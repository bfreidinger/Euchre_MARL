"""
Alternating cooperative training for Euchre.

This trainer is designed to address a few bottlenecks that show up quickly in
plain alternating DQN:

1. Richer local observations for the learners, including legal-action masks and
   compact game-context features.
2. A truly frozen teammate snapshot for each phase, so the active learner sees
   a stationary partner policy during that phase.
3. Fixed rule-based opponents so the benchmark stays consistent across phases.
4. Automatic tracking of the best joint checkpoint by evaluation win rate.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from euchre import rlcard
from euchre.rlcard.agents.dqn_agent_pytorch import DQNAgent
from euchre.rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from euchre.rlcard.agents.random_agent import RandomAgent


BASE_OBS_DIM = 48
ACTION_NUM = 54
PLAYER_ONE_HOT_DIM = 4
DEALER_ONE_HOT_DIM = 4
SUIT_ONE_HOT_DIM = 4
TURN_POSITION_DIM = 4
RELATIVE_SEAT_DIM = 4
EXTRA_OBS_DIM = (
    ACTION_NUM +              # legal action mask
    PLAYER_ONE_HOT_DIM +      # learner id
    DEALER_ONE_HOT_DIM +      # dealer id
    SUIT_ONE_HOT_DIM +        # trump
    SUIT_ONE_HOT_DIM +        # turned down suit
    SUIT_ONE_HOT_DIM +        # lead suit
    TURN_POSITION_DIM +       # cards currently in center (0-3)
    RELATIVE_SEAT_DIM +       # seat relative to dealer
    8                         # scalar context features
)
AUGMENTED_OBS_DIM = BASE_OBS_DIM + EXTRA_OBS_DIM


@dataclass
class AlternatingTrainingConfig:
    rule_init_hands: int = 1000
    phase_episodes: int = 1500
    max_rounds: int = 30
    eval_games: int = 1000
    convergence_tol: float = 0.01
    patience_rounds: int = 4
    trick_reward_scale: float = 0.25
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_memory_size: int = 30000
    replay_memory_init_size: int = 500
    update_target_every: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 30000
    mlp_layers: tuple[int, ...] = (256, 256, 128)
    seed: int | None = None


class AlternatingCooperativeTrainer:
    """Alternate training between teammates 0 and 2."""

    def __init__(self, config: AlternatingTrainingConfig):
        self.config = config
        self.env = rlcard.make('euchre', config={'num_players': 4, 'seed': config.seed})
        self.agents = {
            0: self._make_learning_agent('coop_p0'),
            2: self._make_learning_agent('coop_p2'),
        }
        self.rule_opponents = {1: EuchreRuleAgent(), 3: EuchreRuleAgent()}
        self.random_opponents = {1: RandomAgent(ACTION_NUM), 3: RandomAgent(ACTION_NUM)}
        self.phase_history: List[Dict[str, float]] = []
        self.best_result: Dict[str, float] | None = None

    @staticmethod
    def _format_pct(value: float) -> str:
        return f"{value * 100:6.2f}%"

    @staticmethod
    def _one_hot(index: int | None, size: int) -> np.ndarray:
        vec = np.zeros(size, dtype=np.float32)
        if index is not None and 0 <= index < size:
            vec[index] = 1.0
        return vec

    @staticmethod
    def _suit_one_hot(suit: str | None) -> np.ndarray:
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        return AlternatingCooperativeTrainer._one_hot(suit_map.get(suit), SUIT_ONE_HOT_DIM)

    def _augment_state(self, raw_state: dict, player_id: int) -> dict:
        game = self.env.game
        legal_mask = np.zeros(ACTION_NUM, dtype=np.float32)
        legal_mask[raw_state['legal_actions']] = 1.0

        dealer_id = getattr(game, 'dealer_player_id', None)
        turned_down = raw_state.get('turned_down')
        relative_seat = None if dealer_id is None else (player_id - dealer_id) % 4

        team_tricks = float(game.score[0] + game.score[2]) / 5.0
        opp_tricks = float(game.score[1] + game.score[3]) / 5.0
        context = np.array([
            float(raw_state.get('trump_called', False)),
            float(raw_state.get('flipped') is not None),
            float(turned_down is not None),
            float(len(raw_state['hand'])) / 6.0,
            float(len(raw_state['center'])) / 4.0,
            team_tricks,
            opp_tricks,
            (team_tricks - opp_tricks),
        ], dtype=np.float32)

        extra = np.concatenate([
            legal_mask,
            self._one_hot(player_id, PLAYER_ONE_HOT_DIM),
            self._one_hot(dealer_id, DEALER_ONE_HOT_DIM),
            self._suit_one_hot(raw_state.get('trump')),
            self._suit_one_hot(turned_down),
            self._suit_one_hot(raw_state.get('lead_suit')),
            self._one_hot(len(raw_state['center']), TURN_POSITION_DIM),
            self._one_hot(relative_seat, RELATIVE_SEAT_DIM),
            context,
        ]).astype(np.float32)

        state = dict(raw_state)
        state['obs'] = np.concatenate([raw_state['obs'].astype(np.float32), extra]).astype(np.float32)
        return state

    def _make_learning_agent(self, scope: str) -> DQNAgent:
        return DQNAgent(
            scope=scope,
            action_num=ACTION_NUM,
            state_shape=[AUGMENTED_OBS_DIM],
            mlp_layers=list(self.config.mlp_layers),
            replay_memory_size=self.config.replay_memory_size,
            replay_memory_init_size=self.config.replay_memory_init_size,
            update_target_estimator_every=self.config.update_target_every,
            discount_factor=self.config.gamma,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay_steps=self.config.epsilon_decay_steps,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )

    @staticmethod
    def _team_tricks(game) -> int:
        return game.score[0] + game.score[2]

    @staticmethod
    def _opp_tricks(game) -> int:
        return game.score[1] + game.score[3]

    def _policy_action(
        self,
        player_id: int,
        raw_state: dict,
        train_player: int | None,
        frozen_teammate=None,
        opp_agents: Dict[int, object] | None = None,
    ) -> int:
        opp_agents = opp_agents or self.rule_opponents

        if player_id in opp_agents:
            action, _ = opp_agents[player_id].eval_step(raw_state)
            return action

        learner_state = self._augment_state(raw_state, player_id)
        if train_player is not None and player_id == train_player:
            return self.agents[player_id].step(learner_state)

        policy_agent = frozen_teammate if frozen_teammate is not None else self.agents[player_id]
        action, _ = policy_agent.eval_step(learner_state)
        return action

    def _phase_teammate_snapshot(self, frozen_player: int):
        teammate = deepcopy(self.agents[frozen_player])
        teammate.q_estimator.qnet.eval()
        teammate.target_estimator.qnet.eval()
        return teammate

    @staticmethod
    def _clone_rule_agent():
        return EuchreRuleAgent()

    def _collect_rule_demonstrations(self, player_id: int, num_hands: int) -> tuple[np.ndarray, np.ndarray]:
        states = []
        actions = []
        teammates = {0: self._clone_rule_agent(), 2: self._clone_rule_agent()}
        opponents = {1: self._clone_rule_agent(), 3: self._clone_rule_agent()}

        for _ in range(num_hands):
            raw_state, current_player = self.env.reset()
            while not self.env.is_over():
                if current_player in teammates:
                    action, _ = teammates[current_player].eval_step(raw_state)
                    if current_player == player_id:
                        states.append(self._augment_state(raw_state, current_player)['obs'])
                        actions.append(action)
                else:
                    action, _ = opponents[current_player].eval_step(raw_state)
                raw_state, current_player = self.env.step(action)

        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

    def initialize_from_rule_policy(self) -> None:
        if self.config.rule_init_hands <= 0:
            return

        print(f"Bootstrapping from rule policy with {self.config.rule_init_hands} imitation hands per learner")
        for player_id in (0, 2):
            states, actions = self._collect_rule_demonstrations(player_id, self.config.rule_init_hands)
            if len(states) == 0:
                continue

            agent = self.agents[player_id]
            qnet = agent.q_estimator.qnet
            optimizer = agent.q_estimator.optimizer
            qnet.train()

            batch_size = min(self.config.batch_size, len(states))
            epochs = 3
            losses = []
            for _ in range(epochs):
                permutation = np.random.permutation(len(states))
                for start in range(0, len(states), batch_size):
                    idx = permutation[start:start + batch_size]
                    batch_states = torch.from_numpy(states[idx]).float().to(agent.device)
                    batch_actions = torch.from_numpy(actions[idx]).long().to(agent.device)
                    logits = qnet(batch_states)
                    loss = F.cross_entropy(logits, batch_actions)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

            qnet.eval()
            agent.target_estimator = deepcopy(agent.q_estimator)
            print(
                f"  P{player_id} rule init: {len(states)} samples, "
                f"avg imitation loss {np.mean(losses):.4f}"
            )

    def _print_banner(self) -> None:
        print("=" * 72)
        print("Alternating Cooperative Euchre Training")
        print("  Team learner : players 0 and 2")
        print("  Opponents    : rule-based players 1 and 3")
        print(f"  Obs dim      : {AUGMENTED_OBS_DIM} (base {BASE_OBS_DIM} + context {EXTRA_OBS_DIM})")
        print(f"  Rule init    : {self.config.rule_init_hands} imitation hands")
        print(f"  Phase hands  : {self.config.phase_episodes}")
        print(f"  Max rounds   : {self.config.max_rounds}")
        print(f"  Eval hands   : {self.config.eval_games}")
        print("=" * 72)

    def _print_phase_summary(self, round_idx: int, phase: Dict[str, float]) -> None:
        print(
            f"  Round {round_idx:>2} | Train P{phase['train_player']} | "
            f"train payoff {phase['phase_avg_training_payoff']:+.3f} | "
            f"WIN RATE {self._format_pct(phase['eval_win_rate'])} | "
            f"eval payoff {phase['eval_avg_payoff']:+.3f}"
        )

    def _print_round_summary(self, round_idx: int, p0_phase: Dict[str, float], p2_phase: Dict[str, float]) -> None:
        best_phase = max((p0_phase, p2_phase), key=lambda phase: (phase['eval_win_rate'], phase['eval_avg_payoff']))
        print("-" * 72)
        print(
            f"End of round {round_idx}: current joint WIN RATE "
            f"{self._format_pct(p2_phase['eval_win_rate'])} | "
            f"avg payoff {p2_phase['eval_avg_payoff']:+.3f}"
        )
        print(
            f"  Best phase this round: after training P{best_phase['train_player']} "
            f"the team reached {self._format_pct(best_phase['eval_win_rate'])}"
        )
        if self.best_result is not None:
            print(
                f"  Best overall so far : {self._format_pct(self.best_result['eval_win_rate'])} "
                f"after training P{self.best_result['train_player']} in round {self.best_result['round']}"
            )
        print("-" * 72)

    def play_training_episode(self, train_player: int, frozen_teammate, opp_agents: Dict[int, object]) -> float:
        """
        Play one hand while training only `train_player`.

        The active learner sees a stationary teammate snapshot during this phase.
        """
        active_agent = self.agents[train_player]
        raw_state, player_id = self.env.reset()
        pending_state = None
        pending_action = None
        pending_reward = 0.0
        prev_team_diff = self._team_tricks(self.env.game) - self._opp_tricks(self.env.game)

        while not self.env.is_over():
            if player_id == train_player and pending_state is not None:
                next_train_state = self._augment_state(raw_state, train_player)
                active_agent.feed((
                    pending_state,
                    pending_action,
                    pending_reward,
                    next_train_state,
                    False,
                ))
                pending_state = None
                pending_action = None
                pending_reward = 0.0

            action = self._policy_action(
                player_id,
                raw_state,
                train_player=train_player,
                frozen_teammate=frozen_teammate,
                opp_agents=opp_agents,
            )

            if player_id == train_player:
                pending_state = self._augment_state(raw_state, train_player)
                pending_action = action

            prev_center_len = len(self.env.game.center)
            next_raw_state, next_player_id = self.env.step(action)
            hand_ended = self.env.is_over()
            trick_ended = (prev_center_len == 3 and len(self.env.game.center) == 0)

            if pending_state is not None and trick_ended:
                team_diff = self._team_tricks(self.env.game) - self._opp_tricks(self.env.game)
                pending_reward += self.config.trick_reward_scale * (team_diff - prev_team_diff)
                prev_team_diff = team_diff

            if hand_ended:
                if pending_state is not None:
                    terminal_state = self._augment_state(self.env.get_state(train_player), train_player)
                    terminal_reward = self.env.game.get_payoffs().get(train_player, 0.0)
                    active_agent.feed((
                        pending_state,
                        pending_action,
                        pending_reward + terminal_reward,
                        terminal_state,
                        True,
                    ))
                break

            raw_state = next_raw_state
            player_id = next_player_id

        return self.env.game.get_payoffs().get(0, 0.0)

    def _is_new_best(self, result: Dict[str, float]) -> bool:
        if self.best_result is None:
            return True
        return (
            result['eval_win_rate'],
            result['eval_avg_payoff'],
        ) > (
            self.best_result['eval_win_rate'],
            self.best_result['eval_avg_payoff'],
        )

    def run_phase(self, round_idx: int, train_player: int, num_episodes: int, best_checkpoint_path: Path | None = None) -> Dict[str, float]:
        self.agents[train_player].memory.clear()
        frozen_player = 2 if train_player == 0 else 0
        frozen_teammate = self._phase_teammate_snapshot(frozen_player)

        payoffs = []
        for _ in range(num_episodes):
            payoffs.append(self.play_training_episode(train_player, frozen_teammate, self.rule_opponents))

        win_rate, avg_payoff = self.evaluate_joint_policy(self.config.eval_games)
        win_count = int(round(win_rate * self.config.eval_games))
        result = {
            'round': round_idx,
            'train_player': train_player,
            'episodes': num_episodes,
            'phase_avg_training_payoff': float(np.mean(payoffs)) if payoffs else 0.0,
            'eval_win_rate': win_rate,
            'eval_win_pct': win_rate * 100.0,
            'eval_wins': win_count,
            'eval_games': self.config.eval_games,
            'eval_avg_payoff': avg_payoff,
        }
        self.phase_history.append(result)

        if self._is_new_best(result):
            self.best_result = dict(result)
            if best_checkpoint_path is not None:
                self.save_checkpoint(best_checkpoint_path)

        return result

    def evaluate_joint_policy(self, num_games: int) -> tuple[float, float]:
        wins = 0
        total = 0.0

        for _ in range(num_games):
            raw_state, player_id = self.env.reset()
            while not self.env.is_over():
                action = self._policy_action(
                    player_id,
                    raw_state,
                    train_player=None,
                    frozen_teammate=None,
                    opp_agents=self.rule_opponents,
                )
                raw_state, player_id = self.env.step(action)

            payoff = self.env.game.get_payoffs().get(0, 0.0)
            total += payoff
            if payoff > 0:
                wins += 1

        return wins / num_games, total / num_games

    def train(self, best_checkpoint_path: Path | None = None) -> List[Dict[str, float]]:
        stagnant_rounds = 0
        last_round_eval = None
        self._print_banner()

        for round_idx in range(1, self.config.max_rounds + 1):
            print(f"\nRound {round_idx}/{self.config.max_rounds}")

            round_phases = []
            for train_player in (0, 2):
                phase = self.run_phase(round_idx, train_player, self.config.phase_episodes, best_checkpoint_path=best_checkpoint_path)
                round_phases.append(phase)
                self._print_phase_summary(round_idx, phase)

            self._print_round_summary(round_idx, round_phases[0], round_phases[1])

            round_eval = self.phase_history[-1]['eval_avg_payoff']
            if last_round_eval is not None:
                if abs(round_eval - last_round_eval) <= self.config.convergence_tol:
                    stagnant_rounds += 1
                else:
                    stagnant_rounds = 0

                if stagnant_rounds >= self.config.patience_rounds:
                    print(
                        "Stopping early: joint payoff change stayed within "
                        f"{self.config.convergence_tol:.3f} for {self.config.patience_rounds} rounds."
                    )
                    break

            last_round_eval = round_eval

        return self.phase_history

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': asdict(self.config),
            'obs_dim': AUGMENTED_OBS_DIM,
            'agent0': self.agents[0].get_state_dict(),
            'agent2': self.agents[2].get_state_dict(),
            'history': self.phase_history,
            'best_result': self.best_result,
        }, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Alternating cooperative Euchre training')
    parser.add_argument('--rule-init-hands', type=int, default=1000)
    parser.add_argument('--phase-episodes', type=int, default=1500)
    parser.add_argument('--max-rounds', type=int, default=30)
    parser.add_argument('--eval-games', type=int, default=1000)
    parser.add_argument('--convergence-tol', type=float, default=0.01)
    parser.add_argument('--patience-rounds', type=int, default=4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/alternating_coop.pt')
    parser.add_argument('--best-checkpoint', type=str, default='checkpoints/alternating_coop.best.pt')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AlternatingTrainingConfig(
        rule_init_hands=args.rule_init_hands,
        phase_episodes=args.phase_episodes,
        max_rounds=args.max_rounds,
        eval_games=args.eval_games,
        convergence_tol=args.convergence_tol,
        patience_rounds=args.patience_rounds,
        seed=args.seed,
    )

    trainer = AlternatingCooperativeTrainer(config)
    trainer.initialize_from_rule_policy()
    history = trainer.train(best_checkpoint_path=Path(args.best_checkpoint))
    trainer.save_checkpoint(args.checkpoint)

    final_win_rate, final_avg_payoff = trainer.evaluate_joint_policy(config.eval_games)
    print("\n" + "=" * 72)
    print("Final greedy evaluation")
    print(f"  WIN RATE        : {final_win_rate * 100:6.2f}% ({int(round(final_win_rate * config.eval_games))}/{config.eval_games})")
    print(f"  Avg payoff      : {final_avg_payoff:+.3f}")
    print(f"  Phases run      : {len(history)}")
    print(f"  Final checkpoint: {args.checkpoint}")
    print(f"  Best checkpoint : {args.best_checkpoint}")
    if trainer.best_result is not None:
        print(
            "  Best observed   : "
            f"{trainer.best_result['eval_win_pct']:.2f}% after training P{trainer.best_result['train_player']} "
            f"in round {trainer.best_result['round']}"
        )
    print("=" * 72)


if __name__ == '__main__':
    main()
