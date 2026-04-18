"""
QMIX cooperative training for Euchre — mixed personality team.

Agent 0 (aggressive) + Agent 2 (timid) trained together as one QMIX team.
Team 1&3: fixed EuchreRuleAgents (opponents).

Both agents share a single Q_tot through the QMIX mixing network.
The team reward is shaped based on WHICH agent triggered key events:

  Agent 0 calls trump  → bonus  (+0.3)   (aggressive rewarded for calling)
  Agent 2 calls trump  → penalty (-0.2)   (timid penalized for calling)
  March (5-0 sweep)    → bonus  (+0.5)
  Euchre by agent 0    → softened (x0.6)  (aggressive shrugs off risk)
  Euchre by agent 2    → amplified (x1.5) (timid punished for risky call)
  Defensive win        → bonus  (+0.3)    (timid rewarded for safe play)
  Trick reward         → +0.25 per trick

Both agents see the SAME reward per transition — but because that reward
depends on who acted, the mixer learns to let agent 0 drive offensive
decisions and agent 2 play conservatively.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent, JointMemory
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from rlcard.utils.euchre_utils import ACTION_LIST

# ── Hyperparameters ────────────────────────────────────────────────────────────

OBS_DIM      = 48
ACTION_NUM   = 54
GLOBAL_DIM   = 127
MIX_EMBED    = 64

EPISODES          = 5_000_000
BATCH_SIZE        = 64
MEMORY_SIZE       = 500_000
WARMUP_EPISODES   = 50_000
TRAIN_EVERY       = 1
TARGET_SYNC_EVERY = 10_000
EVAL_EVERY        = 100_000
EVAL_GAMES        = 500

GAMMA        = 0.99
LR           = 5e-4
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_STEPS = 15_000_000


# ── Mixing Network ─────────────────────────────────────────────────────────────

class MixingNetwork(nn.Module):
    def __init__(self, n_agents: int, global_dim: int, embed_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w = nn.Sequential(
            nn.Linear(global_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents),
        )
        self.hyper_b = nn.Sequential(
            nn.Linear(global_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, q_vals: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        w = torch.abs(self.hyper_w(global_state))
        b = self.hyper_b(global_state)
        return (q_vals * w).sum(dim=1, keepdim=True) + b


# ── QMIX System ────────────────────────────────────────────────────────────────

class QMIXSystem:
    def __init__(self, agent0: DQNAgent, agent2: DQNAgent,
                 global_dim: int = GLOBAL_DIM,
                 embed_dim: int = MIX_EMBED,
                 lr: float = LR,
                 gamma: float = GAMMA):
        self.agent0 = agent0
        self.agent2 = agent2
        self.gamma  = gamma
        self.train_t = 0
        self.device = agent0.device

        self.mixer        = MixingNetwork(2, global_dim, embed_dim).to(self.device)
        self.target_mixer = deepcopy(self.mixer)

        all_params = (
            list(agent0.q_estimator.qnet.parameters()) +
            list(agent2.q_estimator.qnet.parameters()) +
            list(self.mixer.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    def _t(self, arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=self.device)

    def sync_targets(self):
        self.agent0.target_estimator = deepcopy(self.agent0.q_estimator)
        self.agent2.target_estimator = deepcopy(self.agent2.q_estimator)
        self.target_mixer = deepcopy(self.mixer)

    def save(self, path: str):
        torch.save({
            'agent0_qnet': self.agent0.q_estimator.qnet.state_dict(),
            'agent2_qnet': self.agent2.q_estimator.qnet.state_dict(),
            'mixer': self.mixer.state_dict(),
        }, path)
        print(f"QMIX model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent0.q_estimator.qnet.load_state_dict(ckpt['agent0_qnet'])
        self.agent2.q_estimator.qnet.load_state_dict(ckpt['agent2_qnet'])
        self.mixer.load_state_dict(ckpt['mixer'])
        self.sync_targets()
        print(f"QMIX model loaded from {path}")

    def train(self, joint_memory: JointMemory) -> float:
        (obs1, obs2, a1, a2,
         next_obs1, next_obs2,
         reward, gs, next_gs,
         nl1, nl2, done) = joint_memory.sample()

        obs1      = self._t(obs1)
        obs2      = self._t(obs2)
        a1        = self._t(a1, torch.long)
        a2        = self._t(a2, torch.long)
        next_obs1 = self._t(next_obs1)
        next_obs2 = self._t(next_obs2)
        reward    = self._t(reward)
        gs        = self._t(gs)
        next_gs   = self._t(next_gs)
        nl1       = self._t(nl1)
        nl2       = self._t(nl2)
        done      = self._t(done)

        self.agent0.q_estimator.qnet.train()
        self.agent2.q_estimator.qnet.train()
        self.mixer.train()

        q0_all = self.agent0.q_estimator.qnet(obs1)
        q2_all = self.agent2.q_estimator.qnet(obs2)
        q0 = q0_all.gather(1, a1.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, a2.unsqueeze(1)).squeeze(1)

        q_vals = torch.stack([q0, q2], dim=1)
        q_tot  = self.mixer(q_vals, gs)

        with torch.no_grad():
            self.agent0.target_estimator.qnet.eval()
            self.agent2.target_estimator.qnet.eval()
            self.target_mixer.eval()

            q0_online = self.agent0.q_estimator.qnet(next_obs1)
            q2_online = self.agent2.q_estimator.qnet(next_obs2)
            q0_online[nl1 == 0] = -1e9
            q2_online[nl2 == 0] = -1e9
            best_a0 = q0_online.argmax(dim=1)
            best_a2 = q2_online.argmax(dim=1)

            q0_next = self.agent0.target_estimator.qnet(next_obs1)
            q2_next = self.agent2.target_estimator.qnet(next_obs2)
            max_q0  = q0_next.gather(1, best_a0.unsqueeze(1)).squeeze(1)
            max_q2  = q2_next.gather(1, best_a2.unsqueeze(1)).squeeze(1)

            q_next_vals = torch.stack([max_q0, max_q2], dim=1)
            q_tot_next  = self.target_mixer(q_next_vals, next_gs)
            not_done    = (1.0 - done).unsqueeze(1)
            target      = reward.unsqueeze(1) + self.gamma * q_tot_next * not_done

        loss = F.mse_loss(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent0.q_estimator.qnet.parameters()) +
            list(self.agent2.q_estimator.qnet.parameters()) +
            list(self.mixer.parameters()),
            max_norm=10.0
        )
        self.optimizer.step()

        self.agent0.q_estimator.qnet.eval()
        self.agent2.q_estimator.qnet.eval()
        self.mixer.eval()

        self.train_t += 1
        return loss.item()


# ── Episode Runner ─────────────────────────────────────────────────────────────

def run_episode(env, qmix: QMIXSystem, opp_agents: dict,
                joint_memory: JointMemory) -> float:
    """
    Play one full hand with mixed-personality reward shaping.

    Both agents receive the SAME team reward per transition, but that
    reward depends on WHICH agent called trump:
      - Agent 0 calling → aggressive shaping (bonus for calling, soft euchre)
      - Agent 2 calling → timid shaping (penalty for calling, harsh euchre)

    Returns the final game payoff for the QMIX team (player 0's payoff).
    """
    game = env.game
    state, player_id = env.reset()

    buf: dict = {}
    prev_gs          = env.get_global_state()
    prev_team_tricks = 0
    trump_caller     = None  # which agent called trump (0, 2, or None)

    def _legal_mask(state_dict):
        mask = np.zeros(ACTION_NUM, dtype=np.float32)
        for a in state_dict['legal_actions']:
            mask[a] = 1.0
        return mask

    def flush(reward: float, done: bool = False):
        nonlocal prev_gs
        if 0 not in buf or 2 not in buf:
            return

        next_gs  = env.get_global_state()
        obs0, a0 = buf[0]
        obs2, a2 = buf[2]

        next_state0 = env._extract_state(game.get_state(0))
        next_state2 = env._extract_state(game.get_state(2))
        next_obs0   = next_state0['obs']
        next_obs2   = next_state2['obs']
        next_legal0 = _legal_mask(next_state0) if not done else np.ones(ACTION_NUM, dtype=np.float32)
        next_legal2 = _legal_mask(next_state2) if not done else np.ones(ACTION_NUM, dtype=np.float32)

        joint_memory.save(obs0, obs2, a0, a2,
                          next_obs0, next_obs2,
                          reward, prev_gs, next_gs,
                          next_legal0, next_legal2, done)
        buf.clear()
        prev_gs = next_gs

    while not env.is_over():

        # ── action selection ──────────────────────────────────────────────────
        if player_id == 0:
            action = qmix.agent0.step(state)
            qmix.agent0.total_t += 1
        elif player_id == 2:
            action = qmix.agent2.step(state)
            qmix.agent2.total_t += 1
        else:
            action = opp_agents[player_id].step(state)

        # ── buffer team agent transitions ─────────────────────────────────────
        if player_id in (0, 2):
            buf[player_id] = (state['obs'].copy(), action)

        # ── detect which agent called trump ───────────────────────────────────
        if player_id in (0, 2) and trump_caller is None:
            action_str = ACTION_LIST[action]
            if action_str == 'pick' or action_str.startswith('call'):
                trump_caller = player_id

        prev_center_len = len(game.center)
        next_state, next_player_id = env.step(action)
        curr_center_len = len(game.center)

        trick_ended = (prev_center_len == 3 and curr_center_len == 0)
        hand_ended  = env.is_over()

        # ── flush joint transition with mixed-personality reward ───────────────
        if hand_ended:
            base_payoff = game.get_payoffs().get(0, 0)
            shaped_reward = base_payoff

            team_tricks = game.score[0] + game.score[2]
            team_won = base_payoff > 0

            if team_won and team_tricks == 5:
                # March
                shaped_reward += 0.5

            elif not team_won and trump_caller is not None:
                # Euchred — penalty depends on who called
                if trump_caller == 0:
                    # Agent 0 (aggressive) called — soften penalty
                    shaped_reward = base_payoff * 0.6
                else:
                    # Agent 2 (timid) called — amplify penalty
                    shaped_reward = base_payoff * 1.5

            elif team_won and trump_caller is None:
                # Defensive win (opponents called, we won)
                shaped_reward += 0.3

            flush(shaped_reward, done=True)

        elif trick_ended:
            new_tricks = game.score[0] + game.score[2]
            trick_r    = (new_tricks - prev_team_tricks) * 0.25
            prev_team_tricks = new_tricks
            flush(trick_r)

        elif 0 in buf and 2 in buf and len(game.center) == 0:
            # Both acted during bidding — reward/penalize based on who called
            if trump_caller == 0:
                flush(0.3)    # aggressive agent called — good
            elif trump_caller == 2:
                flush(-0.2)   # timid agent called — bad
            else:
                flush(0.0)    # no one called yet

        state     = next_state
        player_id = next_player_id

    return game.get_payoffs().get(0, 0)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(env, qmix: QMIXSystem, opp_agents: dict,
             n_games: int = EVAL_GAMES) -> tuple[float, float]:
    wins   = 0
    total  = 0.0

    for _ in range(n_games):
        state, player_id = env.reset()
        while not env.is_over():
            if player_id == 0:
                action, _ = qmix.agent0.eval_step(state)
            elif player_id == 2:
                action, _ = qmix.agent2.eval_step(state)
            else:
                action, _ = opp_agents[player_id].eval_step(state)
            state, player_id = env.step(action)

        payoffs = env.game.get_payoffs()
        p = payoffs.get(0, 0)
        total += p
        if p > 0:
            wins += 1

    return wins / n_games, total / n_games


# ── Main Training Loop ─────────────────────────────────────────────────────────

if __name__ == '__main__':

    env = rlcard.make('euchre', config={'num_players': 4})

    agent0 = DQNAgent(
        scope='aggressive',
        action_num=ACTION_NUM,
        state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_STEPS,
        replay_memory_size=MEMORY_SIZE,
        replay_memory_init_size=100,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    agent2 = DQNAgent(
        scope='timid',
        action_num=ACTION_NUM,
        state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_STEPS,
        replay_memory_size=MEMORY_SIZE,
        replay_memory_init_size=100,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    opp_agents = {1: EuchreRuleAgent(), 3: EuchreRuleAgent()}

    joint_memory = JointMemory(memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)
    qmix         = QMIXSystem(agent0, agent2)

    env.set_agents([agent0, opp_agents[1], agent2, opp_agents[3]])

    print("=" * 60)
    print("QMIX Mixed-Personality Euchre Training")
    print("  Player 0 : AGGRESSIVE (risk-taker, trump caller)")
    print("  Player 2 : TIMID      (conservative, defensive)")
    print("  Players 1 & 3 : fixed rule agents (opponents)")
    print(f"  Episodes : {EPISODES}   Warmup: {WARMUP_EPISODES}")
    print("=" * 60)
    print(f"{'Episode':>8}  {'AvgLoss':>9}  {'WinRate':>8}  {'AvgPayoff':>10}")
    print("-" * 44)

    recent_losses = []
    eval_episodes = []
    eval_win_rates = []
    eval_avg_payoffs = []

    save_dir = os.path.dirname(os.path.abspath(__file__))

    for ep in range(1, EPISODES + 1):
        run_episode(env, qmix, opp_agents, joint_memory)

        if ep >= WARMUP_EPISODES and len(joint_memory) >= BATCH_SIZE:
            loss = qmix.train(joint_memory)
            recent_losses.append(loss)

        if ep % TARGET_SYNC_EVERY == 0:
            qmix.sync_targets()
            print(f"  [ep {ep:>4}] target networks synced")

        if ep % EVAL_EVERY == 0:
            qmix.save(os.path.join(save_dir, 'qmix_mixed.pt'))

            win_rate, avg_payoff = evaluate(env, qmix, opp_agents)
            avg_loss = np.mean(recent_losses[-EVAL_EVERY:]) if recent_losses else float('nan')
            eps0 = qmix.agent0.epsilons[min(qmix.agent0.total_t, EPSILON_STEPS - 1)]
            print(f"{ep:>8}  {avg_loss:>9.4f}  {win_rate*100:>7.1f}%  {avg_payoff:>+10.3f}"
                  f"   ε={eps0:.3f}")
            eval_episodes.append(ep)
            eval_win_rates.append(win_rate * 100)
            eval_avg_payoffs.append(avg_payoff)

    # ── final evaluation ──────────────────────────────────────────────────────
    print("=" * 60)
    win_rate, avg_payoff = evaluate(env, qmix, opp_agents, n_games=2000)
    print(f"Final (2000 games):  win={win_rate*100:.1f}%  avg_payoff={avg_payoff:+.3f}")

    qmix.save(os.path.join(save_dir, 'qmix_mixed.pt'))

    # ── plot learning curves ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(eval_episodes, eval_win_rates, marker='o', linewidth=2, color='steelblue')
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% baseline')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('QMIX Mixed (Aggressive + Timid) vs Rule Agents')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eval_episodes, eval_avg_payoffs, marker='o', linewidth=2, color='darkorange')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, label='break-even')
    ax2.set_ylabel('Avg Payoff')
    ax2.set_xlabel('Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'qmix_mixed_learning_curve.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Learning curve saved to {plot_path}")
    plt.show()
