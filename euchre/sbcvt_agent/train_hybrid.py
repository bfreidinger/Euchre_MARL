"""
Hybrid MCTS-QMIX training for Euchre — train from scratch.

Bidding phase  : MCTS decides call/pass using target-network Q-values for
                 card-play rollouts and rule-based for all bidding inside
                 rollouts. No Q-prior bias toward calling.

Card-play phase: QMIX epsilon-greedy (epsilon 1.0 -> 0.05, same schedule
                 as the original training).

Only card-play transitions are stored in the replay buffer. QMIX never
sees a bidding state, so its Q-values stay unbiased for card play.

After each TARGET_SYNC_EVERY episode, qmix.sync_targets() is called and
sbcvt.target_mixer is refreshed so rollouts stay stable.

Usage:
  python train_hybrid.py
  python train_hybrid.py --episodes 2000000 --sims 10
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent, JointMemory
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent

from BasicQMIX.train_qmix import (
    QMIXSystem, OBS_DIM, ACTION_NUM,
    BATCH_SIZE, TARGET_SYNC_EVERY, LR,
    EPSILON_START, EPSILON_END, EPSILON_STEPS,
    MEMORY_SIZE, WARMUP_EPISODES,
)
from sbcvt_agent.sbcvt_agent import SBCVTAgent

# ── Defaults ──────────────────────────────────────────────────────────────────

_BASIC_QMIX = os.path.join(os.path.dirname(__file__), '..', 'BasicQMIX')

CKPT_OUT   = os.path.join(_BASIC_QMIX, 'qmix_hybrid.pt')
EPISODES   = 2_000_000
MCTS_SIMS  = 10
MCTS_DEPTH = 5
EVAL_EVERY = 10_000
EVAL_GAMES = 200

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_out', default=CKPT_OUT)
    p.add_argument('--episodes', default=EPISODES,   type=int)
    p.add_argument('--sims',     default=MCTS_SIMS,  type=int)
    p.add_argument('--depth',    default=MCTS_DEPTH, type=int)
    return p.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _legal_mask(state_dict):
    mask = np.zeros(ACTION_NUM, dtype=np.float32)
    for a in state_dict['legal_actions']:
        mask[a] = 1.0
    return mask

# ── Episode Runner ─────────────────────────────────────────────────────────────

def run_episode_hybrid(env, qmix, opp_agents, joint_memory, sbcvt):
    """
    Play one hand.

    Bidding decisions  -> MCTS (sbcvt.step), not stored in replay buffer.
    Card-play decisions -> QMIX epsilon-greedy, stored as joint transitions.
    """
    game = env.game
    state, player_id = env.reset()
    sbcvt.new_hand()

    buf = {}
    prev_gs = env.get_global_state()
    prev_team_tricks = 0

    def flush(trick_reward, done=False):
        nonlocal prev_gs
        if 0 not in buf or 2 not in buf:
            return
        next_gs = env.get_global_state()
        obs0, a0 = buf[0]
        obs2, a2 = buf[2]
        next_state0 = env._extract_state(game.get_state(0))
        next_state2 = env._extract_state(game.get_state(2))
        next_obs0 = next_state0['obs']
        next_obs2 = next_state2['obs']
        next_legal0 = (_legal_mask(next_state0) if not done
                       else np.ones(ACTION_NUM, dtype=np.float32))
        next_legal2 = (_legal_mask(next_state2) if not done
                       else np.ones(ACTION_NUM, dtype=np.float32))
        joint_memory.save(obs0, obs2, a0, a2,
                          next_obs0, next_obs2,
                          trick_reward, prev_gs, next_gs,
                          next_legal0, next_legal2, done)
        buf.clear()
        prev_gs = next_gs

    while not env.is_over():
        is_bidding = game.trump is None

        if player_id == 0:
            if is_bidding:
                action = sbcvt.step(state, player_id=0)
                sbcvt.record_action(0, action, game)
            else:
                action = qmix.agent0.step(state)
                buf[0] = (state['obs'].copy(), action)
            qmix.agent0.total_t += 1
        elif player_id == 2:
            if is_bidding:
                action = sbcvt.step(state, player_id=2)
                sbcvt.record_action(2, action, game)
            else:
                action = qmix.agent2.step(state)
                buf[2] = (state['obs'].copy(), action)
            qmix.agent2.total_t += 1
        else:
            action = opp_agents[player_id].step(state)

        prev_center_len = len(game.center)
        next_state, next_player_id = env.step(action)
        curr_center_len = len(game.center)

        trick_ended = (prev_center_len == 3 and curr_center_len == 0)
        hand_ended  = env.is_over()

        if hand_ended:
            payoffs = game.get_payoffs()
            flush(payoffs.get(0, 0), done=True)
        elif trick_ended:
            new_tricks   = game.score[0] + game.score[2]
            trick_r      = (new_tricks - prev_team_tricks) * 0.25
            prev_team_tricks = new_tricks
            flush(trick_r)
        # No flush during bidding — those transitions are not stored.

        state     = next_state
        player_id = next_player_id

    payoff  = game.get_payoffs().get(0, 0)
    called  = game.calling_player in {0, 2}
    euchred = called and payoff == -2
    return payoff, called, euchred

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_hybrid(env, qmix, sbcvt, opp_agents, n_games):
    """MCTS for bidding, greedy QMIX for card play."""
    wins = calls = euchres = 0
    total_payoff = 0.0

    for _ in range(n_games):
        game = env.game
        state, player_id = env.reset()
        sbcvt.new_hand()

        while not env.is_over():
            is_bidding = game.trump is None

            if player_id == 0:
                if is_bidding:
                    action = sbcvt.step(state, player_id=0)
                    sbcvt.record_action(0, action, game)
                else:
                    action, _ = qmix.agent0.eval_step(state)
            elif player_id == 2:
                if is_bidding:
                    action = sbcvt.step(state, player_id=2)
                    sbcvt.record_action(2, action, game)
                else:
                    action, _ = qmix.agent2.eval_step(state)
            else:
                action, _ = opp_agents[player_id].eval_step(state)

            state, player_id = env.step(action)

        payoff = env.game.get_payoffs().get(0, 0)
        total_payoff += payoff
        if payoff > 0:
            wins += 1
        if env.game.calling_player in {0, 2}:
            calls += 1
            if payoff == -2:
                euchres += 1

    return wins / n_games, total_payoff / n_games, calls / n_games, euchres / max(calls, 1)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = rlcard.make('euchre', config={'num_players': 4})

    agent0 = DQNAgent(
        scope='agent0', action_num=ACTION_NUM, state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_STEPS,
        replay_memory_size=MEMORY_SIZE, replay_memory_init_size=100,
        batch_size=BATCH_SIZE, learning_rate=LR,
    )
    agent2 = DQNAgent(
        scope='agent2', action_num=ACTION_NUM, state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_STEPS,
        replay_memory_size=MEMORY_SIZE, replay_memory_init_size=100,
        batch_size=BATCH_SIZE, learning_rate=LR,
    )

    qmix = QMIXSystem(agent0, agent2, lr=LR)

    sbcvt = SBCVTAgent(
        agent0=agent0, agent2=agent2, mixer=qmix.mixer,
        env=env, num_sims=args.sims, max_depth=args.depth, device=device,
        use_target_nets=True,
        rollout_bid_rule=True,
        rollout_to_end=True,
        target_mixer=qmix.target_mixer,
    )

    opp_agents   = {1: EuchreRuleAgent(), 3: EuchreRuleAgent()}
    joint_memory = JointMemory(memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)
    env.set_agents([agent0, opp_agents[1], agent2, opp_agents[3]])

    print('=' * 72)
    print('  Hybrid MCTS-QMIX Training (fresh weights)')
    print(f'  Checkpoint out : {args.ckpt_out}')
    print(f'  Episodes       : {args.episodes:,}')
    print(f'  MCTS sims      : {args.sims}  depth: {args.depth}  (bidding only)')
    print(f'  Rollout bidding: rule-based')
    print(f'  Rollout cards  : QMIX target nets (synced every {TARGET_SYNC_EVERY:,} eps)')
    print('=' * 72)
    print(f"{'Episode':>8}  {'Loss':>8}  {'WinRate':>8}  {'CallRate':>9}"
          f"  {'Euchr/Call':>10}  {'ε':>6}")
    print('-' * 60)

    recent_losses = []
    eval_episodes, eval_wrs, eval_calls, eval_euchres = [], [], [], []

    for ep in range(1, args.episodes + 1):
        run_episode_hybrid(env, qmix, opp_agents, joint_memory, sbcvt)

        if ep >= WARMUP_EPISODES and len(joint_memory) >= BATCH_SIZE:
            loss = qmix.train(joint_memory)
            recent_losses.append(loss)

        if ep % TARGET_SYNC_EVERY == 0:
            qmix.sync_targets()
            sbcvt.target_mixer = qmix.target_mixer  # keep rollout mixer in sync

        if ep % EVAL_EVERY == 0:
            wr, avg_p, call_r, euchre_r = evaluate_hybrid(
                env, qmix, sbcvt, opp_agents, EVAL_GAMES)
            avg_loss = np.mean(recent_losses[-EVAL_EVERY:]) if recent_losses else float('nan')
            eps = qmix.agent0.epsilons[min(qmix.agent0.total_t, EPSILON_STEPS - 1)]
            print(f"{ep:>8}  {avg_loss:>8.4f}  {wr*100:>7.1f}%  {call_r*100:>8.1f}%"
                  f"  {euchre_r*100:>9.1f}%  {eps:>6.3f}")
            eval_episodes.append(ep)
            eval_wrs.append(wr * 100)
            eval_calls.append(call_r * 100)
            eval_euchres.append(euchre_r * 100)

    qmix.save(args.ckpt_out)

    _, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(eval_episodes, eval_wrs, marker='o', color='steelblue')
    axes[0].axhline(50, color='gray', linestyle='--', linewidth=1)
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_title('Hybrid MCTS-QMIX vs Rule-Based (fresh training)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_episodes, eval_calls, marker='o', color='darkorange')
    axes[1].set_ylabel('Call Rate (%)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(eval_episodes, eval_euchres, marker='o', color='crimson')
    axes[2].set_ylabel('Euchre Rate When Calling (%)')
    axes[2].set_xlabel('Episode')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(_BASIC_QMIX, 'qmix_hybrid_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f'Learning curves saved to {plot_path}')
    plt.show()


if __name__ == '__main__':
    main()
