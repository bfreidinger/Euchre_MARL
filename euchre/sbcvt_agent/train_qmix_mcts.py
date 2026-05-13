"""
Fine-tune QMIX with MCTS-guided bidding.

Loads qmix_euchre.pt and continues training using SBCVT to choose
trump-calling actions instead of epsilon-greedy. Card play stays
epsilon-greedy. The replay buffer starts fresh so old over-calling
transitions don't counteract the corrective gradient signal.

MCTS epsilon decays 0.5 → 0.05 over the first 60% of training,
forcing exploration of both call and pass in early fine-tuning and
converging to near-greedy MCTS by the end.

Usage:
  python train_qmix_mcts.py
  python train_qmix_mcts.py --episodes 300000 --sims 30 --lr_mult 2
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
    run_episode, evaluate,
)
from sbcvt_agent.sbcvt_agent import SBCVTAgent

# ── Defaults ──────────────────────────────────────────────────────────────────

_BASIC_QMIX = os.path.join(os.path.dirname(__file__), '..', 'BasicQMIX')

CKPT_IN  = os.path.join(_BASIC_QMIX, 'qmix_euchre.pt')
CKPT_OUT = os.path.join(_BASIC_QMIX, 'qmix_mcts_finetuned.pt')

EPISODES        = 500_000
LR_MULT         = 3          # multiply base LR to speed up unlearning
MEMORY_SIZE     = 100_000    # fresh buffer — smaller than original 500k
WARMUP          = 1_000

MCTS_SIMS       = 20
MCTS_DEPTH      = 5
MCTS_EPS_START  = 0.5
MCTS_EPS_END    = 0.05
MCTS_EPS_STEPS  = 300_000    # decay over first 60% of fine-tuning

CARD_EPSILON    = 0.05       # fixed epsilon for card-play steps

EVAL_EVERY      = 10_000
EVAL_GAMES      = 500

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_in',  default=CKPT_IN)
    p.add_argument('--ckpt_out', default=CKPT_OUT)
    p.add_argument('--episodes', default=EPISODES,    type=int)
    p.add_argument('--sims',     default=MCTS_SIMS,   type=int)
    p.add_argument('--depth',    default=MCTS_DEPTH,  type=int)
    p.add_argument('--lr_mult',  default=LR_MULT,     type=float)
    return p.parse_args()

# ── Epsilon schedule ──────────────────────────────────────────────────────────

def mcts_epsilon(ep: int) -> float:
    frac = min(ep / MCTS_EPS_STEPS, 1.0)
    return MCTS_EPS_START + frac * (MCTS_EPS_END - MCTS_EPS_START)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    fine_lr = LR * args.lr_mult
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = rlcard.make('euchre', config={'num_players': 4})

    # Build agents with fixed low card-play epsilon (fine-tuning, not exploring)
    agent0 = DQNAgent(
        scope='agent0', action_num=ACTION_NUM, state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=CARD_EPSILON, epsilon_end=CARD_EPSILON, epsilon_decay_steps=1,
    )
    agent2 = DQNAgent(
        scope='agent2', action_num=ACTION_NUM, state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=CARD_EPSILON, epsilon_end=CARD_EPSILON, epsilon_decay_steps=1,
    )

    # Load pre-trained weights; QMIXSystem creates a fresh optimizer at fine_lr
    qmix = QMIXSystem(agent0, agent2, lr=fine_lr)
    qmix.load(args.ckpt_in)

    sbcvt = SBCVTAgent(
        agent0=qmix.agent0, agent2=qmix.agent2, mixer=qmix.mixer,
        env=env, num_sims=args.sims, max_depth=args.depth, device=device,
    )

    opp_agents   = {1: EuchreRuleAgent(), 3: EuchreRuleAgent()}
    joint_memory = JointMemory(memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)

    env.set_agents([agent0, opp_agents[1], agent2, opp_agents[3]])

    print('=' * 72)
    print('  QMIX Fine-Tuning with MCTS-Guided Bidding')
    print(f'  Checkpoint in : {args.ckpt_in}')
    print(f'  Checkpoint out: {args.ckpt_out}')
    print(f'  Episodes : {args.episodes:,}   LR: {fine_lr:.2e} ({args.lr_mult}× base)')
    print(f'  MCTS sims: {args.sims}   depth: {args.depth}')
    print(f'  MCTS ε   : {MCTS_EPS_START} → {MCTS_EPS_END} over {MCTS_EPS_STEPS:,} eps')
    print(f'  Buffer   : {MEMORY_SIZE:,} (fresh — old biased data discarded)')
    print('=' * 72)
    print(f"{'Episode':>8}  {'Loss':>8}  {'WinRate':>8}  {'CallRate':>9}"
          f"  {'Euchr/Call':>10}  {'MCTSε':>6}")
    print('-' * 60)

    recent_losses = []
    eval_episodes, eval_wrs, eval_calls, eval_euchres = [], [], [], []

    for ep in range(1, args.episodes + 1):
        sbcvt.epsilon = mcts_epsilon(ep)
        run_episode(env, qmix, opp_agents, joint_memory, sbcvt_agent=sbcvt)

        if ep >= WARMUP and len(joint_memory) >= BATCH_SIZE:
            loss = qmix.train(joint_memory)
            recent_losses.append(loss)

        if ep % TARGET_SYNC_EVERY == 0:
            qmix.sync_targets()

        if ep % EVAL_EVERY == 0:
            wr, _, call_r, euchre_r = evaluate(env, qmix, opp_agents, EVAL_GAMES)
            avg_loss = np.mean(recent_losses[-EVAL_EVERY:]) if recent_losses else float('nan')
            print(f"{ep:>8}  {avg_loss:>8.4f}  {wr*100:>7.1f}%  {call_r*100:>8.1f}%"
                  f"  {euchre_r*100:>9.1f}%  {sbcvt.epsilon:>6.3f}")
            eval_episodes.append(ep)
            eval_wrs.append(wr * 100)
            eval_calls.append(call_r * 100)
            eval_euchres.append(euchre_r * 100)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    qmix.save(args.ckpt_out)

    # ── Learning curves ───────────────────────────────────────────────────────
    _, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(eval_episodes, eval_wrs, marker='o', color='steelblue')
    axes[0].axhline(50, color='gray', linestyle='--', linewidth=1)
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_title('QMIX MCTS Fine-Tuning vs Rule-Based')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_episodes, eval_calls, marker='o', color='darkorange')
    axes[1].set_ylabel('Call Rate (%)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(eval_episodes, eval_euchres, marker='o', color='crimson')
    axes[2].set_ylabel('Euchre Rate When Calling (%)')
    axes[2].set_xlabel('Episode')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(_BASIC_QMIX, 'qmix_mcts_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f'Learning curves saved to {plot_path}')
    plt.show()


if __name__ == '__main__':
    main()
