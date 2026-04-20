"""
Standalone DQN training for Euchre — no cooperative mixing network.

Each agent (players 0 & 2) trains independently, optimizing its own
individual payoff via standard Double-DQN.  Opponents (players 1 & 3)
are fixed rule-based agents — the same setup as QMIX training so results
are directly comparable.

Checkpoint saved to dqn_euchre.pt with keys:
    agent0_q_estimator      — player 0 Q-network weights
    agent0_target_estimator — player 0 target-network weights
    agent2_q_estimator      — player 2 Q-network weights
    agent2_target_estimator — player 2 target-network weights
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Kept identical to QMIX so the two runs are directly comparable.

OBS_DIM    = 48
ACTION_NUM = 54

EPISODES          = 300_000
BATCH_SIZE        = 64
MEMORY_SIZE       = 100_000
WARMUP_EPISODES   = 500
TARGET_SYNC_EVERY = 1_000
EVAL_EVERY        = 50_000
EVAL_GAMES        = 100

LR            = 1e-4
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_STEPS = 500_000

CKPT_PATH = os.path.join(os.path.dirname(__file__), 'dqn_euchre.pt')

RESUME = False  # set True to continue training from CKPT_PATH


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_agent(scope: str) -> DQNAgent:
    return DQNAgent(
        scope=scope,
        action_num=ACTION_NUM,
        state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        learning_rate=LR,
        discount_factor=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_STEPS,
        replay_memory_size=MEMORY_SIZE,
        replay_memory_init_size=BATCH_SIZE,
        batch_size=BATCH_SIZE,
        update_target_estimator_every=TARGET_SYNC_EVERY,
    )


def save(agent0: DQNAgent, agent2: DQNAgent, path: str):
    ckpt = {}
    ckpt.update(agent0.get_state_dict())   # agent0_q_estimator, agent0_target_estimator
    ckpt.update(agent2.get_state_dict())   # agent2_q_estimator, agent2_target_estimator
    torch.save(ckpt, path)
    print(f"DQN checkpoint saved to {path}")


def load(agent0: DQNAgent, agent2: DQNAgent, path: str):
    ckpt = torch.load(path, map_location=agent0.device)
    agent0.load(ckpt)
    agent2.load(ckpt)
    print(f"DQN checkpoint loaded from {path}")


def evaluate(env, agent0: DQNAgent, agent2: DQNAgent,
             opp1: EuchreRuleAgent, opp3: EuchreRuleAgent,
             n_games: int = EVAL_GAMES):
    """Greedy evaluation; returns (win_rate, avg_payoff) for the DQN team."""
    wins  = 0
    total = 0.0
    env.set_agents([agent0, opp1, agent2, opp3])

    for _ in range(n_games):
        state, player_id = env.reset()
        while not env.is_over():
            if player_id == 0:
                action, _ = agent0.eval_step(state)
            elif player_id == 2:
                action, _ = agent2.eval_step(state)
            elif player_id == 1:
                action, _ = opp1.eval_step(state)
            else:
                action, _ = opp3.eval_step(state)
            state, player_id = env.step(action)

        p = env.game.get_payoffs().get(0, 0)
        total += p
        if p > 0:
            wins += 1

    return wins / n_games, total / n_games


# ── Training loop ──────────────────────────────────────────────────────────────

if __name__ == '__main__':

    env = rlcard.make('euchre', config={'num_players': 4})

    agent0 = make_agent('agent0')
    agent2 = make_agent('agent2')
    opp1   = EuchreRuleAgent()
    opp3   = EuchreRuleAgent()

    if RESUME and os.path.exists(CKPT_PATH):
        load(agent0, agent2, CKPT_PATH)
        # Pin epsilon to its trained floor without touching total_t,
        # so the replay buffer still fills normally during warmup.
        for a in (agent0, agent2):
            a.epsilons = np.ones(EPSILON_STEPS) * EPSILON_END
    elif RESUME:
        print(f"Warning: RESUME=True but {CKPT_PATH} not found — starting fresh.")

    env.set_agents([agent0, opp1, agent2, opp3])

    print("=" * 60)
    print("Standalone DQN Euchre Training")
    print("  Team DQN  : Player 0 + Player 2 (independent DQN)")
    print("  Team Rule : Player 1 + Player 3 (fixed rule agents)")
    print(f"  Episodes  : {EPISODES}   Warmup: {WARMUP_EPISODES}")
    print("=" * 60)
    print(f"{'Episode':>8}  {'WinRate':>8}  {'AvgPayoff':>10}")
    print("-" * 32)

    eval_episodes   = []
    eval_win_rates  = []
    eval_avg_payoffs = []

    for ep in range(1, EPISODES + 1):
        trajectories, payoffs = env.run(is_training=True)

        # env.run() does NOT call feed() — it only collects trajectories.
        # We must feed each transition manually so total_t increments,
        # epsilon decays, memory fills, and gradient updates fire.
        for transition in trajectories[0]:
            agent0.feed(transition)
        for transition in trajectories[2]:
            agent2.feed(transition)

        if ep % EVAL_EVERY == 0:
            win_rate, avg_payoff = evaluate(env, agent0, agent2, opp1, opp3)
            eps = agent0.epsilons[min(agent0.total_t, EPSILON_STEPS - 1)]
            print(f"{ep:>8}  {win_rate*100:>7.1f}%  {avg_payoff:>+10.3f}   ε={eps:.3f}")
            eval_episodes.append(ep)
            eval_win_rates.append(win_rate * 100)
            eval_avg_payoffs.append(avg_payoff)
            # restore training agents after evaluate() swapped them
            env.set_agents([agent0, opp1, agent2, opp3])

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("=" * 60)
    win_rate, avg_payoff = evaluate(env, agent0, agent2, opp1, opp3, n_games=200)
    print(f"Final (200 games):  win={win_rate*100:.1f}%  avg_payoff={avg_payoff:+.3f}")

    save(agent0, agent2, CKPT_PATH)

    # ── Learning curve ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(eval_episodes, eval_win_rates, marker='o', linewidth=2, color='steelblue')
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% baseline')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Standalone DQN vs Rule Agents — Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eval_episodes, eval_avg_payoffs, marker='o', linewidth=2, color='darkorange')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, label='break-even')
    ax2.set_ylabel('Avg Payoff')
    ax2.set_xlabel('Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'dqn_learning_curve.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Learning curve saved to {plot_path}")
    plt.show()
