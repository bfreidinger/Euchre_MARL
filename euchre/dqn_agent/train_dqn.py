"""
Standalone DQN training for Euchre — no cooperative mixing network.

A single shared network plays as both players 0 and 2, receiving transitions
from both positions each episode. This doubles the training signal per update
and forces the agent to learn a policy that works from any team seat.

Checkpoint saved to dqn_euchre.pt with keys:
    shared_q_estimator      — shared Q-network weights
    shared_target_estimator — shared target-network weights
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

OBS_DIM    = 48
ACTION_NUM = 54

EPISODES          = 30_000
BATCH_SIZE        = 64
MEMORY_SIZE       = 100_000
WARMUP_EPISODES   = 200
TARGET_SYNC_EVERY = 1_000
EVAL_EVERY        = 2_000
EVAL_GAMES        = 50

LR            = 1e-4
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_STEPS = 15_000

CKPT_PATH = os.path.join(os.path.dirname(__file__), 'dqn_euchre_shared.pt')

RESUME = True  # set True to continue training from CKPT_PATH


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_agent() -> DQNAgent:
    return DQNAgent(
        scope='shared',
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


def save(agent: DQNAgent, path: str):
    torch.save(agent.get_state_dict(), path)
    print(f"DQN checkpoint saved to {path}")


def load(agent: DQNAgent, path: str):
    ckpt = torch.load(path, map_location=agent.device)
    agent.load(ckpt)
    print(f"DQN checkpoint loaded from {path}")


def evaluate(env, agent: DQNAgent,
             opp1: EuchreRuleAgent, opp3: EuchreRuleAgent,
             n_games: int = EVAL_GAMES):
    """Greedy evaluation; returns (win_rate, avg_payoff) for the DQN team."""
    wins  = 0
    total = 0.0
    env.set_agents([agent, opp1, agent, opp3])

    for _ in range(n_games):
        state, player_id = env.reset()
        while not env.is_over():
            if player_id in (0, 2):
                action, _ = agent.eval_step(state)
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

    agent = make_agent()
    opp1  = EuchreRuleAgent()
    opp3  = EuchreRuleAgent()

    if RESUME and os.path.exists(CKPT_PATH):
        load(agent, CKPT_PATH)
        agent.epsilons = np.ones(EPSILON_STEPS) * EPSILON_END
    elif RESUME:
        print(f"Warning: RESUME=True but {CKPT_PATH} not found — starting fresh.")

    env.set_agents([agent, opp1, agent, opp3])

    print("=" * 60)
    print("Standalone DQN Euchre Training  (shared network)")
    print("  Team DQN  : Player 0 + Player 2 (shared weights)")
    print("  Team Rule : Player 1 + Player 3 (fixed rule agents)")
    print(f"  Episodes  : {EPISODES}   Warmup: {WARMUP_EPISODES}")
    print("=" * 60)
    print(f"{'Episode':>8}  {'WinRate':>8}  {'AvgPayoff':>10}")
    print("-" * 32)

    eval_episodes    = []
    eval_win_rates   = []
    eval_avg_payoffs = []

    for ep in range(1, EPISODES + 1):
        trajectories, payoffs = env.run(is_training=True)

        # Feed transitions from both team positions into the shared agent.
        # This gives 2x experience per episode and trains a position-agnostic policy.
        for transition in trajectories[0]:
            agent.feed(transition)
        for transition in trajectories[2]:
            agent.feed(transition)

        if ep % EVAL_EVERY == 0:
            win_rate, avg_payoff = evaluate(env, agent, opp1, opp3)
            eps = agent.epsilons[min(agent.total_t, EPSILON_STEPS - 1)]
            print(f"{ep:>8}  {win_rate*100:>7.1f}%  {avg_payoff:>+10.3f}   ε={eps:.3f}")
            eval_episodes.append(ep)
            eval_win_rates.append(win_rate * 100)
            eval_avg_payoffs.append(avg_payoff)
            env.set_agents([agent, opp1, agent, opp3])

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("=" * 60)
    win_rate, avg_payoff = evaluate(env, agent, opp1, opp3, n_games=200)
    print(f"Final (200 games):  win={win_rate*100:.1f}%  avg_payoff={avg_payoff:+.3f}")

    save(agent, CKPT_PATH)

    # ── Learning curve ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(eval_episodes, eval_win_rates, marker='o', linewidth=2, color='steelblue')
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% baseline')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Standalone DQN vs Rule Agents — Learning Curves (shared network)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eval_episodes, eval_avg_payoffs, marker='o', linewidth=2, color='darkorange')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, label='break-even')
    ax2.set_ylabel('Avg Payoff')
    ax2.set_xlabel('Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'dqn_learning_curve_shared.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Learning curve saved to {plot_path}")
    plt.show()
