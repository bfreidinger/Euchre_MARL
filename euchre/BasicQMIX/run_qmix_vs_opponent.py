"""
Evaluate a saved QMIX checkpoint against random, rule-based, and DQN opponents.
Runs all three matchups and plots win rate curves on the same graph.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import rlcard
import matplotlib.pyplot as plt
import numpy as np
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from train_qmix import QMIXSystem, OBS_DIM, ACTION_NUM, MIX_EMBED

# ── Configuration ──────────────────────────────────────────────────────────────

NUM_GAMES  = 500
SHOW_HANDS = True
WIN_TARGET = 10

DQN_CKPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'dqn_agent', 'dqn_euchre.pt')

OPPONENTS = ['random', 'rule', 'dqn']

# ── Build QMIX once ────────────────────────────────────────────────────────────

agent0 = DQNAgent(scope='agent0', action_num=ACTION_NUM,
                  state_shape=[OBS_DIM], mlp_layers=[128, 128])
agent2 = DQNAgent(scope='agent2', action_num=ACTION_NUM,
                  state_shape=[OBS_DIM], mlp_layers=[128, 128])

qmix = QMIXSystem(agent0, agent2)
ckpt_path = os.path.join(os.path.dirname(__file__), 'qmix_euchre.pt')
qmix.load(ckpt_path)

env = rlcard.make('euchre', config={'num_players': 4})

# ── Helper: build opponent agents ──────────────────────────────────────────────

def make_opponents(opp_type):
    if opp_type == 'random':
        return RandomAgent(ACTION_NUM), RandomAgent(ACTION_NUM), 'Random'
    elif opp_type == 'dqn':
        opp1 = DQNAgent(scope='agent0', action_num=ACTION_NUM,
                        state_shape=[OBS_DIM], mlp_layers=[128, 128])
        opp3 = DQNAgent(scope='agent2', action_num=ACTION_NUM,
                        state_shape=[OBS_DIM], mlp_layers=[128, 128])
        dqn_ckpt = torch.load(DQN_CKPT_PATH, map_location=opp1.device)
        opp1.load(dqn_ckpt)
        opp3.load(dqn_ckpt)
        return opp1, opp3, 'DQN'
    else:
        return EuchreRuleAgent(), EuchreRuleAgent(), 'Rule-Based'

# ── Run all matchups ───────────────────────────────────────────────────────────

all_results = {}  # opp_label -> np.array of 1/0 per game

for opp_type in OPPONENTS:
    opp1, opp3, opp_label = make_opponents(opp_type)
    env.set_agents([agent0, opp1, agent2, opp3])

    qmix_match_wins = 0
    total_hands     = 0
    game_results    = []

    print(f"\n{'='*60}")
    print(f"Evaluating QMIX vs {opp_label}  ({NUM_GAMES} games, first to {WIN_TARGET} pts)")
    print('='*60)

    for game_idx in range(NUM_GAMES):
        qmix_score = 0
        opp_score  = 0
        hands_this_game = 0

        print(f"\nGame {game_idx + 1}  (QMIX vs {opp_label})")
        print(f"  {'Hand':<6} {'Winner':<22} {'Pts':<5} QMIX {qmix_score:>2} — {opp_score:<2} OPP")
        print(f"  {'-'*52}")

        while qmix_score < WIN_TARGET and opp_score < WIN_TARGET:
            state, player_id = env.reset()

            while not env.is_over():
                if player_id == 0:
                    action, _ = qmix.agent0.eval_step(state)
                elif player_id == 2:
                    action, _ = qmix.agent2.eval_step(state)
                elif player_id == 1:
                    action, _ = opp1.eval_step(state)
                else:
                    action, _ = opp3.eval_step(state)
                state, player_id = env.step(action)

            payoffs   = env.game.get_payoffs()
            qmix_hand = payoffs.get(0, 0)

            if qmix_hand > 0:
                qmix_score += qmix_hand
                pts       = qmix_hand
                winner    = 'QMIX' + (' (march!)' if qmix_hand == 2 else '')
            else:
                opp_score += abs(qmix_hand)
                pts       = abs(qmix_hand)
                winner    = 'OPP ' + (' (march!)' if abs(qmix_hand) == 2 else '')

            score_str = f"QMIX {qmix_score:>2} — {opp_score:<2} OPP"
            hands_this_game += 1
            if SHOW_HANDS:
                print(f"  Hand {hands_this_game:<3}  {winner:<22} +{pts}    {score_str}")

        winner_label = 'QMIX' if qmix_score >= WIN_TARGET else 'OPP'
        print(f"  {'-'*52}")
        print(f"  >> {winner_label} wins game {game_idx + 1} "
              f"({qmix_score}–{opp_score}) in {hands_this_game} hands")

        if qmix_score >= WIN_TARGET:
            qmix_match_wins += 1
            game_results.append(1)
        else:
            game_results.append(0)
        total_hands += hands_this_game

    print(f"\nResults vs {opp_label}:")
    print(f"  QMIX: {qmix_match_wins}/{NUM_GAMES} ({100*qmix_match_wins/NUM_GAMES:.1f}%)")
    print(f"  Avg hands per game: {total_hands/NUM_GAMES:.1f}")

    all_results[opp_label] = np.array(game_results)

# ── Win rate plot ───────────────────────────────────────────────────────────────

WINDOW = max(1, NUM_GAMES // 10)
COLORS = {'Random': '#2196f3', 'Rule-Based': '#e53935', 'DQN': '#43a047'}

game_numbers = np.arange(1, NUM_GAMES + 1)
rolling_x    = np.arange(WINDOW, NUM_GAMES + 1)

fig, ax = plt.subplots(figsize=(11, 5))

for opp_label, results in all_results.items():
    rolling_wr = np.convolve(results, np.ones(WINDOW) / WINDOW, mode='valid')
    final_wr   = results.mean() * 100
    ax.plot(rolling_x, rolling_wr * 100,
            color=COLORS[opp_label], linewidth=2.2,
            label=f'vs {opp_label}  (final {final_wr:.1f}%)')

ax.axhline(50, color='#9e9e9e', linewidth=1.2, linestyle='--', label='50% baseline')

ax.set_xlim(1, NUM_GAMES)
ax.set_ylim(0, 100)
ax.set_xlabel('Game', fontsize=12)
ax.set_ylabel('Win Rate (%)', fontsize=12)
ax.set_title(f'QMIX Win Rate vs All Opponents  ({NUM_GAMES} games, first to {WIN_TARGET} pts)',
             fontsize=14, fontweight='bold', pad=12)

ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'qmix_vs_all_winrate.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nWin rate graph saved to: {out_path}")
plt.show()
