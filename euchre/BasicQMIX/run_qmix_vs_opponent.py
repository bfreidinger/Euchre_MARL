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
from rlcard.utils.euchre_utils import ACTION_LIST
from bidding_observation import BASE_OBS_DIM, augment_state
from train_qmix import QMIXSystem, OBS_DIM, ACTION_NUM, MIX_EMBED

# ── Configuration ──────────────────────────────────────────────────────────────

NUM_GAMES  = 500
SHOW_HANDS = True
WIN_TARGET = 10

DQN_CKPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'dqn_agent', 'dqn_euchre_shared.pt')

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
        shared = DQNAgent(scope='shared', action_num=ACTION_NUM,
                          state_shape=[BASE_OBS_DIM], mlp_layers=[128, 128])
        dqn_ckpt = torch.load(DQN_CKPT_PATH, map_location=shared.device)
        shared.load(dqn_ckpt)
        return shared, shared, 'DQN'
    else:
        return EuchreRuleAgent(), EuchreRuleAgent(), 'Rule-Based'

# ── Run all matchups ───────────────────────────────────────────────────────────

all_results   = {}   # opp_label -> np.array of 1/0 per game
all_hand_stats = {}  # opp_label -> dict of hand outcome counts

for opp_type in OPPONENTS:
    opp1, opp3, opp_label = make_opponents(opp_type)
    env.set_agents([agent0, opp1, agent2, opp3])

    qmix_match_wins = 0
    total_hands     = 0
    game_results    = []
    hand_stats      = {'march': 0, 'win': 0, 'loss': 0, 'opp_march': 0,
                       'qmix_euchre': 0, 'opp_euchre': 0,
                       'total_hands': 0, 'match_wins': 0, 'match_losses': 0}

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
            trump_caller = None

            while not env.is_over():
                if player_id == 0:
                    action, _ = qmix.agent0.eval_step(augment_state(env, state, 0))
                elif player_id == 2:
                    action, _ = qmix.agent2.eval_step(augment_state(env, state, 2))
                elif player_id == 1:
                    action, _ = opp1.eval_step(state)
                else:
                    action, _ = opp3.eval_step(state)

                action_name = ACTION_LIST[action]
                if trump_caller is None and (action_name == 'pick' or action_name.startswith('call-')):
                    trump_caller = player_id
                state, player_id = env.step(action)

            payoffs   = env.game.get_payoffs()
            qmix_hand = payoffs.get(0, 0)
            hand_stats['total_hands'] += 1

            caller      = env.game.calling_player
            maker_team  = {caller, (caller + 2) % 4}
            was_euchred = (env.game.score[caller] + env.game.score[(caller + 2) % 4]) < 3

            if qmix_hand == 2:
                qmix_score += 2
                if was_euchred:  # QMIX defended and euchred the opponents
                    pts, winner = 2, 'QMIX (euchre!)'
                    hand_stats['qmix_euchre'] += 1
                else:            # QMIX called trump and marched
                    pts, winner = 2, 'QMIX (march!)'
                    hand_stats['march'] += 1
            elif qmix_hand == 1:
                qmix_score += 1
                pts, winner = 1, 'QMIX'
                hand_stats['win'] += 1
            elif qmix_hand == -1:
                opp_score += 1
                pts, winner = 1, 'OPP'
                hand_stats['loss'] += 1
            else:  # -2
                opp_score += 2
                if was_euchred:  # OPP defended and euchred QMIX
                    pts, winner = 2, 'OPP (euchre!)'
                    hand_stats['opp_euchre'] += 1
                else:            # OPP called trump and marched
                    pts, winner = 2, 'OPP (march!)'
                    hand_stats['opp_march'] += 1

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
            hand_stats['match_wins'] += 1
            game_results.append(1)
        else:
            hand_stats['match_losses'] += 1
            game_results.append(0)
        total_hands += hands_this_game

    all_results[opp_label]    = np.array(game_results)
    all_hand_stats[opp_label] = hand_stats

    wr = qmix_match_wins / NUM_GAMES
    ci = 1.96 * np.sqrt(wr * (1 - wr) / NUM_GAMES)
    h  = hand_stats
    print(f"\nResults vs {opp_label}:")
    print(f"  Game win rate : {qmix_match_wins}/{NUM_GAMES}  ({100*wr:.1f}% ± {100*ci:.1f}%)")
    print(f"  Avg hands/game: {total_hands/NUM_GAMES:.1f}")
    print(f"  Hand outcomes : {total_hands} total hands")
    print(f"    QMIX march    : {h['march']:>5}  ({100*h['march']/total_hands:.1f}%)")
    print(f"    QMIX euchre   : {h['qmix_euchre']:>5}  ({100*h['qmix_euchre']/total_hands:.1f}%)")
    print(f"    QMIX win      : {h['win']:>5}  ({100*h['win']/total_hands:.1f}%)")
    print(f"    OPP win       : {h['loss']:>5}  ({100*h['loss']/total_hands:.1f}%)")
    print(f"    OPP euchre    : {h['opp_euchre']:>5}  ({100*h['opp_euchre']/total_hands:.1f}%)")
    print(f"    OPP march     : {h['opp_march']:>5}  ({100*h['opp_march']/total_hands:.1f}%)")

# ── Bar chart ──────────────────────────────────────────────────────────────────

COLORS = {'Random': '#2196f3', 'Rule-Based': '#e53935', 'DQN': '#43a047'}

labels     = list(all_results.keys())
win_rates  = [all_results[l].mean() * 100 for l in labels]
error_bars = [1.96 * np.sqrt((wr/100) * (1 - wr/100) / NUM_GAMES) * 100
              for wr in win_rates]
colors     = [COLORS[l] for l in labels]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(labels, win_rates, yerr=error_bars, capsize=6,
              color=colors, alpha=0.85, width=0.5,
              error_kw={'elinewidth': 1.8, 'ecolor': '#555555'})

ax.axhline(50, color='#9e9e9e', linewidth=1.2, linestyle='--', label='50% baseline')

for bar, wr in zip(bars, win_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
            f'{wr:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylim(0, 100)
ax.set_ylabel('Win Rate (%)', fontsize=12)
ax.set_title(f'QMIX Win Rate vs All Opponents\n'
             f'({NUM_GAMES} games, first to {WIN_TARGET} pts — 95% CI error bars)',
             fontsize=13, fontweight='bold', pad=12)

ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'qmix_vs_all_winrate.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nWin rate graph saved to: {out_path}")
plt.show()

# ── Build summary rows ────────────────────────────────────────────────────────

all_summary_rows = []
for label, h in all_hand_stats.items():
    t = h['total_hands']
    all_summary_rows.append({
        'Opponent':               label,
        'Match Win %':            all_results[label].mean() * 100,
        'Hand Win %':             (h['march'] + h['win'] + h['qmix_euchre']) / t * 100,
        'QMIX Marched %':         h['march'] / t * 100,
        'Got Marched %':          h['opp_march'] / t * 100,
        'QMIX Called Win %':      (h['march'] + h['win']) / t * 100,
        'QMIX Called Lost %':     h['opp_euchre'] / t * 100,
        'Opp Called QMIX Loss %': h['loss'] / t * 100,
        'Opp Called Marched %':   h['opp_march'] / t * 100,
    })

# ── Summary table for presentation ────────────────────────────────────────────

table_columns = [
    'Opponent',
    'Match Win %',
    'Hand Win %',
    'QMIX Marched %',
    'Got Marched %',
    'QMIX Called Win %',
    'QMIX Called Lost %',
    'Opp Called QMIX Loss %',
    'Opp Called Marched %',
]

table_text = []
for row in all_summary_rows:
    table_text.append([
        row['Opponent'],
        f"{row['Match Win %']:.1f}",
        f"{row['Hand Win %']:.1f}",
        f"{row['QMIX Marched %']:.1f}",
        f"{row['Got Marched %']:.1f}",
        f"{row['QMIX Called Win %']:.1f}",
        f"{row['QMIX Called Lost %']:.1f}",
        f"{row['Opp Called QMIX Loss %']:.1f}",
        f"{row['Opp Called Marched %']:.1f}",
    ])

fig, ax = plt.subplots(figsize=(14, 2.2 + 0.55 * len(table_text)))
ax.axis('off')
table = ax.table(
    cellText=table_text,
    colLabels=table_columns,
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
ax.set_title('QMIX Evaluation Summary Table (%)', fontsize=14, fontweight='bold', pad=12)
table_path = os.path.join(os.path.dirname(__file__), 'qmix_summary_table.png')
plt.tight_layout()
plt.savefig(table_path, dpi=180, bbox_inches='tight')
print(f"Summary table saved to: {table_path}")
plt.show()

# ── Presentation-friendly event-rate bars ────────────────────────────────────

event_specs = [
    ('QMIX Marched %', 'QMIX Marched'),
    ('Got Marched %', 'QMIX Got Marched'),
    ('QMIX Called Win %', 'QMIX Called + Won'),
    ('QMIX Called Lost %', 'QMIX Called + Lost'),
    ('Opp Called QMIX Loss %', 'Opp Called + QMIX Lost'),
    ('Opp Called Marched %', 'Opp Called + Marched'),
]

x = np.arange(len(all_summary_rows))
width = 0.12
fig, ax = plt.subplots(figsize=(14, 6))
for idx, (key, label) in enumerate(event_specs):
    values = [row[key] for row in all_summary_rows]
    ax.bar(x + (idx - (len(event_specs) - 1) / 2) * width, values, width=width, label=label)

ax.set_xticks(x)
ax.set_xticklabels([row['Opponent'] for row in all_summary_rows])
ax.set_ylabel('Percent of Hands')
ax.set_title('QMIX Hand Outcome Breakdown by Opponent', fontsize=14, fontweight='bold', pad=12)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=9, ncols=2)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

breakdown_path = os.path.join(os.path.dirname(__file__), 'qmix_outcome_breakdown.png')
plt.tight_layout()
plt.savefig(breakdown_path, dpi=150, bbox_inches='tight')
print(f"Outcome breakdown graph saved to: {breakdown_path}")
plt.show()
