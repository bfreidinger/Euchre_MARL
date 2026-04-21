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
all_summary_rows = []


def init_hand_stats():
    return {
        'match_wins': 0,
        'match_losses': 0,
        'total_hands': 0,
        'hand_wins': 0,
        'hand_losses': 0,
        'qmix_marches': 0,
        'qmix_got_marched': 0,
        'qmix_called': 0,
        'opp_called': 0,
        'qmix_called_and_won': 0,
        'qmix_called_and_lost': 0,
        'opp_called_and_qmix_lost': 0,
        'opp_called_and_marched': 0,
    }


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def summarize_stats(opp_label, match_results, stats):
    total_matches = len(match_results)
    row = {
        'Opponent': opp_label,
        'Match Win %': pct(stats['match_wins'], total_matches),
        'Hand Win %': pct(stats['hand_wins'], stats['total_hands']),
        'QMIX Marched %': pct(stats['qmix_marches'], stats['total_hands']),
        'Got Marched %': pct(stats['qmix_got_marched'], stats['total_hands']),
        'QMIX Called Win %': pct(stats['qmix_called_and_won'], stats['total_hands']),
        'QMIX Called Lost %': pct(stats['qmix_called_and_lost'], stats['total_hands']),
        'Opp Called QMIX Loss %': pct(stats['opp_called_and_qmix_lost'], stats['total_hands']),
        'Opp Called Marched %': pct(stats['opp_called_and_marched'], stats['total_hands']),
    }
    return row

for opp_type in OPPONENTS:
    opp1, opp3, opp_label = make_opponents(opp_type)
    env.set_agents([agent0, opp1, agent2, opp3])

    qmix_match_wins = 0
    total_hands     = 0
    game_results    = []
    hand_stats      = init_hand_stats()

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
                    action, _ = qmix.agent0.eval_step(state)
                elif player_id == 2:
                    action, _ = qmix.agent2.eval_step(state)
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

            if qmix_hand > 0:
                qmix_score += qmix_hand
                pts       = qmix_hand
                winner    = 'QMIX' + (' (march!)' if qmix_hand == 2 else '')
                hand_stats['hand_wins'] += 1
                if qmix_hand == 2:
                    hand_stats['qmix_marches'] += 1
            else:
                opp_score += abs(qmix_hand)
                pts       = abs(qmix_hand)
                winner    = 'OPP ' + (' (march!)' if abs(qmix_hand) == 2 else '')
                hand_stats['hand_losses'] += 1
                if abs(qmix_hand) == 2:
                    hand_stats['qmix_got_marched'] += 1

            if trump_caller in (0, 2):
                hand_stats['qmix_called'] += 1
                if qmix_hand > 0:
                    hand_stats['qmix_called_and_won'] += 1
                else:
                    hand_stats['qmix_called_and_lost'] += 1
            elif trump_caller in (1, 3):
                hand_stats['opp_called'] += 1
                if qmix_hand < 0:
                    hand_stats['opp_called_and_qmix_lost'] += 1
                    if abs(qmix_hand) == 2:
                        hand_stats['opp_called_and_marched'] += 1

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

    print(f"\nResults vs {opp_label}:")
    print(f"  QMIX: {qmix_match_wins}/{NUM_GAMES} ({100*qmix_match_wins/NUM_GAMES:.1f}%)")
    print(f"  Avg hands per game: {total_hands/NUM_GAMES:.1f}")
    print(f"  Hand win rate: {pct(hand_stats['hand_wins'], hand_stats['total_hands']):.1f}%")
    print(f"  QMIX marched: {hand_stats['qmix_marches']} ({pct(hand_stats['qmix_marches'], hand_stats['total_hands']):.1f}%)")
    print(f"  QMIX got marched: {hand_stats['qmix_got_marched']} ({pct(hand_stats['qmix_got_marched'], hand_stats['total_hands']):.1f}%)")
    print(f"  QMIX called and won: {hand_stats['qmix_called_and_won']} "
          f"({pct(hand_stats['qmix_called_and_won'], hand_stats['total_hands']):.1f}% of hands, "
          f"{pct(hand_stats['qmix_called_and_won'], hand_stats['qmix_called']):.1f}% of QMIX calls)")
    print(f"  QMIX called and lost: {hand_stats['qmix_called_and_lost']} "
          f"({pct(hand_stats['qmix_called_and_lost'], hand_stats['total_hands']):.1f}% of hands, "
          f"{pct(hand_stats['qmix_called_and_lost'], hand_stats['qmix_called']):.1f}% of QMIX calls)")
    print(f"  Opp called and QMIX lost: {hand_stats['opp_called_and_qmix_lost']} "
          f"({pct(hand_stats['opp_called_and_qmix_lost'], hand_stats['total_hands']):.1f}% of hands)")
    print(f"  Opp called and marched: {hand_stats['opp_called_and_marched']} "
          f"({pct(hand_stats['opp_called_and_marched'], hand_stats['total_hands']):.1f}% of hands)")

    all_results[opp_label] = np.array(game_results)
    all_summary_rows.append(summarize_stats(opp_label, game_results, hand_stats))

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
