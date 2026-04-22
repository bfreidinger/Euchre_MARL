"""
Compare QMIX against baseline opponent teams and generate slide-ready figures.

This script evaluates one QMIX checkpoint against:
  - Random opponents
  - Rule-based opponents
  - Standalone DQN opponents

It saves the same kinds of outputs as compare_saved_models.py:
  - CSV metrics
  - summary table image
  - win-rate / payoff comparison chart
  - grouped outcome-breakdown chart
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import torch

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.euchre_utils import ACTION_LIST
from bidding_observation import BASE_OBS_DIM, augment_state

from train_qmix import QMIXSystem, OBS_DIM, ACTION_NUM


ROOT = os.path.dirname(os.path.abspath(__file__))
DQN_DIR = os.path.join(ROOT, '..', 'dqn_agent')

QMIX_CHECKPOINTS = {
    'neutral': os.path.join(ROOT, 'qmix_neutral.pt'),
    'base': os.path.join(ROOT, 'qmix_euchre.pt'),
}

OPPONENT_SPECS = [
    ('random', 'Random'),
    ('rule', 'Rule-Based'),
    ('dqn', 'Standalone DQN'),
]


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def make_opponents(opp_type, dqn_ckpt_path=None):
    if opp_type == 'random':
        return RandomAgent(ACTION_NUM), RandomAgent(ACTION_NUM), 'Random'
    if opp_type == 'dqn':
        if dqn_ckpt_path is None or not os.path.exists(dqn_ckpt_path):
            raise FileNotFoundError('DQN checkpoint not found for DQN opponents.')
        opp1 = DQNAgent(scope='agent0', action_num=ACTION_NUM,
                        state_shape=[BASE_OBS_DIM], mlp_layers=[128, 128])
        opp3 = DQNAgent(scope='agent2', action_num=ACTION_NUM,
                        state_shape=[BASE_OBS_DIM], mlp_layers=[128, 128])
        checkpoint = torch.load(dqn_ckpt_path, map_location=opp1.device)
        opp1.load(checkpoint)
        opp3.load(checkpoint)
        return opp1, opp3, 'Standalone DQN'
    return EuchreRuleAgent(), EuchreRuleAgent(), 'Rule-Based'


def build_qmix_team(qmix_path):
    agent0 = DQNAgent(scope='agent0', action_num=ACTION_NUM,
                      state_shape=[OBS_DIM], mlp_layers=[128, 128])
    agent2 = DQNAgent(scope='agent2', action_num=ACTION_NUM,
                      state_shape=[OBS_DIM], mlp_layers=[128, 128])
    qmix = QMIXSystem(agent0, agent2)
    qmix.load(qmix_path)
    return qmix.agent0, qmix.agent2


def init_stats():
    return {
        'hands': 0,
        'wins': 0,
        'losses': 0,
        'total_payoff': 0.0,
        'marched': 0,
        'got_marched': 0,
        'called': 0,
        'opp_called': 0,
        'called_won': 0,
        'called_lost': 0,
        'opp_called_qmix_lost': 0,
        'opp_called_marched': 0,
    }


def evaluate_matchup(env, qmix_path, opp_type, num_hands, dqn_ckpt_path):
    team0, team2 = build_qmix_team(qmix_path)
    opp1, opp3, opp_label = make_opponents(opp_type, dqn_ckpt_path=dqn_ckpt_path)
    env.set_agents([team0, opp1, team2, opp3])
    stats = init_stats()

    for _ in range(num_hands):
        state, player_id = env.reset()
        trump_caller = None

        while not env.is_over():
            if player_id == 0:
                action, _ = team0.eval_step(augment_state(env, state, 0))
            elif player_id == 2:
                action, _ = team2.eval_step(augment_state(env, state, 2))
            elif player_id == 1:
                action, _ = opp1.eval_step(state)
            else:
                action, _ = opp3.eval_step(state)

            action_name = ACTION_LIST[action]
            if trump_caller is None and (action_name == 'pick' or action_name.startswith('call-')):
                trump_caller = player_id

            state, player_id = env.step(action)

        payoff = env.game.get_payoffs().get(0, 0.0)
        stats['hands'] += 1
        stats['total_payoff'] += payoff

        if payoff > 0:
            stats['wins'] += 1
            if payoff == 2:
                stats['marched'] += 1
        else:
            stats['losses'] += 1
            if abs(payoff) == 2:
                stats['got_marched'] += 1

        if trump_caller in (0, 2):
            stats['called'] += 1
            if payoff > 0:
                stats['called_won'] += 1
            else:
                stats['called_lost'] += 1
        elif trump_caller in (1, 3):
            stats['opp_called'] += 1
            if payoff < 0:
                stats['opp_called_qmix_lost'] += 1
                if abs(payoff) == 2:
                    stats['opp_called_marched'] += 1

    return {
        'Opponent': opp_label,
        'Win Rate %': pct(stats['wins'], stats['hands']),
        'Avg Payoff': stats['total_payoff'] / stats['hands'] if stats['hands'] else 0.0,
        'QMIX Marched %': pct(stats['marched'], stats['hands']),
        'QMIX Got Marched %': pct(stats['got_marched'], stats['hands']),
        'QMIX Called + Won %': pct(stats['called_won'], stats['hands']),
        'QMIX Called + Lost %': pct(stats['called_lost'], stats['hands']),
        'Opp Called + QMIX Lost %': pct(stats['opp_called_qmix_lost'], stats['hands']),
        'Opp Called + Marched %': pct(stats['opp_called_marched'], stats['hands']),
        'QMIX Called Win Given Call %': pct(stats['called_won'], stats['called']),
        'QMIX Called Loss Given Call %': pct(stats['called_lost'], stats['called']),
    }


def save_csv(rows, path):
    columns = list(rows[0].keys())
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_table(rows, path, title):
    table_columns = [
        'Opponent',
        'Win Rate %',
        'Avg Payoff',
        'QMIX Marched %',
        'QMIX Got Marched %',
        'QMIX Called + Won %',
        'QMIX Called + Lost %',
        'Opp Called + QMIX Lost %',
        'Opp Called + Marched %',
    ]
    cell_text = []
    for row in rows:
        cell_text.append([
            row['Opponent'],
            f"{row['Win Rate %']:.1f}",
            f"{row['Avg Payoff']:+.3f}",
            f"{row['QMIX Marched %']:.1f}",
            f"{row['QMIX Got Marched %']:.1f}",
            f"{row['QMIX Called + Won %']:.1f}",
            f"{row['QMIX Called + Lost %']:.1f}",
            f"{row['Opp Called + QMIX Lost %']:.1f}",
            f"{row['Opp Called + Marched %']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(15, 2.2 + 0.6 * len(rows)))
    ax.axis('off')
    table = ax.table(
        cellText=cell_text,
        colLabels=table_columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_overview_chart(rows, path, title):
    opponents = [row['Opponent'] for row in rows]
    win_rates = [row['Win Rate %'] for row in rows]
    avg_payoffs = [row['Avg Payoff'] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].bar(opponents, win_rates, color='#2f7ed8')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_title('QMIX Win Rate by Opponent')
    axes[0].axhline(50, color='gray', linestyle='--', linewidth=1)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(opponents, avg_payoffs, color='#f28f43')
    axes[1].set_ylabel('Average Payoff')
    axes[1].set_title('QMIX Avg Payoff by Opponent')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_breakdown_chart(rows, path, title):
    event_specs = [
        ('QMIX Marched %', 'QMIX Marched'),
        ('QMIX Got Marched %', 'QMIX Got Marched'),
        ('QMIX Called + Won %', 'QMIX Called + Won'),
        ('QMIX Called + Lost %', 'QMIX Called + Lost'),
        ('Opp Called + QMIX Lost %', 'Opp Called + QMIX Lost'),
        ('Opp Called + Marched %', 'Opp Called + Marched'),
    ]

    x = np.arange(len(rows))
    width = 0.12
    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, (key, label) in enumerate(event_specs):
        values = [row[key] for row in rows]
        ax.bar(x + (idx - (len(event_specs) - 1) / 2) * width, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([row['Opponent'] for row in rows])
    ax.set_ylabel('Percent of Hands')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9, ncols=2)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare QMIX vs baseline opponent teams')
    parser.add_argument('--hands', type=int, default=1000, help='Number of hands per matchup')
    parser.add_argument('--qmix', choices=sorted(QMIX_CHECKPOINTS.keys()), default='neutral',
                        help='Which QMIX checkpoint to evaluate')
    parser.add_argument('--outdir', type=str, default=os.path.join(ROOT, 'qmix_vs_baselines_outputs'),
                        help='Directory for CSV and image outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    qmix_path = QMIX_CHECKPOINTS[args.qmix]
    if not os.path.exists(qmix_path):
        raise FileNotFoundError(f"QMIX checkpoint not found: {qmix_path}")

    env = rlcard.make('euchre', config={'num_players': 4})
    dqn_ckpt_path = os.path.join(DQN_DIR, 'dqn_euchre.pt')

    rows = []
    for opp_type, opp_label in OPPONENT_SPECS:
        print(f"Evaluating QMIX ({args.qmix}) vs {opp_label} over {args.hands} hands...")
        row = evaluate_matchup(env, qmix_path, opp_type, args.hands, dqn_ckpt_path)
        rows.append(row)
        print(
            f"  win={row['Win Rate %']:.1f}% | payoff={row['Avg Payoff']:+.3f} | "
            f"marched={row['QMIX Marched %']:.1f}% | got_marched={row['QMIX Got Marched %']:.1f}%"
        )

    suffix = f"qmix_{args.qmix}"
    csv_path = os.path.join(args.outdir, f'{suffix}_vs_baselines.csv')
    table_path = os.path.join(args.outdir, f'{suffix}_vs_baselines_table.png')
    overview_path = os.path.join(args.outdir, f'{suffix}_vs_baselines_overview.png')
    breakdown_path = os.path.join(args.outdir, f'{suffix}_vs_baselines_breakdown.png')

    save_csv(rows, csv_path)
    save_summary_table(rows, table_path, f'QMIX ({args.qmix.title()}) vs Baseline Opponents')
    save_overview_chart(rows, overview_path, f'QMIX ({args.qmix.title()}) Performance by Opponent')
    save_breakdown_chart(rows, breakdown_path, f'QMIX ({args.qmix.title()}) Outcome Breakdown by Opponent')

    print("\nSaved outputs:")
    print(f"  CSV       : {csv_path}")
    print(f"  Table PNG : {table_path}")
    print(f"  Overview  : {overview_path}")
    print(f"  Breakdown : {breakdown_path}")


if __name__ == '__main__':
    main()
