"""
Compare saved Euchre policies and generate slide-ready figures.

By default this evaluates all available learned team checkpoints against
rule-based opponents over many independent hands, then saves:

  - a summary table image
  - a win-rate / payoff comparison chart
  - a grouped outcome-breakdown chart
  - a CSV with the same metrics
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
PERSONALITIES_DIR = os.path.join(ROOT, '..', 'Personalities')
DQN_DIR = os.path.join(ROOT, '..', 'dqn_agent')


MODEL_SPECS = [
    {
        'key': 'random',
        'label': 'Random Team',
        'kind': 'baseline_random',
        'path': None,
    },
    {
        'key': 'rule',
        'label': 'Rule-Based Team',
        'kind': 'baseline_rule',
        'path': None,
    },
    {
        'key': 'dqn',
        'label': 'Standalone DQN',
        'kind': 'dqn',
        'path': os.path.join(DQN_DIR, 'dqn_euchre.pt'),
    },
    {
        'key': 'qmix_neutral',
        'label': 'QMIX Neutral',
        'kind': 'qmix',
        'path': os.path.join(ROOT, 'qmix_neutral.pt'),
    },
    {
        'key': 'qmix_aggressive',
        'label': 'QMIX Aggressive',
        'kind': 'qmix',
        'path': os.path.join(PERSONALITIES_DIR, 'qmix_aggressive.pt'),
    },
    {
        'key': 'qmix_timid',
        'label': 'QMIX Timid',
        'kind': 'qmix',
        'path': os.path.join(PERSONALITIES_DIR, 'qmix_timid.pt'),
    },
]


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def available_specs():
    specs = []
    for spec in MODEL_SPECS:
        if spec['path'] is None or os.path.exists(spec['path']):
            specs.append(spec)
    return specs


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
        return opp1, opp3, 'DQN'
    return EuchreRuleAgent(), EuchreRuleAgent(), 'Rule-Based'


def build_team(spec):
    if spec['kind'] == 'baseline_random':
        return RandomAgent(ACTION_NUM), RandomAgent(ACTION_NUM), False

    if spec['kind'] == 'baseline_rule':
        return EuchreRuleAgent(), EuchreRuleAgent(), False

    if spec['kind'] == 'dqn':
        agent0 = DQNAgent(scope='agent0', action_num=ACTION_NUM,
                          state_shape=[BASE_OBS_DIM], mlp_layers=[128, 128])
        agent2 = DQNAgent(scope='agent2', action_num=ACTION_NUM,
                          state_shape=[BASE_OBS_DIM], mlp_layers=[128, 128])
        checkpoint = torch.load(spec['path'], map_location=agent0.device)
        agent0.load(checkpoint)
        agent2.load(checkpoint)
        return agent0, agent2, False

    if spec['kind'] == 'qmix':
        agent0 = DQNAgent(scope='agent0', action_num=ACTION_NUM,
                          state_shape=[OBS_DIM], mlp_layers=[128, 128])
        agent2 = DQNAgent(scope='agent2', action_num=ACTION_NUM,
                          state_shape=[OBS_DIM], mlp_layers=[128, 128])
        qmix = QMIXSystem(agent0, agent2)
        qmix.load(spec['path'])
        return qmix.agent0, qmix.agent2, True

    raise ValueError(f"Unknown model kind: {spec['kind']}")


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


def evaluate_team(env, spec, opp1, opp3, num_hands):
    team0, team2, use_augmented_obs = build_team(spec)
    env.set_agents([team0, opp1, team2, opp3])
    stats = init_stats()

    for _ in range(num_hands):
        state, player_id = env.reset()
        trump_caller = None

        while not env.is_over():
            if player_id == 0:
                action_state = augment_state(env, state, 0) if use_augmented_obs else state
                action, _ = team0.eval_step(action_state)
            elif player_id == 2:
                action_state = augment_state(env, state, 2) if use_augmented_obs else state
                action, _ = team2.eval_step(action_state)
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
        'Model': spec['label'],
        'Win Rate %': pct(stats['wins'], stats['hands']),
        'Avg Payoff': stats['total_payoff'] / stats['hands'] if stats['hands'] else 0.0,
        'Marched %': pct(stats['marched'], stats['hands']),
        'Got Marched %': pct(stats['got_marched'], stats['hands']),
        'Called + Won %': pct(stats['called_won'], stats['hands']),
        'Called + Lost %': pct(stats['called_lost'], stats['hands']),
        'Opp Called + Loss %': pct(stats['opp_called_qmix_lost'], stats['hands']),
        'Opp Called + Marched %': pct(stats['opp_called_marched'], stats['hands']),
        'Called Win Given Call %': pct(stats['called_won'], stats['called']),
        'Called Loss Given Call %': pct(stats['called_lost'], stats['called']),
    }


def save_csv(rows, path):
    columns = list(rows[0].keys())
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_table(rows, path, title):
    table_columns = [
        'Model',
        'Win Rate %',
        'Avg Payoff',
        'Marched %',
        'Got Marched %',
        'Called + Won %',
        'Called + Lost %',
        'Opp Called + Loss %',
        'Opp Called + Marched %',
    ]
    cell_text = []
    for row in rows:
        cell_text.append([
            row['Model'],
            f"{row['Win Rate %']:.1f}",
            f"{row['Avg Payoff']:+.3f}",
            f"{row['Marched %']:.1f}",
            f"{row['Got Marched %']:.1f}",
            f"{row['Called + Won %']:.1f}",
            f"{row['Called + Lost %']:.1f}",
            f"{row['Opp Called + Loss %']:.1f}",
            f"{row['Opp Called + Marched %']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(15, 2.2 + 0.55 * len(rows)))
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
    models = [row['Model'] for row in rows]
    win_rates = [row['Win Rate %'] for row in rows]
    avg_payoffs = [row['Avg Payoff'] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].bar(models, win_rates, color='#2f7ed8')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_title('Win Rate by Model')
    axes[0].axhline(50, color='gray', linestyle='--', linewidth=1)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=25)

    axes[1].bar(models, avg_payoffs, color='#f28f43')
    axes[1].set_ylabel('Average Payoff')
    axes[1].set_title('Average Payoff by Model')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=25)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_breakdown_chart(rows, path, title):
    event_specs = [
        ('Marched %', 'Marched'),
        ('Got Marched %', 'Got Marched'),
        ('Called + Won %', 'Called + Won'),
        ('Called + Lost %', 'Called + Lost'),
        ('Opp Called + Loss %', 'Opp Called + Loss'),
        ('Opp Called + Marched %', 'Opp Called + Marched'),
    ]

    x = np.arange(len(rows))
    width = 0.12
    fig, ax = plt.subplots(figsize=(15, 6.5))

    for idx, (key, label) in enumerate(event_specs):
        values = [row[key] for row in rows]
        ax.bar(x + (idx - (len(event_specs) - 1) / 2) * width, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([row['Model'] for row in rows], rotation=25, ha='right')
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
    parser = argparse.ArgumentParser(description='Compare saved Euchre models and generate presentation figures')
    parser.add_argument('--hands', type=int, default=1000, help='Number of hands per model evaluation')
    parser.add_argument('--opponent', choices=['rule', 'random', 'dqn'], default='rule', help='Opponent team type')
    parser.add_argument('--outdir', type=str, default=os.path.join(ROOT, 'comparison_outputs'),
                        help='Directory for CSV and image outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    env = rlcard.make('euchre', config={'num_players': 4})
    dqn_ckpt_path = os.path.join(DQN_DIR, 'dqn_euchre.pt')
    opp1, opp3, opp_label = make_opponents(args.opponent, dqn_ckpt_path=dqn_ckpt_path)

    rows = []
    for spec in available_specs():
        print(f"Evaluating {spec['label']} vs {opp_label} over {args.hands} hands...")
        row = evaluate_team(env, spec, opp1, opp3, args.hands)
        rows.append(row)
        print(
            f"  win={row['Win Rate %']:.1f}% | payoff={row['Avg Payoff']:+.3f} | "
            f"marched={row['Marched %']:.1f}% | got_marched={row['Got Marched %']:.1f}%"
        )

    csv_path = os.path.join(args.outdir, f'model_comparison_vs_{args.opponent}.csv')
    table_path = os.path.join(args.outdir, f'model_comparison_table_vs_{args.opponent}.png')
    overview_path = os.path.join(args.outdir, f'model_comparison_overview_vs_{args.opponent}.png')
    breakdown_path = os.path.join(args.outdir, f'model_comparison_breakdown_vs_{args.opponent}.png')

    save_csv(rows, csv_path)
    save_summary_table(rows, table_path, f'Model Comparison vs {opp_label} (%)')
    save_overview_chart(rows, overview_path, f'Model Performance vs {opp_label}')
    save_breakdown_chart(rows, breakdown_path, f'Model Outcome Breakdown vs {opp_label}')

    print("\nSaved outputs:")
    print(f"  CSV       : {csv_path}")
    print(f"  Table PNG : {table_path}")
    print(f"  Overview  : {overview_path}")
    print(f"  Breakdown : {breakdown_path}")


if __name__ == '__main__':
    main()
