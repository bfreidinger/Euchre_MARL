"""
Compare baseline teams to QMIX and generate slide-ready figures.

This is a focused version of compare_saved_models.py that includes:
  - Random Team
  - Rule-Based Team
  - Standalone DQN
  - QMIX Neutral

It saves the same kinds of outputs:
  - CSV metrics
  - summary table image
  - win-rate / payoff comparison chart
  - grouped outcome-breakdown chart
"""

import argparse
import os

from compare_saved_models import (
    DQN_DIR,
    ROOT,
    available_specs,
    evaluate_team,
    make_opponents,
    save_breakdown_chart,
    save_csv,
    save_overview_chart,
    save_summary_table,
)
from euchre import rlcard


INCLUDED_KEYS = {'random', 'rule', 'dqn', 'qmix_neutral'}


def parse_args():
    parser = argparse.ArgumentParser(description='Compare baseline teams against QMIX')
    parser.add_argument('--hands', type=int, default=1000, help='Number of hands per model evaluation')
    parser.add_argument('--opponent', choices=['rule', 'random', 'dqn'], default='rule', help='Opponent team type')
    parser.add_argument('--outdir', type=str, default=os.path.join(ROOT, 'baseline_vs_qmix_outputs'),
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
        if spec['key'] not in INCLUDED_KEYS:
            continue
        print(f"Evaluating {spec['label']} vs {opp_label} over {args.hands} hands...")
        row = evaluate_team(env, spec, opp1, opp3, args.hands)
        rows.append(row)
        print(
            f"  win={row['Win Rate %']:.1f}% | payoff={row['Avg Payoff']:+.3f} | "
            f"marched={row['Marched %']:.1f}% | got_marched={row['Got Marched %']:.1f}%"
        )

    csv_path = os.path.join(args.outdir, f'baseline_vs_qmix_{args.opponent}.csv')
    table_path = os.path.join(args.outdir, f'baseline_vs_qmix_table_{args.opponent}.png')
    overview_path = os.path.join(args.outdir, f'baseline_vs_qmix_overview_{args.opponent}.png')
    breakdown_path = os.path.join(args.outdir, f'baseline_vs_qmix_breakdown_{args.opponent}.png')

    save_csv(rows, csv_path)
    save_summary_table(rows, table_path, f'Baselines vs QMIX ({opp_label} Opponents)')
    save_overview_chart(rows, overview_path, f'Baseline Team Comparison vs {opp_label}')
    save_breakdown_chart(rows, breakdown_path, f'Baseline Outcome Breakdown vs {opp_label}')

    print("\nSaved outputs:")
    print(f"  CSV       : {csv_path}")
    print(f"  Table PNG : {table_path}")
    print(f"  Overview  : {overview_path}")
    print(f"  Breakdown : {breakdown_path}")


if __name__ == '__main__':
    main()
