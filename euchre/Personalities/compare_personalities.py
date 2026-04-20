"""
Compare QMIX personality agents (timid, aggressive, mixed) head-to-head.
Each "game" is a full match played to 10 points (multiple hands).

Usage:
    python compare_personalities.py
    python compare_personalities.py --games 500
    python compare_personalities.py --target_score 10
    python compare_personalities.py --opponent random
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from Personalities.train_qmix_personalities import QMIXSystem, OBS_DIM, ACTION_NUM, MIX_EMBED

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--opponent', choices=['random', 'rule'], default='rule',
                    help='Opponent type (default: rule)')
parser.add_argument('--games', type=int, default=500,
                    help='Number of full games to play (default: 500)')
parser.add_argument('--target_score', type=int, default=10,
                    help='Points needed to win a game (default: 10)')
args = parser.parse_args()

NUM_GAMES    = args.games
TARGET_SCORE = args.target_score
CKPT_DIR     = os.path.dirname(os.path.abspath(__file__))

PERSONALITIES = ['timid', 'aggressive', 'mixed']
COLORS        = {'timid': '#5B9BD5', 'aggressive': '#E74C3C', 'mixed': '#2ECC71'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_qmix():
    """Create a fresh QMIXSystem with untrained DQN agents."""
    agent0 = DQNAgent(
        scope='agent0', action_num=ACTION_NUM,
        state_shape=[OBS_DIM], mlp_layers=[128, 128],
    )
    agent2 = DQNAgent(
        scope='agent2', action_num=ACTION_NUM,
        state_shape=[OBS_DIM], mlp_layers=[128, 128],
    )
    return QMIXSystem(agent0, agent2)


def play_one_hand(qmix, env, opp_agents):
    """
    Play a single hand and return (payoff, hand_stats dict).
    hand_stats keys:
        team_called   – True if QMIX team (player 0 or 2) called trump
        tricks_won    – number of tricks won by QMIX team
        is_march      – True if QMIX team won all 5 tricks
        is_euchre     – True if QMIX team called trump but lost
        got_euchred   – True if opponents called trump but lost (QMIX euchred them)
    """
    state, player_id = env.reset()
    while not env.is_over():
        if player_id == 0:
            action, _ = qmix.agent0.eval_step(state)
        elif player_id == 2:
            action, _ = qmix.agent2.eval_step(state)
        else:
            action, _ = opp_agents[player_id].eval_step(state)
        state, player_id = env.step(action)

    game = env.game
    payoff = game.get_payoffs().get(0, 0)
    team_tricks = game.score[0] + game.score[2]
    caller = getattr(game, 'calling_player', None)
    team_called = caller in (0, 2)

    return payoff, {
        'team_called':  team_called,
        'tricks_won':   team_tricks,
        'is_march':     team_called and team_tricks == 5,
        'is_euchre':    team_called and payoff < 0,
        'got_euchred':  (not team_called) and payoff > 0 and team_tricks >= 3,
    }


def evaluate(qmix, env, opp_agents, num_games, target_score, seed=42):
    """
    Play num_games full games (each first to target_score points).
    Returns (wins, avg_margin, per-game margins, aggregate hand stats).
    """
    np.random.seed(seed)
    wins = 0
    margins = []

    # Aggregate hand-level stats across all games
    total_hands      = 0
    total_hands_won  = 0
    total_tricks     = 0
    total_trump_calls = 0
    total_marches    = 0
    total_euchres    = 0     # times QMIX called and lost
    total_euchred_opp = 0    # times QMIX euchred the opponents

    for _ in range(num_games):
        qmix_score = 0
        opp_score  = 0

        while qmix_score < target_score and opp_score < target_score:
            payoff, hand_stats = play_one_hand(qmix, env, opp_agents)
            total_hands += 1

            if payoff > 0:
                qmix_score += abs(payoff)
                total_hands_won += 1
            else:
                opp_score += abs(payoff)

            total_tricks      += hand_stats['tricks_won']
            total_trump_calls += int(hand_stats['team_called'])
            total_marches     += int(hand_stats['is_march'])
            total_euchres     += int(hand_stats['is_euchre'])
            total_euchred_opp += int(hand_stats['got_euchred'])

        margin = qmix_score - opp_score
        margins.append(margin)
        if qmix_score >= target_score:
            wins += 1

    hand_stats_agg = {
        'total_hands':       total_hands,
        'hands_won':         total_hands_won,
        'hand_win_rate':     100 * total_hands_won / total_hands,
        'avg_tricks':        total_tricks / total_hands,
        'trump_calls':       total_trump_calls,
        'trump_call_rate':   100 * total_trump_calls / total_hands,
        'marches':           total_marches,
        'march_rate':        100 * total_marches / total_hands,
        'euchres':           total_euchres,
        'euchre_rate':       100 * total_euchres / total_hands,
        'euchred_opp':       total_euchred_opp,
        'euchred_opp_rate':  100 * total_euchred_opp / total_hands,
    }

    return wins, np.mean(margins), margins, hand_stats_agg


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    env = rlcard.make('euchre', config={'num_players': 4})

    opp_label = args.opponent.capitalize()
    results = {}  # personality -> {wins, win_rate, avg_payoff, payoffs}

    for name in PERSONALITIES:
        ckpt = os.path.join(CKPT_DIR, f'qmix_{name}.pt')
        if not os.path.exists(ckpt):
            print(f"  ⚠ Checkpoint not found: {ckpt} — skipping {name}")
            continue

        qmix = build_qmix()
        qmix.load(ckpt)

        # Build fresh opponents per personality so state is clean
        if args.opponent == 'random':
            opp_agents = {1: RandomAgent(ACTION_NUM), 3: RandomAgent(ACTION_NUM)}
        else:
            opp_agents = {1: EuchreRuleAgent(), 3: EuchreRuleAgent()}

        env.set_agents([qmix.agent0, opp_agents[1], qmix.agent2, opp_agents[3]])

        print(f"Evaluating {name:>12} over {NUM_GAMES} games (first to {TARGET_SCORE}) vs {opp_label}…")
        wins, avg_margin, margins, hand_stats = evaluate(qmix, env, opp_agents, NUM_GAMES, TARGET_SCORE)
        win_rate = 100 * wins / NUM_GAMES
        results[name] = dict(wins=wins, win_rate=win_rate,
                             avg_margin=avg_margin, margins=margins,
                             **hand_stats)
        print(f"  {name:>12}: wins={wins}/{NUM_GAMES} ({win_rate:.1f}%)  "
              f"avg margin={avg_margin:+.1f}  "
              f"hands={hand_stats['total_hands']}  "
              f"trump calls={hand_stats['trump_call_rate']:.1f}%  "
              f"euchres={hand_stats['euchre_rate']:.1f}%")

    if not results:
        print("No checkpoints found — nothing to plot.")
        sys.exit(1)

    # ── Compute 95% confidence intervals ─────────────────────────────────────
    for name in PERSONALITIES:
        if name not in results:
            continue
        r = results[name]
        p_hat = r['wins'] / NUM_GAMES
        # Wilson score interval for win rate
        z = 1.96
        denom = 1 + z**2 / NUM_GAMES
        center = (p_hat + z**2 / (2 * NUM_GAMES)) / denom
        half_w = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * NUM_GAMES)) / NUM_GAMES) / denom
        r['wr_ci_low']  = (center - half_w) * 100
        r['wr_ci_high'] = (center + half_w) * 100
        r['wr_ci_err']  = half_w * 100

        # CI for average margin
        margins_arr = np.array(r['margins'])
        sem = stats.sem(margins_arr)
        ci = sem * 1.96
        r['margin_ci_low']  = r['avg_margin'] - ci
        r['margin_ci_high'] = r['avg_margin'] + ci
        r['margin_ci_err']  = ci

    # ── Print summary tables ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  Each game is first to {TARGET_SCORE} points  |  {NUM_GAMES} games per personality")
    print("-" * 78)
    print(f"{'Personality':>12}  {'Wins':>6}  {'Win %':>7}  {'95% CI':>15}  {'Avg Margin':>11}  {'95% CI':>15}")
    print("-" * 78)
    for name in PERSONALITIES:
        if name in results:
            r = results[name]
            print(f"{name:>12}  {r['wins']:>6}  {r['win_rate']:>6.1f}%  "
                  f"[{r['wr_ci_low']:>5.1f}, {r['wr_ci_high']:>5.1f}]%  "
                  f"{r['avg_margin']:>+11.1f}  "
                  f"[{r['margin_ci_low']:>+6.1f}, {r['margin_ci_high']:>+6.1f}]")
    print("=" * 78)

    # ── Detailed hand-level stats table ───────────────────────────────────────
    print(f"\n{'':>12}  {'Hands':>7}  {'Hand Win%':>9}  {'Avg Tricks':>10}  "
          f"{'Trump Call%':>11}  {'March%':>7}  {'Euchre%':>8}  {'Euchred Opp%':>12}")
    print("-" * 88)
    for name in PERSONALITIES:
        if name in results:
            r = results[name]
            print(f"{name:>12}  {r['total_hands']:>7}  {r['hand_win_rate']:>8.1f}%  "
                  f"{r['avg_tricks']:>10.2f}  {r['trump_call_rate']:>10.1f}%  "
                  f"{r['march_rate']:>6.1f}%  {r['euchre_rate']:>7.1f}%  "
                  f"{r['euchred_opp_rate']:>11.1f}%")
    print("=" * 88)

    # ── Plot ──────────────────────────────────────────────────────────────────
    names       = [n for n in PERSONALITIES if n in results]
    win_rates   = [results[n]['win_rate']      for n in names]
    wr_errs     = [results[n]['wr_ci_err']     for n in names]
    avg_margins = [results[n]['avg_margin']     for n in names]
    margin_errs = [results[n]['margin_ci_err']  for n in names]
    colors      = [COLORS[n] for n in names]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, top=0.90)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── (1) Game win rate ─────────────────────────────────────────────────────
    bars1 = ax1.bar(names, win_rates, yerr=wr_errs, capsize=6,
                    color=colors, edgecolor='black', linewidth=0.8,
                    error_kw={'linewidth': 1.5, 'color': 'black'})
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% baseline')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title(f'Game Win Rate ({NUM_GAMES} games, first to {TARGET_SCORE})')
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(win_rates) + max(wr_errs) + 12)
    for bar, val, err in zip(bars1, win_rates, wr_errs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 1.0,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # ── (2) Average score margin ──────────────────────────────────────────────
    bars2 = ax2.bar(names, avg_margins, yerr=margin_errs, capsize=6,
                    color=colors, edgecolor='black', linewidth=0.8,
                    error_kw={'linewidth': 1.5, 'color': 'black'})
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, label='break-even')
    ax2.set_ylabel('Avg Score Margin')
    ax2.set_title(f'Avg Score Margin')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    max_margin_top = max(m + e for m, e in zip(avg_margins, margin_errs))
    ax2.set_ylim(0, max_margin_top + 0.6)
    for bar, val, err in zip(bars2, avg_margins, margin_errs):
        offset = err + 0.05
        ax2.text(bar.get_x() + bar.get_width() / 2, val + offset,
                 f'{val:+.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── (3) Hand-level rates: trump calls, marches, euchres ───────────────────
    stat_labels = ['Trump Call %', 'March %', 'Euchre %', 'Euchred Opp %']
    stat_keys   = ['trump_call_rate', 'march_rate', 'euchre_rate', 'euchred_opp_rate']

    x = np.arange(len(stat_labels))
    bar_width = 0.25
    for i, name in enumerate(names):
        vals = [results[name][k] for k in stat_keys]
        ax3.bar(x + i * bar_width, vals, bar_width, label=name,
                color=COLORS[name], edgecolor='black', linewidth=0.6)
        for j, v in enumerate(vals):
            ax3.text(x[j] + i * bar_width, v + 0.3,
                     f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax3.set_xticks(x + bar_width * (len(names) - 1) / 2)
    ax3.set_xticklabels(stat_labels)
    ax3.set_ylabel('Rate (%)')
    ax3.set_title('Hand-Level Event Rates')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # ── (4) Hand win rate & avg tricks per hand ───────────────────────────────
    hand_wrs   = [results[n]['hand_win_rate'] for n in names]
    avg_tricks = [results[n]['avg_tricks']    for n in names]

    x2 = np.arange(2)
    bar_width2 = 0.22
    for i, name in enumerate(names):
        vals = [results[name]['hand_win_rate'], results[name]['avg_tricks']]
        bars = ax4.bar(x2 + i * bar_width2, vals, bar_width2, label=name,
                       color=COLORS[name], edgecolor='black', linewidth=0.6)
        for j, v in enumerate(vals):
            fmt = f'{v:.1f}%' if j == 0 else f'{v:.2f}'
            ax4.text(x2[j] + i * bar_width2, v + 0.3, fmt,
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_xticks(x2 + bar_width2 * (len(names) - 1) / 2)
    ax4.set_xticklabels(['Hand Win Rate (%)', 'Avg Tricks / Hand'])
    ax4.set_title('Per-Hand Performance')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    fig.suptitle(f'QMIX Personality Comparison vs {opp_label} (First to {TARGET_SCORE})',
                 fontsize=15, fontweight='bold')
    plot_path = os.path.join(CKPT_DIR, 'personality_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")
    plt.show()
