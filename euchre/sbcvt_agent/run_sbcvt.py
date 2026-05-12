"""
Run the SBCVT agent against chosen opponents, games to WIN_TARGET points.

Player assignments:
  0   : SBCVT (MCTS + QMIX)
  2   : partner — 'sbcvt' (also runs MCTS) or 'qmix' (greedy, faster)
  1,3 : opponents — 'random', 'rule', or 'qmix' (from a checkpoint)

Usage:
  python run_sbcvt.py                         # defaults below
  python run_sbcvt.py --ckpt my_model.pt --opp rule --sims 200 --games 50
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from rlcard.utils.euchre_utils import ACTION_LIST

from BasicQMIX.train_qmix import QMIXSystem, OBS_DIM, ACTION_NUM, MIX_EMBED, GLOBAL_DIM
from sbcvt_agent import SBCVTAgent

# ---------------------------------------------------------------------------
# Defaults (override via CLI or edit here)
# ---------------------------------------------------------------------------

DEFAULT_CKPT    = os.path.join(os.path.dirname(__file__), '..', 'BasicQMIX', 'qmix_euchre.pt')
DEFAULT_OPP     = 'rule'   # 'random' | 'rule' | 'qmix'
DEFAULT_PARTNER = 'qmix'     # 'qmix'   | 'sbcvt'
DEFAULT_SIMS    = 100        # MCTS simulations per decision
DEFAULT_DEPTH   = 5          # max rollout depth before QMIX leaf eval
DEFAULT_GAMES   = 100        # number of full games (each first to WIN_TARGET)
WIN_TARGET      = 10

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    default=DEFAULT_CKPT,    help='QMIX checkpoint path')
    p.add_argument('--opp',     default=DEFAULT_OPP,     choices=['random', 'rule', 'qmix'])
    p.add_argument('--partner', default=DEFAULT_PARTNER, choices=['qmix', 'sbcvt'])
    p.add_argument('--sims',    default=DEFAULT_SIMS,    type=int)
    p.add_argument('--depth',   default=DEFAULT_DEPTH,   type=int)
    p.add_argument('--games',   default=DEFAULT_GAMES,   type=int)
    p.add_argument('--quiet',   action='store_true',     help='suppress per-hand output')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------

def build_qmix(ckpt_path: str, device):
    """Load a QMIXSystem from checkpoint and return (qmix, agent0, agent2)."""
    agent0 = DQNAgent(scope='agent0', action_num=ACTION_NUM,
                      state_shape=[OBS_DIM], mlp_layers=[128, 128])
    agent2 = DQNAgent(scope='agent2', action_num=ACTION_NUM,
                      state_shape=[OBS_DIM], mlp_layers=[128, 128])
    qmix = QMIXSystem(agent0, agent2)
    qmix.load(ckpt_path)
    return qmix


def make_opponents(opp_type: str, ckpt_path: str, device):
    """Return (agent1, agent3, label) for the chosen opponent type."""
    if opp_type == 'random':
        return RandomAgent(ACTION_NUM), RandomAgent(ACTION_NUM), 'Random'
    elif opp_type == 'rule':
        return EuchreRuleAgent(), EuchreRuleAgent(), 'Rule-Based'
    else:  # 'qmix'
        opp_qmix = build_qmix(ckpt_path, device)
        return opp_qmix.agent0, opp_qmix.agent2, 'QMIX-Opponent'


# ---------------------------------------------------------------------------
# Hand outcome classification
# ---------------------------------------------------------------------------

def classify_hand(payoff0: float, game) -> tuple[str, int]:
    """
    Returns (label, points_delta_for_sbcvt_team) given player-0's payoff
    and the finished game object.
    """
    caller = game.calling_player
    maker_team = {caller, (caller + 2) % 4}
    euchred = (game.score[caller] + game.score[(caller + 2) % 4]) < 3

    if payoff0 == 2:
        label = 'SBCVT (euchre!)' if euchred else 'SBCVT (march!)'
    elif payoff0 == 1:
        label = 'SBCVT'
    elif payoff0 == -1:
        label = 'OPP'
    else:  # -2
        label = 'OPP (euchre!)' if euchred else 'OPP (march!)'

    return label, int(payoff0)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build SBCVT team
    qmix = build_qmix(args.ckpt, device)
    env  = rlcard.make('euchre', config={'num_players': 4})

    sbcvt0 = SBCVTAgent(
        agent0=qmix.agent0,
        agent2=qmix.agent2,
        mixer=qmix.mixer,
        env=env,
        num_sims=args.sims,
        max_depth=args.depth,
        device=device,
    )

    if args.partner == 'sbcvt':
        partner2 = sbcvt0   # same agent, step() selects for whichever player_id is passed
        partner_label = 'SBCVT'
    else:
        partner2 = qmix.agent2   # greedy QMIX eval_step
        partner_label = 'QMIX-greedy'

    opp1, opp3, opp_label = make_opponents(args.opp, args.ckpt, device)

    # Stats trackers
    hand_stats = {
        'march': 0, 'euchre_for': 0, 'win': 0,
        'loss': 0,  'euchre_against': 0, 'march_against': 0,
        'total_hands': 0,
    }
    game_results = []
    total_hands  = 0

    print(f"\n{'='*62}")
    print(f"  SBCVT (sims={args.sims}, depth={args.depth})")
    print(f"  Partner  : {partner_label}")
    print(f"  Opponents: {opp_label}")
    print(f"  Games    : {args.games} × first to {WIN_TARGET} pts")
    print(f"{'='*62}")

    for game_idx in range(args.games):
        sbcvt_score = 0
        opp_score   = 0
        hands_this_game = 0

        if not args.quiet:
            print(f"\nGame {game_idx + 1}")
            print(f"  {'Hand':<6} {'Winner':<24} {'Pts':<5} SBCVT {sbcvt_score:>2} — {opp_score:<2} OPP")
            print(f"  {'-'*54}")

        while sbcvt_score < WIN_TARGET and opp_score < WIN_TARGET:
            state, player_id = env.reset()
            sbcvt0.new_hand()

            while not env.is_over():
                if player_id == 0:
                    action = sbcvt0.step(state, player_id=0)
                elif player_id == 2:
                    if args.partner == 'sbcvt':
                        action = sbcvt0.step(state, player_id=2)
                    else:
                        action, _ = partner2.eval_step(state)
                elif player_id == 1:
                    action, _ = opp1.eval_step(state)
                else:
                    action, _ = opp3.eval_step(state)

                if player_id in (0, 2):
                    sbcvt0.record_action(player_id, action, env.game)
                state, player_id = env.step(action)

            payoffs = env.game.get_payoffs()
            payoff0 = payoffs.get(0, 0)
            label, delta = classify_hand(payoff0, env.game)

            if delta == 2:
                sbcvt_score += 2
                if 'euchre' in label:
                    hand_stats['euchre_for'] += 1
                else:
                    hand_stats['march'] += 1
            elif delta == 1:
                sbcvt_score += 1
                hand_stats['win'] += 1
            elif delta == -1:
                opp_score += 1
                hand_stats['loss'] += 1
            else:
                opp_score += 2
                if 'euchre' in label:
                    hand_stats['euchre_against'] += 1
                else:
                    hand_stats['march_against'] += 1

            hands_this_game += 1
            hand_stats['total_hands'] += 1

            if not args.quiet:
                score_str = f"SBCVT {sbcvt_score:>2} — {opp_score:<2} OPP"
                print(f"  Hand {hands_this_game:<3}  {label:<24} +{abs(delta)}    {score_str}")

        total_hands += hands_this_game
        won = sbcvt_score >= WIN_TARGET
        game_results.append(1 if won else 0)

        if not args.quiet:
            winner_label = 'SBCVT' if won else 'OPP'
            print(f"  {'-'*54}")
            print(f"  >> {winner_label} wins game {game_idx + 1} "
                  f"({sbcvt_score}–{opp_score}) in {hands_this_game} hands")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    n = args.games
    wins = sum(game_results)
    wr   = wins / n
    ci   = 1.96 * np.sqrt(wr * (1 - wr) / n) if n > 1 else 0.0
    h    = hand_stats
    t    = max(h['total_hands'], 1)

    print(f"\n{'='*62}")
    print(f"  Results: SBCVT vs {opp_label}  ({n} games)")
    print(f"{'='*62}")
    print(f"  Game win rate   : {wins}/{n}  ({100*wr:.1f}% ± {100*ci:.1f}%)")
    print(f"  Avg hands/game  : {total_hands/n:.1f}")
    print(f"\n  Hand outcomes ({t} total):")
    print(f"    SBCVT march      : {h['march']:>5}  ({100*h['march']/t:.1f}%)")
    print(f"    SBCVT euchred opp: {h['euchre_for']:>5}  ({100*h['euchre_for']/t:.1f}%)")
    print(f"    SBCVT won        : {h['win']:>5}  ({100*h['win']/t:.1f}%)")
    print(f"    OPP won          : {h['loss']:>5}  ({100*h['loss']/t:.1f}%)")
    print(f"    OPP euchred SBCVT: {h['euchre_against']:>5}  ({100*h['euchre_against']/t:.1f}%)")
    print(f"    OPP march        : {h['march_against']:>5}  ({100*h['march_against']/t:.1f}%)")
    print(f"{'='*62}\n")

    return game_results


if __name__ == '__main__':
    args = parse_args()
    run(args)
