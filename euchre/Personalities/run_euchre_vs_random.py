"""
Evaluate a saved QMIX checkpoint against random or rule-based opponents.

Usage:
    python run_euchre_vs_random.py                      # vs rule-based (default)
    python run_euchre_vs_random.py --opponent random    # vs random
    python run_euchre_vs_random.py --opponent rule      # vs rule-based
    python run_euchre_vs_random.py --games 500          # number of evaluation hands
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from Personalities.train_qmix_personalities import QMIXSystem, OBS_DIM, ACTION_NUM, MIX_EMBED

# ── CLI args ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--opponent', choices=['random', 'rule'], default='rule',
                    help='Opponent type: random or rule-based (default: rule)')
parser.add_argument('--games', type=int, default=1000,
                    help='Number of hands to evaluate (default: 1000)')
args = parser.parse_args()

# ── Build agents ───────────────────────────────────────────────────────────────

# Scopes must match those used during training so the checkpoint keys align.
agent0 = DQNAgent(
    scope='agent0',
    action_num=ACTION_NUM,
    state_shape=[OBS_DIM],
    mlp_layers=[128, 128],
)
agent2 = DQNAgent(
    scope='agent2',
    action_num=ACTION_NUM,
    state_shape=[OBS_DIM],
    mlp_layers=[128, 128],
)

qmix = QMIXSystem(agent0, agent2)
ckpt_path = os.path.join(os.path.dirname(__file__), 'qmix_euchre.pt')
qmix.load(ckpt_path)

if args.opponent == 'random':
    opp1 = RandomAgent(ACTION_NUM)
    opp3 = RandomAgent(ACTION_NUM)
    opp_label = 'Random'
else:
    opp1 = EuchreRuleAgent()
    opp3 = EuchreRuleAgent()
    opp_label = 'Rule-based'

# ── Evaluation loop ────────────────────────────────────────────────────────────

env = rlcard.make('euchre', config={'num_players': 4})
# Players 0 & 2 are the QMIX team; players 1 & 3 are opponents.
env.set_agents([agent0, opp1, agent2, opp3])

num_games   = args.games
wins        = 0
total_payoff = {i: 0.0 for i in range(4)}

print(f"Evaluating QMIX (players 0 & 2) vs {opp_label} (players 1 & 3) "
      f"over {num_games} hands...")
print("-" * 60)

for game_idx in range(num_games):
    state, player_id = env.reset()

    while not env.is_over():
        if player_id == 0:
            action, _ = qmix.agent0.eval_step(state)
        elif player_id == 2:
            action, _ = qmix.agent2.eval_step(state)
        else:
            action, _ = opp1.eval_step(state) if player_id == 1 else opp3.eval_step(state)
        state, player_id = env.step(action)

    payoffs = env.game.get_payoffs()
    for p, score in payoffs.items():
        total_payoff[p] += score

    qmix_payoff = payoffs.get(0, 0)
    if qmix_payoff > 0:
        wins += 1

    if (game_idx + 1) % 100 == 0:
        win_pct = 100 * wins / (game_idx + 1)
        avg_pay = total_payoff[0] / (game_idx + 1)
        print(f"  Hand {game_idx + 1:>5}: win rate {win_pct:.1f}%  avg payoff {avg_pay:+.3f}")

# ── Summary ────────────────────────────────────────────────────────────────────

print("=" * 60)
print(f"Results over {num_games} hands  (opponent: {opp_label})")
print(f"  QMIX team   (players 0 & 2): "
      f"wins={wins}/{num_games} ({100*wins/num_games:.1f}%)  "
      f"total payoff={total_payoff[0]+total_payoff[2]:+.1f}")
print(f"  Opponent    (players 1 & 3): "
      f"wins={num_games-wins}/{num_games} ({100*(num_games-wins)/num_games:.1f}%)  "
      f"total payoff={total_payoff[1]+total_payoff[3]:+.1f}")
print(f"  Avg payoff per hand (QMIX team player 0): "
      f"{total_payoff[0]/num_games:+.3f}")