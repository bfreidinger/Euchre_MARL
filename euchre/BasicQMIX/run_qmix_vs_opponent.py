"""
Evaluate a saved QMIX checkpoint against random or rule-based opponents.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from train_qmix import QMIXSystem, OBS_DIM, ACTION_NUM, MIX_EMBED

# ── Configuration ──────────────────────────────────────────────────────────────

OPPONENT   = 'rule'   # 'rule' or 'random'
NUM_GAMES  = 20
SHOW_HANDS = True     # print score after every hand

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

if OPPONENT == 'random':
    opp1 = RandomAgent(ACTION_NUM)
    opp3 = RandomAgent(ACTION_NUM)
    opp_label = 'Random'
else:
    opp1 = EuchreRuleAgent()
    opp3 = EuchreRuleAgent()
    opp_label = 'Rule-based'

# ── Evaluation loop ────────────────────────────────────────────────────────────
# Each "game" is a race to WIN_TARGET points.
# Winning a hand scores +1 (normal win) or +2 (march); losses score 0.
# First team to WIN_TARGET points wins the game.

WIN_TARGET = 10

env = rlcard.make('euchre', config={'num_players': 4})
env.set_agents([agent0, opp1, agent2, opp3])

num_games      = NUM_GAMES
qmix_match_wins = 0
total_hands    = 0

print(f"Evaluating QMIX (players 0 & 2) vs {opp_label} (players 1 & 3)")
print(f"  {num_games} games, first to {WIN_TARGET} points wins each game")
print("-" * 60)

for game_idx in range(num_games):
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

        payoffs = env.game.get_payoffs()
        qmix_hand = payoffs.get(0, 0)

        if qmix_hand > 0:
            qmix_score += qmix_hand
            pts     = qmix_hand
            winner  = 'QMIX' + (' (march!)' if qmix_hand == 2 else '')
            score_str = f"QMIX {qmix_score:>2} — {opp_score:<2} OPP"
        else:
            opp_score += abs(qmix_hand)
            pts     = abs(qmix_hand)
            winner  = 'OPP ' + (' (march!)' if abs(qmix_hand) == 2 else '')
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
    total_hands += hands_this_game

# ── Summary ────────────────────────────────────────────────────────────────────

print("=" * 60)
print(f"Results over {num_games} games to {WIN_TARGET} pts  (opponent: {opp_label})")
print(f"  QMIX team   (players 0 & 2): "
      f"{qmix_match_wins}/{num_games} ({100*qmix_match_wins/num_games:.1f}%)")
print(f"  Opponent    (players 1 & 3): "
      f"{num_games - qmix_match_wins}/{num_games} "
      f"({100*(num_games - qmix_match_wins)/num_games:.1f}%)")
print(f"  Avg hands per game: {total_hands/num_games:.1f}")
