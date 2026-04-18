"""
QMIX cooperative training for Euchre — neutral (no personality shaping).

Team 0&2 (QMIX):
  Player 0 and Player 2 are identical cooperative agents.
  No personality shaping — both learn purely from game rewards.

Team 1&3: fixed EuchreRuleAgents (opponents).

Reward structure:
  • Tricks 1-4: +0.25 per trick won by the team (intermediate shaping).
  • Trick 5 / hand end: final game payoff (+1 normal win, +2 march, -1 loss, -2 euchred).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent, JointMemory
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from train_qmix import QMIXSystem, evaluate

# ── Hyperparameters ────────────────────────────────────────────────────────────

OBS_DIM      = 48
ACTION_NUM   = 54
GLOBAL_DIM   = 127
MIX_EMBED    = 64

EPISODES          = 4000
BATCH_SIZE        = 32
MEMORY_SIZE       = 15000
WARMUP_EPISODES   = 300
TRAIN_EVERY       = 1
TARGET_SYNC_EVERY = 300
EVAL_EVERY        = 400
EVAL_GAMES        = 150

GAMMA        = 0.99
LR           = 5e-4
EPSILON_START = 1.0
EPSILON_END   = 0.08
EPSILON_STEPS = EPISODES * 3


# ── Episode Runner (no personality shaping) ───────────────────────────────────

def run_episode(env, qmix: QMIXSystem, opp_agents: dict,
                joint_memory: JointMemory) -> float:
    """
    Play one full hand and store joint transitions.

    Joint transitions are flushed:
      • After each completed trick (reward = score delta × 0.25).
      • At hand end (reward = final game payoff ±1 or ±2).
      • After both agents have acted during the bidding phase (reward = 0).

    Returns the final game payoff for the QMIX team (player 0's payoff).
    """
    game = env.game
    state, player_id = env.reset()

    buf: dict = {}
    prev_gs          = env.get_global_state()
    prev_team_tricks = 0

    def _legal_mask(state_dict):
        """Build a binary mask of shape (ACTION_NUM,) from a state's legal actions."""
        mask = np.zeros(ACTION_NUM, dtype=np.float32)
        for a in state_dict['legal_actions']:
            mask[a] = 1.0
        return mask

    def flush(reward: float, done: bool = False):
        nonlocal prev_gs
        if 0 not in buf or 2 not in buf:
            return

        next_gs = env.get_global_state()
        obs0, a0 = buf[0]
        obs2, a2 = buf[2]

        next_state0 = env._extract_state(game.get_state(0))
        next_state2 = env._extract_state(game.get_state(2))
        next_obs0 = next_state0['obs']
        next_obs2 = next_state2['obs']
        next_legal0 = _legal_mask(next_state0) if not done else np.ones(ACTION_NUM, dtype=np.float32)
        next_legal2 = _legal_mask(next_state2) if not done else np.ones(ACTION_NUM, dtype=np.float32)

        joint_memory.save(obs0, obs2, a0, a2,
                          next_obs0, next_obs2,
                          reward, prev_gs, next_gs,
                          next_legal0, next_legal2, done)
        buf.clear()
        prev_gs = next_gs

    while not env.is_over():

        # ── action selection ──────────────────────────────────────────────────
        if player_id == 0:
            action = qmix.agent0.step(state)
            qmix.agent0.total_t += 1
        elif player_id == 2:
            action = qmix.agent2.step(state)
            qmix.agent2.total_t += 1
        else:
            action = opp_agents[player_id].step(state)

        # ── buffer team agent transitions (no shaping) ────────────────────────
        if player_id in (0, 2):
            buf[player_id] = (state['obs'].copy(), action)

        prev_center_len = len(game.center)
        next_state, next_player_id = env.step(action)
        curr_center_len = len(game.center)

        trick_ended = (prev_center_len == 3 and curr_center_len == 0)
        hand_ended  = env.is_over()

        # ── flush joint transition ────────────────────────────────────────────
        if hand_ended:
            payoffs = game.get_payoffs()
            flush(payoffs.get(0, 0), done=True)
        elif trick_ended:
            new_tricks = game.score[0] + game.score[2]
            trick_r    = (new_tricks - prev_team_tricks) * 0.25
            prev_team_tricks = new_tricks
            flush(trick_r)
        elif 0 in buf and 2 in buf:
            flush(0.0)

        state     = next_state
        player_id = next_player_id

    return game.get_payoffs().get(0, 0)


# ── Main Training Loop ─────────────────────────────────────────────────────────

if __name__ == '__main__':

    env = rlcard.make('euchre', config={'num_players': 4})

    # ── create agents (identical architecture, no personality distinction) ─────
    agent0 = DQNAgent(
        scope='qmix_p0',
        action_num=ACTION_NUM,
        state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_STEPS,
        replay_memory_size=MEMORY_SIZE,
        replay_memory_init_size=100,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    agent2 = DQNAgent(
        scope='qmix_p2',
        action_num=ACTION_NUM,
        state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_STEPS,
        replay_memory_size=MEMORY_SIZE,
        replay_memory_init_size=100,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
    )
    opp_agents = {1: EuchreRuleAgent(), 3: EuchreRuleAgent()}

    # ── QMIX system + replay buffer ───────────────────────────────────────────
    joint_memory = JointMemory(memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)
    qmix         = QMIXSystem(agent0, agent2)

    env.set_agents([agent0, opp_agents[1], agent2, opp_agents[3]])

    # ── training ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("QMIX Cooperative Euchre Training (Neutral — no shaping)")
    print("  Team QMIX : Player 0 + Player 2  (identical)")
    print("  Team Rule : Player 1 (Rule) + Player 3 (Rule)")
    print(f"  Episodes  : {EPISODES}   Warmup: {WARMUP_EPISODES}")
    print("=" * 60)
    print(f"{'Episode':>8}  {'AvgLoss':>9}  {'WinRate':>8}  {'AvgPayoff':>10}")
    print("-" * 44)

    recent_losses = []

    for ep in range(1, EPISODES + 1):
        run_episode(env, qmix, opp_agents, joint_memory)

        if ep >= WARMUP_EPISODES and len(joint_memory) >= BATCH_SIZE:
            loss = qmix.train(joint_memory)
            recent_losses.append(loss)

        if ep % TARGET_SYNC_EVERY == 0:
            qmix.sync_targets()
            print(f"  [ep {ep:>4}] target networks synced")

        if ep % EVAL_EVERY == 0:
            win_rate, avg_payoff = evaluate(env, qmix, opp_agents)
            avg_loss = np.mean(recent_losses[-EVAL_EVERY:]) if recent_losses else float('nan')
            eps0 = qmix.agent0.epsilons[min(qmix.agent0.total_t, EPSILON_STEPS - 1)]
            print(f"{ep:>8}  {avg_loss:>9.4f}  {win_rate*100:>7.1f}%  {avg_payoff:>+10.3f}"
                  f"   ε={eps0:.3f}")

    # ── final evaluation ──────────────────────────────────────────────────────
    print("=" * 60)
    win_rate, avg_payoff = evaluate(env, qmix, opp_agents, n_games=1000)
    print(f"Final (1000 games):  win={win_rate*100:.1f}%  avg_payoff={avg_payoff:+.3f}")

    qmix.save(os.path.join(os.path.dirname(__file__), 'qmix_neutral.pt'))
