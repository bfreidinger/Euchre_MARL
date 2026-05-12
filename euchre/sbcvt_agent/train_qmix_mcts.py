"""
MCTS-augmented QMIX training (AlphaZero-style).

Every MCTS_EVERY episodes, epsilon-greedy action selection is replaced with
MCTS using the current Q-networks as prior and leaf evaluator. All TD updates
are identical to standard QMIX training. The only thing that changes is the
quality of actions stored in the replay buffer.

This creates the AlphaZero feedback loop:
  better Q-values → better MCTS → better replay data → better Q-values

Usage:
  python train_qmix_mcts.py                          # train from scratch
  python train_qmix_mcts.py --resume qmix_euchre.pt  # fine-tune existing ckpt
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent, JointMemory
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent

from BasicQMIX.train_qmix import (
    QMIXSystem,
    OBS_DIM, ACTION_NUM, GLOBAL_DIM, MIX_EMBED,
    EPISODES, BATCH_SIZE, MEMORY_SIZE, WARMUP_EPISODES,
    TARGET_SYNC_EVERY, EVAL_EVERY, EVAL_GAMES,
    LR, EPSILON_START, EPSILON_END, EPSILON_STEPS,
    run_episode, evaluate,
)
from sbcvt_agent import SBCVTAgent

# ── MCTS training hyperparameters ─────────────────────────────────────────────
# Keep MCTS cheap during training so the loop stays tractable.
# 20 sims × ~20 decisions/hand ≈ 400 forward passes per episode.

MCTS_EVERY = 10    # use MCTS every Nth episode  (10 % of all episodes)
MCTS_SIMS  = 20    # simulations per decision during training
MCTS_DEPTH = 5  22    # max rollout depth before QMIX leaf eval

CKPT_OUT = os.path.join(os.path.dirname(__file__), '..', 'BasicQMIX', 'qmix_mcts.pt')


# ── MCTS episode runner ────────────────────────────────────────────────────────

def run_episode_mcts(env, qmix: QMIXSystem, sbcvt: SBCVTAgent,
                     opp_agents: dict, joint_memory: JointMemory) -> float:
    """
    Play one hand using MCTS action selection for the QMIX team.

    Structurally identical to run_episode() — same flush logic, same reward
    shaping, same JointMemory.save() signature. The only change is that
    agent.step() (epsilon-greedy) is replaced by sbcvt.step() (MCTS).

    Because MCTS produces better actions, the transitions pushed into the
    replay buffer are higher quality, which improves subsequent TD updates.
    """
    game = env.game
    state, player_id = env.reset()
    sbcvt.new_hand()

    buf: dict = {}
    prev_gs          = env.get_global_state()
    prev_team_tricks = 0

    def _legal_mask(state_dict):
        mask = np.zeros(ACTION_NUM, dtype=np.float32)
        for a in state_dict['legal_actions']:
            mask[a] = 1.0
        return mask

    def flush(trick_reward: float, done: bool = False):
        nonlocal prev_gs
        if 0 not in buf or 2 not in buf:
            return
        next_gs = env.get_global_state()
        obs0, a0 = buf[0]
        obs2, a2 = buf[2]
        next_state0 = env._extract_state(game.get_state(0))
        next_state2 = env._extract_state(game.get_state(2))
        next_obs0   = next_state0['obs']
        next_obs2   = next_state2['obs']
        next_legal0 = _legal_mask(next_state0) if not done else np.ones(ACTION_NUM, dtype=np.float32)
        next_legal2 = _legal_mask(next_state2) if not done else np.ones(ACTION_NUM, dtype=np.float32)
        joint_memory.save(obs0, obs2, a0, a2,
                          next_obs0, next_obs2,
                          trick_reward, prev_gs, next_gs,
                          next_legal0, next_legal2, done)
        buf.clear()
        prev_gs = next_gs

    while not env.is_over():

        if player_id == 0:
            action = sbcvt.step(state, player_id=0)
            qmix.agent0.total_t += 1   # keep epsilon schedule on track
        elif player_id == 2:
            action = sbcvt.step(state, player_id=2)
            qmix.agent2.total_t += 1
        else:
            action = opp_agents[player_id].step(state)

        if player_id in (0, 2):
            buf[player_id] = (state['obs'].copy(), action)
            sbcvt.record_action(player_id, action, game)

        prev_center_len = len(game.center)
        next_state, next_player_id = env.step(action)
        curr_center_len = len(game.center)

        trick_ended = (prev_center_len == 3 and curr_center_len == 0)
        hand_ended  = env.is_over()

        if hand_ended:
            payoffs = game.get_payoffs()
            flush(payoffs.get(0, 0), done=True)
        elif trick_ended:
            new_tricks       = game.score[0] + game.score[2]
            trick_r          = (new_tricks - prev_team_tricks) * 0.25
            prev_team_tricks = new_tricks
            flush(trick_r)
        elif 0 in buf and 2 in buf and len(game.center) == 0:
            flush(0.0)

        state     = next_state
        player_id = next_player_id

    return game.get_payoffs().get(0, 0)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None,
                        help='path to existing QMIX checkpoint to fine-tune')
    args = parser.parse_args()

    env = rlcard.make('euchre', config={'num_players': 4})

    # ── agents ────────────────────────────────────────────────────────────────
    agent0 = DQNAgent(
        scope='agent0', action_num=ACTION_NUM, state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_STEPS,
        replay_memory_size=MEMORY_SIZE, replay_memory_init_size=100,
        batch_size=BATCH_SIZE, learning_rate=LR,
    )
    agent2 = DQNAgent(
        scope='agent2', action_num=ACTION_NUM, state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_STEPS,
        replay_memory_size=MEMORY_SIZE, replay_memory_init_size=100,
        batch_size=BATCH_SIZE, learning_rate=LR,
    )
    opp_agents   = {1: EuchreRuleAgent(), 3: EuchreRuleAgent()}
    joint_memory = JointMemory(memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)
    qmix         = QMIXSystem(agent0, agent2)

    if args.resume:
        qmix.load(args.resume)
        print(f"Resumed from: {args.resume}")

    env.set_agents([agent0, opp_agents[1], agent2, opp_agents[3]])

    # SBCVT agent always wraps the live Q-networks — no separate copy needed.
    # When Q-networks improve, MCTS automatically benefits on the next call.
    sbcvt = SBCVTAgent(
        agent0=agent0,
        agent2=agent2,
        mixer=qmix.mixer,
        env=env,
        num_sims=MCTS_SIMS,
        max_depth=MCTS_DEPTH,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    # ── training loop ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("MCTS-Augmented QMIX Training")
    print(f"  Team QMIX  : Player 0 + Player 2")
    print(f"  Team Rule  : Player 1 + Player 3  (fixed rule agents)")
    print(f"  Episodes   : {EPISODES}   Warmup: {WARMUP_EPISODES}")
    print(f"  MCTS every : {MCTS_EVERY} episodes  "
          f"({100/MCTS_EVERY:.0f}% of training)  "
          f"sims={MCTS_SIMS}  depth={MCTS_DEPTH}")
    print("=" * 65)
    print(f"{'Episode':>8}  {'Type':>6}  {'AvgLoss':>9}  {'WinRate':>8}  {'AvgPayoff':>10}")
    print("-" * 50)

    recent_losses  = []
    eval_episodes  = []
    eval_win_rates = []
    eval_payoffs   = []
    mcts_episodes  = 0
    eps_episodes   = 0

    for ep in range(1, EPISODES + 1):

        warmed_up = ep >= WARMUP_EPISODES
        use_mcts  = warmed_up and (ep % MCTS_EVERY == 0)

        if use_mcts:
            run_episode_mcts(env, qmix, sbcvt, opp_agents, joint_memory)
            mcts_episodes += 1
        else:
            run_episode(env, qmix, opp_agents, joint_memory)
            eps_episodes += 1

        if warmed_up and len(joint_memory) >= BATCH_SIZE:
            loss = qmix.train(joint_memory)
            if loss is not None:
                recent_losses.append(loss)

        if ep % TARGET_SYNC_EVERY == 0:
            qmix.sync_targets()

        if ep % EVAL_EVERY == 0:
            win_rate, avg_payoff = evaluate(env, qmix, opp_agents)
            avg_loss = np.mean(recent_losses[-EVAL_EVERY:]) if recent_losses else float('nan')
            eps_val  = qmix.agent0.epsilons[min(qmix.agent0.total_t, EPSILON_STEPS - 1)]
            ep_type  = 'MCTS' if use_mcts else 'eps'
            print(f"{ep:>8}  {ep_type:>6}  {avg_loss:>9.4f}  "
                  f"{win_rate*100:>7.1f}%  {avg_payoff:>+10.3f}  ε={eps_val:.3f}")
            eval_episodes.append(ep)
            eval_win_rates.append(win_rate * 100)
            eval_payoffs.append(avg_payoff)

    # ── final eval + save ─────────────────────────────────────────────────────
    print("=" * 65)
    win_rate, avg_payoff = evaluate(env, qmix, opp_agents, n_games=2000)
    print(f"Final (2000 hands):  win={win_rate*100:.1f}%  avg_payoff={avg_payoff:+.3f}")
    print(f"MCTS episodes: {mcts_episodes}  eps-greedy episodes: {eps_episodes}")
    qmix.save(CKPT_OUT)

    # ── learning curves ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(eval_episodes, eval_win_rates, marker='o', linewidth=2, color='steelblue')
    ax1.axhline(50, color='grey', linestyle='--', linewidth=1)
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('MCTS-Augmented QMIX — Win Rate vs Rule Agents')
    ax1.grid(alpha=0.3)

    ax2.plot(eval_episodes, eval_payoffs, marker='o', linewidth=2, color='darkorange')
    ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax2.set_ylabel('Avg Payoff')
    ax2.set_xlabel('Episode')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'BasicQMIX', 'qmix_mcts_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Learning curves saved to: {plot_path}")
    plt.show()
