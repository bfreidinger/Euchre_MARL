"""
QMIX training loop for cooperative Euchre (players 0 & 2 as one team).

Algorithm: Rashid et al., "QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent Reinforcement Learning" (ICML 2018).

Implementation structure adapted from Lizhi-sjtu/MARL-code-pytorch (QMIX_SMAC),
simplified to use feedforward Q-networks with transition-level replay, since
Euchre episodes are short (5 tricks) and the state is effectively Markovian
given the trick history encoded in the observation.

Per-agent Q-networks are kept separate rather than shared because the two
team players (seats 0 and 2) have distinct positional information (lead vs.
third-to-act) that a shared-parameter network with agent-ID input would
have to relearn from scratch.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from rlcard.envs.euchre import EuchreEnv
from rlcard.agents.dqn_agent_pytorch import DQNAgent, JointMemory
from rlcard.agents.qmix_mixer import QMIXMixer

ENV_CONFIG = {
    'allow_step_back': False,
    'allow_raw_data': False,
    'record_action': False,
    'single_agent_mode': False,
    'active_player': 0,
    'seed': None,
}

TEAM_PLAYERS = {0, 2}   # agents controlled by QMIX
OPP_PLAYERS  = {1, 3}   # opponents (rule/random agents)
ACTION_NUM   = 54


class QMIXTrainer:
    def __init__(self,
                 agent0,           # DQNAgent for player 0
                 agent2,           # DQNAgent for player 2
                 opp_agents,       # dict {1: agent, 3: agent} for opponents
                 joint_memory,     # JointMemory instance
                 env=None):
        self.agent0 = agent0
        self.agent2 = agent2
        self.opp_agents = opp_agents
        self.joint_memory = joint_memory
        self.env = env or EuchreEnv(ENV_CONFIG)

    def run_episode(self):
        """Play one full hand and populate joint_memory with trick-level transitions.

        Returns the shared reward received by team {0, 2}.
        """
        state, player_id = self.env.game.init_game()
        state = self.env._extract_state(state)

        trick_global_state = None
        pending = {}
        cards_at_trick_start = None

        def _legal_mask(state_dict):
            mask = np.zeros(ACTION_NUM, dtype=np.float32)
            for a in state_dict['legal_actions']:
                mask[a] = 1.0
            return mask

        while not self.env.game.is_over():
            cur_hand_size = len(self.env.game.players[0].hand)
            if cards_at_trick_start != cur_hand_size and len(self.env.game.center) == 0:
                trick_global_state = self.env.get_global_state()
                cards_at_trick_start = cur_hand_size
                pending = {}

            obs = state['obs']

            if player_id == 0:
                action = self.agent0.step(state)
            elif player_id == 2:
                action = self.agent2.step(state)
            else:
                action = self.opp_agents[player_id].step(state)

            if player_id in TEAM_PLAYERS:
                pending[player_id] = (obs, action)

            next_state, next_player_id = self.env.step(action)

            if TEAM_PLAYERS.issubset(pending.keys()):
                hand_done = self.env.game.is_over()
                next_global_state = self.env.get_global_state()
                payoffs = self.env.game.get_payoffs() if hand_done else {}
                reward = payoffs.get(0, 0.0)

                obs1, a1 = pending[0]
                obs2, a2 = pending[2]
                next_obs1 = next_state['obs'] if next_player_id == 0 else \
                            self.env._extract_state(self.env.game.get_state(0))['obs']
                next_obs2 = next_state['obs'] if next_player_id == 2 else \
                            self.env._extract_state(self.env.game.get_state(2))['obs']

                next_state0 = self.env._extract_state(self.env.game.get_state(0))
                next_state2 = self.env._extract_state(self.env.game.get_state(2))
                next_legal0 = np.ones(ACTION_NUM, dtype=np.float32) if hand_done else _legal_mask(next_state0)
                next_legal2 = np.ones(ACTION_NUM, dtype=np.float32) if hand_done else _legal_mask(next_state2)

                self.joint_memory.save(
                    obs1, obs2, a1, a2,
                    next_obs1, next_obs2,
                    reward,
                    trick_global_state,
                    next_global_state,
                    next_legal0, next_legal2,
                    hand_done,
                )
                pending = {}

            state = next_state
            player_id = next_player_id

        payoffs = self.env.game.get_payoffs()
        return payoffs.get(0, 0.0)


class QMIXAgent:
    """Ties together two per-agent Q-networks and the QMIX mixer.

    Owns:
      - agent0, agent2         : DQNAgents whose q_estimator.qnet is trained
      - eval_mix_net           : QMIXMixer (online)
      - target_agent0/2        : frozen Q-net copies updated every target_update_freq steps
      - target_mix_net         : frozen mixer copy
      - eval_parameters        : flat list of all trainable params for optimizer + grad clip
      - A single Adam optimizer across all three networks
    """

    def __init__(self,
                 agent0,
                 agent2,
                 opp_agents,
                 joint_memory,
                 env=None,
                 gamma=0.99,
                 learning_rate=1e-4,
                 target_update_freq=200,
                 batch_size=32,
                 replay_memory_init_size=200,
                 state_dim=127,
                 mixing_hidden_dim=32,
                 hyper_hidden_dim=64,
                 device=None):

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.replay_memory_init_size = replay_memory_init_size
        self.train_step = 0

        self.agent0 = agent0
        self.agent2 = agent2

        self.eval_mix_net   = QMIXMixer(state_dim, n_agents=2,
                                        mixing_hidden_dim=mixing_hidden_dim,
                                        hyper_hidden_dim=hyper_hidden_dim).to(self.device)
        self.target_mix_net = deepcopy(self.eval_mix_net)

        # Frozen target Q-nets — initialized as copies, synced via load_state_dict
        self.target_agent0 = deepcopy(agent0)
        self.target_agent2 = deepcopy(agent2)

        self.eval_parameters = (
            list(agent0.q_estimator.qnet.parameters()) +
            list(agent2.q_estimator.qnet.parameters()) +
            list(self.eval_mix_net.parameters())
        )
        self.optimizer = torch.optim.Adam(self.eval_parameters, lr=learning_rate)

        self._runner = QMIXTrainer(agent0, agent2, opp_agents, joint_memory, env)
        self.joint_memory = joint_memory

    def run_episode(self):
        """Collect one hand into joint_memory; return team reward."""
        return self._runner.run_episode()

    def _sync_targets(self):
        self.target_agent0.q_estimator.qnet.load_state_dict(
            self.agent0.q_estimator.qnet.state_dict())
        self.target_agent2.q_estimator.qnet.load_state_dict(
            self.agent2.q_estimator.qnet.state_dict())
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

    def train(self):
        """One QMIX gradient update. Returns loss or None if buffer too small."""
        if len(self.joint_memory) < self.replay_memory_init_size:
            return None

        (obs1, obs2, a1, a2,
         next_obs1, next_obs2,
         reward, gs, next_gs,
         nl1, nl2, done) = self.joint_memory.sample()

        obs1_t      = torch.FloatTensor(obs1).to(self.device)
        obs2_t      = torch.FloatTensor(obs2).to(self.device)
        a1_t        = torch.LongTensor(a1).to(self.device)
        a2_t        = torch.LongTensor(a2).to(self.device)
        next_obs1_t = torch.FloatTensor(next_obs1).to(self.device)
        next_obs2_t = torch.FloatTensor(next_obs2).to(self.device)
        reward_t    = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        gs_t        = torch.FloatTensor(gs).to(self.device)
        next_gs_t   = torch.FloatTensor(next_gs).to(self.device)
        nl1_t       = torch.FloatTensor(nl1).to(self.device)
        nl2_t       = torch.FloatTensor(nl2).to(self.device)
        done_t      = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # ── current Q_tot ─────────────────────────────────────────────────────
        q1 = self.agent0.q_estimator.qnet(obs1_t)
        q2 = self.agent2.q_estimator.qnet(obs2_t)
        q1_taken = q1.gather(1, a1_t.unsqueeze(1))
        q2_taken = q2.gather(1, a2_t.unsqueeze(1))
        chosen_action_qvals = torch.cat([q1_taken, q2_taken], dim=1)   # (B, 2)
        q_total_eval = self.eval_mix_net(chosen_action_qvals, gs_t)    # (B, 1)

        # ── target Q_tot (Double-DQN style) ───────────────────────────────────
        with torch.no_grad():
            # Online net selects best legal next action; target net evaluates it
            q1_online = self.agent0.q_estimator.qnet(next_obs1_t)
            q2_online = self.agent2.q_estimator.qnet(next_obs2_t)
            q1_online[nl1_t == 0] = -1e9
            q2_online[nl2_t == 0] = -1e9
            best_a1 = q1_online.argmax(dim=1, keepdim=True)
            best_a2 = q2_online.argmax(dim=1, keepdim=True)

            q1_next = self.target_agent0.q_estimator.qnet(next_obs1_t)
            q2_next = self.target_agent2.q_estimator.qnet(next_obs2_t)
            q1_target_max = q1_next.gather(1, best_a1)
            q2_target_max = q2_next.gather(1, best_a2)

            q_vals_next    = torch.cat([q1_target_max, q2_target_max], dim=1)
            q_total_target = self.target_mix_net(q_vals_next, next_gs_t)
            targets = reward_t + self.gamma * (1.0 - done_t) * q_total_target

        # ── gradient step ─────────────────────────────────────────────────────
        loss = F.mse_loss(q_total_eval, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, max_norm=10.0)
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._sync_targets()

        return loss.item()
