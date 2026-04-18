"""
QMIX training loop for cooperative Euchre (players 0 & 2 as one team).

Sequential-turn handling follows Option 1 from the Phase 2 spec:
  - global state s_t is snapshotted once at the start of each trick
  - both agents' transitions within that trick share the same s_t / s_t'
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

        # Per-trick staging buffers
        trick_global_state = None   # snapshotted once per trick
        pending = {}                # {player_id: (obs, action)} within current trick

        cards_at_trick_start = None

        while not self.env.game.is_over():
            # --- snapshot global state at the start of each trick ---
            cur_hand_size = len(self.env.game.players[0].hand)
            if cards_at_trick_start != cur_hand_size and len(self.env.game.center) == 0:
                trick_global_state = self.env.get_global_state()
                cards_at_trick_start = cur_hand_size
                pending = {}

            obs = state['obs']

            # --- select action ---
            if player_id == 0:
                action = self.agent0.step(state)
            elif player_id == 2:
                action = self.agent2.step(state)
            else:
                action = self.opp_agents[player_id].step(state)

            # store pending transition for team players
            if player_id in TEAM_PLAYERS:
                pending[player_id] = (obs, action)

            # --- step environment ---
            next_state, next_player_id = self.env.step(action)

            # --- when both team players have acted in this trick, save joint transition ---
            if TEAM_PLAYERS.issubset(pending.keys()):
                next_global_state = self.env.get_global_state()
                payoffs = self.env.game.get_payoffs() if self.env.game.is_over() else {}
                reward = payoffs.get(0, 0.0)   # team reward (same for 0 and 2)

                obs1, a1 = pending[0]
                obs2, a2 = pending[2]
                next_obs1 = next_state['obs'] if next_player_id == 0 else \
                            self.env._extract_state(self.env.game.get_state(0))['obs']
                next_obs2 = next_state['obs'] if next_player_id == 2 else \
                            self.env._extract_state(self.env.game.get_state(2))['obs']

                self.joint_memory.save(
                    obs1, obs2, a1, a2,
                    next_obs1, next_obs2,
                    reward,
                    trick_global_state,
                    next_global_state,
                )
                pending = {}

            state = next_state
            player_id = next_player_id

        payoffs = self.env.game.get_payoffs()
        return payoffs.get(0, 0.0)


class QMIXAgent:
    """Ties together two per-agent Q-networks and the QMIX mixer.

    Owns:
      - agent0, agent2       : DQNAgents whose q_estimator.qnet is trained
      - mixer                : QMIXMixer
      - target_agent0/2      : frozen copies updated every target_update_every steps
      - target_mixer         : frozen copy of mixer
      - A single Adam optimizer across all three networks
    """

    def __init__(self,
                 agent0,
                 agent2,
                 opp_agents,
                 joint_memory,
                 env=None,
                 discount_factor=0.99,
                 learning_rate=1e-4,
                 target_update_every=200,
                 batch_size=32,
                 replay_memory_init_size=200,
                 state_dim=127,
                 mixing_hidden_dim=32,
                 hyper_hidden_dim=64,
                 device=None):

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.discount_factor = discount_factor
        self.target_update_every = target_update_every
        self.batch_size = batch_size
        self.replay_memory_init_size = replay_memory_init_size
        self.train_t = 0

        # Per-agent Q-networks (reuse DQNAgent's estimator)
        self.agent0 = agent0
        self.agent2 = agent2

        # Mixer
        self.mixer = QMIXMixer(state_dim, n_agents=2,
                               mixing_hidden_dim=mixing_hidden_dim,
                               hyper_hidden_dim=hyper_hidden_dim).to(self.device)

        # Target networks (frozen copies, periodically synced)
        self.target_agent0 = deepcopy(agent0)
        self.target_agent2 = deepcopy(agent2)
        self.target_mixer  = deepcopy(self.mixer)

        # Single optimizer over all trainable parameters
        params = (list(agent0.q_estimator.qnet.parameters()) +
                  list(agent2.q_estimator.qnet.parameters()) +
                  list(self.mixer.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        # Episode runner
        self._runner = QMIXTrainer(agent0, agent2, opp_agents, joint_memory, env)
        self.joint_memory = joint_memory

    def run_episode(self):
        """Collect one hand into joint_memory; return team reward."""
        return self._runner.run_episode()

    def train(self):
        """One QMIX gradient update. Returns loss or None if buffer too small."""
        if len(self.joint_memory) < self.replay_memory_init_size:
            return None

        (obs1, obs2, a1, a2,
         next_obs1, next_obs2,
         reward, gs, next_gs) = self.joint_memory.sample()

        # --- tensors ---
        obs1_t      = torch.FloatTensor(obs1).to(self.device)
        obs2_t      = torch.FloatTensor(obs2).to(self.device)
        a1_t        = torch.LongTensor(a1).to(self.device)
        a2_t        = torch.LongTensor(a2).to(self.device)
        next_obs1_t = torch.FloatTensor(next_obs1).to(self.device)
        next_obs2_t = torch.FloatTensor(next_obs2).to(self.device)
        reward_t    = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        gs_t        = torch.FloatTensor(gs).to(self.device)
        next_gs_t   = torch.FloatTensor(next_gs).to(self.device)

        # --- current Q_tot ---
        q1 = self.agent0.q_estimator.qnet(obs1_t)           # (batch, action_num)
        q2 = self.agent2.q_estimator.qnet(obs2_t)
        q1_taken = q1.gather(1, a1_t.unsqueeze(1))          # (batch, 1)
        q2_taken = q2.gather(1, a2_t.unsqueeze(1))
        q_agents = torch.cat([q1_taken, q2_taken], dim=1)   # (batch, 2)
        q_tot = self.mixer(q_agents, gs_t)                  # (batch, 1)

        # --- target Q_tot (no grad) ---
        with torch.no_grad():
            q1_next = self.target_agent0.q_estimator.qnet(next_obs1_t)
            q2_next = self.target_agent2.q_estimator.qnet(next_obs2_t)
            q1_next_max = q1_next.max(1)[0].unsqueeze(1)    # (batch, 1)
            q2_next_max = q2_next.max(1)[0].unsqueeze(1)
            q_agents_next = torch.cat([q1_next_max, q2_next_max], dim=1)
            q_tot_next = self.target_mixer(q_agents_next, next_gs_t)
            target = reward_t + self.discount_factor * q_tot_next

        # --- loss and update ---
        loss = F.mse_loss(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- sync target networks ---
        self.train_t += 1
        if self.train_t % self.target_update_every == 0:
            self.target_agent0 = deepcopy(self.agent0)
            self.target_agent2 = deepcopy(self.agent2)
            self.target_mixer  = deepcopy(self.mixer)

        return loss.item()
