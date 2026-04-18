"""
QMIX Mixing Network for cooperative Euchre.

Takes individual Q-values Q1, Q2 from each teammate agent and the
127-dim global state, and produces a monotone Q_tot via hypernetworks.
Monotonicity is enforced by taking abs() of all mixing weights so that
dQ_tot/dQi >= 0 for all i.

Architecture follows Rashid et al. (2018), QMIX.
"""

import numpy as np
import torch
import torch.nn as nn


class QMIXMixer(nn.Module):
    """
    Args:
        state_dim (int): dimension of global state vector (127)
        n_agents (int): number of cooperative agents (2)
        mixing_hidden_dim (int): width of the mixing layer (32)
        hyper_hidden_dim (int): width of the hypernetwork hidden layer (64)
    """

    def __init__(self,
                 state_dim=127,
                 n_agents=2,
                 mixing_hidden_dim=32,
                 hyper_hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.mixing_hidden_dim = mixing_hidden_dim

        # Hypernetwork for first mixing layer weights W1: s -> (n_agents * mixing_hidden_dim,)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, n_agents * mixing_hidden_dim),
        )

        # Hypernetwork for first mixing layer bias b1: s -> (mixing_hidden_dim,)
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)

        # Hypernetwork for second mixing layer weights W2: s -> (mixing_hidden_dim,)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, mixing_hidden_dim),
        )

        # Hypernetwork for final bias b2: s -> scalar
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

    def forward(self, q_values, global_states):
        """
        Args:
            q_values (Tensor): shape (batch, n_agents) — Q1 and Q2 for chosen actions
            global_states (Tensor): shape (batch, state_dim)

        Returns:
            q_tot (Tensor): shape (batch, 1) — monotone joint Q-value
        """
        batch = q_values.size(0)
        q_values = q_values.view(batch, 1, self.n_agents)   # (batch, 1, n_agents)

        # First mixing layer
        w1 = torch.abs(self.hyper_w1(global_states))        # (batch, n_agents * mixing_hidden_dim)
        w1 = w1.view(batch, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(global_states).view(batch, 1, self.mixing_hidden_dim)
        hidden = torch.nn.functional.elu(
            torch.bmm(q_values, w1) + b1                    # (batch, 1, mixing_hidden_dim)
        )

        # Second mixing layer
        w2 = torch.abs(self.hyper_w2(global_states))        # (batch, mixing_hidden_dim)
        w2 = w2.view(batch, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(global_states).view(batch, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2                  # (batch, 1, 1)

        return q_tot.view(batch, 1)
