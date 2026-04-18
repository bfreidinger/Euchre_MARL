import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import unittest
import numpy as np
import torch

from rlcard.agents.dqn_agent_pytorch import DQNAgent, JointMemory
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.qmix_mixer import QMIXMixer
from BasicQMIX.qmix_trainer import QMIXTrainer, QMIXAgent

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

STATE_DIM  = 127
OBS_DIM    = 48
ACTION_NUM = 54
BATCH      = 8

def make_agents():
    agent0 = DQNAgent(scope='p0', state_shape=[OBS_DIM], action_num=ACTION_NUM,
                      mlp_layers=[64, 64])
    agent2 = DQNAgent(scope='p2', state_shape=[OBS_DIM], action_num=ACTION_NUM,
                      mlp_layers=[64, 64])
    opp_agents = {1: RandomAgent(ACTION_NUM), 3: RandomAgent(ACTION_NUM)}
    return agent0, agent2, opp_agents

def make_joint_memory(size=500, batch=BATCH):
    return JointMemory(memory_size=size, batch_size=batch)

def fill_memory(mem, n=None):
    """Push n random joint transitions into mem (default: mem.batch_size * 2)."""
    n = n or mem.batch_size * 2
    for _ in range(n):
        mem.save(
            obs1=np.random.rand(OBS_DIM).astype(np.float32),
            obs2=np.random.rand(OBS_DIM).astype(np.float32),
            action1=np.random.randint(0, ACTION_NUM),
            action2=np.random.randint(0, ACTION_NUM),
            next_obs1=np.random.rand(OBS_DIM).astype(np.float32),
            next_obs2=np.random.rand(OBS_DIM).astype(np.float32),
            reward=np.random.randn(),
            global_state=np.random.rand(STATE_DIM).astype(np.float32),
            next_global_state=np.random.rand(STATE_DIM).astype(np.float32),
        )


# ---------------------------------------------------------------------------
# QMIXMixer tests
# ---------------------------------------------------------------------------

class TestQMIXMixer(unittest.TestCase):

    def setUp(self):
        self.mixer = QMIXMixer(state_dim=STATE_DIM, n_agents=2,
                               mixing_hidden_dim=16, hyper_hidden_dim=32)

    def test_output_shape(self):
        q = torch.randn(BATCH, 2)
        s = torch.randn(BATCH, STATE_DIM)
        out = self.mixer(q, s)
        self.assertEqual(out.shape, (BATCH, 1),
                         f"Expected (batch, 1), got {out.shape}")

    def test_monotonicity(self):
        """Increasing each Qi should never decrease Q_tot."""
        torch.manual_seed(0)
        s = torch.randn(1, STATE_DIM)
        base_q = torch.tensor([[0.5, 0.5]])
        base_tot = self.mixer(base_q, s).item()

        for i in range(2):
            higher_q = base_q.clone()
            higher_q[0, i] += 2.0
            higher_tot = self.mixer(higher_q, s).item()
            self.assertGreaterEqual(higher_tot, base_tot - 1e-5,
                f"Monotonicity violated for agent {i}: "
                f"Q_tot decreased from {base_tot:.4f} to {higher_tot:.4f}")

    def test_gradients_flow(self):
        """Loss.backward() should produce gradients in all mixer parameters."""
        q = torch.randn(BATCH, 2, requires_grad=True)
        s = torch.randn(BATCH, STATE_DIM)
        out = self.mixer(q, s)
        loss = out.mean()
        loss.backward()
        for name, p in self.mixer.named_parameters():
            self.assertIsNotNone(p.grad, f"No gradient for mixer param: {name}")


# ---------------------------------------------------------------------------
# JointMemory tests
# ---------------------------------------------------------------------------

class TestJointMemory(unittest.TestCase):

    def test_save_and_len(self):
        mem = make_joint_memory(size=100, batch=BATCH)
        self.assertEqual(len(mem), 0)
        fill_memory(mem, n=10)
        self.assertEqual(len(mem), 10)

    def test_overflow_eviction(self):
        """Buffer should never exceed memory_size."""
        mem = make_joint_memory(size=10, batch=4)
        fill_memory(mem, n=20)
        self.assertEqual(len(mem), 10)

    def test_sample_shapes(self):
        mem = make_joint_memory(size=200, batch=BATCH)
        fill_memory(mem, n=50)
        (obs1, obs2, a1, a2,
         next_obs1, next_obs2, reward, gs, next_gs) = mem.sample()

        self.assertEqual(obs1.shape,     (BATCH, OBS_DIM))
        self.assertEqual(obs2.shape,     (BATCH, OBS_DIM))
        self.assertEqual(a1.shape,       (BATCH,))
        self.assertEqual(a2.shape,       (BATCH,))
        self.assertEqual(next_obs1.shape,(BATCH, OBS_DIM))
        self.assertEqual(next_obs2.shape,(BATCH, OBS_DIM))
        self.assertEqual(reward.shape,   (BATCH,))
        self.assertEqual(gs.shape,       (BATCH, STATE_DIM))
        self.assertEqual(next_gs.shape,  (BATCH, STATE_DIM))


# ---------------------------------------------------------------------------
# QMIXTrainer tests
# ---------------------------------------------------------------------------

class TestQMIXTrainer(unittest.TestCase):

    def setUp(self):
        agent0, agent2, opp_agents = make_agents()
        self.mem = make_joint_memory(size=1000, batch=BATCH)
        self.trainer = QMIXTrainer(agent0, agent2, opp_agents, self.mem)

    def test_run_episode_returns_float(self):
        reward = self.trainer.run_episode()
        self.assertIsInstance(reward, (int, float),
                              "run_episode() should return a numeric reward")

    def test_run_episode_populates_memory(self):
        """Each hand has 5 tricks so memory should gain at most 5 entries."""
        before = len(self.mem)
        self.trainer.run_episode()
        after = len(self.mem)
        self.assertGreater(after, before, "Memory should grow after an episode")
        self.assertLessEqual(after - before, 5,
                             "At most 5 trick-level transitions per hand")

    def test_reward_is_valid(self):
        """Euchre payoffs are in {-2, -1, 1, 2}."""
        for _ in range(10):
            r = self.trainer.run_episode()
            self.assertIn(r, {-2.0, -1.0, 1.0, 2.0},
                          f"Unexpected reward value: {r}")


# ---------------------------------------------------------------------------
# QMIXAgent tests
# ---------------------------------------------------------------------------

class TestQMIXAgent(unittest.TestCase):

    def setUp(self):
        agent0, agent2, opp_agents = make_agents()
        self.mem = make_joint_memory(size=2000, batch=BATCH)
        self.qmix = QMIXAgent(
            agent0, agent2, opp_agents, self.mem,
            replay_memory_init_size=BATCH * 2,
            batch_size=BATCH,
            target_update_every=50,
            mixing_hidden_dim=16,
            hyper_hidden_dim=32,
        )

    def test_train_returns_none_when_buffer_small(self):
        """train() should return None until replay_memory_init_size is reached."""
        result = self.qmix.train()
        self.assertIsNone(result,
            "train() should return None when buffer is below init size")

    def test_train_returns_loss_after_warmup(self):
        """train() should return a positive float loss after enough transitions."""
        fill_memory(self.mem, n=BATCH * 3)
        loss = self.qmix.train()
        self.assertIsNotNone(loss, "train() should return a loss value")
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0, "Loss should be positive")

    def test_loss_decreases_over_training(self):
        """Loss should trend downward over many updates on fixed data."""
        fill_memory(self.mem, n=500)
        losses = [self.qmix.train() for _ in range(200)]
        first_10  = np.mean(losses[:10])
        last_10   = np.mean(losses[-10:])
        self.assertLess(last_10, first_10,
            f"Loss did not decrease: first_10={first_10:.4f}, last_10={last_10:.4f}")

    def test_target_networks_sync(self):
        """Target networks should update after target_update_every steps."""
        fill_memory(self.mem, n=500)
        # Grab a param from target mixer before training
        param_before = next(self.qmix.target_mixer.parameters()).clone()
        # Train enough steps to trigger a target sync (target_update_every=50)
        for _ in range(55):
            self.qmix.train()
        param_after = next(self.qmix.target_mixer.parameters()).clone()
        # After sync, target should match current mixer
        current = next(self.qmix.mixer.parameters())
        self.assertTrue(torch.allclose(param_after, current),
            "Target mixer should match current mixer after sync")

    def test_end_to_end_episode_then_train(self):
        """run_episode() + train() together should not raise errors."""
        for _ in range(40):
            self.qmix.run_episode()
        loss = self.qmix.train()
        # May still be None if fewer than init_size transitions collected
        if loss is not None:
            self.assertIsInstance(loss, float)


if __name__ == '__main__':
    unittest.main(verbosity=2)
