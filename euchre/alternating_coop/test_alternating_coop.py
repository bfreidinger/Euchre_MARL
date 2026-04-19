import unittest

from euchre.alternating_coop import (
    AUGMENTED_OBS_DIM,
    AlternatingCooperativeTrainer,
    AlternatingTrainingConfig,
)


def make_test_config():
    return AlternatingTrainingConfig(
        rule_init_hands=2,
        phase_episodes=2,
        max_rounds=1,
        eval_games=4,
        convergence_tol=0.0,
        patience_rounds=1,
        batch_size=2,
        replay_memory_size=200,
        replay_memory_init_size=2,
        update_target_every=10,
        epsilon_decay_steps=100,
        mlp_layers=(32,),
        seed=7,
    )


class TestAlternatingCoopTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = AlternatingCooperativeTrainer(make_test_config())

    def test_run_phase_trains_only_active_player(self):
        self.assertEqual(len(self.trainer.agents[0].memory.memory), 0)
        self.assertEqual(len(self.trainer.agents[2].memory.memory), 0)

        phase0 = self.trainer.run_phase(round_idx=1, train_player=0, num_episodes=2)
        p0_after_first = len(self.trainer.agents[0].memory.memory)
        p2_after_first = len(self.trainer.agents[2].memory.memory)

        self.assertGreater(p0_after_first, 0)
        self.assertEqual(p2_after_first, 0)
        self.assertIn('eval_win_rate', phase0)
        self.assertIn('eval_win_pct', phase0)
        self.assertIn('eval_wins', phase0)
        self.assertEqual(phase0['eval_games'], 4)
        self.assertEqual(phase0['round'], 1)

        self.trainer.run_phase(round_idx=1, train_player=2, num_episodes=2)
        self.assertEqual(len(self.trainer.agents[0].memory.memory), p0_after_first)
        self.assertGreater(len(self.trainer.agents[2].memory.memory), p2_after_first)

    def test_train_alternates_between_players(self):
        history = self.trainer.train()
        self.assertEqual(len(history), 2)
        self.assertEqual([phase['train_player'] for phase in history], [0, 2])

    def test_augmented_observation_size(self):
        raw_state, player_id = self.trainer.env.reset()
        aug = self.trainer._augment_state(raw_state, player_id)
        self.assertEqual(aug['obs'].shape[0], AUGMENTED_OBS_DIM)

    def test_tracks_best_result(self):
        history = self.trainer.train()
        self.assertEqual(len(history), 2)
        self.assertIsNotNone(self.trainer.best_result)
        self.assertIn('eval_win_rate', self.trainer.best_result)

    def test_rule_initialization_collects_samples(self):
        states, actions = self.trainer._collect_rule_demonstrations(player_id=0, num_hands=1)
        self.assertGreater(len(states), 0)
        self.assertEqual(len(states), len(actions))
        self.assertEqual(states.shape[1], AUGMENTED_OBS_DIM)


if __name__ == '__main__':
    unittest.main(verbosity=2)
