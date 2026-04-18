from rlcard.envs import Env
from rlcard.games.euchre import Game
from rlcard.utils.euchre_utils import ACTION_SPACE, ACTION_LIST
import numpy as np

class EuchreEnv(Env):

    def __init__(self, config):
        self.game = Game()
        self.name = "euchre"

        self.actions = ACTION_LIST
        self.state_shape = [len(self.actions)]
        super().__init__(config)

    def _extract_state(self, state):
        def vec(s):
            suit = {"C":0, "D":1, "H":2, "S":3}
            rank = {"9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
            if len(s)==1:
                return np.array([ suit[s[0]] ])
            else:
                return np.array([ suit[s[0]], rank[s[1]] ])

        state['legal_actions'] = self._get_legal_actions()
        state['raw_legal_actions'] = self.game.get_legal_actions()

        obs = []
        if state['trump'] is not None:
            obs += [ vec(state['trump']) ]
        else:
            obs += [ np.array([-1]) ]
        if state['flipped'] is not None:
            obs += [ vec(state['flipped']) ]
        else:
            obs += [ np.array([-1,-1]) ]
        if state['lead_suit'] is not None:
            obs += [ vec(state['lead_suit']) ]
        else:
            obs += [ np.array([-1]) ]
        obs += [ vec(e) for e in state['hand'] ]
        obs += [ np.zeros(2*(6-len(state['hand'])))-1 ]
        obs += [ vec(e.get_index()) for e in state['center'] ]
        obs += [ np.zeros(2*(4-len(state['center'])))-1 ]
        obs += [ state['seen'] ]
        state['obs'] = np.hstack(obs)
        
        return state

    def _decode_action(self, action_id):
        return ACTION_LIST[action_id]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = [ACTION_SPACE[action] for action in legal_actions]
        return legal_ids

    def get_payoffs(self):
        return self.game.get_payoffs()

    def get_global_state(self):
        deck = [k for k, v in sorted(ACTION_SPACE.items(),
                key=lambda x: x[1]) if 6 <= ACTION_SPACE[k] <= 29]
        # 4 binary hand vectors (96 dims)
        hands = []
        for p in self.game.players:
            held = {c.get_index() for c in p.hand}
            hands.append(np.array([1.0 if c in held else 0.0 for c in deck]))
        # trump one-hot (4 dims)
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        trump_oh = np.zeros(4)
        if self.game.trump is not None:
            trump_oh[suit_map[self.game.trump]] = 1.0
        # seen (24 dims)
        # team scores (2 dims)
        s = self.game.score
        scores = np.array([s[0] + s[2], s[1] + s[3]], dtype=float)
        # trick number (1 dim)
        trick_num = np.array([5 - len(self.game.players[0].hand)], dtype=float)
        return np.concatenate(hands + [trump_oh, self.game.seen, scores, trick_num])
        # total: 96 + 4 + 24 + 2 + 1 = 127 dims