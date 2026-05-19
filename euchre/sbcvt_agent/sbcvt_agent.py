"""
SBCVT Agent: Statistical Belief search with Cooperative Value factorization for Euchre.

At inference time, wraps a trained QMIXAgent and replaces the greedy argmax
with an MCTS search where:
  - The belief sampler produces plausible hidden deals for the acting player.
  - QMIX Q-values serve as a prior (PUCT, eq. 28) and leaf evaluator (V_mix, eq. 26).
  - Partner actions inside simulations are drawn from the partner's Q-policy (eq. 25).
  - Opponent actions inside simulations follow the rule-based agent policy.

Training is unchanged — this module is inference-only.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import math
import numpy as np
import torch
from copy import deepcopy

from rlcard.utils.euchre_utils import ACTION_SPACE, ACTION_LIST
from rlcard.core import Card
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent


# ---------------------------------------------------------------------------
# Deck / encoding constants
# ---------------------------------------------------------------------------

DECK = [k for k, v in sorted(ACTION_SPACE.items(), key=lambda x: x[1])
        if 6 <= v <= 29]   # 24 playable cards in ACTION_SPACE order

SUIT_MAP = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
RANK_MAP = {'9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


# ---------------------------------------------------------------------------
# Module-level helpers for obs / state extraction
# ---------------------------------------------------------------------------

def _card_from_name(name: str) -> Card:
    """Reconstruct a Card from its index string (e.g. 'HJ' -> Card('H','J'))."""
    return Card(name[0], name[1:])


def _vec(s: str) -> np.ndarray:
    """Encode a card-index string ('HJ') or suit string ('H') to a numeric vector."""
    if len(s) == 1:
        return np.array([SUIT_MAP[s[0]]], dtype=np.float32)
    return np.array([SUIT_MAP[s[0]], RANK_MAP[s[1]]], dtype=np.float32)


def _extract_obs(game, player_id: int) -> np.ndarray:
    """
    Build the 48-dim local observation vector for player_id directly from a
    game object, mirroring EuchreEnv._extract_state().

    Dimension breakdown:
      trump     : 1
      flipped   : 2
      lead_suit : 1
      hand      : 12  (up to 6 cards × 2, padded with -1)
      center    : 8   (up to 4 cards × 2, padded with -1)
      seen      : 24
      ─────────────
      total     : 48
    """
    state = game.get_state(player_id)
    obs = []

    obs.append(_vec(state['trump']) if state['trump'] is not None else np.array([-1.0]))
    obs.append(_vec(state['flipped']) if state['flipped'] is not None else np.array([-1.0, -1.0]))
    obs.append(_vec(state['lead_suit']) if state['lead_suit'] is not None else np.array([-1.0]))

    hand = state['hand']  # list of card-index strings from cards2list()
    for c in hand:
        obs.append(_vec(c))
    obs.append(np.full(2 * (6 - len(hand)), -1.0, dtype=np.float32))

    # state['center'] stores Card objects from game.center
    center = state['center']
    for c in center:
        obs.append(_vec(c.get_index()))
    obs.append(np.full(2 * (4 - len(center)), -1.0, dtype=np.float32))

    obs.append(state['seen'].astype(np.float32))

    return np.hstack(obs)


def _extract_global_state(game) -> np.ndarray:
    """
    Build the 127-dim global state vector directly from a game object,
    mirroring EuchreEnv.get_global_state().

    Dimension breakdown:
      4 binary hand vectors : 4 × 24 = 96
      trump one-hot         : 4
      seen                  : 24
      team scores           : 2
      trick number          : 1
      ──────────────────────────
      total                 : 127
    """
    hands = []
    for p in game.players:
        held = {c.get_index() for c in p.hand}
        hands.append(np.array([1.0 if c in held else 0.0 for c in DECK], dtype=np.float32))

    trump_oh = np.zeros(4, dtype=np.float32)
    if game.trump is not None:
        trump_oh[SUIT_MAP[game.trump]] = 1.0

    s = game.score
    scores = np.array([s[0] + s[2], s[1] + s[3]], dtype=np.float32)
    trick_num = np.array([5 - len(game.players[0].hand)], dtype=np.float32)

    return np.concatenate(hands + [trump_oh, game.seen.astype(np.float32), scores, trick_num])


# ---------------------------------------------------------------------------
# Bidding snapshot  (used by statistical belief weighting, eq. 7)
# ---------------------------------------------------------------------------

class BiddingSnapshot:
    """One bidding decision: who acted, which action, and the public flipped card."""
    __slots__ = ('player_id', 'action_id', 'flipped_name')

    def __init__(self, player_id: int, action_id: int, flipped_name):
        self.player_id    = player_id
        self.action_id    = action_id
        self.flipped_name = flipped_name   # 'HJ'-style index string or None (round-2 bidding)


def _bidding_obs(flipped_name, hand_names: list) -> np.ndarray:
    """
    Build the 48-dim obs for a player at a bidding decision point.

    At bidding time trump is unknown, lead_suit is None, center is empty,
    and no tricks have been played (seen = zeros). Only the hand and the
    flipped card vary across sampled deals.
    """
    obs = []
    obs.append(np.array([-1.0]))                                             # trump: unknown
    obs.append(_vec(flipped_name) if flipped_name is not None
               else np.array([-1.0, -1.0]))                                  # flipped
    obs.append(np.array([-1.0]))                                             # lead_suit: None

    for c in hand_names:
        obs.append(_vec(c))
    obs.append(np.full(2 * (6 - len(hand_names)), -1.0, dtype=np.float32))  # hand padding

    obs.append(np.full(8, -1.0, dtype=np.float32))                          # center: empty
    obs.append(np.zeros(24, dtype=np.float32))                              # seen: none

    return np.hstack(obs)


# ---------------------------------------------------------------------------
# Belief Sampler
# ---------------------------------------------------------------------------

class BeliefSampler:
    """
    Uniform determinization: given what player `player_id` observes,
    randomly redistributes unseen cards to the other three players while
    preserving each player's known hand size.

    What the acting player observes:
      - Their own hand        (exact cards known)
      - game.seen             (24-bit mask of all cards played in past tricks)
      - game.center           (cards played so far in the current trick)
      - game.flipped_card     (public during round-1 bidding; excluded from pool)
      - game.turned_down_card (exact card turned down in round 1, if present)

    There are always 3 kitty cards unknown to everyone — random.sample handles
    them by leaving them out of the deal implicitly.

    Statistical belief (weighted by opponent/partner likelihoods, eq. 7 in
    the paper) is left as a TODO — uniform sampling is the baseline.
    """

    def __init__(self, player_id: int):
        self.player_id = player_id

    def sample(self, game) -> object:
        """
        Returns a deepcopy of `game` with hidden hands replaced by a
        uniformly random legal deal consistent with player_id's observations.
        """
        my_hand = {c.get_index() for c in game.players[self.player_id].hand}
        seen_cards = {DECK[i] for i in range(24) if game.seen[i] == 1}
        center_cards = {c.get_index() for c in game.center}

        known = my_hand | seen_cards | center_cards

        # Flipped card is public during round-1 bidding — not in any hand.
        if game.flipped_card is not None:
            known.add(game.flipped_card.get_index())

        # Turned-down card is out of play after the dealer passes in round 1.
        # game.py stores its full index in turned_down_card for exactly this purpose.
        if getattr(game, 'turned_down_card', None) is not None:
            known.add(game.turned_down_card)

        # Hidden pool: pool is larger than total_needed by the 3 kitty cards.
        # random.sample draws without replacement, leaving kitty cards out implicitly.
        hidden = [c for c in DECK if c not in known]

        other_players = [p for p in range(4) if p != self.player_id]
        hand_sizes = {p: len(game.players[p].hand) for p in other_players}
        total_needed = sum(hand_sizes.values())

        sampled_cards = random.sample(hidden, total_needed)

        sampled_game = deepcopy(game)
        offset = 0
        for p in other_players:
            size = hand_sizes[p]
            sampled_game.players[p].hand = [
                _card_from_name(c) for c in sampled_cards[offset: offset + size]
            ]
            offset += size

        return sampled_game

    # TODO: weighted_sample(game, partner_model, opp_model)
    #   Weight each candidate deal by the likelihood that partner/opponents
    #   would have bid and played as observed (eq. 7 in the paper).


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    A node in the MCTS tree keyed by the acting player's information set.
    Stores per-action visit counts N[a] and cumulative returns W[a].
    """

    def __init__(self, legal_actions: list):
        self.legal_actions = legal_actions
        self.N = {a: 0 for a in legal_actions}
        self.W = {a: 0.0 for a in legal_actions}
        self.children = {}   # action_id -> MCTSNode

    def q_hat(self, action) -> float:
        if self.N[action] == 0:
            return 0.0
        return self.W[action] / self.N[action]

    def total_visits(self) -> int:
        return sum(self.N.values())


# ---------------------------------------------------------------------------
# SBCVT Agent
# ---------------------------------------------------------------------------

class SBCVTAgent:
    """
    Inference-time wrapper around a trained QMIXAgent.

    For each decision, runs M MCTS simulations using:
      - BeliefSampler  to sample hidden deals
      - QMIX Q-network as PUCT prior (eq. 25 / 28)
      - QMIX V_mix     as leaf evaluator (eq. 26)
      - Partner Q-policy for simulating partner actions
      - Random play for opponent actions

    Args:
        agent0:      trained DQNAgent for player 0
        agent2:      trained DQNAgent for player 2
        mixer:       trained mixing network (MixingNetwork or QMIXMixer)
        env:         EuchreEnv (provides the live game for belief sampling)
        num_sims:    M, number of MCTS simulations per decision
        max_depth:   d_max, rollout depth before leaf evaluation
        c_puct:      exploration constant in PUCT (eq. 28)
        temperature: softmax temperature for prior policy (eq. 25)
        device:      torch device
    """

    def __init__(self,
                 agent0,
                 agent2,
                 mixer,
                 env,
                 num_sims: int = 100,
                 max_depth: int = 5,
                 c_puct: float = 1.0,
                 temperature: float = 1.0,
                 n_candidates: int = 200,
                 epsilon: float = 0.0,
                 device=None,
                 use_target_nets: bool = False,
                 rollout_bid_rule: bool = False,
                 rollout_to_end: bool = False,
                 target_mixer=None,
                 team: tuple = (0, 2)):

        self._p0, self._p1 = team          # e.g. (0,2) or (1,3)
        self._team_ids = set(team)
        self._payoff_player = team[0]       # whose payoff represents the team reward
        self.agents = {team[0]: agent0, team[1]: agent2}
        self.mixer = mixer
        self.env = env
        self.num_sims = num_sims
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.device = device or torch.device('cpu')

        self.epsilon = epsilon

        # When True, card-play rollouts use the agents' target networks for
        # stability. target_mixer must be kept in sync by the training loop
        # (sbcvt.target_mixer = qmix.target_mixer after each sync_targets()).
        self.use_target_nets = use_target_nets
        self.target_mixer = target_mixer

        # When True, non-acting-player bidding inside rollouts uses the
        # rule-based agent instead of the Q-network (avoids propagating
        # a biased or random Q-prior into MCTS bidding evaluations).
        self.rollout_bid_rule = rollout_bid_rule

        # When True, unvisited leaf nodes are evaluated by playing the game
        # to completion with rule-based agents rather than using the Q-network
        # value estimate. This gives actual final payoffs (+1/+2/-1/-2) and
        # is correct from episode 1 regardless of Q-network training progress.
        self.rollout_to_end = rollout_to_end

        self.belief_samplers = {
            self._p0: BeliefSampler(player_id=self._p0),
            self._p1: BeliefSampler(player_id=self._p1),
        }
        self._bidding_snapshots: list = []
        self._rule_agent = EuchreRuleAgent()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, state: dict, player_id: int) -> int:
        """
        Select an action for `player_id` using MCTS over sampled hidden deals.

        During the bidding phase, belief samples are weighted by the likelihood
        of observed partner bidding actions under the Q-network (eq. 7).
        After trump is set, uniform determinization is used.

        Args:
            state:     extracted state dict from EuchreEnv (contains 'legal_actions')
            player_id: 0 or 2

        Returns:
            action_id (int) in ACTION_SPACE
        """
        game = self.env.game
        sampler = self.belief_samplers[player_id]
        legal_actions = state['legal_actions']

        root = MCTSNode(legal_actions)

        # Build a weighted pool once per step when in bidding phase and partner has acted.
        partner_id = self._p1 if player_id == self._p0 else self._p0
        partner_snaps = [s for s in self._bidding_snapshots if s.player_id == partner_id]
        use_weighted = game.trump is None and bool(partner_snaps)

        if use_weighted:
            pool, pool_weights = self._build_weighted_pool(game, player_id, partner_id, partner_snaps)

        for _ in range(self.num_sims):
            if use_weighted:
                idx = np.random.choice(len(pool), p=pool_weights)
                sampled_game = deepcopy(pool[idx])
            else:
                sampled_game = sampler.sample(game)
            self._simulate(sampled_game, root, player_id, depth=0)

        return self._select_action(root)

    # ------------------------------------------------------------------
    # MCTS core  (Algorithm 1 & 2 from the paper)
    # ------------------------------------------------------------------

    def _simulate(self, game, node: MCTSNode, acting_player: int, depth: int) -> float:
        """
        Recursive MCTS simulation following Algorithm 2.
        Returns the discounted return from this node onward.

        Expansion rule: always select an action at the current node. If the
        child reached by that action is new (never visited), evaluate it with
        the QMIX leaf value and stop. If the child already exists in the tree,
        recurse. This is the standard "expand on first child visit" pattern.
        """
        if game.is_over():
            payoffs = game.get_payoffs()
            return payoffs.get(self._payoff_player, 0.0)

        if depth >= self.max_depth:
            return self._rollout_to_end(game) if self.rollout_to_end else self._leaf_value(game)

        current_player = game.current_player

        if current_player == acting_player:
            action_id = self._puct_select(node, current_player, game)
        elif current_player in self._team_ids:
            if self.rollout_bid_rule and game.trump is None:
                action_id = self._opponent_action(game)  # rule-based for partner bidding
            else:
                action_id = self._partner_action(game, current_player)
        else:
            action_id = self._opponent_action(game)

        game.step(ACTION_LIST[action_id])

        # First visit to this child: evaluate and add to tree.
        # Subsequent visits: recurse deeper into the existing child node.
        if action_id not in node.children:
            next_legal = [ACTION_SPACE[a] for a in game.get_legal_actions()]
            node.children[action_id] = MCTSNode(next_legal)
            G = (self._rollout_to_end(game) if self.rollout_to_end
                 else self._leaf_value(game))
        else:
            G = self._simulate(game, node.children[action_id], acting_player, depth + 1)

        if current_player == acting_player:
            node.N[action_id] += 1
            node.W[action_id] += G

        return G

    def _puct_select(self, node: MCTSNode, player_id: int, game) -> int:
        """
        PUCT selection rule (eq. 28). Uses QMIX Q-values as a prior policy
        over legal actions (eq. 25) to bias exploration toward promising moves.

        When self.epsilon > 0, a random legal action is chosen with that
        probability before PUCT runs (used during training for exploration).
        """
        legal = node.legal_actions
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(legal)
        obs = _extract_obs(game, player_id)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_vals = self.agents[player_id].q_estimator.qnet(obs_t)[0]  # (54,)

        legal_q = torch.tensor(
            [q_vals[a].item() for a in legal], dtype=torch.float32
        ) / self.temperature
        prior = torch.softmax(legal_q, dim=0).cpu().numpy()

        total = max(node.total_visits(), 1)
        best_action, best_score = None, -float('inf')
        for a, p in zip(legal, prior):
            score = node.q_hat(a) + self.c_puct * p * math.sqrt(total) / (1 + node.N[a])
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def _leaf_value(self, game) -> float:
        """
        V_mix(s, h): evaluate a leaf using the QMIX mixer (eq. 26).

        By Proposition 1 (monotone mixer), independently maximizing each
        agent's Q-value gives the optimal joint action, so we pass the
        per-agent max Q-values directly to the mixer.
        """
        if game.is_over():
            payoffs = game.get_payoffs()
            return payoffs.get(self._payoff_player, 0.0)

        obs0 = _extract_obs(game, self._p0)
        obs2 = _extract_obs(game, self._p1)
        gs   = _extract_global_state(game)

        obs0_t = torch.FloatTensor(obs0).unsqueeze(0).to(self.device)
        obs2_t = torch.FloatTensor(obs2).unsqueeze(0).to(self.device)
        gs_t   = torch.FloatTensor(gs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_target_nets:
                q0 = self.agents[self._p0].target_estimator.qnet(obs0_t)
                q2 = self.agents[self._p1].target_estimator.qnet(obs2_t)
                mixer = self.target_mixer if self.target_mixer is not None else self.mixer
            else:
                q0 = self.agents[self._p0].q_estimator.qnet(obs0_t)
                q2 = self.agents[self._p1].q_estimator.qnet(obs2_t)
                mixer = self.mixer
            q0_max = q0.max(dim=1)[0]                        # (1,)
            q2_max = q2.max(dim=1)[0]                        # (1,)
            q_agents = torch.stack([q0_max, q2_max], dim=1) # (1, 2)
            v_mix = mixer(q_agents, gs_t)                    # (1, 1)

        return v_mix.item()

    def _rollout_to_end(self, game) -> float:
        """
        Play the game to completion using rule-based agents for all players.
        Returns the actual final payoff for player 0's team (+1/+2/-1/-2).
        The game object is mutated in place — callers must pass a copy.
        """
        while not game.is_over():
            player_id  = game.current_player
            legal_strs = game.get_legal_actions()
            state = {
                'raw_legal_actions': legal_strs,
                'hand':        [c.get_index() for c in game.players[player_id].hand],
                'trump_called': game.trump is not None,
                'trump':        game.trump,
                'turned_down':  game.turned_down,
                'lead_suit':    game.lead_suit,
                'flipped':      (game.flipped_card.get_index()
                                 if game.flipped_card is not None else None),
                'center':       game.center,
                'order':        game.order,
                'seen':         game.seen,
            }
            action_id = self._rule_agent.step(state)
            game.step(ACTION_LIST[action_id])
        return game.get_payoffs().get(self._payoff_player, 0.0)

    def _partner_action(self, game, partner_id: int) -> int:
        """
        Sample a partner action from the softmax of their Q-network (eq. 25),
        restricted to legal actions.
        """
        legal = [ACTION_SPACE[a] for a in game.get_legal_actions()]
        obs = _extract_obs(game, partner_id)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_target_nets:
                q_vals = self.agents[partner_id].target_estimator.qnet(obs_t)[0]
            else:
                q_vals = self.agents[partner_id].q_estimator.qnet(obs_t)[0]

        legal_q = torch.tensor(
            [q_vals[a].item() for a in legal], dtype=torch.float32
        ) / self.temperature
        probs = torch.softmax(legal_q, dim=0).cpu().numpy()
        return int(np.random.choice(legal, p=probs))

    def _opponent_action(self, game) -> int:
        """Rule-based action for opponents during rollouts."""
        player_id = game.current_player
        legal_strs = game.get_legal_actions()
        state = {
            'raw_legal_actions': legal_strs,
            'hand': [c.get_index() for c in game.players[player_id].hand],
            'trump_called': game.trump is not None,
            'trump': game.trump,
            'turned_down': game.turned_down,
            'lead_suit': game.lead_suit,
            'flipped': game.flipped_card.get_index() if game.flipped_card is not None else None,
            'center': game.center,
            'order': game.order,
            'seen': game.seen,
        }
        return self._rule_agent.step(state)

    # ------------------------------------------------------------------
    # Action selection from root statistics
    # ------------------------------------------------------------------

    def _select_action(self, root: MCTSNode) -> int:
        """Return most-visited action at the root (robust child selection)."""
        return max(root.legal_actions, key=lambda a: root.N[a])

    # ------------------------------------------------------------------
    # Statistical belief (eq. 7) — bidding-phase hand weighting
    # ------------------------------------------------------------------

    def new_hand(self) -> None:
        """Clear bidding history for a new hand. Call after each env.reset()."""
        self._bidding_snapshots = []

    def record_action(self, player_id: int, action_id: int, game) -> None:
        """
        Record a QMIX-team bidding action before env.step().
        No-op once trump is set (card-play phase).

        Args:
            player_id:  0 or 2
            action_id:  ACTION_SPACE index of the chosen action
            game:       live game object (used to read the current flipped card)
        """
        if game.trump is not None:
            return
        flipped = game.flipped_card
        self._bidding_snapshots.append(BiddingSnapshot(
            player_id    = player_id,
            action_id    = action_id,
            flipped_name = flipped.get_index() if flipped is not None else None,
        ))

    def _build_weighted_pool(self, game, acting_player: int,
                             partner_id: int, partner_snaps: list):
        """
        Generate n_candidates uniform deals and weight each by the likelihood
        of the partner's observed bidding actions under their Q-network (eq. 7).

        Batches all Q-network forward passes across the pool for efficiency.

        Returns:
            pool         : list of sampled game copies (deepcopies of game)
            pool_weights : np.ndarray of shape (n_candidates,), sums to 1
        """
        sampler       = self.belief_samplers[acting_player]
        partner_agent = self.agents[partner_id]
        n             = self.n_candidates

        pool  = [sampler.sample(game) for _ in range(n)]
        log_w = np.zeros(n, dtype=np.float64)

        for snap in partner_snaps:
            # Build partner's obs at this bidding point for every candidate deal.
            obs_batch = []
            for cand_game in pool:
                hand_names = [c.get_index() for c in cand_game.players[partner_id].hand]
                obs_batch.append(_bidding_obs(snap.flipped_name, hand_names))

            obs_t = torch.FloatTensor(np.stack(obs_batch)).to(self.device)   # (n, 48)
            with torch.no_grad():
                q_vals = partner_agent.q_estimator.qnet(obs_t)               # (n, ACTION_NUM)

            log_probs = torch.log_softmax(q_vals / self.temperature, dim=1)[:, snap.action_id]
            log_w += log_probs.cpu().numpy()

        log_w -= log_w.max()    # numerical stability before exp
        weights = np.exp(log_w)
        weights /= weights.sum()

        return pool, weights
