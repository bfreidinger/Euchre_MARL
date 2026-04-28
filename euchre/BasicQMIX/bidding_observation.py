import numpy as np


BASE_OBS_DIM = 48
SUIT_ORDER = ("C", "D", "H", "S")
SUIT_TO_IDX = {suit: idx for idx, suit in enumerate(SUIT_ORDER)}
LEFT_BOWER_BY_TRUMP = {"C": "SJ", "S": "CJ", "H": "DJ", "D": "HJ"}
RIGHT_BOWER_BY_TRUMP = {suit: f"{suit}J" for suit in SUIT_ORDER}
HIGH_TRUMP_CARDS = {"A", "K", "Q"}

BIDDING_FEATURE_DIM = (
    1 +   # in bidding branch
    2 +   # bidding round one-hot
    4 +   # seat relative to dealer one-hot
    1 +   # partner is dealer
    4 +   # turned down suit one-hot
    4 +   # suit counts in hand
    4 +   # trump candidate counts by suit
    4 +   # right bower by suit
    4 +   # left bower by suit
    4 +   # high trump cards by suit
    4 +   # off-suit aces by suit
    1     # partner already passed this round
)
OBS_DIM = BASE_OBS_DIM + BIDDING_FEATURE_DIM


def _one_hot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= index < size:
        vec[index] = 1.0
    return vec


def _suit_one_hot(suit):
    return _one_hot(SUIT_TO_IDX.get(suit), len(SUIT_ORDER))


def _card_codes(state):
    return list(state.get("hand", []))


def _is_bidding_branch(game):
    return game.trump is None


def _bidding_round_features(game):
    if game.trump is not None:
        return np.zeros(2, dtype=np.float32)
    if game.turned_down is None:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0], dtype=np.float32)


def _dealer_relative_features(game, player_id):
    if game.dealer_player_id is None:
        return np.zeros(4, dtype=np.float32)
    relative = (player_id - game.dealer_player_id) % 4
    return _one_hot(relative, 4)


def _partner_is_dealer(game, player_id):
    if game.dealer_player_id is None:
        return np.array([0.0], dtype=np.float32)
    partner_id = (player_id + 2) % 4
    return np.array([1.0 if game.dealer_player_id == partner_id else 0.0], dtype=np.float32)


def _count_cards_by_suit(hand_cards):
    counts = np.zeros(4, dtype=np.float32)
    for card in hand_cards:
        counts[SUIT_TO_IDX[card[0]]] += 1.0
    return counts / 5.0


def _trump_strength_features(hand_cards):
    counts = np.zeros(4, dtype=np.float32)
    right_bower = np.zeros(4, dtype=np.float32)
    left_bower = np.zeros(4, dtype=np.float32)
    high_trump = np.zeros(4, dtype=np.float32)
    off_suit_aces = np.zeros(4, dtype=np.float32)

    hand_set = set(hand_cards)
    for suit, idx in SUIT_TO_IDX.items():
        suit_cards = [card for card in hand_cards if card[0] == suit]
        right_bower[idx] = 1.0 if RIGHT_BOWER_BY_TRUMP[suit] in hand_set else 0.0
        left_bower[idx] = 1.0 if LEFT_BOWER_BY_TRUMP[suit] in hand_set else 0.0
        counts[idx] = min(len(suit_cards) + left_bower[idx], 5.0) / 5.0

        high_count = sum(1 for card in suit_cards if card[1] in HIGH_TRUMP_CARDS)
        high_trump[idx] = high_count / 5.0

        ace_count = sum(1 for card in hand_cards if card[1] == "A" and card[0] != suit)
        off_suit_aces[idx] = ace_count / 4.0

    return counts, right_bower, left_bower, high_trump, off_suit_aces


def _partner_passed_this_round(game, player_id):
    """1.0 if the partner bid before this player in the current round (and thus passed)."""
    if game.trump is not None or game.dealer_player_id is None:
        return np.array([0.0], dtype=np.float32)
    partner_id = (player_id + 2) % 4
    # Acting order within a round: left of dealer = 0, dealer = 3
    my_order      = (player_id  - game.dealer_player_id - 1) % 4
    partner_order = (partner_id - game.dealer_player_id - 1) % 4
    # If partner acted first and it's still our turn, partner must have passed
    return np.array([float(partner_order < my_order)], dtype=np.float32)


def augment_state(env, state, player_id):
    game = env.game
    obs = state["obs"].astype(np.float32)

    if not _is_bidding_branch(game):
        zeros = np.zeros(BIDDING_FEATURE_DIM, dtype=np.float32)
        state = dict(state)
        state["obs"] = np.concatenate([obs, zeros]).astype(np.float32)
        return state

    hand_cards = _card_codes(state)
    suit_counts = _count_cards_by_suit(hand_cards)
    trump_counts, right_bower, left_bower, high_trump, off_suit_aces = _trump_strength_features(hand_cards)

    extra = np.concatenate([
        np.array([1.0], dtype=np.float32),
        _bidding_round_features(game),
        _dealer_relative_features(game, player_id),
        _partner_is_dealer(game, player_id),
        _suit_one_hot(game.turned_down),
        suit_counts,
        trump_counts,
        right_bower,
        left_bower,
        high_trump,
        off_suit_aces,
        _partner_passed_this_round(game, player_id),
    ]).astype(np.float32)

    state = dict(state)
    state["obs"] = np.concatenate([obs, extra]).astype(np.float32)
    return state
