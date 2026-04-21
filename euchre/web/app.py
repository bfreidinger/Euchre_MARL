import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from http import cookies
import json
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from jinja2 import Environment, FileSystemLoader, select_autoescape
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import rlcard
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.euchre_utils import ACTION_LIST


ACTION_NUM = 54
OBS_DIM = 48
MATCH_TARGET = 10

CHECKPOINTS = {
    "baseline": os.path.join(ROOT_DIR, "BasicQMIX", "qmix_euchre.pt"),
    "neutral": os.path.join(ROOT_DIR, "BasicQMIX", "qmix_neutral.pt"),
    "aggressive": os.path.join(ROOT_DIR, "Personalities", "qmix_aggressive.pt"),
    "timid": os.path.join(ROOT_DIR, "Personalities", "qmix_timid.pt"),
}

POLICY_LABELS = {
    "rule": "Rule-based",
    "random": "Random",
    "baseline": "Baseline QMIX",
    "neutral": "Neutral QMIX",
    "aggressive": "Aggressive QMIX",
    "timid": "Timid QMIX",
}

SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
RANK_LABELS = {"A": "A", "K": "K", "Q": "Q", "J": "J", "T": "10", "9": "9"}
PLAYER_NAMES = {
    0: "You",
    1: "Left Opponent",
    2: "Partner",
    3: "Right Opponent",
}

ENV = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")),
    autoescape=select_autoescape(["html"]),
)
SESSIONS = {}
SESSION_LOCK = threading.Lock()
CHECKPOINT_CACHE = {}


def load_checkpoint(path):
    if path not in CHECKPOINT_CACHE:
        CHECKPOINT_CACHE[path] = torch.load(path, map_location="cpu")
    return CHECKPOINT_CACHE[path]


def build_model_policy(family, role):
    if family == "rule":
        return EuchreRuleAgent()
    if family == "random":
        return RandomAgent(ACTION_NUM)

    checkpoint = load_checkpoint(CHECKPOINTS[family])
    key = "agent0_qnet" if role == "opp_left" else "agent2_qnet"
    agent = DQNAgent(
        scope=f"{family}_{role}",
        action_num=ACTION_NUM,
        state_shape=[OBS_DIM],
        mlp_layers=[128, 128],
    )
    agent.q_estimator.qnet.load_state_dict(checkpoint[key])
    agent.target_estimator.qnet.load_state_dict(checkpoint[key])
    return agent


def format_card(card):
    if not card:
        return "None"
    suit = card[0]
    rank = card[1]
    return f"{RANK_LABELS.get(rank, rank)}{SUIT_SYMBOLS.get(suit, suit)}"


def card_payload(card, face_up=True, label=None):
    if not card:
        return {
            "face_up": face_up,
            "rank": "",
            "suit": "",
            "display": label or "Empty",
            "color": "neutral",
        }
    suit = card[0]
    rank = card[1]
    color = "red" if suit in {"H", "D"} else "black"
    return {
        "face_up": face_up,
        "rank": RANK_LABELS.get(rank, rank),
        "suit": SUIT_SYMBOLS.get(suit, suit),
        "display": format_card(card),
        "color": color,
    }


def format_action(action):
    if action == "pass":
        return "Pass"
    if action == "pick":
        return "Order Up"
    if action.startswith("call-"):
        suit = action.split("-", 1)[1]
        return f"Call {SUIT_SYMBOLS[suit]}"
    if action.startswith("discard-"):
        return f"Discard {format_card(action.split('-', 1)[1])}"
    return f"Play {format_card(action)}"


def action_kind(action):
    if action in {"pass", "pick"} or action.startswith("call-"):
        return "bid"
    if action.startswith("discard-"):
        return "discard"
    return "play"


def suit_color(suit):
    return "red" if suit in {"H", "D", "♥", "♦"} else "black"


def action_visuals(action_text):
    payload = {
        "suit": "",
        "suit_symbol": "",
        "suit_color": "neutral",
        "card": None,
    }
    kind = action_kind(action_text)
    if kind in {"play", "discard"}:
        card_code = action_text.split("-", 1)[1] if action_text.startswith("discard-") else action_text
        payload["card"] = card_payload(card_code)
        payload["suit"] = card_code[0]
        payload["suit_symbol"] = SUIT_SYMBOLS.get(card_code[0], card_code[0])
        payload["suit_color"] = suit_color(card_code[0])
    elif action_text.startswith("call-"):
        suit = action_text.split("-", 1)[1]
        payload["suit"] = suit
        payload["suit_symbol"] = SUIT_SYMBOLS.get(suit, suit)
        payload["suit_color"] = suit_color(suit)
    return payload


def redirect(start_response, location):
    start_response("303 See Other", [("Location", location)])
    return [b""]


def json_response(start_response, payload, status="200 OK"):
    data = json.dumps(payload).encode("utf-8")
    start_response(status, [("Content-Type", "application/json; charset=utf-8")])
    return [data]


def parse_post(environ):
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or 0)
    except ValueError:
        length = 0
    body = environ["wsgi.input"].read(length).decode("utf-8")
    data = parse_qs(body)
    return {key: values[0] for key, values in data.items()}


def get_session_id(environ):
    raw_cookie = environ.get("HTTP_COOKIE", "")
    if not raw_cookie:
        return None
    parsed = cookies.SimpleCookie()
    parsed.load(raw_cookie)
    if "euchre_session" not in parsed:
        return None
    return parsed["euchre_session"].value


@dataclass
class MatchSession:
    mode: str
    south_family: str
    partner_family: str
    opp_left_family: str
    opp_right_family: str
    south_agent: object | None = field(init=False, default=None)
    partner_agent: object = field(init=False)
    opp_left_agent: object = field(init=False)
    opp_right_agent: object = field(init=False)
    env: object = field(init=False)
    state: dict = field(init=False, default=None)
    current_player: int = field(init=False, default=0)
    user_score: int = field(init=False, default=0)
    opp_score: int = field(init=False, default=0)
    match_over: bool = field(init=False, default=False)
    hand_over: bool = field(init=False, default=False)
    hand_result: str = field(init=False, default="")
    status: str = field(init=False, default="")
    action_log: list = field(init=False, default_factory=list)
    animation_tick: int = field(init=False, default=0)
    last_move: dict | None = field(init=False, default=None)
    reveal_trick: list = field(init=False, default_factory=list)
    reveal_winner: int | None = field(init=False, default=None)
    trump_caller: int | None = field(init=False, default=None)
    trump_call_label: str = field(init=False, default="")
    current_trick_actions: list = field(init=False, default_factory=list)
    completed_tricks: list = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.mode == "spectator":
            self.south_agent = build_model_policy(self.south_family, "south")
        self.partner_agent = build_model_policy(self.partner_family, "partner")
        self.opp_left_agent = build_model_policy(self.opp_left_family, "opp_left")
        self.opp_right_agent = build_model_policy(self.opp_right_family, "opp_right")
        self.env = rlcard.make("euchre", config={"num_players": 4})
        self.start_hand()

    def start_hand(self):
        self.hand_over = False
        self.hand_result = ""
        self.status = "New hand started."
        self.action_log = []
        self.last_move = None
        self.reveal_trick = []
        self.reveal_winner = None
        self.trump_caller = None
        self.trump_call_label = ""
        self.current_trick_actions = []
        self.completed_tricks = []
        self.state, self.current_player = self.env.reset()
        self._update_turn_status()

    def _agent_for_player(self, player_id):
        if player_id == 0:
            return self.south_agent
        if player_id == 1:
            return self.opp_left_agent
        if player_id == 2:
            return self.partner_agent
        if player_id == 3:
            return self.opp_right_agent
        return None

    def _append_log(self, player_id, action_text):
        self.action_log.append(f"{PLAYER_NAMES[player_id]}: {format_action(action_text)}")
        self.animation_tick += 1
        visuals = action_visuals(action_text)
        self.last_move = {
            "player_id": player_id,
            "player_name": PLAYER_NAMES[player_id],
            "raw": action_text,
            "label": format_action(action_text),
            "kind": action_kind(action_text),
            "card": visuals["card"],
        }
        if action_kind(action_text) == "play":
            self.current_trick_actions.append(
                {
                    "player_id": player_id,
                    "player_name": PLAYER_NAMES[player_id],
                    "label": format_action(action_text),
                    "card": visuals["card"],
                }
            )
        if action_text == "pick" or action_text.startswith("call-"):
            self.trump_caller = player_id
            self.trump_call_label = format_action(action_text)

    def _capture_trick_reveal(self, acting_player, action_text):
        if action_kind(action_text) != "play":
            self.reveal_trick = []
            self.reveal_winner = None
            return

        prior_center = list(self.env.game.center)
        prior_order = list(self.env.game.order)
        trick_will_end = len(prior_center) == 3
        if not trick_will_end:
            self.reveal_trick = []
            self.reveal_winner = None
            return

        self.reveal_trick = [
            {
                "owner": PLAYER_NAMES[owner],
                "player_id": owner,
                "card": card_payload(card.get_index()),
            }
            for owner, card in zip(prior_order, prior_center)
        ]
        self.reveal_trick.append(
            {
                "owner": PLAYER_NAMES[acting_player],
                "player_id": acting_player,
                "card": card_payload(action_text),
            }
        )

    def _update_turn_status(self):
        if self.match_over:
            return
        if self.hand_over:
            self.status = self.hand_result or "Hand over."
            return
        if self.reveal_trick and self.reveal_winner is not None:
            self.status = (
                f"{PLAYER_NAMES[self.reveal_winner]} took the trick. "
                f"{PLAYER_NAMES[self.current_player]} is up next."
            )
            return
        if self.current_player == 0 and self.mode != "spectator":
            self.status = "Your turn."
        else:
            self.status = f"{PLAYER_NAMES[self.current_player]} is up next."

    def _finalize_revealed_trick(self):
        if not self.reveal_trick or self.reveal_winner is None:
            return
        winner_team = "You + Partner" if self.reveal_winner in {0, 2} else "Opponents"
        summary = {
            "winner_id": self.reveal_winner,
            "winner_name": PLAYER_NAMES[self.reveal_winner],
            "winner_team": winner_team,
            "plays": list(self.reveal_trick),
        }
        self.completed_tricks.append(summary)
        self.completed_tricks = self.completed_tricks[-5:]
        self.current_trick_actions = []

    def play_human_action(self, action_id):
        if self.mode == "spectator":
            self.status = "Spectator mode is running policies for every seat."
            return
        if self.match_over:
            self.status = "The match is already over. Start a new match to keep playing."
            return
        if self.hand_over:
            self.status = "This hand is over. Start the next hand."
            return
        if self.current_player != 0:
            self.status = "It is not your turn yet."
            return
        if action_id not in self.state["legal_actions"]:
            self.status = "That action is not legal in the current state."
            return

        action_text = ACTION_LIST[action_id]
        acting_player = self.current_player
        self._capture_trick_reveal(acting_player, action_text)
        self._append_log(0, action_text)
        self.state, self.current_player = self.env.step(action_id)
        if self.reveal_trick:
            self.reveal_winner = self.current_player
        if self.env.is_over():
            self._finalize_hand()
        else:
            self._update_turn_status()

    def advance_ai_turn(self):
        if self.match_over or self.hand_over or (self.current_player == 0 and self.mode != "spectator"):
            self._update_turn_status()
            return False

        agent = self._agent_for_player(self.current_player)
        action_id, _ = agent.eval_step(self.state)
        action_text = ACTION_LIST[action_id]
        acting_player = self.current_player
        self._capture_trick_reveal(acting_player, action_text)
        self._append_log(acting_player, action_text)
        self.state, self.current_player = self.env.step(action_id)
        if self.reveal_trick:
            self.reveal_winner = self.current_player

        if self.env.is_over():
            self._finalize_hand()
        else:
            self._update_turn_status()
        return True

    def _finalize_hand(self):
        self.hand_over = True
        self._finalize_revealed_trick()
        payoffs = self.env.game.get_payoffs()
        human_team_points = max(payoffs.get(0, 0), 0)
        opp_team_points = max(-payoffs.get(0, 0), 0)
        self.user_score += human_team_points
        self.opp_score += opp_team_points

        if human_team_points:
            result = f"You and your partner won the hand for {human_team_points} point"
            if human_team_points != 1:
                result += "s"
            if human_team_points == 2:
                result += " (march or euchre)"
        else:
            result = f"The opponents won the hand for {opp_team_points} point"
            if opp_team_points != 1:
                result += "s"

        self.hand_result = result + "."
        self.status = self.hand_result
        if self.user_score >= MATCH_TARGET or self.opp_score >= MATCH_TARGET:
            self.match_over = True
            if self.user_score >= MATCH_TARGET:
                self.status = f"Match over: you and your partner reached {self.user_score}."
            else:
                self.status = f"Match over: the opponents reached {self.opp_score}."

    def next_hand(self):
        if self.match_over:
            self.status = "The match is already over. Start a new match to play again."
            return
        if not self.hand_over:
            self.status = "Finish the current hand before starting a new one."
            return
        self.start_hand()

    def _clear_reveal_if_present(self):
        self._finalize_revealed_trick()
        self.reveal_trick = []
        self.reveal_winner = None

    def template_context(self):
        public_state = self.state or {}
        human_state = self.env.get_state(0)
        center_cards = []
        center_by_player = {player_id: None for player_id in PLAYER_NAMES}
        active_center = self.reveal_trick
        if not active_center and hasattr(self.env.game, "center"):
            for owner, card in zip(self.env.game.order, self.env.game.center):
                payload = {
                    "owner": PLAYER_NAMES[owner],
                    "player_id": owner,
                    "card": card_payload(card.get_index()),
                }
                center_cards.append(payload)
                center_by_player[owner] = payload
        else:
            for payload in active_center:
                center_cards.append(payload)
                center_by_player[payload["player_id"]] = payload

        legal_actions = []
        grouped_actions = {"bid": [], "play": [], "discard": []}
        if self.current_player == 0 and self.mode != "spectator" and not self.hand_over and self.state:
            raw_actions = self.state["raw_legal_actions"]
            for action_id, action_text in zip(self.state["legal_actions"], raw_actions):
                item = {
                    "id": action_id,
                    "label": format_action(action_text),
                    "raw": action_text,
                    "kind": action_kind(action_text),
                    **action_visuals(action_text),
                }
                legal_actions.append(item)
                grouped_actions[item["kind"]].append(item)

        action_lookup = {}
        for item in legal_actions:
            if item["kind"] == "play":
                action_lookup[item["raw"]] = item
            elif item["kind"] == "discard":
                action_lookup[item["raw"].split("-", 1)[1]] = item

        visible_hand = []
        for card in human_state.get("hand", []):
            payload = card_payload(card)
            if card in action_lookup:
                payload["action_id"] = action_lookup[card]["id"]
                payload["action_kind"] = action_lookup[card]["kind"]
            else:
                payload["action_id"] = None
                payload["action_kind"] = None
            visible_hand.append(payload)
        def make_visible_cards(player_id: int):
            if self.mode == "spectator" or player_id == 0:
                return [card_payload(card) for card in self.env.game.get_state(player_id)["hand"]]
            count = len(self.env.game.players[player_id].hand)
            return [card_payload("SA", face_up=False, label="Hidden")] * count

        seat_info = {
            0: {
                "name": PLAYER_NAMES[0],
                "position": "south",
                "count": len(self.env.game.players[0].hand),
                "cards": make_visible_cards(0),
                "face_up": True,
                "is_current": self.current_player == 0,
                "is_dealer": self.env.game.dealer_player_id == 0,
            },
            1: {
                "name": PLAYER_NAMES[1],
                "position": "west",
                "count": len(self.env.game.players[1].hand),
                "cards": make_visible_cards(1),
                "face_up": self.mode == "spectator",
                "is_current": self.current_player == 1,
                "is_dealer": self.env.game.dealer_player_id == 1,
            },
            2: {
                "name": PLAYER_NAMES[2],
                "position": "north",
                "count": len(self.env.game.players[2].hand),
                "cards": make_visible_cards(2),
                "face_up": self.mode == "spectator",
                "is_current": self.current_player == 2,
                "is_dealer": self.env.game.dealer_player_id == 2,
            },
            3: {
                "name": PLAYER_NAMES[3],
                "position": "east",
                "count": len(self.env.game.players[3].hand),
                "cards": make_visible_cards(3),
                "face_up": self.mode == "spectator",
                "is_current": self.current_player == 3,
                "is_dealer": self.env.game.dealer_player_id == 3,
            },
        }

        bidding_prompt = None
        if grouped_actions["bid"]:
            if any(item["raw"] == "pick" for item in grouped_actions["bid"]):
                bidding_prompt = "Order up the flipped card or pass."
            elif any(item["raw"].startswith("call-") for item in grouped_actions["bid"]):
                bidding_prompt = "Choose the trump suit or pass if allowed."
        elif grouped_actions["discard"]:
            bidding_prompt = "Dealer must discard one card after ordering up."
        elif grouped_actions["play"]:
            bidding_prompt = "Choose a card to play to the trick."

        current_options = []
        if self.state and not self.hand_over:
            for action_id, action_text in zip(self.state.get("legal_actions", []), self.state.get("raw_legal_actions", [])):
                current_options.append(
                    {
                        "id": action_id,
                        "raw": action_text,
                        "label": format_action(action_text),
                        "kind": action_kind(action_text),
                        **action_visuals(action_text),
                    }
                )

        visible_trick = []
        if self.reveal_trick:
            for play in self.reveal_trick:
                visible_trick.append(
                    {
                        "player_name": play["owner"],
                        "card": play["card"],
                    }
                )
        else:
            visible_trick = list(self.current_trick_actions)

        recent_trick = self.completed_tricks[-1] if self.completed_tricks else None

        return {
            "policy_labels": POLICY_LABELS,
            "selected": {
                "mode": self.mode,
                "south": self.south_family,
                "partner": self.partner_family,
                "opp_left": self.opp_left_family,
                "opp_right": self.opp_right_family,
            },
            "match_target": MATCH_TARGET,
            "score": {
                "you": self.user_score,
                "opp": self.opp_score,
            },
            "status": self.status,
            "hand_result": self.hand_result,
            "match_over": self.match_over,
            "hand_over": self.hand_over,
            "animation_tick": self.animation_tick,
            "mode": self.mode,
            "spectator_mode": self.mode == "spectator",
            "auto_advance": ((self.mode == "spectator" or self.current_player != 0) and not self.hand_over and not self.match_over),
            "ai_thinking": ((self.mode == "spectator" or self.current_player != 0) and not self.hand_over and not self.match_over),
            "current_player": PLAYER_NAMES.get(self.current_player, "Unknown"),
            "current_player_id": self.current_player,
            "dealer": PLAYER_NAMES.get(self.env.game.dealer_player_id, "Unknown"),
            "dealer_id": self.env.game.dealer_player_id,
            "trump": format_card(f"{self.env.game.trump}9")[1:] if self.env.game.trump else "Not chosen",
            "trump_caller": PLAYER_NAMES.get(self.trump_caller, "Not called yet") if self.trump_caller is not None else "Not called yet",
            "trump_call_label": self.trump_call_label or "No trump call yet",
            "flipped": card_payload(self.env.game.flipped_card.get_index()) if self.env.game.flipped_card else None,
            "turned_down": (
                SUIT_SYMBOLS.get(self.env.game.turned_down, self.env.game.turned_down)
                if self.env.game.turned_down
                else "None"
            ),
            "team_tricks": self.env.game.score[0] + self.env.game.score[2],
            "opp_tricks": self.env.game.score[1] + self.env.game.score[3],
            "center_cards": center_cards,
            "center_by_player": center_by_player,
            "last_move": self.last_move,
            "reveal_winner": self.reveal_winner,
            "hand_cards": visible_hand,
            "seats": seat_info,
            "legal_actions": legal_actions,
            "grouped_actions": grouped_actions,
            "bidding_prompt": bidding_prompt,
            "current_options": current_options,
            "current_trick_actions": visible_trick,
            "recent_trick": recent_trick,
            "action_log": list(reversed(self.action_log[-16:])),
        }


def render(start_response, session=None):
    template = ENV.get_template("index.html")
    context = {
        "policy_labels": POLICY_LABELS,
        "selected": {
            "mode": "human",
            "south": "baseline",
            "partner": "rule",
            "opp_left": "rule",
            "opp_right": "rule",
        },
        "has_session": session is not None,
    }
    if session is not None:
        context.update(session.template_context())
        context["has_session"] = True
        context["state_json"] = json.dumps(session.template_context())
    else:
        context["state_json"] = "null"

    html = template.render(**context).encode("utf-8")
    headers = [("Content-Type", "text/html; charset=utf-8")]
    start_response("200 OK", headers)
    return [html]


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()

    if path == "/static/style.css":
        css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")
        with open(css_path, "rb") as handle:
            payload = handle.read()
        start_response("200 OK", [("Content-Type", "text/css; charset=utf-8")])
        return [payload]
    if path == "/static/app.js":
        js_path = os.path.join(os.path.dirname(__file__), "static", "app.js")
        with open(js_path, "rb") as handle:
            payload = handle.read()
        start_response("200 OK", [("Content-Type", "application/javascript; charset=utf-8")])
        return [payload]

    session_id = get_session_id(environ)
    session = SESSIONS.get(session_id)

    if method == "POST" and path == "/new-match":
        form = parse_post(environ)
        session = MatchSession(
            mode=form.get("mode", "human"),
            south_family=form.get("south", "baseline"),
            partner_family=form.get("partner", "rule"),
            opp_left_family=form.get("opp_left", "rule"),
            opp_right_family=form.get("opp_right", "rule"),
        )
        session_id = uuid.uuid4().hex
        with SESSION_LOCK:
            SESSIONS[session_id] = session
        if environ.get("HTTP_X_REQUESTED_WITH") == "fetch":
            start_response("200 OK", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Set-Cookie", f"euchre_session={session_id}; Path=/; SameSite=Lax"),
            ])
            return [json.dumps(session.template_context()).encode("utf-8")]
        start_response("303 See Other", [
            ("Location", "/"),
            ("Set-Cookie", f"euchre_session={session_id}; Path=/; SameSite=Lax"),
        ])
        return [b""]

    if method == "POST" and path == "/act":
        if session is not None:
            session._clear_reveal_if_present()
            form = parse_post(environ)
            try:
                action_id = int(form["action"])
            except (KeyError, ValueError):
                session.status = "Could not read that action."
            else:
                session.play_human_action(action_id)
        if environ.get("HTTP_X_REQUESTED_WITH") == "fetch":
            return json_response(start_response, session.template_context() if session else {"error": "No active session"}, "200 OK" if session else "400 Bad Request")
        return redirect(start_response, "/")

    if method == "POST" and path == "/next-hand":
        if session is not None:
            session.next_hand()
        if environ.get("HTTP_X_REQUESTED_WITH") == "fetch":
            return json_response(start_response, session.template_context() if session else {"error": "No active session"}, "200 OK" if session else "400 Bad Request")
        return redirect(start_response, "/")

    if method == "POST" and path == "/advance":
        if session is None:
            return json_response(start_response, {"error": "No active session"}, "400 Bad Request")
        if session.reveal_trick:
            session._clear_reveal_if_present()
            session._update_turn_status()
            return json_response(start_response, session.template_context())
        session.advance_ai_turn()
        return json_response(start_response, session.template_context())

    return render(start_response, session=session)


# Standard WSGI entrypoint for gunicorn.
application = app


def main():
    host = os.environ.get("EUCHRE_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT") or os.environ.get("EUCHRE_WEB_PORT", "8000"))
    with make_server(host, port, app) as server:
        print(f"Euchre web app running at http://{host}:{port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
