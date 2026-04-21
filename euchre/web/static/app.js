(function () {
  const initialStateEl = document.getElementById("initial-state");
  const setupForm = document.getElementById("setup-form");
  const nextHandBtn = document.getElementById("next-hand-btn");
  const tableStage = document.getElementById("table-stage");
  const modeSelect = document.getElementById("mode-select");
  const southPolicyWrap = document.getElementById("south-policy-wrap");

  let state = initialStateEl ? JSON.parse(initialStateEl.textContent || "null") : null;
  let autoAdvanceTimer = null;
  const TURN_DELAY_MS = 2600;
  const TRICK_REVEAL_MS = 1400;
  let selectedSourceRect = null;

  const suitColors = {
    "♥": "red",
    "♦": "red",
    "♠": "black",
    "♣": "black",
  };

  function seatKey(id) {
    return String(id);
  }

  function post(url, params) {
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "X-Requested-With": "fetch",
      },
      body: new URLSearchParams(params),
      credentials: "same-origin",
    }).then((response) => response.json());
  }

  function cardFaceHtml(card, extraClass = "") {
    const color = card.color || suitColors[card.suit] || "black";
    return `
      <div class="card-shell face ${color} ${extraClass}">
        <span class="rank">${card.rank || ""}</span>
        <span class="suit">${card.suit || ""}</span>
      </div>
    `;
  }

  function cardBackHtml(side = false) {
    return `<div class="card-shell back ${side ? "side" : ""}"></div>`;
  }

  function seatPositionClass(playerId) {
    return {
      0: "south",
      1: "west",
      2: "north",
      3: "east",
    }[playerId];
  }

  function animateClassFor(playerId) {
    return `enter-from-${seatPositionClass(playerId)}`;
  }

  function updateText(id, value) {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = value;
    }
  }

  function updateHtml(id, value) {
    const el = document.getElementById(id);
    if (el) {
      el.innerHTML = value;
    }
  }

  function renderSeat(playerId) {
    const seat = state.seats[seatKey(playerId)];
    const seatEl = document.getElementById(`seat-${playerId}`);
    if (!seat || !seatEl) {
      return;
    }

    seatEl.classList.toggle("active", Boolean(seat.is_current));
    seatEl.classList.toggle(
      "thinking",
      Boolean(state.ai_thinking) && state.current_player_id === playerId
    );
    updateText(`seat-${playerId}-name`, seat.name);
    updateText(
      `seat-${playerId}-meta`,
      `${seat.count} cards${seat.is_dealer ? " · dealer" : ""}`
    );

    const fanEl = document.getElementById(`seat-${playerId}-fan`);
    if (fanEl) {
      if (playerId === 0) {
        fanEl.innerHTML = "";
      } else {
        const side = playerId === 1 || playerId === 3;
        if (seat.face_up) {
          fanEl.innerHTML = (seat.cards || [])
            .map(
              (card) => `
                <div class="card-shell face ${card.color} ${side ? "side" : ""} mini-face">
                  <span class="rank">${card.rank}</span>
                  <span class="suit">${card.suit}</span>
                </div>
              `
            )
            .join("");
        } else {
          fanEl.innerHTML = new Array(seat.count).fill(cardBackHtml(side)).join("");
        }
      }
    }
  }

  function renderHand() {
    const handEl = document.getElementById("human-hand");
    if (!handEl) {
      return;
    }
    handEl.innerHTML = (state.hand_cards || [])
      .map(
        (card) => {
          if (card.action_id) {
            return `
              <button
                type="button"
                class="card-shell face ${card.color} hand-card-large hand-action action-trigger"
                data-action="${card.action_id}"
                data-card-origin="hand">
                <span class="rank">${card.rank}</span>
                <span class="suit">${card.suit}</span>
              </button>
            `;
          }
          return `
            <div class="card-shell face ${card.color} hand-card-large disabled-card">
              <span class="rank">${card.rank}</span>
              <span class="suit">${card.suit}</span>
            </div>
          `;
        }
      )
      .join("");
  }

  function renderFlipped() {
    const wrapper = document.getElementById("flipped-wrapper");
    const cardEl = document.getElementById("flipped-card");
    const rankEl = document.getElementById("flipped-rank");
    const suitEl = document.getElementById("flipped-suit");
    if (!wrapper || !cardEl || !rankEl || !suitEl) {
      return;
    }
    if (!state.flipped) {
      wrapper.hidden = true;
      return;
    }
    wrapper.hidden = false;
    cardEl.className = `card-shell face ${state.flipped.color}`;
    rankEl.textContent = state.flipped.rank;
    suitEl.textContent = state.flipped.suit;
  }

  function renderCenter(previousState) {
    const trickArea = document.getElementById("trick-area");
    if (!trickArea) {
      return;
    }

    const previous = previousState ? previousState.center_by_player || {} : {};
    const current = state.center_by_player || {};
    const order = [
      [2, "north"],
      [1, "west"],
      [3, "east"],
      [0, "south"],
    ];

    trickArea.innerHTML = order
      .map(([playerId, className]) => {
        const item = current[seatKey(playerId)];
        const hadCard = Boolean(previous[seatKey(playerId)]);
        const isNewCard = Boolean(item) && !hadCard;
        if (item) {
          return `
            <div class="trick-slot trick-${className}">
              ${cardFaceHtml(item.card, isNewCard ? `enter-card ${animateClassFor(playerId)}` : "")}
              <span class="slot-label">${item.owner}</span>
            </div>
          `;
        }
        return `
          <div class="trick-slot trick-${className}">
            <div class="ghost-slot"></div>
            <span class="slot-label">${state.seats[seatKey(playerId)].name}</span>
          </div>
        `;
      })
      .join("");
  }

  function animateFloatingCard(card, playerId, sourceRect) {
    const trickArea = document.getElementById("trick-area");
    if (!trickArea || !card) {
      return;
    }

    const targetSlot = trickArea.querySelector(`.trick-${seatPositionClass(playerId)}`);
    if (!targetSlot) {
      return;
    }

    let startRect = sourceRect;
    if (!startRect) {
      const seatEl = document.getElementById(`seat-${playerId}`);
      if (!seatEl) {
        return;
      }
      startRect = seatEl.getBoundingClientRect();
    }

    const targetRect = targetSlot.getBoundingClientRect();
    const floating = document.createElement("div");
    floating.className = "floating-card-layer";
    floating.innerHTML = cardFaceHtml(card, "floating-card");
    document.body.appendChild(floating);

    const cardEl = floating.firstElementChild;
    const startX = startRect.left + startRect.width / 2 - 43;
    const startY = startRect.top + startRect.height / 2 - 62;
    const endX = targetRect.left + targetRect.width / 2 - 43;
    const endY = targetRect.top + targetRect.height / 2 - 62;

    cardEl.style.left = `${startX}px`;
    cardEl.style.top = `${startY}px`;
    cardEl.style.transform = "translate3d(0, 0, 0) scale(1)";

    requestAnimationFrame(() => {
      cardEl.style.transform = `translate3d(${endX - startX}px, ${endY - startY}px, 0) scale(0.98)`;
    });

    window.setTimeout(() => {
      floating.remove();
    }, 760);
  }

  function actionButtonHtml(action) {
    if (action.kind === "play" || action.kind === "discard") {
      const verb = action.kind === "play" ? "Play" : "Discard";
      return `
        <button type="button" data-action="${action.id}" class="action-trigger card-button">
          <span>${action.card.rank}${action.card.suit}</span>
          <small>${verb}</small>
        </button>
      `;
    }
    const klass = action.raw === "pass" ? "secondary" : "accent";
    return `
      <button type="button" data-action="${action.id}" class="action-trigger ${klass}">
        ${action.label}
      </button>
    `;
  }

  function renderActions() {
    const bidSection = document.getElementById("bid-section");
    const discardSection = document.getElementById("discard-section");
    const playSection = document.getElementById("play-section");
    const bidActions = document.getElementById("bid-actions");
    const discardActions = document.getElementById("discard-actions");
    const playActions = document.getElementById("play-actions");
    const noActions = document.getElementById("no-actions");
    const prompt = document.getElementById("bidding-prompt");

    if (prompt) {
      prompt.textContent = state.bidding_prompt || "Waiting for the next action.";
    }

    const grouped = state.grouped_actions || { bid: [], discard: [], play: [] };
    if (bidSection && bidActions) {
      bidSection.hidden = grouped.bid.length === 0;
      bidActions.innerHTML = grouped.bid.map(actionButtonHtml).join("");
    }
    if (discardSection && discardActions) {
      discardSection.hidden = grouped.discard.length === 0;
      discardActions.innerHTML = grouped.discard.map(actionButtonHtml).join("");
    }
    if (playSection && playActions) {
      playSection.hidden = grouped.play.length === 0;
      playActions.innerHTML = grouped.play.map(actionButtonHtml).join("");
    }
    if (noActions) {
      noActions.hidden = (state.legal_actions || []).length > 0 || state.spectator_mode;
    }
  }

  function renderCurrentOptions() {
    const wrap = document.getElementById("current-options");
    if (!wrap) {
      return;
    }
    const options = state.current_options || [];
    if (!options.length) {
      wrap.innerHTML = `<p class="muted">No options available right now.</p>`;
      return;
    }
    wrap.innerHTML = options
      .map(
        (option) => `
          <div class="option-chip ${option.suit_color || ""}">
            <span class="option-label">${option.label}</span>
            ${option.suit_symbol ? `<span class="option-suit">${option.suit_symbol}</span>` : ""}
          </div>
        `
      )
      .join("");
  }

  function renderTrickPanels() {
    const currentTrickEl = document.getElementById("current-trick-ledger");
    const recentTrickEl = document.getElementById("recent-trick-panel");
    if (currentTrickEl) {
      const plays = state.current_trick_actions || [];
      currentTrickEl.innerHTML = plays.length
        ? plays
            .map(
              (play) => `
                <div class="trick-row">
                  <span class="trick-player">${play.player_name}</span>
                  <span class="trick-card ${play.card.color}">
                    <span class="trick-rank">${play.card.rank}</span>
                    <span class="trick-suit">${play.card.suit}</span>
                  </span>
                </div>
              `
            )
            .join("")
        : `<p class="muted">No cards have been played in this trick yet.</p>`;
    }
    if (recentTrickEl) {
      const recent = state.recent_trick;
      recentTrickEl.innerHTML = recent
        ? `
          <p class="recent-trick-title">Last Trick: ${recent.winner_name} won for ${recent.winner_team}</p>
          <div class="recent-trick-cards">
            ${recent.plays
              .map(
                (play) => `
                  <div class="recent-trick-card">
                    <span class="recent-trick-owner">${play.owner}</span>
                    <span class="trick-card ${play.card.color}">
                      <span class="trick-rank">${play.card.rank}</span>
                      <span class="trick-suit">${play.card.suit}</span>
                    </span>
                  </div>
                `
              )
              .join("")}
          </div>
        `
        : `<p class="muted">The first completed trick will appear here.</p>`;
    }
  }

  function renderLog() {
    const logEl = document.getElementById("action-log");
    const emptyEl = document.getElementById("empty-log");
    if (!logEl) {
      return;
    }
    const log = state.action_log || [];
    if (log.length === 0) {
      logEl.hidden = true;
      logEl.innerHTML = "";
      if (emptyEl) {
        emptyEl.hidden = false;
      }
      return;
    }
    if (emptyEl) {
      emptyEl.hidden = true;
    }
    logEl.hidden = false;
    logEl.innerHTML = log.map((item) => `<li>${item}</li>`).join("");
  }

  function applyState(nextState, previousState) {
    const pendingMove = nextState && nextState.last_move ? nextState.last_move : null;
    state = nextState;
    if (!state) {
      return;
    }

    updateText("score-you", state.score.you);
    updateText("score-opp", state.score.opp);
    updateText("table-status", state.status);
    updateText("dealer-label", state.dealer);
    updateText("trump-label", state.trump);
    updateText("trump-caller-label", state.trump_caller);
    updateText("trump-call-action", state.trump_call_label);
    updateText("tricks-label", `${state.team_tricks} - ${state.opp_tricks}`);
    updateText("current-player-label", state.current_player);
    updateText("turned-down-label", state.turned_down);

    const handResult = document.getElementById("hand-result");
    if (handResult) {
      handResult.hidden = !state.hand_result;
      handResult.textContent = state.hand_result || "";
    }

    for (const playerId of [0, 1, 2, 3]) {
      renderSeat(playerId);
    }
    renderFlipped();
    renderCenter(previousState);
    renderHand();
    renderActions();
    renderCurrentOptions();
    renderTrickPanels();
    renderLog();

    if (
      pendingMove &&
      pendingMove.card &&
      (!previousState || nextState.animation_tick !== previousState.animation_tick)
    ) {
      animateFloatingCard(
        pendingMove.card,
        pendingMove.player_id,
        pendingMove.player_id === 0 ? selectedSourceRect : null
      );
    }
    selectedSourceRect = null;

    if (tableStage) {
      tableStage.dataset.tick = String(state.animation_tick || 0);
    }

    scheduleAutoAdvance();
  }

  function clearAutoAdvance() {
    if (autoAdvanceTimer) {
      clearTimeout(autoAdvanceTimer);
      autoAdvanceTimer = null;
    }
  }

  function scheduleAutoAdvance() {
    clearAutoAdvance();
    if (!state || !state.auto_advance) {
      return;
    }
    const delay = state.reveal_winner !== null ? TRICK_REVEAL_MS : TURN_DELAY_MS;
    autoAdvanceTimer = setTimeout(async () => {
      const previousState = state;
      const nextState = await post("/advance", {});
      applyState(nextState, previousState);
    }, delay);
  }

  async function sendAction(actionId) {
    clearAutoAdvance();
    const previousState = state;
    const nextState = await post("/act", { action: actionId });
    applyState(nextState, previousState);
  }

  async function startNewMatch(formData) {
    clearAutoAdvance();
    const hadTable = Boolean(document.getElementById("table-stage"));
    const previousState = state;
    const nextState = await post("/new-match", formData);
    if (!hadTable) {
      window.location.reload();
      return;
    }
    applyState(nextState, previousState);
  }

  async function nextHand() {
    clearAutoAdvance();
    const previousState = state;
    const nextState = await post("/next-hand", {});
    applyState(nextState, previousState);
  }

  document.addEventListener("click", (event) => {
    const actionButton = event.target.closest(".action-trigger");
    if (actionButton) {
      event.preventDefault();
      const actionId = actionButton.getAttribute("data-action");
      if (actionId) {
        if (actionButton.dataset.cardOrigin === "hand") {
          selectedSourceRect = actionButton.getBoundingClientRect();
        } else {
          selectedSourceRect = null;
        }
        sendAction(actionId);
      }
    }
  });

  if (setupForm) {
    setupForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const formData = {};
      new FormData(setupForm).forEach((value, key) => {
        formData[key] = value;
      });
      startNewMatch(formData);
    });
  }

  if (modeSelect && southPolicyWrap) {
    modeSelect.addEventListener("change", () => {
      southPolicyWrap.hidden = modeSelect.value !== "spectator";
    });
  }

  if (nextHandBtn) {
    nextHandBtn.addEventListener("click", (event) => {
      event.preventDefault();
      nextHand();
    });
  }

  if (state) {
    scheduleAutoAdvance();
  }
})();
