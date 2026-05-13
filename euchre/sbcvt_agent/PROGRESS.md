# SBCVT Agent — Progress Log

## Goal

Beat a rule-based Euchre agent at a win rate significantly above 50% using a cooperative
multi-agent system. The SBCVT agent wraps a trained QMIX model with MCTS at inference
time to produce better decisions than greedy Q-value selection alone.

The core problem we are trying to solve: the SBCVT agent is **calling trump too
aggressively**, resulting in roughly 34–35% of all hands ending in the opponent
euchring us. Getting euchred surrenders 2 points to the opponent, which wipes out
the scoring advantage from winning more hands overall.

---

## Architecture

- **QMIX**: Two DQN agents (players 0 and 2) trained cooperatively against rule-based
  opponents for 5M episodes. A monotone mixing network combines their Q-values into a
  joint Q_tot. Trained checkpoint: `BasicQMIX/qmix_euchre.pt`.

- **SBCVT** (`sbcvt_agent/sbcvt_agent.py`): Inference-time wrapper around QMIX. At each
  decision, runs MCTS over sampled hidden deals (determinization). QMIX Q-values serve
  as the PUCT prior and leaf evaluator. Partner actions in simulations are sampled from
  the partner's Q-policy; opponent actions use a configurable model.

- **Evaluation** (`sbcvt_agent/run_sbcvt.py`): Runs N full games (first to 10 points)
  and reports win rate, hand outcomes, and euchre/march rates.

---

## What We Measured

### Baseline: SBCVT (both partners) vs Rule-Based — 100 games
```
Game win rate      : 45%
OPP euchred SBCVT  : 35.2% of hands
SBCVT march        : 11.5% of hands
```
Despite winning more hands outright, SBCVT loses the scoring race because every euchre
gives the opponent 2 points. Scoring math: SBCVT earns ~761 pts vs opponent's ~853 pts.

---

## What We Tried

### Option A — Fix rollout opponent model (completed)

**What**: `_opponent_action` in `sbcvt_agent.py` originally picked a random legal action.
We replaced it with `EuchreRuleAgent.step()`, so simulated opponents now call trump with
the same ≥3-card threshold as the real opponents.

**Result**:
```
Game win rate      : 49%  (up from 45%)
OPP euchred SBCVT  : 34.2%  (down from 35.2%)
```
Marginal improvement. The rollout model wasn't the main culprit. The Q-network's biased
prior (trained to value calling highly) is the dominant problem — MCTS can't overcome it
even with accurate opponent rollouts.

---

### Option B — MCTS-guided bidding during fine-tuning (in progress, not working)

**What**: Modified `run_episode` in `train_qmix.py` to use SBCVT instead of epsilon-greedy
for trump-calling decisions during training. Card play stays epsilon-greedy. Replay buffer
starts fresh to avoid old biased transitions. MCTS epsilon decays 0.5 → 0.05 over 300k
episodes to force exploration of both call and pass early on.

Fine-tuning script: `sbcvt_agent/train_qmix_mcts.py`
- Loads `qmix_euchre.pt`
- 3× base learning rate to speed up unlearning
- Fresh 100k replay buffer
- 20 MCTS sims per bidding decision

**Result** (230k episodes in):
```
Call rate          : ~93%  (essentially unchanged — still calling almost every hand)
Euchre/call rate   : ~43%  (barely moving)
Win rate           : ~53%  (marginally better, but call rate unchanged)
```
The fine-tuning is not correcting the overcalling bias. Root cause: even with MCTS epsilon
at 0.5, the biased Q-prior pushes the majority of non-random simulations toward calling,
so calling still accumulates the most visits and wins root selection. The replay buffer
remains dominated by calling transitions. TD updates cannot overcome the strength of the
prior bias within a reasonable number of episodes.

---

## Root Cause Analysis

Three compounding biases all push MCTS toward calling trump:

| Bias | Source |
|------|--------|
| Q-prior biased toward calling | 5M episodes of training where calling was net-profitable |
| Leaf value inflated at bidding nodes | `_leaf_value` takes max Q, which = call value when calling has higher Q |
| ~~Random opponents pass too often~~ | Fixed in Option A |

The fundamental issue is a **chicken-and-egg loop**: the Q-network's prior is biased toward
calling → MCTS follows the biased prior → replay buffer stays biased → Q-network doesn't
unlearn. TD updates alone cannot break this loop when the prior is this strong.

---

## Next Step — Option C: KL Divergence Policy Loss

**Idea**: At each bidding decision during training, run MCTS and record the visit
distribution `N[a] / ΣN` as a policy target. Add an auxiliary KL divergence loss that
directly pushes the Q-network's softmax to match this distribution:

```
L_total = L_QMIX_TD  +  λ · KL( π_MCTS  ||  softmax(Q_bid / τ) )
```

This breaks the chicken-and-egg loop by *directly correcting Q-values for bidding*
rather than waiting for TD to slowly propagate the signal. The Q-network is forced to
assign lower Q to calling when MCTS (with rule-based rollouts) says to pass, regardless
of what the prior currently says.

**Why this is different from Option B**: Option B only changes which actions go into the
replay buffer and hopes TD self-corrects. Option C adds a loss term that explicitly
penalizes the Q-network for disagreeing with the MCTS policy — a much stronger and more
direct training signal.

**Implementation plan**:
1. After each MCTS bidding decision, store `(obs, legal_actions, mcts_visit_dist)`.
2. After each episode, compute KL loss over the stored bidding targets.
3. Backprop KL loss through the Q-network (same optimizer as QMIX TD loss).
4. Tune λ to balance TD accuracy (card play) vs policy correction (bidding).
