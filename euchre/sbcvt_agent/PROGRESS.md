# Hybrid MCTS-QMIX Agent — Architecture & Progress

## Goal

Beat a rule-based Euchre agent at a win rate significantly above 50% using a cooperative
multi-agent system that cleanly separates two distinct sub-problems:

- **Bidding (call/pass trump)**: a look-ahead reasoning problem best solved by search
- **Card play**: a cooperative coordination problem best solved by learned multi-agent RL

---

## Why Previous Approaches Failed

The original SBCVT agent wrapped a QMIX model trained for 5M episodes with MCTS at
inference time. QMIX Q-values served as both the PUCT prior and the leaf evaluator inside
MCTS. This caused a compounding overcalling bias:

| Bias | Source |
|------|--------|
| Q-prior biased toward calling | 5M episodes where calling was net-profitable on average |
| Leaf value inflated at bidding nodes | `_leaf_value` takes max Q, which equals the call Q when calling has higher value |
| Chicken-and-egg loop | Biased prior → MCTS votes for call → replay buffer stays biased → Q-network never unlearns |

Two fine-tuning attempts (fixing rollout opponents, MCTS-guided bidding during fine-tuning)
produced only marginal improvement because the biased Q-prior was too strong to overcome
through TD updates alone.

---

## Current Architecture: Hybrid MCTS-QMIX

### Core Idea

Bidding and card play are handled by entirely separate systems that each do only what they
are best at. QMIX never learns a bidding policy, so its Q-values are never corrupted by
calling bias.

```
Bidding phase  →  MCTS with rule-based rollouts to game completion
Card-play phase →  QMIX (cooperative DQN with mixing network)
```

---

### Training (`sbcvt_agent/train_hybrid.py`)

QMIX is trained from scratch with fresh random weights. Every episode proceeds as follows:

**Bidding phase (trump calling):**
- When player 0 or player 2 must decide whether to call or pass, `SBCVTAgent.step()` is
  called to run MCTS over sampled hidden deals (determinization).
- Each MCTS simulation:
  1. Samples a plausible hidden deal (randomizes unseen cards to other players)
  2. Selects bid or pass at the root using PUCT
  3. All remaining bidding decisions (partner and opponents) are played out by the
     rule-based agent
  4. All five tricks are played out to game completion by the rule-based agent
  5. Returns the actual final payoff: +1 (win), +2 (march), -1 (loss), -2 (euchred)
- After all `num_sims` simulations, the action with the most visits is chosen — whichever
  of call/pass produced better average payoffs across sampled deals wins.
- **Bidding transitions are NOT stored in the replay buffer.** QMIX never trains on
  bidding states, so its Q-values stay unbiased for card play.

**Card-play phase:**
- Player 0 and player 2 select cards using QMIX epsilon-greedy (`agent.step()`).
- These transitions ARE stored in the replay buffer as joint (obs0, obs2, action0, action2,
  reward, global_state, next_state, done) tuples.
- Intermediate trick rewards: +0.25 per trick won by the team.
- Terminal reward: final game payoff (+1/+2/-1/-2).
- Standard QMIX TD updates train both Q-networks and the mixing network jointly.

**Why rollout-to-end matters:**
The original SBCVT used the Q-network's `_leaf_value` to evaluate MCTS leaf nodes. With
fresh random weights, those estimates are meaningless noise. Using rule-based rollouts to
game completion gives real payoffs from episode 1, making MCTS useful immediately.

**Why rule-based inside rollouts:**
Since QMIX starts with random weights, using it for card-play rollouts inside MCTS would
just add noise. Rule-based play is stable, reasonable, and fast. The QMIX target networks
are configured for rollouts but with `rollout_to_end=True` the rule-based agent handles
everything inside simulations.

**Target network stability:**
After every `TARGET_SYNC_EVERY` episodes, QMIX target networks are synced and
`sbcvt.target_mixer` is refreshed so the rollout leaf evaluator always uses the most
recent stable snapshot.

**Training hyperparameters:**
```
Episodes        : 2,000,000 (default)
MCTS sims       : 10 per bidding decision (default)
MCTS depth      : 10 (falls back to rule-based rollout at depth limit)
Epsilon         : 1.0 → 0.05 over 15M total_t steps (card play exploration)
Learning rate   : 5e-4
Replay buffer   : 500,000 transitions (card play only)
Warmup          : 50,000 episodes before any gradient updates
Target sync     : every 10,000 episodes
Eval            : every 10,000 episodes (200 hands, MCTS bidding + greedy QMIX cards)
Checkpoint out  : BasicQMIX/qmix_hybrid.pt
```

Run training:
```
python3 sbcvt_agent/train_hybrid.py
python3 sbcvt_agent/train_hybrid.py --episodes 2000000 --sims 10 --depth 10
```

---

### SBCVTAgent Flags (`sbcvt_agent/sbcvt_agent.py`)

The SBCVTAgent class supports several modes controlled by constructor flags:

| Flag | Default | Effect |
|------|---------|--------|
| `rollout_to_end` | False | Replace Q-network leaf evaluation with rule-based rollout to game completion |
| `rollout_bid_rule` | False | Use rule-based agent for non-acting-player bidding inside rollouts |
| `use_target_nets` | False | Use target Q-networks (instead of online nets) for card-play rollouts |
| `target_mixer` | None | Target mixing network; must be refreshed after each `sync_targets()` call |

For hybrid training/evaluation, all four are set: `rollout_to_end=True`,
`rollout_bid_rule=True`, `use_target_nets=True`.

---

### Evaluation (`sbcvt_agent/run_sbcvt.py`)

Runs N full games (first team to 10 points wins). The `--hybrid` flag mirrors the training
setup: MCTS for bidding decisions, greedy QMIX (`eval_step`) for card play.

```
python3 sbcvt_agent/run_sbcvt.py --hybrid --partner sbcvt --opp rule --games 100
```

| Flag | Default | Options |
|------|---------|---------|
| `--hybrid` | off | Enables MCTS-bid / QMIX-cards split |
| `--ckpt` | `BasicQMIX/qmix_hybrid.pt` | Path to checkpoint |
| `--sims` | 100 | MCTS simulations per bidding decision |
| `--depth` | 10 | Max tree depth |
| `--partner` | `qmix` | `qmix` (greedy) or `sbcvt` (also runs MCTS) |
| `--opp` | `rule` | `rule`, `random`, or `qmix` |
| `--games` | 100 | Number of full games |
| `--quiet` | off | Suppress per-hand output |

Output includes: game win rate ± 95% CI, avg hands per game, and a full hand outcome
breakdown (march, euchre for/against, normal win/loss).

---

## Key Metrics to Watch

| Metric | Bad | Good |
|--------|-----|------|
| Call rate | >85% (overcalling) | 55–70% |
| Euchre/call | >30% | <20–25% |
| Game win rate | <50% | >55% |

The original QMIX-only agent called ~93% of hands and was euchred ~43% of the time it
called. The hybrid architecture eliminates the Q-prior calling bias entirely — MCTS
evaluates call vs pass purely on simulated outcomes.
