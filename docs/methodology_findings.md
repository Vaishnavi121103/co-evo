# Methodology findings from calibrating against real EMBER-2018

These results come from calibrating the harness on the real dataset before
running the policy study. They are reported here because each one is a
*negative* result that changes how a feature-space co-evolution experiment
must be built, and because two of them are invisible in synthetic data --
they only appear at EMBER's dimensionality and with a genuinely strong
classifier.

## Setup

* **Data.** EMBER-2018 (v2), the official release, md5-verified. Raw features
  are vectorized with EMBER's own extractor (`feature_version=2`), giving the
  canonical **2381-dimensional** vector. A balanced subsample is drawn; the
  evaluation pool is disjoint from training data (enforced by sha256).
* **Defender.** Histogram gradient-boosted trees (the algorithmic stand-in for
  EMBER's LightGBM reference model).
* **Attack surface.** Only feature blocks an append/add-style,
  functionality-preserving mutation can touch: byte histogram, byte-entropy
  histogram, strings, sections, imports.

**Defender baseline.** 98.50% clean accuracy (`max_iter=150`), 98.42% at
`max_iter=80` for half the fit cost. This matches published EMBER baselines
and confirms the pipeline reproduces the reference setting.

| Defender config | Clean accuracy | Fit time |
|---|---|---|
| `max_iter=150, lr=0.05` | 0.9850 | 35.4 s |
| `max_iter=80,  lr=0.08` | 0.9842 | 17.0 s |
| `max_iter=60,  lr=0.10, max_features=0.2` | 0.9800 | 11.7 s |

`max_iter=80` is used throughout: 0.08% accuracy for a 2x cost reduction.

## Finding 1 — "add-only" is not a faithful constraint on EMBER features

A functionality-preserving mutation only ever *adds* to a binary, so it is
tempting to model every mutable feature as monotonically increasing. On
EMBER's actual feature vector that is wrong:

* the byte and byte-entropy histograms are **normalized**, so raising one bin
  necessarily lowers others;
* the section and import blocks are produced by `FeatureHasher`, which emits
  **signed** values -- adding an import can move a hashed feature *down*.

Only a handful of features are genuine monotone counters (string counts,
section counts). Modelling all ~1076 mutable features as add-only is both
unfaithful and self-defeating: it can only push a sample further into
malicious territory. Correcting this leaves 5 add-only features and 1071
bidirectional ones.

## Finding 2 — random mutation directions do not scale to EMBER dimensionality

The natural first action space is a set of random perturbation directions over
the mutable features. It works in a low-dimensional synthetic feature space
and **fails silently** at EMBER scale:

| Action space | Attack success (best over step sizes) |
|---|---|
| 16 random directions, add-only | 0.041 |
| 16 random directions, bidirectional | 0.014 |
| **16 block-wise mimicry directions** | **0.130 – 0.194** |

The diagnostic is that attack success was *insensitive to step size* -- a
larger budget bought nothing. In ~10^3 dimensions a random vector is nearly
orthogonal to any direction that lowers the malicious score, so no step
budget produces evasion.

Real gym-malware actions are not random either: they are semantic edits with
systematic effects. The faithful feature-space analogue used here is a
**mimicry** action set -- move along `mean(benign) - mean(malicious)`,
restricted to one contiguous block of mutable features at a time, so each
action means *"make this aspect of the file look more like benign software."*
With mimicry actions, attack success becomes monotone in step size, which is
the signature of an action space that is actually doing work.

## Finding 3 — a probability-difference reward gives no learning signal

A confident classifier saturates: it reports `p(malicious) ~ 0.999` over many
consecutive mutations. A reward shaped on the probability difference is then
~0 almost everywhere, and the agent receives a near-constant per-step penalty
whichever action it picks. With no gradient, Q-values stay flat and the greedy
policy collapses onto one arbitrary action.

Shaping on **log-odds** keeps the signal dense, since `log(p/(1-p))` still
moves while `p` is pinned near 1. This measurably helps DQN
(0.007 -> 0.030 attack success at `step_scale=4.0`) but, as Finding 4 records,
does not by itself make RL competitive.

## Finding 4 — RL attackers underperform a stochastic mimicry policy

Against the same mimicry action space and query budget:

| Attacker | Attack success |
|---|---|
| Stochastic policy over mimicry actions | **0.194** |
| DQN, 150 episodes | 0.052 |
| DQN, 1000 cumulative episodes | 0.104 |
| PPO, 150 episodes (greedy eval) | 0.030 – 0.097 |

DQN improves with a cumulative budget -- which is what the co-evolution loop
actually provides, since the attacker persists across rounds -- but plateaus
around 0.104, roughly half the stochastic baseline.

The mechanism is structural, not a tuning failure. A **greedy** policy applies
its single argmax action repeatedly, so it edits only *one* feature block for
the whole episode. A stochastic policy mixes directions and accumulates
movement across *all* blocks, which is a strictly better composite move in
this action space. Greedy action selection is therefore handicapped by
construction here, independent of how well the value function is learned.

This is consistent with the instability this literature reports for RL malware
evasion agents (and with the risk register's expectation that DQN can converge
to a near-random policy), but it identifies a specific cause rather than
attributing it to optimisation noise.

**Consequence for the study.** The retraining-policy comparison is the
contribution; the attacker only needs to supply strong, consistent, and
reproducible adversarial pressure. The main factorial study therefore uses the
strongest available attacker, and the attacker-algorithm comparison above is
reported as a secondary finding rather than assumed.
