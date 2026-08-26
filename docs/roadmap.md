# Co-Evolutionary Malware Detection Using RL and Adaptive Defensive Retraining
## Project Framework & Roadmap (Revised Novelty)

**Student:** Vaishnavi Suryawanshi (25MAI10053) | **Branch:** ME AIML
**Supervisors:** Dr. Manoj Kumar Pandey, Dr. Amit Vajpayee | **Evaluator:** Dr. Richa Sharma

---

## 1. Revised Problem Statement

Machine-learning-based malware classifiers face adaptive, RL-driven evasion attacks. The open
question is no longer *whether* an RL agent can evade a static classifier — that is well
established — but *how a defender that retrains periodically behaves under sustained adversarial
pressure*: does it converge to a robust boundary, oscillate, or stay perpetually behind the
attacker, and **what retraining policy (cadence, data selection, retraining trigger) produces the
best robustness-per-unit-cost tradeoff.**

## 2. Novelty Statement

Prior single-round evasion studies (gym-malware, DQEAF, MERLIN) evaluate attackers against a frozen
classifier and stop at first evasion. Since 2022 a second wave formalizes the attacker–defender
interaction as an iterated / game-theoretic process (Ebrahimi et al. 2022; a 2026 bilevel-
optimization formulation; RAID-venue RL-hardening work).

**What remains open, and what this dissertation targets, is a systematic, controlled comparison of
retraining *policies* within the co-evolutionary framing.** Existing work each proposes and
evaluates a single retraining/defense mechanism in isolation, on its own dataset and attacker. None
systematically varies **retraining cadence**, **retraining data-selection strategy**, and
**attacker budget** as independent variables under a shared harness, or reports the resulting
robustness-vs-cost frontier. The contribution is therefore methodological and empirical:
(a) a reusable co-evolutionary benchmark harness with pluggable attacker/defender/retraining-policy
combinations, (b) a controlled factorial study isolating which retraining-policy variable drives
convergence vs. oscillation, and (c) practical retraining-cadence guidance for EDR vendors expressed
as a cost-robustness tradeoff.

## 3. Related-Work Landscape

| Category | Representative work | What it does | Gap it leaves open |
|---|---|---|---|
| Single-round evasion | Anderson et al. 2018 (gym-malware); MERLIN (Quertier et al. 2022) | RL agent evades a frozen classifier once | No retraining loop |
| Minimax / alternating | Ebrahimi et al. 2022, IEEE ICDMW | Alternates strengthening attacker & detector | Single retraining strategy; no cadence comparison |
| Bilevel co-evolution | 2026 bilevel-optimization paper | Explicit co-evolutionary loop, real families | One defense mechanism; no factorial study |
| RL iterative hardening | RAID "problem-space" hardening | Many adversarial retraining rounds → ~0% ASR | Focused on mechanism, not policy comparison |
| Concept-drift-aware | DRMD (2026) | RL for classify/retrain/reject under natural drift | Natural, not adversarial, drift |

**Positioning:** the controlled-comparison / benchmark layer on top of this cluster — not a new
attacker or defense mechanism.

## 4. System Architecture

```
                 ┌─────────────────────────────┐
                 │   Co-Evolution Orchestrator   │
                 │  (round counter, policy switch)│
                 └───────────┬─────────────────┘
           ┌─────────────────┼─────────────────┐
           ▼                                     ▼
 ┌─────────────────────┐               ┌─────────────────────┐
 │   Attacker Module     │               │   Defender Module    │
 │  RL agent (DQN/PPO)   │◄──feedback───►│  GBDT/EMBER or MLP    │
 │  PE-mutation actions  │   (labels)    │  + retraining policy  │
 └─────────────────────┘               └─────────────────────┘
           └───────────────┬─────────────────────┘
                             ▼
                 ┌─────────────────────────────┐
                 │   Evaluation Harness           │
                 │  evasion / round, cost,        │
                 │  oscillation index, queries    │
                 └─────────────────────────────┘
```

## 5. Experimental Design

**Independent variables (factorial grid):**
1. Retraining cadence — {every round, every N rounds, threshold-triggered}
2. Data-selection strategy — {full replay, hard-example mining, bounded buffer}
3. Attacker algorithm — {DQN, PPO} (secondary robustness axis)

**Dependent variables / metrics:**
- Evasion rate per round (primary)
- Rounds-to-convergence (or divergence flag)
- **Oscillation index** — trailing-window std of evasion rate (key novel metric)
- Retraining cost (wall-clock + samples) per round
- Attacker query complexity (queries to find an evasive variant)

**Baselines:** frozen classifier (lower bound); minimax alternation (reproduce at small scale);
naive "retrain every round on everything" (the practical vendor default).

**Datasets:** EMBER static features for the classifier side (a synthetic EMBER-like feature space
ships for immediate, download-free experimentation); a labelled binary corpus for the PE-mutation
attacker once institutional/legal sign-off is in place.

## 6. Roadmap (≈10 months)

| Phase | Duration | Deliverable |
|---|---|---|
| 0. Proposal & lit consolidation | Weeks 1–3 | Updated novelty; annotated bibliography |
| 1. Setup & baseline reproduction | Weeks 4–10 | Classifier baseline; RL attacker vs. frozen classifier |
| 2. Orchestrator + policies | Weeks 11–18 | Co-evolution loop; oscillation-index metric |
| 3. Full experimental matrix | Weeks 19–28 | All policy combinations run |
| 4. Analysis & baseline comparison | Weeks 29–34 | Statistical comparison across policies |
| 5. Writing | Weeks 35–40 | Thesis draft + paper extract |
| 6. Buffer / defense prep | Weeks 41–44 | Revisions, defense slides |

**Critical-path risk:** Phase 3 is compute-heavy — start early, treat grid size as adjustable
(a reduced 2×2 first, expand as compute permits).

## 7. Tools, Datasets & Compute

- Python, PyTorch, scikit-learn (LightGBM optional for exact EMBER model)
- EMBER dataset (public, feature-vector based)
- gym-malware / MAB-malware environment (extend rather than rebuild)
- Malware corpus with institutional/legal sign-off, VM-isolated, for the mutation environment
- GPU access for RL training rounds (Phase 3 bottleneck)

## 8. Target Venues

IEEE ICDMW, IEEE S&P Workshops (SPW), IEEE TrustCom, IEEE Access, IEEE TIFS; also RAID, ACM ASIACCS.

## 9. Risk Register

| Risk | Mitigation |
|---|---|
| Novelty flagged as covered | Position as controlled-comparison layer; cite 2022–2026 cluster |
| Factorial grid too expensive | Reduced 2×2 first, expand if time/compute allow |
| Malware corpus access delays | Confirm access early; EMBER unblocks defender side in parallel |
| RL training instability | Budget tuning time; PPO fallback if DQN underperforms |

## 10. Selected References

[5] M. R. Ebrahimi et al., "An Adversarial Reinforcement Learning Framework for Robust ML-based
Malware Detection," *IEEE ICDMW*, pp. 567–576, 2022.
[6] "Adversarial Co-Evolution of Malware and Detection Models: A Bilevel Optimization Perspective,"
arXiv:2604.22569, 2026.
[7] "How to Train your Antivirus: RL-based Hardening through the Problem Space," *RAID*.
[8] "DRMD: Deep Reinforcement Learning for Malware Detection under Concept Drift,"
arXiv:2508.18839, 2026.

*(Verify full titles/authors and pull DOIs against publisher pages before formal submission.)*

---

## How the code maps to this roadmap

| Roadmap element | Where it lives |
|---|---|
| Co-Evolution Orchestrator | `src/coevomal/orchestrator.py` |
| Attacker Module (DQN / PPO / random) | `src/coevomal/attackers/` |
| Defender Module (GBDT / MLP) | `src/coevomal/defenders/` |
| Retraining policies (cadence × data-selection × mode) | `src/coevomal/policies/retraining.py` |
| Evaluation harness (evasion, oscillation index, cost, queries) | `src/coevomal/evaluation/metrics.py` |
| PE-mutation environment (feature-space proxy) | `src/coevomal/environment/malware_env.py` |
| Factorial study (Phase 3) | `experiments/run_factorial.py` |
| EMBER extension point | `src/coevomal/environment/dataset.py::load_ember` |
