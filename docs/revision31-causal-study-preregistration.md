# Revision 31 causal study preregistration

## Decision

Revision 31 will run a 2×2 factorial study with five matched seeds per cell. Branch width is static within each run (`K=1` or `K=4`). Complexity remains dynamic in every run, but its level schedule is either outcome-adaptive or outcome-blinded and pre-scheduled.

The study graduates to a larger model only if the `K=4` adaptive treatment learns internally, transfers to a sealed 120-task external pack, and beats `K=1` across matched seeds.

The machine-readable protocol is
[`research/studies/revision31-causal-study.json`](../research/studies/revision31-causal-study.json).

## Why this study

Revision 30 did not establish meaningful post-training. Its external evaluation produced small, inconsistent gains, and its trajectories exposed two tool-contract failures:

- the model copied the illustrative path `relative/path.py` as if it existed;
- after an exact edit was rejected, the model repeated stale edits without enough current file state to recover.

Revision 31 changes only this action interface. It does not change the reward, optimizer, policy-credit rule, task templates, branch restoration, or hidden verifier. That makes the experiment a test of whether a usable tool contract unlocks the existing branch-aware learner, rather than another simultaneous algorithm rewrite.

## Factorial design

The four cells are:

| Cell | Branch width | Complexity schedule |
| --- | ---: | --- |
| `k1_scheduled_dynamic` | 1 | Promote after every two completed validation windows |
| `k4_scheduled_dynamic` | 4 | Promote after every two completed validation windows |
| `k1_adaptive` | 1 | Promote after two retained zero-regression mastery windows |
| `k4_adaptive` | 4 | Promote after two retained zero-regression mastery windows |

The scheduled treatment is still dynamically complex: it traverses the same ordered complexity levels and uses the nearest harder level as its probe. It is outcome-blinded, so the comparison separates static branch width from outcome-driven complexity adaptation.

Each cell uses seeds 137, 269, 443, 617, and 887. Conditions sharing a seed also share the validation and test split bases. Every condition receives exactly 320 sampled completions. `K=4` collects four tasks per update and `K=1` collects sixteen so that both attempt sixteen new continuations per update before replay.

The primary estimands are paired within seed:

1. the branching main effect, `K=4 − K=1`, averaged across curriculum policies;
2. the curriculum main effect, adaptive minus scheduled dynamic, averaged across branch widths;
3. their interaction;
4. each trained adapter’s change from the frozen base policy.

These are estimands, not claims of complete causal identification. Model sampling and validation-directed task targeting remain stochastic mediators. Matched seeds, split bases, completion budgets, and the outcome-blinded schedule remove the largest avoidable confounds.

## Tool contract

The revision-31 system prompt leads with the output contract and contains no fake non-empty path. It tells the model to copy every non-empty path from an observed `list`, `search`, or `read` result. A rejected exact edit returns bounded current file content and directs the model to reread before retrying.

The motivating revision-30 failure shapes are fixtures for local conformance tests:

1. an invented `relative/path.py` read is rejected and followed by an observed-path recovery;
2. a stale edit returns current content, and a reread-derived edit succeeds.

The implementation is frozen before the external pack is authored.

## External evaluation

After trainer freeze, a 120-task pack is authored and sealed:

- 40 SQLite and data-repair tasks;
- 40 filesystem and CLI-debugging tasks;
- 40 micro-repository code-repair tasks.

Every completed adapter and the frozen base policy are evaluated. External outcomes remain unopened until all 20 training conditions are terminal. Every seed, adapter, and regression is reported before aggregate statistics.

## Graduation gates

All four required gates must pass:

1. **Operational:** all 20 runs complete final evaluation, persist their adapters, verify teardown, and maintain at least 99% action-protocol validity.
2. **Internal learning:** at least four of five `K=4` adaptive seeds have a positive clean paired gain; the median gain is at least one task; at most two tasks regress across those seeds.
3. **External transfer:** at least four of five `K=4` adaptive adapters improve; their median gain is at least 6 of 120 tasks; at most one seed regresses; the aggregate paired McNemar exact p-value is at most 0.05.
4. **Branching:** at least four of five matched seed effects favor `K=4`, and the median main effect is at least 3 of 120 tasks.

If any gate fails, the study result is `FAIL` and no larger-model run is authorized by this protocol. If every gate passes, the first larger-model experiment is one `K=4` adaptive replication with the same 320-completion budget and a new frozen split.

## Stop and reporting rules

Provider safety ceilings, final-evaluation reserves, and exact completion budgets remain binding. A failed or incomplete condition is visible and does not silently disappear from the denominator. Repeating a condition requires an explicit amendment and preserves the failed attempt.

The final report will contain every cell and seed, the main effects and interaction, external paired transitions, cost, hardware, provider receipt, and teardown evidence.
