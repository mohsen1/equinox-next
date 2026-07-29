# Independent-prefix K=4 comparison

## Decision

The later control for the active shared-prefix learner is an opt-in, static-`K=4`
independent-prefix group-relative run. Revision 31’s `K=1` cell remains a branch-width
ablation; it is not this control.

The implementation lives in
`research/runpod/repository_repair_independent_prefix_grpo.py`. Importing the module does
not change the active larger-model profile. No RunPod launcher or immutable workload
manifest selects it. Executing that module directly installs the process-local comparison
hooks, captures the trainer result, replaces shared-prefix-only result fields, validates
the independent topology and accounting, and only then emits the result.
It runs whichever byte-verified base configuration is already installed in that process;
it is not the future frozen 7B matched profile or a provider authorization.

Proofs, progress events, branch snapshots, and the dashboard use one condition object:

```json
{
  "schema_version": 1,
  "condition_id": "independent_prefix_grpo_k4",
  "short_label": "Independent · K=4",
  "prefix_topology": "independent",
  "branch_width": 4,
  "group_credit": "sibling_relative",
  "shared_prefix": false
}
```

Consumers must use these typed fields instead of inferring the condition from a run name
or workload prose.

## Comparison semantics

Each group uses one task and four trajectories. All four trajectories restore the same
initial task state, then sample every model-generated action independently. There is no
model-generated shared prefix and no decision checkpoint.

The optimizer uses:

- the same static group width, model, adapter initialization, reward, verifier, task
  generator, sampling temperature, top-p, KL coefficient, learning rate, curriculum,
  replay policy, validation schedule, and test split as the shared-prefix source;
- leave-one-out, standardized sibling-group advantages over the four terminal returns;
- signed credit on every sampled action, including malformed or rejected actions; and
- per-trajectory normalization, so the action weights for one trajectory sum to that
  trajectory’s advantage.

This is a single-pass GRPO update. The sampled policy and update policy are identical, so
the initial policy ratio is one and a clipping branch cannot activate. The existing
disabled-adapter KL term remains active for every sampled action. A later profile must
stay single-pass or add and declare old-policy ratio and clipping state.

## Matched-budget admission

A shared-prefix result creates the comparison contract. An independent-prefix result is
admissible only when all of these checks pass:

1. Both runs use static `K=4`.
2. Model and tokenizer-facing interface identities, optimization seed, reward, verifier,
   sampling settings, optimizer controls, curriculum, replay, validation, test split,
   maximum updates, and runtime ceiling match.
3. Complete ordered branch evidence covers every counted task group. A run with
   discarded groups that lack branch evidence is ineligible.
4. Realized task groups, excluded groups, completed groups, sampled trajectories,
   terminal completions, total sampled actions, and total sampled completion tokens match
   exactly.
5. The canonical task/seed schedule digest matches. Its material includes branch order,
   update and collection index, task ID, complexity level, replay and curriculum roles,
   and all four sibling sampling seeds.
6. The result declares independent prefixes, no restored decision checkpoint, full
   trajectory gradient credit, and the comparison profile identity.

`total_sampled_completion_tokens` must include every training action, including the
shared-prefix tokens in the source run. Every source snapshot records its shared-prefix
completion tokens, each sibling records its post-prefix completion tokens, and
`sampled_completion_tokens` must equal their exact sum. An independent snapshot has no
shared-prefix component, so its four sibling totals must equal
`sampled_completion_tokens` exactly. The snapshot totals must then equal the run total
exactly; `>=` evidence is never accepted. The contract refuses to build when any token
field, full branch coverage, or exact reconciliation is unavailable.

The installed comparison wraps the generic branch-evidence token counter only for
snapshots carrying the explicit independent topology marker. Those snapshots require
`shared_prefix: null`, a matched initial state, four siblings, and exact sibling-token
reconciliation. All other snapshots are delegated to the unchanged strict shared-prefix
counter. The same hook is used by checkpoint persistence, resume validation, and final
result reconciliation.

Post-prefix actions and sibling-only tokens remain topology-specific disclosure counters;
they are recorded but are not required to equal each other. The contract rejects a near
match instead of silently treating it as evidence. Cost, elapsed time, malformed actions,
and exact solve outcomes remain reportable outcomes beyond the shared runtime ceiling.

## Required work before a paid comparison

The current module provides collection, credit, serialization, checkpoint-safe token
accounting, a validated result entrypoint, and matched-budget admission checks. Before
spending GPU time:

1. freeze a new comparison profile around the completed shared-prefix source result;
2. add an operator that records total training completion tokens and stops at the exact
   realized action, token, and completion budgets without skipping retention validation
   or final evaluation;
3. include the module in a new immutable bundle and source contract;
4. run local self-tests and a cheap capacity screen; and
5. authorize a separate bounded provider run.

Do not reuse the active shared-prefix pilot authorization or mutate its manifest. A
comparison that misses any realized budget or the task/seed schedule is exploratory and
cannot support the matched-budget claim.
