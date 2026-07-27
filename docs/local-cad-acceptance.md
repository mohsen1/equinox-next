# Local CAD acceptance scenario

The release passes when `scripts/acceptance` completes from a clean checkout with no
cloud credentials.

## Scenario

1. Start PostgreSQL, MinIO, the orchestrator, execution service, fixture worker, and
   dashboard with Docker Compose.
2. Seed one bounded technical-drawing task: a mounting plate with a central bore, four
   corner holes, and a raised boss.
3. Launch an independent-rollout baseline and a width-four branch-aware BPO-local run
   through typed API requests. Both manifests explicitly name
   `LocalFixtureComputeProvider` and `DeterministicJudgeFixture`.
4. Execute a shared three-action CAD prefix. Accept each action exactly once and record a
   semantic state, isolated runtime cursor, source and candidate render, geometry report,
   proof bundle, verification DAG, deterministic pointwise result, metric observations, and named
   rewards.
5. At the declared boundary, capture a logical environment snapshot and a policy-specific
   decision checkpoint. Restore four child cursors with distinct runtime identities,
   fencing tokens, deterministic RNG derivations, copied task horizon, and reserved
   budgets.
6. Complete four sibling continuations. Member 2 retries one infrastructure failure.
   Member 3 produces a valid negative constraint result. Member 4 produces an abstained
   or disputed judge result, which creates no judge-derived zero.
7. Compare sibling proof bundles in blinded deterministic order. Store the presentation
   permutation and a group result with preferences, a tie, confidence, and admitted
   sibling-preference metrics where valid.
8. Materialize the exact admitted trees, proof bundles, verification runs, judge results,
   reward signals, and eligibility decisions. Commit one policy successor with
   compare-and-swap.
9. Deliver duplicate and out-of-order command/results. Assert that transitions, accepted
   verifier nodes, judge results, metrics, rewards, jobs, and optimizer commits remain
   unique.
10. Rejudge an existing proof bundle with `cad.pointwise.mock@2`. Assert that candidate
    execution, geometry, and rendering counts do not change and that the original result
    used by the committed iteration remains immutable.
11. Restart the orchestrator and execution service. Reconcile accepted work and resume an
    incomplete verification graph without duplicating completed nodes.
12. Cancel an active run. Assert live allocations, jobs, sessions, cursors, leases, and
    ephemeral snapshots are released while scientific metadata remains queryable.
13. Reproduce a completed run from its normalized immutable manifest and retain the
    source-run link.
14. Open the run, iteration, tree, proof, verification, judge, and sibling-comparison
    routes. Assert live API data, URL-addressable selection, keyboard outline views, and
    light/dark accessibility.

## Release invariants

- Local provider registries contain exactly `LocalFixtureComputeProvider` and
  `DeterministicJudgeFixture`; tests reject configuration containing real-provider names or
  endpoints.
- Infrastructure, verifier, judge-provider, abstention, disagreement, malformed output,
  and integrity failures never become candidate reward zero.
- Artifact digests are verified before scientific acceptance.
- Snapshot fidelity is labeled `logical_restore`; no process-memory or microVM claim is
  made.
- No hidden task reference, rubric, calibration case, provider credential, or Docker
  control capability enters a candidate or policy runtime.
