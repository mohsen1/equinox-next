---
version: 1
slug: "services-dashboard-src-app-tsx"
primary_target: "services/dashboard/src/App.tsx"
related_targets: ["services/dashboard/src/styles.css","services/dashboard/src/pages/rollout.tsx","services/dashboard/src/pages/workspace.tsx"]
---

Scope: the Equinox Next web dashboard, especially run, rollout-tree, and verification
inspection routes. Mode: Operate.

Audience: RL researchers and research operators. Job: observe a run causally, decide
whether collected evidence is valid, and trace one committed policy update back to every
action and exact lineage. Primary actions: launch, follow, select, compare, inspect,
cancel, reproduce, and rejudge. Proof is persisted API data: states, transitions,
attempts, renders, deterministic reports, model assessments, named rewards, and immutable
manifests.

Direction: The distilled Verification Drafting Table. The entry experience is a short
experiment queue, not a telemetry wall. Launch is environment-first. Completed runs expose
a first-class Trace tab before evidence and operations: run → trajectory → shared prefix →
checkpoint → sibling lane → selected action. ReactFlow is the spatial canvas; the selected
action inspector is the reading surface. Evidence and provenance stay synchronized to URL
selection instead of becoming distant tables.

Memorable moment: the researcher opens Trace, sees the shared path split explicitly at a
snapshot checkpoint, chooses one of four sibling lanes, then steps backward and forward
through actions while outcome, verification, proof, and provenance update beside the graph.

Constraints: flat/no decorative shadows, one product accent, automatic light/dark, WCAG
2.2 AA, keyboard outline equivalents, URL-addressable selection, and no hidden evidence
payloads. Branch execution status, verifier result, and training eligibility must never be
collapsed into one ambiguous state. Mobile defaults to Outline; Graph remains an intentional
pan/zoom mode.
