# Revision 30 freeze

Revision 30 is frozen at commit
`e6a139375a4e6ec92df362873237b398aa0041c0`. The trainer, repository-repair
environment, dependency lock, model revision, workload revision, objective, and
reference-run artifacts are recorded in
`research/frozen/repository-repair-revision-30.json`.

The frozen source files must remain byte-for-byte unchanged:

- `research/runpod/repository_repair_rl.py`
- `research/runpod/repository_repair_env.py`
- `requirements.txt`

The confirmatory study may add orchestration outside those files. A replication may
override optimization, validation, and test seeds before calling the frozen trainer;
the result must record every override. The no-update control may suppress optimizer
mutation while retaining the same forward, backward, collection, validation, and
evaluation workload. The K=1 ablation must identify its alternative advantage estimator
because leave-one-sibling-out advantage is undefined at width one. Neither control may
be reported as an unmodified revision-30 training run.

The freeze test recomputes each source digest and checks that the same bytes exist in
the recorded Git commit. Study infrastructure must fail closed when the manifest,
worktree, or frozen commit disagree.

External evaluation tasks are intentionally excluded from this freeze. They must be
authored and committed after this boundary so that revision-30 development could not
adapt to their outcomes. Candidate prompts may include task instructions and visible
repository state, but never hidden verifier logic or expected patches.
