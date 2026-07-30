# Larger-model RunPod runbook

The supported campaign is one integrated H100 eligibility screen followed by at most one
screen-authorized training pilot. The screen stages and verifies the workload on the same
H100 that runs the screen. Do not create a separate CPU or GPU staging pod.

The immutable profile is
`qwen2.5-coder-7b-runpod-h100@10`. It pins:

- `Qwen/Qwen2.5-Coder-7B-Instruct` at revision
  `c03e6d358207e414f1eca0bb1891e29f1db0e242`;
- one Secure Cloud `NVIDIA H100 80GB HBM3`;
- the digest-pinned PyTorch 2.8 and CUDA 12.8 image;
- static branching at `K=4`;
- adaptive curriculum complexity with replay;
- eligibility workload `larger-model-eligibility-screen@10`;
- training workload `runpod-repository-repair-large-model-pilot@7`; and
- transactional adapter, optimizer, and policy-lineage retention.

The profile also pins `larger-model-dependencies.lock` by size and SHA-256. During the
authenticated preparation window, the bootstrap installs its 30 non-Torch Python 3.12
packages from hash-verified binary wheels into a private `/tmp` tree. Torch comes only
from the digest-pinned base image. The bootstrap then disables package-network access,
marks the installed tree read-only, re-hashes it, and records the exact private-tree
digest. Bytes under `/workspace/equinox-state/python` and the historical `@6` dependency
directory are not trusted inputs.

Changing any pinned model, source, image, runtime, budget, handoff, or scientific setting
requires a new profile. Never edit a receipt or authorization to cross a profile change.

## Campaign envelope

| Guard | Screen | Pilot |
| --- | ---: | ---: |
| Maximum hourly rate | $4.00 | $3.25 |
| Maximum total cost | $3.35 | $13.00 |
| Provider lifetime | 2,880 seconds | 14,280 seconds |
| Pre-create readiness deadline | 600 seconds | 360 seconds |
| Scientific target | 1,500 seconds | 9,000 seconds |
| Result retrieval reserve | 60 seconds | 180 seconds |
| Teardown reserve | 120 seconds | 120 seconds |
| Workload attempts | 1 | 1 |

The screen lifetime is exactly:

```text
600  create, attest, upload, stage, verify, and activate
1500 scientific target
300  final-evaluation reserve
300  retry reserve
60   result retrieval
120  teardown
----
2880 seconds
```

The cost gate conservatively bills 3,000 seconds. At the maximum $4 hourly rate this is
$3.3333, below the $3.35 screen cap.

The campaign began at a RunPod balance of `$45.2834247943` and has a `$25` authorization.
Immediately before either allocation, the launcher reads the current provider balance
and reserves the full current stage, the still-authorized pilot, and `$0.30` of storage.
If the resulting worst case exceeds `$25`, it stops before create.

No AWS resource is permitted.

## What the eligibility screen proves

The screen does not train the policy. It verifies that this exact model, runtime, volume,
task interface, and branch distribution are suitable for the pilot.

It evaluates eight deterministic level-0 baselines and eight independent `K=4` sibling
groups. Branching begins after the model has observed the repository root. The screen
requires:

- exact action-protocol validity and checkpoint admission;
- mixed solved and failed siblings;
- at least two informative branch groups;
- enough memory for the pilot sequence shape;
- deterministic eager attention with math SDP, strict deterministic algorithms, and
  the frozen cuBLAS workspace configuration;
- a projected final evaluation inside the pilot reserve;
- no final-test access;
- no persistent policy mutation; and
- exact restoration of optimizer state.

An operationally valid screen can return `eligible: false`. That is evidence, not
permission to rerun until a sample passes.

## Before allocation

Use the existing 50 GB RunPod network volume containing the pinned model snapshot.
Export its exact identity:

```bash
export EQUINOX_RUNPOD_NETWORK_VOLUME_ID=euh248b2p5
export EQUINOX_RUNPOD_DATA_CENTER_IDS=EU-FR-1
```

Then verify:

```bash
runpodctl pod list --all
runpodctl network-volume get "$EQUINOX_RUNPOD_NETWORK_VOLUME_ID"
runpodctl user
```

Stop unless all of the following are true:

- no pod exists;
- the configured volume is the sole expected volume in `EU-FR-1`;
- `currentSpendPerHr` is no more than the storage-only ceiling;
- the dashboard is healthy at `http://127.0.0.1:33100/runs`;
- the API is healthy at `http://127.0.0.1:8180/healthz`;
- the shared operator lease is absent; and
- every participating source byte matches immutable Git `HEAD`.

Unrelated untracked files do not enter the bundle. A changed participating file blocks
allocation.

## Run the read-only screen preflight

```bash
./scripts/screen-larger-model --preflight-only
```

Preflight performs no provider mutation. It must verify:

- exact profile, model, source contract, image digest, volume, and H100 inventory;
- the historical `@6` volume receipt only as a narrow preallocation seed;
- the deterministic current-HEAD workload bundle;
- the 2,880-second lifetime and $3.35 cap;
- the 3,000-second conservative cost bound;
- the live dashboard and observer API;
- the campaign ledger; and
- zero active pods immediately before returning.

The historical receipt cannot authorize science or dependency bytes. The paid H100 must
re-hash the complete model snapshot, install the profile-pinned dependency lock into a
private tree, and produce fresh current-profile evidence.

## Run the integrated screen

```bash
./scripts/screen-larger-model
```

The launcher registers the observable execution before allocation. It then starts one
absolute 600-second deadline immediately before the sole create request. That deadline
never resets.

The provider request is restricted to:

- exactly one Secure H100;
- the immutable `tag@sha256` image;
- the exact network volume mounted at `/workspace`;
- `8000/http`;
- SSH disabled; and
- provider auto-termination after exactly 2,880 seconds.

The launcher verifies the returned pod ID, image, GPU count and identity, cloud type,
volume, data center, mount, and hourly rate before sending workload bytes.

### Staging and activation

The pod starts only the authenticated bootstrap. Its creation environment contains a
bearer token and planned identity: profile, Git commit, source contract, bootstrap
digest, bundle digest, bundle size and content-addressed path, and volume identity. It
contains no bundle bytes and no unborn receipt digest.

The host verifies `/bootstrap-health`, then sends the exact XZ bundle once to `/bundle`.
The bootstrap:

1. checks the exact path, bearer token, content type, size, and SHA-256 digest;
2. rejects links, duplicate members, non-regular members, extra files, path traversal,
   trailing compressed data, and excessive expanded size;
3. stages the bytes at the content-addressed volume path without following symlinks;
4. writes and fsyncs the canonical stage receipt;
5. verifies the bundled dependency-lock bytes, installs only hash-matched binary wheels
   during authenticated preparation, closes package-network access, and re-hashes the
   read-only private dependency tree;
6. copies the exact allowlisted code into a separate content-addressed, read-only
   private tree and re-hashes it;
7. re-hashes every pinned model-snapshot file;
8. verifies exact dependency, Torch, CUDA, H100, memory, and bf16 identities; and
9. runs a fresh-process real AdamW persist, fsync, reopen, restore, and advance
   checkpoint probe on the mounted network volume.

The host retrieves and independently verifies the stage receipt, volume-readiness
receipt, and Torch evidence. Only then does it send the canonical activation object to
`/activate-staged-bundle`. The activation contract is
`authenticated-proxy-stage-activation@2`.

The first structured progress must echo the exact commit, source, bootstrap, bundle,
stage, dependency-lock, private dependency tree, private code tree, volume, Torch,
activation, and provider-volume identities. A missing or changed field stops the run
before its scientific output is accepted. The same evidence is revalidated on every
restart.

If the upload or activation response is interrupted, the launcher reconciles only through
authenticated exact health or progress. It never blindly uploads a second bundle.

## Durable screen evidence

The launcher retrieves the screen result before teardown, deletes the exact pod, and
requires three successful zero-pod, storage-only provider polls.

Only after confirmed teardown does it publish this six-artifact set:

- current-profile volume-readiness receipt;
- current-profile bundle-stage receipt;
- Torch retention evidence;
- scientific screen result;
- provider receipt; and
- campaign attempt record.

The files are written to an immutable hidden generation. One fsynced
`runpod-proof-….artifact-set.json` commit manifest is the visibility boundary. Consumers
must resolve artifacts through that manifest. A directory glob or a partial generation
is not proof.

The observer may report `SUCCEEDED` only with:

```json
{
  "artifact_set_committed": true,
  "artifact_set_manifest_digest": "sha256:…"
}
```

The prior `@6` receipts are preserved in immutable profile-qualified archives. They are
not overwritten or promoted to current evidence.

## Verify pilot authorization

```bash
./scripts/run-larger-model-pilot --preflight-only
```

The wrapper finds a committed current-profile artifact set, verifies every artifact
digest, and passes only its resolved paths to the authorization gate. Authorization
requires:

- `eligible: true`;
- exact screen counts and every required gate;
- no training, persistent mutation, or test-split access;
- the exact current HEAD, bundle, bootstrap, model, volume, and runtime;
- the nested readiness and Torch evidence;
- the exact activation digest;
- the committed provider receipt; and
- confirmed teardown.

The pilot rebuilds the canonical bundle from current HEAD. Any source or operational
identity change requires another explicitly authorized screen.

Screen evidence authorizes a pilot; it does not substitute for pilot preparation. The
pilot allocation must independently install the same hash lock into a new private tree,
materialize code into a new private tree, re-attest the mounted snapshot and runtime,
and run a fresh retention checkpoint probe. The pilot stops before model loading if any
new digest differs from the authorization or current profile.

The retention probe proves the checkpoint mechanism, not a reusable checkpoint
identity. Each allocation receives a fresh, unlogged 32-byte HMAC key and run identity.
Every adapter file, safetensors file, and training-state file is bound into a canonical
checkpoint manifest. Resume accepts only the runner-anchored generation and manifest
digest, reopens files through no-follow descriptors, copies the authenticated bytes to
a private directory, and loads training state with `weights_only=True`. Legacy,
unbound, symlinked, hard-linked, replaced, or replayed checkpoints fail closed. The
readiness proof and pilot therefore share the stable authentication mechanism revision,
not an ephemeral HMAC or per-run manifest digest.

## Run the training pilot

```bash
./scripts/run-larger-model-pilot
```

The pilot runs static `K=4` branching with adaptive complexity and replay. Positive policy
credit is restricted to verified successful sibling actions and the immediately upstream
fresh reads that enabled the fixing edit.

The paid worker must start Python with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; setting it
after CUDA initialization is not valid. The pilot then requires eager attention,
disables flash and memory-efficient SDP, enables math SDP, disables TF32 and cuDNN
benchmarking, and enables deterministic algorithms with `warn_only=False`. Model
loading and the final result both record this exact contract. Any missing setting or
unavailable deterministic kernel stops the run instead of weakening reproducibility.

Each candidate update is transactional. A regression restores the retained adapter,
optimizer, effective policy lineage, pending examples, and retained validation
observation. Four consecutive rejected regression windows stop the run. Mastery can
advance only from a retained zero-regression improvement.

Before the first retained promotion, collection allocates three of every four new task
groups to the active frontier and one to the nearest probe. After that promotion it
returns to the `2:2` adaptive mix. The transition is recorded in progress and result
evidence; static `K=4`, the Wilson promotion gate, and the sealed final evaluation do
not change.

The proof must distinguish:

- attempted updates;
- effective optimizer updates;
- retained policy updates;
- rolled-back updates;
- provisional versus committed validation outcomes; and
- the exact branch groups and actions used by each update.

A completed pilot reports one canonical outcome:

- `MEANINGFUL_POST_TRAINING` when the sealed hypothesis passes;
- `NEGATIVE_EXPERIMENT_COMPLETED` when the run is complete and probative but the
  hypothesis fails; or
- `INCONCLUSIVE_EXPERIMENT_COMPLETED` when the run cannot support either conclusion.

`meaningful_post_training` is the authoritative boolean. A complete, probative negative
result is valid experimental evidence, but it must use claim strength `NEGATIVE_RESULT`.
Only a passing hypothesis may use `EXPLORATORY_SINGLE_SEED`. Dynamic complexity is
reported separately and counts as progressed only after at least one retained
promotion.

## Hard stop conditions

Every allocated-state failure must delete the exact pod and reconcile provider state.
Stop for any of the following:

- unknown or changed provider state;
- an image, GPU, volume, mount, data-center, or rate mismatch;
- expiry of the absolute readiness deadline;
- insufficient remaining provider lifetime;
- a dependency lock, private-tree, bundle, receipt, Torch, snapshot, activation, or
  first-progress mismatch;
- an uncommitted artifact set;
- model-load or no-progress timeout;
- projected or accrued cost above the cap;
- CUDA out of memory;
- a changed campaign ledger;
- unresolved teardown; or
- an existing shared operator lease.

Do not retry an unchanged deterministic failure or ambiguous create. Unknown provider
state is not proof of absence.

## Recovery and teardown

The shared lease is:

```text
~/.local/state/equinox/runpod/operator.lock/lease.json
```

Never delete it merely to force another launch. Reconcile the recorded pod name and ID,
confirm three successful zero-pod and storage-only polls, and recover any artifact
publication journal before releasing it.

### Republish a committed result after an observer outage

Use this path only when an artifact set committed after confirmed teardown but the
screen's terminal execution PUT or the pilot's proof POST failed or lost its response.
It does not invoke RunPod, allocate a GPU, consume pilot authorization, or create an
operator lease.

The local account and its proof directory are the replay trust boundary. The command
rejects foreign-owned files or directories and any group- or world-writable proof,
generation, or artifact path. Do not copy a set through a shared writable directory.

Verify and reconstruct the exact request without network access:

```bash
./scripts/republish-runpod-proof \
  --proof-directory "$PWD/var/research-proofs" \
  --publication-id runpod-proof-… \
  --dry-run
```

Then replay the proof to the local API:

```bash
EQUINOX_API_ROOT=http://127.0.0.1:8180 \
EQUINOX_INTERNAL_TOKEN="$EQUINOX_INTERNAL_TOKEN" \
./scripts/republish-runpod-proof \
  --proof-directory "$PWD/var/research-proofs" \
  --publication-id runpod-proof-…
```

Plain HTTP is accepted only for numeric loopback addresses. Use HTTPS for every hostname
or remote API root; the command refuses to send the internal token over non-loopback
HTTP.

For a screen set, the command reconstructs the terminal execution update from the
committed receipt and the already-registered execution, PUTs the exact typed envelope,
and verifies `SUCCEEDED`. For a pilot set, it re-verifies the model artifact and committed
source screen, POSTs the exact proof, and verifies both the execution and proof records.
Repeating either operation is safe; a prior pilot ingestion returns
`already_recorded: true`.

Stop instead of replaying when the set is partial or tampered, the execution is missing,
the source screen is not committed remotely, the provider or workload lineage differs,
or a failed execution has a non-recoverable error. Screen recovery accepts only the
specific post-teardown terminal-publication failure; it does not reopen other failed
experiments.

After the pilot completes, or after the campaign is explicitly abandoned, retain and
verify the committed proof set, then delete the network volume:

```bash
runpodctl network-volume delete "$EQUINOX_RUNPOD_NETWORK_VOLUME_ID"
runpodctl network-volume list
runpodctl user
```

The volume remains billable after GPU teardown.

## Observe the run

Open `http://127.0.0.1:33100/runs`.

The Runs index stays concise: name, status, learning progress, GPU, estimated cost, and
relative update time. The detail page shows the real lifecycle:

```text
Prepare → Stage bundle → Verify → Screen/Run → Teardown → Publish
```

Use Trajectory to inspect the shared prefix, exact checkpoint, four sibling branches,
every action and observation, verifier result, sibling-relative signal, optimizer input,
retention decision, and dynamic-complexity transition. Use Evidence for immutable
identities and receipts. Do not infer learning from GPU utilization or a successful
teardown alone.
