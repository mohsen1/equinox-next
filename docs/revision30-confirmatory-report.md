# Revision 30 confirmatory study

Status: **FAIL**

No seed or regression is averaged away. Every paid execution appears below.

Frozen workload: `runpod-repository-repair-causal-credit@30`

Frozen source: `e6a139375a4e6ec92df362873237b398aa0041c0`

## Decisive tests

| Test | Status |
|---|---:|
| k4 training beats frozen policy k4 | FAIL |
| k4 training beats matched k1 | FAIL |
| gains repeat across fresh seeds | FAIL |
| all retained adapters evaluated externally | PASS |
| fresh k4 gains transfer without regressions | FAIL |
| regression guard remains clean | PASS |

## Causal matching caveats

- `k4_training_beats_frozen_policy_k4`: 17 of 20 task positions matched; 3 differed. Adaptive task routing is policy-mediated and therefore part of the treatment. Matching the generator inputs does not guarantee an identical realized training schedule.
- `k4_training_beats_matched_k1`: 5 of 28 task positions matched; 23 differed. Adaptive task routing is policy-mediated and therefore part of the treatment. Matching the generator inputs does not guarantee an identical realized training schedule.

## Optimization conditions

| Condition | Status | Seed | K | Updates | Completions | Complexity | Initial | Final | Gain | + | − |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| k4_train_seed113 | SUCCEEDED | 113 | 4 | 15 | 440 | L1 / sampled L2 | 0.3125 | 0.5 | 0.1875 | 9 | 0 |
| k4_train_seed211 | FAILED | 211 | 4 | — | — | — | — | — | — | — | — |
| k4_train_seed307 | SUCCEEDED | 307 | 4 | 8 | 156 | L0 / sampled L2 | 0.291667 | 0.291667 | 0 | 0 | 0 |
| k4_train_seed701 | SUCCEEDED | 701 | 4 | 16 | 416 | L0 / sampled L2 | 0.291666 | 0.458333 | 0.166667 | 8 | 0 |
| k4_no_update_seed307 | SUCCEEDED | 307 | 4 | 0 | 156 | L0 / sampled L1 | 0.291667 | 0.291667 | 0 | 0 | 0 |
| k1_train_seed307 | SUCCEEDED | 307 | 1 | 14 | 156 | L0 / sampled L2 | 0.291667 | 0.541667 | 0.25 | 13 | 1 |

## External evaluation

Pack: `revision30-post-freeze-external-pack@1`

| Policy | Seed | Exact | Micro repo | SQLite | Filesystem / CLI | + | − |
|---|---:|---:|---:|---:|---:|---:|---:|
| disabled_adapter_base | — | 1 / 9 | 0 | 1 | 0 | — | — |
| k4_train_seed113 | 113 | 2 / 9 | 1 | 1 | 0 | 1 | 0 |
| k4_train_seed307 | 307 | 1 / 9 | 0 | 1 | 0 | 0 | 0 |
| k4_train_seed701 | 701 | 2 / 9 | 1 | 1 | 0 | 1 | 0 |
| k4_no_update_seed307 | 307 | 1 / 9 | 0 | 1 | 0 | 0 | 0 |
| k1_train_seed307 | 307 | 0 / 9 | 0 | 0 | 0 | 0 | 1 |

### External task transitions

| Adapter | Task | Domain | Base | Adapter | Transition |
|---|---|---|---:|---:|---|
| k4_train_seed113 | external-micro-label-normalization | micro_repository | fail | fail | unchanged |
| k4_train_seed113 | external-micro-retry-delay | micro_repository | fail | pass | improved |
| k4_train_seed113 | external-micro-route-prefix | micro_repository | fail | fail | unchanged |
| k4_train_seed113 | external-sqlite-email-normalization | sqlite_data_repair | fail | fail | unchanged |
| k4_train_seed113 | external-sqlite-stock-floor | sqlite_data_repair | fail | fail | unchanged |
| k4_train_seed113 | external-sqlite-job-completion | sqlite_data_repair | pass | pass | unchanged |
| k4_train_seed113 | external-cli-rename-suffix | filesystem_cli | fail | fail | unchanged |
| k4_train_seed113 | external-cli-numeric-sort | filesystem_cli | fail | fail | unchanged |
| k4_train_seed113 | external-cli-literal-grep | filesystem_cli | fail | fail | unchanged |
| k4_train_seed307 | external-micro-label-normalization | micro_repository | fail | fail | unchanged |
| k4_train_seed307 | external-micro-retry-delay | micro_repository | fail | fail | unchanged |
| k4_train_seed307 | external-micro-route-prefix | micro_repository | fail | fail | unchanged |
| k4_train_seed307 | external-sqlite-email-normalization | sqlite_data_repair | fail | fail | unchanged |
| k4_train_seed307 | external-sqlite-stock-floor | sqlite_data_repair | fail | fail | unchanged |
| k4_train_seed307 | external-sqlite-job-completion | sqlite_data_repair | pass | pass | unchanged |
| k4_train_seed307 | external-cli-rename-suffix | filesystem_cli | fail | fail | unchanged |
| k4_train_seed307 | external-cli-numeric-sort | filesystem_cli | fail | fail | unchanged |
| k4_train_seed307 | external-cli-literal-grep | filesystem_cli | fail | fail | unchanged |
| k4_train_seed701 | external-micro-label-normalization | micro_repository | fail | fail | unchanged |
| k4_train_seed701 | external-micro-retry-delay | micro_repository | fail | pass | improved |
| k4_train_seed701 | external-micro-route-prefix | micro_repository | fail | fail | unchanged |
| k4_train_seed701 | external-sqlite-email-normalization | sqlite_data_repair | fail | fail | unchanged |
| k4_train_seed701 | external-sqlite-stock-floor | sqlite_data_repair | fail | fail | unchanged |
| k4_train_seed701 | external-sqlite-job-completion | sqlite_data_repair | pass | pass | unchanged |
| k4_train_seed701 | external-cli-rename-suffix | filesystem_cli | fail | fail | unchanged |
| k4_train_seed701 | external-cli-numeric-sort | filesystem_cli | fail | fail | unchanged |
| k4_train_seed701 | external-cli-literal-grep | filesystem_cli | fail | fail | unchanged |
| k4_no_update_seed307 | external-micro-label-normalization | micro_repository | fail | fail | unchanged |
| k4_no_update_seed307 | external-micro-retry-delay | micro_repository | fail | fail | unchanged |
| k4_no_update_seed307 | external-micro-route-prefix | micro_repository | fail | fail | unchanged |
| k4_no_update_seed307 | external-sqlite-email-normalization | sqlite_data_repair | fail | fail | unchanged |
| k4_no_update_seed307 | external-sqlite-stock-floor | sqlite_data_repair | fail | fail | unchanged |
| k4_no_update_seed307 | external-sqlite-job-completion | sqlite_data_repair | pass | pass | unchanged |
| k4_no_update_seed307 | external-cli-rename-suffix | filesystem_cli | fail | fail | unchanged |
| k4_no_update_seed307 | external-cli-numeric-sort | filesystem_cli | fail | fail | unchanged |
| k4_no_update_seed307 | external-cli-literal-grep | filesystem_cli | fail | fail | unchanged |
| k1_train_seed307 | external-micro-label-normalization | micro_repository | fail | fail | unchanged |
| k1_train_seed307 | external-micro-retry-delay | micro_repository | fail | fail | unchanged |
| k1_train_seed307 | external-micro-route-prefix | micro_repository | fail | fail | unchanged |
| k1_train_seed307 | external-sqlite-email-normalization | sqlite_data_repair | fail | fail | unchanged |
| k1_train_seed307 | external-sqlite-stock-floor | sqlite_data_repair | fail | fail | unchanged |
| k1_train_seed307 | external-sqlite-job-completion | sqlite_data_repair | pass | fail | regressed |
| k1_train_seed307 | external-cli-rename-suffix | filesystem_cli | fail | fail | unchanged |
| k1_train_seed307 | external-cli-numeric-sort | filesystem_cli | fail | fail | unchanged |
| k1_train_seed307 | external-cli-literal-grep | filesystem_cli | fail | fail | unchanged |

## Every execution

| Execution | Condition | Outcome | GPU | Cost (estimated USD) | Teardown |
|---|---|---|---|---:|---:|
| runpod-proof-20260728T142404Z | k4_train_seed113 | SUCCEEDED | NVIDIA RTX PRO 4500 Blackwell | 1.137499 | yes |
| runpod-proof-20260728T173307Z | k4_train_seed211 | FAILED | NVIDIA A40 | 0.0159 | yes |
| runpod-proof-20260728T173849Z | k4_train_seed211 | FAILED | NVIDIA A40 | 0.018659 | yes |
| runpod-proof-20260728T175652Z | k4_train_seed307 | ELIGIBLE | NVIDIA A40 | 0.018411 | yes |
| runpod-proof-20260728T180041Z | k4_train_seed401 | INELIGIBLE | NVIDIA A40 | 0.023327 | yes |
| runpod-proof-20260728T180557Z | k4_train_seed503 | INELIGIBLE | NVIDIA A40 | 0.02143 | yes |
| runpod-proof-20260728T180933Z | protocol_screen_seed601 | FAILED | NVIDIA A40 | — | yes |
| runpod-proof-20260728T181027Z | k4_train_seed601 | INELIGIBLE | NVIDIA RTX PRO 4500 Blackwell | 0.03823 | yes |
| runpod-proof-20260728T181623Z | k4_train_seed701 | ELIGIBLE | NVIDIA RTX PRO 4500 Blackwell | 0.065337 | yes |
| runpod-proof-20260728T182638Z | k4_train_seed307 | FAILED | NVIDIA RTX PRO 4500 Blackwell | 0.039191 | yes |
| runpod-proof-20260728T184042Z | k4_train_seed701 | SUCCEEDED | NVIDIA RTX PRO 4500 Blackwell | 1.133015 | yes |
| runpod-proof-20260728T201339Z | k4_train_seed307 | SUCCEEDED | NVIDIA A40 | 0.321196 | yes |
| runpod-proof-20260728T205904Z | k4_no_update_seed307 | SUCCEEDED | NVIDIA A40 | 0.281729 | yes |
| runpod-proof-20260728T214307Z | k1_train_seed307 | SUCCEEDED | NVIDIA A40 | 0.378751 | yes |
| runpod-proof-20260728T223646Z | revision30-external-20260728t223641z | FAILED | NVIDIA RTX PRO 4500 Blackwell | 0.005602 | yes |
| runpod-proof-20260728T223932Z | revision30-external-20260728t223927z | FAILED | NVIDIA RTX PRO 4500 Blackwell | 0.004722 | yes |
| runpod-proof-20260728T224155Z | revision30-external-20260728t224150z | FAILED | NVIDIA A40 | 0.003626 | yes |
| runpod-proof-20260728T224553Z | revision30-external-20260728t224547z | FAILED | NVIDIA A40 | 0 | yes |
| runpod-proof-20260728T224755Z | revision30-external-20260728t224749z | FAILED | NVIDIA A40 | 0 | yes |
| runpod-proof-20260728T225016Z | revision30-external-20260728t225011z | FAILED | NVIDIA A40 | 0.022404 | yes |
| runpod-proof-20260728T230039Z | revision30-external-20260728t230033z | FAILED | NVIDIA A40 | — | yes |
| runpod-proof-20260728T230057Z | revision30-external-20260728t230051z | SUCCEEDED | NVIDIA RTX PRO 4500 Blackwell | 0.236615 | yes |

## Every operator attempt

| Condition | Stage | Outcome | Started | Finished |
|---|---|---|---|---|
| k4_train_seed211 | study_condition_preflight | passed | 2026-07-28T17:30:16Z | 2026-07-28T17:30:18Z |
| k4_train_seed211 | study_condition_run | failed | 2026-07-28T17:33:06Z | 2026-07-28T17:35:50Z |
| k4_train_seed211 | study_condition_run | failed | 2026-07-28T17:38:48Z | 2026-07-28T17:41:42Z |
| protocol_screen_seed307 | eligibility_screen_preflight | passed | 2026-07-28T17:55:10Z | 2026-07-28T17:55:12Z |
| protocol_screen_seed307 | eligibility_screen_run | eligible | 2026-07-28T17:56:50Z | 2026-07-28T17:59:48Z |
| protocol_screen_seed401 | eligibility_screen_preflight | passed | 2026-07-28T18:00:29Z | 2026-07-28T18:00:31Z |
| protocol_screen_seed401 | eligibility_screen_run | ineligible | 2026-07-28T18:00:39Z | 2026-07-28T18:04:14Z |
| protocol_screen_seed503 | eligibility_screen_preflight | passed | 2026-07-28T18:05:34Z | 2026-07-28T18:05:35Z |
| protocol_screen_seed503 | eligibility_screen_run | ineligible | 2026-07-28T18:05:56Z | 2026-07-28T18:09:05Z |
| protocol_screen_seed601 | eligibility_screen_preflight | passed | 2026-07-28T18:09:20Z | 2026-07-28T18:09:21Z |
| protocol_screen_seed601 | eligibility_screen_run | failed | 2026-07-28T18:09:32Z | 2026-07-28T18:09:35Z |
| protocol_screen_seed601 | eligibility_screen_preflight | passed | 2026-07-28T18:10:13Z | 2026-07-28T18:10:15Z |
| protocol_screen_seed601 | eligibility_screen_run | ineligible | 2026-07-28T18:10:26Z | 2026-07-28T18:13:45Z |
| protocol_screen_seed701 | eligibility_screen_preflight | passed | 2026-07-28T18:16:10Z | 2026-07-28T18:16:12Z |
| protocol_screen_seed701 | eligibility_screen_run | eligible | 2026-07-28T18:16:22Z | 2026-07-28T18:22:14Z |
| k4_train_seed307 | study_condition_preflight | passed | 2026-07-28T18:26:06Z | 2026-07-28T18:26:07Z |
| k4_train_seed307 | study_condition_run | failed | 2026-07-28T18:26:37Z | 2026-07-28T18:30:12Z |
| k4_train_seed701 | study_condition_preflight | passed | 2026-07-28T18:40:40Z | 2026-07-28T18:40:41Z |
| k4_train_seed701 | study_condition_run | passed | 2026-07-28T18:40:41Z | 2026-07-28T20:12:58Z |
| k4_train_seed307 | study_condition_preflight | passed | 2026-07-28T20:13:33Z | 2026-07-28T20:13:34Z |
| k4_train_seed307 | study_condition_run | passed | 2026-07-28T20:13:37Z | 2026-07-28T20:57:57Z |
| k4_no_update_seed307 | study_condition_preflight | passed | 2026-07-28T20:58:58Z | 2026-07-28T20:58:59Z |
| k4_no_update_seed307 | study_condition_run | passed | 2026-07-28T20:59:03Z | 2026-07-28T21:37:58Z |
| k1_train_seed307 | study_condition_preflight | passed | 2026-07-28T21:39:46Z | 2026-07-28T21:39:47Z |
| k1_train_seed307 | study_condition_run | failed | 2026-07-28T21:39:49Z | 2026-07-28T21:39:54Z |
| k1_train_seed307 | study_condition_run | passed | 2026-07-28T21:43:05Z | 2026-07-28T22:35:15Z |
| revision30-external-20260728t223641z | external_evaluation_run | failed | 2026-07-28T22:36:45Z | 2026-07-28T22:37:14Z |
| revision30-external-20260728t223927z | external_evaluation_run | failed | 2026-07-28T22:39:31Z | 2026-07-28T22:39:55Z |
| revision30-external-20260728t224150z | external_evaluation_run | failed | 2026-07-28T22:41:54Z | 2026-07-28T22:42:25Z |
| revision30-external-20260728t224547z | external_evaluation_run | failed | 2026-07-28T22:45:52Z | 2026-07-28T22:47:38Z |
| revision30-external-20260728t224749z | external_evaluation_run | failed | 2026-07-28T22:47:54Z | 2026-07-28T22:49:28Z |
| revision30-external-20260728t225011z | external_evaluation_run | failed | 2026-07-28T22:50:15Z | 2026-07-28T22:58:14Z |
| revision30-external-20260728t230033z | external_evaluation_run | failed | 2026-07-28T23:00:38Z | 2026-07-28T23:00:41Z |
| revision30-external-20260728t230051z | external_evaluation_run | passed | 2026-07-28T23:00:55Z | 2026-07-28T23:22:25Z |

## Failures

- Provider `runpod-proof-20260728T173307Z` · k4_train_seed211 · The remote workload failed with exit code 1.
- Provider `runpod-proof-20260728T173849Z` · k4_train_seed211 · The remote workload failed with exit code 1.
- Provider `runpod-proof-20260728T180933Z` · protocol_screen_seed601 · RunPod pod creation failed before a provider handle was returned.
- Provider `runpod-proof-20260728T182638Z` · k4_train_seed307 · The remote workload failed with exit code 1.
- Provider `runpod-proof-20260728T223646Z` · revision30-external-20260728t223641z · curl: (56) The requested URL returned error: 404
- Provider `runpod-proof-20260728T223932Z` · revision30-external-20260728t223927z · curl: (56) The requested URL returned error: 404
- Provider `runpod-proof-20260728T224155Z` · revision30-external-20260728t224150z · curl: (56) The requested URL returned error: 404
- Provider `runpod-proof-20260728T224553Z` · revision30-external-20260728t224547z · No RunPod pod or active hourly spend remained.
- Provider `runpod-proof-20260728T224755Z` · revision30-external-20260728t224749z · No RunPod pod or active hourly spend remained.
- Provider `runpod-proof-20260728T225016Z` · revision30-external-20260728t225011z · No RunPod pod or active hourly spend remained.
- Provider `runpod-proof-20260728T230039Z` · revision30-external-20260728t230033z · {"error":"There are no longer any instances available with the requested specifications. Please refresh and try again."}
Usage:
  runpodctl pod create [flags]

Flags:
      --cloud-type string          cloud type (SECURE or COMMUNITY) (default "SECURE")
      --compliance string          comma-separated compliance requirements (e.g., HIPAA,SOC_2_TYPE_2)
      --compute-type string        compute type (GPU or CPU) (default "GPU")
      --container-disk-in-gb int   container disk size in gb (default 20)
      --country-code string        limit pod to a specific country (e.g., US, DE)
      --data-center-ids string     comma-separated list of data center ids
      --docker-args string         docker cmd arguments
      --env string                 environment variables as json object
      --global-networking          enable global networking (secure cloud only)
      --gpu-count int              number of gpus (default 1)
      --gpu-id string              gpu id (from 'runpodctl gpu list')
  -h, --help                       help for create
      --image string               docker image name (required if no template)
      --min-cuda-version string    minimum cuda version (e.g., 12.6)
      --name string                pod name
      --network-volume-id string   network volume id to attach
      --ports string               comma-separated list of ports (e.g., '8888/http,22/tcp')
      --public-ip                  require public ip (community cloud only)
      --registry-auth-id string    container registry auth id (from 'runpodctl registry list')
      --ssh                        enable ssh on the pod (default true)
      --stop-after string          auto-stop datetime (e.g., 2026-04-15T00:00:00Z)
      --template-id string         template id (use 'runpodctl template search' to find templates)
      --terminate-after string     auto-terminate datetime (e.g., 2026-04-15T00:00:00Z)
      --volume-in-gb int           volume size in gb
      --volume-mount-path string   volume mount path (default "/workspace")

Global Flags:
  -o, --output string   output format (json, yaml) (default "json")

{"error":"failed to create pod: There are no longer any instances available with the requested specifications. Please refresh and try again."}
- Operator `k4_train_seed211` at `2026-07-28T17:33:06Z` · exit 1
- Operator `k4_train_seed211` at `2026-07-28T17:38:48Z` · exit 1
- Operator `protocol_screen_seed601` at `2026-07-28T18:09:32Z` · exit 1
- Operator `k4_train_seed307` at `2026-07-28T18:26:37Z` · exit 1
- Operator `k1_train_seed307` at `2026-07-28T21:39:49Z` · exit 1
- Operator `revision30-external-20260728t223641z` at `2026-07-28T22:36:45Z` · exit 1
- Operator `revision30-external-20260728t223927z` at `2026-07-28T22:39:31Z` · exit 1
- Operator `revision30-external-20260728t224150z` at `2026-07-28T22:41:54Z` · exit 1
- Operator `revision30-external-20260728t224547z` at `2026-07-28T22:45:52Z` · exit -15
- Operator `revision30-external-20260728t224749z` at `2026-07-28T22:47:54Z` · exit -15
- Operator `revision30-external-20260728t225011z` at `2026-07-28T22:50:15Z` · exit 1
- Operator `revision30-external-20260728t230033z` at `2026-07-28T23:00:38Z` · exit 1

Report digest: `sha256:063ad17d91ec6d11f4c31ce6790528829fd66619a6f783e579b8f3ad4714ecac`
