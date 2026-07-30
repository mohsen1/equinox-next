import { useApi } from "../api";
import {
  AsyncState,
  formatDate,
  friendlyStatus,
  KeyValue,
  MachineId,
  Notice,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";
import {
  decodeResearchComputeExecution,
  decodeRunsResponse,
  type RunsResponse,
} from "../contracts";
import { formatEstimatedCost, formatRelativeTime } from "../format";
import { Link, useParams } from "../router";
import {
  booleanValue,
  complexityForLevel,
  formatDuration,
  formatRateInterval,
  intervalValue,
  numberValue,
  percent,
  recordValue,
  resultItems,
  runPercentage,
  shortModelName,
  stringValue,
} from "../runs-helpers";
import type {
  ResearchComputeExecution,
  ResearchObserverEvidence,
  ResearchValidationSummary,
} from "../types";
import { ResearchRunTabs } from "./research-trajectory";

const LARGER_MODEL_ELIGIBILITY_WORKLOADS = new Set([
  "repository-repair-larger-model-eligibility",
  "repository-repair-larger-model-eligibility-screen",
  "repository-repair-protocol-eligibility",
  "repository-repair-protocol-eligibility-screen",
]);

export function RunsPage() {
  const runs = useApi<RunsResponse>("/v1/runs", 2_000, decodeRunsResponse);
  const items =
    runs.data?.research_items.filter(
      (run) => run.execution_id !== "runpod-proof-ui-rehearsal",
    ) ?? [];

  return (
    <>
      <PageHeader title="Runs" />
      <div className="content workspace-content">
        <AsyncState
          loading={runs.loading}
          error={runs.error}
          stale={Boolean(runs.data && runs.error)}
          onRetry={runs.retry}
          empty={!items.length}
        >
          {items.length ? (
            <section className="run-queue" aria-label="Training runs">
              <div className="run-list-header" aria-hidden="true">
                <span>Run</span>
                <span>Status</span>
                <span>Progress</span>
                <span>GPU</span>
                <span>Total cost</span>
                <span>Updated</span>
              </div>
              <div className="run-list">
                {items.map((run) => (
                  <RunRow key={run.execution_id} run={run} />
                ))}
              </div>
            </section>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

function RunRow({ run }: { run: ResearchComputeExecution }) {
  const update = numberValue(run.progress.update);
  const maximumUpdates = numberValue(run.progress.maximum_updates);
  const level = numberValue(run.progress.current_level);
  const maximumLevel = numberValue(run.progress.maximum_level);
  const phase = stringValue(run.progress.phase);
  const attempt = numberValue(run.progress.attempt);
  const percentage = runPercentage(
    run.status,
    update,
    maximumUpdates,
    phase,
    attempt,
  );
  const progressLabel =
    update !== null
      ? `Update ${update}${maximumUpdates !== null ? ` of ${maximumUpdates}` : ""}`
      : observerProgressLabel(run, phase);
  const displayStatus = observerExecutionStatus(run);

  return (
    <Link
      className="run-row research-run-row"
      to={`/runs/research/${run.execution_id}`}
    >
      <span className="run-main">
        <strong>{run.name}</strong>
        <span>
          {shortModelName(run.model_id)} · K={run.branch_width} ·{" "}
          {run.complexity_strategy}
        </span>
      </span>
      <StatusBadge status={displayStatus} />
      <span className="run-progress">
        <span className="progress-track" aria-hidden="true">
          <span style={{ width: `${percentage}%` }} />
        </span>
        <span>{progressLabel}</span>
      </span>
      <span className="run-gpu">
        {run.allocated_gpu ?? "Awaiting allocation"}
        {level !== null ? (
          <small>
            Level {level}
            {maximumLevel !== null ? ` of ${maximumLevel}` : ""}
          </small>
        ) : null}
      </span>
      <span className="run-cost">{formatEstimatedCost(run.cost)}</span>
      <time
        dateTime={run.updated_at}
        aria-label={formatDate(run.updated_at)}
        title={formatDate(run.updated_at)}
      >
        {formatRelativeTime(run.updated_at)}
      </time>
    </Link>
  );
}

type ObserverStageKey =
  | "preparing"
  | "staging_bundle"
  | "verifying_runtime"
  | "screening"
  | "running"
  | "teardown"
  | "publication";

type ObserverStageState = "complete" | "current" | "failed" | "pending";

interface ObserverStage {
  key: ObserverStageKey;
  label: string;
  state: ObserverStageState;
}

const OBSERVER_STAGE_LABELS: Record<ObserverStageKey, string> = {
  preparing: "Prepare",
  staging_bundle: "Stage bundle",
  verifying_runtime: "Verify runtime/model",
  screening: "Screen",
  running: "Run",
  teardown: "Teardown",
  publication: "Publish",
};

const OBSERVER_EVIDENCE_KEYS = [
  "phase",
  "message",
  "profile_id",
  "head_commit",
  "source_contract_digest",
  "bootstrap_source_digest",
  "workload_bundle_digest",
  "workload_bundle_size_bytes",
  "workload_bundle_path",
  "bundle_stage_receipt_digest",
  "volume_readiness_receipt_digest",
  "torch_retention_evidence_digest",
  "bundle_activation_digest",
  "artifact_set_manifest_digest",
  "artifact_set_committed",
  "network_volume_id",
  "network_volume_data_center_id",
  "network_volume_size_gb",
  "model_snapshot_digest",
  "pinned_snapshot_digest",
  "model_revision",
  "gate_results",
  "screen_completed",
  "eligible",
  "larger_model_eligible",
  "evaluation_completed",
  "evaluation_total",
  "branch_groups_completed",
  "branch_groups_total",
  "update",
  "maximum_updates",
  "current_level",
  "maximum_level",
] as const satisfies ReadonlyArray<keyof ResearchObserverEvidence>;

function isEligibilityScreenExecution(run: ResearchComputeExecution): boolean {
  return (
    LARGER_MODEL_ELIGIBILITY_WORKLOADS.has(run.workload_id) ||
    run.workload_id.includes("eligibility")
  );
}

function observerEvidence(
  run: ResearchComputeExecution,
): ResearchObserverEvidence {
  if (run.observer_evidence) return run.observer_evidence;
  const evidence: ResearchObserverEvidence = {};
  const sources = [run.progress, run.resource_profile];
  for (const key of OBSERVER_EVIDENCE_KEYS) {
    for (const source of sources) {
      const value = source[key];
      if (value !== undefined && value !== null) {
        Object.assign(evidence, { [key]: value });
        break;
      }
    }
  }
  return evidence;
}

function observerPhaseStage(
  phase: string | null,
  status: ResearchComputeExecution["status"],
): ObserverStageKey {
  const normalized = phase?.toLowerCase() ?? "";
  if (
    normalized === "complete" ||
    normalized.includes("publish") ||
    normalized.includes("persisted")
  ) {
    return "publication";
  }
  if (
    normalized === "finalizing" ||
    normalized.includes("teardown") ||
    normalized.includes("release")
  ) {
    return "teardown";
  }
  if (
    normalized === "training" ||
    normalized === "resuming" ||
    normalized.includes("policy_update") ||
    normalized.includes("final_evaluation") ||
    normalized === "evaluation"
  ) {
    return "running";
  }
  if (
    normalized.includes("screen") ||
    normalized.includes("eligibility") ||
    normalized.includes("baseline_evaluation") ||
    normalized.includes("branch_collection")
  ) {
    return "screening";
  }
  if (
    normalized === "verifying_runtime" ||
    normalized === "activating" ||
    normalized === "model_loading" ||
    normalized.includes("runtime") ||
    normalized.includes("model")
  ) {
    return "verifying_runtime";
  }
  if (normalized === "staging_bundle" || normalized.includes("bundle_stag")) {
    return "staging_bundle";
  }
  if (status === "FINALIZING") return "teardown";
  if (status === "RUNNING") return "running";
  return "preparing";
}

function observerCurrentStage(
  run: ResearchComputeExecution,
  evidence: ResearchObserverEvidence,
  phase: string | null,
): ObserverStageKey {
  if (phase && phase !== "failed") return observerPhaseStage(phase, run.status);
  if (
    numberValue(evidence.branch_groups_completed) !== null ||
    recordValue(evidence.gate_results)
  ) {
    return "screening";
  }
  if (
    numberValue(evidence.update) !== null ||
    numberValue(evidence.current_level) !== null
  ) {
    return "running";
  }
  if (
    stringValue(evidence.bundle_activation_digest) ||
    stringValue(evidence.torch_retention_evidence_digest) ||
    stringValue(evidence.model_snapshot_digest) ||
    stringValue(evidence.pinned_snapshot_digest)
  ) {
    return "verifying_runtime";
  }
  if (stringValue(evidence.bundle_stage_receipt_digest)) {
    return "staging_bundle";
  }
  return observerPhaseStage(phase, run.status);
}

export function hasDurablePublication(run: ResearchComputeExecution): boolean {
  const evidence = observerEvidence(run);
  const digest = stringValue(evidence.artifact_set_manifest_digest);
  return Boolean(
    evidence.artifact_set_committed === true &&
      digest?.startsWith("sha256:") &&
      digest.length === 71,
  );
}

export function observerExecutionStatus(
  run: ResearchComputeExecution,
): ResearchComputeExecution["status"] {
  return run.status === "SUCCEEDED" &&
    run.artifact_publication_required === true &&
    !hasDurablePublication(run)
    ? "FINALIZING"
    : run.status;
}

export function integratedObserverStages(
  run: ResearchComputeExecution,
  eligibilityScreen = isEligibilityScreenExecution(run),
): ObserverStage[] {
  const keys: ObserverStageKey[] = [
    "preparing",
    "staging_bundle",
    "verifying_runtime",
    "screening",
    ...(eligibilityScreen ? [] : (["running"] as ObserverStageKey[])),
    "teardown",
    "publication",
  ];
  const evidence = observerEvidence(run);
  const phase = stringValue(evidence.phase) ?? stringValue(run.progress.phase);
  const currentKey = observerCurrentStage(run, evidence, phase);
  const currentIndex = Math.max(0, keys.indexOf(currentKey));
  const published =
    hasDurablePublication(run) ||
    (run.status === "SUCCEEDED" && run.artifact_publication_required !== true);
  const failurePublished = Boolean(run.failure_receipt_digest);

  return keys.map((key, index) => {
    let state: ObserverStageState =
      published || index < currentIndex
        ? "complete"
        : index === currentIndex
          ? "current"
          : "pending";

    if (key === "teardown" && run.teardown_confirmed) state = "complete";
    if (key === "publication" && (published || failurePublished)) {
      state = "complete";
    }
    if (
      run.status === "FAILED" &&
      key === currentKey &&
      key !== "publication" &&
      !(key === "teardown" && run.teardown_confirmed)
    ) {
      state = "failed";
    }

    return { key, label: OBSERVER_STAGE_LABELS[key], state };
  });
}

function observerProgressLabel(
  run: ResearchComputeExecution,
  phase: string | null,
): string {
  if (run.status === "SUCCEEDED" && hasDurablePublication(run)) {
    return "Published";
  }
  if (
    run.status === "SUCCEEDED" &&
    run.artifact_publication_required !== true
  ) {
    return "Complete";
  }
  if (run.status === "FAILED") return "Failed";
  const evidence = observerEvidence(run);
  const branchGroups = numberValue(evidence.branch_groups_completed);
  const branchTotal = numberValue(evidence.branch_groups_total);
  if (
    observerPhaseStage(phase, run.status) === "screening" &&
    branchGroups !== null
  ) {
    return `Screening · ${branchGroups}${
      branchTotal !== null ? ` of ${branchTotal}` : ""
    } groups`;
  }
  const labels: Partial<Record<string, string>> = {
    preparing: "Preparing",
    staging_bundle: "Staging bundle",
    verifying_runtime: "Verifying runtime",
    activating: "Activating bundle",
    finalizing: run.teardown_confirmed ? "Publishing" : "Teardown",
    complete: hasDurablePublication(run) ? "Published" : "Publishing",
  };
  return labels[phase ?? ""] ?? friendlyStatus(phase ?? run.status);
}

export function ResearchRunPage() {
  const { executionId = "" } = useParams<{ executionId: string }>();
  const execution = useApi<ResearchComputeExecution>(
    `/v1/research-compute-executions/${encodeURIComponent(executionId)}`,
    2_000,
    decodeResearchComputeExecution,
  );
  const run = execution.data;
  const claimStrength = stringValue(run?.progress.claim_strength);
  const isEligibilityScreen =
    run !== null && run !== undefined && isEligibilityScreenExecution(run);
  const eligibility = run ? largerModelEligibility(run) : null;
  const stages = run ? integratedObserverStages(run, isEligibilityScreen) : [];
  const durablePublication = run ? hasDurablePublication(run) : false;

  return (
    <>
      <PageHeader
        eyebrow={<Link to="/runs">Runs</Link>}
        title={run?.name ?? "Run"}
        description={
          run
            ? `${shortModelName(run.model_id)} · K=${run.branch_width} · ${run.complexity_strategy}`
            : undefined
        }
        actions={
          run ? (
            <StatusBadge status={observerExecutionStatus(run)} />
          ) : undefined
        }
      />
      <ResearchRunTabs executionId={executionId} active="overview" />
      <div className="content workspace-content">
        <AsyncState
          loading={execution.loading}
          error={execution.error}
          stale={Boolean(execution.data && execution.error)}
          onRetry={execution.retry}
        >
          {run ? (
            <div className="research-observer">
              {run.status === "FAILED" ? (
                <Notice tone="negative" title="Run failed">
                  <KeyValue items={failureItems(run)} />
                </Notice>
              ) : run.status === "SUCCEEDED" &&
                run.artifact_publication_required === true &&
                !durablePublication ? (
                <Notice tone="warning" title="Publication pending">
                  A durable proof receipt has not been recorded.
                </Notice>
              ) : null}

              <section className="observer-stage" aria-label="Run lifecycle">
                {stages.map((stage) => (
                  <div
                    key={stage.label}
                    className={`observer-stage-item ${stage.state}`}
                    aria-label={`${stage.label}: ${friendlyStatus(stage.state)}`}
                  >
                    <span>{stage.label}</span>
                    <span className="observer-stage-status" aria-hidden="true">
                      {stage.state === "complete"
                        ? "✓"
                        : stage.state === "failed"
                          ? "!"
                          : stage.state === "current"
                            ? "→"
                            : "○"}
                    </span>
                  </div>
                ))}
              </section>

              <div className="run-overview research-observer-grid">
                {isEligibilityScreen && eligibility ? (
                  <Section
                    title="Eligibility"
                    aside={
                      <StatusBadge
                        status={
                          eligibility.eligible === null
                            ? run.status === "SUCCEEDED" ||
                              run.status === "FAILED"
                              ? "INCOMPLETE"
                              : "SCREENING"
                            : eligibility.eligible
                              ? "ELIGIBLE"
                              : "INELIGIBLE"
                        }
                      />
                    }
                  >
                    <KeyValue items={eligibility.items} />
                  </Section>
                ) : (
                  <Section title="Progress">
                    <KeyValue items={progressItems(run)} />
                  </Section>
                )}
                <Section title="Compute">
                  <KeyValue items={allocationItems(run)} />
                </Section>
              </div>

              <Section title="Evidence">
                <KeyValue items={observerEvidenceItems(run)} />
              </Section>

              {!isEligibilityScreen && validationRows(run).length ? (
                <Section title="Validation">
                  <ValidationHistory
                    rows={validationRows(run)}
                    best={run.progress.best_validation}
                  />
                </Section>
              ) : null}

              {run.status === "SUCCEEDED" && !isEligibilityScreen ? (
                <Section title="Result">
                  <KeyValue items={resultItems(run, claimStrength)} />
                </Section>
              ) : null}

              <Section
                title="Identity"
                aside={<MachineId value={run.execution_id} />}
              >
                <KeyValue
                  items={[
                    { label: "Workload", value: run.workload_id },
                    { label: "Model", value: run.model_id ?? "Not reported" },
                    {
                      label: "Branching",
                      value: `Static K=${run.branch_width}`,
                    },
                    { label: "Complexity", value: run.complexity_strategy },
                    {
                      label: "Proof",
                      value:
                        run.proof_id &&
                        (hasDurablePublication(run) ||
                          run.artifact_publication_required !== true) ? (
                          <Link to={`/proofs/${run.proof_id}`}>
                            <MachineId value={run.proof_id} />
                          </Link>
                        ) : run.status === "FAILED" ? (
                          "Not produced"
                        ) : isEligibilityScreen &&
                          run.status === "SUCCEEDED" ? (
                          "Not required"
                        ) : (
                          "Pending publication"
                        ),
                    },
                    ...(run.status === "SUCCEEDED" && run.failure_receipt_digest
                      ? [
                          {
                            label: "Prior failure receipt",
                            value: (
                              <MachineId value={run.failure_receipt_digest} />
                            ),
                          },
                        ]
                      : []),
                    { label: "Updated", value: formatDate(run.updated_at) },
                  ]}
                />
              </Section>
            </div>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

function failureItems(run: ResearchComputeExecution) {
  const legacyRemoteMessage = stringValue(run.progress.error);
  const remoteError = executionError(
    run.progress.remote_error,
    legacyRemoteMessage,
  );
  const operatorError = executionError(run.progress.operator_error);
  return [
    ...(remoteError
      ? [
          {
            label: "Remote",
            value: <FailureError error={remoteError} />,
            span: true,
          },
        ]
      : []),
    ...(operatorError
      ? [
          {
            label: "Operator",
            value: <FailureError error={operatorError} />,
            span: true,
          },
        ]
      : []),
    {
      label: "Failure receipt",
      value: run.failure_receipt_digest ? (
        <MachineId value={run.failure_receipt_digest} />
      ) : (
        "Not persisted"
      ),
    },
    {
      label: "Teardown",
      value: run.teardown_confirmed ? "Confirmed" : "Unconfirmed",
    },
  ];
}

function observerEvidenceItems(run: ResearchComputeExecution) {
  const evidence = observerEvidence(run);
  const phase = stringValue(evidence.phase);
  const message = stringValue(evidence.message);
  const phaseLabel = phase ? friendlyStatus(phase) : null;
  const showMessage =
    message !== null &&
    normalizeObserverText(message) !== normalizeObserverText(phaseLabel);
  const gates = recordValue(evidence.gate_results);
  const gateEntries = Object.entries(gates ?? {}).filter(
    (entry): entry is [string, boolean] => typeof entry[1] === "boolean",
  );
  const branchGroups = numberValue(evidence.branch_groups_completed);
  const branchGroupTotal = numberValue(evidence.branch_groups_total);
  const evaluation = numberValue(evidence.evaluation_completed);
  const evaluationTotal = numberValue(evidence.evaluation_total);
  const modelSnapshot =
    stringValue(evidence.model_snapshot_digest) ??
    stringValue(evidence.pinned_snapshot_digest);
  const networkVolume = stringValue(evidence.network_volume_id);
  const networkDataCenter = stringValue(evidence.network_volume_data_center_id);
  const networkSize = numberValue(evidence.network_volume_size_gb);

  return [
    ...(phase ? [{ label: "Phase", value: friendlyStatus(phase) }] : []),
    ...(showMessage ? [{ label: "Message", value: message, span: true }] : []),
    ...machineEvidence("Profile", stringValue(evidence.profile_id)),
    ...machineEvidence("Commit", stringValue(evidence.head_commit)),
    ...machineEvidence(
      "Source contract",
      stringValue(evidence.source_contract_digest),
    ),
    ...machineEvidence(
      "Bootstrap",
      stringValue(evidence.bootstrap_source_digest),
    ),
    ...machineEvidence("Bundle", stringValue(evidence.workload_bundle_digest)),
    ...(numberValue(evidence.workload_bundle_size_bytes) !== null
      ? [
          {
            label: "Bundle size",
            value: formatBytes(
              numberValue(evidence.workload_bundle_size_bytes) ?? 0,
            ),
          },
        ]
      : []),
    ...(stringValue(evidence.workload_bundle_path)
      ? [
          {
            label: "Bundle path",
            value: <code>{stringValue(evidence.workload_bundle_path)}</code>,
            span: true,
          },
        ]
      : []),
    ...machineEvidence(
      "Bundle stage",
      stringValue(evidence.bundle_stage_receipt_digest),
    ),
    ...machineEvidence(
      "Volume readiness",
      stringValue(evidence.volume_readiness_receipt_digest),
    ),
    ...machineEvidence(
      "Torch retention",
      stringValue(evidence.torch_retention_evidence_digest),
    ),
    ...machineEvidence(
      "Activation",
      stringValue(evidence.bundle_activation_digest),
    ),
    ...(stringValue(evidence.artifact_set_manifest_digest)
      ? [
          {
            label: "Artifact set",
            value: (
              <>
                <MachineId
                  value={stringValue(evidence.artifact_set_manifest_digest)!}
                />
                {evidence.artifact_set_committed === true ? " · committed" : ""}
              </>
            ),
          },
        ]
      : evidence.artifact_set_committed !== undefined
        ? [
            {
              label: "Artifact set",
              value:
                evidence.artifact_set_committed === true
                  ? "Commit marker only"
                  : "Not committed",
            },
          ]
        : run.status === "SUCCEEDED" &&
            run.artifact_publication_required !== true
          ? [{ label: "Artifact set", value: "Legacy · non-atomic" }]
          : []),
    ...machineEvidence("Model snapshot", modelSnapshot),
    ...machineEvidence("Model revision", stringValue(evidence.model_revision)),
    ...(networkVolume
      ? [
          {
            label: "Network volume",
            value: (
              <>
                <MachineId value={networkVolume} />
                {networkDataCenter ? ` · ${networkDataCenter}` : ""}
                {networkSize !== null ? ` · ${networkSize} GB` : ""}
              </>
            ),
          },
        ]
      : []),
    ...(branchGroups !== null
      ? [
          {
            label: "Branch groups",
            value: `${branchGroups}${
              branchGroupTotal !== null ? ` / ${branchGroupTotal}` : ""
            }`,
          },
        ]
      : []),
    ...(evaluation !== null
      ? [
          {
            label: "Evaluation",
            value: `${evaluation}${
              evaluationTotal !== null ? ` / ${evaluationTotal}` : ""
            }`,
          },
        ]
      : []),
    ...(gateEntries.length
      ? [
          {
            label: "Gates",
            value: gateEntries
              .map(
                ([name, passed]) =>
                  `${friendlyStatus(name)}: ${passed ? "pass" : "fail"}`,
              )
              .join(" · "),
            span: true,
          },
        ]
      : []),
    ...(run.receipt_digest && run.proof_id && hasDurablePublication(run)
      ? [
          {
            label: "Proof receipt",
            value: <MachineId value={run.receipt_digest} />,
          },
        ]
      : []),
    ...(run.failure_receipt_digest
      ? [
          {
            label: "Failure receipt",
            value: <MachineId value={run.failure_receipt_digest} />,
          },
        ]
      : []),
  ];
}

function machineEvidence(label: string, value: string | null) {
  return value
    ? [
        {
          label,
          value: <MachineId value={value} />,
        },
      ]
    : [];
}

function formatBytes(value: number): string {
  if (value < 1_000) return `${value} B`;
  if (value < 1_000_000) return `${(value / 1_000).toFixed(1)} KB`;
  return `${(value / 1_000_000).toFixed(1)} MB`;
}

function normalizeObserverText(value: string | null): string {
  return (value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

interface ExecutionError {
  code: string | null;
  message: string;
  exitCode: number | null;
}

function executionError(
  value: unknown,
  fallbackMessage: string | null = null,
): ExecutionError | null {
  const error = recordValue(value);
  const message = stringValue(error?.message) ?? fallbackMessage;
  if (!message) return null;
  return {
    code: stringValue(error?.code),
    message,
    exitCode: numberValue(error?.exit_code),
  };
}

function FailureError({ error }: { error: ExecutionError }) {
  return (
    <span className="failure-error">
      {error.code ? <code>{error.code}</code> : null}
      <span>{error.message}</span>
      {error.exitCode !== null ? (
        <small>Process exit {error.exitCode}</small>
      ) : null}
    </span>
  );
}

function largerModelEligibility(run: ResearchComputeExecution) {
  const progress = run.progress;
  const terminal = run.status === "SUCCEEDED" || run.status === "FAILED";
  const missingRate =
    run.status === "FAILED"
      ? "Incomplete"
      : terminal
        ? "Unavailable"
        : "Awaiting samples";
  const evidence =
    recordValue(progress.larger_model_eligibility) ??
    recordValue(progress.eligibility);
  const value = (key: string): unknown =>
    progress[key] ?? evidence?.[key] ?? null;
  const eligible = firstBoolean(
    value("eligible"),
    value("larger_model_eligible"),
  );
  const profile = firstString(
    value("larger_model_profile_id"),
    value("profile_id"),
    value("profile"),
  );
  const revision = firstString(value("model_revision"), value("model_commit"));
  const checkpointAdmission = firstNumber(
    value("checkpoint_admission_rate"),
    value("branch_checkpoint_rate"),
    value("checkpoint_rate"),
  );
  const informativeBranching = firstNumber(
    value("informative_branching_rate"),
    value("informative_group_rate"),
  );
  const peakGpuMemoryGb = gpuMemoryGb(
    firstNumber(
      value("peak_gpu_memory_gb"),
      value("peak_cuda_memory_gb"),
      value("gpu_peak_memory_gb"),
      value("peak_reserved_vram_gb"),
    ),
    firstNumber(
      value("peak_gpu_memory_bytes"),
      value("peak_cuda_memory_bytes"),
      value("peak_reserved_vram_bytes"),
    ),
  );
  const peakReservedVramFraction = firstNumber(
    value("peak_reserved_vram_fraction"),
  );
  const gateResults = recordValue(value("gate_results"));
  const mutationDetected = firstBoolean(value("policy_mutation_detected"));
  const mutationVerified = firstBoolean(
    value("policy_mutation_verified"),
    value("no_policy_mutation_verified"),
    value("policy_parameters_unchanged"),
    value("policy_unchanged"),
    gateResults?.policy_unchanged,
  );
  const mutationEnabled = firstBoolean(value("policy_mutation_enabled"));

  return {
    eligible,
    items: [
      {
        label: "Revision",
        value: revision ? (
          <MachineId value={revision} />
        ) : terminal ? (
          "Unavailable"
        ) : (
          "Awaiting model load"
        ),
      },
      {
        label: "Profile",
        value: profile ?? (terminal ? "Unavailable" : "Awaiting screen"),
      },
      {
        label: "Checkpoint admission",
        value: percent(checkpointAdmission, missingRate),
      },
      {
        label: "Informative branching",
        value: percent(informativeBranching, missingRate),
      },
      {
        label: "Peak GPU memory",
        value:
          peakReservedVramFraction !== null
            ? `${percent(peakReservedVramFraction)} reserved`
            : peakGpuMemoryGb === null
              ? terminal
                ? "Unavailable"
                : "Awaiting model load"
              : `${formatDecimal(peakGpuMemoryGb)} GB`,
      },
      {
        label: "Policy mutation",
        value:
          mutationDetected === false && mutationVerified === true
            ? "None · verified"
            : mutationDetected === true ||
                mutationVerified === false ||
                mutationEnabled === true
              ? "Detected"
              : mutationDetected === false
                ? `None · verification ${terminal ? "unavailable" : "pending"}`
                : mutationEnabled === false
                  ? `Disabled · verification ${
                      terminal ? "unavailable" : "pending"
                    }`
                  : terminal
                    ? "Verification unavailable"
                    : "Verification pending",
      },
    ],
  };
}

function firstBoolean(...values: unknown[]): boolean | null {
  for (const value of values) {
    const parsed = booleanValue(value);
    if (parsed !== null) return parsed;
  }
  return null;
}

function firstNumber(...values: unknown[]): number | null {
  for (const value of values) {
    const parsed = numberValue(value);
    if (parsed !== null) return parsed;
  }
  return null;
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    const parsed = stringValue(value);
    if (parsed !== null) return parsed;
  }
  return null;
}

function gpuMemoryGb(
  gigabytes: number | null,
  bytes: number | null,
): number | null {
  if (gigabytes !== null) return gigabytes;
  return bytes === null ? null : bytes / 1_000_000_000;
}

function formatDecimal(value: number): string {
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
}

function progressItems(run: ResearchComputeExecution) {
  const progress = run.progress;
  const update = numberValue(progress.update);
  const maximumUpdates = numberValue(progress.maximum_updates);
  const currentLevel = numberValue(progress.current_level);
  const maximumLevel = numberValue(progress.maximum_level);
  const maximumSampledLevel = numberValue(
    progress.maximum_sampled_complexity_level,
  );
  const curriculumDecision = recordValue(progress.curriculum_decision);
  const frontierProbeLevel = numberValue(
    curriculumDecision?.frontier_probe_level_used,
  );
  const nextFrontierProbeLevel = numberValue(
    curriculumDecision?.next_frontier_probe_level,
  );
  const frontierProbeSignal = probeSignal(
    stringValue(curriculumDecision?.frontier_probe_decision),
  );
  const validationRate = numberValue(progress.exact_rate);
  const validationInterval = intervalValue(progress.exact_rate_95ci);
  const validationExamples = numberValue(progress.validation_examples);
  const exactRateSource = stringValue(progress.exact_rate_source);
  const aggregateTestMean = exactRateSource === "aggregate_test_mean";
  const finalEvaluationReward = exactRateSource === "final_evaluation_reward";
  const evaluationExamples =
    aggregateTestMean || finalEvaluationReward
      ? null
      : (numberValue(progress.evaluation_examples) ?? validationExamples);
  const evaluationSplit = stringValue(progress.evaluation_split);
  const sampledCompletions = numberValue(progress.sampled_completions);
  const elapsed = numberValue(progress.elapsed_seconds);
  const informativeGroupRate = numberValue(progress.informative_group_rate);
  const policyUpdates = numberValue(progress.policy_update_count);
  const attemptedPolicyUpdates = numberValue(
    progress.attempted_policy_update_count,
  );
  const effectivePolicyUpdates = numberValue(
    progress.effective_policy_update_count,
  );
  const retainedPolicyUpdates = numberValue(
    progress.retained_policy_update_count,
  );
  const retentionRollbacks = numberValue(progress.retention_rollback_count);
  const policyLineage =
    attemptedPolicyUpdates === null
      ? policyUpdates === null
        ? null
        : `${policyUpdates} policy`
      : `${attemptedPolicyUpdates} attempted · ${
          effectivePolicyUpdates ?? "—"
        } effective · ${retainedPolicyUpdates ?? "—"} retained${
          retentionRollbacks !== null && retentionRollbacks > 0
            ? ` · ${retentionRollbacks} rolled back`
            : ""
        }`;
  const pendingSignalGroups = numberValue(
    progress.pending_informative_group_count,
  );
  const protocolValidity = numberValue(progress.action_protocol_validity_rate);
  const recentMalformedRate = numberValue(
    progress.recent_malformed_action_rate,
  );
  const totalActions = numberValue(progress.total_sampled_actions);
  const trainingRemaining = numberValue(progress.training_remaining_seconds);
  const evaluationReserve = numberValue(
    progress.final_evaluation_reserve_seconds,
  );
  const baselineRate =
    evaluationSplit === "test"
      ? (progress.initial_level_exact_rate ?? null)
      : (progress.baseline_validation?.exact_rate ?? null);
  const activeComplexity = complexityForLevel(
    currentLevel,
    progress.active_complexity,
  );
  const evaluationCompleted = numberValue(progress.evaluation_completed);
  const evaluationTotal = numberValue(progress.evaluation_total);
  const progressPhase = stringValue(progress.phase);
  const evaluating =
    progressPhase?.includes("evaluation") === true &&
    evaluationCompleted !== null &&
    evaluationTotal !== null &&
    evaluationTotal > 0;

  return [
    {
      label: "Phase",
      value:
        run.status === "SUCCEEDED"
          ? "Complete"
          : friendlyStatus(progressPhase ?? run.status),
    },
    {
      label: evaluating ? "Evaluation" : "Updates",
      value: evaluating
        ? `${evaluationCompleted} / ${evaluationTotal} tasks`
        : update === null
          ? "Not started"
          : `${update}${maximumUpdates !== null ? ` / ${maximumUpdates}` : ""}${
              policyLineage !== null ? ` · ${policyLineage}` : ""
            }${
              pendingSignalGroups !== null && pendingSignalGroups > 0
                ? ` · ${pendingSignalGroups} pending`
                : ""
            }`,
    },
    {
      label: "Curriculum",
      value:
        currentLevel === null
          ? "Awaiting evaluation"
          : `Level ${currentLevel}${
              maximumLevel !== null ? ` of ${maximumLevel}` : ""
            }${
              maximumSampledLevel !== null && maximumSampledLevel > currentLevel
                ? ` · probed through level ${maximumSampledLevel}`
                : ""
            }${
              activeComplexity
                ? ` · ${activeComplexity.file_count} files · ${activeComplexity.fault_count} fault${
                    activeComplexity.fault_count === 1 ? "" : "s"
                  } · horizon ${activeComplexity.repair_horizon}`
                : ""
            }`,
    },
    ...(frontierProbeLevel === null
      ? []
      : [
          {
            label: "Probe",
            value: `Level ${frontierProbeLevel}${
              nextFrontierProbeLevel !== null &&
              nextFrontierProbeLevel !== frontierProbeLevel
                ? ` → ${nextFrontierProbeLevel}`
                : ""
            }${frontierProbeSignal ? ` · ${frontierProbeSignal}` : ""}`,
          },
        ]),
    {
      label: aggregateTestMean
        ? "Test mean"
        : finalEvaluationReward
          ? "Final reward"
          : evaluationSplit === "test"
            ? "Test solve"
            : evaluationSplit === "validation"
              ? "Validation solve"
              : "Solve rate",
      value: `${formatRateInterval(
        validationRate,
        validationInterval,
        evaluationExamples,
        "Awaiting evaluation",
      )}${
        baselineRate !== null && validationRate !== null
          ? ` · ${formatSignedPoints(validationRate - baselineRate)} vs baseline`
          : ""
      }`,
    },
    ...(informativeGroupRate === null
      ? []
      : [
          { label: "Informative groups", value: percent(informativeGroupRate) },
        ]),
    ...(protocolValidity === null && recentMalformedRate === null
      ? []
      : [
          {
            label: "Action protocol",
            value: `${
              protocolValidity === null
                ? "Validity pending"
                : `${percent(protocolValidity)} valid`
            }${
              recentMalformedRate === null
                ? ""
                : ` · ${percent(recentMalformedRate)} malformed recent`
            }`,
          },
        ]),
    {
      label: totalActions === null ? "Completions" : "Actions",
      value:
        totalActions?.toLocaleString() ??
        sampledCompletions?.toLocaleString() ??
        "0",
    },
    ...(run.status === "PROVISIONING" ||
    run.status === "RUNNING" ||
    run.status === "FINALIZING"
      ? [
          {
            label: "Reserve",
            value:
              trainingRemaining === null && evaluationReserve === null
                ? "Measuring"
                : `${
                    trainingRemaining === null
                      ? "Training closed"
                      : `${formatDuration(trainingRemaining)} training`
                  }${
                    evaluationReserve === null
                      ? ""
                      : ` · ${formatDuration(evaluationReserve)} evaluation`
                  }`,
          },
        ]
      : []),
    {
      label: "Elapsed",
      value: elapsed === null ? "—" : formatDuration(elapsed),
    },
  ];
}

function probeSignal(reason: string | null) {
  switch (reason) {
    case "mixed_correctness_contrast_retained":
      return "mixed K=4";
    case "all_siblings_solved_raise_probe":
    case "hardest_probe_all_solved":
      return "4/4 solved";
    case "no_siblings_solved_lower_probe":
    case "nearest_probe_all_failed":
      return "0/4 solved";
    case "heterogeneous_saturation_hold_probe":
      return "split extremes";
    case "curriculum_promotion_reset_to_nearest_probe":
      return "promoted";
    case "maximum_level_reached":
      return "maximum";
    case "insufficient_probe_evidence":
      return "awaiting K=4";
    default:
      return null;
  }
}

function validationRows(
  run: ResearchComputeExecution,
): ResearchValidationSummary[] {
  const baseline = run.progress.baseline_validation;
  const history = run.progress.validation_history ?? [];
  return [...(baseline ? [{ ...baseline, update: 0 }] : []), ...history];
}

function ValidationHistory({
  rows,
  best,
}: {
  rows: ResearchValidationSummary[];
  best?: ResearchValidationSummary;
}) {
  return (
    <div className="validation-history">
      <table>
        <thead>
          <tr>
            <th>Checkpoint</th>
            <th>Exact</th>
            <th>95% interval</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const isBaseline = index === 0 && row.update === 0;
            const isBest =
              best?.update !== undefined && row.update === best.update;
            return (
              <tr key={`${row.update ?? "baseline"}-${row.level ?? 0}`}>
                <td>
                  {isBaseline ? "Baseline" : `Update ${row.update ?? "—"}`}
                  <small>Level {row.level ?? 0}</small>
                </td>
                <td>
                  {row.exact_rate === undefined ? "—" : percent(row.exact_rate)}
                  {row.exact_successes !== undefined &&
                  row.examples !== undefined ? (
                    <small>
                      {row.exact_successes} / {row.examples}
                    </small>
                  ) : null}
                </td>
                <td>{formatInterval(row.exact_rate_95ci)}</td>
                <td>
                  <span>
                    {row.retention_transaction_disposition === "retain"
                      ? isBest
                        ? "Best retained"
                        : "Retained"
                      : row.retention_transaction_disposition === "rollback"
                        ? "Rolled back"
                        : row.retention_transaction_disposition ===
                            "provisional"
                          ? "Provisional"
                          : isBest
                            ? "Best retained"
                            : row.retention_guard_passed === false
                              ? "Guard rejected"
                              : row.mastered
                                ? `Mastery ${row.mastery_streak ?? 1}`
                                : row.regression_streak
                                  ? `Regression ${row.regression_streak}`
                                  : isBaseline
                                    ? "Reference"
                                    : "Continue"}
                  </span>
                  {row.fixed_guard_paired_change &&
                  row.fixed_guard_levels?.length ? (
                    <small>
                      Fixed {row.fixed_guard_levels.join(", ")} · +
                      {row.fixed_guard_paired_change.improved} / −
                      {row.fixed_guard_paired_change.regressed}
                      {row.curriculum_paired_change
                        ? ` · rotating +${row.curriculum_paired_change.improved} / −${row.curriculum_paired_change.regressed}`
                        : ""}
                    </small>
                  ) : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function formatInterval(value?: [number, number]): string {
  if (!value) return "—";
  return `${percent(value[0])}–${percent(value[1])}`;
}

function formatSignedPoints(value: number): string {
  const points = value * 100;
  if (Math.abs(points) < 0.05) return "0.0 pts";
  return `${points > 0 ? "+" : ""}${points.toFixed(1)} pts`;
}

function allocationItems(run: ResearchComputeExecution) {
  const hourlyRate = numberValue(run.resource_profile.hourly_cost_usd);
  const spendingCap = numberValue(run.resource_profile.maximum_hourly_cost_usd);
  return [
    {
      label: "GPU",
      value:
        stringValue(run.resource_profile.gpu_id) ??
        run.allocated_gpu ??
        "Awaiting allocation",
    },
    {
      label: "Cloud",
      value:
        stringValue(run.resource_profile.cloud_type) ?? "Awaiting allocation",
    },
    {
      label: "Hourly rate",
      value:
        hourlyRate === null ? "Not reported" : `$${hourlyRate.toFixed(3)}/hour`,
    },
    {
      label: "Spending cap",
      value:
        spendingCap === null
          ? "Not reported"
          : `$${spendingCap.toFixed(2)}/hour`,
    },
    {
      label: "Provider handle",
      value: run.provider_handle ? (
        <MachineId value={run.provider_handle} />
      ) : (
        "Not allocated"
      ),
    },
    { label: "Started", value: formatDate(run.started_at) },
    {
      label: "Teardown",
      value: run.teardown_confirmed
        ? "Confirmed"
        : run.status === "FINALIZING"
          ? "In progress"
          : "Pending",
    },
  ];
}
