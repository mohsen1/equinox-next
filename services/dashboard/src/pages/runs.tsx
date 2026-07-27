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
import type { ResearchComputeExecution } from "../types";
import { ResearchRunTabs } from "./research-trajectory";

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
  const percentage =
    update !== null && maximumUpdates !== null && maximumUpdates > 0
      ? Math.min(100, (update / maximumUpdates) * 100)
      : phasePercentage(run.status, phase);
  const progressLabel =
    update !== null
      ? `Update ${update}${maximumUpdates !== null ? ` of ${maximumUpdates}` : ""}`
      : friendlyStatus(phase ?? run.status);

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
      <StatusBadge status={run.status} />
      <span className="run-progress">
        <span className="progress-track" aria-hidden="true">
          <span style={{ width: `${percentage}%` }} />
        </span>
        <span>{progressLabel}</span>
      </span>
      <span className="run-gpu">
        {run.allocated_gpu ?? "Awaiting allocation"}
        <small>
          {level !== null
            ? `Level ${level}${maximumLevel !== null ? ` of ${maximumLevel}` : ""}`
            : "No evaluation yet"}
        </small>
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

export function ResearchRunPage() {
  const { executionId = "" } = useParams<{ executionId: string }>();
  const execution = useApi<ResearchComputeExecution>(
    `/v1/research-compute-executions/${encodeURIComponent(executionId)}`,
    2_000,
    decodeResearchComputeExecution,
  );
  const run = execution.data;
  const phase = stringValue(run?.progress.phase);
  const error = stringValue(run?.progress.error);
  const claimStrength = stringValue(run?.progress.claim_strength);

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
        actions={run ? <StatusBadge status={run.status} /> : undefined}
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
              {error ? (
                <Notice tone="negative" title="Run failed">
                  {error}
                </Notice>
              ) : run.status === "SUCCEEDED" ? (
                <Notice title="Run complete">
                  {run.teardown_confirmed
                    ? "Result persisted. Compute release confirmed."
                    : "Result persisted. Compute release is not confirmed."}
                </Notice>
              ) : null}

              <section className="observer-stage" aria-label="Run lifecycle">
                {observerStages(run.status, phase, run.teardown_confirmed).map(
                  (stage) => (
                    <div
                      key={stage.label}
                      className={`observer-stage-item ${stage.state}`}
                    >
                      <span>{stage.label}</span>
                      <small>{stage.detail}</small>
                    </div>
                  ),
                )}
              </section>

              <div className="run-overview research-observer-grid">
                <Section title="Progress">
                  <KeyValue items={progressItems(run)} />
                </Section>
                <Section title="Compute">
                  <KeyValue items={allocationItems(run)} />
                </Section>
              </div>

              {run.status === "SUCCEEDED" ? (
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
                      value: run.proof_id ? (
                        <Link to={`/proofs/${run.proof_id}`}>
                          <MachineId value={run.proof_id} />
                        </Link>
                      ) : (
                        "Pending"
                      ),
                    },
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

function progressItems(run: ResearchComputeExecution) {
  const progress = run.progress;
  const update = numberValue(progress.update);
  const maximumUpdates = numberValue(progress.maximum_updates);
  const currentLevel = numberValue(progress.current_level);
  const maximumLevel = numberValue(progress.maximum_level);
  const validationRate = numberValue(progress.exact_rate);
  const sampledCompletions = numberValue(progress.sampled_completions);
  const elapsed = numberValue(progress.elapsed_seconds);
  const informativeGroupRate = numberValue(progress.informative_group_rate);

  return [
    {
      label: "Phase",
      value:
        run.status === "SUCCEEDED"
          ? "Complete"
          : friendlyStatus(stringValue(progress.phase) ?? run.status),
    },
    {
      label: "Updates",
      value:
        update === null
          ? "Not started"
          : `${update}${maximumUpdates !== null ? ` / ${maximumUpdates}` : ""}`,
    },
    {
      label: "Curriculum",
      value:
        currentLevel === null
          ? "Awaiting evaluation"
          : `Level ${currentLevel}${maximumLevel !== null ? ` of ${maximumLevel}` : ""}`,
    },
    {
      label: "Validation exact",
      value: percent(validationRate, "Awaiting evaluation"),
    },
    ...(informativeGroupRate === null
      ? []
      : [
          { label: "Informative groups", value: percent(informativeGroupRate) },
        ]),
    {
      label: "Completions",
      value: sampledCompletions?.toLocaleString() ?? "0",
    },
    {
      label: "Elapsed",
      value: elapsed === null ? "—" : formatDuration(elapsed),
    },
  ];
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

function resultItems(
  run: ResearchComputeExecution,
  claimStrength: string | null,
) {
  const progress = run.progress;
  const initial = numberValue(progress.initial_exact_rate);
  const final = numberValue(progress.final_exact_rate);
  const gain = numberValue(progress.reward_gain);
  const promotions = numberValue(progress.promotion_count);
  const adapterPersisted = booleanValue(progress.adapter_persisted);
  return [
    {
      label: "Claim",
      value:
        claimStrength === "EXPLORATORY_SINGLE_SEED"
          ? "Exploratory · one seed"
          : claimStrength
            ? friendlyStatus(claimStrength)
            : "Not reported",
    },
    { label: "Baseline test", value: percent(initial, "Not reported") },
    { label: "Final test", value: percent(final, "Not reported") },
    {
      label: "Test gain",
      value:
        gain === null
          ? "Not reported"
          : `${gain >= 0 ? "+" : ""}${(gain * 100).toFixed(1)} pts`,
    },
    { label: "Promotions", value: promotions?.toLocaleString() ?? "0" },
    {
      label: "Stop reason",
      value: friendlyStatus(
        stringValue(progress.stop_reason) ?? "not reported",
      ),
    },
    {
      label: "Adapter",
      value:
        adapterPersisted === null
          ? "Not reported"
          : adapterPersisted
            ? "Verified"
            : "Missing",
    },
  ];
}

function observerStages(
  status: ResearchComputeExecution["status"],
  phase: string | null,
  teardownConfirmed: boolean,
): Array<{ label: string; detail: string; state: string }> {
  const learning = ["training", "evaluation", "baseline_evaluation"];
  const current =
    status === "SUCCEEDED"
      ? 3
      : status === "FINALIZING"
        ? 2
        : phase && learning.includes(phase)
          ? 1
          : 0;
  const labels = [
    ["Allocate", "Provider capacity"],
    ["Learn", "Sample · verify · update"],
    ["Finalize", "Persist result"],
    ["Release", "Confirm teardown"],
  ];
  return labels.map(([label, detail], index) => ({
    label,
    detail,
    state:
      status === "SUCCEEDED"
        ? "complete"
        : status === "FAILED"
          ? index === 3 && teardownConfirmed
            ? "complete"
            : index === current
              ? "failed"
              : index < current
                ? "complete"
                : "pending"
          : index < current
            ? "complete"
            : index === current
              ? "current"
              : "pending",
  }));
}

function phasePercentage(
  status: ResearchComputeExecution["status"],
  phase: string | null,
): number {
  if (status === "SUCCEEDED" || status === "FAILED") return 100;
  if (status === "FINALIZING") return 92;
  if (status === "PROVISIONING") return phase === "container_starting" ? 12 : 5;
  if (phase === "dependency_setup") return 20;
  if (phase === "model_loading") return 28;
  if (phase === "baseline_evaluation") return 36;
  if (phase === "training") return 50;
  if (phase === "evaluation") return 70;
  return 18;
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length ? value : null;
}

function booleanValue(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function shortModelName(value: string | null): string {
  if (!value) return "Model pending";
  return value.split("/").at(-1) ?? value;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)} sec`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function percent(value: number | null, fallback = "Not reported"): string {
  return value === null ? fallback : `${(value * 100).toFixed(1)}%`;
}
