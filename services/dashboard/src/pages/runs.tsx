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
  formatDuration,
  formatRateInterval,
  intervalValue,
  numberValue,
  observerStages,
  percent,
  resultItems,
  runPercentage,
  shortModelName,
  stringValue,
} from "../runs-helpers";
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
  const attempt = numberValue(run?.progress.attempt);

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
                {observerStages(
                  run.status,
                  phase,
                  run.teardown_confirmed,
                  attempt,
                  numberValue(run.progress.update) !== null ||
                    numberValue(run.progress.current_level) !== null,
                ).map((stage) => (
                  <div
                    key={stage.label}
                    className={`observer-stage-item ${stage.state}`}
                    aria-label={`${stage.label}: ${friendlyStatus(stage.state)}`}
                  >
                    <span>{stage.label}</span>
                    <small>{stage.detail}</small>
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
      label: aggregateTestMean
        ? "Test mean"
        : finalEvaluationReward
          ? "Final reward"
          : evaluationSplit === "test"
            ? "Test solve"
            : evaluationSplit === "validation"
              ? "Validation solve"
              : "Solve rate",
      value: formatRateInterval(
        validationRate,
        validationInterval,
        evaluationExamples,
        "Awaiting evaluation",
      ),
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
