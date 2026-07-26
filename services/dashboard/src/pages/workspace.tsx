/**
 * THESIS: The dashboard is an experiment queue, not a wall of telemetry.
 * OWN-WORLD: Quiet paper surfaces, one oxide interaction color, and broad ruled rows.
 * STORY: Choose an environment, launch a bounded run, then follow collection into proof.
 * FIRST VIEWPORT: A short run queue with one primary action and no provider boilerplate.
 * FORM: Distilled Operate surface; evidence stays available through progressive disclosure.
 */
import { type FormEvent, useEffect, useMemo, useState } from "react";
import { api, useApi } from "../api";
import {
  AsyncState,
  displayRunName,
  formatDate,
  friendlyStatus,
  KeyValue,
  MachineId,
  Notice,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";
import { Link, useNavigate, useParams, useSearchParams } from "../router";
import type {
  ComplexityConfig,
  ComplexityResponse,
  ComplexityState,
  EnvironmentsResponse,
  EnvironmentSpec,
  ResearchComputeExecution,
  RunRolloutTree,
  RunSummary,
} from "../types";

const TERMINAL = new Set(["SUCCEEDED", "FAILED", "CANCELED"]);

const DEFAULT_COMPLEXITY: ComplexityConfig = {
  strategy: "adaptive",
  minimum_level: 0,
  initial_level: 0,
  maximum_level: 12,
  sampling_band: 4,
  mastery_threshold: 0.9,
  evaluation_window: 32,
  promotion_step: 1,
};

export function RunsPage() {
  const runs = useApi<{
    items: RunSummary[];
    research_items: ResearchComputeExecution[];
  }>("/v1/runs", 2_000);
  const visibleRuns =
    runs.data?.research_items.filter(
      (run) => run.execution_id !== "runpod-proof-ui-rehearsal",
    ) ?? [];
  const activeCount = visibleRuns.filter(
    (run) => !TERMINAL.has(run.status),
  ).length;
  const totalCount = visibleRuns.length;

  return (
    <>
      <PageHeader
        title="Runs"
        description={
          runs.data
            ? `${activeCount} active · ${totalCount} total`
            : "Your experiment queue"
        }
      />
      <div className="content workspace-content">
        <AsyncState
          loading={runs.loading}
          error={runs.error}
          empty={!visibleRuns.length}
        >
          {visibleRuns.length ? (
            <section
              className="run-queue research-run-queue"
              aria-label="Training runs"
            >
              <header className="quiet-heading">
                <div>
                  <h2>Training runs</h2>
                  <p>
                    RunPod executions with live learning progress and verified
                    teardown.
                  </p>
                </div>
              </header>
              <div className="run-list">
                {visibleRuns.map((run) => (
                  <ResearchRunRow key={run.execution_id} run={run} />
                ))}
              </div>
            </section>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

function ResearchRunRow({ run }: { run: ResearchComputeExecution }) {
  const update = numberValue(run.progress.update);
  const maximumUpdates = numberValue(run.progress.maximum_updates);
  const percentage =
    update !== null && maximumUpdates !== null && maximumUpdates > 0
      ? Math.min(100, (update / maximumUpdates) * 100)
      : phasePercentage(run.status, stringValue(run.progress.phase));
  const level = numberValue(run.progress.current_level);
  const maximumLevel = numberValue(run.progress.maximum_level);
  const phase = stringValue(run.progress.phase);
  const progressLabel =
    update !== null
      ? `Update ${update}${maximumUpdates !== null ? ` of ${maximumUpdates}` : ""}`
      : phase
        ? friendlyStatus(phase)
        : friendlyStatus(run.status);

  return (
    <Link
      className="run-row research-run-row"
      to={`/runs/research/${run.execution_id}`}
    >
      <span className="run-main">
        <strong>{run.name}</strong>
        <span>{shortModelName(run.model_id)} · RunPod</span>
      </span>
      <StatusBadge status={run.status} />
      <span className="run-progress">
        <span className="progress-track" aria-hidden="true">
          <span style={{ width: `${percentage}%` }} />
        </span>
        <span>{progressLabel}</span>
      </span>
      <span className="run-complexity">
        Adaptive · K={run.branch_width}
        <small>
          {level !== null
            ? `level ${level}${maximumLevel !== null ? ` of ${maximumLevel}` : ""}`
            : "awaiting first evaluation"}
        </small>
      </span>
      <time dateTime={run.updated_at}>{formatDate(run.updated_at)}</time>
    </Link>
  );
}

export function ResearchRunPage() {
  const { executionId = "" } = useParams<{ executionId: string }>();
  const execution = useApi<ResearchComputeExecution>(
    `/v1/research-compute-executions/${executionId}`,
    2_000,
  );
  const run = execution.data;
  const progress = run?.progress ?? {};
  const update = numberValue(progress.update);
  const maximumUpdates = numberValue(progress.maximum_updates);
  const currentLevel = numberValue(progress.current_level);
  const maximumLevel = numberValue(progress.maximum_level);
  const exactRate = numberValue(progress.exact_rate);
  const sampledCompletions = numberValue(progress.sampled_completions);
  const elapsed = numberValue(progress.elapsed_seconds);
  const phase = stringValue(progress.phase);
  const message = stringValue(progress.message);
  const error = stringValue(progress.error);
  const initialExactRate = numberValue(progress.initial_exact_rate);
  const finalExactRate = numberValue(progress.final_exact_rate);
  const rewardGain = numberValue(progress.reward_gain);
  const promotionCount = numberValue(progress.promotion_count);
  const stopReason = stringValue(progress.stop_reason);
  const hypothesisPassed = booleanValue(progress.hypothesis_passed);
  const adapterPersisted = booleanValue(progress.adapter_persisted);

  return (
    <>
      <PageHeader
        eyebrow={<Link to="/runs">Runs</Link>}
        title={run?.name ?? "Research compute"}
        description={
          run
            ? `${shortModelName(run.model_id)} · static K=${run.branch_width} · adaptive complexity`
            : "Loading the operator-side execution record"
        }
        actions={run ? <StatusBadge status={run.status} /> : undefined}
      />
      <div className="content workspace-content">
        <AsyncState loading={execution.loading} error={execution.error}>
          {run ? (
            <div className="research-observer">
              {error ? (
                <Notice tone="negative" title="External run failed">
                  {error}
                </Notice>
              ) : run.status === "SUCCEEDED" ? (
                <Notice
                  title={
                    hypothesisPassed === false
                      ? "Run complete; hypothesis not supported"
                      : "Run complete"
                  }
                >
                  {hypothesisPassed === false
                    ? "The execution and teardown succeeded, but held-out exact accuracy did not improve."
                    : "The result is persisted and RunPod teardown is confirmed."}
                </Notice>
              ) : (
                <Notice title="Live external execution">
                  {message ??
                    "The operator harness is publishing progress from RunPod."}
                </Notice>
              )}

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
                <Section title="Current progress">
                  <KeyValue
                    items={[
                      {
                        label: "Phase",
                        value:
                          run.status === "SUCCEEDED"
                            ? "Complete"
                            : phase
                              ? friendlyStatus(phase)
                              : friendlyStatus(run.status),
                      },
                      {
                        label: "Training",
                        value:
                          update !== null
                            ? `${update}${maximumUpdates !== null ? ` / ${maximumUpdates}` : ""} updates`
                            : "Not started",
                      },
                      {
                        label: "Curriculum",
                        value:
                          currentLevel !== null
                            ? `Level ${currentLevel}${maximumLevel !== null ? ` of ${maximumLevel}` : ""}`
                            : "Awaiting evaluation",
                      },
                      {
                        label: "Held-out exact",
                        value:
                          exactRate !== null
                            ? `${(exactRate * 100).toFixed(1)}%`
                            : "Awaiting evaluation",
                      },
                      {
                        label: "Completions",
                        value:
                          sampledCompletions !== null
                            ? sampledCompletions.toLocaleString()
                            : "0",
                      },
                      {
                        label: "Elapsed",
                        value: elapsed !== null ? formatDuration(elapsed) : "—",
                      },
                    ]}
                  />
                </Section>
                <Section title="Allocation">
                  <KeyValue
                    items={[
                      {
                        label: "GPU",
                        value:
                          stringValue(run.resource_profile.gpu_id) ??
                          "Awaiting allocation",
                      },
                      {
                        label: "Cloud",
                        value:
                          stringValue(run.resource_profile.cloud_type) ??
                          "Awaiting allocation",
                      },
                      {
                        label: "Hourly rate",
                        value:
                          numberValue(run.resource_profile.hourly_cost_usd) !==
                          null
                            ? `$${numberValue(
                                run.resource_profile.hourly_cost_usd,
                              )?.toFixed(3)}/hour`
                            : "Not reported",
                      },
                      {
                        label: "Spending cap",
                        value:
                          numberValue(
                            run.resource_profile.maximum_hourly_cost_usd,
                          ) !== null
                            ? `$${numberValue(
                                run.resource_profile.maximum_hourly_cost_usd,
                              )?.toFixed(2)}/hour`
                            : "Not reported",
                      },
                      {
                        label: "Provider handle",
                        value: run.provider_handle ? (
                          <MachineId value={run.provider_handle} />
                        ) : (
                          "Not allocated"
                        ),
                      },
                      {
                        label: "Started",
                        value: formatDate(run.started_at),
                      },
                      {
                        label: "Teardown",
                        value: run.teardown_confirmed
                          ? "Confirmed"
                          : run.status === "FINALIZING"
                            ? "In progress"
                            : "Pending",
                      },
                    ]}
                  />
                </Section>
              </div>

              {run.status === "SUCCEEDED" ? (
                <Section title="Learning outcome">
                  <KeyValue
                    items={[
                      {
                        label: "Hypothesis",
                        value:
                          hypothesisPassed === null
                            ? "Not reported"
                            : hypothesisPassed
                              ? "Supported"
                              : "Not supported",
                      },
                      {
                        label: "Initial exact",
                        value:
                          initialExactRate !== null
                            ? `${(initialExactRate * 100).toFixed(1)}%`
                            : "Not reported",
                      },
                      {
                        label: "Final exact",
                        value:
                          finalExactRate !== null
                            ? `${(finalExactRate * 100).toFixed(1)}%`
                            : exactRate !== null
                              ? `${(exactRate * 100).toFixed(1)}%`
                              : "Not reported",
                      },
                      {
                        label: "Exact gain",
                        value:
                          rewardGain !== null
                            ? `${rewardGain >= 0 ? "+" : ""}${(
                                rewardGain * 100
                              ).toFixed(1)} pts`
                            : "Not reported",
                      },
                      {
                        label: "Promotions",
                        value:
                          promotionCount !== null
                            ? promotionCount.toLocaleString()
                            : "0",
                      },
                      {
                        label: "Stop reason",
                        value: stopReason
                          ? friendlyStatus(stopReason)
                          : "Not reported",
                      },
                      {
                        label: "Adapter",
                        value:
                          adapterPersisted === null
                            ? "Not reported"
                            : adapterPersisted
                              ? "Persisted"
                              : "Not persisted",
                      },
                    ]}
                  />
                </Section>
              ) : null}

              <Section
                title="Run identity"
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
                        <MachineId value={run.proof_id} />
                      ) : (
                        "Recorded after teardown"
                      ),
                    },
                    {
                      label: "Last heartbeat",
                      value: formatDate(run.updated_at),
                    },
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

function observerStages(
  status: ResearchComputeExecution["status"],
  phase: string | null,
  teardownConfirmed: boolean,
): Array<{ label: string; detail: string; state: string }> {
  const learningPhases = ["training", "evaluation", "baseline_evaluation"];
  const phaseIndex =
    status === "SUCCEEDED"
      ? 3
      : status === "FINALIZING"
        ? 2
        : phase && learningPhases.includes(phase)
          ? 1
          : 0;
  const failedIndex =
    phase === "finalizing"
      ? 2
      : phase && learningPhases.includes(phase)
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
          ? index === 3
            ? teardownConfirmed
              ? "complete"
              : "failed"
            : index < failedIndex
              ? "complete"
              : index === failedIndex
                ? "failed"
                : "pending"
          : index < phaseIndex
            ? "complete"
            : index === phaseIndex
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
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

export function NewRunPage() {
  const navigate = useNavigate();
  const [launchParams] = useSearchParams();
  const catalog = useApi<EnvironmentsResponse>("/v1/environments");
  const [environmentId, setEnvironmentId] = useState(
    launchParams.get("environment") ?? "cad.reconstruction",
  );
  const [algorithm, setAlgorithm] =
    useState<RunSummary["algorithm"]>("bpo_local_metric");
  const [name, setName] = useState("SQLite repair exploration");
  const [seed, setSeed] = useState(17);
  const [complexity, setComplexity] =
    useState<ComplexityConfig>(DEFAULT_COMPLEXITY);
  const [catalogInitialized, setCatalogInitialized] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const environment = catalog.data?.items.find(
    (item) => item.environment_id === environmentId,
  );
  const branchWidth = algorithm === "bpo_local_metric" ? 4 : 1;
  const preview = useMemo(
    () => ({
      environment: environmentId,
      task_revision: environment?.task_revision,
      algorithm,
      branch: { mode: "static", width: branchWidth },
      complexity,
      seed,
    }),
    [algorithm, branchWidth, complexity, environment, environmentId, seed],
  );

  useEffect(() => {
    if (!environment || catalogInitialized) return;
    setName(`${environment.short_name} exploration`);
    setComplexity({
      ...DEFAULT_COMPLEXITY,
      initial_level: environment.complexity.default_initial_level,
      maximum_level: environment.complexity.default_max_level,
    });
    setCatalogInitialized(true);
  }, [catalogInitialized, environment]);

  function chooseEnvironment(next: EnvironmentSpec) {
    setEnvironmentId(next.environment_id);
    setName(`${next.short_name} exploration`);
    setComplexity({
      ...DEFAULT_COMPLEXITY,
      initial_level: next.complexity.default_initial_level,
      maximum_level: next.complexity.default_max_level,
    });
  }

  function setComplexityNumber(key: keyof ComplexityConfig, value: number) {
    setComplexity((current) => ({ ...current, [key]: value }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!environment?.launch_enabled) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await api<{ run_id: string }>("/v1/runs", {
        method: "POST",
        body: JSON.stringify({
          name,
          algorithm,
          environment_id: environment.environment_id,
          policy_compute_provider: "MockRunPodProvider",
          judge_provider: "MockJudgeProvider",
          task_revision: environment.task_revision,
          branch: {
            mode: "static",
            width: branchWidth,
            decision_after_actions: 3,
            rng_mode: "split_stream",
          },
          complexity,
          budgets: {
            transitions: 64,
            environment_cpu_seconds: 600,
            render_cpu_seconds: 300,
            judge_input_tokens: 100000,
          },
          seed,
          retention_class: "local-research",
        }),
      });
      navigate(`/runs/${result.run_id}`);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Run could not be launched.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <PageHeader
        eyebrow={<Link to="/runs">Runs</Link>}
        title="New run"
        description="Choose the environment first. Equinox resolves the rest."
      />
      <form className="content launch-workspace" onSubmit={submit}>
        <div className="launch-primary">
          {error ? (
            <Notice tone="negative" title="Run could not be launched">
              {error}
            </Notice>
          ) : null}
          <section className="form-group" aria-labelledby="environment-label">
            <div className="form-group-heading">
              <div>
                <h2 id="environment-label">Environment</h2>
                <p>Each environment owns its task generator and verifier.</p>
              </div>
              <Link to="/environments">Compare environments</Link>
            </div>
            <AsyncState loading={catalog.loading} error={catalog.error}>
              <div className="environment-choices">
                {catalog.data?.items.map((item) => (
                  <label
                    key={item.environment_id}
                    className={
                      item.environment_id === environmentId
                        ? "environment-choice selected"
                        : "environment-choice"
                    }
                  >
                    <input
                      type="radio"
                      name="environment"
                      value={item.environment_id}
                      checked={item.environment_id === environmentId}
                      onChange={() => chooseEnvironment(item)}
                    />
                    <span>
                      <strong>{item.name}</strong>
                      <small>{item.summary}</small>
                    </span>
                    <StatusBadge
                      status={
                        item.launch_enabled
                          ? "LOCAL_READY"
                          : "CONFIGURATION_DRAFT"
                      }
                    />
                  </label>
                ))}
              </div>
            </AsyncState>
          </section>

          <section className="form-group" aria-labelledby="run-setup-label">
            <div className="form-group-heading">
              <div>
                <h2 id="run-setup-label">Run setup</h2>
                <p>
                  A concise identity and a reproducible collection strategy.
                </p>
              </div>
            </div>
            <div className="field-grid">
              <label className="field">
                <span>Run name</span>
                <input
                  value={name}
                  minLength={3}
                  maxLength={80}
                  onChange={(event) => setName(event.target.value)}
                  required
                />
              </label>
              <label className="field">
                <span>Seed</span>
                <input
                  type="number"
                  min={0}
                  max={2147483647}
                  value={seed}
                  onChange={(event) =>
                    setSeed(
                      Number.isNaN(event.currentTarget.valueAsNumber)
                        ? 0
                        : event.currentTarget.valueAsNumber,
                    )
                  }
                />
              </label>
            </div>
            <fieldset className="inline-choice">
              <legend>Collection</legend>
              <label>
                <input
                  type="radio"
                  name="algorithm"
                  checked={algorithm === "bpo_local_metric"}
                  onChange={() => setAlgorithm("bpo_local_metric")}
                />
                <span>
                  <strong>Branch-aware</strong>
                  Static K=4
                </span>
              </label>
              <label>
                <input
                  type="radio"
                  name="algorithm"
                  checked={algorithm === "independent_rollout_baseline"}
                  onChange={() => setAlgorithm("independent_rollout_baseline")}
                />
                <span>
                  <strong>Independent baseline</strong>
                  Four complete rollouts
                </span>
              </label>
            </fieldset>
          </section>

          <section className="form-group" aria-labelledby="complexity-label">
            <div className="form-group-heading">
              <div>
                <h2 id="complexity-label">Adaptive difficulty</h2>
                <p>
                  Keep tasks within a rolling capability band and promote after
                  demonstrated mastery.
                </p>
              </div>
              <StatusBadge status="REQUIRED" />
            </div>
            <div className="complexity-summary">
              <span>
                Start <strong>{complexity.initial_level}</strong>
              </span>
              <span aria-hidden="true">→</span>
              <span>
                Maximum <strong>{complexity.maximum_level}</strong>
              </span>
              <span className="complexity-rule">
                Promote at{" "}
                <strong>
                  {Math.round(complexity.mastery_threshold * 100)}%
                </strong>{" "}
                over <strong>{complexity.evaluation_window}</strong> outcomes
              </span>
            </div>
            <details className="advanced-settings">
              <summary>Adjust curriculum</summary>
              <div className="field-grid compact">
                <label className="field">
                  <span>Starting level</span>
                  <input
                    type="number"
                    min={complexity.minimum_level}
                    max={complexity.maximum_level}
                    value={complexity.initial_level}
                    onChange={(event) =>
                      setComplexityNumber(
                        "initial_level",
                        event.currentTarget.valueAsNumber,
                      )
                    }
                  />
                </label>
                <label className="field">
                  <span>Maximum level</span>
                  <input
                    type="number"
                    min={complexity.initial_level}
                    max={100}
                    value={complexity.maximum_level}
                    onChange={(event) =>
                      setComplexityNumber(
                        "maximum_level",
                        event.currentTarget.valueAsNumber,
                      )
                    }
                  />
                </label>
                <label className="field">
                  <span>Sampling band</span>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={complexity.sampling_band}
                    onChange={(event) =>
                      setComplexityNumber(
                        "sampling_band",
                        event.currentTarget.valueAsNumber,
                      )
                    }
                  />
                </label>
                <label className="field">
                  <span>Evaluation window</span>
                  <input
                    type="number"
                    min={8}
                    max={1024}
                    value={complexity.evaluation_window}
                    onChange={(event) =>
                      setComplexityNumber(
                        "evaluation_window",
                        event.currentTarget.valueAsNumber,
                      )
                    }
                  />
                </label>
              </div>
            </details>
          </section>
        </div>

        <aside className="launch-summary">
          <span className="summary-kicker">
            {environment?.launch_enabled
              ? "Ready to launch"
              : "Configuration draft"}
          </span>
          <h2>{environment?.short_name ?? "Environment"}</h2>
          <dl>
            <div>
              <dt>Collection</dt>
              <dd>{algorithmLabel(algorithm)}</dd>
            </div>
            <div>
              <dt>Sampling</dt>
              <dd>
                {algorithm === "bpo_local_metric"
                  ? `Static K=${branchWidth}`
                  : "No branching · 4 independent rollouts"}
              </dd>
            </div>
            <div>
              <dt>Difficulty</dt>
              <dd>
                Adaptive · {complexity.initial_level}→{complexity.maximum_level}
              </dd>
            </div>
            <div>
              <dt>Compute</dt>
              <dd>
                {environment?.launch_enabled
                  ? "Local mock provider"
                  : "RunPool adapter pending"}
              </dd>
            </div>
          </dl>
          {!environment?.launch_enabled ? (
            <p className="adapter-note">
              This configuration is a draft. Launching is held until the RunPool
              image, action protocol, snapshot, and verifier pass conformance.
            </p>
          ) : null}
          <button
            className="button primary launch-button"
            disabled={submitting || !environment?.launch_enabled}
          >
            {submitting
              ? "Launching…"
              : environment?.launch_enabled
                ? "Launch run"
                : "RunPool adapter required"}
          </button>
          <details className="manifest-disclosure">
            <summary>Manifest preview</summary>
            <pre>{JSON.stringify(preview, null, 2)}</pre>
          </details>
        </aside>
      </form>
    </>
  );
}

interface MetricObservation {
  descriptor: string;
  value: unknown;
  unit: string;
  subject_id: string;
  created_at: string;
}

interface RewardSignal {
  reward_signal_id: string;
  subject_type: string;
  subject_id: string;
  name: string;
  value: number;
  reward_pipeline_id: string;
  metric_observation_ids: string[];
  created_at: string;
}

interface RunDetailResponse {
  run: RunSummary;
  attempts: Array<Record<string, any>>;
  allocations: Array<Record<string, any>>;
  policy_versions: Array<Record<string, any>>;
  metrics: MetricObservation[];
  reward_signals: RewardSignal[];
  failures: Array<Record<string, any>>;
  providers: Record<string, string>;
  calibration: { status: string; message: string };
}

interface RunEvent {
  cursor: number;
  event_id: string;
  event_type: string;
  aggregate_type: string;
  aggregate_id: string;
  producer: string;
  payload: Record<string, unknown>;
  occurred_at: string;
}

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const view = searchParams.get("view") ?? "overview";
  const detail = useApi<RunDetailResponse>(`/v1/runs/${runId}`, 1_000);
  const complexity = useApi<ComplexityResponse>(
    `/v1/runs/${runId}/complexity`,
    2_000,
  );
  const batches = useApi<{ items: Array<Record<string, any>> }>(
    `/v1/runs/${runId}/collection-batches`,
    1_000,
  );
  const iterations = useApi<{ items: Array<Record<string, any>> }>(
    `/v1/runs/${runId}/iterations`,
    1_000,
  );
  const events = useApi<{ items: RunEvent[] }>(
    `/v1/runs/${runId}/events`,
    1_000,
  );
  const trees = useApi<{ items: RunRolloutTree[] }>(
    `/v1/runs/${runId}/rollout-trees`,
    1_000,
  );
  const [actionError, setActionError] = useState<string | null>(null);
  const run = detail.data?.run;

  function selectView(next: string) {
    const params = new URLSearchParams(searchParams);
    if (next === "overview") params.delete("view");
    else params.set("view", next);
    setSearchParams(params);
  }

  async function reproduce() {
    try {
      const result = await api<{ run_id: string }>(
        `/v1/runs/${runId}/reproduce`,
        { method: "POST" },
      );
      navigate(`/runs/${result.run_id}`);
    } catch (cause) {
      setActionError(
        cause instanceof Error ? cause.message : "Reproduction failed.",
      );
    }
  }

  async function cancel() {
    try {
      await api(`/v1/runs/${runId}/cancel`, {
        method: "POST",
        body: JSON.stringify({
          mode: "terminate",
          operation_id: crypto.randomUUID(),
        }),
      });
    } catch (cause) {
      setActionError(
        cause instanceof Error ? cause.message : "Cancellation failed.",
      );
    }
  }

  return (
    <>
      <PageHeader
        eyebrow={<Link to="/runs">Runs</Link>}
        title={run ? displayRunName(run.name) : "Run"}
        description={
          run
            ? `${environmentLabel(run.manifest.environment?.id)} · ${algorithmLabel(run.algorithm)}`
            : "Loading run…"
        }
        actions={
          run ? (
            <>
              <StatusBadge status={run.status} />
              <button
                className="button secondary"
                type="button"
                onClick={reproduce}
              >
                Reproduce
              </button>
              {!TERMINAL.has(run.status) ? (
                <button
                  className="button danger"
                  type="button"
                  onClick={cancel}
                >
                  Cancel
                </button>
              ) : null}
            </>
          ) : null
        }
      />
      <div className="run-tabs" role="tablist" aria-label="Run sections">
        {["overview", "trace", "evidence", "operations"].map((item) => (
          <button
            key={item}
            type="button"
            role="tab"
            aria-selected={view === item}
            onClick={() => selectView(item)}
          >
            {item.charAt(0).toUpperCase() + item.slice(1)}
          </button>
        ))}
      </div>
      <div className="content workspace-content">
        <AsyncState loading={detail.loading} error={detail.error}>
          {actionError ? (
            <Notice tone="negative" title="Run action failed">
              {actionError}
            </Notice>
          ) : null}
          {detail.data && view === "overview" ? (
            <RunOverview
              detail={detail.data}
              complexity={complexity.data}
              complexityLoading={complexity.loading}
              complexityError={complexity.error}
              batches={batches.data?.items ?? []}
              batchesLoading={batches.loading}
              batchesError={batches.error}
              iterations={iterations.data?.items ?? []}
              iterationsLoading={iterations.loading}
              iterationsError={iterations.error}
            />
          ) : null}
          {detail.data && view === "evidence" ? (
            <RunEvidence detail={detail.data} />
          ) : null}
          {detail.data && view === "trace" ? (
            <RunTrace
              run={detail.data.run}
              trees={trees.data?.items ?? []}
              loading={trees.loading}
              error={trees.error}
            />
          ) : null}
          {detail.data && view === "operations" ? (
            <RunOperations
              detail={detail.data}
              events={events.data?.items ?? []}
              eventsLoading={events.loading}
              eventsError={events.error}
            />
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

function RunOverview({
  detail,
  complexity,
  complexityLoading,
  complexityError,
  batches,
  batchesLoading,
  batchesError,
  iterations,
  iterationsLoading,
  iterationsError,
}: {
  detail: RunDetailResponse;
  complexity: ComplexityResponse | null;
  complexityLoading: boolean;
  complexityError: Error | null;
  batches: Array<Record<string, any>>;
  batchesLoading: boolean;
  batchesError: Error | null;
  iterations: Array<Record<string, any>>;
  iterationsLoading: boolean;
  iterationsError: Error | null;
}) {
  const activeComplexity: ComplexityState | null =
    complexity?.available === false ? null : complexity;
  const legacyComplexity = complexity?.available === false;
  const stage =
    detail.run.status === "SUCCEEDED"
      ? 4
      : iterations.length
        ? 3
        : detail.metrics.length
          ? 2
          : batches.length
            ? 1
            : 0;
  return (
    <>
      <section className="run-hero">
        <div className="stage-track" aria-label={`Run stage ${stage} of 4`}>
          {["Setup", "Collect", "Verify", "Update"].map((label, index) => (
            <span
              key={label}
              className={
                index < stage ? "complete" : index === stage ? "current" : ""
              }
            >
              <i aria-hidden="true" />
              {label}
            </span>
          ))}
        </div>
        <div className="run-next">
          <span>Current outcome</span>
          <strong>
            {detail.run.status === "SUCCEEDED"
              ? "Evidence collected and policy update committed"
              : detail.run.status.toLowerCase().replaceAll("_", " ")}
          </strong>
          {batches.length ? (
            <Link to={`/runs/${detail.run.run_id}?view=trace`}>
              Follow every action →
            </Link>
          ) : null}
        </div>
      </section>

      <section className="complexity-panel">
        <div>
          <span className="summary-kicker">Adaptive difficulty</span>
          <h2>
            {activeComplexity
              ? `Level ${activeComplexity.current_level}`
              : complexityLoading
                ? "Reading curriculum state…"
                : legacyComplexity
                  ? "Legacy run"
                  : "Complexity unavailable"}
          </h2>
          <p>
            {activeComplexity
              ? `Sampling levels ${activeComplexity.active_range[0]}–${activeComplexity.active_range[1]}. Promote at ${Math.round(activeComplexity.mastery_threshold * 100)}% over ${activeComplexity.evaluation_window} outcomes.`
              : legacyComplexity
                ? "This run predates persisted curriculum state. Reproduce it to start from the current adaptive difficulty contract."
                : complexityError
                  ? `The curriculum state could not be loaded: ${complexityError.message}`
                  : "Waiting for the persisted curriculum state."}
          </p>
        </div>
        {activeComplexity ? (
          <div className="complexity-meter">
            <span>
              {activeComplexity.window_progress.attempts}/
              {activeComplexity.window_progress.required} observed
            </span>
            <span className="progress-track">
              <span
                style={{
                  width: `${Math.min(
                    100,
                    (activeComplexity.window_progress.attempts /
                      activeComplexity.window_progress.required) *
                      100,
                  )}%`,
                }}
              />
            </span>
            <small>{activeComplexity.promotion_count} promotions</small>
          </div>
        ) : null}
      </section>

      <div className="overview-split">
        <section className="plain-section">
          <div className="quiet-heading">
            <div>
              <h2>Collection</h2>
              <p>
                {batchesError
                  ? "Unavailable"
                  : batchesLoading
                    ? "Loading…"
                    : `${batches.length} persisted ${batches.length === 1 ? "batch" : "batches"}`}
              </p>
            </div>
          </div>
          {batchesError ? (
            <p className="muted-block">
              Collection data could not be loaded: {batchesError.message}
            </p>
          ) : batchesLoading ? (
            <p className="muted-block">Reading collection batches…</p>
          ) : batches.length ? (
            <div className="compact-list">
              {batches.map((batch) => (
                <div key={batch.collection_batch_id}>
                  <span>
                    <strong>
                      {batch.tree_count} rollout{" "}
                      {Number(batch.tree_count) === 1 ? "tree" : "trees"}
                    </strong>
                    <MachineId value={batch.collection_batch_id} copy={false} />
                  </span>
                  <StatusBadge status={batch.status} />
                </div>
              ))}
            </div>
          ) : (
            <p className="muted-block">Collection has not started.</p>
          )}
        </section>
        <section className="plain-section">
          <div className="quiet-heading">
            <div>
              <h2>Training updates</h2>
              <p>
                {iterationsError
                  ? "Unavailable"
                  : iterationsLoading
                    ? "Loading…"
                    : `${iterations.length} committed ${iterations.length === 1 ? "iteration" : "iterations"}`}
              </p>
            </div>
          </div>
          {iterationsError ? (
            <p className="muted-block">
              Training updates could not be loaded: {iterationsError.message}
            </p>
          ) : iterationsLoading ? (
            <p className="muted-block">Reading training updates…</p>
          ) : iterations.length ? (
            <div className="compact-list">
              {iterations.map((iteration) => (
                <Link
                  key={iteration.training_iteration_id}
                  to={`/runs/${detail.run.run_id}/iterations/${iteration.training_iteration_id}`}
                >
                  <span>
                    <strong>
                      {iteration.tree_count} trees · {iteration.excluded_count}{" "}
                      excluded
                    </strong>
                    <MachineId
                      value={iteration.training_iteration_id}
                      copy={false}
                    />
                  </span>
                  <StatusBadge status={iteration.status} />
                </Link>
              ))}
            </div>
          ) : (
            <p className="muted-block">No policy update has been committed.</p>
          )}
        </section>
      </div>
    </>
  );
}

function RunTrace({
  run,
  trees,
  loading,
  error,
}: {
  run: RunSummary;
  trees: RunRolloutTree[];
  loading: boolean;
  error: Error | null;
}) {
  const actionCount = trees.reduce(
    (total, tree) => total + tree.transition_count,
    0,
  );
  const siblingCount = trees.reduce(
    (total, tree) => total + tree.sibling_count,
    0,
  );
  const exceptions = trees.reduce(
    (total, tree) => total + tree.exception_count,
    0,
  );
  const branchAware = run.algorithm === "bpo_local_metric";
  return (
    <>
      <section className="trace-intro">
        <div>
          <span className="summary-kicker">Run trace</span>
          <h2>
            {loading
              ? "Reading trajectories…"
              : error
                ? "Trace unavailable"
                : `${actionCount} actions across ${trees.length} ${trees.length === 1 ? "trajectory" : "trajectories"}`}
          </h2>
          <p>
            {branchAware
              ? "The shared prefix is executed once, then the checkpoint fans out into four isolated sibling paths."
              : "Each rollout is independent, with no shared decision checkpoint."}
          </p>
        </div>
        {!loading && !error && trees.length ? (
          <dl className="trace-totals">
            <div>
              <dt>Static width</dt>
              <dd>{branchAware ? "K=4" : "K=1"}</dd>
            </div>
            <div>
              <dt>Sibling paths</dt>
              <dd>{siblingCount}</dd>
            </div>
            <div>
              <dt>Exceptions</dt>
              <dd>{exceptions}</dd>
            </div>
          </dl>
        ) : null}
      </section>
      <div className="trace-grammar" aria-label="How to read a rollout trace">
        <span>Start</span>
        <i aria-hidden="true" />
        <span>Shared actions</span>
        <i aria-hidden="true" />
        <strong>Checkpoint</strong>
        <i className="fan" aria-hidden="true" />
        <span>{branchAware ? "4 sibling paths" : "Independent path"}</span>
        <i aria-hidden="true" />
        <span>Verification</span>
      </div>
      <section className="plain-section trace-list-section">
        <div className="quiet-heading">
          <div>
            <h2>Collected trajectories</h2>
            <p>
              Select one to inspect every state, action, outcome, and proof.
            </p>
          </div>
        </div>
        {error ? (
          <p className="muted-block">
            The rollout trace could not be loaded: {error.message}
          </p>
        ) : loading ? (
          <p className="muted-block">Reading rollout trees…</p>
        ) : trees.length ? (
          <div className="trace-list">
            {trees.map((tree, index) => (
              <Link
                key={tree.rollout_tree_id}
                className="trace-row"
                to={`/rollout-trees/${tree.rollout_tree_id}`}
              >
                <span className="trace-index" aria-hidden="true">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className="trace-main">
                  <strong>
                    {tree.transition_count} actions · {tree.state_count} states
                  </strong>
                  <MachineId value={tree.rollout_tree_id} copy={false} />
                </span>
                <span>
                  {tree.sibling_count
                    ? `${tree.sibling_count} siblings`
                    : "Independent"}
                  <small>
                    {tree.excluded_count} excluded · {tree.exception_count}{" "}
                    exceptions
                  </small>
                </span>
                <StatusBadge status={tree.status} />
                <span className="trace-open">Open explorer →</span>
              </Link>
            ))}
          </div>
        ) : (
          <p className="muted-block">
            No trajectories have been persisted for this run yet.
          </p>
        )}
      </section>
    </>
  );
}

function RunEvidence({ detail }: { detail: RunDetailResponse }) {
  const metrics = latestByName(detail.metrics, (item) => item.descriptor);
  const rewards = latestByName(detail.reward_signals, (item) => item.name);
  return (
    <>
      <Notice tone="warning" title="Model assessment boundary">
        {detail.calibration.message}
      </Notice>
      <section className="plain-section">
        <div className="quiet-heading">
          <div>
            <h2>Latest evidence</h2>
            <p>Deterministic facts remain separate from model assessments.</p>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Authority</th>
                <th>Observation</th>
                <th>Value</th>
                <th>Recorded</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((metric) => (
                <tr key={metric.descriptor}>
                  <td>
                    <StatusBadge
                      status={
                        metric.descriptor.startsWith("judge.")
                          ? "MODEL"
                          : "DETERMINISTIC"
                      }
                    />
                  </td>
                  <td>{metric.descriptor.replaceAll(".", " / ")}</td>
                  <td>
                    <strong>{observationValue(metric.value)}</strong>{" "}
                    <small>{metric.unit}</small>
                  </td>
                  <td>{formatDate(metric.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="plain-section">
        <div className="quiet-heading">
          <div>
            <h2>Reward signals</h2>
            <p>Named transforms admitted to training.</p>
          </div>
        </div>
        <div className="compact-list rewards">
          {rewards.map((reward) => (
            <div key={reward.name}>
              <span>
                <strong>{reward.name.replaceAll("_", " ")}</strong>
                <small>
                  {reward.metric_observation_ids.length} source facts
                </small>
              </span>
              <strong>{reward.value.toFixed(4)}</strong>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

function RunOperations({
  detail,
  events,
  eventsLoading,
  eventsError,
}: {
  detail: RunDetailResponse;
  events: RunEvent[];
  eventsLoading: boolean;
  eventsError: Error | null;
}) {
  return (
    <>
      {detail.failures.length ? (
        <Notice tone="negative" title="Run failure">
          <pre>{JSON.stringify(detail.failures, null, 2)}</pre>
        </Notice>
      ) : null}
      <section className="plain-section">
        <div className="quiet-heading">
          <div>
            <h2>Attempts</h2>
            <p>Operational retries do not duplicate scientific facts.</p>
          </div>
        </div>
        <div className="compact-list">
          {detail.attempts.map((attempt) => (
            <div key={attempt.attempt_id}>
              <span>
                <strong>Attempt {attempt.attempt_number}</strong>
                <MachineId value={attempt.attempt_id} copy={false} />
              </span>
              <StatusBadge status={attempt.status} />
            </div>
          ))}
        </div>
      </section>
      <section className="plain-section">
        <div className="quiet-heading">
          <div>
            <h2>Recent events</h2>
            <p>The latest durable lifecycle changes.</p>
          </div>
        </div>
        <div className="event-list">
          {eventsError ? (
            <p className="muted-block">
              Events could not be loaded: {eventsError.message}
            </p>
          ) : eventsLoading ? (
            <p className="muted-block">Reading durable events…</p>
          ) : (
            events
              .slice(-12)
              .reverse()
              .map((event) => (
                <details key={event.event_id}>
                  <summary>
                    <span>
                      <strong>{event.event_type.replaceAll(".", " / ")}</strong>
                      <small>
                        {event.producer} · {formatDate(event.occurred_at)}
                      </small>
                    </span>
                    <span>#{event.cursor}</span>
                  </summary>
                  <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                </details>
              ))
          )}
        </div>
      </section>
      <details className="provenance-disclosure">
        <summary>Providers and provenance</summary>
        <dl>
          {Object.entries(detail.providers).map(([label, value]) => (
            <div key={label}>
              <dt>{label.replace("_", " ")}</dt>
              <dd>{value}</dd>
            </div>
          ))}
          <div>
            <dt>Manifest</dt>
            <dd>
              <MachineId value={detail.run.manifest_digest} />
            </dd>
          </div>
        </dl>
      </details>
    </>
  );
}

function latestByName<T extends { created_at: string }>(
  items: T[],
  key: (item: T) => string,
): T[] {
  const latest = new Map<string, T>();
  for (const item of items) {
    const previous = latest.get(key(item));
    if (!previous || item.created_at > previous.created_at)
      latest.set(key(item), item);
  }
  return [...latest.values()].sort((left, right) =>
    key(left).localeCompare(key(right)),
  );
}

function observationValue(value: unknown): string {
  if (typeof value === "number")
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function environmentLabel(id?: string) {
  if (!id || id === "cad.reconstruction") return "CAD reconstruction";
  if (id === "sqlite.repair") return "SQLite repair";
  if (id === "cli.debug") return "CLI debugging";
  if (id === "repository.repair") return "Repository repair";
  return id;
}

function algorithmLabel(algorithm: RunSummary["algorithm"]) {
  return algorithm === "bpo_local_metric"
    ? "Branch-aware · K=4"
    : "Independent baseline";
}
