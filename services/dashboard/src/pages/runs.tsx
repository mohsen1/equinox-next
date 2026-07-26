import { FormEvent, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "../router";
import { api, useApi } from "../api";
import {
  AsyncState,
  formatDate,
  KeyValue,
  MachineId,
  Notice,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";
import type { RunSummary } from "../types";

export function RunsPage() {
  const { data, error, loading, setData } = useApi<{ items: RunSummary[] }>(
    "/v1/runs",
    2_000,
  );
  const [seeding, setSeeding] = useState(false);
  const [seedError, setSeedError] = useState<string | null>(null);

  async function seed() {
    setSeeding(true);
    setSeedError(null);
    try {
      await api("/internal/seed", { method: "POST" });
      const runs = await api<{ items: RunSummary[] }>("/v1/runs");
      setData(runs);
    } catch (cause) {
      setSeedError(
        cause instanceof Error
          ? cause.message
          : "CAD proofs could not be seeded.",
      );
    } finally {
      setSeeding(false);
    }
  }

  return (
    <>
      <PageHeader
        eyebrow="Research operations"
        title="Runs"
        description="Persisted local experiments, exact provider boundaries, and evidence health."
        actions={
          <>
            <button
              className="button secondary"
              type="button"
              onClick={seed}
              disabled={seeding}
            >
              {seeding ? "Seeding…" : "Seed CAD proofs"}
            </button>
            <Link className="button primary" to="/runs/new">
              Launch run
            </Link>
          </>
        }
      />
      <div className="content">
        <Notice title="Local provider boundary">
          Policy compute resolves <code>MockRunPodProvider</code>; model
          assessments resolve <code>MockJudgeProvider</code>. External capacity
          is disabled.
        </Notice>
        {seedError ? (
          <Notice tone="negative" title="Seed failed">
            {seedError}
          </Notice>
        ) : null}
        <AsyncState loading={loading} error={error} empty={!data?.items.length}>
          <Section
            title="Active and recent runs"
            aside={<span>{data?.items.length ?? 0} persisted</span>}
          >
            <div className="table-wrap">
              <table className="data-table runs-table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Lifecycle</th>
                    <th>Algorithm</th>
                    <th>Evidence</th>
                    <th>Exceptions</th>
                    <th>Local cost</th>
                    <th>Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.items.map((run) => (
                    <tr key={run.run_id}>
                      <td>
                        <Link className="row-link" to={`/runs/${run.run_id}`}>
                          <strong>{run.name}</strong>
                          <MachineId value={run.run_id} copy={false} />
                        </Link>
                      </td>
                      <td>
                        <StatusBadge status={run.status} />
                      </td>
                      <td>
                        <span>{algorithmLabel(run.algorithm)}</span>
                        <small>
                          {run.manifest.environment?.snapshot_fidelity ??
                            "not recorded"}
                        </small>
                      </td>
                      <td>
                        <span>
                          {run.rollout_tree_count} trees ·{" "}
                          {run.verification_run_count} verifications
                        </span>
                        <small>{run.iteration_count} iteration</small>
                      </td>
                      <td>
                        <span>
                          {run.retry_count} retries · {run.abstention_count}{" "}
                          abstentions
                        </span>
                        <small>typed outcomes</small>
                      </td>
                      <td>
                        <span>
                          {run.cost.execution_credits.toFixed(3)} credits
                        </span>
                        <small>judge: 0 external</small>
                      </td>
                      <td>{formatDate(run.updated_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </AsyncState>
      </div>
    </>
  );
}

function algorithmLabel(algorithm: RunSummary["algorithm"]) {
  return algorithm === "bpo_local_metric"
    ? "BPO local metric"
    : "Independent baseline";
}

export function NewRunPage() {
  const navigate = useNavigate();
  const [algorithm, setAlgorithm] =
    useState<RunSummary["algorithm"]>("bpo_local_metric");
  const [name, setName] = useState("Branch-aware CAD proof");
  const [seed, setSeed] = useState(17);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const branchWidth = algorithm === "bpo_local_metric" ? 4 : 1;
  const preview = useMemo(
    () => ({
      profile: "local-contract-proof",
      environment: "cad.reconstruction@1.0.0",
      task_revision: "mounting-plate@sha256:fixture-v1",
      algorithm,
      policy_compute_provider: "MockRunPodProvider",
      execution_provider: "ComposeExecutionProvider",
      snapshot_fidelity: "logical_restore",
      judge_provider: "MockJudgeProvider",
      judge_spec: "cad.pointwise.mock@1",
      branch: {
        width: branchWidth,
        decision_after_actions: 3,
        rng_mode: "split_stream",
      },
      seed,
      credentials_required: [],
    }),
    [algorithm, branchWidth, seed],
  );

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const result = await api<{ run_id: string }>("/v1/runs", {
        method: "POST",
        body: JSON.stringify({
          name,
          algorithm,
          policy_compute_provider: "MockRunPodProvider",
          judge_provider: "MockJudgeProvider",
          task_revision: "mounting-plate@sha256:fixture-v1",
          branch: {
            width: branchWidth,
            decision_after_actions: 3,
            rng_mode: "split_stream",
          },
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
        title="Launch a local CAD run"
        description="Typed configuration with mock-only providers and a normalized manifest preview."
      />
      <form className="content launch-layout" onSubmit={submit}>
        <div className="launch-form">
          {error ? (
            <Notice tone="negative" title="Run could not be launched">
              {error}
            </Notice>
          ) : null}
          <Section title="Scientific intent">
            <label className="field">
              <span>Run name</span>
              <input
                value={name}
                minLength={3}
                maxLength={80}
                onChange={(event) => setName(event.target.value)}
                required
              />
              <small>Shown in run and iteration lineage.</small>
            </label>
            <fieldset className="choice-field">
              <legend>Collection algorithm</legend>
              <label
                className={
                  algorithm === "bpo_local_metric"
                    ? "choice selected"
                    : "choice"
                }
              >
                <input
                  type="radio"
                  name="algorithm"
                  value="bpo_local_metric"
                  checked={algorithm === "bpo_local_metric"}
                  onChange={() => setAlgorithm("bpo_local_metric")}
                />
                <span>
                  <strong>Branch-aware BPO local</strong>
                  Shared three-action prefix, one logical checkpoint, four
                  isolated siblings.
                </span>
              </label>
              <label
                className={
                  algorithm === "independent_rollout_baseline"
                    ? "choice selected"
                    : "choice"
                }
              >
                <input
                  type="radio"
                  name="algorithm"
                  value="independent_rollout_baseline"
                  checked={algorithm === "independent_rollout_baseline"}
                  onChange={() => setAlgorithm("independent_rollout_baseline")}
                />
                <span>
                  <strong>Independent rollout baseline</strong>
                  Four complete trees without shared state or sibling
                  comparison.
                </span>
              </label>
            </fieldset>
            <label className="field compact-field">
              <span>Deterministic seed</span>
              <input
                type="number"
                min={0}
                max={2147483647}
                value={seed}
                onChange={(event) => {
                  const value = event.currentTarget.valueAsNumber;
                  setSeed(Number.isNaN(value) ? 0 : value);
                }}
              />
            </label>
          </Section>
          <Section title="Resolved providers">
            <div className="provider-list">
              <div>
                <span className="provider-icon" aria-hidden="true">
                  P
                </span>
                <span>
                  <strong>MockRunPodProvider</strong>
                  Deterministic policy allocation · local CPU · no RunPod
                  resolution path
                </span>
                <StatusBadge status="READY" />
              </div>
              <div>
                <span className="provider-icon" aria-hidden="true">
                  J
                </span>
                <span>
                  <strong>MockJudgeProvider</strong>
                  Contract fixtures only · no model endpoint or credentials
                </span>
                <StatusBadge status="READY" />
              </div>
            </div>
          </Section>
          <div className="form-actions">
            <Link className="button secondary" to="/runs">
              Cancel
            </Link>
            <button className="button primary" disabled={submitting}>
              {submitting ? "Authorizing…" : "Launch run"}
            </button>
          </div>
        </div>
        <aside className="manifest-panel">
          <h2>Normalized manifest preview</h2>
          <p>
            Read-only. The orchestrator canonicalizes this input and pins its
            digest.
          </p>
          <pre>{JSON.stringify(preview, null, 2)}</pre>
          <div className="cost-summary">
            <strong>Estimated local cost</strong>
            <span>
              {algorithm === "bpo_local_metric" ? "0.60" : "0.90"} execution
              credits
            </span>
            <span>0 external judge credits</span>
          </div>
        </aside>
      </form>
    </>
  );
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

interface RunEventsResponse {
  items: RunEvent[];
  next_cursor: number | null;
}

function latestNamed<T>(
  items: T[],
  key: (item: T) => string,
  timestamp: (item: T) => string,
): T[] {
  const latest = new Map<string, T>();
  for (const item of items) {
    const previous = latest.get(key(item));
    if (!previous || timestamp(item) > timestamp(previous))
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

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const navigate = useNavigate();
  const detail = useApi<RunDetailResponse>(`/v1/runs/${runId}`, 1_000);
  const batches = useApi<{ items: Array<Record<string, any>> }>(
    `/v1/runs/${runId}/collection-batches`,
    1_000,
  );
  const iterations = useApi<{ items: Array<Record<string, any>> }>(
    `/v1/runs/${runId}/iterations`,
    1_000,
  );
  const events = useApi<RunEventsResponse>(`/v1/runs/${runId}/events`, 1_000);
  const [actionError, setActionError] = useState<string | null>(null);
  const metricRows = useMemo(
    () =>
      latestNamed(
        detail.data?.metrics ?? [],
        (metric) => metric.descriptor,
        (metric) => metric.created_at,
      ),
    [detail.data?.metrics],
  );
  const rewardRows = useMemo(
    () =>
      latestNamed(
        detail.data?.reward_signals ?? [],
        (reward) => reward.name,
        (reward) => reward.created_at,
      ),
    [detail.data?.reward_signals],
  );
  const executionCredits =
    detail.data?.metrics
      .filter((metric) => metric.descriptor === "deterministic.execution_cost")
      .reduce(
        (total, metric) =>
          total + (typeof metric.value === "number" ? metric.value : 0),
        0,
      ) ?? 0;

  async function reproduce() {
    try {
      const result = await api<{ run_id: string }>(
        `/v1/runs/${runId}/reproduce`,
        {
          method: "POST",
        },
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

  const run = detail.data?.run;
  return (
    <>
      <PageHeader
        eyebrow={<Link to="/runs">Runs</Link>}
        title={run?.name ?? "Run"}
        description={
          run ? (
            <>
              <MachineId value={run.run_id} /> · {algorithmLabel(run.algorithm)}
            </>
          ) : (
            "Loading persisted scientific intent…"
          )
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
              {!["SUCCEEDED", "FAILED", "CANCELED"].includes(run.status) ? (
                <button
                  className="button danger"
                  type="button"
                  onClick={cancel}
                >
                  Cancel run
                </button>
              ) : null}
            </>
          ) : null
        }
      />
      <div className="content">
        <AsyncState loading={detail.loading} error={detail.error}>
          {actionError ? (
            <Notice tone="negative" title="Action failed">
              {actionError}
            </Notice>
          ) : null}
          {detail.data ? (
            <>
              <div className="run-overview">
                <Section title="Lifecycle">
                  <KeyValue
                    items={[
                      {
                        label: "Observed",
                        value: <StatusBadge status={detail.data.run.status} />,
                      },
                      {
                        label: "Desired",
                        value: detail.data.run.desired_state,
                      },
                      { label: "Attempts", value: detail.data.attempts.length },
                      {
                        label: "Cleanup",
                        value: (
                          <StatusBadge
                            status={detail.data.run.cleanup_status}
                          />
                        ),
                      },
                    ]}
                  />
                </Section>
                <Section title="Provider boundary">
                  <KeyValue
                    items={Object.entries(detail.data.providers).map(
                      ([label, value]) => ({
                        label: label.replace("_", " "),
                        value: <code>{value}</code>,
                      }),
                    )}
                  />
                </Section>
                <Section title="Evidence">
                  <KeyValue
                    items={[
                      { label: "Metrics", value: detail.data.metrics.length },
                      {
                        label: "Named rewards",
                        value: detail.data.reward_signals.length,
                      },
                      {
                        label: "Policy versions",
                        value: detail.data.policy_versions.length,
                      },
                      {
                        label: "Execution cost",
                        value: `${executionCredits.toFixed(4)} credits`,
                      },
                      {
                        label: "External judge cost",
                        value: "0 credits (mock)",
                      },
                    ]}
                  />
                </Section>
              </div>
              <Notice tone="warning" title="Model assessment boundary">
                {detail.data.calibration.message}
              </Notice>
              {detail.data.failures.length ? (
                <Notice tone="negative" title="Run failure">
                  <pre>{JSON.stringify(detail.data.failures, null, 2)}</pre>
                </Notice>
              ) : null}
              <Section
                title="Metric observations"
                aside={
                  <span>
                    Latest of {detail.data.metrics.length} persisted ·{" "}
                    {metricRows.length} descriptors
                  </span>
                }
              >
                <AsyncState empty={!metricRows.length}>
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Authority</th>
                          <th>Descriptor</th>
                          <th>Value</th>
                          <th>Subject</th>
                          <th>Recorded</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metricRows.map((metric) => (
                          <tr key={metric.descriptor}>
                            <td>
                              <StatusBadge
                                status={
                                  metric.descriptor.startsWith("judge.")
                                    ? "JUDGE"
                                    : "DETERMINISTIC"
                                }
                              />
                            </td>
                            <td>
                              <code>{metric.descriptor}</code>
                            </td>
                            <td>
                              {observationValue(metric.value)}{" "}
                              <small>{metric.unit}</small>
                            </td>
                            <td>
                              <MachineId
                                value={metric.subject_id}
                                copy={false}
                              />
                            </td>
                            <td>{formatDate(metric.created_at)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </AsyncState>
              </Section>
              <Section
                title="Named reward signals"
                aside={
                  <span>
                    Latest of {detail.data.reward_signals.length} persisted ·{" "}
                    {rewardRows.length} names
                  </span>
                }
              >
                <AsyncState empty={!rewardRows.length}>
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Signal</th>
                          <th>Value</th>
                          <th>Pipeline</th>
                          <th>Subject</th>
                          <th>Metric facts</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rewardRows.map((reward) => (
                          <tr key={reward.name}>
                            <td>{reward.name.replaceAll("_", " ")}</td>
                            <td>
                              <strong>{reward.value.toFixed(4)}</strong>
                            </td>
                            <td>
                              <code>{reward.reward_pipeline_id}</code>
                            </td>
                            <td>
                              <span>{reward.subject_type.toLowerCase()}</span>
                              <MachineId
                                value={reward.subject_id}
                                copy={false}
                              />
                            </td>
                            <td>{reward.metric_observation_ids.length}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </AsyncState>
              </Section>
              <Section title="Collection batches">
                <AsyncState
                  loading={batches.loading}
                  error={batches.error}
                  empty={!batches.data?.items.length}
                >
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Batch</th>
                          <th>Status</th>
                          <th>Policy</th>
                          <th>Trees</th>
                          <th>Verification</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batches.data?.items.map((batch) => (
                          <tr key={batch.collection_batch_id}>
                            <td>
                              <MachineId value={batch.collection_batch_id} />
                            </td>
                            <td>
                              <StatusBadge status={batch.status} />
                            </td>
                            <td>
                              <MachineId
                                value={batch.behavior_policy_version_id}
                                copy={false}
                              />
                            </td>
                            <td>{batch.tree_count}</td>
                            <td>
                              <code>{batch.verification_plan_id}</code>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </AsyncState>
              </Section>
              <Section title="Training iterations">
                <AsyncState
                  loading={iterations.loading}
                  error={iterations.error}
                  empty={!iterations.data?.items.length}
                >
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Iteration</th>
                          <th>Status</th>
                          <th>Input → output policy</th>
                          <th>Trees</th>
                          <th>Excluded</th>
                          <th>Input manifest</th>
                        </tr>
                      </thead>
                      <tbody>
                        {iterations.data?.items.map((iteration) => (
                          <tr key={iteration.training_iteration_id}>
                            <td>
                              <Link
                                className="row-link inline"
                                to={`/runs/${runId}/iterations/${iteration.training_iteration_id}`}
                              >
                                <MachineId
                                  value={iteration.training_iteration_id}
                                  copy={false}
                                />
                              </Link>
                            </td>
                            <td>
                              <StatusBadge status={iteration.status} />
                            </td>
                            <td>
                              <MachineId
                                value={iteration.input_policy_version_id}
                                copy={false}
                              />{" "}
                              →{" "}
                              {iteration.output_policy_version_id ? (
                                <MachineId
                                  value={iteration.output_policy_version_id}
                                  copy={false}
                                />
                              ) : (
                                "pending"
                              )}
                            </td>
                            <td>{iteration.tree_count}</td>
                            <td>{iteration.excluded_count}</td>
                            <td>
                              {iteration.iteration_input_digest ? (
                                <MachineId
                                  value={iteration.iteration_input_digest}
                                />
                              ) : (
                                "not materialized"
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </AsyncState>
              </Section>
              <Section title="Attempts and resources">
                <AsyncState empty={!detail.data.attempts.length}>
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Attempt</th>
                          <th>Lifecycle</th>
                          <th>Heartbeat</th>
                          <th>Allocation</th>
                          <th>Observed resource</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.data.attempts.map((attempt) => {
                          const allocation = detail.data!.allocations.find(
                            (item) =>
                              item.run_attempt_id === attempt.attempt_id,
                          );
                          return (
                            <tr key={attempt.attempt_id}>
                              <td>
                                <MachineId value={attempt.attempt_id} />
                              </td>
                              <td>
                                <StatusBadge status={attempt.status} />
                              </td>
                              <td>
                                {attempt.heartbeat_at
                                  ? formatDate(attempt.heartbeat_at)
                                  : "Not received"}
                              </td>
                              <td>
                                <code>{attempt.provider_name}</code>
                              </td>
                              <td>
                                <StatusBadge
                                  status={
                                    allocation?.observed_state ?? "UNKNOWN"
                                  }
                                />
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </AsyncState>
              </Section>
              <Section
                title="Run event log"
                aside={
                  <span>
                    Latest {Math.min(events.data?.items.length ?? 0, 20)} events
                  </span>
                }
              >
                <AsyncState
                  loading={events.loading}
                  error={events.error}
                  empty={!events.data?.items.length}
                >
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Cursor</th>
                          <th>Event</th>
                          <th>Producer</th>
                          <th>Aggregate</th>
                          <th>Payload</th>
                          <th>Occurred</th>
                        </tr>
                      </thead>
                      <tbody>
                        {events.data?.items
                          .slice(-20)
                          .reverse()
                          .map((event) => (
                            <tr key={event.event_id}>
                              <td>{event.cursor}</td>
                              <td>
                                <code>{event.event_type}</code>
                              </td>
                              <td>{event.producer}</td>
                              <td>
                                <span>{event.aggregate_type}</span>
                                <MachineId
                                  value={event.aggregate_id}
                                  copy={false}
                                />
                              </td>
                              <td>
                                <details className="event-payload">
                                  <summary>
                                    {Object.keys(event.payload).length} fields
                                  </summary>
                                  <pre className="json-block compact">
                                    {JSON.stringify(event.payload, null, 2)}
                                  </pre>
                                </details>
                              </td>
                              <td>{formatDate(event.occurred_at)}</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </AsyncState>
              </Section>
              <Section title="Provenance">
                <KeyValue
                  items={[
                    {
                      label: "Manifest digest",
                      value: (
                        <MachineId value={detail.data.run.manifest_digest} />
                      ),
                      span: true,
                    },
                    {
                      label: "Task",
                      value: (
                        <code>
                          {String(detail.data.run.manifest.task_revision)}
                        </code>
                      ),
                    },
                    {
                      label: "Snapshot fidelity",
                      value: <code>logical_restore</code>,
                    },
                    {
                      label: "Reward pipeline",
                      value: <code>cad-local-rewards@1</code>,
                    },
                    {
                      label: "Calibration",
                      value: <StatusBadge status="MOCK_CONTRACT_ONLY" />,
                    },
                  ]}
                />
              </Section>
            </>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

interface IterationDetail {
  iteration: Record<string, any>;
  rollout_trees: Array<Record<string, any>>;
  eligibility_decisions: Array<Record<string, any>>;
}

export function IterationPage() {
  const { runId = "", iterationId = "" } = useParams();
  const { data, error, loading } = useApi<IterationDetail>(
    `/v1/iterations/${iterationId}`,
  );
  const decisions = (treeId: string) =>
    data?.eligibility_decisions.filter(
      (decision) => decision.rollout_tree_id === treeId,
    ) ?? [];

  return (
    <>
      <PageHeader
        eyebrow={
          <>
            <Link to="/runs">Runs</Link> /{" "}
            <Link to={`/runs/${runId}`}>Run</Link>
          </>
        }
        title="Committed iteration"
        description={<MachineId value={iterationId} />}
        actions={data ? <StatusBadge status={data.iteration.status} /> : null}
      />
      <div className="content">
        <AsyncState loading={loading} error={error}>
          {data ? (
            <>
              <div className="run-overview">
                <Section title="Policy advance">
                  <KeyValue
                    items={[
                      {
                        label: "Input",
                        value: (
                          <MachineId
                            value={data.iteration.input_policy_version_id}
                          />
                        ),
                      },
                      {
                        label: "Output",
                        value: data.iteration.output_policy_version_id ? (
                          <MachineId
                            value={data.iteration.output_policy_version_id}
                          />
                        ) : (
                          "Pending commit"
                        ),
                      },
                      {
                        label: "Commit operation",
                        value: data.iteration.commit_operation_id ? (
                          <MachineId
                            value={data.iteration.commit_operation_id}
                          />
                        ) : (
                          "Pending commit"
                        ),
                      },
                    ]}
                  />
                </Section>
                <Section title="Immutable inputs">
                  <KeyValue
                    items={[
                      {
                        label: "Trees considered",
                        value: data.rollout_trees.length,
                      },
                      {
                        label: "Eligibility records",
                        value: data.eligibility_decisions.length,
                      },
                      {
                        label: "Manifest",
                        value: data.iteration.iteration_input_digest ? (
                          <MachineId
                            value={data.iteration.iteration_input_digest}
                          />
                        ) : (
                          "Pending materialization"
                        ),
                      },
                    ]}
                  />
                </Section>
              </div>
              <Section title="Every rollout tree considered">
                <AsyncState
                  empty={
                    !data.rollout_trees.length ||
                    !data.eligibility_decisions.length
                  }
                >
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Tree</th>
                          <th>Collection status</th>
                          <th>Eligibility</th>
                          <th>Reason</th>
                          <th>Digest</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.rollout_trees.flatMap((tree) => {
                          const treeDecisions = decisions(tree.rollout_tree_id);
                          return treeDecisions.map((decision, index) => (
                            <tr key={decision.decision_id}>
                              <td>
                                {index === 0 ? (
                                  <Link
                                    className="row-link inline"
                                    to={`/rollout-trees/${tree.rollout_tree_id}`}
                                  >
                                    <MachineId
                                      value={tree.rollout_tree_id}
                                      copy={false}
                                    />
                                  </Link>
                                ) : (
                                  <span className="muted">
                                    ↳ sibling decision
                                  </span>
                                )}
                              </td>
                              <td>
                                <StatusBadge status={tree.status} />
                              </td>
                              <td>
                                <StatusBadge status={decision.status} />
                              </td>
                              <td>
                                {decision.reason_code.replaceAll("_", " ")}
                              </td>
                              <td>
                                {index === 0 ? (
                                  <MachineId value={tree.digest} />
                                ) : (
                                  "shared tree"
                                )}
                              </td>
                            </tr>
                          ));
                        })}
                      </tbody>
                    </table>
                  </div>
                </AsyncState>
              </Section>
              <Section title="Iteration-input manifest">
                <p className="section-copy">
                  This is the exact materializer input accepted before the
                  compare-and-swap policy commit.
                </p>
                {data.iteration.iteration_input_manifest ? (
                  <details className="manifest-details">
                    <summary>Inspect exact manifest</summary>
                    <pre className="json-block">
                      {JSON.stringify(
                        data.iteration.iteration_input_manifest,
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                ) : (
                  <Notice title="Manifest pending">
                    Manifest materialization is still pending.
                  </Notice>
                )}
              </Section>
            </>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}
