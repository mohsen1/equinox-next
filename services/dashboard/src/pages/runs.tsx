import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "../router";
import { api, useApi } from "../api";
import {
  AsyncState,
  EvidenceStrengthBadge,
  ExecutionBadge,
  formatDate,
  KeyValue,
  LearningOutcomeBadge,
  MachineId,
  Notice,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";
import type { RunSummary, StudySummary } from "../types";

export function RunsPage() {
  const [params, setParams] = useSearchParams();
  const [query, setQuery] = useState(params.get("q") ?? "");
  const requestParams = new URLSearchParams({ limit: "50" });
  for (const key of ["q", "status", "sort"]) {
    const value = params.get(key);
    if (value) requestParams.set(key, value);
  }
  const { data, error, loading, setData, refresh } = useApi<{
    items: RunSummary[];
    total: number;
  }>(`/v1/runs?${requestParams.toString()}`, 10_000);
  const studies = useApi<{ items: StudySummary[] }>("/v1/studies", 10_000);
  const [seeding, setSeeding] = useState(false);
  const [seedError, setSeedError] = useState<string | null>(null);

  async function seed() {
    setSeeding(true);
    setSeedError(null);
    try {
      await api("/internal/seed", { method: "POST" });
      const runs = await api<{ items: RunSummary[]; total: number }>(
        `/v1/runs?${requestParams.toString()}`,
      );
      setData(runs);
      studies.refresh();
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

  function applyFilters(event: FormEvent) {
    event.preventDefault();
    const next = new URLSearchParams(params);
    if (query.trim()) next.set("q", query.trim());
    else next.delete("q");
    setParams(next);
  }

  return (
    <>
      <PageHeader
        eyebrow="Research operations"
        title="Runs"
        description="Studies, matched conditions, and the individual executions that produced immutable evidence."
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
        {seedError ? (
          <Notice tone="negative" title="Seed failed">
            {seedError}
          </Notice>
        ) : null}
        <AsyncState loading={studies.loading} error={studies.error}>
          {studies.data?.items.length ? (
            <Section
              title="Studies"
              aside={<span>{studies.data.items.length} defined</span>}
            >
              <div className="study-index">
                {studies.data.items.map((study) => (
                  <Link
                    key={study.study_id}
                    to={`/studies/${study.study_id}`}
                    className="study-row"
                  >
                    <span className="study-identity">
                      <strong>{study.research_question}</strong>
                      <span>
                        <code>{study.protocol_revision}</code> ·{" "}
                        {study.run_count} run{study.run_count === 1 ? "" : "s"}
                      </span>
                    </span>
                    <span className="study-condition-summary">
                      {study.conditions.map((condition) => (
                        <span key={condition.condition}>
                          {condition.condition
                            .toLowerCase()
                            .replaceAll("_", " ")}
                          <small>
                            {condition.completed_count}/{condition.run_count}
                          </small>
                        </span>
                      ))}
                    </span>
                    <span className="study-match">
                      <StatusBadge status={study.comparison.status} />
                      <small>
                        {study.comparison.paired_seeds.length
                          ? `Seed ${study.comparison.paired_seeds.join(", ")}`
                          : "Awaiting paired seed"}
                      </small>
                    </span>
                  </Link>
                ))}
              </div>
            </Section>
          ) : null}
        </AsyncState>
        <form className="index-controls" onSubmit={applyFilters}>
          <label className="search-field">
            <span>Search runs</span>
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Name or run ID"
            />
          </label>
          <label>
            <span>Execution</span>
            <select
              value={params.get("status") ?? ""}
              onChange={(event) => {
                const next = new URLSearchParams(params);
                if (event.target.value) next.set("status", event.target.value);
                else next.delete("status");
                setParams(next);
              }}
            >
              <option value="">All states</option>
              <option value="RUNNING">Running</option>
              <option value="SUCCEEDED">Completed</option>
              <option value="FAILED">Failed</option>
              <option value="CANCELED">Canceled</option>
            </select>
          </label>
          <label>
            <span>Sort</span>
            <select
              value={params.get("sort") ?? "updated_desc"}
              onChange={(event) => {
                const next = new URLSearchParams(params);
                next.set("sort", event.target.value);
                setParams(next);
              }}
            >
              <option value="updated_desc">Recently updated</option>
              <option value="created_desc">Recently created</option>
              <option value="name_asc">Name A–Z</option>
            </select>
          </label>
          <button className="button secondary" type="submit">
            Apply
          </button>
          <button className="text-button" type="button" onClick={refresh}>
            Refresh
          </button>
          <span className="index-count">{data?.total ?? 0} runs</span>
        </form>
        <AsyncState loading={loading} error={error} empty={!data?.items.length}>
          <Section
            title="Individual runs"
            aside={<span>{data?.items.length ?? 0} shown</span>}
          >
            <div className="table-wrap desktop-index">
              <table className="data-table runs-table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Condition</th>
                    <th>Execution</th>
                    <th>Learning</th>
                    <th>Evidence</th>
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
                        <span>
                          {run.study.condition
                            .toLowerCase()
                            .replaceAll("_", " ")}
                        </span>
                        <small>seed {run.manifest.seed ?? "—"}</small>
                      </td>
                      <td>
                        <ExecutionBadge status={run.status} />
                      </td>
                      <td>
                        <LearningOutcomeBadge outcome={run.learning_outcome} />
                        <small>
                          {run.committed_iteration_count} committed update
                          {run.committed_iteration_count === 1 ? "" : "s"}
                        </small>
                      </td>
                      <td>
                        <span>
                          {run.proof_count} proofs · {run.rollout_tree_count}{" "}
                          trees
                        </span>
                        <small>
                          {run.retry_count} retries · {run.abstention_count}{" "}
                          abstentions
                        </small>
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
            <ul className="mobile-index run-cards">
              {data?.items.map((run) => (
                <li key={run.run_id}>
                  <Link to={`/runs/${run.run_id}`}>
                    <span className="mobile-card-title">
                      <strong>{run.name}</strong>
                      <small>
                        {run.study.condition.toLowerCase().replaceAll("_", " ")}{" "}
                        · seed {run.manifest.seed ?? "—"}
                      </small>
                    </span>
                    <ExecutionBadge status={run.status} />
                    <span className="mobile-card-summary">{run.summary}</span>
                    <span className="mobile-card-footer">
                      <LearningOutcomeBadge outcome={run.learning_outcome} />
                      <small>{run.proof_count} proofs</small>
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </Section>
        </AsyncState>
        <Notice title="Local provider boundary">
          Policy compute resolves <code>MockRunPodProvider</code>; model
          assessments resolve <code>MockJudgeProvider</code>. External capacity
          and held-out evaluation are disabled.
        </Notice>
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
  const studyId = "study_cad_branching_contract_v1";
  const studyCondition =
    algorithm === "bpo_local_metric" ? "BRANCH_AWARE" : "INDEPENDENT_CONTROL";
  const researchQuestion =
    "Does a shared decision checkpoint produce more useful CAD continuations than independent rollouts under a matched protocol?";
  const preview = useMemo(
    () => ({
      profile: "local-contract-proof",
      study: {
        study_id: studyId,
        condition: studyCondition,
        research_question: researchQuestion,
        protocol_revision: "cad-contract-protocol@1",
      },
      data_protocol: {
        source: "generated",
        generator_revision: "cad-fixture-generator@1",
        split_policy: "training fixture only; no held-out evaluation",
        sampling: "paired deterministic seed",
      },
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
    [algorithm, branchWidth, researchQuestion, seed, studyCondition, studyId],
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
          study_id: studyId,
          study_condition: studyCondition,
          research_question: researchQuestion,
          protocol_revision: "cad-contract-protocol@1",
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
            <div className="research-question">
              <span>Study question</span>
              <strong>{researchQuestion}</strong>
              <small>
                Conditions are compared only when task, model, evaluation pack,
                budget, and paired-seed policy match.
              </small>
            </div>
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
  study: {
    study_id: string;
    condition: string;
    research_question: string;
    protocol_revision: string;
    match_contract_digest: string;
  };
  data_protocol: {
    source: string;
    digest: string;
    generator: {
      id: string;
      revision: string;
      digest: string;
      task_revision: string;
      seed: number;
      sampling: string;
    };
    splits: Record<
      string,
      {
        task_groups: number;
        candidate_trajectories?: number;
        reason?: string;
      }
    >;
    sampling_policy: {
      strategy: string;
      seed: number;
      temperature: number;
    };
    quality_checks: Record<string, string>;
    gradient_lineage: {
      status: string;
      contributing_rollout_trees: number;
      contributing_proofs: number;
      contributing_reward_signals: number;
      eligibility_decisions_recorded: number;
      consumed_by_policy_version: string | null;
      iteration_input_digest: string | null;
    };
  };
  outcome: {
    learning_outcome: string;
    evidence_strength: string;
    summary: string;
  };
  proof_count: number;
  evaluation: {
    held_out_examples: number;
    test_result: number | null;
    claim: string;
  };
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
  const detail = useApi<RunDetailResponse>(`/v1/runs/${runId}`);
  const live = useApi<RunSummary>(`/v1/runs/${runId}/summary`, 3_000);
  const batches = useApi<{ items: Array<Record<string, any>> }>(
    `/v1/runs/${runId}/collection-batches`,
  );
  const iterations = useApi<{ items: Array<Record<string, any>> }>(
    `/v1/runs/${runId}/iterations`,
  );
  const events = useApi<RunEventsResponse>(
    `/v1/runs/${runId}/events?limit=50`,
    10_000,
  );
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

  useEffect(() => {
    if (!detail.data || !live.data) return;
    if (
      detail.data.run.status !== live.data.status ||
      detail.data.run.updated_at !== live.data.updated_at
    ) {
      detail.refresh();
      batches.refresh();
      iterations.refresh();
    }
  }, [
    batches.refresh,
    detail.data,
    detail.refresh,
    iterations.refresh,
    live.data,
  ]);

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
  const liveStatus = live.data?.status ?? run?.status;
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
              {liveStatus ? <ExecutionBadge status={liveStatus} /> : null}
              <button
                className="button secondary"
                type="button"
                onClick={reproduce}
              >
                Reproduce
              </button>
              {liveStatus &&
              !["SUCCEEDED", "FAILED", "CANCELED"].includes(liveStatus) ? (
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
              <section className="decision-statement">
                <span>Run conclusion</span>
                <h2>{detail.data.outcome.summary}</h2>
                <div className="decision-badges">
                  <ExecutionBadge
                    status={liveStatus ?? detail.data.run.status}
                  />
                  <LearningOutcomeBadge
                    outcome={detail.data.outcome.learning_outcome}
                  />
                  <EvidenceStrengthBadge
                    strength={detail.data.outcome.evidence_strength}
                  />
                </div>
              </section>
              <nav className="run-local-nav" aria-label="Run sections">
                <a href="#overview">Overview</a>
                <a href="#trajectory">Trajectory</a>
                <a href="#evidence">Evidence</a>
                <a href="#technical-details">Technical details</a>
              </nav>
              <div className="run-overview" id="overview">
                <Section title="Execution">
                  <KeyValue
                    items={[
                      {
                        label: "Observed",
                        value: (
                          <ExecutionBadge
                            status={liveStatus ?? detail.data.run.status}
                          />
                        ),
                      },
                      {
                        label: "Control intent",
                        value:
                          detail.data.run.desired_state === "RUNNING"
                            ? "Allow completion"
                            : "Cancel requested",
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
                <Section title="Learning outcome">
                  <KeyValue
                    items={[
                      {
                        label: "Result",
                        value: (
                          <LearningOutcomeBadge
                            outcome={detail.data.outcome.learning_outcome}
                          />
                        ),
                      },
                      {
                        label: "Held-out examples",
                        value: detail.data.evaluation.held_out_examples,
                      },
                      {
                        label: "Policy versions",
                        value: detail.data.policy_versions.length,
                      },
                      {
                        label: "Evidence strength",
                        value: (
                          <EvidenceStrengthBadge
                            strength={detail.data.outcome.evidence_strength}
                          />
                        ),
                      },
                    ]}
                  />
                </Section>
                <Section title="Evidence">
                  <KeyValue
                    items={[
                      {
                        label: "Proof bundles",
                        value: (
                          <Link
                            to={`/proofs?q=${encodeURIComponent(detail.data.run.run_id)}`}
                          >
                            {detail.data.proof_count}
                          </Link>
                        ),
                      },
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
              <div className="protocol-overview">
                <Section
                  title="Research question"
                  aside={
                    <Link to={`/studies/${detail.data.study.study_id}`}>
                      Compare study
                    </Link>
                  }
                >
                  <div className="research-brief">
                    <p>{detail.data.study.research_question}</p>
                    <dl>
                      <div>
                        <dt>Condition</dt>
                        <dd>
                          {detail.data.study.condition
                            .toLowerCase()
                            .replaceAll("_", " ")}
                        </dd>
                      </div>
                      <div>
                        <dt>Protocol</dt>
                        <dd>
                          <code>{detail.data.study.protocol_revision}</code>
                        </dd>
                      </div>
                      <div>
                        <dt>Seed</dt>
                        <dd>
                          {detail.data.data_protocol.sampling_policy.seed}
                        </dd>
                      </div>
                    </dl>
                  </div>
                </Section>
                <Section
                  title="Data and protocol"
                  aside={
                    <StatusBadge
                      status={detail.data.data_protocol.gradient_lineage.status}
                    />
                  }
                >
                  <KeyValue
                    items={[
                      {
                        label: "Training source",
                        value: detail.data.data_protocol.source,
                      },
                      {
                        label: "Generator",
                        value: (
                          <code>
                            {detail.data.data_protocol.generator.revision}
                          </code>
                        ),
                      },
                      {
                        label: "Training groups",
                        value:
                          detail.data.data_protocol.splits.training.task_groups,
                      },
                      {
                        label: "Candidate trajectories",
                        value:
                          detail.data.data_protocol.splits.training
                            .candidate_trajectories ?? 0,
                      },
                      {
                        label: "Held-out splits",
                        value: "0 validation · 0 guard · 0 test",
                      },
                      {
                        label: "Protocol digest",
                        value: (
                          <MachineId value={detail.data.data_protocol.digest} />
                        ),
                        span: true,
                      },
                    ]}
                  />
                </Section>
              </div>
              <Section
                title="Gradient contribution lineage"
                aside={
                  detail.data.data_protocol.gradient_lineage
                    .consumed_by_policy_version ? (
                    <MachineId
                      value={
                        detail.data.data_protocol.gradient_lineage
                          .consumed_by_policy_version
                      }
                      copy={false}
                    />
                  ) : (
                    <span>Awaiting materialization</span>
                  )
                }
              >
                <div className="lineage-strip">
                  <span>
                    <strong>
                      {
                        detail.data.data_protocol.gradient_lineage
                          .contributing_rollout_trees
                      }
                    </strong>
                    eligible rollout trees
                  </span>
                  <span>
                    <strong>
                      {
                        detail.data.data_protocol.gradient_lineage
                          .contributing_proofs
                      }
                    </strong>
                    proof bundles
                  </span>
                  <span>
                    <strong>
                      {
                        detail.data.data_protocol.gradient_lineage
                          .contributing_reward_signals
                      }
                    </strong>
                    named reward signals
                  </span>
                  <span>
                    <strong>
                      {
                        detail.data.data_protocol.gradient_lineage
                          .eligibility_decisions_recorded
                      }
                    </strong>
                    eligibility decisions
                  </span>
                </div>
              </Section>
              <Notice tone="warning" title="Model assessment boundary">
                {detail.data.calibration.message} {detail.data.evaluation.claim}
              </Notice>
              {detail.data.failures.length ? (
                <Notice tone="negative" title="Run failure">
                  <pre>{JSON.stringify(detail.data.failures, null, 2)}</pre>
                </Notice>
              ) : null}
              <Section id="trajectory" title="Collection batches">
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
              <Section
                id="evidence"
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
              <Section id="technical-details" title="Attempts and resources">
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
              <div className="run-overview iteration-overview">
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
