import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { useApi } from "../api";
import {
  AsyncState,
  formatDate,
  MachineId,
  PageHeader,
  StatusBadge,
} from "../components";
import {
  decodeResearchStudyReport,
  decodeRunsResponse,
  decodeStudiesResponse,
  type RunsResponse,
  type StudiesResponse,
} from "../contracts";
import { formatEstimatedCost, formatRelativeTime } from "../format";
import { Link } from "../router";
import { shortModelName } from "../runs-helpers";
import type {
  ResearchComputeExecution,
  ResearchStudyCondition,
  ResearchStudyReport,
} from "../types";

interface ConditionRow extends ResearchStudyCondition {
  id: string;
}

type ConditionSort = "id" | "seed" | "branch_width" | "final_successes";

const chartTooltipStyle = {
  background: "var(--surface)",
  border: "1px solid var(--rule-strong)",
  borderRadius: "4px",
  color: "var(--ink)",
  fontSize: "0.75rem",
};

export function DashboardPage() {
  const runs = useApi<RunsResponse>("/v1/runs", 2_000, decodeRunsResponse);
  const studies = useApi<StudiesResponse>(
    "/v1/studies",
    10_000,
    decodeStudiesResponse,
  );
  const researchRuns = [...(runs.data?.research_items ?? [])]
    .filter((run) => run.execution_id !== "runpod-proof-ui-rehearsal")
    .sort(
      (left, right) =>
        Date.parse(right.updated_at) - Date.parse(left.updated_at),
    );
  const latestRun = researchRuns[0];
  const latestStudy =
    studies.data &&
    [...studies.data.items].sort(
      (left, right) =>
        Date.parse(right.generated_at) - Date.parse(left.generated_at),
    )[0];

  return (
    <>
      <PageHeader title="Dashboard" />
      <div className="content dashboard-content">
        <LatestRun
          run={latestRun}
          loading={runs.loading}
          error={runs.error}
          retry={runs.retry}
        />

        <AsyncState
          loading={studies.loading}
          error={studies.error}
          stale={Boolean(studies.data && studies.error)}
          onRetry={studies.retry}
        >
          {latestStudy ? (
            <StudyReadout studyId={latestStudy.study_id} />
          ) : studies.data ? (
            <section className="state-panel empty-state">
              <strong>No study evidence yet</strong>
              <Link className="button secondary" to="/runs">
                View runs
              </Link>
            </section>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

export default DashboardPage;

function LatestRun({
  run,
  loading,
  error,
  retry,
}: {
  run: ResearchComputeExecution | undefined;
  loading: boolean;
  error: Error | null;
  retry: () => void;
}) {
  return (
    <section className="dashboard-panel dashboard-latest-run">
      <header className="panel-heading">
        <h2>Latest run</h2>
      </header>
      <AsyncState
        loading={loading}
        error={error}
        stale={Boolean(run && error)}
        onRetry={retry}
      >
        {run ? (
          <Link
            className="dashboard-run-row"
            to={`/runs/research/${run.execution_id}`}
          >
            <span className="dashboard-run-identity">
              <strong>{run.name}</strong>
              <small>
                {shortModelName(run.model_id)} · K={run.branch_width} · adaptive
              </small>
            </span>
            <StatusBadge status={run.status} />
            <span>{run.allocated_gpu ?? "Awaiting allocation"}</span>
            <span>{formatEstimatedCost(run.cost)}</span>
            <time
              dateTime={run.updated_at}
              aria-label={formatDate(run.updated_at)}
              title={formatDate(run.updated_at)}
            >
              {formatRelativeTime(run.updated_at)}
            </time>
          </Link>
        ) : !loading && !error ? (
          <div className="state-panel empty-state">
            <strong>No runs yet</strong>
          </div>
        ) : null}
      </AsyncState>
    </section>
  );
}

function StudyReadout({ studyId }: { studyId: string }) {
  const study = useApi<ResearchStudyReport>(
    `/v1/studies/${encodeURIComponent(studyId)}`,
    10_000,
    decodeResearchStudyReport,
  );

  return (
    <AsyncState
      loading={study.loading}
      error={study.error}
      stale={Boolean(study.data && study.error)}
      onRetry={study.retry}
    >
      {study.data ? <StudyEvidence report={study.data} /> : null}
    </AsyncState>
  );
}

function StudyEvidence({ report }: { report: ResearchStudyReport }) {
  const decisions = Object.entries(report.decisions);
  const passed = decisions.filter(
    ([, decision]) => decision.status === "PASS",
  ).length;
  const conditionCount = Object.keys(report.conditions).length;

  return (
    <>
      <section className="dashboard-decision" aria-labelledby="study-decision">
        <div className="decision-copy">
          <Link
            className="decision-study-link"
            to={`/studies/${encodeURIComponent(report.study_id)}`}
          >
            {report.study_id}
          </Link>
          <h2 id="study-decision">
            {report.overall_status === "PASS"
              ? "The latest study passed its preregistered gates."
              : "The latest study did not pass its preregistered gates."}
          </h2>
          <p>
            {report.freeze.model?.id ?? "Model unavailable"} · {passed} of{" "}
            {decisions.length} gates passed across {conditionCount} conditions.
          </p>
        </div>
        <div className="decision-verdict">
          <StatusBadge status={report.overall_status} />
          <Link to={`/studies/${encodeURIComponent(report.study_id)}`}>
            Open study
          </Link>
        </div>
        <div className="decision-source">
          <time dateTime={report.generated_at}>
            {formatDate(report.generated_at)}
          </time>
          <MachineId value={report.report_digest} copy={false} />
        </div>
      </section>

      <div className="dashboard-primary-grid">
        <ChartPanel title="Internal evaluation">
          <InternalEvaluationChart report={report} />
        </ChartPanel>
        <DecisionGates report={report} />
      </div>

      <div className="dashboard-chart-grid dashboard-single-chart">
        <ChartPanel title="Provider spend">
          <ProviderCostChart report={report} />
        </ChartPanel>
      </div>

      <ConditionLedger report={report} />

      <section className="dashboard-panel provenance-panel">
        <header className="panel-heading">
          <h2>Study provenance</h2>
          <StatusBadge status="DIGEST_RECORDED" />
        </header>
        <dl>
          <div>
            <dt>Study</dt>
            <dd>
              <code>{report.study_id}</code>
            </dd>
          </div>
          <div>
            <dt>Workload</dt>
            <dd>
              <code>{report.freeze.frozen_workload_revision ?? "—"}</code>
            </dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>{report.freeze.model?.id ?? "—"}</dd>
          </div>
          <div>
            <dt>Freeze commit</dt>
            <dd>
              <code>{report.freeze.frozen_source_commit ?? "—"}</code>
            </dd>
          </div>
          <div>
            <dt>Executions</dt>
            <dd>{report.executions.length}</dd>
          </div>
          <div>
            <dt>Aggregation</dt>
            <dd>{report.aggregation_policy}</dd>
          </div>
        </dl>
      </section>
    </>
  );
}

function ChartPanel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="dashboard-panel chart-panel">
      <header className="panel-heading">
        <h2>{title}</h2>
      </header>
      {children}
    </section>
  );
}

function InternalEvaluationChart({ report }: { report: ResearchStudyReport }) {
  const data = Object.entries(report.conditions).map(([id, condition]) => ({
    condition: shortCondition(id),
    initial:
      condition.initial_successes === undefined ||
      condition.initial_successes === null
        ? undefined
        : condition.initial_successes * 100,
    final:
      condition.final_successes === undefined ||
      condition.final_successes === null
        ? undefined
        : condition.final_successes * 100,
  }));

  return (
    <>
      <div
        className="chart-frame"
        role="img"
        aria-label="Initial and final internal evaluation success rates by study condition."
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 14, right: 12, left: -10, bottom: 36 }}
          >
            <CartesianGrid stroke="var(--rule)" vertical={false} />
            <XAxis
              dataKey="condition"
              stroke="var(--ink-muted)"
              tick={{ fill: "var(--ink-muted)", fontSize: 10 }}
              interval={0}
              height={54}
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={(value) => `${String(value)}%`}
              stroke="var(--ink-muted)"
              tick={{ fill: "var(--ink-muted)", fontSize: 10 }}
              width={52}
            />
            <Tooltip
              contentStyle={chartTooltipStyle}
              formatter={(value) => `${Number(value).toFixed(1)}%`}
            />
            <Legend wrapperStyle={{ fontSize: "0.72rem" }} />
            <Bar
              dataKey="initial"
              name="Initial"
              fill="var(--rule-strong)"
              radius={[2, 2, 0, 0]}
              isAnimationActive={false}
            />
            <Bar
              dataKey="final"
              name="Final"
              fill="var(--accent)"
              radius={[2, 2, 0, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ChartDataTable
        label="Internal evaluation data"
        headings={["Condition", "Initial", "Final"]}
        rows={Object.entries(report.conditions).map(([id, condition]) => [
          id,
          formatPercent(condition.initial_successes),
          formatPercent(condition.final_successes),
        ])}
      />
    </>
  );
}

function ProviderCostChart({ report }: { report: ResearchStudyReport }) {
  let knownCumulative = 0;
  const data = report.executions.map((execution, index) => {
    const cost = execution.estimated_cost_usd;
    const costKnown = cost !== undefined && cost !== null;
    if (costKnown) knownCumulative += cost;
    return {
      attempt: index + 1,
      cumulative: costKnown ? Number(knownCumulative.toFixed(6)) : null,
      knownCumulative: costKnown ? knownCumulative : null,
      cost,
      execution,
    };
  });
  const knownTotal = estimatedStudyCost(report);
  const unknownCount = unknownStudyCostCount(report);

  return (
    <>
      <div
        className="chart-frame"
        role="img"
        aria-label="Cumulative known estimated provider cost across recorded study executions. Gaps mark executions with unavailable cost."
      >
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 14, right: 14, left: -2, bottom: 8 }}
          >
            <CartesianGrid stroke="var(--rule)" vertical={false} />
            <XAxis
              dataKey="attempt"
              stroke="var(--ink-muted)"
              tick={{ fill: "var(--ink-muted)", fontSize: 10 }}
            />
            <YAxis
              tickFormatter={(value) => `$${Number(value).toFixed(0)}`}
              stroke="var(--ink-muted)"
              tick={{ fill: "var(--ink-muted)", fontSize: 10 }}
            />
            <Tooltip
              contentStyle={chartTooltipStyle}
              formatter={(value) => formatMoney(Number(value))}
            />
            <Area
              type="stepAfter"
              dataKey="cumulative"
              stroke="var(--accent)"
              strokeWidth={2}
              fill="var(--accent-soft)"
              connectNulls={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-footnotes">
        <span>
          <strong>{formatMoney(knownTotal)}</strong> known estimated
        </span>
        <span>
          <strong>{unknownCount}</strong> cost{" "}
          {unknownCount === 1 ? "unavailable" : "values unavailable"}
        </span>
        <span>
          <strong>
            {
              report.executions.filter(
                (execution) => execution.teardown_confirmed,
              ).length
            }
            /{report.executions.length}
          </strong>{" "}
          teardown confirmed
        </span>
      </div>
      <ChartDataTable
        label="Provider execution cost data"
        headings={[
          "Execution",
          "Outcome",
          "Estimated cost",
          "Known cumulative",
          "Teardown",
        ]}
        rows={data.map(({ execution, cost, knownCumulative }) => [
          execution.execution_id,
          execution.outcome,
          cost === undefined || cost === null
            ? "Unavailable"
            : formatMoney(cost),
          knownCumulative === null ? "—" : formatMoney(knownCumulative),
          execution.teardown_confirmed ? "Confirmed" : "Unconfirmed",
        ])}
      />
    </>
  );
}

function DecisionGates({ report }: { report: ResearchStudyReport }) {
  const decisions = Object.entries(report.decisions);
  return (
    <section className="dashboard-panel decision-gates">
      <header className="panel-heading">
        <h2>Decision gates</h2>
        <span>{decisions.length}</span>
      </header>
      <ol>
        {decisions.map(([id, decision]) => (
          <li key={id}>
            {decision.status === "PASS" ? (
              <CheckCircle2 aria-hidden="true" />
            ) : (
              <XCircle aria-hidden="true" />
            )}
            <strong>{humanize(id)}</strong>
            <StatusBadge status={decision.status} />
          </li>
        ))}
      </ol>
    </section>
  );
}

function ChartDataTable({
  label,
  headings,
  rows,
}: {
  label: string;
  headings: string[];
  rows: string[][];
}) {
  return (
    <details className="chart-data">
      <summary>Exact data</summary>
      <div className="table-wrap">
        <table className="data-table">
          <caption>{label}</caption>
          <thead>
            <tr>
              {headings.map((heading) => (
                <th key={heading}>{heading}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.join("-")}>
                {row.map((value, index) => (
                  <td key={`${index}-${value}`}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function ConditionLedger({ report }: { report: ResearchStudyReport }) {
  const [sort, setSort] = useState<ConditionSort>("seed");
  const [descending, setDescending] = useState(false);
  const rows = useMemo(
    () =>
      sortConditions(
        Object.entries(report.conditions).map(([id, condition]) => ({
          id,
          ...condition,
        })),
        sort,
        descending,
      ),
    [descending, report.conditions, sort],
  );

  function changeSort(next: ConditionSort) {
    if (next === sort) {
      setDescending((value) => !value);
    } else {
      setSort(next);
      setDescending(false);
    }
  }

  return (
    <section className="dashboard-panel condition-ledger">
      <header className="panel-heading">
        <h2>Condition evidence</h2>
        <span>{rows.length}</span>
      </header>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <SortableHeader
                label="Condition"
                column="id"
                active={sort}
                descending={descending}
                onSort={changeSort}
              />
              <SortableHeader
                label="Seed"
                column="seed"
                active={sort}
                descending={descending}
                onSort={changeSort}
              />
              <SortableHeader
                label="K"
                column="branch_width"
                active={sort}
                descending={descending}
                onSort={changeSort}
              />
              <th>Outcome</th>
              <th>Initial</th>
              <SortableHeader
                label="Final"
                column="final_successes"
                active={sort}
                descending={descending}
                onSort={changeSort}
              />
              <th>Paired change</th>
              <th>Completions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>
                  <span className="condition-cell">
                    <strong>{row.id}</strong>
                    <small>{humanize(row.role ?? "unspecified")}</small>
                  </span>
                </td>
                <td>{row.seed ?? "—"}</td>
                <td>{row.branch_width ?? "—"}</td>
                <td>
                  <StatusBadge status={row.status ?? "UNKNOWN"} />
                </td>
                <td>{formatPercent(row.initial_successes)}</td>
                <td>{formatPercent(row.final_successes)}</td>
                <td>{formatPairedChange(row)}</td>
                <td>{row.sampled_completions?.toLocaleString() ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SortableHeader({
  label,
  column,
  active,
  descending,
  onSort,
}: {
  label: string;
  column: ConditionSort;
  active: ConditionSort;
  descending: boolean;
  onSort: (column: ConditionSort) => void;
}) {
  const selected = active === column;
  return (
    <th
      aria-sort={
        selected ? (descending ? "descending" : "ascending") : undefined
      }
    >
      <button
        className="sort-button"
        type="button"
        onClick={() => onSort(column)}
      >
        {label}
        {selected ? (
          descending ? (
            <ArrowDown aria-hidden="true" />
          ) : (
            <ArrowUp aria-hidden="true" />
          )
        ) : (
          <ArrowUpDown aria-hidden="true" />
        )}
      </button>
    </th>
  );
}

export function sortConditions(
  rows: ConditionRow[],
  sort: ConditionSort,
  descending = false,
): ConditionRow[] {
  const direction = descending ? -1 : 1;
  return [...rows].sort((left, right) => {
    const leftValue = left[sort];
    const rightValue = right[sort];
    const leftMissing = leftValue === undefined || leftValue === null;
    const rightMissing = rightValue === undefined || rightValue === null;
    if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
    if (leftMissing) return left.id.localeCompare(right.id);
    return (
      direction *
      String(leftValue).localeCompare(String(rightValue), undefined, {
        numeric: true,
      })
    );
  });
}

export function estimatedStudyCost(report: ResearchStudyReport): number {
  return report.executions.reduce(
    (total, execution) => total + (execution.estimated_cost_usd ?? 0),
    0,
  );
}

export function unknownStudyCostCount(report: ResearchStudyReport): number {
  return report.executions.filter(
    (execution) =>
      execution.estimated_cost_usd === undefined ||
      execution.estimated_cost_usd === null,
  ).length;
}

function formatPercent(value: number | null | undefined): string {
  return value === undefined || value === null
    ? "—"
    : `${(value * 100).toFixed(1)}%`;
}

function formatPairedChange(condition: ResearchStudyCondition): string {
  if (
    condition.paired_improved === undefined ||
    condition.paired_improved === null ||
    condition.paired_regressed === undefined ||
    condition.paired_regressed === null
  ) {
    return "—";
  }
  return `+${condition.paired_improved} / −${condition.paired_regressed}`;
}

function formatMoney(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function humanize(value: string): string {
  const normalized = value.replaceAll("_", " ");
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function shortCondition(value: string): string {
  return value
    .replace("k4_train", "K4 train")
    .replace("k4_no_update", "K4 frozen")
    .replace("k1_train", "K1 train")
    .replaceAll("_", " ");
}
