import { useEffect, useMemo, useState } from "react";
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
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
  CircleDollarSign,
  Database,
  GitBranch,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { formatDistanceToNowStrict } from "date-fns";
import { useApi } from "../api";
import { MachineId, PageHeader, StatusBadge } from "../components";
import { revision30Study, type Revision30Condition } from "../data/revision30";
import { Link } from "../router";
import type { RunSummary } from "../types";

interface EnvironmentSummary {
  usage: {
    rollout_trees: number;
    committed_iterations: number;
    verification_runs: number;
  };
}

interface ConditionRow extends Revision30Condition {
  externalSuccesses: number | null;
  externalRate: number | null;
}

const chartTooltipStyle = {
  background: "var(--surface)",
  border: "1px solid var(--rule-strong)",
  borderRadius: "4px",
  color: "var(--ink)",
  fontSize: "0.75rem",
};

const percent = (value: number) =>
  new Intl.NumberFormat("en", {
    style: "percent",
    minimumFractionDigits: 0,
    maximumFractionDigits: 1,
  }).format(value);

const money = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);

const utcTimestamp = (value: string) =>
  new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).format(new Date(value));

export function DashboardPage() {
  const runs = useApi<{ items: RunSummary[]; total: number }>(
    "/v1/runs?limit=100&sort=updated_desc",
    10_000,
  );
  const proofs = useApi<{ total: number }>("/v1/proofs?limit=1", 10_000);
  const environments = useApi<{ items: EnvironmentSummary[] }>(
    "/v1/environments",
    15_000,
  );

  const environmentUsage = environments.data?.items.reduce(
    (total, item) => ({
      rollout_trees: total.rollout_trees + item.usage.rollout_trees,
      committed_iterations:
        total.committed_iterations + item.usage.committed_iterations,
      verification_runs: total.verification_runs + item.usage.verification_runs,
    }),
    { rollout_trees: 0, committed_iterations: 0, verification_runs: 0 },
  );
  const localCost =
    runs.data?.items.reduce(
      (total, run) => total + run.cost.execution_credits,
      0,
    ) ?? null;

  return (
    <>
      <PageHeader
        eyebrow="Research command center"
        title="Dashboard"
        description="One decision surface for live contract operations and the archived confirmatory study—separated by evidence profile."
        actions={
          <>
            <Link className="button secondary" to="/proofs">
              Inspect proofs
            </Link>
            <Link className="button primary" to="/runs/new">
              Launch run
            </Link>
          </>
        }
      />
      <div className="content dashboard-content">
        <section
          className="dashboard-decision"
          aria-labelledby="study-decision"
        >
          <div className="decision-copy">
            <span className="decision-kicker">
              Revision 30 · confirmatory readout
            </span>
            <h2 id="study-decision">
              The study did not confirm a K=4 training advantage.
            </h2>
            <p>
              Internal gains failed to repeat across both fresh seeds, K=4 did
              not beat either matched control. External evaluation improved two
              of 45 adapter-task pairs and regressed one. Regression guards and
              complete external coverage passed.
            </p>
          </div>
          <div className="decision-verdict">
            <StatusBadge status={revision30Study.overallStatus} />
            <strong>2 of 6 gates passed</strong>
            <span>
              Report frozen {utcTimestamp(revision30Study.source.generatedAt)}{" "}
              UTC
            </span>
          </div>
          <div className="decision-source">
            <span>
              Main worktree source <code>{revision30Study.source.path}</code>
            </span>
            <MachineId
              value={revision30Study.source.reportDigest}
              copy={false}
            />
          </div>
        </section>

        <section className="live-instruments" aria-labelledby="live-profile">
          <div className="live-heading">
            <span className="live-signal" aria-hidden="true" />
            <div>
              <h2 id="live-profile">Live local-contract profile</h2>
              <p>Mock providers only · operational counts, not model claims</p>
            </div>
          </div>
          <Instrument
            icon={<Database aria-hidden="true" />}
            label="Runs"
            value={liveValue(runs.loading, runs.error, runs.data?.total)}
            detail={
              runs.data
                ? `${runs.data.items.filter((run) => run.status === "SUCCEEDED").length} completed`
                : "Reading control plane"
            }
          />
          <Instrument
            icon={<ShieldCheck aria-hidden="true" />}
            label="Proof bundles"
            value={liveValue(proofs.loading, proofs.error, proofs.data?.total)}
            detail={`${environmentUsage?.verification_runs ?? "—"} verifications`}
          />
          <Instrument
            icon={<GitBranch aria-hidden="true" />}
            label="Committed updates"
            value={liveValue(
              environments.loading,
              environments.error,
              environmentUsage?.committed_iterations,
            )}
            detail={`${environmentUsage?.rollout_trees ?? "—"} rollout trees`}
          />
          <Instrument
            icon={<CircleDollarSign aria-hidden="true" />}
            label="Execution cost"
            value={
              localCost === null
                ? liveValue(runs.loading, runs.error, undefined)
                : `${localCost.toFixed(4)} cr`
            }
            detail="Local fixture credits"
          />
        </section>

        <div className="dashboard-primary-grid">
          <ChartPanel
            title="Internal evaluation"
            description="Paired test rate before and after each completed condition. The failed seed has no evaluation."
            aside="48 tasks per completed condition"
          >
            <InternalEvaluationChart />
          </ChartPanel>
          <section className="dashboard-panel decision-gates">
            <header className="panel-heading">
              <div>
                <h2>Confirmatory gates</h2>
                <p>
                  Every predeclared question retains its pass or fail outcome.
                </p>
              </div>
              <span>6 decisions</span>
            </header>
            <ol>
              {revision30Study.decisions.map((decision) => (
                <li key={decision.id}>
                  {decision.status === "PASS" ? (
                    <CheckCircle2 aria-hidden="true" />
                  ) : (
                    <XCircle aria-hidden="true" />
                  )}
                  <span>
                    <strong>{decision.label}</strong>
                    <small>{decision.detail}</small>
                  </span>
                  <StatusBadge status={decision.status} />
                </li>
              ))}
            </ol>
          </section>
        </div>

        <div className="dashboard-chart-grid">
          <ChartPanel
            title="Post-freeze external pack"
            description="Exact task success on nine disjoint tasks. This is transfer evidence, shown separately from internal evaluation."
            aside="3 domains · 3 tasks each"
          >
            <ExternalEvaluationChart />
          </ChartPanel>
          <ChartPanel
            title="Provider spend"
            description="Cumulative estimated RunPod cost across every recorded execution, including failed attempts."
            aside={`${revision30Study.executions.length} executions`}
          >
            <ProviderCostChart />
          </ChartPanel>
        </div>

        <ConditionLedger />

        <section className="dashboard-panel provenance-panel">
          <header className="panel-heading">
            <div>
              <h2>Frozen study provenance</h2>
              <p>{revision30Study.aggregationPolicy}</p>
            </div>
            <StatusBadge status="IMMUTABLE" />
          </header>
          <dl>
            <div>
              <dt>Study</dt>
              <dd>
                <code>{revision30Study.studyId}</code>
              </dd>
            </div>
            <div>
              <dt>Workload</dt>
              <dd>
                <code>{revision30Study.freeze.workload}</code>
              </dd>
            </div>
            <div>
              <dt>Model</dt>
              <dd>{revision30Study.freeze.model}</dd>
            </div>
            <div>
              <dt>Freeze commit</dt>
              <dd>
                <code>{revision30Study.freeze.commit}</code>
              </dd>
            </div>
            <div>
              <dt>Imported from</dt>
              <dd>
                <code>
                  {revision30Study.source.worktreeCommit.slice(0, 12)}
                </code>
              </dd>
            </div>
            <div>
              <dt>Age</dt>
              <dd>
                {formatDistanceToNowStrict(
                  new Date(revision30Study.source.generatedAt),
                  { addSuffix: true },
                )}
              </dd>
            </div>
          </dl>
        </section>
      </div>
    </>
  );
}

export default DashboardPage;

function liveValue(
  loading: boolean,
  error: Error | null,
  value: number | undefined,
) {
  if (value !== undefined) return String(value);
  if (loading) return "…";
  if (error) return "Unavailable";
  return "0";
}

function Instrument({
  icon,
  label,
  value,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="instrument">
      <span className="instrument-icon">{icon}</span>
      <span>
        <small>{label}</small>
        <strong>{value}</strong>
      </span>
      <span className="instrument-detail">{detail}</span>
    </div>
  );
}

function ChartPanel({
  title,
  description,
  aside,
  children,
}: {
  title: string;
  description: string;
  aside: string;
  children: React.ReactNode;
}) {
  return (
    <section className="dashboard-panel chart-panel">
      <header className="panel-heading">
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        <span>{aside}</span>
      </header>
      {children}
    </section>
  );
}

function InternalEvaluationChart() {
  const compact = useMediaQuery("(max-width: 520px)");
  const data = revision30Study.conditions.map((condition) => ({
    condition: compact
      ? condition.label
          .replace("K=4 train", "K4 train")
          .replace("K=4 frozen", "K4 frozen")
          .replace("K=1 train", "K1 train")
      : condition.label.replace(" · ", "\n"),
    initial:
      condition.initialRate === null ? undefined : condition.initialRate * 100,
    final: condition.finalRate === null ? undefined : condition.finalRate * 100,
  }));

  return (
    <>
      <div
        className={`chart-frame${compact ? " compact-internal-chart" : ""}`}
        role="img"
        aria-label="Grouped bar chart comparing initial and final internal evaluation rates by study condition."
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout={compact ? "vertical" : "horizontal"}
            margin={
              compact
                ? { top: 10, right: 16, left: 12, bottom: 8 }
                : { top: 14, right: 12, left: -10, bottom: 36 }
            }
          >
            <CartesianGrid
              stroke="var(--rule)"
              vertical={!compact}
              horizontal={compact}
            />
            <XAxis
              dataKey={compact ? undefined : "condition"}
              type={compact ? "number" : "category"}
              domain={compact ? [0, 60] : undefined}
              tickFormatter={compact ? (value) => `${value}%` : undefined}
              stroke="var(--ink-muted)"
              tick={{ fill: "var(--ink-muted)", fontSize: 10 }}
              interval={0}
              height={compact ? 30 : 54}
            />
            <YAxis
              dataKey={compact ? "condition" : undefined}
              type={compact ? "category" : "number"}
              domain={compact ? undefined : [0, 60]}
              tickFormatter={compact ? undefined : (value) => `${value}%`}
              width={compact ? 112 : 60}
              stroke="var(--ink-muted)"
              tick={{ fill: "var(--ink-muted)", fontSize: 10 }}
            />
            <Tooltip
              contentStyle={chartTooltipStyle}
              formatter={(value) => [`${Number(value).toFixed(1)}%`]}
            />
            <Legend wrapperStyle={{ fontSize: "0.72rem" }} />
            <Bar
              dataKey="initial"
              name="Initial"
              fill="var(--rule-strong)"
              radius={compact ? [0, 2, 2, 0] : [2, 2, 0, 0]}
              isAnimationActive={false}
            />
            <Bar
              dataKey="final"
              name="Final"
              fill="var(--accent)"
              radius={compact ? [0, 2, 2, 0] : [2, 2, 0, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ChartDataTable
        label="Internal evaluation data"
        headings={["Condition", "Initial", "Final"]}
        rows={revision30Study.conditions.map((condition) => [
          condition.label,
          condition.initialRate === null
            ? "Not evaluated"
            : percent(condition.initialRate),
          condition.finalRate === null
            ? "Not evaluated"
            : percent(condition.finalRate),
        ])}
      />
    </>
  );
}

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(
    () => typeof window !== "undefined" && window.matchMedia(query).matches,
  );

  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [query]);

  return matches;
}

function ExternalEvaluationChart() {
  const data = [
    {
      condition: "Disabled adapter",
      successes: revision30Study.external.base.exactSuccesses,
    },
    ...revision30Study.external.adapters.map((adapter) => ({
      condition:
        revision30Study.conditions.find(
          (condition) => condition.id === adapter.conditionId,
        )?.label ?? adapter.conditionId,
      successes: adapter.exactSuccesses,
    })),
  ];

  return (
    <>
      <div
        className="chart-frame"
        role="img"
        aria-label="Horizontal bar chart of exact successes out of nine external tasks for the disabled adapter and five retained policies."
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 18, left: 30, bottom: 8 }}
          >
            <CartesianGrid stroke="var(--rule)" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 9]}
              ticks={[0, 3, 6, 9]}
              stroke="var(--ink-muted)"
              tick={{ fill: "var(--ink-muted)", fontSize: 10 }}
            />
            <YAxis
              type="category"
              dataKey="condition"
              width={126}
              stroke="var(--ink-muted)"
              tick={{ fill: "var(--ink-muted)", fontSize: 10 }}
            />
            <ReferenceLine
              x={1}
              stroke="var(--rule-strong)"
              strokeDasharray="3 3"
            />
            <Tooltip
              contentStyle={chartTooltipStyle}
              formatter={(value) => [
                `${String(value)} of 9`,
                "Exact successes",
              ]}
            />
            <Bar
              dataKey="successes"
              name="Exact successes"
              fill="var(--accent)"
              radius={[0, 2, 2, 0]}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ChartDataTable
        label="External evaluation data"
        headings={["Policy", "Exact successes", "Rate"]}
        rows={data.map((item) => [
          item.condition,
          `${item.successes} of 9`,
          percent(item.successes / 9),
        ])}
      />
    </>
  );
}

function ProviderCostChart() {
  let cumulative = 0;
  const data = revision30Study.executions.map((execution, index) => {
    cumulative += execution.costUsd ?? 0;
    return {
      attempt: index + 1,
      cumulative: Number(cumulative.toFixed(6)),
      cost: execution.costUsd ?? 0,
      outcome: execution.outcome,
    };
  });

  return (
    <>
      <div
        className="chart-frame"
        role="img"
        aria-label="Area chart of cumulative estimated provider cost across 22 recorded executions."
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
              label={{
                value: "Execution",
                position: "insideBottom",
                offset: -3,
                fill: "var(--ink-muted)",
                fontSize: 10,
              }}
            />
            <YAxis
              tickFormatter={(value) => `$${Number(value).toFixed(0)}`}
              stroke="var(--ink-muted)"
              tick={{ fill: "var(--ink-muted)", fontSize: 10 }}
            />
            <Tooltip
              contentStyle={chartTooltipStyle}
              formatter={(value, name) => [
                money(Number(value)),
                name === "cumulative" ? "Cumulative cost" : "Execution cost",
              ]}
            />
            <Area
              type="stepAfter"
              dataKey="cumulative"
              stroke="var(--accent)"
              strokeWidth={2}
              fill="var(--accent-soft)"
              name="cumulative"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-footnotes">
        <span>
          <strong>{money(cumulative)}</strong> recorded spend
        </span>
        <span>
          <strong>
            {
              revision30Study.executions.filter(
                (execution) => execution.outcome === "FAILED",
              ).length
            }
          </strong>{" "}
          failed provider executions included
        </span>
        <span>
          <strong>22/22</strong> teardown receipts confirmed
        </span>
      </div>
    </>
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
      <summary>View exact data</summary>
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
                  <td key={`${value}-${index}`}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function ConditionLedger() {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "seed", desc: false },
  ]);
  const data = useMemo<ConditionRow[]>(
    () =>
      revision30Study.conditions.map((condition) => {
        const external = revision30Study.external.adapters.find(
          (adapter) => adapter.conditionId === condition.id,
        );
        return {
          ...condition,
          externalSuccesses: external?.exactSuccesses ?? null,
          externalRate: external?.exactRate ?? null,
        };
      }),
    [],
  );
  const columns = useMemo<ColumnDef<ConditionRow>[]>(
    () => [
      {
        accessorKey: "label",
        header: "Condition",
        cell: ({ row }) => (
          <span className="condition-cell">
            <strong>{row.original.label}</strong>
            <small>{row.original.role}</small>
          </span>
        ),
      },
      {
        accessorKey: "seed",
        header: "Seed",
      },
      {
        accessorKey: "status",
        header: "Execution",
        cell: ({ getValue }) => <StatusBadge status={String(getValue())} />,
      },
      {
        id: "internal",
        accessorFn: (row) => row.finalRate,
        header: "Internal eval",
        cell: ({ row }) =>
          row.original.initialRate === null ||
          row.original.finalRate === null ? (
            "Not evaluated"
          ) : (
            <span className="rate-change">
              {percent(row.original.initialRate)}
              <span aria-hidden="true">→</span>
              <strong>{percent(row.original.finalRate)}</strong>
            </span>
          ),
      },
      {
        accessorKey: "externalRate",
        header: "External pack",
        cell: ({ row }) =>
          row.original.externalSuccesses === null
            ? "Not retained"
            : `${row.original.externalSuccesses}/9 · ${percent(row.original.externalRate!)}`,
      },
      {
        accessorKey: "policyUpdates",
        header: "Updates",
        cell: ({ getValue }) =>
          getValue<number | null>() === null ? "—" : String(getValue()),
      },
      {
        accessorKey: "sampledCompletions",
        header: "Completions",
        cell: ({ getValue }) =>
          getValue<number | null>() === null
            ? "—"
            : Number(getValue()).toLocaleString(),
      },
    ],
    [],
  );
  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    enableSortingRemoval: false,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <section className="dashboard-panel condition-ledger">
      <header className="panel-heading">
        <div>
          <h2>Condition evidence ledger</h2>
          <p>
            Per-seed outcomes stay visible; sorting never replaces the
            individual record with an average.
          </p>
        </div>
        <span>Sortable columns</span>
      </header>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    aria-sort={
                      header.column.getIsSorted() === "asc"
                        ? "ascending"
                        : header.column.getIsSorted() === "desc"
                          ? "descending"
                          : undefined
                    }
                  >
                    {header.isPlaceholder ? null : header.column.getCanSort() ? (
                      <button
                        className="sort-button"
                        type="button"
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                        {header.column.getIsSorted() === "asc" ? (
                          <ArrowUp aria-hidden="true" />
                        ) : header.column.getIsSorted() === "desc" ? (
                          <ArrowDown aria-hidden="true" />
                        ) : (
                          <ArrowUpDown aria-hidden="true" />
                        )}
                      </button>
                    ) : (
                      flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
