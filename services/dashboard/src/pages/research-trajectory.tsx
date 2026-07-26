/**
 * THESIS: A run is understood as a sequence of verified learning checkpoints, not a KPI summary.
 * OWN-WORLD: Drafting-table lanes, ruled evidence panes, and one oxide-blue selected path.
 * STORY: Follow curriculum movement, then inspect the four sampled actions behind each saved branch.
 * FIRST VIEWPORT: One graph fills the canvas; selection opens exact verifier evidence in a fixed inspector.
 * FORM: Operate surface; React Flow graph with an equivalent outline and URL-addressed selection.
 */
import { useEffect, useMemo } from "react";
import {
  Background,
  Controls,
  type Edge,
  Handle,
  MarkerType,
  type Node,
  type NodeProps,
  Position,
  ReactFlow,
} from "@xyflow/react";
import { useApi } from "../api";
import {
  AsyncState,
  friendlyStatus,
  PageHeader,
  StatusBadge,
} from "../components";
import { Link, useParams, useSearchParams } from "../router";
import type {
  ResearchLevelObservation,
  ResearchTrajectory,
  ResearchTrajectoryCheckpoint,
  ResearchTrajectoryPromotion,
  ResearchTrajectoryResponse,
} from "../types";
import {
  branchSnapshotLabel,
  ResearchBranchWorkspace,
} from "./research-branches";

type ViewMode = "graph" | "outline";
type TrajectoryTrack = "curriculum" | "branches";
type StepKind = "baseline" | "checkpoint" | "final";

interface TrajectoryStep {
  id: string;
  kind: StepKind;
  title: string;
  update: number;
  level: number;
  exactRate: number;
  observation?: ResearchLevelObservation;
  levelObservations?: Record<string, ResearchLevelObservation>;
  checkpoint?: ResearchTrajectoryCheckpoint;
  promotion?: ResearchTrajectoryPromotion;
}

interface ResearchNodeData extends Record<string, unknown> {
  stepId?: string;
  kind?: StepKind;
  title: string;
  detail?: string;
  exactRate?: number;
  promotedTo?: number;
  level?: number;
}

type ResearchFlowNode = Node<ResearchNodeData>;

const nodeTypes = {
  checkpoint: ResearchCheckpointNode,
  lane: ResearchLaneNode,
};

export function ResearchRunTabs({
  executionId,
  active,
}: {
  executionId: string;
  active: "overview" | "trajectory";
}) {
  return (
    <nav className="run-tabs" aria-label="Run sections">
      <Link
        to={`/runs/research/${executionId}`}
        aria-current={active === "overview" ? "page" : undefined}
      >
        Overview
      </Link>
      <Link
        to={`/runs/research/${executionId}/trajectory`}
        aria-current={active === "trajectory" ? "page" : undefined}
      >
        Trajectory
      </Link>
    </nav>
  );
}

export function ResearchTrajectoryPage() {
  const { executionId = "" } = useParams<{ executionId: string }>();
  const [params, setParams] = useSearchParams();
  const response = useApi<ResearchTrajectoryResponse>(
    `/v1/research-compute-executions/${executionId}/trajectory`,
    2_000,
  );
  const run = response.data?.execution;
  const trajectory = response.data?.trajectory;
  const defaultView: ViewMode =
    typeof window !== "undefined" &&
    window.matchMedia?.("(max-width: 760px)").matches
      ? "outline"
      : "graph";
  const requestedView = params.get("view");
  const view: ViewMode =
    requestedView === "graph" || requestedView === "outline"
      ? requestedView
      : defaultView;
  const track: TrajectoryTrack =
    params.get("track") === "branches" ? "branches" : "curriculum";
  const steps = useMemo(
    () => (trajectory ? buildTrajectorySteps(trajectory) : []),
    [trajectory],
  );
  const selectedId = params.get("step") ?? steps.at(-1)?.id ?? "";
  const selectedIndex = steps.findIndex((step) => step.id === selectedId);
  const selectedStep =
    steps[selectedIndex] ?? steps.at(-1) ?? steps.at(0) ?? null;
  const branchSnapshots = trajectory?.branch_snapshots ?? [];
  const selectedSnapshot =
    branchSnapshots.find(
      (snapshot) => snapshot.snapshot_id === params.get("branch"),
    ) ??
    branchSnapshots.at(-1) ??
    null;
  const requestedSibling = Number(params.get("sibling"));
  const selectedSibling =
    selectedSnapshot?.siblings.find(
      (sibling) => sibling.index === requestedSibling,
    ) ??
    selectedSnapshot?.siblings.find(
      (sibling) => sibling.index === selectedSnapshot.best_sibling_index,
    ) ??
    selectedSnapshot?.siblings[0] ??
    null;
  const flow = useMemo(
    () =>
      trajectory
        ? buildTrajectoryFlow(trajectory, steps, selectedStep?.id ?? "")
        : { nodes: [], edges: [] },
    [selectedStep?.id, steps, trajectory],
  );

  useEffect(() => {
    if (track !== "curriculum" || !steps.length || params.get("step")) return;
    const next = new URLSearchParams(params);
    next.set("step", steps.at(-1)!.id);
    if (!params.get("view")) next.set("view", defaultView);
    setParams(next, { replace: true });
  }, [defaultView, params, setParams, steps, track]);

  useEffect(() => {
    if (
      track !== "branches" ||
      !selectedSnapshot ||
      !selectedSibling ||
      (params.get("branch") === selectedSnapshot.snapshot_id &&
        params.get("sibling") === String(selectedSibling.index))
    ) {
      return;
    }
    const next = new URLSearchParams(params);
    next.set("branch", selectedSnapshot.snapshot_id);
    next.set("sibling", String(selectedSibling.index));
    if (!next.get("view")) next.set("view", defaultView);
    setParams(next, { replace: true });
  }, [
    defaultView,
    params,
    selectedSibling,
    selectedSnapshot,
    setParams,
    track,
  ]);

  function selectStep(stepId: string) {
    const next = new URLSearchParams(params);
    next.set("step", stepId);
    if (!next.get("view")) next.set("view", defaultView);
    setParams(next);
  }

  function setView(nextView: ViewMode) {
    const next = new URLSearchParams(params);
    next.set("view", nextView);
    setParams(next);
  }

  function setTrack(nextTrack: TrajectoryTrack) {
    const next = new URLSearchParams(params);
    next.set("track", nextTrack);
    if (!next.get("view")) next.set("view", defaultView);
    if (nextTrack === "branches" && selectedSnapshot && selectedSibling) {
      next.set("branch", selectedSnapshot.snapshot_id);
      next.set("sibling", String(selectedSibling.index));
    }
    setParams(next);
  }

  function selectSnapshot(snapshotId: string) {
    const snapshot = branchSnapshots.find(
      (candidate) => candidate.snapshot_id === snapshotId,
    );
    if (!snapshot) return;
    const next = new URLSearchParams(params);
    next.set("branch", snapshot.snapshot_id);
    next.set("sibling", String(snapshot.best_sibling_index));
    setParams(next);
  }

  function selectSibling(index: number) {
    const next = new URLSearchParams(params);
    next.set("sibling", String(index));
    setParams(next);
  }

  return (
    <>
      <PageHeader
        eyebrow={<Link to="/runs">Runs</Link>}
        title={run?.name ?? "Run trajectory"}
        description={
          run
            ? `${shortModelName(run.model_id)} · static K=${run.branch_width} · adaptive complexity`
            : undefined
        }
        actions={run ? <StatusBadge status={run.status} /> : undefined}
      />
      <ResearchRunTabs executionId={executionId} active="trajectory" />
      <AsyncState loading={response.loading} error={response.error}>
        {run && !trajectory ? (
          <div className="content workspace-content">
            <div className="empty-state">
              <h2>No persisted trajectory</h2>
              <p>This run did not produce a proof with training checkpoints.</p>
            </div>
          </div>
        ) : null}
        {run && trajectory ? (
          <div className="research-trajectory-page">
            <div className="research-trajectory-toolbar">
              <div className="trajectory-toolbar-main">
                <div className="trajectory-result">
                  <span>
                    {formatPercent(trajectory.initial_exact_rate)} →{" "}
                    {formatPercent(trajectory.final_exact_rate)}
                  </span>
                  <span>
                    Level {trajectory.reached_level} of{" "}
                    {trajectory.maximum_level}
                  </span>
                  <span>{trajectory.updates_completed} updates</span>
                </div>
                {track === "branches" && branchSnapshots.length ? (
                  <label className="branch-snapshot-select">
                    <span>Checkpoint</span>
                    <select
                      value={selectedSnapshot?.snapshot_id ?? ""}
                      onChange={(event) =>
                        selectSnapshot(event.currentTarget.value)
                      }
                    >
                      {branchSnapshots.map((snapshot) => (
                        <option
                          key={snapshot.snapshot_id}
                          value={snapshot.snapshot_id}
                        >
                          {branchSnapshotLabel(snapshot)}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
              </div>
              <div className="trajectory-toolbar-controls">
                <div className="segmented" aria-label="Trajectory track">
                  <button
                    type="button"
                    aria-pressed={track === "curriculum"}
                    onClick={() => setTrack("curriculum")}
                  >
                    Curriculum
                  </button>
                  <button
                    type="button"
                    aria-pressed={track === "branches"}
                    onClick={() => setTrack("branches")}
                  >
                    Branches
                  </button>
                </div>
                <div
                  className="segmented"
                  aria-label="Trajectory representation"
                >
                  <button
                    type="button"
                    aria-pressed={view === "graph"}
                    onClick={() => setView("graph")}
                  >
                    Graph
                  </button>
                  <button
                    type="button"
                    aria-pressed={view === "outline"}
                    onClick={() => setView("outline")}
                  >
                    Outline
                  </button>
                </div>
              </div>
            </div>
            {track === "curriculum" && selectedStep ? (
              <div className="research-trajectory-workspace">
                <section
                  className="research-trajectory-canvas"
                  aria-label="Training trajectory"
                >
                  {view === "graph" ? (
                    <ReactFlow
                      nodes={flow.nodes}
                      edges={flow.edges}
                      nodeTypes={nodeTypes}
                      onNodeClick={(_, node) => {
                        const stepId = node.data.stepId;
                        if (typeof stepId === "string") selectStep(stepId);
                      }}
                      fitView
                      fitViewOptions={{ padding: 0.18 }}
                      minZoom={0.35}
                      maxZoom={1.5}
                      nodesDraggable={false}
                      nodesConnectable={false}
                      nodesFocusable
                      edgesFocusable={false}
                      autoPanOnNodeFocus
                      aria-label="Verified training checkpoints arranged by complexity level."
                      proOptions={{ hideAttribution: true }}
                    >
                      <Background color="var(--rule)" gap={32} size={1} />
                      <Controls showInteractive={false} />
                    </ReactFlow>
                  ) : (
                    <TrajectoryOutline
                      steps={steps}
                      selectedId={selectedStep.id}
                      select={selectStep}
                    />
                  )}
                </section>
                <TrajectoryInspector
                  trajectory={trajectory}
                  step={selectedStep}
                  previous={steps[selectedIndex - 1]}
                  next={steps[selectedIndex + 1]}
                  select={selectStep}
                />
              </div>
            ) : null}
            {track === "branches" && selectedSnapshot && selectedSibling ? (
              <ResearchBranchWorkspace
                snapshot={selectedSnapshot}
                selectedSibling={selectedSibling}
                view={view}
                selectSibling={selectSibling}
              />
            ) : null}
            {track === "branches" && !branchSnapshots.length ? (
              <div className="research-branch-empty">
                <h2>No saved branches</h2>
                <p>
                  Workload revision @2 did not persist sibling responses. New
                  runs save one K=4 group per domain at every evaluation.
                </p>
              </div>
            ) : null}
          </div>
        ) : null}
      </AsyncState>
    </>
  );
}

export function buildTrajectorySteps(
  trajectory: ResearchTrajectory,
): TrajectoryStep[] {
  const steps: TrajectoryStep[] = [
    {
      id: "baseline",
      kind: "baseline",
      title: "Baseline",
      update: 0,
      level: 0,
      exactRate: trajectory.initial_exact_rate,
      levelObservations: trajectory.initial_by_level,
    },
  ];
  for (const checkpoint of trajectory.checkpoints) {
    steps.push({
      id: `update-${checkpoint.update}`,
      kind: "checkpoint",
      title: `Update ${checkpoint.update}`,
      update: checkpoint.update,
      level: checkpoint.level,
      exactRate: checkpoint.exact_rate,
      observation: checkpoint,
      checkpoint,
      promotion: trajectory.promotions.find(
        (promotion) => promotion.update === checkpoint.update,
      ),
    });
  }
  steps.push({
    id: "final",
    kind: "final",
    title: "Final policy",
    update: trajectory.updates_completed,
    level: trajectory.reached_level,
    exactRate: trajectory.final_exact_rate,
    levelObservations: trajectory.final_by_level,
  });
  return steps;
}

function buildTrajectoryFlow(
  trajectory: ResearchTrajectory,
  steps: TrajectoryStep[],
  selectedId: string,
): { nodes: ResearchFlowNode[]; edges: Edge[] } {
  const laneWidth = 250;
  const rowHeight = 154;
  const nodes: ResearchFlowNode[] = [];
  for (let level = 0; level <= trajectory.maximum_level; level += 1) {
    nodes.push({
      id: `lane-${level}`,
      type: "lane",
      position: { x: level * laneWidth, y: 0 },
      selectable: false,
      focusable: false,
      draggable: false,
      data: {
        title: `Level ${level}`,
        detail: level <= trajectory.reached_level ? "Reached" : "Not reached",
        level,
      },
    });
  }
  const laneRows = new Map<number, number>();
  steps.forEach((step) => {
    const laneRow = laneRows.get(step.level) ?? 0;
    laneRows.set(step.level, laneRow + 1);
    nodes.push({
      id: step.id,
      type: "checkpoint",
      position: {
        x: step.level * laneWidth,
        y: 76 + laneRow * rowHeight,
      },
      selected: step.id === selectedId,
      data: {
        stepId: step.id,
        kind: step.kind,
        title: step.title,
        detail:
          step.kind === "checkpoint"
            ? `${step.checkpoint?.mastery_streak ?? 0} mastery window${
                step.checkpoint?.mastery_streak === 1 ? "" : "s"
              }`
            : step.kind === "baseline"
              ? "Fixed suite"
              : friendlyStatus(trajectory.stop_reason),
        exactRate: step.exactRate,
        promotedTo: step.promotion?.to_level,
        level: step.level,
      },
    });
  });
  const edges = steps.slice(1).map((step, index) => {
    const source = steps[index];
    const promoted = source.promotion !== undefined;
    return {
      id: `${source.id}-${step.id}`,
      source: source.id,
      target: step.id,
      type: "smoothstep",
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 14,
        height: 14,
      },
      animated: false,
      className: promoted ? "promotion-edge" : undefined,
      style: {
        stroke:
          source.id === selectedId || step.id === selectedId
            ? "var(--accent)"
            : "var(--rule-strong)",
        strokeWidth:
          source.id === selectedId || step.id === selectedId ? 2 : 1.25,
      },
    };
  });
  return { nodes, edges };
}

function ResearchCheckpointNode({
  data,
  selected,
}: NodeProps<ResearchFlowNode>) {
  return (
    <div
      className={`research-checkpoint-node ${selected ? "selected" : ""} ${
        data.kind ?? ""
      }`}
    >
      <Handle type="target" position={Position.Top} />
      <span>{data.title}</span>
      <strong>{formatPercent(data.exactRate ?? 0)}</strong>
      <small>{data.detail}</small>
      {data.promotedTo !== undefined ? (
        <em>Promoted to level {data.promotedTo}</em>
      ) : null}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function ResearchLaneNode({ data }: NodeProps<ResearchFlowNode>) {
  return (
    <div className="research-lane-node">
      <strong>{data.title}</strong>
      <span>{data.detail}</span>
    </div>
  );
}

export function TrajectoryOutline({
  steps,
  selectedId,
  select,
}: {
  steps: TrajectoryStep[];
  selectedId: string;
  select: (stepId: string) => void;
}) {
  return (
    <div className="research-trajectory-outline">
      <table>
        <caption>Training checkpoints</caption>
        <thead>
          <tr>
            <th>Step</th>
            <th>Level</th>
            <th>Exact</th>
            <th>RL signal</th>
            <th>Promotion</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((step) => (
            <tr
              key={step.id}
              className={step.id === selectedId ? "selected" : undefined}
            >
              <td>
                <button type="button" onClick={() => select(step.id)}>
                  {step.title}
                </button>
              </td>
              <td>{step.level}</td>
              <td>{formatPercent(step.exactRate)}</td>
              <td>
                {step.checkpoint?.informative_group_rate !== undefined
                  ? formatPercent(step.checkpoint.informative_group_rate)
                  : "—"}
              </td>
              <td>
                {step.promotion
                  ? `Level ${step.promotion.from_level} → ${step.promotion.to_level}`
                  : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TrajectoryInspector({
  trajectory,
  step,
  previous,
  next,
  select,
}: {
  trajectory: ResearchTrajectory;
  step: TrajectoryStep;
  previous?: TrajectoryStep;
  next?: TrajectoryStep;
  select: (stepId: string) => void;
}) {
  const checkpoint = step.checkpoint;
  const observations = step.levelObservations
    ? Object.entries(step.levelObservations).sort(
        ([left], [right]) => Number(left) - Number(right),
      )
    : [];
  return (
    <aside className="research-trajectory-inspector" aria-label="Selected step">
      <header>
        <span>
          {step.kind === "checkpoint" ? `Level ${step.level}` : "All levels"}
        </span>
        {step.promotion ? (
          <StatusBadge status="PROMOTED" />
        ) : (
          <StatusBadge
            status={step.kind === "final" ? "SUCCEEDED" : "VERIFIED"}
          />
        )}
      </header>
      <div className="trajectory-selected-step">
        <h2>{step.title}</h2>
        <strong>{formatPercent(step.exactRate)}</strong>
        <span>held-out exact</span>
      </div>
      {checkpoint ? (
        <>
          <dl className="trajectory-step-facts">
            <Fact
              label="Branch pass"
              value={formatOptionalPercent(
                checkpoint.training_branch_pass_rate,
              )}
            />
            <Fact
              label="RL signal"
              value={formatOptionalPercent(checkpoint.informative_group_rate)}
            />
            <Fact
              label="Teacher fallback"
              value={formatOptionalPercent(checkpoint.teacher_fallback_rate)}
            />
            <Fact
              label="Mastery"
              value={`${checkpoint.mastery_streak ?? 0} / 2 windows`}
            />
            <Fact
              label="Policy updates"
              value={String(checkpoint.policy_update_count ?? 0)}
            />
            <Fact
              label="Teacher updates"
              value={String(checkpoint.teacher_update_count ?? 0)}
            />
            <Fact
              label="Gradient norm"
              value={formatDecimal(checkpoint.gradient_norm)}
            />
            <Fact
              label="Elapsed"
              value={formatDuration(checkpoint.elapsed_seconds)}
            />
          </dl>
          <DomainResults observation={checkpoint} />
        </>
      ) : (
        <div className="trajectory-level-results">
          {observations.map(([level, observation]) => (
            <div key={level}>
              <span>Level {level}</span>
              <strong>{formatPercent(observation.exact_rate)}</strong>
            </div>
          ))}
        </div>
      )}
      {step.promotion ? (
        <div className="trajectory-promotion">
          <span>Complexity promotion</span>
          <strong>
            Level {step.promotion.from_level} → {step.promotion.to_level}
          </strong>
          <small>
            {step.promotion.mastery_windows} mastery windows · minimum domain{" "}
            {formatPercent(step.promotion.minimum_domain_exact_rate)}
          </small>
        </div>
      ) : null}
      {step.kind === "final" ? (
        <dl className="trajectory-step-facts final">
          <Fact
            label="Exact gain"
            value={`${trajectory.exact_gain >= 0 ? "+" : ""}${(
              trajectory.exact_gain * 100
            ).toFixed(1)} pts`}
          />
          <Fact
            label="Completions"
            value={trajectory.total_sampled_completions.toLocaleString()}
          />
          <Fact
            label="RL signal"
            value={formatPercent(trajectory.informative_group_rate)}
          />
          <Fact
            label="Teacher fallback"
            value={formatPercent(trajectory.teacher_fallback_rate)}
          />
        </dl>
      ) : null}
      <div className="step-navigation" aria-label="Step navigation">
        <button
          type="button"
          disabled={!previous}
          onClick={() => previous && select(previous.id)}
        >
          ← Previous
        </button>
        <button
          type="button"
          disabled={!next}
          onClick={() => next && select(next.id)}
        >
          Next →
        </button>
      </div>
    </aside>
  );
}

function DomainResults({
  observation,
}: {
  observation: ResearchLevelObservation;
}) {
  return (
    <div className="trajectory-domain-results">
      {Object.entries(observation.per_domain).map(([domain, result]) => (
        <div key={domain}>
          <span>{friendlyStatus(domain)}</span>
          <strong>{formatPercent(result.exact_rate)}</strong>
          <small>{result.mean_reward.toFixed(3)} reward</small>
        </div>
      ))}
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatOptionalPercent(value: number | undefined): string {
  return value === undefined ? "—" : formatPercent(value);
}

function formatDecimal(value: number | undefined): string {
  return value === undefined ? "—" : value.toFixed(3);
}

function formatDuration(value: number | undefined): string {
  if (value === undefined) return "—";
  if (value < 60) return `${Math.round(value)} sec`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

function shortModelName(value: string | null): string {
  if (!value) return "Model pending";
  return value.split("/").at(-1) ?? value;
}
