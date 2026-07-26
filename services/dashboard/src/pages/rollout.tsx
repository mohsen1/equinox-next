import { useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  type Edge,
  MarkerType,
  type Node,
  ReactFlow,
} from "@xyflow/react";
import { Link, useParams, useSearchParams } from "../router";
import { api, useApi } from "../api";
import {
  AsyncState,
  displayRunName,
  EvidenceRender,
  formatDate,
  KeyValue,
  MachineId,
  Notice,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";
import type {
  GraphEdge,
  ProofBundle,
  RolloutGraph,
  VerificationDetail,
  VerificationStep,
} from "../types";

interface BranchComparison {
  branch_group: Record<string, any>;
  members: Array<
    Record<string, any> & {
      proof_bundle: ProofBundle | null;
      reward_signals: Array<Record<string, any>>;
    }
  >;
  model_assessment: Record<string, any> | null;
  presentation_order: number[];
  member_bindings: Array<{
    label: string;
    branch_member_id: string;
    sibling_index: number;
    proof_bundle_digest: string;
  }>;
}

export function RolloutTreePage() {
  const { treeId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const graph = useApi<RolloutGraph>(
    `/v1/rollout-trees/${treeId}/graph`,
    1_000,
  );
  const selectedVerification = params.get("verification");
  const verification = useApi<VerificationDetail>(
    selectedVerification
      ? `/v1/verification-runs/${selectedVerification}`
      : null,
    1_000,
  );
  const branchGroupId =
    params.get("branch") ?? graph.data?.branch_groups[0]?.branch_group_id;
  const comparison = useApi<BranchComparison>(
    branchGroupId ? `/v1/branch-groups/${branchGroupId}/comparison` : null,
    1_000,
  );
  const [rejudging, setRejudging] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const defaultView =
    typeof window !== "undefined" &&
    window.matchMedia?.("(max-width: 760px)").matches
      ? "outline"
      : "graph";
  const view = params.get("view") ?? defaultView;
  const outcomeFilter = params.get("filter") ?? "all";
  const filteredGraph = useMemo(
    () =>
      graph.data
        ? filterGraph(graph.data, outcomeFilter)
        : (graph.data ?? null),
    [graph.data, outcomeFilter],
  );

  useEffect(() => {
    if (!filteredGraph) return;
    const selectedMember = params.get("member");
    const memberVisible =
      !selectedMember ||
      filteredGraph.branch_members.some(
        (member) => member.branch_member_id === selectedMember,
      );
    if (params.get("state") && memberVisible) return;
    const defaultEdge =
      [...filteredGraph.edges]
        .reverse()
        .find((edge) => edge.branch_member_id) ?? filteredGraph.edges.at(-1);
    if (!defaultEdge) return;
    const next = new URLSearchParams(params);
    next.set("state", defaultEdge.target);
    next.set("verification", defaultEdge.verification_run_id);
    if (defaultEdge.branch_member_id)
      next.set("member", defaultEdge.branch_member_id);
    else next.delete("member");
    if (filteredGraph.branch_groups[0]) {
      next.set(
        "branch",
        String(filteredGraph.branch_groups[0].branch_group_id),
      );
    }
    if (!params.get("view")) next.set("view", defaultView);
    setParams(next, { replace: true });
  }, [defaultView, filteredGraph, params, setParams]);

  const selectedState = params.get("state");
  const selectedStep = params.get("step");
  const selectedEdge = filteredGraph?.edges.find(
    (edge) => edge.target === selectedState,
  );
  const selectedStateRecord = filteredGraph?.nodes.find(
    (state) => state.id === selectedState,
  );
  const selectedMember = filteredGraph?.branch_members.find(
    (member) => member.branch_member_id === selectedEdge?.branch_member_id,
  );
  const causalEdges = filteredGraph
    ? causalEdgeSequence(filteredGraph, selectedEdge?.branch_member_id ?? null)
    : [];
  const selectedEdgeIndex = selectedEdge
    ? causalEdges.findIndex((edge) => edge.id === selectedEdge.id)
    : -1;
  function selectEdge(edge: GraphEdge) {
    const next = new URLSearchParams(params);
    next.set("state", edge.target);
    next.set("verification", edge.verification_run_id);
    if (edge.branch_member_id) next.set("member", edge.branch_member_id);
    else next.delete("member");
    setParams(next);
  }
  const flow = useMemo(
    () =>
      filteredGraph
        ? buildFlow(
            filteredGraph,
            selectedState,
            selectedEdge?.branch_member_id ?? null,
            (stateId, edge) => {
              const next = new URLSearchParams(params);
              next.set("state", stateId);
              if (edge) next.set("verification", edge.verification_run_id);
              if (edge?.branch_member_id)
                next.set("member", edge.branch_member_id);
              else next.delete("member");
              setParams(next);
            },
          )
        : { nodes: [], edges: [] },
    [
      filteredGraph,
      params,
      selectedEdge?.branch_member_id,
      selectedState,
      setParams,
    ],
  );

  const proof = verification.data?.proof_bundle?.manifest;
  async function rejudge() {
    const runId = verification.data?.verification_run.run_id;
    if (!runId || !selectedVerification) return;
    setRejudging(true);
    setActionError(null);
    try {
      const result = await api<{ verification_run_id: string }>(
        `/v1/runs/${runId}/rejudge`,
        {
          method: "POST",
          body: JSON.stringify({
            verification_run_id: selectedVerification,
            fixture_scenario: "valid",
          }),
        },
      );
      const next = new URLSearchParams(params);
      next.set("verification", result.verification_run_id);
      next.set("step", "pointwise-judge");
      setParams(next);
    } catch (cause) {
      setActionError(
        cause instanceof Error ? cause.message : "Rejudge failed.",
      );
    } finally {
      setRejudging(false);
    }
  }

  return (
    <>
      <PageHeader
        eyebrow={
          <>
            {graph.data?.tree.run_id ? (
              <Link to={`/runs/${String(graph.data.tree.run_id)}?view=trace`}>
                Run trace
              </Link>
            ) : (
              <Link to="/runs">Runs</Link>
            )}{" "}
            / trajectory
          </>
        }
        title="Trajectory explorer"
        description={
          graph.data ? (
            <>
              {displayRunName(String(graph.data.tree.run_name ?? "Run"))} ·{" "}
              {graph.data.edges.length} actions ·{" "}
              {graph.data.branch_members.length
                ? `static K=${graph.data.branch_members.length}`
                : "independent"}
            </>
          ) : (
            <MachineId value={treeId} />
          )
        }
        actions={
          <>
            <div className="segmented" aria-label="Trajectory representation">
              <button
                type="button"
                aria-pressed={view === "graph"}
                onClick={() => updateParam(params, setParams, "view", "graph")}
              >
                Graph
              </button>
              <button
                type="button"
                aria-pressed={view === "outline"}
                onClick={() =>
                  updateParam(params, setParams, "view", "outline")
                }
              >
                Outline
              </button>
            </div>
            <button
              className="button secondary"
              type="button"
              onClick={rejudge}
              disabled={rejudging}
            >
              {rejudging ? "Rejudging…" : "Rejudge proof"}
            </button>
          </>
        }
      />
      <AsyncState
        loading={graph.loading}
        error={graph.error}
        empty={Boolean(graph.data && !graph.data.nodes.length)}
      >
        {graph.data && filteredGraph ? (
          <div className="trajectory-page">
            {actionError ? (
              <Notice tone="negative" title="Action failed">
                {actionError}
              </Notice>
            ) : null}
            <section className="trajectory-summary">
              <div className="trace-reading">
                <span>Start</span>
                <i aria-hidden="true" />
                <span>
                  Shared prefix
                  <small>
                    {
                      filteredGraph.edges.filter(
                        (edge) => !edge.branch_member_id,
                      ).length
                    }{" "}
                    actions
                  </small>
                </span>
                <i aria-hidden="true" />
                <strong>Checkpoint</strong>
                <i className="fan" aria-hidden="true" />
                <span>
                  K={staticBranchWidth(filteredGraph)}
                  <small>
                    {filteredGraph.branch_members.length} visible sibling
                    {filteredGraph.branch_members.length === 1 ? "" : "s"}
                  </small>
                </span>
              </div>
              <dl>
                <div>
                  <dt>States</dt>
                  <dd>{filteredGraph.nodes.length}</dd>
                </div>
                <div>
                  <dt>Actions</dt>
                  <dd>{filteredGraph.edges.length}</dd>
                </div>
                <div>
                  <dt>Exceptions</dt>
                  <dd>
                    {
                      filteredGraph.branch_members.filter(
                        (member) =>
                          member.failure_mode || member.status !== "SUCCEEDED",
                      ).length
                    }
                  </dd>
                </div>
              </dl>
            </section>
            <div className="trajectory-controls">
              <label>
                Show
                <select
                  value={params.get("filter") ?? "all"}
                  onChange={(event) =>
                    updateParam(params, setParams, "filter", event.target.value)
                  }
                >
                  <option value="all">All typed outcomes</option>
                  <option value="accepted">Accepted</option>
                  <option value="exceptions">
                    Retries, failures, abstentions
                  </option>
                </select>
              </label>
              <span>
                {filteredGraph.nodes.length} of {graph.data.nodes.length} states
                · {filteredGraph.branch_members.length} of{" "}
                {graph.data.branch_members.length} sibling paths
              </span>
              {graph.data.environment_snapshots[0] ? (
                <span>
                  Snapshot fidelity: <code>logical_restore</code>
                </span>
              ) : null}
            </div>
            <div className="trajectory-workspace">
              <section className="graph-region" aria-label="Rollout trajectory">
                {view === "graph" ? (
                  <ReactFlow
                    nodes={flow.nodes}
                    edges={flow.edges}
                    fitView
                    minZoom={0.35}
                    maxZoom={1.5}
                    nodesDraggable={false}
                    nodesConnectable={false}
                    elementsSelectable
                    nodesFocusable
                    edgesFocusable
                    autoPanOnNodeFocus
                    aria-label="Branch-aware rollout tree. Use Tab to move through states."
                    proOptions={{ hideAttribution: true }}
                  >
                    <Background color="var(--rule)" gap={32} size={1} />
                    <Controls showInteractive={false} />
                  </ReactFlow>
                ) : (
                  <TrajectoryOutline
                    graph={filteredGraph}
                    selectedState={selectedState}
                    select={(stateId, edge) => {
                      const next = new URLSearchParams(params);
                      next.set("state", stateId);
                      if (edge)
                        next.set("verification", edge.verification_run_id);
                      if (edge?.branch_member_id)
                        next.set("member", edge.branch_member_id);
                      setParams(next);
                    }}
                  />
                )}
              </section>
              <TraceInspector
                graph={filteredGraph}
                state={selectedStateRecord}
                edge={selectedEdge}
                member={selectedMember}
                previous={
                  selectedEdgeIndex > 0
                    ? causalEdges[selectedEdgeIndex - 1]
                    : undefined
                }
                next={
                  selectedEdgeIndex >= 0
                    ? causalEdges[selectedEdgeIndex + 1]
                    : undefined
                }
                select={selectEdge}
              />
            </div>
            <div className="branch-lane-strip" aria-label="Sibling paths">
              <span className="branch-lane-label">Sibling paths</span>
              {filteredGraph.branch_members.map((member) => {
                const memberEdges = graph.data!.edges.filter(
                  (edge) => edge.branch_member_id === member.branch_member_id,
                );
                const terminalEdge = memberEdges.at(-1);
                return (
                  <button
                    type="button"
                    key={member.branch_member_id}
                    aria-pressed={
                      selectedMember?.branch_member_id ===
                      member.branch_member_id
                    }
                    onClick={() => terminalEdge && selectEdge(terminalEdge)}
                  >
                    <span>
                      <strong>Sibling {member.sibling_index + 1}</strong>
                      <small>{memberEdges.length} actions</small>
                    </span>
                    <span className="lane-outcomes">
                      <small>
                        Exec{" "}
                        {(member.terminal_outcome ?? "complete")
                          .toLowerCase()
                          .replaceAll("_", " ")}
                      </small>
                      <small>
                        Verify{" "}
                        {(member.verification_status ?? "unknown")
                          .toLowerCase()
                          .replaceAll("_", " ")}
                        {member.failure_mode
                          ? ` · ${member.failure_mode.toLowerCase().replaceAll("_", " ")}`
                          : ""}
                      </small>
                      {Number(member.retry_count ?? 0) > 0 ? (
                        <small>{member.retry_count} recovered retries</small>
                      ) : null}
                      <StatusBadge
                        status={member.eligibility_status ?? member.status}
                      />
                    </span>
                  </button>
                );
              })}
            </div>
            <EvidenceComparator
              proof={proof}
              comparison={comparison.data}
              selectedMember={params.get("member")}
              visibleMemberIds={filteredGraph.branch_members.map(
                (member) => member.branch_member_id,
              )}
            />
            <div className="proof-inspector-grid">
              <VerificationSummary
                loading={verification.loading}
                error={verification.error}
                detail={verification.data}
                selectedStep={selectedStep}
                selectStep={(step) =>
                  updateParam(params, setParams, "step", step.step_id)
                }
              />
              <RewardSummary
                verificationId={selectedVerification}
                proof={proof}
                comparison={comparison.data}
                selectedMember={
                  comparison.data?.members.find(
                    (member) =>
                      member.branch_member_id === params.get("member"),
                  ) ?? null
                }
              />
            </div>
          </div>
        ) : null}
      </AsyncState>
    </>
  );
}

function TraceInspector({
  graph,
  state,
  edge,
  member,
  previous,
  next,
  select,
}: {
  graph: RolloutGraph;
  state: RolloutGraph["nodes"][number] | undefined;
  edge: GraphEdge | undefined;
  member: RolloutGraph["branch_members"][number] | undefined;
  previous: GraphEdge | undefined;
  next: GraphEdge | undefined;
  select: (edge: GraphEdge) => void;
}) {
  const checkpoint = graph.decision_checkpoints[0];
  const snapshot = graph.environment_snapshots[0];
  const actionArguments = edge
    ? Object.fromEntries(
        Object.entries(edge.action).filter(([key]) => key !== "kind"),
      )
    : {};
  return (
    <aside className="trace-inspector" aria-label="Selected action">
      <header className="inspector-heading">
        <span>Selected action</span>
        <StatusBadge
          status={edge?.outcome ?? state?.semantic_status ?? "READY"}
        />
      </header>
      {state ? (
        <>
          <div className="selected-action">
            <span>S{state.sequence}</span>
            <h2>
              {edge ? edge.action.kind.replaceAll("_", " ") : "Initial state"}
            </h2>
            <p>
              {member
                ? `Sibling ${member.sibling_index + 1}`
                : edge
                  ? "Shared prefix"
                  : "Trajectory start"}
            </p>
          </div>
          <dl className="action-facts">
            <div>
              <dt>Result</dt>
              <dd>{edge?.outcome.replaceAll("_", " ") ?? "Ready"}</dd>
            </div>
            <div>
              <dt>State</dt>
              <dd>{state.semantic_status.replaceAll("_", " ")}</dd>
            </div>
            {member ? (
              <div>
                <dt>Verification</dt>
                <dd>
                  {(member.verification_status ?? "unknown").replaceAll(
                    "_",
                    " ",
                  )}
                  {member.failure_mode
                    ? ` · ${member.failure_mode.replaceAll("_", " ")}`
                    : ""}
                </dd>
              </div>
            ) : null}
            {member ? (
              <div>
                <dt>Training</dt>
                <dd>
                  {(member.eligibility_status ?? member.status).replaceAll(
                    "_",
                    " ",
                  )}
                </dd>
              </div>
            ) : null}
            {edge?.created_at ? (
              <div>
                <dt>Recorded</dt>
                <dd>{formatDate(edge.created_at)}</dd>
              </div>
            ) : null}
          </dl>
          <div className="step-navigation" aria-label="Step navigation">
            <button
              type="button"
              disabled={!previous}
              onClick={() => previous && select(previous)}
            >
              ← Previous
            </button>
            <button
              type="button"
              disabled={!next}
              onClick={() => next && select(next)}
            >
              Next →
            </button>
          </div>
          {edge && !member && !next && checkpoint ? (
            <p className="checkpoint-prompt">
              Shared prefix complete. Choose a sibling path below to continue
              past the checkpoint.
            </p>
          ) : null}
          {edge?.verification_run_id ? (
            <Link
              className="button secondary inspector-action"
              to={`/verification-runs/${edge.verification_run_id}`}
            >
              Inspect verification
            </Link>
          ) : null}
          {Object.keys(actionArguments).length ? (
            <details className="trace-provenance">
              <summary>Action input</summary>
              <pre>{JSON.stringify(actionArguments, null, 2)}</pre>
            </details>
          ) : null}
          <details className="trace-provenance">
            <summary>Action provenance</summary>
            <dl>
              <div>
                <dt>Transition</dt>
                <dd>
                  {edge ? (
                    <MachineId value={edge.id} copy={false} />
                  ) : (
                    "Initial state"
                  )}
                </dd>
              </div>
              <div>
                <dt>State</dt>
                <dd>
                  <MachineId value={state.id} copy={false} />
                </dd>
              </div>
              {edge?.proof_bundle_id ? (
                <div>
                  <dt>Proof</dt>
                  <dd>
                    <MachineId value={edge.proof_bundle_id} copy={false} />
                  </dd>
                </div>
              ) : null}
              {edge?.operation_id ? (
                <div>
                  <dt>Operation</dt>
                  <dd>
                    <MachineId value={edge.operation_id} copy={false} />
                  </dd>
                </div>
              ) : null}
              {edge?.action_artifact_id ? (
                <div>
                  <dt>Action artifact</dt>
                  <dd>
                    <MachineId value={edge.action_artifact_id} copy={false} />
                  </dd>
                </div>
              ) : null}
              {edge?.runtime_cursor_id ? (
                <div>
                  <dt>Runtime cursor</dt>
                  <dd>
                    <MachineId value={edge.runtime_cursor_id} copy={false} /> v
                    {edge.cursor_version}
                  </dd>
                </div>
              ) : null}
              {state.payload.observation?.digest ? (
                <div>
                  <dt>Observation</dt>
                  <dd>
                    <MachineId
                      value={state.payload.observation.digest}
                      copy={false}
                    />
                  </dd>
                </div>
              ) : null}
              {state.payload.logical_state?.digest ? (
                <div>
                  <dt>Logical state</dt>
                  <dd>
                    <MachineId
                      value={state.payload.logical_state.digest}
                      copy={false}
                    />
                  </dd>
                </div>
              ) : null}
            </dl>
          </details>
        </>
      ) : (
        <p className="muted-block">Select an action in the graph.</p>
      )}
      <section className="checkpoint-context">
        <div className="inspector-heading">
          <span>{checkpoint ? "Branch checkpoint" : "Trajectory context"}</span>
          <span>K={staticBranchWidth(graph)}</span>
        </div>
        {checkpoint ? (
          <dl>
            <div>
              <dt>Snapshot</dt>
              <dd>
                <MachineId
                  value={String(snapshot?.snapshot_id ?? "not recorded")}
                  copy={false}
                />
              </dd>
            </div>
            <div>
              <dt>Fidelity</dt>
              <dd>
                {String(
                  snapshot?.obtained_fidelity ?? "logical_restore",
                ).replaceAll("_", " ")}
              </dd>
            </div>
          </dl>
        ) : (
          <p className="muted-block">
            Independent rollout with no shared checkpoint.
          </p>
        )}
      </section>
    </aside>
  );
}

function filterGraph(graph: RolloutGraph, filter: string): RolloutGraph {
  if (filter === "all") return graph;
  const includedMembers = graph.branch_members.filter((member) => {
    const exceptional =
      Boolean(member.failure_mode) ||
      member.status !== "SUCCEEDED" ||
      Number(member.retry_count ?? 0) > 0;
    return filter === "exceptions" ? exceptional : !exceptional;
  });
  const includedIds = new Set(
    includedMembers.map((member) => member.branch_member_id),
  );
  const incoming = new Map(graph.edges.map((edge) => [edge.target, edge]));
  return {
    ...graph,
    branch_members: includedMembers,
    edges: graph.edges.filter(
      (edge) =>
        !edge.branch_member_id || includedIds.has(edge.branch_member_id),
    ),
    nodes: graph.nodes.filter((node) => {
      const edge = incoming.get(node.id);
      return !edge?.branch_member_id || includedIds.has(edge.branch_member_id);
    }),
  };
}

export function causalEdgeSequence(
  graph: RolloutGraph,
  selectedMemberId: string | null,
): GraphEdge[] {
  return graph.edges.filter(
    (edge) =>
      !edge.branch_member_id ||
      (selectedMemberId && edge.branch_member_id === selectedMemberId),
  );
}

function staticBranchWidth(graph: RolloutGraph): number {
  return Number(
    graph.branch_groups[0]?.width ?? (graph.branch_members.length || 1),
  );
}

export function buildFlow(
  graph: RolloutGraph,
  selectedState: string | null,
  selectedMemberId: string | null,
  select: (stateId: string, edge: GraphEdge | undefined) => void,
): { nodes: Node[]; edges: Edge[] } {
  const incoming = new Map(graph.edges.map((edge) => [edge.target, edge]));
  const memberIndex = new Map(
    graph.branch_members.map((member, visibleIndex) => [
      member.branch_member_id,
      visibleIndex,
    ]),
  );
  const prefixStates = graph.nodes.filter(
    (state) => !incoming.get(state.id)?.branch_member_id,
  );
  const laneGap = 164;
  const laneStartX = 24;
  const centerX =
    laneStartX + (Math.max(1, graph.branch_members.length) - 1) * (laneGap / 2);
  const stepGap = 96;
  const checkpointY = Math.max(
    stepGap,
    (prefixStates.length - 1) * stepGap + stepGap,
  );
  const firstBranchEdge = graph.edges.find((edge) => edge.branch_member_id);
  const branchSourceId = firstBranchEdge?.source;
  const checkpointId = graph.decision_checkpoints[0]
    ? `checkpoint-${String(graph.decision_checkpoints[0].checkpoint_id)}`
    : null;
  const perMemberIndex = new Map<string, number>();
  const nodes: Node[] = graph.nodes.map((state) => {
    const edge = incoming.get(state.id);
    const memberId = edge?.branch_member_id;
    let x = centerX;
    let y =
      prefixStates.findIndex((item) => item.id === state.id) * stepGap + 16;
    if (memberId) {
      const count = perMemberIndex.get(memberId) ?? 0;
      perMemberIndex.set(memberId, count + 1);
      x = laneStartX + (memberIndex.get(memberId) ?? 0) * laneGap;
      y = checkpointY + 112 + count * stepGap;
    }
    const label = edge
      ? edge.action.kind.replaceAll("_", " ")
      : "Initial state";
    const onSelectedPath =
      !memberId || (selectedMemberId && memberId === selectedMemberId);
    const classes = [
      "flow-node",
      state.id === selectedState ? "selected" : "",
      onSelectedPath ? "on-path" : "",
      state.semantic_status === "TERMINATED" ? "terminal" : "",
    ]
      .filter(Boolean)
      .join(" ");
    return {
      id: state.id,
      position: { x, y },
      data: {
        label: (
          <button
            className="flow-node-button"
            type="button"
            onClick={() => select(state.id, edge)}
            aria-label={`${label}, state ${state.sequence}, ${state.semantic_status}`}
          >
            <span className="flow-node-title">{label}</span>
            <span className="flow-node-meta">
              <code>S{state.sequence}</code>
              <small>
                {memberId
                  ? `Sibling ${(memberIndex.get(memberId) ?? 0) + 1}`
                  : edge
                    ? "Shared"
                    : "Start"}
              </small>
            </span>
          </button>
        ),
      },
      className: classes,
      draggable: false,
      selectable: true,
      ariaLabel: `${label}, state ${state.sequence}, ${state.semantic_status}`,
    };
  });
  if (checkpointId) {
    nodes.push({
      id: checkpointId,
      position: { x: centerX + 8, y: checkpointY + 8 },
      data: {
        label: (
          <div className="checkpoint-node-label">
            <span aria-hidden="true">◆</span>
            <strong>Checkpoint</strong>
            <small>fan out K={staticBranchWidth(graph)}</small>
          </div>
        ),
      },
      className: "flow-checkpoint",
      draggable: false,
      selectable: false,
      ariaLabel: `Branch checkpoint, static width ${staticBranchWidth(graph)}`,
    });
  }
  const edges: Edge[] = graph.edges.map((edge) => {
    const onSelectedPath =
      !edge.branch_member_id || edge.branch_member_id === selectedMemberId;
    const exactSelection = edge.target === selectedState;
    return {
      id: edge.id,
      source:
        checkpointId && edge.branch_member_id && edge.source === branchSourceId
          ? checkpointId
          : edge.source,
      target: edge.target,
      type: "smoothstep",
      animated: exactSelection,
      markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
      className: [
        "flow-edge",
        onSelectedPath ? "on-path" : "",
        exactSelection ? "selected" : "",
      ]
        .filter(Boolean)
        .join(" "),
      ariaLabel: `${edge.action.kind.replaceAll("_", " ")} transition, ${edge.outcome}`,
    };
  });
  if (checkpointId && branchSourceId) {
    edges.push({
      id: `${checkpointId}-entry`,
      source: branchSourceId,
      target: checkpointId,
      type: "smoothstep",
      markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
      className: "flow-edge on-path checkpoint-edge",
      ariaLabel: `Checkpoint created after shared prefix`,
    });
  }
  return { nodes, edges };
}

export function TrajectoryOutline({
  graph,
  selectedState,
  select,
}: {
  graph: RolloutGraph;
  selectedState: string | null;
  select: (stateId: string, edge: GraphEdge | undefined) => void;
}) {
  const incoming = new Map(graph.edges.map((edge) => [edge.target, edge]));
  return (
    <div className="outline-table-wrap">
      <table className="data-table outline-table">
        <caption>Keyboard-navigable equivalent of the rollout graph</caption>
        <thead>
          <tr>
            <th>State</th>
            <th>Relationship</th>
            <th>Action</th>
            <th>Outcome</th>
            <th>Verification</th>
          </tr>
        </thead>
        <tbody>
          {graph.nodes.map((state) => {
            const edge = incoming.get(state.id);
            const member = graph.branch_members.find(
              (item) => item.branch_member_id === edge?.branch_member_id,
            );
            return (
              <tr
                key={state.id}
                className={selectedState === state.id ? "selected" : ""}
              >
                <td>
                  <button
                    className="table-button"
                    type="button"
                    onClick={() => select(state.id, edge)}
                  >
                    S{state.sequence}{" "}
                    <MachineId value={state.id} copy={false} />
                  </button>
                </td>
                <td>
                  {member
                    ? `Sibling ${member.sibling_index + 1}`
                    : "Shared prefix"}
                </td>
                <td>{edge?.action.kind.replaceAll("_", " ") ?? "reset"}</td>
                <td>
                  <StatusBadge
                    status={edge?.outcome ?? state.semantic_status}
                  />
                </td>
                <td>
                  {edge ? (
                    <MachineId value={edge.verification_run_id} copy={false} />
                  ) : (
                    "—"
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function EvidenceComparator({
  proof,
  comparison,
  selectedMember,
  visibleMemberIds,
}: {
  proof: ProofBundle | undefined;
  comparison: BranchComparison | null;
  selectedMember: string | null;
  visibleMemberIds: string[];
}) {
  if (!proof) {
    return (
      <div className="state-panel partial-state">
        <strong>Select an accepted transition</strong>
        <p>
          Its reference, source, candidate, and sibling render evidence will
          appear here.
        </p>
      </div>
    );
  }
  const siblingRenders =
    comparison?.members
      .filter(
        (member) =>
          member.proof_bundle &&
          visibleMemberIds.includes(String(member.branch_member_id)) &&
          !(
            member.branch_member_id === selectedMember &&
            member.proof_bundle.candidate_render.digest ===
              proof.candidate_render.digest
          ),
      )
      .map((member) => ({
        label: `Sibling ${Number(member.sibling_index) + 1} terminal`,
        artifact: member.proof_bundle!.candidate_render,
        selected: false,
        status: String(member.status),
      })) ?? [];
  const renders = [
    {
      label: "Reference",
      artifact: proof.task_reference,
      selected: false,
      status: "REFERENCE",
    },
    {
      label: "Action source",
      artifact: proof.source_render,
      selected: false,
      status: "ACCEPTED",
    },
    {
      label: "Selected action result",
      artifact: proof.candidate_render,
      selected: true,
      status: "SELECTED",
    },
    ...siblingRenders,
  ];
  return (
    <Section
      title="Synchronized visual evidence"
      aside={
        <span>
          Renderer <code>cad-fixture-renderer@1</code>
        </span>
      }
      className="evidence-section"
    >
      <div className="render-strip">
        {renders.map((render, index) => (
          <EvidenceRender
            key={`${render.label}-${render.artifact.artifact_id}-${index}`}
            artifact={render.artifact}
            label={render.label}
            selected={render.selected}
            status={render.status}
          />
        ))}
      </div>
    </Section>
  );
}

function VerificationSummary({
  loading,
  error,
  detail,
  selectedStep,
  selectStep,
}: {
  loading: boolean;
  error: Error | null;
  detail: VerificationDetail | null;
  selectedStep: string | null;
  selectStep: (step: VerificationStep) => void;
}) {
  return (
    <Section
      title="Verification components"
      aside={
        detail ? (
          <Link
            to={`/verification-runs/${String(detail.verification_run.verification_run_id)}`}
          >
            Open DAG inspector
          </Link>
        ) : null
      }
    >
      <AsyncState loading={loading} error={error}>
        {detail ? (
          <>
            <div
              className="step-list"
              role="list"
              aria-label="Verification steps"
            >
              {detail.steps.map((step) => (
                <button
                  key={step.step_run_id}
                  type="button"
                  className={
                    selectedStep === step.step_id ? "step selected" : "step"
                  }
                  onClick={() => selectStep(step)}
                >
                  <span>
                    <strong>{step.step_id.replaceAll("-", " ")}</strong>
                    <small>
                      {step.step_type.replaceAll("_", " ")} ·{" "}
                      {step.attempt_count} attempt
                      {step.attempt_count === 1 ? "" : "s"} ·{" "}
                      {step.cache_status.toLowerCase()}
                    </small>
                  </span>
                  <StatusBadge status={step.status} />
                </button>
              ))}
            </div>
            {detail.judge ? (
              <Notice
                tone={
                  detail.judge.result.abstained ||
                  detail.judge.result.disagreement
                    ? "warning"
                    : "neutral"
                }
                title="Model assessment"
              >
                <p>{detail.judge.result.explanation}</p>
                <span>
                  Confidence {Math.round(detail.judge.result.confidence * 100)}%
                  · {detail.judge.result.provider_model_identity}
                </span>
              </Notice>
            ) : null}
          </>
        ) : null}
      </AsyncState>
    </Section>
  );
}

function RewardSummary({
  verificationId,
  proof,
  comparison,
  selectedMember,
}: {
  verificationId: string | null;
  proof: ProofBundle | undefined;
  comparison: BranchComparison | null;
  selectedMember: BranchComparison["members"][number] | null;
}) {
  const assessment = comparison?.model_assessment;
  const presentation = comparison?.member_bindings
    .map((binding) => `${binding.label} = Sibling ${binding.sibling_index + 1}`)
    .join(" → ");
  return (
    <Section title="Proof and reward lineage">
      {proof ? (
        <>
          <KeyValue
            items={[
              {
                label: "Proof bundle",
                value: <MachineId value={proof.proof_bundle_id} />,
              },
              {
                label: "Proof digest",
                value: <MachineId value={proof.digest} />,
              },
              {
                label: "Verification",
                value: verificationId ? (
                  <MachineId value={verificationId} />
                ) : (
                  "—"
                ),
              },
              {
                label: "Reward pipeline",
                value: <code>cad-local-rewards@1</code>,
              },
              {
                label: "Source render",
                value: <MachineId value={proof.source_render.digest} />,
              },
              {
                label: "Candidate render",
                value: <MachineId value={proof.candidate_render.digest} />,
              },
            ]}
          />
          {selectedMember ? (
            <>
              <h3>
                Sibling {Number(selectedMember.sibling_index) + 1} rewards
              </h3>
              <div className="table-wrap">
                <table className="data-table compact-table">
                  <thead>
                    <tr>
                      <th>Signal</th>
                      <th>Value</th>
                      <th>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedMember.reward_signals.map((reward) => (
                      <tr key={String(reward.reward_signal_id)}>
                        <td>{String(reward.name).replaceAll("_", " ")}</td>
                        <td>{Number(reward.value).toFixed(4)}</td>
                        <td>
                          <MachineId
                            value={String(reward.reward_signal_id)}
                            copy={false}
                          />
                          <small>
                            {Number(reward.metric_observation_ids?.length ?? 0)}{" "}
                            metric facts
                          </small>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}
          {assessment ? (
            <>
              <h3>Blinded sibling assessment</h3>
              <Notice
                tone={
                  assessment.abstained || assessment.disagreement
                    ? "warning"
                    : "neutral"
                }
                title="Group judge result"
              >
                <p>{String(assessment.explanation)}</p>
                <span>
                  <StatusBadge status={String(assessment.outcome)} /> ·
                  confidence{" "}
                  {Math.round(Number(assessment.confidence ?? 0) * 100)}% ·{" "}
                  {assessment.tie ? "tie recorded" : "ranked"} · order{" "}
                  <code>{presentation}</code>
                </span>
              </Notice>
              <KeyValue
                items={[
                  {
                    label: "Judge result",
                    value: (
                      <MachineId
                        value={String(assessment.judge_result_id)}
                        copy={false}
                      />
                    ),
                  },
                  {
                    label: "Specification",
                    value: <code>{String(assessment.judge_spec_id)}</code>,
                  },
                  {
                    label: "Proof digest",
                    value: (
                      <MachineId
                        value={String(assessment.proof_bundle_digest)}
                        copy={false}
                      />
                    ),
                  },
                ]}
              />
            </>
          ) : comparison ? (
            <p className="muted">
              This branch group has no persisted group-judge assessment.
            </p>
          ) : null}
        </>
      ) : (
        <p className="muted">
          Select a transition to inspect its immutable proof and reward sources.
        </p>
      )}
    </Section>
  );
}

function updateParam(
  params: URLSearchParams,
  setParams: ReturnType<typeof useSearchParams>[1],
  key: string,
  value: string,
) {
  const next = new URLSearchParams(params);
  next.set(key, value);
  setParams(next);
}
