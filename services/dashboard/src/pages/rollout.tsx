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
  EvidenceRender,
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

interface TransitionExplanation {
  transition_id: string;
  lane: string;
  local_step: number;
  task: {
    title: string;
    revision: string;
    families: string[];
    known_checks: string[];
    study_question: string;
  };
  checkpoint: {
    policy_version_id: string;
    source_state_id: string;
    source_sequence: number;
    source_status: string;
  };
  action: {
    kind: string;
    label: string;
    parameters: Record<string, unknown>;
  };
  effect: {
    destination_state_id: string;
    destination_sequence: number;
    destination_status: string;
    transition_outcome: string;
    terminal_quality: number | null;
  };
  verifier: {
    status: string;
    steps: Array<{
      step_id: string;
      status: string;
      attempt_count: number;
      cache_status: string;
    }>;
    deterministic_facts: Array<{
      descriptor: string;
      value: unknown;
      unit: string;
    }>;
    model_assessment: {
      outcome: string;
      explanation: string | null;
    } | null;
  };
  learning: {
    eligibility: string;
    eligibility_reason: string | null;
    reward_signals: Array<{
      reward_signal_id: string;
      name: string;
      value: number;
      metric_observation_ids: string[];
    }>;
    optimizer_status: string;
    contributed_to_policy_version: string | null;
    iteration_input_digest: string | null;
  };
}

export function RolloutTreePage() {
  const { treeId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const graph = useApi<RolloutGraph>(`/v1/rollout-trees/${treeId}/graph`);
  const selectedVerification = params.get("verification");
  const verification = useApi<VerificationDetail>(
    selectedVerification
      ? `/v1/verification-runs/${selectedVerification}`
      : null,
  );
  const branchGroupId =
    params.get("branch") ?? graph.data?.branch_groups[0]?.branch_group_id;
  const comparison = useApi<BranchComparison>(
    branchGroupId ? `/v1/branch-groups/${branchGroupId}/comparison` : null,
  );
  const [rejudging, setRejudging] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const requestedView = params.get("view");
  const view =
    requestedView === "graph" || requestedView === "outline"
      ? requestedView
      : window.matchMedia("(max-width: 760px)").matches
        ? "outline"
        : "graph";
  const requestedFilter = params.get("filter");
  const outcomeFilter =
    requestedFilter === "accepted" || requestedFilter === "exceptions"
      ? requestedFilter
      : "all";
  const filteredGraph = useMemo(
    () =>
      graph.data
        ? filterGraph(graph.data, outcomeFilter)
        : (graph.data ?? null),
    [graph.data, outcomeFilter],
  );

  useEffect(() => {
    if (!filteredGraph) return;
    const next = new URLSearchParams(params);
    const selectedState = params.get("state");
    const selectedEdge = filteredGraph.edges.find(
      (edge) => edge.target === selectedState,
    );
    const selectedMember = selectedEdge?.branch_member_id ?? null;
    const memberVisible =
      !selectedMember ||
      filteredGraph.branch_members.some(
        (member) => member.branch_member_id === selectedMember,
      );
    const activeEdge =
      selectedEdge && memberVisible
        ? selectedEdge
        : ([...filteredGraph.edges]
            .reverse()
            .find((edge) => edge.branch_member_id) ??
          filteredGraph.edges.at(-1));
    if (activeEdge) {
      next.set("state", activeEdge.target);
      next.set("verification", activeEdge.verification_run_id);
      if (activeEdge.branch_member_id)
        next.set("member", activeEdge.branch_member_id);
      else next.delete("member");
    }
    if (filteredGraph.branch_groups[0]) {
      next.set(
        "branch",
        String(filteredGraph.branch_groups[0].branch_group_id),
      );
    }
    if (requestedView !== "graph" && requestedView !== "outline")
      next.set("view", view);
    if (
      requestedFilter !== null &&
      requestedFilter !== "all" &&
      requestedFilter !== "accepted" &&
      requestedFilter !== "exceptions"
    )
      next.set("filter", "all");
    if (next.toString() !== params.toString())
      setParams(next, { replace: true });
  }, [filteredGraph, params, requestedFilter, requestedView, setParams, view]);

  const selectedState = params.get("state");
  const selectedStep = params.get("step");
  const selectedEdge = graph.data?.edges.find(
    (edge) => edge.target === selectedState,
  );
  const explanation = useApi<TransitionExplanation>(
    selectedEdge ? `/v1/transitions/${selectedEdge.id}/explanation` : null,
  );
  const flow = useMemo(
    () =>
      filteredGraph
        ? buildFlow(filteredGraph, selectedState, (stateId, edge) => {
            const next = new URLSearchParams(params);
            next.set("state", stateId);
            if (edge) next.set("verification", edge.verification_run_id);
            if (edge?.branch_member_id)
              next.set("member", edge.branch_member_id);
            setParams(next);
          })
        : { nodes: [], edges: [] },
    [filteredGraph, params, selectedState, setParams],
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
            <Link to="/runs">Runs</Link> / rollout tree
          </>
        }
        title="Shared-prefix CAD trajectory"
        description={<MachineId value={treeId} />}
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
            <div className="trajectory-controls">
              <label>
                Outcome filter
                <select
                  value={outcomeFilter}
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
                · {filteredGraph.edges.length} transitions ·{" "}
                {filteredGraph.branch_members.length} of{" "}
                {graph.data.branch_members.length} siblings
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
              <aside className="branch-outline">
                <div className="inspector-heading">
                  <span>Branch group</span>
                  {graph.data.branch_groups[0] ? (
                    <StatusBadge
                      status={String(graph.data.branch_groups[0].status)}
                    />
                  ) : (
                    <StatusBadge status="INDEPENDENT" />
                  )}
                </div>
                {graph.data.decision_checkpoints[0] ? (
                  <KeyValue
                    items={[
                      {
                        label: "Checkpoint",
                        value: (
                          <MachineId
                            value={String(
                              graph.data.decision_checkpoints[0].checkpoint_id,
                            )}
                            copy={false}
                          />
                        ),
                      },
                      {
                        label: "Snapshot",
                        value: (
                          <MachineId
                            value={String(
                              graph.data.environment_snapshots[0]?.snapshot_id,
                            )}
                            copy={false}
                          />
                        ),
                      },
                      {
                        label: "Fidelity",
                        value: <code>logical_restore</code>,
                      },
                      {
                        label: "Width",
                        value: graph.data.branch_members.length,
                      },
                    ]}
                  />
                ) : (
                  <p className="muted">
                    Independent rollout: no shared decision checkpoint.
                  </p>
                )}
                <div className="member-list">
                  {filteredGraph.branch_members.map((member) => (
                    <button
                      type="button"
                      key={member.branch_member_id}
                      className={
                        params.get("member") === member.branch_member_id
                          ? "member selected"
                          : "member"
                      }
                      onClick={() => {
                        const edge = [...graph.data!.edges]
                          .reverse()
                          .find(
                            (item) =>
                              item.branch_member_id === member.branch_member_id,
                          );
                        if (!edge) return;
                        const next = new URLSearchParams(params);
                        next.set("member", member.branch_member_id);
                        next.set("state", edge.target);
                        next.set("verification", edge.verification_run_id);
                        setParams(next);
                      }}
                    >
                      <span>Sibling {member.sibling_index + 1}</span>
                      <StatusBadge status={member.status} />
                      {member.failure_mode ? (
                        <small>
                          {member.failure_mode.replaceAll("_", " ")}
                        </small>
                      ) : null}
                    </button>
                  ))}
                </div>
              </aside>
            </div>
            <StepExplanation
              loading={explanation.loading}
              error={explanation.error}
              explanation={explanation.data}
              member={graph.data.branch_members.find(
                (item) =>
                  item.branch_member_id === selectedEdge?.branch_member_id,
              )}
            />
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

function StepExplanation({
  loading,
  error,
  explanation,
  member,
}: {
  loading: boolean;
  error: Error | null;
  explanation: TransitionExplanation | null;
  member:
    | {
        sibling_index: number;
        status: string;
      }
    | undefined;
}) {
  return (
    <Section
      id="step-explanation"
      title={
        explanation
          ? `${member ? `Sibling ${member.sibling_index + 1}` : "Shared prefix"} · step ${explanation.local_step}`
          : "Selected step"
      }
      aside={
        explanation ? (
          <StatusBadge status={explanation.verifier.status} />
        ) : undefined
      }
      className="step-explanation"
    >
      <AsyncState loading={loading} error={error}>
        {explanation ? (
          <>
            <a className="mobile-back-link" href="#trajectory-outline">
              ← Back to trajectory steps
            </a>
            <div className="explanation-grid">
              <article>
                <span>Task</span>
                <h3>{explanation.task.title}</h3>
                <p>
                  Families: {explanation.task.families.join(", ")}. Known
                  checks: {explanation.task.known_checks.join(", ")}.
                </p>
              </article>
              <article>
                <span>Checkpoint knowledge</span>
                <h3>
                  State {explanation.checkpoint.source_sequence} ·{" "}
                  {explanation.checkpoint.source_status.toLowerCase()}
                </h3>
                <p>
                  Policy{" "}
                  <MachineId
                    value={explanation.checkpoint.policy_version_id}
                    copy={false}
                  />{" "}
                  produced this continuation.
                </p>
              </article>
              <article>
                <span>Action</span>
                <h3>{explanation.action.label}</h3>
                <p>
                  {Object.keys(explanation.action.parameters).length
                    ? Object.entries(explanation.action.parameters)
                        .map(([key, value]) => `${key} ${String(value)}`)
                        .join(" · ")
                    : "No additional action parameters."}
                </p>
              </article>
              <article>
                <span>Effect</span>
                <h3>
                  State {explanation.checkpoint.source_sequence} →{" "}
                  {explanation.effect.destination_sequence}
                </h3>
                <p>
                  {explanation.effect.transition_outcome
                    .toLowerCase()
                    .replaceAll("_", " ")}
                  {explanation.effect.terminal_quality !== null
                    ? ` · terminal quality ${Number(explanation.effect.terminal_quality).toFixed(3)}`
                    : ""}
                </p>
              </article>
              <article>
                <span>Verifier observation</span>
                <h3>
                  {explanation.verifier.steps.length} components ·{" "}
                  {explanation.verifier.steps.reduce(
                    (total, step) => total + step.attempt_count,
                    0,
                  )}{" "}
                  attempts
                </h3>
                <ul>
                  {explanation.verifier.deterministic_facts
                    .slice(0, 4)
                    .map((fact) => (
                      <li key={fact.descriptor}>
                        {fact.descriptor
                          .replace("deterministic.", "")
                          .replaceAll("_", " ")}
                        : <strong>{String(fact.value)}</strong>{" "}
                        <small>{fact.unit}</small>
                      </li>
                    ))}
                </ul>
              </article>
              <article>
                <span>Credit and update</span>
                <h3>
                  {explanation.learning.eligibility
                    .toLowerCase()
                    .replaceAll("_", " ")}
                </h3>
                <p>
                  {explanation.learning.reward_signals.length} named reward
                  signals. Optimizer{" "}
                  {explanation.learning.optimizer_status.toLowerCase()}.
                </p>
                {explanation.learning.contributed_to_policy_version ? (
                  <p>
                    Contributed to{" "}
                    <MachineId
                      value={explanation.learning.contributed_to_policy_version}
                      copy={false}
                    />
                  </p>
                ) : (
                  <p>No optimizer contribution is recorded for this step.</p>
                )}
              </article>
            </div>
            <details className="manifest-details">
              <summary>Technical details</summary>
              <pre className="json-block">
                {JSON.stringify(explanation, null, 2)}
              </pre>
            </details>
          </>
        ) : null}
      </AsyncState>
    </Section>
  );
}

function filterGraph(graph: RolloutGraph, filter: string): RolloutGraph {
  if (filter === "all") return graph;
  const includedMembers = graph.branch_members.filter((member) => {
    const exceptional =
      Boolean(member.failure_mode) || member.status !== "SUCCEEDED";
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

function buildFlow(
  graph: RolloutGraph,
  selectedState: string | null,
  select: (stateId: string, edge: GraphEdge | undefined) => void,
): { nodes: Node[]; edges: Edge[] } {
  const incoming = new Map(graph.edges.map((edge) => [edge.target, edge]));
  const memberIndex = new Map(
    graph.branch_members.map((member) => [
      member.branch_member_id,
      member.sibling_index,
    ]),
  );
  const prefixStates = graph.nodes.filter(
    (state) => !incoming.get(state.id)?.branch_member_id,
  );
  const branchPointX = Math.max(160, (prefixStates.length - 1) * 160);
  const perMemberIndex = new Map<string, number>();
  const nodes: Node[] = graph.nodes.map((state) => {
    const edge = incoming.get(state.id);
    const memberId = edge?.branch_member_id;
    let x = prefixStates.findIndex((item) => item.id === state.id) * 160;
    let y = 70;
    if (memberId) {
      const count = perMemberIndex.get(memberId) ?? 0;
      perMemberIndex.set(memberId, count + 1);
      x = branchPointX + (count + 1) * 160;
      y = 190 + (memberIndex.get(memberId) ?? 0) * 105;
    }
    const label = edge ? edge.action.kind.replaceAll("_", " ") : "reset";
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
            <span>{label}</span>
            <code>
              {edge
                ? edge.branch_member_id
                  ? `B${(memberIndex.get(edge.branch_member_id) ?? 0) + 1}.${edge.local_step}`
                  : `P${edge.local_step}`
                : "START"}
            </code>
          </button>
        ),
      },
      className:
        state.id === selectedState ? "flow-node selected" : "flow-node",
      draggable: false,
      selectable: true,
      ariaLabel: `${label}, state ${state.sequence}, ${state.semantic_status}`,
    };
  });
  const edges: Edge[] = graph.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    type: "smoothstep",
    animated: edge.target === selectedState,
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
    className:
      edge.target === selectedState ? "flow-edge selected" : "flow-edge",
    ariaLabel: `${edge.action.kind.replaceAll("_", " ")} transition, ${edge.outcome}`,
  }));
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
    <div className="outline-table-wrap" id="trajectory-outline">
      <table className="data-table outline-table desktop-index">
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
                    {edge
                      ? `${member ? `Sibling ${member.sibling_index + 1}` : "Prefix"} · step ${edge.local_step}`
                      : "Initial state"}{" "}
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
      <ol className="mobile-index trajectory-cards">
        {graph.nodes.flatMap((state) => {
          const edge = incoming.get(state.id);
          if (!edge) return [];
          const member = graph.branch_members.find(
            (item) => item.branch_member_id === edge.branch_member_id,
          );
          return [
            <li
              key={state.id}
              className={selectedState === state.id ? "selected" : ""}
            >
              <button
                type="button"
                onClick={() => {
                  select(state.id, edge);
                  window.requestAnimationFrame(() =>
                    document
                      .getElementById("step-explanation")
                      ?.scrollIntoView({ behavior: "smooth", block: "start" }),
                  );
                }}
              >
                <span>
                  <strong>
                    {member
                      ? `Sibling ${member.sibling_index + 1}`
                      : "Shared prefix"}{" "}
                    · step {edge.local_step}
                  </strong>
                  <small>{edge.action_label}</small>
                </span>
                <StatusBadge status={edge.outcome} />
                <span className="trajectory-card-footer">
                  State {state.sequence} ·{" "}
                  {state.semantic_status.toLowerCase().replaceAll("_", " ")}
                </span>
              </button>
            </li>,
          ];
        })}
      </ol>
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
          visibleMemberIds.includes(String(member.branch_member_id)),
      )
      .map((member) => ({
        label: `Sibling ${Number(member.sibling_index) + 1}`,
        artifact: member.proof_bundle!.candidate_render,
        selected: member.branch_member_id === selectedMember,
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
      label: "Source",
      artifact: proof.source_render,
      selected: false,
      status: "ACCEPTED",
    },
    {
      label: "Candidate",
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
