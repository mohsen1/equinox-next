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
import { friendlyStatus, StatusBadge } from "../components";
import type {
  ResearchBranchSibling,
  ResearchBranchSnapshot,
  ResearchBranchStep,
} from "../types";

type BranchView = "graph" | "outline";

interface BranchNodeData extends Record<string, unknown> {
  siblingIndex?: number;
  actionId?: string;
  title: string;
  detail: string;
  action?: string | Record<string, string> | null;
  reward?: number;
  passed?: boolean;
  best?: boolean;
  multiStep?: boolean;
  prefix?: boolean;
}

interface BranchSelection {
  actionId: string;
  title: string;
  sibling: ResearchBranchSibling | null;
  step: ResearchBranchStep | null;
}

type BranchFlowNode = Node<BranchNodeData>;

const branchNodeTypes = {
  task: BranchTaskNode,
  checkpoint: BranchCheckpointNode,
  action: BranchActionNode,
  sibling: BranchSiblingNode,
};

export function ResearchBranchWorkspace({
  snapshot,
  selectedSibling,
  selectedActionId,
  view,
  selectSibling,
  selectAction,
}: {
  snapshot: ResearchBranchSnapshot;
  selectedSibling: ResearchBranchSibling | null;
  selectedActionId: string;
  view: BranchView;
  selectSibling: (index: number) => void;
  selectAction: (actionId: string, siblingIndex?: number) => void;
}) {
  const flow = buildBranchFlow(
    snapshot,
    selectedSibling?.index ?? -1,
    selectedActionId,
  );
  const navigation = branchNavigation(snapshot, selectedSibling);
  const selected =
    navigation.find((item) => item.actionId === selectedActionId) ??
    navigation.at(-1) ??
    null;
  const selectedPosition = selected
    ? navigation.findIndex((item) => item.actionId === selected.actionId)
    : -1;
  const continuationCount = snapshot.siblings.length;
  const continuationLabel = `${continuationCount} restored continuation${
    continuationCount === 1 ? "" : "s"
  }`;

  return (
    <div className="research-trajectory-workspace branch-workspace">
      <section
        className="research-trajectory-canvas branch-canvas"
        aria-label={`Shared prefix and ${continuationLabel}`}
      >
        {view === "graph" ? (
          <ReactFlow
            nodes={flow.nodes}
            edges={flow.edges}
            nodeTypes={branchNodeTypes}
            onNodeClick={(_, node) => {
              if (typeof node.data.siblingIndex === "number") {
                selectSibling(node.data.siblingIndex);
              }
              if (typeof node.data.actionId === "string") {
                selectAction(node.data.actionId, node.data.siblingIndex);
              }
            }}
            fitView
            fitViewOptions={{ padding: 0.14 }}
            minZoom={0.25}
            maxZoom={1.5}
            nodesDraggable={false}
            nodesConnectable={false}
            nodesFocusable
            edgesFocusable={false}
            autoPanOnNodeFocus
            aria-label={`One diagnostic prefix restored into ${continuationCount} multi-step continuation${
              continuationCount === 1 ? "" : "s"
            }.`}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="var(--rule)" gap={32} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
        ) : (
          <BranchOutline
            snapshot={snapshot}
            selectedIndex={selectedSibling?.index ?? -1}
            selectedActionId={selectedActionId}
            select={selectSibling}
            selectAction={selectAction}
          />
        )}
      </section>
      <BranchInspector
        snapshot={snapshot}
        selection={selected}
        previous={navigation[selectedPosition - 1]}
        next={navigation[selectedPosition + 1]}
        selectAction={selectAction}
      />
    </div>
  );
}

export function buildBranchFlow(
  snapshot: ResearchBranchSnapshot,
  selectedIndex: number,
  selectedActionId = "",
): { nodes: BranchFlowNode[]; edges: Edge[] } {
  if (!snapshot.shared_prefix) {
    return buildLegacyBranchFlow(snapshot, selectedIndex);
  }

  const nodes: BranchFlowNode[] = [];
  const edges: Edge[] = [];
  const prefix = snapshot.shared_prefix.steps;
  const rootId = `${snapshot.snapshot_id}-task`;
  const checkpointId = `${snapshot.snapshot_id}-checkpoint`;
  const laneWidth = 190;
  const centerX = Math.max(0, ((snapshot.siblings.length - 1) * laneWidth) / 2);
  nodes.push({
    id: rootId,
    type: "task",
    position: { x: centerX - 41, y: 18 },
    selectable: false,
    focusable: false,
    draggable: false,
    data: {
      title: friendlyStatus(snapshot.domain),
      detail: `Level ${snapshot.level} · ${snapshot.task?.complexity.file_count ?? "—"} files`,
      multiStep: true,
    },
  });

  let previousId = rootId;
  prefix.forEach((step, position) => {
    const actionId = prefixActionId(step);
    const nodeId = `${snapshot.snapshot_id}-${actionId}`;
    nodes.push({
      id: nodeId,
      type: "action",
      position: { x: centerX, y: 132 + position * 104 },
      selected: actionId === selectedActionId,
      data: {
        actionId,
        title: `Prefix ${position + 1}`,
        detail: stepOutcome(step),
        action: step.action,
        passed: step.accepted,
        prefix: true,
      },
    });
    edges.push(branchEdge(previousId, nodeId, actionId === selectedActionId));
    previousId = nodeId;
  });

  const checkpointY = 132 + prefix.length * 104;
  nodes.push({
    id: checkpointId,
    type: "checkpoint",
    position: { x: centerX + 4, y: checkpointY },
    selectable: false,
    focusable: false,
    draggable: false,
    data: {
      title: "Checkpoint",
      detail: snapshot.checkpoint?.fidelity
        ? friendlyStatus(snapshot.checkpoint.fidelity)
        : friendlyStatus(snapshot.exclusion_reason ?? "Unavailable"),
      passed: snapshot.checkpoint !== null,
    },
  });
  edges.push(branchEdge(previousId, checkpointId, false));

  snapshot.siblings.forEach((sibling, lanePosition) => {
    let lanePreviousId = checkpointId;
    const laneX = lanePosition * laneWidth;
    (sibling.steps ?? []).forEach((step, stepPosition) => {
      const actionId = siblingActionId(sibling, step);
      const nodeId = `${snapshot.snapshot_id}-${actionId}`;
      nodes.push({
        id: nodeId,
        type: "action",
        position: {
          x: laneX,
          y: checkpointY + 138 + stepPosition * 94,
        },
        selected: actionId === selectedActionId,
        data: {
          siblingIndex: sibling.index,
          actionId,
          title:
            stepPosition === 0
              ? `Sibling ${sibling.index + 1} · 1`
              : `Step ${stepPosition + 1}`,
          detail: stepOutcome(step),
          action: step.action,
          reward: step.terminal ? siblingReturn(sibling) : undefined,
          passed: step.accepted,
          best:
            stepPosition === 0 && sibling.index === snapshot.best_sibling_index,
        },
      });
      edges.push(
        branchEdge(
          lanePreviousId,
          nodeId,
          sibling.index === selectedIndex,
          sibling.passed,
        ),
      );
      lanePreviousId = nodeId;
    });
  });
  return { nodes, edges };
}

function buildLegacyBranchFlow(
  snapshot: ResearchBranchSnapshot,
  selectedIndex: number,
): { nodes: BranchFlowNode[]; edges: Edge[] } {
  const nodes: BranchFlowNode[] = [
    {
      id: snapshot.snapshot_id,
      type: "task",
      position: { x: 325, y: 34 },
      selectable: false,
      focusable: false,
      draggable: false,
      data: {
        title: friendlyStatus(snapshot.domain),
        detail: `Update ${snapshot.update} · Level ${snapshot.level}`,
      },
    },
    ...snapshot.siblings.map((sibling, position) => ({
      id: `${snapshot.snapshot_id}-sibling-${sibling.index}`,
      type: "sibling",
      position: { x: position * 224, y: 252 },
      selected: sibling.index === selectedIndex,
      data: {
        siblingIndex: sibling.index,
        actionId: legacyActionId(sibling),
        title: `Sibling ${sibling.index + 1}`,
        detail: sibling.passed ? "Verifier passed" : "Verifier failed",
        action: sibling.action,
        reward: siblingReturn(sibling),
        passed: sibling.passed,
        best: sibling.index === snapshot.best_sibling_index,
      },
    })),
  ];
  const edges = snapshot.siblings.map((sibling) => {
    const selected = sibling.index === selectedIndex;
    const best = sibling.index === snapshot.best_sibling_index;
    return {
      ...branchEdge(
        snapshot.snapshot_id,
        `${snapshot.snapshot_id}-sibling-${sibling.index}`,
        selected,
        sibling.passed,
      ),
      className: [
        sibling.passed ? "branch-edge-passed" : "branch-edge-failed",
        best ? "branch-edge-best" : "",
        selected ? "selected" : "",
      ]
        .filter(Boolean)
        .join(" "),
    };
  });
  return { nodes, edges };
}

function branchEdge(
  source: string,
  target: string,
  selected: boolean,
  passed?: boolean,
): Edge {
  return {
    id: `${source}-${target}`,
    source,
    target,
    type: "smoothstep",
    markerEnd: {
      type: MarkerType.ArrowClosed,
      width: 13,
      height: 13,
    },
    style: {
      stroke: selected
        ? "var(--accent)"
        : passed
          ? "var(--success)"
          : "var(--rule-strong)",
      strokeWidth: selected ? 2 : 1.25,
    },
  };
}

function BranchTaskNode({ data }: NodeProps<BranchFlowNode>) {
  return (
    <div className="research-branch-task-node">
      <span>Task</span>
      <strong>{data.title}</strong>
      <small>{data.detail}</small>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function BranchCheckpointNode({ data }: NodeProps<BranchFlowNode>) {
  return (
    <div
      className={`research-branch-checkpoint-node ${
        data.passed ? "" : "unavailable"
      }`}
    >
      <Handle type="target" position={Position.Top} />
      <strong>{data.title}</strong>
      <small>{data.detail}</small>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function BranchActionNode({ data, selected }: NodeProps<BranchFlowNode>) {
  return (
    <div
      className={`research-branch-action-node ${selected ? "selected" : ""} ${
        data.passed ? "accepted" : "rejected"
      }`}
    >
      <Handle type="target" position={Position.Top} />
      <div>
        <span>{data.title}</span>
        {data.best ? <em>Best</em> : null}
      </div>
      <code title={formatAction(data.action)}>
        {compactAction(data.action)}
      </code>
      <small>{data.detail}</small>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function BranchSiblingNode({ data, selected }: NodeProps<BranchFlowNode>) {
  return (
    <div
      className={`research-branch-sibling-node ${
        selected ? "selected" : ""
      } ${data.passed ? "passed" : "failed"}`}
    >
      <Handle type="target" position={Position.Top} />
      <div>
        <span>{data.title}</span>
        {data.best ? <em>Best reward</em> : null}
      </div>
      <code title={formatAction(data.action)}>
        {compactAction(data.action)}
      </code>
      <div>
        <strong>{formatReward(data.reward ?? 0)}</strong>
        <small>{data.detail}</small>
      </div>
    </div>
  );
}

export function BranchOutline({
  snapshot,
  selectedIndex,
  selectedActionId = "",
  select,
  selectAction = () => undefined,
}: {
  snapshot: ResearchBranchSnapshot;
  selectedIndex: number;
  selectedActionId?: string;
  select: (index: number) => void;
  selectAction?: (actionId: string, siblingIndex?: number) => void;
}) {
  if (snapshot.shared_prefix) {
    return (
      <MultiStepBranchOutline
        snapshot={snapshot}
        selectedActionId={selectedActionId}
        selectAction={selectAction}
      />
    );
  }
  return (
    <div className="research-trajectory-outline branch-outline">
      <table>
        <caption>K={snapshot.siblings.length} sibling actions</caption>
        <thead>
          <tr>
            <th>Sibling</th>
            <th>Action</th>
            <th>Reward</th>
            <th>Format</th>
            <th>Verifier</th>
            <th>Policy signal</th>
          </tr>
        </thead>
        <tbody>
          {snapshot.siblings.map((sibling) => (
            <tr
              key={sibling.index}
              className={
                sibling.index === selectedIndex ? "selected" : undefined
              }
            >
              <td>
                <button type="button" onClick={() => select(sibling.index)}>
                  Sibling {sibling.index + 1}
                  {sibling.index === snapshot.best_sibling_index
                    ? " · Best"
                    : ""}
                </button>
              </td>
              <td>
                <code>{sibling.action ?? "No valid action"}</code>
              </td>
              <td>{formatReward(siblingReturn(sibling))}</td>
              <td>{sibling.format_valid ? "Valid" : "Invalid"}</td>
              <td>{sibling.passed ? "Passed" : "Failed"}</td>
              <td>
                {sibling.policy_signal ? formatSigned(sibling.advantage) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MultiStepBranchOutline({
  snapshot,
  selectedActionId,
  selectAction,
}: {
  snapshot: ResearchBranchSnapshot;
  selectedActionId: string;
  selectAction: (actionId: string, siblingIndex?: number) => void;
}) {
  return (
    <div className="research-trajectory-outline branch-outline">
      <table>
        <caption>
          Shared prefix and K={snapshot.siblings.length} continuation steps
        </caption>
        <thead>
          <tr>
            <th>Lane</th>
            <th>Step</th>
            <th>Action</th>
            <th>Observation</th>
            <th>Verifier</th>
            <th>Reward</th>
          </tr>
        </thead>
        <tbody>
          {snapshot.shared_prefix?.steps.map((step, index) => {
            const actionId = prefixActionId(step);
            return (
              <BranchStepRow
                key={actionId}
                lane="Shared"
                actionId={actionId}
                step={step}
                displayIndex={index + 1}
                selected={actionId === selectedActionId}
                select={() => selectAction(actionId)}
              />
            );
          })}
          {snapshot.siblings.flatMap((sibling) =>
            (sibling.steps ?? []).map((step, index) => {
              const actionId = siblingActionId(sibling, step);
              return (
                <BranchStepRow
                  key={actionId}
                  lane={`Sibling ${sibling.index + 1}`}
                  actionId={actionId}
                  step={step}
                  displayIndex={index + 1}
                  reward={step.terminal ? siblingReturn(sibling) : undefined}
                  selected={actionId === selectedActionId}
                  select={() => selectAction(actionId, sibling.index)}
                />
              );
            }),
          )}
        </tbody>
      </table>
    </div>
  );
}

function BranchStepRow({
  lane,
  step,
  displayIndex,
  reward,
  selected,
  select,
}: {
  lane: string;
  actionId: string;
  step: ResearchBranchStep;
  displayIndex: number;
  reward?: number;
  selected: boolean;
  select: () => void;
}) {
  return (
    <tr className={selected ? "selected" : undefined}>
      <td>
        <button type="button" onClick={select}>
          {lane}
        </button>
      </td>
      <td>{displayIndex}</td>
      <td>
        <code>{compactAction(step.action)}</code>
      </td>
      <td>{compactObservation(step.observation)}</td>
      <td>{step.verifier_passed ? "Passing" : "Failing"}</td>
      <td>{reward === undefined ? "—" : formatReward(reward)}</td>
    </tr>
  );
}

function BranchInspector({
  snapshot,
  selection,
  previous,
  next,
  selectAction,
}: {
  snapshot: ResearchBranchSnapshot;
  selection: BranchSelection | null;
  previous?: BranchSelection;
  next?: BranchSelection;
  selectAction: (actionId: string, siblingIndex?: number) => void;
}) {
  if (!selection) {
    return (
      <aside
        className="research-trajectory-inspector branch-inspector"
        aria-label="Selected branch action"
      >
        <header>
          <span>
            Update {snapshot.update} · Level {snapshot.level}
          </span>
          <StatusBadge status={snapshot.excluded ? "EXCLUDED" : "RUNNING"} />
        </header>
        <div className="trajectory-selected-step">
          <h2>{snapshot.exclusion_reason ?? "Waiting for branch actions"}</h2>
        </div>
      </aside>
    );
  }

  const { sibling, step } = selection;
  const reward = sibling ? siblingReturn(sibling) : (step?.reward ?? 0);
  const status = step?.accepted
    ? step.terminal && !step.verifier_passed
      ? "FAILED"
      : "VERIFIED"
    : sibling?.passed
      ? "SUCCEEDED"
      : "FAILED";
  return (
    <aside
      className="research-trajectory-inspector branch-inspector"
      aria-label="Selected branch action"
    >
      <header>
        <span>
          Update {snapshot.update} · Level {snapshot.level}
        </span>
        <StatusBadge status={status} />
      </header>
      <div className="trajectory-selected-step">
        <h2>{selection.title}</h2>
        <strong>
          {step?.terminal || !step ? formatReward(reward) : step.index + 1}
        </strong>
        <span>{step?.terminal || !step ? "return" : "step"}</span>
      </div>
      {step ? (
        <>
          <dl className="trajectory-step-facts">
            <Fact label="Tool" value={friendlyStatus(step.tool ?? "Invalid")} />
            <Fact
              label="Action"
              value={step.accepted ? "Accepted" : "Rejected"}
            />
            <Fact
              label="Verifier"
              value={
                step.verifier_passed
                  ? "Passing"
                  : `${step.fixed_faults} / ${step.total_faults} fixed`
              }
            />
            <Fact
              label="Terminal"
              value={friendlyStatus(step.terminal_reason ?? "No")}
            />
            <Fact
              label="Advantage"
              value={sibling ? formatSigned(sibling.advantage) : "Masked"}
            />
            <Fact
              label="Policy signal"
              value={
                sibling
                  ? (step.policy_signal ?? sibling.policy_signal)
                    ? "Eligible"
                    : "None"
                  : "Prefix masked"
              }
            />
            <Fact
              label="Group"
              value={
                snapshot.excluded
                  ? friendlyStatus(snapshot.exclusion_reason ?? "Excluded")
                  : snapshot.learning_signal
                    ? "Eligible"
                    : "No relative signal"
              }
            />
            <Fact
              label="Batch weight"
              value={
                (step.effective_batch_weight ??
                  sibling?.effective_batch_weight) === undefined
                  ? "Masked"
                  : formatSigned(
                      step.effective_batch_weight ??
                        sibling?.effective_batch_weight ??
                        0,
                    )
              }
            />
          </dl>
          <div className="branch-evidence">
            <section>
              <span>Action</span>
              <pre>{formatAction(step.action)}</pre>
            </section>
            <section>
              <span>Observation</span>
              <pre>{step.observation}</pre>
            </section>
            <details>
              <summary>State</summary>
              <code>{step.state_digest_after}</code>
            </details>
            {sibling?.reward_components ? (
              <details>
                <summary>
                  Reward ·{" "}
                  {formatReward(sibling.reward_components.terminal_aggregate)}
                </summary>
                <dl className="branch-evidence-facts">
                  <Fact
                    label="Hidden correctness"
                    value={
                      sibling.reward_components.hidden_correctness
                        ? "1.000"
                        : "0.000"
                    }
                  />
                  <Fact
                    label="Public progress"
                    value={formatReward(
                      sibling.reward_components.public_verifier_progress,
                    )}
                  />
                  <Fact
                    label="Action cost"
                    value={formatSigned(
                      sibling.reward_components.accepted_action_cost,
                    )}
                  />
                  <Fact
                    label="Malformed"
                    value={String(
                      sibling.reward_components.malformed_action_count,
                    )}
                  />
                  <Fact
                    label="Submissions"
                    value={String(
                      sibling.reward_components.verifier_submission_count,
                    )}
                  />
                  <Fact
                    label="Completion tokens"
                    value={String(sibling.completion_tokens ?? 0)}
                  />
                </dl>
              </details>
            ) : null}
            {snapshot.optimizer_update ? (
              <details>
                <summary>
                  Optimizer ·{" "}
                  {!snapshot.optimizer_update.applied
                    ? "skipped"
                    : snapshot.optimizer_update.policy_signal_applied === false
                      ? "anchor only"
                      : "applied"}
                </summary>
                <dl className="branch-evidence-facts">
                  <Fact
                    label="Adapter"
                    value={snapshot.optimizer_update.adapter_revision ?? "—"}
                  />
                  <Fact
                    label="Learning rate"
                    value={
                      snapshot.optimizer_update.learning_rate?.toExponential(
                        1,
                      ) ?? "—"
                    }
                  />
                  <Fact
                    label="Gradient norm"
                    value={
                      snapshot.optimizer_update.gradient_norm?.toFixed(3) ?? "—"
                    }
                  />
                  <Fact
                    label="Total loss"
                    value={
                      snapshot.optimizer_update.policy_loss?.toFixed(3) ?? "—"
                    }
                  />
                  <Fact
                    label="REINFORCE"
                    value={
                      snapshot.optimizer_update.reinforce_loss?.toFixed(3) ??
                      "—"
                    }
                  />
                  <Fact
                    label="Reference KL"
                    value={
                      snapshot.optimizer_update.reference_kl?.toFixed(3) ?? "—"
                    }
                  />
                  <Fact
                    label="Policy / anchor examples"
                    value={`${snapshot.optimizer_update.training_examples ?? 0} / ${
                      snapshot.optimizer_update.reference_examples ?? 0
                    }`}
                  />
                  {snapshot.optimizer_update.policy_credit_scope ? (
                    <Fact
                      label="Policy credit"
                      value={
                        snapshot.optimizer_update.policy_credit_scope ===
                        "fault_fixing_edits_from_verified_successful_siblings"
                          ? "Verified fault-fixing edits"
                          : friendlyStatus(
                              snapshot.optimizer_update.policy_credit_scope,
                            )
                      }
                    />
                  ) : null}
                  <Fact
                    label="Signal groups"
                    value={`${
                      snapshot.optimizer_update.policy_signal_group_count ?? 0
                    } applied · ${
                      snapshot.optimizer_update
                        .pending_informative_group_count ?? 0
                    } pending`}
                  />
                  {snapshot.optimizer_update.policy_signal_group_ids?.length ? (
                    <Fact
                      label="Signal sources"
                      value={snapshot.optimizer_update.policy_signal_group_ids.join(
                        " · ",
                      )}
                    />
                  ) : null}
                  {(snapshot.optimizer_update.pending_training_examples ?? 0) >
                  0 ? (
                    <Fact
                      label="Queued examples"
                      value={`${
                        snapshot.optimizer_update.pending_policy_examples ?? 0
                      } policy · ${
                        snapshot.optimizer_update.pending_training_examples ?? 0
                      } total`}
                    />
                  ) : null}
                  {snapshot.optimizer_update
                    .policy_signal_consumed_by_update !== undefined ? (
                    <Fact
                      label="Credit applied"
                      value={`Update ${snapshot.optimizer_update.policy_signal_consumed_by_update}`}
                    />
                  ) : null}
                  <Fact
                    label="Effective weight"
                    value={
                      snapshot.optimizer_update.effective_batch_weight?.toFixed(
                        3,
                      ) ?? "—"
                    }
                  />
                </dl>
              </details>
            ) : null}
            {sibling?.failure_classification?.length ? (
              <details>
                <summary>Failure classification</summary>
                <ul className="branch-failure-list">
                  {sibling.failure_classification.map((item) => (
                    <li key={`${item.category}-${item.source}`}>
                      {friendlyStatus(item.category)}
                      <small>{item.source}</small>
                    </li>
                  ))}
                </ul>
              </details>
            ) : null}
          </div>
        </>
      ) : sibling ? (
        <>
          <dl className="trajectory-step-facts">
            <Fact
              label="Format"
              value={sibling.format_valid ? "Valid" : "Invalid"}
            />
            <Fact
              label="Verifier"
              value={sibling.passed ? "Passed" : "Failed"}
            />
            <Fact label="Advantage" value={formatSigned(sibling.advantage)} />
            <Fact
              label="Policy signal"
              value={sibling.policy_signal ? "Eligible" : "None"}
            />
          </dl>
          <div className="branch-evidence">
            <section>
              <span>Action</span>
              <code>{sibling.action ?? "No valid action parsed"}</code>
            </section>
            <section>
              <span>Raw response</span>
              <pre>{sibling.response || "Empty response"}</pre>
            </section>
          </div>
        </>
      ) : null}
      <div className="step-navigation" aria-label="Action navigation">
        <button
          type="button"
          disabled={!previous}
          onClick={() =>
            previous && selectAction(previous.actionId, previous.sibling?.index)
          }
        >
          ← Previous
        </button>
        <button
          type="button"
          disabled={!next}
          onClick={() =>
            next && selectAction(next.actionId, next.sibling?.index)
          }
        >
          Next →
        </button>
      </div>
    </aside>
  );
}

function branchNavigation(
  snapshot: ResearchBranchSnapshot,
  selectedSibling: ResearchBranchSibling | null,
): BranchSelection[] {
  if (!snapshot.shared_prefix) {
    return snapshot.siblings.map((sibling) => ({
      actionId: legacyActionId(sibling),
      title: `Sibling ${sibling.index + 1}`,
      sibling,
      step: null,
    }));
  }
  const items: BranchSelection[] = snapshot.shared_prefix.steps.map(
    (step, index) => ({
      actionId: prefixActionId(step),
      title: `Shared prefix · step ${index + 1}`,
      sibling: null,
      step,
    }),
  );
  if (selectedSibling) {
    items.push(
      ...(selectedSibling.steps ?? []).map((step, index) => ({
        actionId: siblingActionId(selectedSibling, step),
        title: `Sibling ${selectedSibling.index + 1} · step ${index + 1}`,
        sibling: selectedSibling,
        step,
      })),
    );
  }
  return items;
}

export function defaultBranchActionId(
  snapshot: ResearchBranchSnapshot,
  sibling: ResearchBranchSibling | null,
): string {
  if (snapshot.shared_prefix) {
    const firstSiblingStep = sibling?.steps?.[0];
    if (sibling && firstSiblingStep) {
      return siblingActionId(sibling, firstSiblingStep);
    }
    const lastPrefixStep = snapshot.shared_prefix.steps.at(-1);
    return lastPrefixStep ? prefixActionId(lastPrefixStep) : "";
  }
  return sibling ? legacyActionId(sibling) : "";
}

function prefixActionId(step: ResearchBranchStep): string {
  return `prefix-${step.index}`;
}

function siblingActionId(
  sibling: ResearchBranchSibling,
  step: ResearchBranchStep,
): string {
  return `sibling-${sibling.index}-${step.index}`;
}

function legacyActionId(sibling: ResearchBranchSibling): string {
  return `sibling-${sibling.index}`;
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export function branchSnapshotLabel(snapshot: ResearchBranchSnapshot): string {
  const replay = snapshot.replay ? " · Replay" : "";
  const probe =
    snapshot.curriculum_role === "adjacent_complexity_probe"
      ? " · Complexity probe"
      : "";
  return `Update ${snapshot.update} · Level ${snapshot.level}${replay}${probe}`;
}

function siblingReturn(sibling: ResearchBranchSibling): number {
  return sibling.return ?? sibling.reward ?? 0;
}

function stepOutcome(step: ResearchBranchStep): string {
  if (!step.accepted) return "Rejected";
  if (step.terminal_reason) return friendlyStatus(step.terminal_reason);
  if (step.tool === "test") {
    return `${step.fixed_faults}/${step.total_faults} fixed`;
  }
  return "Accepted";
}

function formatAction(
  action: string | Record<string, string> | null | undefined,
): string {
  if (!action) return "No valid action";
  return typeof action === "string" ? action : JSON.stringify(action, null, 2);
}

function compactAction(
  action: string | Record<string, string> | null | undefined,
): string {
  if (!action) return "No valid action";
  if (typeof action === "object") {
    const tool = action.tool ?? "action";
    const target = action.path ?? action.query ?? "";
    const compactTarget = action.path
      ? (action.path.split("/").at(-1) ?? action.path)
      : target;
    const value = compactTarget ? `${tool} · ${compactTarget}` : tool;
    return value.length > 44 ? `${value.slice(0, 41)}…` : value;
  }
  return action.length > 54 ? `${action.slice(0, 51)}…` : action;
}

function compactObservation(observation: string): string {
  return observation.length > 72 ? `${observation.slice(0, 69)}…` : observation;
}

function formatReward(value: number): string {
  return value.toFixed(3);
}

function formatSigned(value: number): string {
  if (Math.abs(value) < 0.0005) return "0.000";
  return `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
}
