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
import type { ResearchBranchSibling, ResearchBranchSnapshot } from "../types";

type BranchView = "graph" | "outline";

interface BranchNodeData extends Record<string, unknown> {
  siblingIndex?: number;
  title: string;
  detail: string;
  action?: string | null;
  reward?: number;
  passed?: boolean;
  best?: boolean;
}

type BranchFlowNode = Node<BranchNodeData>;

const branchNodeTypes = {
  task: BranchTaskNode,
  sibling: BranchSiblingNode,
};

export function ResearchBranchWorkspace({
  snapshot,
  selectedSibling,
  view,
  selectSibling,
}: {
  snapshot: ResearchBranchSnapshot;
  selectedSibling: ResearchBranchSibling;
  view: BranchView;
  selectSibling: (index: number) => void;
}) {
  const flow = buildBranchFlow(snapshot, selectedSibling.index);
  const siblingPosition = snapshot.siblings.findIndex(
    (sibling) => sibling.index === selectedSibling.index,
  );

  return (
    <div className="research-trajectory-workspace">
      <section
        className="research-trajectory-canvas"
        aria-label="K=4 sibling actions"
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
            }}
            fitView
            fitViewOptions={{ padding: 0.22 }}
            minZoom={0.5}
            maxZoom={1.5}
            nodesDraggable={false}
            nodesConnectable={false}
            nodesFocusable
            edgesFocusable={false}
            autoPanOnNodeFocus
            aria-label="One task branching into four sampled and verified actions."
            proOptions={{ hideAttribution: true }}
          >
            <Background color="var(--rule)" gap={32} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
        ) : (
          <BranchOutline
            snapshot={snapshot}
            selectedIndex={selectedSibling.index}
            select={selectSibling}
          />
        )}
      </section>
      <BranchInspector
        snapshot={snapshot}
        sibling={selectedSibling}
        previous={snapshot.siblings[siblingPosition - 1]}
        next={snapshot.siblings[siblingPosition + 1]}
        select={selectSibling}
      />
    </div>
  );
}

export function buildBranchFlow(
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
        title: `Sibling ${sibling.index + 1}`,
        detail: sibling.passed ? "Verifier passed" : "Verifier failed",
        action: sibling.action,
        reward: sibling.reward,
        passed: sibling.passed,
        best: sibling.index === snapshot.best_sibling_index,
      },
    })),
  ];
  const edges = snapshot.siblings.map((sibling) => {
    const selected = sibling.index === selectedIndex;
    const best = sibling.index === snapshot.best_sibling_index;
    return {
      id: `${snapshot.snapshot_id}-${sibling.index}`,
      source: snapshot.snapshot_id,
      target: `${snapshot.snapshot_id}-sibling-${sibling.index}`,
      type: "smoothstep",
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 14,
        height: 14,
      },
      className: [
        sibling.passed ? "branch-edge-passed" : "branch-edge-failed",
        best ? "branch-edge-best" : "",
        selected ? "selected" : "",
      ]
        .filter(Boolean)
        .join(" "),
      style: {
        stroke: selected
          ? "var(--accent)"
          : sibling.passed
            ? "var(--success)"
            : "var(--rule-strong)",
        strokeWidth: selected ? 2 : 1.25,
      },
    };
  });
  return { nodes, edges };
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
      <code title={data.action ?? "No valid action"}>
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
  select,
}: {
  snapshot: ResearchBranchSnapshot;
  selectedIndex: number;
  select: (index: number) => void;
}) {
  return (
    <div className="research-trajectory-outline branch-outline">
      <table>
        <caption>K=4 sibling actions</caption>
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
              <td>{formatReward(sibling.reward)}</td>
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

function BranchInspector({
  snapshot,
  sibling,
  previous,
  next,
  select,
}: {
  snapshot: ResearchBranchSnapshot;
  sibling: ResearchBranchSibling;
  previous?: ResearchBranchSibling;
  next?: ResearchBranchSibling;
  select: (index: number) => void;
}) {
  const isBest = sibling.index === snapshot.best_sibling_index;
  return (
    <aside
      className="research-trajectory-inspector branch-inspector"
      aria-label="Selected sibling"
    >
      <header>
        <span>
          Update {snapshot.update} · Level {snapshot.level}
        </span>
        <StatusBadge status={sibling.passed ? "SUCCEEDED" : "FAILED"} />
      </header>
      <div className="trajectory-selected-step">
        <h2>Sibling {sibling.index + 1}</h2>
        <strong>{formatReward(sibling.reward)}</strong>
        <span>{isBest ? "best reward" : "reward"}</span>
      </div>
      <dl className="trajectory-step-facts">
        <Fact
          label="Format"
          value={sibling.format_valid ? "Valid" : "Invalid"}
        />
        <Fact label="Verifier" value={sibling.passed ? "Passed" : "Failed"} />
        <Fact label="Advantage" value={formatSigned(sibling.advantage)} />
        <Fact
          label="Policy signal"
          value={sibling.policy_signal ? "Applied" : "None"}
        />
        <Fact
          label="Group signal"
          value={snapshot.learning_signal ? "Informative" : "Uniform"}
        />
        <Fact
          label="Fallback"
          value={snapshot.teacher_fallback ? "Teacher" : "None"}
        />
      </dl>
      <div className="branch-evidence">
        <section>
          <span>Action</span>
          <code>{sibling.action ?? "No valid action parsed"}</code>
        </section>
        <section>
          <span>Expected action</span>
          <code>{snapshot.expected_action}</code>
        </section>
        <section>
          <span>Raw response</span>
          <pre>{sibling.response || "Empty response"}</pre>
        </section>
        <details>
          <summary>Task prompt</summary>
          <pre>{snapshot.prompt}</pre>
        </details>
      </div>
      <div className="step-navigation" aria-label="Sibling navigation">
        <button
          type="button"
          disabled={!previous}
          onClick={() => previous && select(previous.index)}
        >
          ← Previous
        </button>
        <button
          type="button"
          disabled={!next}
          onClick={() => next && select(next.index)}
        >
          Next →
        </button>
      </div>
    </aside>
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

export function branchSnapshotLabel(snapshot: ResearchBranchSnapshot): string {
  return `Update ${snapshot.update} · ${friendlyStatus(snapshot.domain)} · Level ${snapshot.level}`;
}

function compactAction(action: string | null | undefined): string {
  if (!action) return "No valid action";
  return action.length > 54 ? `${action.slice(0, 51)}…` : action;
}

function formatReward(value: number): string {
  return value.toFixed(3);
}

function formatSigned(value: number): string {
  if (Math.abs(value) < 0.0005) return "0.000";
  return `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
}
