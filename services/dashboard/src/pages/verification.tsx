import { useEffect, useMemo } from "react";
import {
  Background,
  Controls,
  type Edge,
  MarkerType,
  type Node,
  ReactFlow,
} from "@xyflow/react";
import { Link, useParams, useSearchParams } from "../router";
import { useApi } from "../api";
import {
  AsyncState,
  KeyValue,
  MachineId,
  Notice,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";
import type { VerificationDetail, VerificationStep } from "../types";

interface VerificationGraphResponse extends VerificationDetail {
  nodes: Array<{
    id: string;
    step_id: string;
    type: string;
    status: string;
    attempt_count: number;
    cache_status: string;
    evidence_roles: string[];
    cost: Record<string, number>;
  }>;
  edges: Array<{
    id: string;
    source_step_id: string;
    target_step_id: string;
  }>;
  accessible_outline: Array<{
    position: number;
    step: string;
    depends_on: string[];
    status: string;
  }>;
}

export function VerificationPage() {
  const { verificationRunId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const { data, error, loading } = useApi<VerificationGraphResponse>(
    `/v1/verification-runs/${verificationRunId}/graph`,
  );
  const selectedStepId = params.get("step");
  const view = params.get("view") ?? "dag";

  useEffect(() => {
    if (!data || selectedStepId) return;
    const next = new URLSearchParams(params);
    next.set(
      "step",
      data.steps.find((step) => step.step_id === "pointwise-judge")?.step_id ??
        data.steps[0]?.step_id,
    );
    setParams(next, { replace: true });
  }, [data, params, selectedStepId, setParams]);

  const selectedStep =
    data?.steps.find((step) => step.step_id === selectedStepId) ??
    data?.steps[0];
  const flow = useMemo(
    () => (data ? buildVerificationFlow(data, selectedStepId) : null),
    [data, selectedStepId],
  );

  return (
    <>
      <PageHeader
        eyebrow={
          <>
            <Link to="/runs">Runs</Link> / verification
          </>
        }
        title="Verification DAG"
        description={<MachineId value={verificationRunId} />}
        actions={
          <div className="segmented" aria-label="Verification representation">
            <button
              type="button"
              aria-pressed={view === "dag"}
              onClick={() => setView(params, setParams, "dag")}
            >
              DAG
            </button>
            <button
              type="button"
              aria-pressed={view === "outline"}
              onClick={() => setView(params, setParams, "outline")}
            >
              Outline
            </button>
          </div>
        }
      />
      <AsyncState
        loading={loading}
        error={error}
        empty={Boolean(data && !data.nodes.length)}
      >
        {data && flow ? (
          <div className="verification-page">
            <Notice
              tone={
                data.judge?.result.abstained || data.judge?.result.disagreement
                  ? "warning"
                  : "neutral"
              }
              title="Evidence boundary"
            >
              {data.model_assessment_notice} Candidate-controlled source and
              renders remain untrusted data; the judge has no tools or writable
              systems.
            </Notice>
            <div className="verification-workspace">
              <section
                className="verification-canvas"
                aria-label="Verification plan"
              >
                {view === "dag" ? (
                  <ReactFlow
                    nodes={flow.nodes}
                    edges={flow.edges}
                    fitView
                    nodesDraggable={false}
                    nodesConnectable={false}
                    nodesFocusable
                    edgesFocusable
                    onNodeClick={(_, node) => {
                      const next = new URLSearchParams(params);
                      next.set("step", String(node.data.stepId));
                      setParams(next);
                    }}
                    aria-label="Verification DAG. Use Tab to navigate steps."
                    proOptions={{ hideAttribution: true }}
                  >
                    <Background color="var(--rule)" gap={32} size={1} />
                    <Controls showInteractive={false} />
                  </ReactFlow>
                ) : (
                  <table className="data-table verification-outline">
                    <caption>
                      Keyboard-navigable equivalent of the verification DAG
                    </caption>
                    <thead>
                      <tr>
                        <th>Order</th>
                        <th>Step</th>
                        <th>Depends on</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.accessible_outline.map((row) => (
                        <tr
                          key={row.step}
                          className={
                            selectedStepId === row.step ? "selected" : ""
                          }
                        >
                          <td>{row.position}</td>
                          <td>
                            <button
                              className="table-button"
                              type="button"
                              onClick={() => {
                                const next = new URLSearchParams(params);
                                next.set("step", row.step);
                                setParams(next);
                              }}
                            >
                              {row.step.replaceAll("-", " ")}
                            </button>
                          </td>
                          <td>{row.depends_on.join(", ") || "None"}</td>
                          <td>
                            <StatusBadge status={row.status} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </section>
              <StepInspector step={selectedStep} detail={data} />
            </div>
            {data.judge ? <JudgeAssessment detail={data} /> : null}
          </div>
        ) : null}
      </AsyncState>
    </>
  );
}

function buildVerificationFlow(
  data: VerificationGraphResponse,
  selectedStepId: string | null,
): { nodes: Node[]; edges: Edge[] } {
  const positions: Record<string, { x: number; y: number }> = {
    geometry: { x: 60, y: 40 },
    render: { x: 60, y: 200 },
    proof: { x: 320, y: 120 },
    "pointwise-judge": { x: 580, y: 200 },
    aggregate: { x: 820, y: 120 },
    "group-judge": { x: 320, y: 120 },
  };
  return {
    nodes: data.nodes.map((node, index) => ({
      id: node.id,
      position: positions[node.step_id] ?? { x: 60 + index * 220, y: 120 },
      data: {
        stepId: node.step_id,
        label: (
          <div className="verification-node-label">
            <strong>{node.step_id.replaceAll("-", " ")}</strong>
            <span>{node.type.replaceAll("_", " ")}</span>
            <StatusBadge status={node.status} />
            <small>
              {node.attempt_count} attempt{node.attempt_count === 1 ? "" : "s"}{" "}
              · {node.cache_status.toLowerCase()}
            </small>
          </div>
        ),
      },
      className:
        selectedStepId === node.step_id
          ? "verification-node selected"
          : "verification-node",
      draggable: false,
      ariaLabel: `${node.step_id}, ${node.status}, ${node.attempt_count} attempts`,
    })),
    edges: data.edges.flatMap((edge) => {
      const source = data.nodes.find(
        (node) => node.step_id === edge.source_step_id,
      );
      const target = data.nodes.find(
        (node) => node.step_id === edge.target_step_id,
      );
      if (!source || !target) return [];
      return [
        {
          id: edge.id,
          source: source.id,
          target: target.id,
          type: "smoothstep",
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
          ariaLabel: `${edge.source_step_id} feeds ${edge.target_step_id}`,
        },
      ];
    }),
  };
}

function StepInspector({
  step,
  detail,
}: {
  step: VerificationStep | undefined;
  detail: VerificationGraphResponse;
}) {
  if (!step) return <aside className="step-inspector">No step selected.</aside>;
  return (
    <aside className="step-inspector">
      <div className="inspector-heading">
        <span>Selected step</span>
        <StatusBadge status={step.status} />
      </div>
      <h2>{step.step_id.replaceAll("-", " ")}</h2>
      <MachineId value={step.step_run_id} />
      <KeyValue
        items={[
          { label: "Type", value: step.step_type.replaceAll("_", " ") },
          { label: "Attempts", value: step.attempt_count },
          { label: "Cache", value: step.cache_status },
          {
            label: "Evidence roles",
            value: step.evidence_roles.join(", "),
            span: true,
          },
          {
            label: "Cost",
            value: Object.entries(step.cost)
              .map(([key, value]) => `${key}: ${value}`)
              .join(" · "),
            span: true,
          },
        ]}
      />
      {step.failure ? (
        <Notice tone="warning" title="Recovered operational attempt">
          <pre>{JSON.stringify(step.failure, null, 2)}</pre>
        </Notice>
      ) : null}
      {Object.keys(step.metrics).length ? (
        <>
          <h3>Typed result</h3>
          <pre className="json-block compact">
            {JSON.stringify(step.metrics, null, 2)}
          </pre>
        </>
      ) : null}
      <h3>Proof input</h3>
      <KeyValue
        items={[
          {
            label: "Proof bundle",
            value: (
              <MachineId
                value={String(detail.proof_bundle?.manifest.proof_bundle_id)}
              />
            ),
          },
          {
            label: "Digest",
            value: (
              <MachineId value={String(detail.proof_bundle?.manifest.digest)} />
            ),
          },
        ]}
      />
    </aside>
  );
}

function JudgeAssessment({ detail }: { detail: VerificationGraphResponse }) {
  const judge = detail.judge!;
  return (
    <Section
      title="Model assessment"
      aside={
        <>
          <StatusBadge status={judge.result.outcome} /> Confidence{" "}
          {Math.round(judge.result.confidence * 100)}%
        </>
      }
    >
      <div className="judge-grid">
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Criterion</th>
                <th>Score</th>
                <th>Confidence</th>
                <th>Evidence roles</th>
              </tr>
            </thead>
            <tbody>
              {judge.result.assessments.map((assessment) => (
                <tr key={assessment.criterion}>
                  <td>{assessment.criterion.replaceAll("_", " ")}</td>
                  <td>{assessment.score.toFixed(2)}</td>
                  <td>{Math.round(assessment.confidence * 100)}%</td>
                  <td>{assessment.evidence_roles.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <KeyValue
            items={[
              {
                label: "Provider model",
                value: <code>{judge.result.provider_model_identity}</code>,
              },
              {
                label: "Specification",
                value: <code>{String(judge.spec.judge_spec_id)}</code>,
              },
              {
                label: "Rubric",
                value: <code>{String(judge.spec.rubric_version)}</code>,
              },
              {
                label: "Calibration",
                value: <StatusBadge status="FIXTURE_CONTRACT_ONLY" />,
              },
              {
                label: "Abstained",
                value: judge.result.abstained ? "Yes" : "No",
              },
              {
                label: "Disagreement",
                value: judge.result.disagreement ? "Yes" : "No",
              },
              {
                label: "Integrity flags",
                value: judge.result.integrity_flags.join(", ") || "None",
              },
              {
                label: "Usage",
                value: Object.entries(judge.result.usage)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join(" · "),
                span: true,
              },
            ]}
          />
          <Notice title="Concise evidence-based explanation">
            {judge.result.explanation}
          </Notice>
        </div>
      </div>
    </Section>
  );
}

function setView(
  params: URLSearchParams,
  setParams: ReturnType<typeof useSearchParams>[1],
  view: string,
) {
  const next = new URLSearchParams(params);
  next.set("view", view);
  setParams(next);
}
