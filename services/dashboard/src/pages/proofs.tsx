import { useApi } from "../api";
import {
  AsyncState,
  formatDate,
  friendlyStatus,
  KeyValue,
  MachineId,
  PageHeader,
  Section,
  shortId,
  StatusBadge,
} from "../components";
import {
  formatDuration,
  formatEstimatedCost,
  formatRelativeTime,
  formatReward,
  formatSignedReward,
} from "../format";
import {
  decodeProofDetail,
  decodeProofsResponse,
  type ProofsResponse,
} from "../contracts";
import { Link, useParams } from "../router";
import type { ResearchProofDetail, ResearchProofSummary } from "../types";

export function ProofsPage() {
  const proofs = useApi<ProofsResponse>(
    "/v1/proofs",
    5_000,
    decodeProofsResponse,
  );
  const items = proofs.data?.items ?? [];

  return (
    <>
      <PageHeader title="Proofs" />
      <div className="content workspace-content">
        <AsyncState
          loading={proofs.loading}
          error={proofs.error}
          stale={Boolean(proofs.data && proofs.error)}
          onRetry={proofs.retry}
        >
          {items.length ? (
            <section className="proof-list" aria-label="Training proofs">
              <div className="proof-list-header" aria-hidden="true">
                <span>Proof</span>
                <span>Completed</span>
                <span>Learning</span>
                <span>Hardware</span>
                <span>Total cost</span>
                <span>Teardown</span>
              </div>
              {items.map((proof) => (
                <ProofRow key={proof.proof_id} proof={proof} />
              ))}
            </section>
          ) : proofs.data ? (
            <div className="state-panel empty-state">
              <strong>No proofs yet</strong>
              <p>Completed runs with confirmed teardown appear here.</p>
              <Link className="button primary" to="/runs">
                View runs
              </Link>
            </div>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

function ProofRow({ proof }: { proof: ResearchProofSummary }) {
  return (
    <Link className="proof-row" to={`/proofs/${proof.proof_id}`}>
      <span className="proof-identity">
        <strong>{proof.run_name ?? "Training proof"}</strong>
        <code>{shortId(proof.proof_id)}</code>
      </span>
      <time
        dateTime={proof.completed_at}
        aria-label={formatDate(proof.completed_at)}
        title={formatDate(proof.completed_at)}
      >
        {formatRelativeTime(proof.completed_at)}
      </time>
      <span className="proof-learning">
        {formatReward(proof.learning.initial_reward)} →{" "}
        {formatReward(proof.learning.final_reward)}
        <small>{formatSignedReward(proof.learning.reward_gain)}</small>
      </span>
      <span className="proof-hardware">
        {proof.hardware.gpu ?? "Unknown GPU"}
        <small>{proof.hardware.provider}</small>
      </span>
      <span>{formatEstimatedCost(proof.cost)}</span>
      <StatusBadge status={proof.teardown_confirmed ? "RELEASED" : "WARNING"} />
    </Link>
  );
}

export function ProofDetailPage() {
  const { proofId = "" } = useParams<{ proofId: string }>();
  const proof = useApi<ResearchProofDetail>(
    `/v1/proofs/${encodeURIComponent(proofId)}`,
    5_000,
    decodeProofDetail,
  );
  const item = proof.data;

  return (
    <>
      <PageHeader
        eyebrow={<Link to="/proofs">Proofs</Link>}
        title={item?.run_name ?? "Training proof"}
        actions={
          item ? (
            <>
              {item.execution_id ? (
                <>
                  <Link
                    className="button secondary"
                    to={`/runs/research/${item.execution_id}`}
                  >
                    View run
                  </Link>
                  <Link
                    className="button primary"
                    to={`/runs/research/${item.execution_id}/trajectory`}
                  >
                    View trajectory
                  </Link>
                </>
              ) : null}
              <StatusBadge
                status={item.teardown_confirmed ? "RELEASED" : "WARNING"}
              />
            </>
          ) : undefined
        }
      />
      <div className="content workspace-content proof-detail-page">
        <AsyncState
          loading={proof.loading}
          error={proof.error}
          stale={Boolean(proof.data && proof.error)}
          onRetry={proof.retry}
        >
          {item ? <ProofEvidence proof={item} /> : null}
        </AsyncState>
      </div>
    </>
  );
}

function ProofEvidence({ proof }: { proof: ResearchProofDetail }) {
  const reached = proof.curriculum.reached_level;
  const maximum = proof.curriculum.maximum_level;
  const curriculum =
    typeof reached === "number"
      ? `Level ${reached}${typeof maximum === "number" ? ` of ${maximum}` : ""}`
      : "Unavailable";

  return (
    <>
      <Section title="Learning outcome">
        <KeyValue
          items={[
            {
              label: "Initial reward",
              value: formatReward(proof.learning.initial_reward),
            },
            {
              label: "Final reward",
              value: formatReward(proof.learning.final_reward),
            },
            {
              label: "Reward gain",
              value: formatSignedReward(proof.learning.reward_gain),
            },
            {
              label: "Claim",
              value:
                proof.learning.claim_strength === "EXPLORATORY_SINGLE_SEED"
                  ? "Exploratory · one seed"
                  : proof.learning.claim_strength
                    ? friendlyStatus(proof.learning.claim_strength)
                    : "Not recorded",
            },
          ]}
        />
      </Section>

      <div className="proof-detail-grid">
        <Section title="Workload and model">
          <KeyValue
            items={[
              { label: "Workload", value: proof.workload.id ?? "Unavailable" },
              {
                label: "Revision",
                value: proof.workload.revision ?? "Unavailable",
              },
              {
                label: "Model",
                value: proof.workload.model_id ?? "Unavailable",
              },
              {
                label: "Model revision",
                value: proof.workload.model_revision ? (
                  <MachineId value={proof.workload.model_revision} />
                ) : (
                  "Unavailable"
                ),
              },
              {
                label: "Algorithm",
                value: proof.workload.algorithm ?? "Unavailable",
              },
              {
                label: "Objective",
                value: proof.workload.objective ?? "Unavailable",
              },
              {
                label: "Branching",
                value:
                  typeof proof.workload.branch_width === "number"
                    ? `Static K=${proof.workload.branch_width}`
                    : "Unavailable",
              },
              {
                label: "Complexity",
                value: proof.workload.complexity_strategy ?? "Unavailable",
              },
              {
                label: "Domains",
                value: proof.workload.task_domains?.join(", ") ?? "Unavailable",
              },
            ]}
          />
        </Section>

        <Section title="Compute">
          <KeyValue
            items={[
              { label: "GPU", value: proof.hardware.gpu ?? "Unavailable" },
              { label: "Provider", value: proof.provider.name },
              {
                label: "Cloud",
                value: proof.hardware.cloud_type ?? "Unavailable",
              },
              { label: "Image", value: proof.hardware.image ?? "Unavailable" },
              {
                label: "Hourly rate",
                value:
                  proof.hardware.hourly_rate_usd === null
                    ? "Unavailable"
                    : `$${proof.hardware.hourly_rate_usd.toFixed(3)}/hour`,
              },
              {
                label: "Runtime",
                value: formatDuration(proof.runtime_seconds),
              },
              {
                label: "Total cost",
                value: formatEstimatedCost(proof.cost),
              },
            ]}
          />
        </Section>

        <Section title="Curriculum">
          <KeyValue
            items={[
              { label: "Progression", value: curriculum },
              {
                label: "Promotions",
                value: proof.curriculum.promotion_count ?? "Unavailable",
              },
              {
                label: "Updates",
                value: proof.curriculum.updates_completed ?? "Unavailable",
              },
              {
                label: "Stop reason",
                value: proof.curriculum.stop_reason
                  ? friendlyStatus(proof.curriculum.stop_reason)
                  : "Unavailable",
              },
              {
                label: "Retention",
                value:
                  proof.curriculum.retention_passed === true
                    ? "Passed"
                    : proof.curriculum.retention_passed === false
                      ? "Failed"
                      : "Not recorded",
              },
            ]}
          />
        </Section>

        <Section title="Evidence">
          <KeyValue
            items={[
              {
                label: "Proof",
                value: <MachineId value={proof.proof_id} />,
                span: true,
              },
              {
                label: "Proof receipt",
                value: <MachineId value={proof.evidence.receipt_digest} />,
                span: true,
              },
              ...(proof.evidence.failure_receipt_digest
                ? [
                    {
                      label: "Prior failure receipt",
                      value: (
                        <MachineId
                          value={proof.evidence.failure_receipt_digest}
                        />
                      ),
                      span: true,
                    },
                  ]
                : []),
              {
                label: "Provider handle",
                value: <MachineId value={proof.provider.handle} />,
                span: true,
              },
              {
                label: "Completed",
                value: (
                  <time dateTime={proof.completed_at}>
                    {formatDate(proof.completed_at)}
                  </time>
                ),
              },
              {
                label: "Teardown",
                value: proof.teardown_confirmed ? "Confirmed" : "Unconfirmed",
              },
            ]}
          />
        </Section>
      </div>
    </>
  );
}
