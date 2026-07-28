import { FormEvent, useState } from "react";
import { useApi } from "../api";
import {
  AsyncState,
  formatDate,
  KeyValue,
  MachineId,
  Notice,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";
import { Link, useParams, useSearchParams } from "../router";

interface ProofIndexItem {
  proof_bundle_id: string;
  digest: string;
  subject_type: string;
  subject_id: string;
  created_at: string;
  verification_run_id: string;
  status: string;
  plan_id: string;
  rejudges_verification_run_id: string | null;
  run_id: string;
  run_name: string;
  study_id: string;
  study_condition: string;
  label: string;
}

export function ProofsPage() {
  const [params, setParams] = useSearchParams();
  const [query, setQuery] = useState(params.get("q") ?? "");
  const path = `/v1/proofs?${new URLSearchParams([
    ["limit", "50"],
    ...(params.get("q") ? [["q", params.get("q")!]] : []),
    ...(params.get("status") ? [["status", params.get("status")!]] : []),
  ]).toString()}`;
  const { data, error, loading } = useApi<{
    items: ProofIndexItem[];
    total: number;
  }>(path, 10_000);

  function submit(event: FormEvent) {
    event.preventDefault();
    const next = new URLSearchParams(params);
    if (query.trim()) next.set("q", query.trim());
    else next.delete("q");
    setParams(next);
  }

  return (
    <>
      <PageHeader
        eyebrow="Immutable evidence"
        title="Proofs"
        description="Inspect and download the receipt, data protocol, exact run configuration, policy lineage, and teardown evidence behind a verification."
      />
      <div className="content">
        <form className="index-controls" onSubmit={submit}>
          <label className="search-field">
            <span>Search proofs</span>
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Run name, proof ID, or digest"
            />
          </label>
          <label>
            <span>Verification status</span>
            <select
              value={params.get("status") ?? ""}
              onChange={(event) => {
                const next = new URLSearchParams(params);
                if (event.target.value) next.set("status", event.target.value);
                else next.delete("status");
                setParams(next);
              }}
            >
              <option value="">All statuses</option>
              <option value="SUCCEEDED">Succeeded</option>
              <option value="ABSTAINED">Abstained</option>
              <option value="DISAGREEMENT">Disagreement</option>
              <option value="INTEGRITY_VIOLATION">Integrity violation</option>
            </select>
          </label>
          <button className="button secondary" type="submit">
            Apply
          </button>
          <span className="index-count">{data?.total ?? 0} proofs</span>
        </form>
        <AsyncState
          loading={loading}
          error={error}
          empty={Boolean(data && !data.items.length)}
        >
          <Section title="Verifiable receipts">
            <div className="table-wrap desktop-index">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Proof</th>
                    <th>Run</th>
                    <th>Study condition</th>
                    <th>Evidence type</th>
                    <th>Verification</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.items.map((proof) => (
                    <tr key={proof.proof_bundle_id}>
                      <td>
                        <Link
                          className="row-link"
                          to={`/proofs/${proof.proof_bundle_id}`}
                        >
                          <strong>{proof.label}</strong>
                          <MachineId
                            value={proof.proof_bundle_id}
                            copy={false}
                          />
                        </Link>
                      </td>
                      <td>
                        <Link to={`/runs/${proof.run_id}`}>
                          {proof.run_name}
                        </Link>
                      </td>
                      <td>
                        {proof.study_condition
                          .toLowerCase()
                          .replaceAll("_", " ")}
                      </td>
                      <td>{proof.subject_type.toLowerCase()}</td>
                      <td>
                        <StatusBadge status={proof.status} />
                      </td>
                      <td>{formatDate(proof.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="mobile-index proof-cards">
              {data?.items.map((proof) => (
                <li key={proof.proof_bundle_id}>
                  <Link to={`/proofs/${proof.proof_bundle_id}`}>
                    <span>
                      <strong>{proof.label}</strong>
                      <small>{proof.run_name}</small>
                    </span>
                    <StatusBadge status={proof.status} />
                    <span>
                      {proof.study_condition.toLowerCase().replaceAll("_", " ")}
                    </span>
                    <time>{formatDate(proof.created_at)}</time>
                  </Link>
                </li>
              ))}
            </ul>
          </Section>
        </AsyncState>
      </div>
    </>
  );
}

interface ProofDetailResponse {
  proof: Record<string, any>;
  run: Record<string, any>;
  study: Record<string, any>;
  data_protocol: Record<string, any>;
  exact_run_manifest: Record<string, any>;
  artifacts: Array<Record<string, any>>;
  policy_versions: Array<Record<string, any>>;
  teardown_receipt: {
    complete: boolean;
    allocations: Array<Record<string, any>>;
  };
  evaluation: { examples: unknown[]; claim: string };
  downloads: Array<{ item: string; label: string }>;
}

export function ProofDetailPage() {
  const { proofBundleId = "" } = useParams();
  const { data, error, loading } = useApi<ProofDetailResponse>(
    `/v1/proofs/${proofBundleId}`,
  );
  return (
    <>
      <PageHeader
        eyebrow={<Link to="/proofs">Proofs</Link>}
        title="Evidence receipt"
        description={
          data ? (
            <MachineId value={String(data.proof.digest)} />
          ) : (
            "Reading receipt…"
          )
        }
        actions={
          data ? (
            <StatusBadge status={String(data.proof.verification_status)} />
          ) : null
        }
      />
      <div className="content proof-detail">
        <AsyncState loading={loading} error={error}>
          {data ? (
            <>
              <section className="decision-statement">
                <span>What this proves</span>
                <h2>
                  Verification produced an immutable evidence bundle for one{" "}
                  {String(data.proof.subject_type).toLowerCase()}. It does not
                  establish held-out model improvement.
                </h2>
                <div className="decision-badges">
                  <StatusBadge
                    status={String(data.proof.verification_status)}
                  />
                  <span>
                    Plan <code>{String(data.proof.plan_id)}</code>
                  </span>
                </div>
              </section>
              <div className="proof-downloads" aria-label="Download evidence">
                {data.downloads.map((download) => (
                  <a
                    className="button secondary"
                    key={download.item}
                    href={`/api/v1/proofs/${proofBundleId}/download?item=${download.item}`}
                    download
                  >
                    {download.label}
                  </a>
                ))}
              </div>
              <div className="run-overview">
                <Section title="Receipt identity">
                  <KeyValue
                    items={[
                      {
                        label: "Proof",
                        value: (
                          <MachineId
                            value={String(data.proof.proof_bundle_id)}
                          />
                        ),
                        span: true,
                      },
                      {
                        label: "Verification",
                        value: (
                          <MachineId
                            value={String(data.proof.verification_run_id)}
                          />
                        ),
                      },
                      {
                        label: "Subject",
                        value: (
                          <MachineId value={String(data.proof.subject_id)} />
                        ),
                      },
                    ]}
                  />
                </Section>
                <Section title="Run and study">
                  <KeyValue
                    items={[
                      {
                        label: "Run",
                        value: (
                          <Link to={`/runs/${String(data.run.run_id)}`}>
                            {String(data.run.name)}
                          </Link>
                        ),
                      },
                      {
                        label: "Study",
                        value: (
                          <Link to={`/studies/${String(data.study.study_id)}`}>
                            {String(data.study.study_id)}
                          </Link>
                        ),
                      },
                      {
                        label: "Condition",
                        value: String(data.study.condition)
                          .toLowerCase()
                          .replaceAll("_", " "),
                      },
                    ]}
                  />
                </Section>
                <Section title="Teardown">
                  <KeyValue
                    items={[
                      {
                        label: "Receipt",
                        value: (
                          <StatusBadge
                            status={
                              data.teardown_receipt.complete
                                ? "SUCCEEDED"
                                : "INCOMPLETE"
                            }
                          />
                        ),
                      },
                      {
                        label: "Allocations",
                        value: data.teardown_receipt.allocations.length,
                      },
                      {
                        label: "Held-out examples",
                        value: data.evaluation.examples.length,
                      },
                    ]}
                  />
                </Section>
              </div>
              <Notice tone="warning" title="Evaluation boundary">
                {data.evaluation.claim}
              </Notice>
              <Section
                title="Artifact inventory"
                aside={<span>{data.artifacts.length} references</span>}
              >
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Role</th>
                        <th>Trust</th>
                        <th>Visibility</th>
                        <th>Media type</th>
                        <th>Digest</th>
                        <th>Bytes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.artifacts.map((artifact) => (
                        <tr
                          key={`${String(artifact.artifact_id)}-${String(artifact.role)}`}
                        >
                          <td>
                            {artifact.downloadable ? (
                              <a
                                href={`/api/v1/artifacts/${String(artifact.artifact_id)}`}
                              >
                                {String(artifact.role).replaceAll("-", " ")}
                              </a>
                            ) : (
                              String(artifact.role).replaceAll("-", " ")
                            )}
                          </td>
                          <td>{String(artifact.trust_class).toLowerCase()}</td>
                          <td>
                            <StatusBadge status={String(artifact.visibility)} />
                          </td>
                          <td>
                            <code>{String(artifact.media_type)}</code>
                          </td>
                          <td>
                            <MachineId value={String(artifact.digest)} />
                          </td>
                          <td>
                            {Number(artifact.size_bytes).toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>
              <Section title="Technical details">
                <details className="manifest-details">
                  <summary>Inspect immutable evidence manifest</summary>
                  <pre className="json-block">
                    {JSON.stringify(data.proof.manifest, null, 2)}
                  </pre>
                </details>
              </Section>
            </>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}
