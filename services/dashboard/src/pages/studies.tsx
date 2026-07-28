import { useApi } from "../api";
import {
  AsyncState,
  EvidenceStrengthBadge,
  ExecutionBadge,
  formatDate,
  LearningOutcomeBadge,
  MachineId,
  Notice,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";
import { Link, useParams } from "../router";
import type { StudySummary } from "../types";

export function StudyPage() {
  const { studyId = "" } = useParams();
  const { data, error, loading } = useApi<StudySummary>(
    `/v1/studies/${studyId}`,
    10_000,
  );

  return (
    <>
      <PageHeader
        eyebrow={<Link to="/runs">Runs</Link>}
        title="Study comparison"
        description={
          data?.research_question ?? "Reading matched study protocol…"
        }
        actions={
          data ? <StatusBadge status={data.comparison.status} /> : undefined
        }
      />
      <div className="content study-page">
        <AsyncState loading={loading} error={error}>
          {data ? (
            <>
              <section className="decision-statement">
                <span>Current conclusion</span>
                <h2>{data.comparison.learning_claim}</h2>
                <div className="decision-badges">
                  <StatusBadge status={data.comparison.status} />
                  <EvidenceStrengthBadge strength="CONTRACT_ONLY" />
                  <span>
                    {data.comparison.paired_seeds.length
                      ? `Paired seed${data.comparison.paired_seeds.length === 1 ? "" : "s"} ${data.comparison.paired_seeds.join(", ")}`
                      : "No paired seed completed yet"}
                  </span>
                </div>
              </section>

              <Section
                title="Conditions and matched controls"
                aside={<span>{data.protocol_revision}</span>}
              >
                <div className="table-wrap">
                  <table className="data-table study-conditions">
                    <caption>
                      Conditions may be compared only when every pinned protocol
                      constraint and at least one seed match.
                    </caption>
                    <thead>
                      <tr>
                        <th>Condition</th>
                        <th>Seeds</th>
                        <th>Execution</th>
                        <th>Learning outcome</th>
                        <th>Held-out test</th>
                        <th>Local cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.conditions.map((condition) => (
                        <tr key={condition.condition}>
                          <td>
                            <strong>
                              {condition.condition
                                .toLowerCase()
                                .replaceAll("_", " ")}
                            </strong>
                            <small>
                              {condition.run_count} persisted run(s)
                            </small>
                          </td>
                          <td>
                            {condition.seeds.length
                              ? condition.seeds.join(", ")
                              : "Pending"}
                          </td>
                          <td>
                            {condition.completed_count}/{condition.run_count}{" "}
                            completed
                          </td>
                          <td>
                            <LearningOutcomeBadge
                              outcome={
                                condition.learning_outcomes[0] ??
                                "NOT_EVALUATED"
                              }
                            />
                          </td>
                          <td>{condition.test_result_label}</td>
                          <td>
                            {condition.execution_credits.toFixed(3)} credits
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>

              <div className="study-protocol">
                <Section title="Comparison guardrails">
                  <ul className="check-list">
                    {data.comparison.constraints.map((constraint) => (
                      <li key={constraint}>
                        <span aria-hidden="true">✓</span>
                        {constraint}
                      </li>
                    ))}
                  </ul>
                </Section>
                <Section title="Pinned protocol">
                  <dl className="protocol-digest">
                    <dt>Protocol revision</dt>
                    <dd>
                      <code>{data.protocol_revision}</code>
                    </dd>
                    <dt>Match digest</dt>
                    <dd>
                      {data.comparison.protocol_digest ? (
                        <MachineId value={data.comparison.protocol_digest} />
                      ) : (
                        "Conflicting protocols"
                      )}
                    </dd>
                    <dt>Last activity</dt>
                    <dd>{formatDate(data.updated_at)}</dd>
                  </dl>
                </Section>
              </div>

              <Section title="Individual runs">
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Run</th>
                        <th>Condition</th>
                        <th>Seed</th>
                        <th>Execution</th>
                        <th>Learning</th>
                        <th>Evidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.runs?.map((run) => (
                        <tr key={run.run_id}>
                          <td>
                            <Link
                              className="row-link"
                              to={`/runs/${run.run_id}`}
                            >
                              <strong>{run.name}</strong>
                              <MachineId value={run.run_id} copy={false} />
                            </Link>
                          </td>
                          <td>
                            {run.study.condition
                              .toLowerCase()
                              .replaceAll("_", " ")}
                          </td>
                          <td>{run.manifest.seed ?? "—"}</td>
                          <td>
                            <ExecutionBadge status={run.status} />
                          </td>
                          <td>
                            <LearningOutcomeBadge
                              outcome={run.learning_outcome}
                            />
                          </td>
                          <td>
                            {run.proof_count} proof
                            {run.proof_count === 1 ? "" : "s"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>

              <Notice tone="warning" title="Scientific boundary">
                A matched protocol makes comparison valid; it does not create a
                learning result. Add held-out validation and test packs before
                claiming model improvement.
              </Notice>
            </>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}
