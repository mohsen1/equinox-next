import { useApi } from "../api";
import {
  AsyncState,
  formatDate,
  MachineId,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";
import { Link, useParams } from "../router";
import type { ResearchStudyReport, ResearchStudySummary } from "../types";

export function StudiesPage() {
  const studies = useApi<{ items: ResearchStudySummary[] }>(
    "/v1/studies",
    10_000,
  );

  return (
    <>
      <PageHeader title="Studies" />
      <div className="content workspace-content">
        <AsyncState
          loading={studies.loading}
          error={studies.error}
          stale={Boolean(studies.data && studies.error)}
          onRetry={studies.retry}
          empty={!studies.data?.items.length}
        >
          {studies.data?.items.length ? (
            <div className="study-list">
              {studies.data.items.map((study) => (
                <Link
                  className="study-row"
                  key={study.study_id}
                  to={`/studies/${study.study_id}`}
                >
                  <span>
                    <strong>{study.study_id}</strong>
                    <small>{study.model_id ?? "Model unavailable"}</small>
                  </span>
                  <StatusBadge status={study.overall_status} />
                  <span>
                    {study.condition_count} conditions
                    <small>{study.execution_count} executions</small>
                  </span>
                  <span>
                    ${study.estimated_provider_cost_usd.toFixed(2)}
                    <small>estimated</small>
                  </span>
                  <time dateTime={study.generated_at}>
                    {formatDate(study.generated_at)}
                  </time>
                </Link>
              ))}
            </div>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

export function StudyPage() {
  const { studyId = "" } = useParams<{ studyId: string }>();
  const study = useApi<ResearchStudyReport>(
    `/v1/studies/${encodeURIComponent(studyId)}`,
    10_000,
  );
  const report = study.data;
  const conditions = report ? Object.entries(report.conditions) : [];
  const decisions = report ? Object.entries(report.decisions) : [];
  const external = report?.external_evaluation;
  const estimatedCost =
    report?.executions.reduce(
      (total, execution) => total + (execution.estimated_cost_usd ?? 0),
      0,
    ) ?? 0;

  return (
    <>
      <PageHeader
        eyebrow={<Link to="/studies">Studies</Link>}
        title={report?.study_id ?? studyId}
        actions={
          report ? <StatusBadge status={report.overall_status} /> : undefined
        }
      />
      <div className="content workspace-content study-page">
        <AsyncState
          loading={study.loading}
          error={study.error}
          stale={Boolean(study.data && study.error)}
          onRetry={study.retry}
        >
          {report ? (
            <>
              <Section title="Decision">
                <div className="study-decision-list">
                  {decisions.map(([decisionId, decision]) => (
                    <div key={decisionId}>
                      <span>{decisionId.replaceAll("_", " ")}</span>
                      <StatusBadge status={decision.status} />
                    </div>
                  ))}
                </div>
              </Section>

              <Section title="Conditions">
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Condition</th>
                        <th>Seed</th>
                        <th>K</th>
                        <th>Status</th>
                        <th>Initial</th>
                        <th>Final</th>
                        <th>Paired change</th>
                        <th>Completions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {conditions.map(([conditionId, condition]) => (
                        <tr key={conditionId}>
                          <td>
                            <strong>{conditionId}</strong>
                            <small>
                              {condition.role?.replaceAll("_", " ") ?? "—"}
                            </small>
                          </td>
                          <td>{condition.seed ?? "—"}</td>
                          <td>{condition.branch_width ?? "—"}</td>
                          <td>
                            <StatusBadge
                              status={condition.status ?? "UNKNOWN"}
                            />
                          </td>
                          <td>{formatRate(condition.initial_successes)}</td>
                          <td>{formatRate(condition.final_successes)}</td>
                          <td>
                            +{condition.paired_improved ?? 0} / −
                            {condition.paired_regressed ?? 0}
                          </td>
                          <td>{condition.sampled_completions ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>

              <div className="study-facts">
                <Section title="External evaluation">
                  <dl className="key-value">
                    <div>
                      <dt>Pack</dt>
                      <dd>{external?.pack_id ?? "Unavailable"}</dd>
                    </div>
                    <div>
                      <dt>Tasks</dt>
                      <dd>{external?.task_count ?? "—"}</dd>
                    </div>
                    <div>
                      <dt>Domains</dt>
                      <dd>
                        {external?.domain_task_counts
                          ? Object.entries(external.domain_task_counts)
                              .map(([domain, count]) => `${domain}: ${count}`)
                              .join(" · ")
                          : "—"}
                      </dd>
                    </div>
                  </dl>
                </Section>
                <Section title="Evidence">
                  <dl className="key-value">
                    <div>
                      <dt>Estimated provider cost</dt>
                      <dd>${estimatedCost.toFixed(2)}</dd>
                    </div>
                    <div>
                      <dt>Provider failures</dt>
                      <dd>{report.failure_count.provider_executions}</dd>
                    </div>
                    <div>
                      <dt>Operator failures</dt>
                      <dd>{report.failure_count.operator_attempts}</dd>
                    </div>
                    <div>
                      <dt>Report</dt>
                      <dd>
                        <MachineId value={report.report_digest} />
                      </dd>
                    </div>
                  </dl>
                </Section>
              </div>
            </>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

function formatRate(value: number | undefined): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}
