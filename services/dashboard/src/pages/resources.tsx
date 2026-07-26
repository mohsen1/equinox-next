import { useApi } from "../api";
import { AsyncState, MachineId, PageHeader, StatusBadge } from "../components";

interface ResourcesResponse {
  allocations: Array<Record<string, any>>;
  research_compute_proofs: Array<{
    proof_id: string;
    provider_name: string;
    provider_handle: string;
    provider_cli_version: string;
    resource_profile: {
      gpu_id?: string;
      image?: string;
      hourly_cost_usd?: number;
    };
    workload: {
      id?: string;
      static_branch_width?: number;
      complexity_strategy?: string;
      scale_case_count?: number;
      maximum_horizon?: number;
      total_action_decisions?: number;
      model_id?: string;
      task_domains?: string[];
      maximum_complexity_level?: number;
      reached_complexity_level?: number;
      total_sampled_completions?: number;
    };
    result: {
      initial_reward?: number;
      final_reward?: number;
      reward_gain?: number;
      promotion_count?: number;
      elapsed_seconds?: number;
      gpu_name?: string;
      hypothesis_passed?: boolean;
    };
    completed_at: string;
    teardown_confirmed: boolean;
  }>;
  counts: {
    allocations: number;
    snapshots: number;
    checkpoints: number;
    pending_operations: number;
  };
  provider_boundaries: Record<string, string[]>;
  external_capacity: number;
}

export function ResourcesPage() {
  const { data, error, loading } = useApi<ResourcesResponse>(
    "/v1/resources",
    2_000,
  );
  return (
    <>
      <PageHeader title="Proofs" />
      <div className="content workspace-content">
        <AsyncState
          loading={loading}
          error={error}
          empty={!data?.research_compute_proofs.length}
        >
          {data ? (
            <section className="section" aria-label="Training proofs">
              <div className="table-wrap">
                <table className="data-table research-proofs-table">
                  <thead>
                    <tr>
                      <th>Proof</th>
                      <th>Workload</th>
                      <th>Hardware</th>
                      <th>Learning signal</th>
                      <th>Curriculum</th>
                      <th>Cleanup</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.research_compute_proofs.map((proof) => (
                      <tr key={proof.proof_id}>
                        <td data-label="Proof">
                          <MachineId value={proof.proof_id} />
                          <small>{proof.provider_name}</small>
                        </td>
                        <td data-label="Workload">
                          {proof.workload.id ?? "Unknown"}
                          <small>
                            Static K={proof.workload.static_branch_width} ·{" "}
                            {proof.workload.complexity_strategy}
                            {proof.workload.scale_case_count
                              ? ` · ${proof.workload.scale_case_count} scale cases`
                              : ""}
                            {proof.workload.maximum_horizon
                              ? ` · horizon ${proof.workload.maximum_horizon}`
                              : ""}
                            {proof.workload.model_id
                              ? ` · ${shortModelName(proof.workload.model_id)}`
                              : ""}
                            {proof.workload.task_domains
                              ? ` · ${proof.workload.task_domains.length} domains`
                              : ""}
                          </small>
                        </td>
                        <td data-label="Hardware">
                          {proof.result.gpu_name ??
                            proof.resource_profile.gpu_id ??
                            "Unknown"}
                          <small>
                            {proof.resource_profile.hourly_cost_usd
                              ? `$${proof.resource_profile.hourly_cost_usd}/hour`
                              : "Rate unavailable"}
                          </small>
                        </td>
                        <td data-label="Learning signal">
                          {proof.result.initial_reward?.toFixed(3)} →{" "}
                          {proof.result.final_reward?.toFixed(3)}
                          <small>
                            {formatSigned(proof.result.reward_gain)} reward ·{" "}
                            {proof.result.elapsed_seconds}s
                            {proof.workload.total_action_decisions
                              ? ` · ${compactNumber(
                                  proof.workload.total_action_decisions,
                                )} decisions`
                              : ""}
                            {proof.workload.total_sampled_completions
                              ? ` · ${compactNumber(
                                  proof.workload.total_sampled_completions,
                                )} completions`
                              : ""}
                          </small>
                        </td>
                        <td data-label="Curriculum">
                          {proof.result.promotion_count ?? 0} promotions
                          {proof.workload.reached_complexity_level !==
                          undefined ? (
                            <small>
                              Level {proof.workload.reached_complexity_level} of{" "}
                              {proof.workload.maximum_complexity_level}
                              {proof.result.hypothesis_passed === true
                                ? " · learning criterion met"
                                : proof.result.hypothesis_passed === false
                                  ? " · criterion not met"
                                  : ""}
                            </small>
                          ) : null}
                        </td>
                        <td data-label="Cleanup">
                          <StatusBadge
                            status={
                              proof.teardown_confirmed ? "RELEASED" : "WARNING"
                            }
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

function compactNumber(value: number): string {
  return new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatSigned(value: number | undefined): string {
  if (value === undefined) return "Unavailable";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function shortModelName(value: string): string {
  return value.split("/").at(-1) ?? value;
}
