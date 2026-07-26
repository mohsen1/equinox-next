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

interface ResourcesResponse {
  allocations: Array<Record<string, any>>;
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
      <PageHeader
        eyebrow="Operational plane"
        title="Resources"
        description="Desired and observed local allocations, snapshots, checkpoints, and provider boundaries."
      />
      <div className="content">
        <AsyncState loading={loading} error={error}>
          {data ? (
            <>
              <Notice title="No external provider capacity">
                This local profile exposes {data.external_capacity} external
                model or RunPod slots. Candidate sessions have no cloud
                credentials.
              </Notice>
              <div className="run-overview">
                <Section title="Live projections">
                  <KeyValue
                    items={[
                      { label: "Allocations", value: data.counts.allocations },
                      { label: "Snapshots", value: data.counts.snapshots },
                      { label: "Checkpoints", value: data.counts.checkpoints },
                      {
                        label: "Pending operations",
                        value: data.counts.pending_operations,
                      },
                    ]}
                  />
                </Section>
                <Section title="Provider registries">
                  <KeyValue
                    items={Object.entries(data.provider_boundaries).map(
                      ([name, values]) => ({
                        label: name.replace("_", " "),
                        value: values.map((value) => (
                          <code key={value}>{value}</code>
                        )),
                      }),
                    )}
                  />
                </Section>
              </div>
              <Section title="Policy-compute allocations">
                <AsyncState empty={!data.allocations.length}>
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Allocation</th>
                          <th>Provider</th>
                          <th>Desired</th>
                          <th>Observed</th>
                          <th>Profile</th>
                          <th>Cleanup</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.allocations.map((allocation) => (
                          <tr key={allocation.allocation_id}>
                            <td>
                              <MachineId value={allocation.allocation_id} />
                            </td>
                            <td>
                              <code>{allocation.provider_name}</code>
                            </td>
                            <td>
                              <StatusBadge status={allocation.desired_state} />
                            </td>
                            <td>
                              <StatusBadge status={allocation.observed_state} />
                            </td>
                            <td>{allocation.resource_profile}</td>
                            <td>
                              {allocation.cleanup_warning ? (
                                <StatusBadge status="WARNING" />
                              ) : (
                                "No warning"
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </AsyncState>
              </Section>
            </>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}
