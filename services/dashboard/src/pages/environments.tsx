import { useApi } from "../api";
import {
  AsyncState,
  KeyValue,
  PageHeader,
  Section,
  StatusBadge,
} from "../components";

interface EnvironmentRecord {
  environment_id: string;
  name: string;
  version: string;
  description: string;
  dataset: {
    source: string;
    revision: string;
    task_families: string[];
  };
  harness: {
    provider: string;
    snapshot_fidelity: string;
    network: string;
  };
  reward_function: {
    pipeline: string;
    deterministic: string[];
    model_assessed: string[];
  };
  readiness: {
    current: string;
    contract_defined: boolean;
    simulator_verified: boolean;
    sandbox_integrated: boolean;
    used_in_training: boolean;
  };
  usage: {
    rollout_trees: number;
    committed_iterations: number;
    verification_runs: number;
  };
}

export function EnvironmentsPage() {
  const { data, error, loading } = useApi<{ items: EnvironmentRecord[] }>(
    "/v1/environments",
    15_000,
  );
  return (
    <>
      <PageHeader
        eyebrow="Research protocol"
        title="Environments"
        description="Dataset, execution harness, reward function, and readiness—kept separate so “defined” never masquerades as “used.”"
      />
      <div className="content">
        <AsyncState
          loading={loading}
          error={error}
          empty={Boolean(data && !data.items.length)}
        >
          {data?.items.map((environment) => (
            <Section
              key={environment.environment_id}
              title={
                <span className="environment-title">
                  {environment.name}
                  <code>{environment.version}</code>
                </span>
              }
              aside={<StatusBadge status={environment.readiness.current} />}
              className="environment-record"
            >
              <p className="environment-description">
                {environment.description}
              </p>
              <ol
                className="readiness-track"
                aria-label="Environment readiness"
              >
                {[
                  ["Contract defined", environment.readiness.contract_defined],
                  [
                    "Simulator verified",
                    environment.readiness.simulator_verified,
                  ],
                  [
                    "Sandbox integrated",
                    environment.readiness.sandbox_integrated,
                  ],
                  ["Used in training", environment.readiness.used_in_training],
                ].map(([label, complete]) => (
                  <li
                    className={complete ? "complete" : ""}
                    key={String(label)}
                  >
                    <span aria-hidden="true">{complete ? "✓" : "○"}</span>
                    {label}
                  </li>
                ))}
              </ol>
              <div className="environment-composition">
                <section>
                  <span className="composition-label">Dataset</span>
                  <h3>Generated CAD tasks</h3>
                  <KeyValue
                    items={[
                      { label: "Source", value: environment.dataset.source },
                      {
                        label: "Revision",
                        value: <code>{environment.dataset.revision}</code>,
                      },
                      {
                        label: "Families",
                        value: environment.dataset.task_families.join(", "),
                      },
                    ]}
                  />
                </section>
                <section>
                  <span className="composition-label">Harness</span>
                  <h3>Isolated local execution</h3>
                  <KeyValue
                    items={[
                      {
                        label: "Provider",
                        value: <code>{environment.harness.provider}</code>,
                      },
                      {
                        label: "Restore fidelity",
                        value: (
                          <code>{environment.harness.snapshot_fidelity}</code>
                        ),
                      },
                      { label: "Network", value: environment.harness.network },
                    ]}
                  />
                </section>
                <section>
                  <span className="composition-label">Reward function</span>
                  <h3>Evidence-derived signals</h3>
                  <KeyValue
                    items={[
                      {
                        label: "Pipeline",
                        value: (
                          <code>{environment.reward_function.pipeline}</code>
                        ),
                      },
                      {
                        label: "Deterministic",
                        value:
                          environment.reward_function.deterministic.join(", "),
                      },
                      {
                        label: "Model-assessed",
                        value:
                          environment.reward_function.model_assessed.join(", "),
                      },
                    ]}
                  />
                </section>
              </div>
              <div className="environment-usage">
                {environment.usage.rollout_trees} rollout trees ·{" "}
                {environment.usage.verification_runs} verifications ·{" "}
                {environment.usage.committed_iterations} committed iterations
              </div>
            </Section>
          ))}
        </AsyncState>
      </div>
    </>
  );
}
