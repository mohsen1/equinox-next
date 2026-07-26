import { useApi } from "../api";
import { AsyncState, PageHeader, StatusBadge } from "../components";
import { useSearchParams } from "../router";
import type { EnvironmentsResponse, EnvironmentSpec } from "../types";

export function EnvironmentsPage() {
  const catalog = useApi<EnvironmentsResponse>("/v1/environments");
  const [searchParams, setSearchParams] = useSearchParams();
  const environments =
    catalog.data?.items.filter(
      (item) => item.environment_id !== "cad.reconstruction",
    ) ?? [];
  const selectedId =
    searchParams.get("environment") ?? environments[0]?.environment_id ?? "";
  const selected = environments.find(
    (item) => item.environment_id === selectedId,
  );

  function select(item: EnvironmentSpec) {
    const next = new URLSearchParams(searchParams);
    next.set("environment", item.environment_id);
    setSearchParams(next);
  }

  return (
    <>
      <PageHeader title="Environments" />
      <div className="content workspace-content">
        <AsyncState loading={catalog.loading} error={catalog.error}>
          {catalog.data ? (
            <div className="environment-workspace">
              <section className="environment-index" aria-label="Environments">
                <div>
                  {environments.map((item) => (
                    <button
                      key={item.environment_id}
                      type="button"
                      aria-pressed={item.environment_id === selectedId}
                      onClick={() => select(item)}
                    >
                      <span>
                        <strong>{item.name}</strong>
                        <small>{item.action_space}</small>
                      </span>
                      <StatusBadge status="VERIFIED" />
                    </button>
                  ))}
                </div>
              </section>
              {selected ? (
                <EnvironmentDetail
                  environment={selected}
                  branchWidth={catalog.data.branching.branch_width}
                />
              ) : null}
            </div>
          ) : null}
        </AsyncState>
      </div>
    </>
  );
}

function EnvironmentDetail({
  environment,
  branchWidth,
}: {
  environment: EnvironmentSpec;
  branchWidth: number;
}) {
  return (
    <article className="environment-detail">
      <h2>{environment.name}</h2>
      <p className="environment-lede">{environment.summary}</p>
      <dl className="environment-contract">
        <div>
          <dt>Agent actions</dt>
          <dd>{environment.action_space}</dd>
        </div>
        <div>
          <dt>Trusted verifier</dt>
          <dd>{environment.verifier}</dd>
        </div>
        <div>
          <dt>Snapshot</dt>
          <dd>{environment.snapshot_strategy}</dd>
        </div>
        <div>
          <dt>Branching</dt>
          <dd>Static K={branchWidth} at valid action boundaries</dd>
        </div>
      </dl>
      <section className="difficulty-contract">
        <div>
          <h3>Dynamic complexity</h3>
          <p>
            Starts at level {environment.complexity.default_initial_level} and
            expands toward level {environment.complexity.default_max_level} as
            the policy demonstrates mastery.
          </p>
        </div>
        <div className="dimension-list" aria-label="Complexity dimensions">
          {environment.complexity.dimensions.map((dimension) => (
            <span key={dimension}>{dimension.replaceAll("_", " ")}</span>
          ))}
        </div>
      </section>
    </article>
  );
}
