# Runs and Proofs workspace

The dashboard keeps run monitoring operational during a transient API outage and moves
completed training evidence into dedicated Proofs routes.

## Service readiness

The dashboard exposes two distinct checks:

- `GET /healthz` confirms that nginx can serve the application.
- `GET /readyz` proxies the orchestrator health check and succeeds only when the API and
  its database dependency are ready.

The nginx proxy resolves the orchestrator through Docker DNS on each recovery window,
uses a two-second upstream connection timeout, and retries only connection, timeout, and
gateway failures. Recreating the orchestrator container does not require recreating the
dashboard.

Dashboard polling permits one request at a time and gives each read eight seconds to
finish. When a refresh fails after data has loaded, the route keeps the last successful
response visible, shows `Live updates paused`, and offers an in-place retry. The next
successful poll clears the warning. An initial failure shows a recoverable error state
instead of an empty page.

## Runs

`GET /v1/runs` includes research execution summaries with:

- `allocated_gpu`, which is present only after a provider handle has been recorded;
- `cost.total_usd`;
- `cost.estimated`;
- `cost.hourly_rate_usd`; and
- `cost.elapsed_seconds`.

The estimated total uses the recorded values:

```text
total_usd = hourly_rate_usd × elapsed_seconds ÷ 3600
```

The total remains unavailable when either input is missing. It is labeled estimated until
a provider-billed amount is ingested. A requested GPU without a provider handle is shown
as `Awaiting allocation`.

The Runs queue shows run name, model and configuration, status, progress, allocated GPU,
estimated total cost, and relative update time. The full localized timestamp remains on
the `time` element for assistive technology and pointer inspection.

## Proofs

User-facing routes:

- `/proofs` lists completed training proofs.
- `/proofs/:proof_id` shows the complete evidence record.
- `/resources` redirects to `/proofs`.

API routes:

- `GET /v1/proofs` returns proof identity, originating run, completion time, learning
  result, hardware, estimated cost, and teardown status.
- `GET /v1/proofs/{proof_id}` adds workload and model revisions, provider image and handle,
  runtime, curriculum progression, receipt digest, and teardown confirmation.

When an execution record exists, proof detail links to the run overview and trajectory.
The proof list does not expose provider handles or receipt digests; those remain in the
detail contract.

## Theme and accessibility

The dashboard follows `prefers-color-scheme` exclusively. It does not write a theme
attribute or expose a manual theme control. Navigation, tables, graph controls, notices,
statuses, and work surfaces use semantic tokens in both modes.

The checked text/background pairs meet WCAG 2.2 AA:

- light mode minimum: `4.62:1`;
- dark mode minimum: `5.51:1`.

Interactive controls use a two-pixel visible focus outline with a two-pixel offset.

## Failure check

To verify recovery locally:

```bash
docker compose stop orchestrator
curl -i http://127.0.0.1:3100/healthz
curl -i http://127.0.0.1:3100/readyz
curl -i http://127.0.0.1:3100/api/v1/runs
docker compose start orchestrator
```

Expected behavior:

- liveness remains `200`;
- readiness fails while the orchestrator is stopped;
- the API returns a gateway failure within two seconds instead of hanging;
- Runs and loaded proof detail remain usable with a paused-update notice; and
- polling recovers without a page reload after readiness returns.
