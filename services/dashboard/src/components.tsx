import { useState, type PropsWithChildren, type ReactNode } from "react";
import { artifactUrl } from "./api";
import { Link, NavLink, useLocation } from "./router";
import type { ArtifactRef } from "./types";

const NAV_ITEMS = [
  { to: "/runs", label: "Runs" },
  { to: "/environments", label: "Environments" },
  { to: "/resources", label: "Proofs" },
];

export function AppShell({ children }: PropsWithChildren) {
  const location = useLocation();
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside className="rail" aria-label="Primary navigation">
        <NavLink className="brand" to="/runs" aria-label="Equinox Next runs">
          <span className="brand-mark" aria-hidden="true">
            EQ
          </span>
          <span>Equinox</span>
        </NavLink>
        <nav>
          {NAV_ITEMS.map((item) => {
            const isActive =
              item.to === "/runs"
                ? location.pathname.startsWith("/runs") ||
                  location.pathname.startsWith("/rollout-trees") ||
                  location.pathname.startsWith("/verification-runs")
                : location.pathname.startsWith(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                aria-current={isActive ? "page" : undefined}
                className={isActive ? "nav-item active" : "nav-item"}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </aside>
      <main id="main" className="main" data-route={location.pathname}>
        {children}
      </main>
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow ? <div className="route-context">{eyebrow}</div> : null}
        <h1>{title}</h1>
        {description ? <p>{description}</p> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const tone = statusTone(status);
  return <span className={`status ${tone}`}>{friendlyStatus(status)}</span>;
}

function statusTone(status: string): string {
  if (
    [
      "SUCCEEDED",
      "ACCEPTED",
      "ADMITTED",
      "READY",
      "CONTINUED",
      "VERIFIED",
    ].includes(status)
  )
    return "positive";
  if (
    ["FAILED", "INTEGRITY_VIOLATION", "EXCLUDED", "CANDIDATE_FAILED"].includes(
      status,
    )
  )
    return "negative";
  if (
    [
      "ABSTAINED",
      "DISAGREEMENT",
      "RETRYING",
      "INFRA_FAILED",
      "CANCEL_REQUESTED",
    ].includes(status)
  )
    return "warning";
  return "neutral";
}

export function friendlyStatus(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function displayRunName(name: string): string {
  const reproductionPrefix = "Reproduction of ";
  if (!name.startsWith(reproductionPrefix)) return name;
  return `${reproductionPrefix}${name.replace(
    new RegExp(`^(?:${reproductionPrefix})+`),
    "",
  )}`;
}

export function MachineId({
  value,
  copy = true,
}: {
  value: string;
  copy?: boolean;
}) {
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">(
    "idle",
  );

  async function copyValue() {
    try {
      await navigator.clipboard.writeText(value);
      setCopyState("copied");
    } catch {
      setCopyState("failed");
    }
  }

  return (
    <span className="machine-id" title={value}>
      <code>{shortId(value)}</code>
      {copy ? (
        <button
          className="copy-button"
          type="button"
          onClick={copyValue}
          aria-label={`Copy ${value}`}
        >
          <span aria-live="polite">
            {copyState === "copied"
              ? "Copied"
              : copyState === "failed"
                ? "Copy failed"
                : "Copy"}
          </span>
        </button>
      ) : null}
    </span>
  );
}

export function shortId(value: string): string {
  if (value.length <= 20) return value;
  return `${value.slice(0, 12)}…${value.slice(-6)}`;
}

export function EvidenceRender({
  artifact,
  label,
  selected = false,
  status,
}: {
  artifact: ArtifactRef;
  label: string;
  selected?: boolean;
  status: string;
}) {
  const visible = artifact.visibility !== "HIDDEN";
  return (
    <figure className={selected ? "render-pane selected" : "render-pane"}>
      <figcaption>
        <span>{label}</span>
        <StatusBadge status={status} />
      </figcaption>
      {visible ? (
        <img
          src={artifactUrl(artifact.artifact_id)}
          alt={`${label} canonical CAD render`}
        />
      ) : (
        <div
          className="render-placeholder"
          role="img"
          aria-label={`${label} evidence is ${artifact.visibility.toLowerCase()}`}
        >
          <span className="visibility-mark" aria-hidden="true">
            ⊘
          </span>
          <strong>Evidence hidden</strong>
          <span>{artifact.role.replaceAll("-", " ")}</span>
          <code>{artifact.visibility}</code>
        </div>
      )}
      <MachineId value={artifact.digest} copy={false} />
    </figure>
  );
}

export function Section({
  title,
  aside,
  children,
  className = "",
}: PropsWithChildren<{
  title: ReactNode;
  aside?: ReactNode;
  className?: string;
}>) {
  return (
    <section className={`section ${className}`}>
      <div className="section-heading">
        <h2>{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function AsyncState({
  loading = false,
  error = null,
  empty,
  children,
}: PropsWithChildren<{
  loading?: boolean;
  error?: Error | null;
  empty?: boolean;
}>) {
  if (loading) {
    return (
      <div className="state-panel loading-state" role="status">
        <span className="loading-line" />
        <span className="loading-line short" />
        <p>Reading persisted evidence…</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="state-panel error-state" role="alert">
        <strong>Evidence could not be loaded</strong>
        <p>{error.message}</p>
        <button
          type="button"
          className="button secondary"
          onClick={() => location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }
  if (empty) {
    return (
      <div className="state-panel empty-state">
        <strong>No persisted records yet</strong>
        <p>Launch or seed a local CAD run to create inspectable lineage.</p>
        <NavLink className="button primary" to="/runs/new">
          Launch run
        </NavLink>
      </div>
    );
  }
  return children;
}

export function KeyValue({
  items,
}: {
  items: Array<{ label: string; value: ReactNode; span?: boolean }>;
}) {
  return (
    <dl className="key-value">
      {items.map((item) => (
        <div key={item.label} className={item.span ? "span" : ""}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Notice({
  tone = "neutral",
  title,
  children,
}: PropsWithChildren<{
  tone?: "neutral" | "warning" | "negative";
  title: string;
}>) {
  return (
    <div
      className={`notice ${tone}`}
      role={tone === "negative" ? "alert" : undefined}
    >
      <strong>{title}</strong>
      <div>{children}</div>
    </div>
  );
}

export function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}
