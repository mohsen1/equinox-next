import { useState, type PropsWithChildren, type ReactNode } from "react";
import { artifactUrl } from "./api";
import { Link, NavLink, useLocation } from "./router";
import type { ArtifactRef } from "./types";

const NAV_ITEMS = [
  { to: "/runs", label: "Runs", glyph: "⌁" },
  { to: "/environments", label: "Environments", glyph: "◇" },
  { to: "/proofs", label: "Proofs", glyph: "⌗" },
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
            E
          </span>
          <span>
            Equinox <strong>Next</strong>
          </span>
        </NavLink>
        <nav>
          {NAV_ITEMS.map((item) => {
            const isActive =
              item.to === "/runs"
                ? location.pathname === "/runs" ||
                  location.pathname.startsWith("/studies/") ||
                  (location.pathname.startsWith("/runs/") &&
                    location.pathname !== "/runs/new")
                : location.pathname === item.to ||
                  location.pathname.startsWith(`${item.to}/`);
            return (
              <Link
                key={item.to}
                to={item.to}
                aria-current={isActive ? "page" : undefined}
                className={isActive ? "nav-item active" : "nav-item"}
              >
                <span aria-hidden="true">{item.glyph}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="rail-note">
          <span className="provider-dot" />
          Local contract proof
          <small>Mocks only · no cloud credentials</small>
        </div>
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
      "MATCHED",
      "USED_IN_TRAINING",
      "MATERIALIZED",
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
      "INCONCLUSIVE",
      "NO_UPDATE",
      "INCOMPLETE",
      "CONTRACT_ONLY",
    ].includes(status)
  )
    return "warning";
  return "neutral";
}

export function ExecutionBadge({ status }: { status: string }) {
  const label: Record<string, string> = {
    SUCCEEDED: "Completed",
    FAILED: "Failed",
    CANCELED: "Canceled",
    CANCEL_REQUESTED: "Cancel requested",
  };
  return (
    <span className={`status ${statusTone(status)}`}>
      {label[status] ?? friendlyStatus(status)}
    </span>
  );
}

export function LearningOutcomeBadge({ outcome }: { outcome: string }) {
  const label: Record<string, string> = {
    INCONCLUSIVE: "Inconclusive",
    NOT_EVALUATED: "Not evaluated",
    NO_UPDATE: "No policy update",
    IMPROVED: "Improved",
    REGRESSED_ROLLED_BACK: "Regressed · rolled back",
  };
  return (
    <span className={`status ${statusTone(outcome)}`}>
      {label[outcome] ?? friendlyStatus(outcome)}
    </span>
  );
}

export function EvidenceStrengthBadge({ strength }: { strength: string }) {
  const label: Record<string, string> = {
    CONTRACT_ONLY: "Contract evidence only",
    EXPLORATORY_SINGLE_SEED: "Exploratory · single seed",
    MATCHED_CONTROL: "Matched control",
    REPLICATED: "Replicated",
    EXTERNALLY_VALIDATED: "Externally validated",
  };
  return (
    <span className={`status ${statusTone(strength)}`}>
      {label[strength] ?? friendlyStatus(strength)}
    </span>
  );
}

export function friendlyStatus(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
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
  id,
}: PropsWithChildren<{
  title: ReactNode;
  aside?: ReactNode;
  className?: string;
  id?: string;
}>) {
  return (
    <section className={`section ${className}`} id={id}>
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
