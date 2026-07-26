import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { AppShell, displayRunName, EvidenceRender } from "./components";
import type { ArtifactRef } from "./types";

describe("displayRunName", () => {
  it("collapses repeated reproduction prefixes", () => {
    expect(
      displayRunName(
        "Reproduction of Reproduction of Active restart contract proof",
      ),
    ).toBe("Reproduction of Active restart contract proof");
    expect(displayRunName("Active restart contract proof")).toBe(
      "Active restart contract proof",
    );
  });
});

describe("AppShell", () => {
  let container: HTMLDivElement;
  let root: Root | undefined;

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    container = document.createElement("div");
    document.body.append(container);
  });

  afterEach(async () => {
    if (root) await act(async () => root?.unmount());
    container.remove();
  });

  it("shows product navigation without prototype workspace status", async () => {
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <AppShell>
          <p>Research runs</p>
        </AppShell>,
      );
    });

    expect(container.textContent).toContain("Runs");
    expect(container.textContent).toContain("Environments");
    expect(container.textContent).toContain("Proofs");
    expect(container.textContent).not.toContain("Local workspace");
    expect(container.textContent).not.toContain(
      "External compute disconnected",
    );
  });
});

describe("EvidenceRender", () => {
  let container: HTMLDivElement;
  let root: Root | undefined;

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    container = document.createElement("div");
    document.body.append(container);
  });

  afterEach(async () => {
    if (root) await act(async () => root?.unmount());
    container.remove();
  });

  it("does not request hidden evidence", async () => {
    const artifact: ArtifactRef = {
      artifact_id: "artifact_hidden",
      digest: "sha256:hidden",
      role: "candidate-render",
      media_type: "image/svg+xml",
      visibility: "HIDDEN",
      trust_class: "UNTRUSTED",
    };
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <EvidenceRender
          artifact={artifact}
          label="Candidate"
          status="SELECTED"
        />,
      );
    });

    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector('[role="img"]')?.textContent).toContain(
      "Evidence hidden",
    );
    expect(container.textContent).toContain("candidate render");
    expect(container.textContent).toContain("sha256:hidden");
  });
});
