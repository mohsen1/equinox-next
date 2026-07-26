import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { EvidenceRender } from "./components";
import type { ArtifactRef } from "./types";

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
