import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TrajectoryOutline } from "./pages/rollout";
import "./styles.css";
import type { RolloutGraph } from "./types";

const graph: RolloutGraph = {
  tree: {},
  nodes: [
    {
      id: "state_root",
      type: "state",
      sequence: 0,
      semantic_status: "READY",
      payload: { state_id: "state_root" },
    },
    {
      id: "state_child",
      type: "state",
      sequence: 1,
      semantic_status: "CONTINUED",
      payload: { state_id: "state_child" },
    },
  ],
  edges: [
    {
      id: "transition_1",
      source: "state_root",
      target: "state_child",
      branch_member_id: "member_1",
      outcome: "CONTINUED",
      action: { kind: "create_base" },
      verification_run_id: "verification_1",
      proof_bundle_id: "proof_1",
    },
  ],
  branch_groups: [],
  branch_members: [
    {
      branch_member_id: "member_1",
      branch_group_id: "group_1",
      sibling_index: 0,
      status: "SUCCEEDED",
      failure_mode: null,
    },
  ],
  decision_checkpoints: [],
  environment_snapshots: [],
  accessible_outline: [],
};

describe("rollout accessibility contracts", () => {
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

  it("provides a keyboard-operable outline equivalent", async () => {
    const select = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root?.render(
        <TrajectoryOutline
          graph={graph}
          selectedState={null}
          select={select}
        />,
      );
    });

    expect(container.querySelector("caption")?.textContent).toContain(
      "Keyboard-navigable equivalent",
    );
    const buttons = container.querySelectorAll("button");
    expect(buttons).toHaveLength(2);
    buttons[1].focus();
    expect(document.activeElement).toBe(buttons[1]);
    buttons[1].click();
    expect(select).toHaveBeenCalledWith("state_child", graph.edges[0]);
  });

  it("defines native dark mode and visible keyboard focus", () => {
    const styles = Array.from(document.styleSheets)
      .flatMap((sheet) => Array.from(sheet.cssRules))
      .map((rule) => rule.cssText)
      .join("\n");
    expect(styles).toContain("color-scheme: light dark");
    expect(styles).toContain("prefers-color-scheme: dark");
    expect(styles).toContain("button:focus-visible");
    expect(styles).toContain("outline: 2px solid var(--focus)");
  });
});
