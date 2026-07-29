import { describe, expect, it } from "vitest";
import { revision30Study } from "./revision30";

describe("Revision 30 dashboard evidence", () => {
  it("preserves the frozen report totals and decision result", () => {
    const totalCost = revision30Study.executions.reduce(
      (total, execution) => total + (execution.costUsd ?? 0),
      0,
    );
    const passCount = revision30Study.decisions.filter(
      (decision) => decision.status === "PASS",
    ).length;

    expect(revision30Study.executions).toHaveLength(22);
    expect(totalCost).toBeCloseTo(3.765644, 6);
    expect(passCount).toBe(2);
    expect(revision30Study.decisions).toHaveLength(6);
  });

  it("keeps external improvements and regressions distinct", () => {
    const improved = revision30Study.external.adapters.reduce(
      (total, adapter) => total + adapter.improved,
      0,
    );
    const regressed = revision30Study.external.adapters.reduce(
      (total, adapter) => total + adapter.regressed,
      0,
    );

    expect(revision30Study.external.adapters).toHaveLength(5);
    expect(improved).toBe(2);
    expect(regressed).toBe(1);
    expect(revision30Study.external.taskCount).toBe(9);
  });
});
