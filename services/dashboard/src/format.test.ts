import { describe, expect, it } from "vitest";

import { formatEstimatedCost, formatRelativeTime } from "./format";

describe("formatRelativeTime", () => {
  const now = Date.parse("2026-07-27T12:00:00Z");

  it("formats recent, minute, and hour updates", () => {
    expect(formatRelativeTime("2026-07-27T11:59:40Z", now)).toBe("just now");
    expect(formatRelativeTime("2026-07-27T11:56:00Z", now)).toBe("4 min ago");
    expect(formatRelativeTime("2026-07-27T10:00:00Z", now)).toBe("2 hr ago");
  });

  it("handles future and invalid timestamps", () => {
    expect(formatRelativeTime("2026-07-27T12:02:00Z", now)).toBe("in 2 min");
    expect(formatRelativeTime("invalid", now)).toBe("Unknown");
  });
});

describe("formatEstimatedCost", () => {
  it("labels estimated totals and preserves provider-billed totals", () => {
    expect(
      formatEstimatedCost({
        total_usd: 0.34128,
        estimated: true,
        hourly_rate_usd: 0.44,
        elapsed_seconds: 2792.289,
      }),
    ).toMatch(/^\$0\.34 est\.$/);
    expect(
      formatEstimatedCost({
        total_usd: 1.25,
        estimated: false,
        hourly_rate_usd: null,
        elapsed_seconds: null,
      }),
    ).toBe("$1.25");
  });

  it("handles missing cost data", () => {
    expect(formatEstimatedCost(undefined)).toBe("Unavailable");
    expect(
      formatEstimatedCost({
        total_usd: null,
        estimated: false,
        hourly_rate_usd: 0.44,
        elapsed_seconds: null,
      }),
    ).toBe("Unavailable");
  });
});
