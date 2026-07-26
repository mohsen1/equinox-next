import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useApi } from "./api";

function LiveValue() {
  const { data } = useApi<{ sequence: number }>("/v1/live", 1_000);
  return <output>{data?.sequence ?? "waiting"}</output>;
}

function RecoverableValue() {
  const { data, error, retry } = useApi<{ sequence: number }>(
    "/v1/live",
    1_000,
  );
  return (
    <>
      <output>{data?.sequence ?? "waiting"}</output>
      <span>{error?.message ?? "connected"}</span>
      <button type="button" onClick={retry}>
        Try again
      </button>
    </>
  );
}

describe("useApi polling", () => {
  let container: HTMLDivElement;
  let root: Root | undefined;

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    vi.useFakeTimers();
    container = document.createElement("div");
    document.body.append(container);
  });

  afterEach(async () => {
    if (root) await act(async () => root?.unmount());
    container.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("refreshes active data without a page reload", async () => {
    let sequence = 0;
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ sequence: ++sequence }),
    }));
    vi.stubGlobal("fetch", fetchMock);
    root = createRoot(container);

    await act(async () => {
      root?.render(<LiveValue />);
      await Promise.resolve();
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(container.textContent).toBe("1");

    await act(async () => {
      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(container.textContent).toBe("2");
  });

  it("keeps the last response usable and recovers after a transient failure", async () => {
    let request = 0;
    const fetchMock = vi.fn(async () => {
      request += 1;
      if (request === 2) throw new TypeError("connection refused");
      return {
        ok: true,
        json: async () => ({ sequence: request }),
      };
    });
    vi.stubGlobal("fetch", fetchMock);
    root = createRoot(container);

    await act(async () => {
      root?.render(<RecoverableValue />);
      await Promise.resolve();
    });
    expect(container.querySelector("output")?.textContent).toBe("1");
    expect(container.textContent).toContain("connected");

    await act(async () => {
      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
    });
    expect(container.querySelector("output")?.textContent).toBe("1");
    expect(container.textContent).toContain("temporarily unavailable");

    await act(async () => {
      container.querySelector("button")?.click();
      await Promise.resolve();
    });
    expect(container.querySelector("output")?.textContent).toBe("3");
    expect(container.textContent).toContain("connected");
  });
});
