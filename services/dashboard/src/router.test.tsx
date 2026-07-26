import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { BrowserRouter, Navigate, Route, Routes } from "./router";

describe("BrowserRouter", () => {
  let container: HTMLDivElement;
  let root: Root | undefined;

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    window.history.replaceState(null, "", "/");
    container = document.createElement("div");
    document.body.append(container);
  });

  afterEach(async () => {
    if (root) await act(async () => root?.unmount());
    container.remove();
  });

  it("renders the redirect target on a fresh root load", async () => {
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Navigate to="/runs" replace />} />
            <Route path="/runs" element={<main>Runs ready</main>} />
          </Routes>
        </BrowserRouter>,
      );
    });

    expect(window.location.pathname).toBe("/runs");
    expect(container.querySelector("main")?.textContent).toBe("Runs ready");
  });

  it("redirects the legacy resources route to Proofs", async () => {
    window.history.replaceState(null, "", "/resources");
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <BrowserRouter>
          <Routes>
            <Route
              path="/resources"
              element={<Navigate to="/proofs" replace />}
            />
            <Route path="/proofs" element={<main>Proofs ready</main>} />
          </Routes>
        </BrowserRouter>,
      );
    });

    expect(window.location.pathname).toBe("/proofs");
    expect(container.querySelector("main")?.textContent).toBe("Proofs ready");
  });
});
