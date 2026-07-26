import { useEffect, useState } from "react";

const API_ROOT = "/api";
const LOAD_TIMEOUT_MS = 8_000;

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_ROOT}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
  } catch (cause) {
    if (cause instanceof Error && cause.name === "AbortError") throw cause;
    throw new Error("The API is temporarily unavailable.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail?.message ??
        body?.detail?.code ??
        `Request failed: ${response.status}`,
    );
  }
  return response.json() as Promise<T>;
}

export function artifactUrl(artifactId: string): string {
  return `${API_ROOT}/v1/artifacts/${artifactId}`;
}

export function useApi<T>(path: string | null, pollIntervalMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(Boolean(path));
  const [refreshing, setRefreshing] = useState(false);
  const [retryToken, setRetryToken] = useState(0);

  useEffect(() => {
    if (!path) {
      setLoading(false);
      return;
    }
    const lifecycleController = new AbortController();
    let requestInFlight = false;

    async function load(initial: boolean) {
      if (requestInFlight) return;
      requestInFlight = true;
      if (initial) setLoading(true);
      else setRefreshing(true);

      const requestController = new AbortController();
      const timeout = window.setTimeout(
        () => requestController.abort("timeout"),
        LOAD_TIMEOUT_MS,
      );
      const abortRequest = () => requestController.abort("unmounted");
      lifecycleController.signal.addEventListener("abort", abortRequest, {
        once: true,
      });

      try {
        const next = await api<T>(path!, { signal: requestController.signal });
        setData(next);
        setError(null);
      } catch (cause) {
        if (!lifecycleController.signal.aborted) {
          const timedOut =
            cause instanceof Error &&
            cause.name === "AbortError" &&
            requestController.signal.reason === "timeout";
          setError(
            timedOut
              ? new Error("The API did not respond. Live updates are paused.")
              : cause instanceof Error
                ? cause
                : new Error("The API is temporarily unavailable."),
          );
        }
      } finally {
        window.clearTimeout(timeout);
        lifecycleController.signal.removeEventListener("abort", abortRequest);
        requestInFlight = false;
        if (initial) setLoading(false);
        else setRefreshing(false);
      }
    }

    void load(true);
    const interval =
      pollIntervalMs > 0
        ? window.setInterval(() => void load(false), pollIntervalMs)
        : null;
    return () => {
      lifecycleController.abort();
      if (interval !== null) window.clearInterval(interval);
    };
  }, [path, pollIntervalMs, retryToken]);

  return {
    data,
    error,
    loading,
    refreshing,
    retry: () => setRetryToken((value) => value + 1),
    setData,
  };
}
