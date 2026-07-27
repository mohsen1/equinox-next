import { useEffect, useState } from "react";

const API_ROOT = "/api";
const LOAD_TIMEOUT_MS = 8_000;

export type Decoder<T> = (value: unknown) => T;

export async function api<T>(
  path: string,
  init?: RequestInit,
  decode?: Decoder<T>,
): Promise<T> {
  let response: Response;
  try {
    const headers = new Headers(init?.headers);
    if (init?.body !== undefined && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    response = await fetch(`${API_ROOT}${path}`, {
      ...init,
      headers,
    });
  } catch (cause) {
    if (cause instanceof Error && cause.name === "AbortError") throw cause;
    throw new Error("The API is temporarily unavailable.");
  }
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    throw new Error(errorMessage(body, response.status));
  }
  const body: unknown = await response.json();
  return decode ? decode(body) : (body as T);
}

export function artifactUrl(artifactId: string): string {
  return `${API_ROOT}/v1/artifacts/${artifactId}`;
}

export function useApi<T>(
  path: string | null,
  pollIntervalMs = 0,
  decode?: Decoder<T>,
) {
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
        const next = await api<T>(
          path!,
          { signal: requestController.signal },
          decode,
        );
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
  }, [path, pollIntervalMs, retryToken, decode]);

  return {
    data,
    error,
    loading,
    refreshing,
    retry: () => setRetryToken((value) => value + 1),
    setData,
  };
}

function errorMessage(body: unknown, status: number): string {
  if (typeof body === "string" && body.trim()) return body;
  if (!body || typeof body !== "object") return `Request failed: ${status}`;
  const detail = "detail" in body ? body.detail : body;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item
          ? String(item.msg)
          : null,
      )
      .filter((item): item is string => Boolean(item));
    if (messages.length) return messages.join("; ");
  }
  if (detail && typeof detail === "object") {
    if ("message" in detail && typeof detail.message === "string") {
      return detail.message;
    }
    if ("code" in detail && typeof detail.code === "string") {
      return detail.code;
    }
  }
  return `Request failed: ${status}`;
}
