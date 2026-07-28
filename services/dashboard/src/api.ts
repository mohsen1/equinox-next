import { useCallback, useEffect, useRef, useState } from "react";

const API_ROOT = "/api";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
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
  const [refreshVersion, setRefreshVersion] = useState(0);
  const hasData = useRef(false);
  const refresh = useCallback(
    () => setRefreshVersion((value) => value + 1),
    [],
  );

  useEffect(() => {
    if (!path) {
      setLoading(false);
      return;
    }
    let disposed = false;
    let timeout: number | null = null;
    let controller: AbortController | null = null;
    async function load(initial: boolean) {
      controller = new AbortController();
      if (initial && !hasData.current) setLoading(true);
      setError(null);
      try {
        const next = await api<T>(path!, { signal: controller.signal });
        if (!disposed) {
          hasData.current = true;
          setData(next);
        }
      } catch (cause) {
        if (!disposed && cause instanceof Error && cause.name !== "AbortError")
          setError(cause);
      } finally {
        if (!disposed && initial) setLoading(false);
      }
    }
    function schedule() {
      if (pollIntervalMs <= 0 || disposed) return;
      timeout = window.setTimeout(async () => {
        if (!document.hidden) await load(false);
        schedule();
      }, pollIntervalMs);
    }
    void load(true).then(schedule);
    return () => {
      disposed = true;
      controller?.abort();
      if (timeout !== null) window.clearTimeout(timeout);
    };
  }, [path, pollIntervalMs, refreshVersion]);

  return { data, error, loading, setData, refresh };
}
