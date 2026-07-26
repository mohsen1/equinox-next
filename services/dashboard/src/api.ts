import { useEffect, useState } from "react";

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

  useEffect(() => {
    if (!path) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    async function load(initial: boolean) {
      if (initial) setLoading(true);
      setError(null);
      try {
        setData(await api<T>(path!, { signal: controller.signal }));
      } catch (cause) {
        if (cause instanceof Error && cause.name !== "AbortError")
          setError(cause);
      } finally {
        if (initial) setLoading(false);
      }
    }
    void load(true);
    const interval =
      pollIntervalMs > 0
        ? window.setInterval(() => void load(false), pollIntervalMs)
        : null;
    return () => {
      controller.abort();
      if (interval !== null) window.clearInterval(interval);
    };
  }, [path, pollIntervalMs]);

  return { data, error, loading, setData };
}
