import type { EstimatedComputeCost } from "./types";

export function formatRelativeTime(value: string, now = Date.now()): string {
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return "Unknown";
  const seconds = Math.round((now - timestamp) / 1_000);
  if (Math.abs(seconds) < 45) return "just now";

  const future = seconds < 0;
  const absoluteSeconds = Math.abs(seconds);
  const [amount, unit] =
    absoluteSeconds < 3_600
      ? [Math.max(1, Math.round(absoluteSeconds / 60)), "min"]
      : absoluteSeconds < 86_400
        ? [Math.max(1, Math.round(absoluteSeconds / 3_600)), "hr"]
        : [Math.max(1, Math.round(absoluteSeconds / 86_400)), "day"];
  const label = unit === "day" && amount !== 1 ? "days" : unit;
  return future ? `in ${amount} ${label}` : `${amount} ${label} ago`;
}

export function formatEstimatedCost(
  cost: EstimatedComputeCost | undefined,
): string {
  if (cost?.total_usd === null || cost?.total_usd === undefined) {
    return "Unavailable";
  }
  const value = new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: cost.total_usd < 0.01 ? 3 : 2,
  }).format(cost.total_usd);
  return cost.estimated ? `${value} est.` : value;
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) {
    return "Unavailable";
  }
  if (seconds < 60) return `${Math.round(seconds)} sec`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return remainder ? `${minutes} min ${remainder} sec` : `${minutes} min`;
}

export function formatReward(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(3)
    : "Unavailable";
}

export function formatSignedReward(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value >= 0 ? "+" : ""}${value.toFixed(3)}`
    : "Unavailable";
}
