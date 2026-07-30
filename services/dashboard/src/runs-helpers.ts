import { friendlyStatus } from "./components";
import type { ResearchComplexity, ResearchComputeExecution } from "./types";

export function postTrainingOutcomeLabel(
  postTrainingOutcome: unknown,
  meaningfulPostTraining: unknown,
  legacyClaim: string | null = null,
): string {
  const outcome = stringValue(postTrainingOutcome);
  const meaningful = booleanValue(meaningfulPostTraining);
  if (meaningful === true) return "Meaningful";
  switch (outcome) {
    case "NEGATIVE_EXPERIMENT_COMPLETED":
      return "Negative result";
    case "INCONCLUSIVE_EXPERIMENT_COMPLETED":
      return "Inconclusive";
  }
  if (meaningful === false) return "Not meaningful";
  if (outcome === "MEANINGFUL_POST_TRAINING") return "Meaningful";
  return legacyClaim ? friendlyStatus(legacyClaim) : "Not recorded";
}

export function dynamicComplexityProgressLabel(value: unknown): string {
  const progressed = booleanValue(value);
  if (progressed === null) return "Not recorded";
  return progressed ? "Progressed" : "No progression";
}

export function resultItems(
  run: ResearchComputeExecution,
  claimStrength: string | null,
) {
  const progress = run.progress;
  const initial = numberValue(progress.initial_exact_rate);
  const final = numberValue(progress.final_exact_rate);
  const gain = numberValue(progress.reward_gain);
  const promotions = numberValue(progress.promotion_count);
  const adapterPersisted = booleanValue(progress.adapter_persisted);
  const pairedTest = recordValue(progress.paired_test_change);
  const pairedImproved = numberValue(pairedTest?.improved);
  const pairedRegressed = numberValue(pairedTest?.regressed);
  const pairedExamples = numberValue(pairedTest?.examples);
  const pairedPValue = numberValue(pairedTest?.mcnemar_exact_p_value);
  const resumedFromCheckpoint = booleanValue(progress.resumed_from_checkpoint);
  const attemptCount = numberValue(progress.attempt_count);
  const reserveCeilingExceeded = booleanValue(
    progress.final_evaluation_reserve_exceeded_ceiling,
  );
  const finalEvaluationPartial = booleanValue(
    progress.final_evaluation_partial,
  );
  const meaningfulPostTraining = booleanValue(
    progress.meaningful_post_training,
  );
  const postTrainingOutcome =
    typeof progress.post_training_outcome === "string"
      ? progress.post_training_outcome
      : null;
  const dynamicComplexityProgressed = booleanValue(
    progress.dynamic_complexity_progressed,
  );
  return [
    {
      label: "Post-training",
      value: postTrainingOutcomeLabel(
        postTrainingOutcome,
        meaningfulPostTraining,
        claimStrength,
      ),
    },
    ...(dynamicComplexityProgressed === null
      ? []
      : [
          {
            label: "Complexity",
            value: dynamicComplexityProgressLabel(dynamicComplexityProgressed),
          },
        ]),
    ...(resumedFromCheckpoint
      ? [
          {
            label: "Attempt",
            value: `Resumed from checkpoint${attemptCount === null ? "" : ` · ${attemptCount}`}`,
          },
          {
            label: "Crash tail",
            value: "Wall-clock cost retained · action count unavailable",
          },
        ]
      : []),
    ...(reserveCeilingExceeded
      ? [
          {
            label: "Evaluation reserve",
            value: "Estimate exceeded ceiling · training stopped",
          },
        ]
      : []),
    ...(finalEvaluationPartial
      ? [
          {
            label: "Evaluation",
            value: "Incomplete · workload deadline reached",
          },
        ]
      : []),
    {
      label: "Baseline test",
      value: `${percent(initial, "Not reported")}${finalEvaluationPartial && initial !== null ? " · partial" : ""}`,
    },
    {
      label: "Final test",
      value: `${percent(final, "Not reported")}${finalEvaluationPartial && final !== null ? " · partial" : ""}`,
    },
    {
      label: "Test gain",
      value:
        gain === null
          ? "Not reported"
          : `${gain >= 0 ? "+" : ""}${(gain * 100).toFixed(1)} pts`,
    },
    ...(finalEvaluationPartial ||
    pairedImproved === null ||
    pairedRegressed === null ||
    pairedExamples === null ||
    pairedExamples <= 0
      ? []
      : [
          {
            label: "Test pairs",
            value: `${pairedImproved} improved · ${pairedRegressed} regressed · n=${pairedExamples}${
              pairedPValue === null
                ? ""
                : pairedPValue < 0.0001
                  ? " · p < 0.0001"
                  : ` · p=${pairedPValue.toFixed(4)}`
            }`,
          },
        ]),
    { label: "Promotions", value: promotions?.toLocaleString() ?? "0" },
    {
      label: "Stop reason",
      value: friendlyStatus(
        stringValue(progress.stop_reason) ?? "not reported",
      ),
    },
    {
      label: "Adapter",
      value:
        adapterPersisted === null
          ? "Not reported"
          : adapterPersisted
            ? "Verified"
            : "Missing",
    },
  ];
}

export function observerStages(
  status: ResearchComputeExecution["status"],
  phase: string | null,
  teardownConfirmed: boolean,
  attempt: number | null,
  hasLearningProgress = false,
): Array<{ label: string; detail: string; state: string }> {
  const learning = [
    "training",
    "resuming",
    "evaluation",
    "baseline_evaluation",
  ];
  const current =
    status === "SUCCEEDED"
      ? 4
      : status === "FINALIZING"
        ? 3
        : phase === "finalizing" || phase === "complete"
          ? 3
          : (attempt !== null && attempt > 1) ||
              hasLearningProgress ||
              (phase && learning.includes(phase))
            ? 2
            : phase === "failed" ||
                phase === "container_starting" ||
                phase === "dependency_setup" ||
                phase === "model_loading"
              ? 1
              : 0;
  const labels = [
    ["Allocate", "Provider capacity"],
    ["Prepare", "Boot · dependencies · model"],
    ["Learn", "Sample · verify · update"],
    ["Finalize", "Persist result"],
    ["Release", "Confirm teardown"],
  ];
  return labels.map(([label, detail], index) => ({
    label,
    detail,
    state:
      status === "SUCCEEDED"
        ? "complete"
        : status === "FAILED"
          ? index === 4 && teardownConfirmed
            ? "complete"
            : index === current
              ? "failed"
              : index < current
                ? "complete"
                : "pending"
          : index < current
            ? "complete"
            : index === current
              ? "current"
              : "pending",
  }));
}

export function phasePercentage(
  status: ResearchComputeExecution["status"],
  phase: string | null,
  attempt: number | null,
): number {
  let percentage = 18;
  if (status === "FINALIZING") percentage = 92;
  else if (status === "PROVISIONING")
    percentage = phase === "container_starting" ? 12 : 5;
  else if (phase === "dependency_setup") percentage = 20;
  else if (phase === "model_loading") percentage = 28;
  else if (phase === "baseline_evaluation") percentage = 36;
  else if (phase === "resuming") percentage = 48;
  else if (phase === "training") percentage = 50;
  else if (phase === "evaluation") percentage = 70;
  else if (phase === "finalizing") percentage = 85;
  else if (phase === "complete") percentage = 92;
  return attempt !== null && attempt > 1
    ? Math.max(48, percentage)
    : percentage;
}

export function runPercentage(
  status: ResearchComputeExecution["status"],
  update: number | null,
  maximumUpdates: number | null,
  phase: string | null,
  attempt: number | null,
): number {
  if (status === "SUCCEEDED") return 100;
  if (status === "FAILED") {
    return update !== null && maximumUpdates !== null && maximumUpdates > 0
      ? Math.min(100, (update / maximumUpdates) * 100)
      : 0;
  }
  if (phase === "finalizing") {
    const phaseProgress = phasePercentage(status, phase, attempt);
    return update !== null && maximumUpdates !== null && maximumUpdates > 0
      ? Math.max(phaseProgress, (update / maximumUpdates) * 100)
      : phaseProgress;
  }
  if (update === 0) return phasePercentage(status, phase, attempt);
  return update !== null && maximumUpdates !== null && maximumUpdates > 0
    ? Math.min(100, (update / maximumUpdates) * 100)
    : phasePercentage(status, phase, attempt);
}

export function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function complexityForLevel(
  currentLevel: number | null,
  activeComplexity: ResearchComplexity | undefined,
): ResearchComplexity | null {
  return currentLevel !== null && activeComplexity?.level === currentLevel
    ? activeComplexity
    : null;
}

export function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length ? value : null;
}

export function booleanValue(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

export function recordValue(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function intervalValue(value: unknown): [number, number] | null {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    value.some((item) => typeof item !== "number" || !Number.isFinite(item))
  ) {
    return null;
  }
  return [value[0] as number, value[1] as number];
}

export function shortModelName(value: string | null): string {
  if (!value) return "Model pending";
  return value.split("/").at(-1) ?? value;
}

export function formatDuration(seconds: number): string {
  const totalSeconds = Math.round(seconds);
  if (totalSeconds < 60) return `${totalSeconds} sec`;
  if (totalSeconds >= 3_600) {
    const hours = Math.floor(totalSeconds / 3_600);
    const minutes = Math.floor((totalSeconds % 3_600) / 60);
    return `${hours}h ${minutes}m`;
  }
  return `${Math.floor(totalSeconds / 60)}m ${totalSeconds % 60}s`;
}

export function percent(
  value: number | null,
  fallback = "Not reported",
): string {
  return value === null ? fallback : `${(value * 100).toFixed(1)}%`;
}

export function formatRateInterval(
  rate: number | null,
  interval: [number, number] | null,
  examples: number | null,
  fallback: string,
): string {
  if (rate === null) return fallback;
  const confidence = interval
    ? ` · 95% CI ${(interval[0] * 100).toFixed(1)}–${(interval[1] * 100).toFixed(1)}%`
    : "";
  const sample = examples === null ? "" : ` · n=${examples}`;
  return `${percent(rate)}${confidence}${sample}`;
}
