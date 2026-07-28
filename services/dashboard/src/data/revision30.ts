export interface Revision30Condition {
  id: string;
  label: string;
  role: string;
  seed: number;
  branchWidth: 1 | 4;
  status: "SUCCEEDED" | "FAILED";
  mutationEnabled: boolean;
  initialRate: number | null;
  finalRate: number | null;
  gain: number | null;
  optimizerUpdates: number | null;
  policyUpdates: number | null;
  sampledCompletions: number | null;
  pairedImproved: number | null;
  pairedRegressed: number | null;
  pValue: number | null;
  stopReason: string | null;
}

export interface Revision30Adapter {
  conditionId: string;
  exactSuccesses: number;
  exactRate: number;
  protocolValidity: number;
  improved: number;
  regressed: number;
  unchanged: number;
}

export interface Revision30Execution {
  id: string;
  condition: string;
  outcome: string;
  gpu: string;
  costUsd: number | null;
  startedAt: string;
  completedAt: string;
  teardownConfirmed: boolean;
}

export const revision30Study = {
  source: {
    worktreeCommit: "d09b10732bb24d476124e2498c72b7173b5ead4b",
    path: "research/studies/revision30-confirmatory-results.json",
    reportDigest:
      "sha256:063ad17d91ec6d11f4c31ce6790528829fd66619a6f783e579b8f3ad4714ecac",
    generatedAt: "2026-07-28T23:27:16+00:00",
  },
  studyId: "repository-repair-confirmatory-study@1",
  overallStatus: "FAIL",
  aggregationPolicy:
    "Every execution, seed, and regression is reported before any conclusion; no mean replaces per-seed outcomes.",
  freeze: {
    commit: "7195efc",
    sourceCommit: "e6a139375a4e6ec92df362873237b398aa0041c0",
    workload: "runpod-repository-repair-causal-credit@30",
    objective: "verified-fix-coverage-retention-policy-gradient@15",
    model: "Qwen/Qwen2.5-Coder-3B-Instruct",
    modelRevision: "488639f1ff808d1d3d0ba301aef8c11461451ec5",
  },
  conditions: [
    {
      id: "k4_train_seed113",
      label: "K=4 train · seed 113",
      role: "Frozen reference",
      seed: 113,
      branchWidth: 4,
      status: "SUCCEEDED",
      mutationEnabled: true,
      initialRate: 0.3125,
      finalRate: 0.5,
      gain: 0.1875,
      optimizerUpdates: 17,
      policyUpdates: 15,
      sampledCompletions: 440,
      pairedImproved: 9,
      pairedRegressed: 0,
      pValue: 0.00390625,
      stopReason: "validation_regression",
    },
    {
      id: "k4_train_seed211",
      label: "K=4 train · seed 211",
      role: "Protocol-ineligible failure",
      seed: 211,
      branchWidth: 4,
      status: "FAILED",
      mutationEnabled: true,
      initialRate: null,
      finalRate: null,
      gain: null,
      optimizerUpdates: null,
      policyUpdates: null,
      sampledCompletions: null,
      pairedImproved: null,
      pairedRegressed: null,
      pValue: null,
      stopReason: null,
    },
    {
      id: "k4_train_seed307",
      label: "K=4 train · seed 307",
      role: "Fresh replication",
      seed: 307,
      branchWidth: 4,
      status: "SUCCEEDED",
      mutationEnabled: true,
      initialRate: 0.291667,
      finalRate: 0.291667,
      gain: 0,
      optimizerUpdates: 8,
      policyUpdates: 8,
      sampledCompletions: 156,
      pairedImproved: 0,
      pairedRegressed: 0,
      pValue: 1,
      stopReason: "validation_regression",
    },
    {
      id: "k4_train_seed701",
      label: "K=4 train · seed 701",
      role: "Fresh replication",
      seed: 701,
      branchWidth: 4,
      status: "SUCCEEDED",
      mutationEnabled: true,
      initialRate: 0.291666,
      finalRate: 0.458333,
      gain: 0.166667,
      optimizerUpdates: 19,
      policyUpdates: 16,
      sampledCompletions: 416,
      pairedImproved: 8,
      pairedRegressed: 0,
      pValue: 0.0078125,
      stopReason: "consecutive_uninformative_groups",
    },
    {
      id: "k4_no_update_seed307",
      label: "K=4 frozen · seed 307",
      role: "Matched frozen-policy control",
      seed: 307,
      branchWidth: 4,
      status: "SUCCEEDED",
      mutationEnabled: false,
      initialRate: 0.291667,
      finalRate: 0.291667,
      gain: 0,
      optimizerUpdates: 0,
      policyUpdates: 0,
      sampledCompletions: 156,
      pairedImproved: 0,
      pairedRegressed: 0,
      pValue: 1,
      stopReason: "study_completion_budget",
    },
    {
      id: "k1_train_seed307",
      label: "K=1 train · seed 307",
      role: "Branch-width ablation",
      seed: 307,
      branchWidth: 1,
      status: "SUCCEEDED",
      mutationEnabled: true,
      initialRate: 0.291667,
      finalRate: 0.541667,
      gain: 0.25,
      optimizerUpdates: 14,
      policyUpdates: 14,
      sampledCompletions: 156,
      pairedImproved: 13,
      pairedRegressed: 1,
      pValue: 0.0018310546875,
      stopReason: "study_completion_budget",
    },
  ] satisfies Revision30Condition[],
  decisions: [
    {
      id: "all_retained_adapters_evaluated_externally",
      label: "Every retained adapter reached the external pack",
      detail: "Five of five retained policies were evaluated on all nine tasks.",
      status: "PASS",
    },
    {
      id: "fresh_k4_gains_transfer_without_regressions",
      label: "Fresh K=4 gains transfer without regressions",
      detail: "Seed 701 improved one task; seed 307 changed none.",
      status: "FAIL",
    },
    {
      id: "gains_repeat_across_fresh_seeds",
      label: "Internal gains repeat across fresh seeds",
      detail: "Seed 307 showed no internal gain; seed 701 gained 16.67 points.",
      status: "FAIL",
    },
    {
      id: "k4_training_beats_frozen_policy_k4",
      label: "K=4 training beats the frozen K=4 control",
      detail: "Both matched seed-307 policies finished at 29.17% on 48 tasks.",
      status: "FAIL",
    },
    {
      id: "k4_training_beats_matched_k1",
      label: "K=4 training beats matched K=1",
      detail: "K=1 finished at 54.17%; K=4 finished at 29.17%.",
      status: "FAIL",
    },
    {
      id: "regression_guard_remains_clean",
      label: "Fresh K=4 regression guards remain clean",
      detail: "Both fresh K=4 runs retained their adapters with zero paired regressions.",
      status: "PASS",
    },
  ] as const,
  external: {
    packId: "revision30-post-freeze-external-pack@1",
    resultDigest:
      "sha256:45e8684c446839e73795c85d56c830a80c17e15d783e7dd06769717c0e0d612d",
    taskCount: 9,
    domainTaskCounts: {
      "Filesystem CLI": 3,
      "Micro repository": 3,
      "SQLite repair": 3,
    },
    base: {
      conditionId: "disabled_adapter_base",
      exactSuccesses: 1,
      exactRate: 0.111111,
      protocolValidity: 0.911765,
      improved: 0,
      regressed: 0,
      unchanged: 9,
    },
    adapters: [
      {
        conditionId: "k4_train_seed113",
        exactSuccesses: 2,
        exactRate: 0.222222,
        protocolValidity: 0.978261,
        improved: 1,
        regressed: 0,
        unchanged: 8,
      },
      {
        conditionId: "k4_train_seed307",
        exactSuccesses: 1,
        exactRate: 0.111111,
        protocolValidity: 0.911765,
        improved: 0,
        regressed: 0,
        unchanged: 9,
      },
      {
        conditionId: "k4_train_seed701",
        exactSuccesses: 2,
        exactRate: 0.222222,
        protocolValidity: 1,
        improved: 1,
        regressed: 0,
        unchanged: 8,
      },
      {
        conditionId: "k4_no_update_seed307",
        exactSuccesses: 1,
        exactRate: 0.111111,
        protocolValidity: 0.911765,
        improved: 0,
        regressed: 0,
        unchanged: 9,
      },
      {
        conditionId: "k1_train_seed307",
        exactSuccesses: 0,
        exactRate: 0,
        protocolValidity: 0.967213,
        improved: 0,
        regressed: 1,
        unchanged: 8,
      },
    ] satisfies Revision30Adapter[],
  },
  executions: [
    ["142404", "k4_train_seed113", "SUCCEEDED", "RTX PRO 4500", 1.137499, "2026-07-28T14:24:04Z", "2026-07-28T15:56:46Z"],
    ["173307", "k4_train_seed211", "FAILED", "A40", 0.0159, "2026-07-28T17:33:07Z", "2026-07-28T17:35:50Z"],
    ["173849", "k4_train_seed211", "FAILED", "A40", 0.018659, "2026-07-28T17:38:49Z", "2026-07-28T17:41:42Z"],
    ["175652", "k4_train_seed307", "ELIGIBLE", "A40", 0.018411, "2026-07-28T17:56:52Z", "2026-07-28T17:59:48Z"],
    ["180041", "k4_train_seed401", "INELIGIBLE", "A40", 0.023327, "2026-07-28T18:00:41Z", "2026-07-28T18:04:14Z"],
    ["180557", "k4_train_seed503", "INELIGIBLE", "A40", 0.02143, "2026-07-28T18:05:57Z", "2026-07-28T18:09:05Z"],
    ["180933", "protocol_screen_seed601", "FAILED", "A40", null, "2026-07-28T18:09:33Z", "2026-07-28T18:09:35Z"],
    ["181027", "k4_train_seed601", "INELIGIBLE", "RTX PRO 4500", 0.03823, "2026-07-28T18:10:27Z", "2026-07-28T18:13:45Z"],
    ["181623", "k4_train_seed701", "ELIGIBLE", "RTX PRO 4500", 0.065337, "2026-07-28T18:16:23Z", "2026-07-28T18:22:14Z"],
    ["182638", "k4_train_seed307", "FAILED", "RTX PRO 4500", 0.039191, "2026-07-28T18:26:38Z", "2026-07-28T18:30:12Z"],
    ["184042", "k4_train_seed701", "SUCCEEDED", "RTX PRO 4500", 1.133015, "2026-07-28T18:40:42Z", "2026-07-28T20:12:58Z"],
    ["201339", "k4_train_seed307", "SUCCEEDED", "A40", 0.321196, "2026-07-28T20:13:39Z", "2026-07-28T20:57:57Z"],
    ["205904", "k4_no_update_seed307", "SUCCEEDED", "A40", 0.281729, "2026-07-28T20:59:04Z", "2026-07-28T21:37:58Z"],
    ["214307", "k1_train_seed307", "SUCCEEDED", "A40", 0.378751, "2026-07-28T21:43:07Z", "2026-07-28T22:35:15Z"],
    ["223646", "external evaluation", "FAILED", "RTX PRO 4500", 0.005602, "2026-07-28T22:36:46Z", "2026-07-28T22:37:13Z"],
    ["223932", "external evaluation", "FAILED", "RTX PRO 4500", 0.004722, "2026-07-28T22:39:32Z", "2026-07-28T22:39:55Z"],
    ["224155", "external evaluation", "FAILED", "A40", 0.003626, "2026-07-28T22:41:55Z", "2026-07-28T22:42:25Z"],
    ["224553", "external evaluation", "FAILED", "A40", 0, "2026-07-28T22:45:53Z", "2026-07-28T22:47:56Z"],
    ["224755", "external evaluation", "FAILED", "A40", 0, "2026-07-28T22:47:55Z", "2026-07-28T22:50:17Z"],
    ["225016", "external evaluation", "FAILED", "A40", 0.022404, "2026-07-28T22:50:16Z", "2026-07-28T23:00:40Z"],
    ["230039", "external evaluation", "FAILED", "A40", null, "2026-07-28T23:00:39Z", "2026-07-28T23:00:41Z"],
    ["230057", "external evaluation", "SUCCEEDED", "RTX PRO 4500", 0.236615, "2026-07-28T23:00:57Z", "2026-07-28T23:22:25Z"],
  ].map(
    ([id, condition, outcome, gpu, costUsd, startedAt, completedAt]) =>
      ({
        id: `runpod-proof-20260728T${id}Z`,
        condition,
        outcome,
        gpu,
        costUsd,
        startedAt,
        completedAt,
        teardownConfirmed: true,
      }) as Revision30Execution,
  ),
} as const;
