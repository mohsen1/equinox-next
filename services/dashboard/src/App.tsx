/**
 * THESIS: A verification drafting table replaces the category's KPI-card dashboard.
 * OWN-WORLD: Cool paper layers, exact rules, oxide-blue interaction, typed semantic state.
 * STORY: Launch a run, follow its shared prefix, inspect proof, then trace the committed update.
 * FIRST VIEWPORT: Compact rail, decision-led route header, and a dense evidence workspace.
 * FORM: Operate surface; shared-prefix/four-lane composition selected from study B, with A's inspector.
 */
import { AppShell } from "./components";
import {
  IterationPage,
  NewRunPage,
  RunDetailPage,
  RunsPage,
} from "./pages/runs";
import { ResourcesPage } from "./pages/resources";
import { RolloutTreePage } from "./pages/rollout";
import { VerificationPage } from "./pages/verification";
import { StudyPage } from "./pages/studies";
import { EnvironmentsPage } from "./pages/environments";
import { ProofDetailPage, ProofsPage } from "./pages/proofs";
import { Navigate, Route, Routes } from "./router";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Navigate to="/runs" replace />} />
        <Route path="/runs" element={<RunsPage />} />
        <Route path="/runs/new" element={<NewRunPage />} />
        <Route path="/runs/:runId" element={<RunDetailPage />} />
        <Route path="/studies/:studyId" element={<StudyPage />} />
        <Route
          path="/runs/:runId/iterations/:iterationId"
          element={<IterationPage />}
        />
        <Route path="/rollout-trees/:treeId" element={<RolloutTreePage />} />
        <Route
          path="/verification-runs/:verificationRunId"
          element={<VerificationPage />}
        />
        <Route path="/resources" element={<ResourcesPage />} />
        <Route path="/environments" element={<EnvironmentsPage />} />
        <Route path="/proofs" element={<ProofsPage />} />
        <Route path="/proofs/:proofBundleId" element={<ProofDetailPage />} />
        <Route path="*" element={<Navigate to="/runs" replace />} />
      </Routes>
    </AppShell>
  );
}
