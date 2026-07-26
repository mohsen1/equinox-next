/**
 * THESIS: A verification drafting table replaces the category's KPI-card dashboard.
 * OWN-WORLD: Cool paper layers, exact rules, oxide-blue interaction, typed semantic state.
 * STORY: Launch a run, follow its shared prefix, inspect proof, then trace the committed update.
 * FIRST VIEWPORT: Compact rail, decision-led route header, and a dense evidence workspace.
 * FORM: Operate surface; shared-prefix/four-lane composition selected from study B, with A's inspector.
 */
import { AppShell } from "./components";
import { EnvironmentsPage } from "./pages/environments";
import { ResearchTrajectoryPage } from "./pages/research-trajectory";
import { ResearchRunPage, RunsPage } from "./pages/workspace";
import { ResourcesPage } from "./pages/resources";
import { Navigate, Route, Routes } from "./router";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Navigate to="/runs" replace />} />
        <Route path="/runs" element={<RunsPage />} />
        <Route
          path="/runs/research/:executionId"
          element={<ResearchRunPage />}
        />
        <Route
          path="/runs/research/:executionId/trajectory"
          element={<ResearchTrajectoryPage />}
        />
        <Route path="/environments" element={<EnvironmentsPage />} />
        <Route path="/resources" element={<ResourcesPage />} />
        <Route path="*" element={<Navigate to="/runs" replace />} />
      </Routes>
    </AppShell>
  );
}
