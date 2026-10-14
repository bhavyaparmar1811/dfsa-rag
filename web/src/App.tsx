import { Suspense, lazy, useEffect } from "react";
import { Route, Routes, useLocation, useParams } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Page, PageHeader } from "./components/ui";
import { ChatPage } from "./features/chat/ChatPage";

// The report pages carry the chart library: load them when first opened.
const EvalsPage = lazy(() =>
  import("./features/evals/EvalsPage").then((m) => ({ default: m.EvalsPage })),
);
const RunDetailPage = lazy(() =>
  import("./features/evals/RunDetailPage").then((m) => ({ default: m.RunDetailPage })),
);
const CostPage = lazy(() =>
  import("./features/cost/CostPage").then((m) => ({ default: m.CostPage })),
);

/** "/" and "/c/:id" share one chat page. Its key changes only when a different
 * conversation is opened, so the first answer of a new chat doesn't remount it
 * when the URL gains the conversation id. */
function ChatRoute() {
  const { id } = useParams();
  const state = useLocation().state as { chatKey?: string } | null;
  const key = state?.chatKey ?? id ?? "new";
  return <ChatPage key={key} chatKey={key} conversationId={id} />;
}

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    document.getElementById("main")?.scrollTo({ top: 0 });
  }, [pathname]);
  return null;
}

export function App() {
  return (
    <AppShell>
      <ScrollToTop />
      <Suspense fallback={null}>
        <Routes>
          <Route path="/" element={<ChatRoute />} />
          <Route path="/c/:id" element={<ChatRoute />} />
          <Route path="/evals" element={<EvalsPage />} />
          <Route path="/evals/runs/:runId" element={<RunDetailPage />} />
          <Route path="/cost" element={<CostPage />} />
          <Route
            path="*"
            element={
              <Page>
                <PageHeader title="Not found">There is no page at this address.</PageHeader>
              </Page>
            }
          />
        </Routes>
      </Suspense>
    </AppShell>
  );
}
