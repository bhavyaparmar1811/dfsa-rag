import { describe, expect, it } from "vitest";
import type { Citation, MessageOut } from "../../api/types";
import { fromMessages, linkCitations } from "./linkify";

const cite = (citation: string, label = "Rule (binding)"): Citation => ({
  citation,
  label,
  key: citation,
  chunk_id: `id-${citation}`,
});

describe("linkCitations", () => {
  it("links each mention of a cited provision to its index", () => {
    const text = "Yes (COB 3.4.2). See COB 3.4.1 for exceptions.";
    expect(linkCitations(text, [cite("COB 3.4.1"), cite("COB 3.4.2")])).toBe(
      "Yes ([COB 3.4.2](cite:1)). See [COB 3.4.1](cite:0) for exceptions.",
    );
  });

  it("prefers the longer citation and never matches inside a longer rule number", () => {
    const text = "COB 3.4.2 Guidance explains COB 3.4.2, not COB 3.4.21 or COB 3.4.2.1.";
    const out = linkCitations(text, [cite("COB 3.4.2"), cite("COB 3.4.2 Guidance", "Guidance (non-binding)")]);
    expect(out).toBe(
      "[COB 3.4.2 Guidance](cite:1) explains [COB 3.4.2](cite:0), not COB 3.4.21 or COB 3.4.2.1.",
    );
  });

  it("keeps a rule's paragraph reference outside the link", () => {
    expect(linkCitations("Rule COB 3.4.2(1) applies.", [cite("COB 3.4.2")])).toBe(
      "Rule [COB 3.4.2](cite:0)(1) applies.",
    );
  });

  it("leaves provisions the answer doesn't cite as plain text", () => {
    expect(linkCitations("See GEN 2.26.1.", [cite("COB 3.4.2")])).toBe("See GEN 2.26.1.");
    expect(linkCitations("No citations.", [])).toBe("No citations.");
  });
});

const msg = (over: Partial<MessageOut>): MessageOut => ({
  id: 1,
  role: "user",
  content: "",
  rewritten_query: null,
  status: null,
  refusal_code: null,
  citations: [],
  trace_id: null,
  cost_usd: 0,
  latency_ms: null,
  created_at: "2026-10-14T12:00:00",
  ...over,
});

describe("fromMessages", () => {
  it("pairs each question with its answer and keeps a changed rewrite", () => {
    const turns = fromMessages([
      msg({ id: 1, content: "Do we need a Client Agreement?", rewritten_query: "Do we need a Client Agreement?" }),
      msg({ id: 2, role: "assistant", content: "Yes.", status: "answered", citations: [cite("COB 3.3.2")], trace_id: "t1" }),
      msg({ id: 3, content: "what about retail?", rewritten_query: "Client Agreement for a Retail Client?" }),
      msg({ id: 4, role: "assistant", content: "No answer.", status: "refused", refusal_code: "low_confidence" }),
    ]);
    expect(turns).toHaveLength(2);
    expect(turns[0]).toMatchObject({ question: "Do we need a Client Agreement?", rewritten: null, state: "answered", traceId: "t1" });
    expect(turns[1]).toMatchObject({ rewritten: "Client Agreement for a Retail Client?", state: "refused", refusalCode: "low_confidence" });
  });

  it("marks an answer resting only on Guidance", () => {
    const [turn] = fromMessages([
      msg({ id: 1, content: "q" }),
      msg({ id: 2, role: "assistant", status: "answered", citations: [cite("COB 3.4.2 Guidance", "Guidance (non-binding)")] }),
    ]);
    expect(turn.guidanceOnly).toBe(true);
  });
});
