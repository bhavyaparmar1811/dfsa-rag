import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import type { GateReport } from "../../api/types";
import { metricLabel, usd } from "../../lib/format";
import { GateChecks, fmtMetric } from "./parts";

beforeAll(() => {
  // motion's whileInView needs an IntersectionObserver, which jsdom lacks.
  globalThis.IntersectionObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  } as unknown as typeof IntersectionObserver;
});
afterEach(cleanup);

describe("formatting", () => {
  it("shows rates as percentages and MRR, nDCG and relevance as decimals", () => {
    expect(fmtMetric("recall@6", 0.9259)).toBe("92.6%");
    expect(fmtMetric("false_refusal_rate", 0.037)).toBe("3.7%");
    expect(fmtMetric("mrr", 0.8599)).toBe("0.860");
    expect(fmtMetric("relevance", 4.9423)).toBe("4.94");
    expect(fmtMetric("errors", 0)).toBe("0");
    expect(fmtMetric("threshold", null)).toBe("unset");
  });

  it("names regression checks after their baseline run", () => {
    expect(metricLabel("recall@6 vs #13")).toBe("Recall@6 vs run #13");
  });

  it("keeps small costs readable", () => {
    expect(usd(0.026574)).toBe("$0.027");
    expect(usd(0.0041)).toBe("$0.0041");
    expect(usd(0.000001)).toBe("< $0.0001");
    expect(usd(1.7007)).toBe("$1.70");
  });
});

describe("GateChecks", () => {
  const report: GateReport = {
    passed: false,
    baseline_run: 13,
    checks: [
      { name: "recall@6", op: ">=", value: 0.9259, limit: 0.92, passed: true, note: "" },
      { name: "false_refusal_rate", op: "<=", value: 0.1, limit: 0.08, passed: false, note: "" },
      { name: "mrr vs #13", op: ">=", value: 0.86, limit: 0.8221, passed: true, note: "baseline 0.8475" },
      { name: "reviewed_only", op: "==", value: "True", limit: "True", passed: true, note: "" },
    ],
  };

  it("groups checks and marks each as passed or failed", () => {
    render(<GateChecks report={report} />);
    for (const group of ["Floors", "Ceilings", "No regression", "Preconditions"])
      expect(screen.getByText(group)).toBeTruthy();
    expect(screen.getAllByLabelText("passed")).toHaveLength(3);
    expect(screen.getAllByLabelText("failed")).toHaveLength(1);
    expect(screen.getByText("MRR vs run #13")).toBeTruthy();
    expect(screen.getByText("was 0.848")).toBeTruthy();
  });
});
