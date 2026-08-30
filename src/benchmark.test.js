import test from "node:test";
import assert from "node:assert/strict";
import { benchmarkOutcomeForRun, normaliseBenchmarkExamples } from "./benchmark.js";

test("normalises common benchmark schemas without assuming failure", () => {
  const [example] = normaliseBenchmarkExamples([{
    id: "gsm-7",
    question: "What is 2 + 2?",
    reference_prompt: "Calculate 1 + 1.",
    answer: "4",
    benchmark_name: "gsm8k",
    split: "test",
    correct: true,
    score: 1
  }], { idFactory: () => "local-id" });
  assert.deepEqual(example, {
    id: "local-id",
    benchmark: "gsm8k",
    task: "",
    split: "test",
    exampleId: "gsm-7",
    prompt: "What is 2 + 2?",
    reference: "Calculate 1 + 1.",
    expected: "4",
    target: "",
    score: 1,
    threshold: null,
    direction: "maximize",
    suppliedOutcome: "passed",
    status: "passed",
    outcomeSource: "benchmark",
    cluster: "Untriaged"
  });
});

test("uses a benchmark threshold when no outcome was supplied", () => {
  const [example] = normaliseBenchmarkExamples([{ prompt: "Example", threshold: 0.7 }], { idFactory: () => "id" });
  assert.deepEqual(benchmarkOutcomeForRun(example, { metric: { value: 0.65 } }), {
    status: "failed",
    source: "maximize threshold 0.7"
  });
});

test("preserves a supplied anomaly outcome", () => {
  const [example] = normaliseBenchmarkExamples([{ prompt: "Example", outcome: "anomalous" }], { idFactory: () => "id" });
  assert.deepEqual(benchmarkOutcomeForRun(example, { metric: { value: 1 } }), {
    status: "anomaly",
    source: "benchmark-supplied outcome"
  });
});

test("treats a numeric result as a score rather than an outcome", () => {
  const [example] = normaliseBenchmarkExamples([{ prompt: "Example", result: 0.25 }], { idFactory: () => "id" });
  assert.equal(example.score, 0.25);
  assert.equal(example.status, "pending");
  assert.equal(example.suppliedOutcome, "");
});
