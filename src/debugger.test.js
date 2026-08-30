import assert from "node:assert/strict";
import test from "node:test";

import {
  appendRunHistory,
  buildHistogram,
  compareRunSnapshots,
  createRunSnapshot,
  normaliseWatchlist,
  watchlistStorageKey,
} from "./debugger.js";

test("histograms ignore non-finite values and account for every observation", () => {
  const histogram = buildHistogram([0, 1, 2, 3, 4, NaN, Infinity, null]);
  assert.equal(histogram.count, 5);
  assert.equal(histogram.bins.reduce((sum, bin) => sum + bin.count, 0), 5);
  assert.deepEqual(histogram.domain, [0, 4]);
  assert.equal(histogram.statistics.median, 2);
  assert.equal(histogram.statistics.mean, 2);
});

test("signed histograms use a zero-centred domain without losing boundary values", () => {
  const histogram = buildHistogram([-4, -1, 0, 0.25, 2], { symmetric: true, maximumBins: 10 });
  assert.deepEqual(histogram.domain, [-4, 4]);
  assert.equal(histogram.bins.length % 2, 0);
  assert.equal(histogram.bins.reduce((sum, bin) => sum + bin.count, 0), 5);
});

test("constant observations receive a finite display domain", () => {
  const histogram = buildHistogram([0, 0, 0, 0]);
  assert.deepEqual(histogram.domain, [-0.5, 0.5]);
  assert.equal(histogram.bins.reduce((sum, bin) => sum + bin.count, 0), 4);
});

function run(id, value, probability, revision = "abc") {
  return {
    runId: id,
    modelId: "test/model",
    revision,
    prompt: `Prompt ${id}`,
    metric: { name: "Target", kind: "target_probability", value, direction: "maximize" },
    target: { tokenId: 1, text: " yes", probability, logit: 2, rank: 1 },
    nextToken: { entropy: 1.2, topK: [{ text: " yes" }] },
    context: { seed: 0, dtype: "float32", device: "cpu", software: { worker: "0.3.0" } },
    layers: [{ layer: 0, residPost: { norm: 2, dla: value }, attentionWrite: { norm: 1, dla: value / 2 }, mlpWrite: { norm: 1, dla: -value / 2 } }],
    evidence: { kind: "observational", causal: false },
  };
}

test("run history deduplicates and keeps newest runs first", () => {
  const first = createRunSnapshot(run("a", 0.2, 0.2));
  const second = createRunSnapshot(run("b", 0.4, 0.4));
  const history = appendRunHistory(appendRunHistory([], first), second);
  assert.deepEqual(history.map((item) => item.id), ["b", "a"]);
  assert.deepEqual(appendRunHistory(history, first).map((item) => item.id), ["a", "b"]);
});

test("run diff reports metric deltas only for compatible scientific contexts", () => {
  const left = createRunSnapshot(run("a", 0.2, 0.2));
  const right = createRunSnapshot(run("b", 0.5, 0.5));
  const comparison = compareRunSnapshots(left, right);
  assert.equal(comparison.compatibility.metricComparable, true);
  assert.ok(Math.abs(comparison.metricDelta - 0.3) < 1e-12);
  assert.ok(Math.abs(comparison.layers[0].residualDlaDelta - 0.3) < 1e-12);
  const incompatible = compareRunSnapshots(left, createRunSnapshot(run("c", 0.8, 0.8, "other")));
  assert.equal(incompatible.metricDelta, null);
  assert.equal(incompatible.compatibility.sameRevision, false);
  const otherTargetSource = run("d", 0.5, 0.5);
  otherTargetSource.target = { ...otherTargetSource.target, tokenId: 2, text: " no" };
  const otherTarget = compareRunSnapshots(left, createRunSnapshot(otherTargetSource));
  assert.equal(otherTarget.targetProbabilityDelta, null);
  assert.equal(otherTarget.compatibility.sameTarget, false);
});

test("watchlist entries are unique, bounded, and evidence-labelled", () => {
  const values = normaliseWatchlist([
    { nodeId: "l0_mlp", label: "Layer 0", evidence: "causal", note: "Patch changed metric." },
    { nodeId: "l0_mlp", label: "Duplicate" },
    { nodeId: "l1_output", evidence: "invalid", note: "x".repeat(3000) },
  ]);
  assert.equal(values.length, 2);
  assert.equal(values[0].evidence, "causal");
  assert.equal(values[1].evidence, "hypothesis");
  assert.equal(values[1].note.length, 2000);
  assert.match(watchlistStorageKey("test/model", "abc"), /test%2Fmodel/);
});
