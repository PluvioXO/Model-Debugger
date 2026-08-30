import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { GPT2_DIAGNOSTIC as record } from "./gpt2-diagnostic-data.js";

const html = readFileSync(new URL("../gpt2-diagnostic.html", import.meta.url), "utf8");

test("record pins the exact GPT-2 run and matched prompt design", () => {
  assert.equal(record.model.id, "openai-community/gpt2");
  assert.match(record.model.revision, /^[a-f0-9]{40}$/);
  assert.equal(record.model.seed, 0);
  assert.equal(record.question.selectedTokens.length, record.question.referenceTokens.length);
  const changed = record.question.selectedTokens
    .map((token, index) => token === record.question.referenceTokens[index] ? null : index)
    .filter(Number.isInteger);
  assert.deepEqual(changed, [record.question.changedTokenIndex]);
  assert.equal(record.question.metric.correctToken, " Paris");
  assert.equal(record.question.metric.incorrectToken, " Berlin");
});

test("record keeps observational and causal evidence distinct", () => {
  assert.equal(record.observation.causal, false);
  assert.equal(record.intervention.causal, true);
  assert.equal(record.observation.firstMaterialLayer, 4);
  assert.match(record.observation.note, /does not establish/i);
  assert.match(record.intervention.note, /full tested grid/i);
  assert.match(record.diagnosis.verdict, /this prompt pair/i);
  assert.ok(record.diagnosis.notSupported.some((claim) => /unique mechanism/i.test(claim)));
});

test("preloaded intervention grid is complete and arithmetically consistent", () => {
  const { baseline, sweeps, primaryResult, accumulatedResult, negativeControl } = record.intervention;
  assert.deepEqual(Object.keys(sweeps), ["attention", "mlp", "residual"]);
  Object.values(sweeps).forEach((values) => {
    assert.equal(values.length, 12);
    assert.ok(values.every(Number.isFinite));
  });
  const attentionPeak = Math.max(...sweeps.attention.map(Math.abs));
  assert.equal(Math.abs(sweeps.attention[9]), attentionPeak);
  assert.equal(primaryResult.signedEffect, sweeps.attention[9]);
  assert.equal(primaryResult.intervened, baseline + primaryResult.signedEffect);
  assert.equal(accumulatedResult.signedEffect, sweeps.residual[10]);
  assert.equal(accumulatedResult.intervened, baseline + accumulatedResult.signedEffect);
  assert.equal(negativeControl.signedEffect, sweeps.attention[0]);
  assert.ok(Math.abs(negativeControl.signedEffect / primaryResult.signedEffect) < 0.01);
});

test("dedicated route document exposes the complete diagnostic flow", () => {
  assert.match(html, /id="question"/);
  assert.match(html, /id="observe"/);
  assert.match(html, /id="intervene"/);
  assert.match(html, /id="diagnosis"/);
  assert.match(html, /Preloaded · no worker required/);
  assert.match(html, /src="\/src\/gpt2-diagnostic\.js"/);
  assert.doesNotMatch(html, /live results/i);
});
