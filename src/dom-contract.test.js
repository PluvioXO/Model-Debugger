import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const application = readFileSync(new URL("./app.js", import.meta.url), "utf8");

test("every queried element ID exists exactly once", () => {
  const queried = [...application.matchAll(/document\.querySelector\("#([A-Za-z][A-Za-z0-9_-]*)"\)/g)]
    .map((match) => match[1]);
  const ids = [...html.matchAll(/\sid="([A-Za-z][A-Za-z0-9_-]*)"/g)].map((match) => match[1]);
  const counts = new Map(ids.map((id) => [id, ids.filter((candidate) => candidate === id).length]));
  assert.deepEqual([...new Set(queried)].filter((id) => !counts.has(id)), []);
  assert.deepEqual([...counts].filter(([, count]) => count !== 1), []);
});

test("debugging workbench exposes all ten workflow surfaces", () => {
  [
    "debugCaseLibraryTitle",
    "debugMetricKindSelect",
    "runtimeCompareButton",
    "debugDivergenceChart",
    "interventionLabTitle",
    "rootCauseTitle",
    "microscopeTitle",
    "benchmarkExplorerTitle",
    "debugDiagnostics",
    "verificationTitle",
  ].forEach((id) => assert.match(html, new RegExp(`id="${id}"`)));
});

test("workspace exposes the integrated trace, sweep, history, watchlist, and lens tools", () => {
  [
    "generationTimelineResult",
    "runCausalSweepButton",
    "runHistoryList",
    "watchlistPanel",
    "nodeAnnotationEditor",
    "logitLensChart",
    "inferenceWaterfall",
    "inferenceWaterfallTotal",
  ].forEach((id) => assert.match(html, new RegExp(`id="${id}"`)));
  assert.match(application, /runtimeApi\("\/generate"/);
  assert.match(application, /runtimeApi\("\/sweep"/);
  assert.match(application, /compareRunSnapshots/);
  assert.match(application, /normaliseWatchlist/);
  assert.match(application, /function applyRuntimeCapabilityVisibility/);
  assert.match(html, /data-runtime-capability="logitLens"/);
  assert.match(html, /data-runtime-capability="rootCauseTrace"/);
});

test("landing page includes the complete advanced field guide", () => {
  const landingStart = html.indexOf('<div id="landing"');
  const tutorialStart = html.indexOf('<section id="tutorial"');
  const workspaceStart = html.indexOf('<header class="topbar"');

  assert.ok(landingStart >= 0 && landingStart < tutorialStart);
  assert.ok(tutorialStart < workspaceStart);
  assert.match(html, /id="tutorial" class="landing-tutorial"/);
  assert.doesNotMatch(html, /class="tutorial-page"/);
  [
    "tutorial-start",
    "tutorial-evidence",
    "tutorial-examples",
    "tutorial-advanced",
    "tutorialCompare",
    "tutorialIntervene",
    "tutorialTrace",
    "tutorialMicroscope",
    "tutorialBenchmark",
    "tutorialVerify",
    "tutorialDiagnostics",
  ].forEach((id) => assert.match(html, new RegExp(`id="${id}"`)));
  [
    "01-checkpoint-map.png",
    "02-inference-profile.png",
    "03-paired-comparison.png",
    "04-intervention-result.png",
    "05-verification-result.png",
  ].forEach((asset) => assert.match(html, new RegExp(`/assets/tutorial/${asset}`)));
});

test("workspace guide control returns to the landing tutorial", () => {
  assert.match(html, /id="tutorialButton"/);
  assert.match(application, /function openLandingTutorial\(\)/);
  assert.match(application, /elements\.tutorial\.scrollIntoView/);
  assert.match(application, /elements\.tutorialButton\.addEventListener\("click", openLandingTutorial\)/);
});

test("GPT-2 development fixture uses the ordinary Hugging Face workflow", () => {
  assert.match(html, /id="gpt2DevExampleButton"/);
  assert.match(application, /GPT2_DEVELOPMENT_MODEL_ID = "openai-community\/gpt2"/);
  assert.match(application, /await importHuggingFaceModel\(\)/);
  assert.doesNotMatch(application, /synthetic activation/i);
});

test("selected-token interventions replay against each prompt's final token", () => {
  assert.match(application, /position: scope === "position" \? -1 : baseRun\.position/);
});

test("Daytona is the default managed runtime and API keys are not persisted client-side", () => {
  assert.match(html, /id="runtimeModeDaytona"[^>]*aria-pressed="true"/);
  assert.match(html, /id="runtimeDaytonaApiKeyInput"[^>]*type="password"/);
  assert.match(html, /id="runtimeDaytonaValidateButton"[^>]*disabled>Check API key<\/button>/);
  assert.match(html, /id="runtimeDaytonaKeyStatus"[^>]*aria-live="polite"/);
  assert.match(html, /id="runtimeDaytonaHfAccess"[^>]*data-state="public"/);
  assert.match(html, /id="runtimeDaytonaHfStatus"/);
  assert.doesNotMatch(html, /id="runtimeDaytonaHfTokenInput"/);
  assert.match(application, /runtimeApi\("\/daytona\/provision"/);
  assert.match(application, /runtimeApi\("\/daytona\/validate"/);
  assert.doesNotMatch(application, /hfToken:/);
  assert.match(application, /setRuntimeMode\("daytona"\)/);
  assert.doesNotMatch(html, /Google Colab|trycloudflare/i);
  assert.doesNotMatch(application, /localStorage\.setItem\([^)]*Daytona/i);
});
