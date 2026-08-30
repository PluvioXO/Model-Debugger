import {
  formatBytes,
  formatCount,
  formatShape
} from "./presentation.js";
import {
  routeGraphEdge,
  routeGraphEdges,
  tensorStackDepth
} from "./graph-routing.js";
import {
  benchmarkOutcomeForRun,
  normaliseBenchmarkExamples
} from "./benchmark.js";
import {
  appendRunHistory,
  buildHistogram,
  compareRunSnapshots,
  createRunSnapshot,
  normaliseWatchlist,
  watchlistStorageKey
} from "./debugger.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const NODE_WIDTH = 244;
const NODE_HEIGHT = 84;
const X_GAP = 116;
const PAGE_PADDING = 54;
const TRANSFORMER_BLOCK_HEADER_HEIGHT = 22;
const TRANSFORMER_BLOCK_HEADER_WIDTH = 176;
const TRANSFORMER_BLOCK_HEADER_INSET = 10;
const TRANSFORMER_BLOCK_HEADER_RISE = 24;
const MIN_SCALE = 0.008;
const MAX_SCALE = 2.5;
const ARROWHEAD_SCREEN_SIZE = 8;
const GPT2_DEVELOPMENT_MODEL_ID = "openai-community/gpt2";
const GPT2_DEVELOPMENT_EXAMPLES = [
  { example_id: "factual-recall", task: "Factual recall", prompt: "The capital of France is", reference_prompt: "The capital of Germany is", expected: " Paris", target: " Paris", uxOutcome: "passed", uxScore: 0.82 },
  { example_id: "arithmetic", task: "Arithmetic", prompt: "Two plus two equals", reference_prompt: "Three plus three equals", expected: " four", target: " four", uxOutcome: "failed", uxScore: 0.18 },
  { example_id: "physical-knowledge", task: "Physical knowledge", prompt: "Water freezes at", reference_prompt: "Water boils at", expected: " zero", target: " zero", uxOutcome: "anomaly", uxScore: 0.47 },
  { example_id: "antonym", task: "Lexical relation", prompt: "The opposite of hot is", reference_prompt: "The opposite of up is", expected: " cold", target: " cold", uxOutcome: "passed", uxScore: 0.76 },
  { example_id: "animal-sound", task: "Commonsense", prompt: "A dog says", reference_prompt: "A cat says", expected: " woof", target: " woof", uxOutcome: "regressed", uxScore: 0.31 },
  { example_id: "syntax", task: "Syntax", prompt: "Because the meeting was late, the manager", reference_prompt: "Because the train was late, the passenger", expected: " was", target: " was", uxOutcome: "observed", uxScore: 0.55 }
];

const elements = {
  landing: document.querySelector("#landing"),
  tutorial: document.querySelector("#tutorial"),
  landingStartButton: document.querySelector("#landingStartButton"),
  workspaceLaunchers: [...document.querySelectorAll("[data-open-workspace]")],
  appHomeButton: document.querySelector("#appHomeButton"),
  tutorialButton: document.querySelector("#tutorialButton"),
  settingsButton: document.querySelector("#settingsButton"),
  settingsButtonIcon: document.querySelector("#settingsButtonIcon"),
  settingsButtonAvatar: document.querySelector("#settingsButtonAvatar"),
  settingsOverlay: document.querySelector("#settingsOverlay"),
  settingsDrawer: document.querySelector("#settingsDrawer"),
  settingsBackdrop: document.querySelector("#settingsBackdrop"),
  closeSettingsButton: document.querySelector("#closeSettingsButton"),
  settingsAccountMount: document.querySelector("#settingsAccountMount"),
  settingsWorkerMount: document.querySelector("#settingsWorkerMount"),
  appStatusText: document.querySelector("#appStatusText"),
  statusDot: document.querySelector(".status-dot"),
  workspace: document.querySelector(".workspace"),
  editorPanel: document.querySelector("#editorPanel"),
  collapseSidebarButton: document.querySelector("#collapseSidebarButton"),
  sidebarToggleButton: document.querySelector("#sidebarToggleButton"),
  sidebarToggleLabel: document.querySelector(".sidebar-toggle-label"),
  hfImport: document.querySelector(".hf-import"),
  hfAccount: document.querySelector("#hfAccount"),
  hfAccountTitle: document.querySelector("#hfAccountTitle"),
  hfAccountMark: document.querySelector("#hfAccountMark"),
  hfAccountAvatar: document.querySelector("#hfAccountAvatar"),
  hfAccountAvatarFallback: document.querySelector("#hfAccountAvatarFallback"),
  hfAccountForm: document.querySelector("#hfAccountForm"),
  hfAccountToggleButton: document.querySelector("#hfAccountToggleButton"),
  hfAccountConnectButton: document.querySelector("#hfAccountConnectButton"),
  hfAccountDetail: document.querySelector("#hfAccountDetail"),
  hfAccountStatus: document.querySelector("#hfAccountStatus"),
  hfAccountProfile: document.querySelector("#hfAccountProfile"),
  hfAccountOrgs: document.querySelector("#hfAccountOrgs"),
  hfAccountProfileLink: document.querySelector("#hfAccountProfileLink"),
  hfTokenInput: document.querySelector("#hfTokenInput"),
  hfModelInput: document.querySelector("#hfModelInput"),
  hfRevisionInput: document.querySelector("#hfRevisionInput"),
  hfImportButton: document.querySelector("#hfImportButton"),
  hfImportStatus: document.querySelector("#hfImportStatus"),
  gpt2DevExampleButton: document.querySelector("#gpt2DevExampleButton"),
  importSummary: document.querySelector("#importSummary"),
  loadedRepositoryLink: document.querySelector("#loadedRepositoryLink"),
  loadedRepository: document.querySelector("#loadedRepository"),
  loadedRevision: document.querySelector("#loadedRevision"),
  loadedCommit: document.querySelector("#loadedCommit"),
  loadedTensors: document.querySelector("#loadedTensors"),
  loadedShards: document.querySelector("#loadedShards"),
  checkpointNote: document.querySelector("#checkpointNote"),
  predictionList: document.querySelector("#predictionList"),
  predictionCount: document.querySelector("#predictionCount"),
  debugCaseSelect: document.querySelector("#debugCaseSelect"),
  newDebugCaseButton: document.querySelector("#newDebugCaseButton"),
  openDebugCaseButton: document.querySelector("#openDebugCaseButton"),
  deleteDebugCaseButton: document.querySelector("#deleteDebugCaseButton"),
  debugCaseLibraryStatus: document.querySelector("#debugCaseLibraryStatus"),
  runtimePanel: document.querySelector("#runtimePanel"),
  runtimeBadge: document.querySelector("#runtimeBadge"),
  runtimeConnectFields: document.querySelector("#runtimeConnectFields"),
  runtimeModeDaytona: document.querySelector("#runtimeModeDaytona"),
  runtimeModeLocal: document.querySelector("#runtimeModeLocal"),
  runtimeDaytonaSetup: document.querySelector("#runtimeDaytonaSetup"),
  runtimeLocalSetup: document.querySelector("#runtimeLocalSetup"),
  runtimeGpuRecommendation: document.querySelector("#runtimeGpuRecommendation"),
  runtimeGpuReason: document.querySelector("#runtimeGpuReason"),
  runtimeGpuSelect: document.querySelector("#runtimeGpuSelect"),
  runtimeDaytonaApiKeyInput: document.querySelector("#runtimeDaytonaApiKeyInput"),
  runtimeDaytonaHfTokenInput: document.querySelector("#runtimeDaytonaHfTokenInput"),
  copyRuntimeCommandButton: document.querySelector("#copyRuntimeCommandButton"),
  runtimeEndpointInput: document.querySelector("#runtimeEndpointInput"),
  runtimeSecretInput: document.querySelector("#runtimeSecretInput"),
  runtimeSecretLabel: document.querySelector("#runtimeSecretLabel"),
  runtimeConnectButton: document.querySelector("#runtimeConnectButton"),
  runtimeControls: document.querySelector("#runtimeControls"),
  runtimeAccelerator: document.querySelector("#runtimeAccelerator"),
  runtimeModel: document.querySelector("#runtimeModel"),
  runtimeLoadButton: document.querySelector("#runtimeLoadButton"),
  runtimePromptInput: document.querySelector("#runtimePromptInput"),
  runtimeControlPromptInput: document.querySelector("#runtimeControlPromptInput"),
  runtimeTargetInput: document.querySelector("#runtimeTargetInput"),
  runtimeTopKInput: document.querySelector("#runtimeTopKInput"),
  runtimeForwardButton: document.querySelector("#runtimeForwardButton"),
  runtimeCompareButton: document.querySelector("#runtimeCompareButton"),
  runtimeResult: document.querySelector("#runtimeResult"),
  runtimeDisconnectButton: document.querySelector("#runtimeDisconnectButton"),
  runtimeStatus: document.querySelector("#runtimeStatus"),
  runModelButton: document.querySelector("#runModelButton"),
  modelInterface: document.querySelector("#modelInterface"),
  closeModelInterfaceButton: document.querySelector("#closeModelInterfaceButton"),
  modelInterfaceSubtitle: document.querySelector("#modelInterfaceSubtitle"),
  modelRunConnection: document.querySelector("#modelRunConnection"),
  modelRunConnectionTitle: document.querySelector("#modelRunConnectionTitle"),
  modelRunConnectionDetail: document.querySelector("#modelRunConnectionDetail"),
  modelRunSettingsButton: document.querySelector("#modelRunSettingsButton"),
  modelRunForm: document.querySelector("#modelRunForm"),
  modelRunStatus: document.querySelector("#modelRunStatus"),
  debugCaseNameInput: document.querySelector("#debugCaseNameInput"),
  debugExpectedInput: document.querySelector("#debugExpectedInput"),
  debugNotesInput: document.querySelector("#debugNotesInput"),
  debugCaseSaveState: document.querySelector("#debugCaseSaveState"),
  saveDebugCaseButton: document.querySelector("#saveDebugCaseButton"),
  duplicateDebugCaseButton: document.querySelector("#duplicateDebugCaseButton"),
  debugMetricSummary: document.querySelector("#debugMetricSummary"),
  debugMetricNameInput: document.querySelector("#debugMetricNameInput"),
  debugMetricKindSelect: document.querySelector("#debugMetricKindSelect"),
  debugMetricAnswerInput: document.querySelector("#debugMetricAnswerInput"),
  debugMetricPositiveInput: document.querySelector("#debugMetricPositiveInput"),
  debugMetricNegativeInput: document.querySelector("#debugMetricNegativeInput"),
  debugSeedInput: document.querySelector("#debugSeedInput"),
  modelRunSummary: document.querySelector("#modelRunSummary"),
  inferenceWaterfallTotal: document.querySelector("#inferenceWaterfallTotal"),
  inferenceWaterfall: document.querySelector("#inferenceWaterfall"),
  inferenceWaterfallNote: document.querySelector("#inferenceWaterfallNote"),
  modelTokenContext: document.querySelector("#modelTokenContext"),
  modelTokenStrip: document.querySelector("#modelTokenStrip"),
  modelPredictionEntropy: document.querySelector("#modelPredictionEntropy"),
  modelPredictionList: document.querySelector("#modelPredictionList"),
  modelAttributionMethod: document.querySelector("#modelAttributionMethod"),
  modelAttributionSummary: document.querySelector("#modelAttributionSummary"),
  modelContributionChart: document.querySelector("#modelContributionChart"),
  componentDlaChart: document.querySelector("#componentDlaChart"),
  metricDistributionScope: document.querySelector("#metricDistributionScope"),
  metricHistoryHistogram: document.querySelector("#metricHistoryHistogram"),
  componentDlaHistogram: document.querySelector("#componentDlaHistogram"),
  residualNormHistogram: document.querySelector("#residualNormHistogram"),
  modelAttributionNote: document.querySelector("#modelAttributionNote"),
  residualNormChart: document.querySelector("#residualNormChart"),
  residualDlaChart: document.querySelector("#residualDlaChart"),
  residualDeltaChart: document.querySelector("#residualDeltaChart"),
  residualCosineChart: document.querySelector("#residualCosineChart"),
  componentNormChart: document.querySelector("#componentNormChart"),
  attentionEntropyMap: document.querySelector("#attentionEntropyMap"),
  logitLensMethod: document.querySelector("#logitLensMethod"),
  logitLensChart: document.querySelector("#logitLensChart"),
  logitLensStages: document.querySelector("#logitLensStages"),
  logitLensNote: document.querySelector("#logitLensNote"),
  modelHookCount: document.querySelector("#modelHookCount"),
  modelHookTable: document.querySelector("#modelHookTable"),
  debugDiagnosticCount: document.querySelector("#debugDiagnosticCount"),
  debugDiagnostics: document.querySelector("#debugDiagnostics"),
  debugComparisonResult: document.querySelector("#debugComparisonResult"),
  debugComparisonMetric: document.querySelector("#debugComparisonMetric"),
  debugComparisonSummary: document.querySelector("#debugComparisonSummary"),
  debugTokenAlignment: document.querySelector("#debugTokenAlignment"),
  debugDivergenceChart: document.querySelector("#debugDivergenceChart"),
  debugDivergenceHistogram: document.querySelector("#debugDivergenceHistogram"),
  debugDivergenceList: document.querySelector("#debugDivergenceList"),
  debugInterventionComponent: document.querySelector("#debugInterventionComponent"),
  debugInterventionMethod: document.querySelector("#debugInterventionMethod"),
  debugInterventionScale: document.querySelector("#debugInterventionScale"),
  debugInterventionScope: document.querySelector("#debugInterventionScope"),
  debugRunInterventionButton: document.querySelector("#debugRunInterventionButton"),
  debugInterventionResult: document.querySelector("#debugInterventionResult"),
  debugTraceCandidates: document.querySelector("#debugTraceCandidates"),
  debugTraceThreshold: document.querySelector("#debugTraceThreshold"),
  debugRunTraceButton: document.querySelector("#debugRunTraceButton"),
  debugTraceResult: document.querySelector("#debugTraceResult"),
  debugMicroscopeHook: document.querySelector("#debugMicroscopeHook"),
  debugMicroscopePosition: document.querySelector("#debugMicroscopePosition"),
  debugMicroscopeHead: document.querySelector("#debugMicroscopeHead"),
  debugMicroscopeLimit: document.querySelector("#debugMicroscopeLimit"),
  debugLoadActivationButton: document.querySelector("#debugLoadActivationButton"),
  debugMicroscopeResult: document.querySelector("#debugMicroscopeResult"),
  debugBenchmarkName: document.querySelector("#debugBenchmarkName"),
  debugBenchmarkFilter: document.querySelector("#debugBenchmarkFilter"),
  debugInboxFile: document.querySelector("#debugInboxFile"),
  debugInboxDataset: document.querySelector("#debugInboxDataset"),
  debugImportInboxButton: document.querySelector("#debugImportInboxButton"),
  debugRunInboxButton: document.querySelector("#debugRunInboxButton"),
  debugInboxCount: document.querySelector("#debugInboxCount"),
  debugInboxList: document.querySelector("#debugInboxList"),
  debugGuardrailsInput: document.querySelector("#debugGuardrailsInput"),
  debugRunVerificationButton: document.querySelector("#debugRunVerificationButton"),
  debugExportReportButton: document.querySelector("#debugExportReportButton"),
  debugVerificationResult: document.querySelector("#debugVerificationResult"),
  generationMaxTokens: document.querySelector("#generationMaxTokens"),
  generationMode: document.querySelector("#generationMode"),
  generationTemperature: document.querySelector("#generationTemperature"),
  generationTopP: document.querySelector("#generationTopP"),
  generationLensStages: document.querySelector("#generationLensStages"),
  runGenerationButton: document.querySelector("#runGenerationButton"),
  generationTimelineResult: document.querySelector("#generationTimelineResult"),
  runHistoryCount: document.querySelector("#runHistoryCount"),
  runHistoryList: document.querySelector("#runHistoryList"),
  runDiffLeft: document.querySelector("#runDiffLeft"),
  runDiffRight: document.querySelector("#runDiffRight"),
  compareRunsButton: document.querySelector("#compareRunsButton"),
  runDiffResult: document.querySelector("#runDiffResult"),
  sweepComponentKind: document.querySelector("#sweepComponentKind"),
  sweepMethod: document.querySelector("#sweepMethod"),
  sweepPositionCount: document.querySelector("#sweepPositionCount"),
  sweepScale: document.querySelector("#sweepScale"),
  runCausalSweepButton: document.querySelector("#runCausalSweepButton"),
  causalSweepResult: document.querySelector("#causalSweepResult"),
  graphPanel: document.querySelector(".graph-panel"),
  graphLoading: document.querySelector("#graphLoading"),
  graphLoadingTitle: document.querySelector("#graphLoadingTitle"),
  graphLoadingDetail: document.querySelector("#graphLoadingDetail"),
  cancelImportButton: document.querySelector("#cancelImportButton"),
  emptyState: document.querySelector("#emptyState"),
  emptyFocusButton: document.querySelector("#emptyFocusButton"),
  modelTitle: document.querySelector("#modelTitle"),
  modelType: document.querySelector("#modelType"),
  modelEvidence: document.querySelector("#modelEvidence"),
  validationStatus: document.querySelector("#validationStatus"),
  topologySummary: document.querySelector("#topologySummary"),
  totalParameters: document.querySelector("#totalParameters"),
  checkpointBreakdown: document.querySelector("#checkpointBreakdown"),
  moduleCount: document.querySelector("#moduleCount"),
  unitLabel: document.querySelector("#unitLabel"),
  memoryEstimate: document.querySelector("#memoryEstimate"),
  trainablePercent: document.querySelector("#trainablePercent"),
  searchShell: document.querySelector(".search-shell"),
  searchInput: document.querySelector("#searchInput"),
  searchCount: document.querySelector("#searchCount"),
  detailLevelHint: document.querySelector("#detailLevelHint"),
  viewport: document.querySelector("#graphViewport"),
  svg: document.querySelector("#graphSvg"),
  arrowheads: [...document.querySelectorAll("[data-arrowhead]")],
  camera: document.querySelector("#camera"),
  groupLayer: document.querySelector("#groupLayer"),
  tensorStackLayer: document.querySelector("#tensorStackLayer"),
  edgeLayer: document.querySelector("#edgeLayer"),
  nodeLayer: document.querySelector("#nodeLayer"),
  zoomOut: document.querySelector("#zoomOutButton"),
  zoomReadout: document.querySelector("#zoomReadout"),
  zoomIn: document.querySelector("#zoomInButton"),
  residualLedgerButton: document.querySelector("#residualLedgerButton"),
  watchlistButton: document.querySelector("#watchlistButton"),
  watchlistCount: document.querySelector("#watchlistCount"),
  watchlistPanel: document.querySelector("#watchlistPanel"),
  closeWatchlistButton: document.querySelector("#closeWatchlistButton"),
  watchlistList: document.querySelector("#watchlistList"),
  fit: document.querySelector("#fitButton"),
  export: document.querySelector("#exportButton"),
  residualLedger: document.querySelector("#residualLedger"),
  closeResidualLedger: document.querySelector("#closeResidualLedgerButton"),
  residualLedgerPosition: document.querySelector("#residualLedgerPosition"),
  residualLedgerMetric: document.querySelector("#residualLedgerMetric"),
  residualLedgerNote: document.querySelector("#residualLedgerNote"),
  residualLedgerList: document.querySelector("#residualLedgerList"),
  inspector: document.querySelector("#inspector"),
  inspectorKind: document.querySelector("#inspectorKind"),
  inspectorName: document.querySelector("#inspectorName"),
  inspectorDescription: document.querySelector("#inspectorDescription"),
  inspectorDetails: document.querySelector("#inspectorDetails"),
  inspectorTabs: document.querySelector("#inspectorTabs"),
  inspectorTensorCount: document.querySelector("#inspectorTensorCount"),
  inspectorConnections: document.querySelector("#inspectorConnections"),
  inspectorTensors: document.querySelector("#inspectorTensors"),
  inspectorHuggingFace: document.querySelector("#inspectorHuggingFace"),
  inspectorRaw: document.querySelector("#inspectorRaw"),
  copyNodePath: document.querySelector("#copyNodePathButton"),
  toggleWatchNodeButton: document.querySelector("#toggleWatchNodeButton"),
  nodeEvidenceSelect: document.querySelector("#nodeEvidenceSelect"),
  nodeAnnotationInput: document.querySelector("#nodeAnnotationInput"),
  saveNodeAnnotationButton: document.querySelector("#saveNodeAnnotationButton"),
  nodeAnnotationStatus: document.querySelector("#nodeAnnotationStatus"),
  inspectorActionStatus: document.querySelector("#inspectorActionStatus"),
  closeInspector: document.querySelector("#closeInspectorButton"),
  graphAnnouncement: document.querySelector("#graphAnnouncement")
};

const state = {
  model: null,
  selectedId: null,
  inspectorTab: "overview",
  inspectorRendered: new Set(),
  inspectorRenderToken: 0,
  transform: { x: 40, y: 40, scale: 1 },
  layout: [],
  bounds: { width: 0, height: 0 },
  drag: null,
  cameraFrame: null,
  motionTimer: null,
  importController: null,
  hfAccount: null,
  hfAccountLoading: false,
  importOutcome: "idle",
  searchCursor: -1,
  copyResetTimer: null,
  searchFrame: null,
  viewportSize: null,
  detailLevel: "full",
  arrowheadScale: null,
  searchIndex: new Map(),
  searchMatches: new Set(),
  layoutById: new Map(),
  edgeRoutes: new Map(),
  layoutColumns: [],
  flowEdgesByColumn: [],
  overviewLayout: [],
  overviewEdges: [],
  flowGroupById: new Map(),
  maxColumn: 0,
  renderedRange: null,
  residualLedgerOpen: false,
  settingsOpen: false,
  settingsReturnFocus: null,
  modelInterfaceOpen: false,
  runtime: { connected: false, reachable: false, worker: null, busy: false, latestRun: null, generationTrace: null, causalSweep: null, mode: "daytona", recommendation: null, quantization: "none", nodeMetrics: new Map(), maxWriteNorm: 0, maxContributionShare: 0 },
  debug: { cases: [], currentCase: null, comparison: null, interventions: [], trace: null, inbox: [], selectedBenchmarkId: null, verification: null, runHistory: [], watchlist: new Map(), watchlistOpen: false, dirty: false }
};

elements.settingsAccountMount.append(elements.hfAccount);
elements.settingsWorkerMount.append(elements.runtimePanel);

const searchTextCache = new WeakMap();

function currentWatchlistStorageKey() {
  const source = state.model?.source ?? {};
  return watchlistStorageKey(source.modelId, source.revision ?? "main");
}

function watchlistValues() {
  return [...state.debug.watchlist.values()];
}

function persistWatchlist() {
  try {
    localStorage.setItem(currentWatchlistStorageKey(), JSON.stringify(watchlistValues()));
  } catch {
    // Case persistence remains available when browser storage is blocked.
  }
}

function restoreWatchlist(caseValues = null) {
  let stored = [];
  try {
    stored = JSON.parse(localStorage.getItem(currentWatchlistStorageKey()) || "[]");
  } catch {
    stored = [];
  }
  const values = normaliseWatchlist(caseValues ?? stored);
  state.debug.watchlist = new Map(values.map((item) => [item.nodeId, item]));
  renderWatchlist();
}

function watchedNode(nodeId) {
  return state.debug.watchlist.get(nodeId) ?? null;
}

function renderWatchlist() {
  elements.watchlistList.replaceChildren();
  const values = watchlistValues();
  elements.watchlistCount.textContent = String(values.length);
  if (!values.length) {
    const empty = document.createElement("p");
    empty.className = "watchlist-empty";
    empty.textContent = "Select a graph component and choose Watch to start a research list.";
    elements.watchlistList.append(empty);
  }
  values.forEach((item) => {
    const row = document.createElement("article");
    row.className = "watchlist-item";
    row.setAttribute("role", "listitem");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "watchlist-item-open";
    const identity = document.createElement("span");
    const label = document.createElement("strong");
    const evidence = document.createElement("small");
    const metric = observedNodeMetric(item.nodeId);
    label.textContent = item.label;
    evidence.textContent = `${item.evidence}${Number.isFinite(metric?.causalEffect) ? ` · Δ ${shortObservedValue(metric.causalEffect, true)}` : Number.isFinite(metric?.dla) ? ` · DLA ${shortObservedValue(metric.dla, true)}` : ""}`;
    identity.append(label, evidence);
    const note = document.createElement("p");
    note.textContent = item.note || "No annotation yet.";
    open.append(identity, note);
    open.addEventListener("click", () => {
      setWatchlistOpen(false);
      focusGraphNode(item.nodeId);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "watchlist-item-remove";
    remove.setAttribute("aria-label", `Remove ${item.label} from watchlist`);
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      state.debug.watchlist.delete(item.nodeId);
      persistWatchlist();
      renderWatchlist();
      state.renderedRange = null;
      if (state.model) renderFlowScene({ force: true });
      if (state.selectedId === item.nodeId) syncNodeAnnotationEditor();
      setDebugDirty();
    });
    row.append(open, remove);
    elements.watchlistList.append(row);
  });
  elements.watchlistButton.setAttribute("aria-label", `Graph watchlist, ${values.length} components`);
}

function setWatchlistOpen(open) {
  state.debug.watchlistOpen = Boolean(open) && Boolean(state.model);
  if (state.debug.watchlistOpen && state.residualLedgerOpen) setResidualLedgerOpen(false);
  elements.watchlistPanel.hidden = !state.debug.watchlistOpen;
  elements.watchlistButton.classList.toggle("active", state.debug.watchlistOpen);
  elements.watchlistButton.setAttribute("aria-expanded", String(state.debug.watchlistOpen));
  if (state.debug.watchlistOpen) renderWatchlist();
}

function selectedGraphNode() {
  if (!state.model || !state.selectedId) return null;
  return state.model.nodeById?.get(state.selectedId)
    ?? state.layout.find((entry) => entry.node.id === state.selectedId)?.node
    ?? null;
}

function syncNodeAnnotationEditor() {
  const node = selectedGraphNode();
  if (!node) return;
  const entry = watchedNode(node.id);
  elements.toggleWatchNodeButton.textContent = entry ? "Unwatch" : "Watch";
  elements.toggleWatchNodeButton.setAttribute("aria-pressed", String(Boolean(entry)));
  elements.nodeEvidenceSelect.value = entry?.evidence ?? "hypothesis";
  elements.nodeAnnotationInput.value = entry?.note ?? "";
  elements.nodeAnnotationStatus.textContent = entry ? "Watched" : "Not watched";
}

function writeSelectedNodeAnnotation({ toggle = false } = {}) {
  const node = selectedGraphNode();
  if (!node) return;
  const existing = watchedNode(node.id);
  if (toggle && existing) {
    state.debug.watchlist.delete(node.id);
  } else {
    state.debug.watchlist.set(node.id, {
      nodeId: node.id,
      label: node.name || node.path || node.id,
      evidence: elements.nodeEvidenceSelect.value,
      note: elements.nodeAnnotationInput.value.trim(),
      createdAt: existing?.createdAt ?? new Date().toISOString()
    });
  }
  persistWatchlist();
  renderWatchlist();
  syncNodeAnnotationEditor();
  state.renderedRange = null;
  renderFlowScene({ force: true });
  setDebugDirty();
}

function recordRun(run, options = {}) {
  if (!run) return;
  try {
    const snapshot = createRunSnapshot(run, options);
    state.debug.runHistory = appendRunHistory(state.debug.runHistory, snapshot);
    renderRunHistory();
    setDebugDirty();
  } catch {
    // Older persisted records may not have stable run identifiers.
  }
}

function runOptionLabel(run) {
  const time = Number.isNaN(Date.parse(run.createdAt)) ? "" : new Date(run.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return `${run.label}${time ? ` · ${time}` : ""}`;
}

function renderRunHistory() {
  const history = state.debug.runHistory;
  elements.runHistoryCount.textContent = `${history.length} run${history.length === 1 ? "" : "s"}`;
  elements.runHistoryList.replaceChildren();
  if (!history.length) {
    const empty = document.createElement("p");
    empty.className = "run-history-empty";
    empty.textContent = "Forward passes and generation traces will appear here.";
    elements.runHistoryList.append(empty);
  }
  history.slice(0, 8).forEach((run) => {
    const row = document.createElement("article");
    row.className = "run-history-item";
    row.setAttribute("role", "listitem");
    const heading = document.createElement("div");
    const label = document.createElement("strong");
    const context = document.createElement("small");
    const result = document.createElement("span");
    label.textContent = run.label;
    context.textContent = `${run.kind} · ${String(run.revision ?? "revision unknown").slice(0, 12)} · seed ${run.context?.seed ?? "—"}`;
    result.textContent = run.metric && Number.isFinite(run.metric.value)
      ? `${run.metric.name}: ${modelMetric(run.metric.value, 4)}`
      : run.output?.topToken ? `Top token: ${run.output.topToken}` : run.completion || "Recorded";
    heading.append(label, context);
    row.append(heading, result);
    elements.runHistoryList.append(row);
  });
  const previousLeft = elements.runDiffLeft.value;
  const previousRight = elements.runDiffRight.value;
  [elements.runDiffLeft, elements.runDiffRight].forEach((select) => {
    select.replaceChildren();
    history.forEach((run) => {
      const option = document.createElement("option");
      option.value = run.id;
      option.textContent = runOptionLabel(run);
      select.append(option);
    });
  });
  let leftId = history.some((run) => run.id === previousLeft) ? previousLeft : history[1]?.id ?? history[0]?.id ?? "";
  let rightId = history.some((run) => run.id === previousRight) ? previousRight : history[0]?.id ?? "";
  if (history.length > 1 && leftId === rightId) {
    rightId = history[0].id;
    leftId = history.find((run) => run.id !== rightId)?.id ?? history[1].id;
  }
  elements.runDiffLeft.value = leftId;
  elements.runDiffRight.value = rightId;
  elements.compareRunsButton.disabled = state.runtime.busy || history.length < 2;
}

function renderRunDiff() {
  const left = state.debug.runHistory.find((run) => run.id === elements.runDiffLeft.value);
  const right = state.debug.runHistory.find((run) => run.id === elements.runDiffRight.value);
  if (!left || !right || left.id === right.id) {
    elements.runDiffResult.hidden = false;
    elements.runDiffResult.textContent = "Choose two different snapshots.";
    return;
  }
  const diff = compareRunSnapshots(left, right);
  elements.runDiffResult.replaceChildren();
  const summary = document.createElement("div");
  summary.className = "debug-summary-grid";
  appendDebugSummary(summary, "Compatibility", diff.compatibility.metricComparable ? "Comparable" : "Descriptive", diff.compatibility.note);
  appendDebugSummary(summary, "Metric Δ", shortObservedValue(diff.metricDelta, true), "later − earlier");
  appendDebugSummary(summary, "Target probability Δ", Number.isFinite(diff.targetProbabilityDelta) ? `${shortObservedValue(diff.targetProbabilityDelta * 100, true)} pp` : "—", "later − earlier");
  appendDebugSummary(summary, "Entropy Δ", shortObservedValue(diff.entropyDelta, true), "nats · later − earlier");
  elements.runDiffResult.append(summary);
  if (diff.layers.length) {
    const chart = document.createElement("div");
    chart.className = "model-chart";
    renderLineChart(chart, [
      { label: "Residual norm Δ", values: diff.layers.map((layer) => layer.residualNormDelta) }
    ], { includeZero: true });
    const dlaChart = document.createElement("div");
    dlaChart.className = "model-chart";
    renderLineChart(dlaChart, [
      { label: "Residual DLA Δ", values: diff.layers.map((layer) => layer.residualDlaDelta) },
      { label: "Attention DLA Δ", values: diff.layers.map((layer) => layer.attentionDlaDelta) },
      { label: "MLP DLA Δ", values: diff.layers.map((layer) => layer.mlpDlaDelta) }
    ], { includeZero: true });
    elements.runDiffResult.append(chart, dlaChart);
  }
  const note = document.createElement("p");
  note.className = "debugger-science-note";
  note.textContent = "Snapshot differences are observational. They do not establish which component caused an output change.";
  elements.runDiffResult.append(note);
  elements.runDiffResult.hidden = false;
}

function setSettingsOpen(open) {
  const shouldOpen = Boolean(open);
  if (shouldOpen === state.settingsOpen) return;
  if (shouldOpen && state.modelInterfaceOpen) setModelInterfaceOpen(false);
  state.settingsOpen = shouldOpen;
  elements.settingsOverlay.hidden = !shouldOpen;
  elements.settingsButton.setAttribute("aria-expanded", String(shouldOpen));
  document.body.classList.toggle("settings-open", shouldOpen);
  if (shouldOpen) {
    state.settingsReturnFocus = document.activeElement;
    requestAnimationFrame(() => elements.settingsDrawer.focus());
  } else {
    state.settingsReturnFocus?.focus?.({ preventScroll: true });
    state.settingsReturnFocus = null;
  }
}

function modelRunReady() {
  const worker = state.runtime.worker;
  return Boolean(
    state.model
    && state.runtime.connected
    && state.runtime.reachable
    && worker?.modelLoaded
    && worker.modelId === state.model.source?.modelId
  );
}

function updateModelRunReadiness() {
  const ready = modelRunReady();
  const hasModel = Boolean(state.model);
  const worker = state.runtime.worker;
  elements.runModelButton.disabled = !hasModel;
  elements.runtimeForwardButton.disabled = !ready || state.runtime.busy;
  elements.runtimeCompareButton.disabled = !ready || state.runtime.busy;
  elements.runGenerationButton.disabled = !ready || state.runtime.busy;
  elements.runCausalSweepButton.disabled = !ready || state.runtime.busy || !state.runtime.latestRun;
  elements.compareRunsButton.disabled = state.runtime.busy || state.debug.runHistory.length < 2;
  elements.modelRunConnection.dataset.state = ready ? "ready" : state.runtime.reachable ? "waiting" : "offline";
  if (!hasModel) {
    elements.modelRunConnectionTitle.textContent = "No checkpoint open";
    elements.modelRunConnectionDetail.textContent = "Open a Hugging Face model before starting a prompt-conditioned run.";
    elements.modelRunStatus.textContent = "Open a model to begin.";
  } else if (!state.runtime.reachable) {
    elements.modelRunConnectionTitle.textContent = "No execution worker";
    elements.modelRunConnectionDetail.textContent = "Connect a Daytona GPU or local worker from Settings.";
    elements.modelRunStatus.textContent = "Connect an execution worker to begin.";
  } else if (!worker?.modelLoaded) {
    elements.modelRunConnectionTitle.textContent = `${worker?.accelerator ?? "Worker"} connected`;
    elements.modelRunConnectionDetail.textContent = `Load ${state.model.source.modelId} from Settings before running.`;
    elements.modelRunStatus.textContent = "The current model is not loaded into the worker.";
  } else if (worker.modelId !== state.model.source?.modelId) {
    elements.modelRunConnectionTitle.textContent = "Worker model does not match";
    elements.modelRunConnectionDetail.textContent = `Worker: ${worker.modelId}. Graph: ${state.model.source.modelId}.`;
    elements.modelRunStatus.textContent = "Load the current graph model from Settings.";
  } else {
    elements.modelRunConnectionTitle.textContent = `${worker.modelId} ready`;
    elements.modelRunConnectionDetail.textContent = `${worker.accelerator ?? worker.device ?? "Execution worker"} · ${worker.dtype ?? "automatic precision"}`;
    if (!state.runtime.busy && !state.runtime.latestRun && !elements.modelRunStatus.classList.contains("error")) elements.modelRunStatus.textContent = "Ready for one hooked forward pass.";
  }
  if (state.runtime.latestRun) populateDebugComponentControls(state.runtime.latestRun, state.debug.comparison);
}

function setModelInterfaceOpen(open) {
  const shouldOpen = Boolean(open) && Boolean(state.model);
  state.modelInterfaceOpen = shouldOpen;
  elements.modelInterface.hidden = !shouldOpen;
  elements.runModelButton.classList.toggle("active", shouldOpen);
  elements.runModelButton.setAttribute("aria-expanded", String(shouldOpen));
  if (!shouldOpen) return;
  setResidualLedgerOpen(false);
  setWatchlistOpen(false);
  closeInspector();
  elements.modelInterfaceSubtitle.textContent = state.model?.source?.modelId
    ? `${state.model.source.modelId} · last-token analysis`
    : "Run one forward pass and inspect the computation.";
  updateModelRunReadiness();
  requestAnimationFrame(() => (modelRunReady() ? elements.runtimePromptInput : elements.modelRunSettingsButton).focus());
}

function setAppView(view, { focus = true, updateHistory = true, fit = true } = {}) {
  const nextView = view === "workspace" ? "workspace" : "landing";
  if (nextView !== "workspace" && state.settingsOpen) setSettingsOpen(false);
  document.body.dataset.view = nextView;
  elements.landing.setAttribute("aria-hidden", String(nextView !== "landing"));
  if (updateHistory) {
    const route = nextView === "workspace" ? "#workspace" : "#home";
    history.replaceState(null, "", route);
  }
  if (nextView === "workspace") {
    requestAnimationFrame(() => {
      handleViewportResize();
      if (state.model && fit) fitGraph();
      if (focus) elements.hfModelInput.focus({ preventScroll: true });
    });
  } else if (focus) {
    document.querySelector(".landing-nav-cta")?.focus({ preventScroll: true });
  }
}

function appViewFromHash(hash = location.hash) {
  if (hash === "#workspace") return "workspace";
  return "landing";
}

function openLandingTutorial() {
  setAppView("landing", { focus: false, updateHistory: false });
  history.replaceState(null, "", "#tutorial");
  requestAnimationFrame(() => elements.tutorial.scrollIntoView({ block: "start" }));
}

function svgElement(tag, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
  return element;
}

function truncate(text, max = 28) {
  const value = String(text);
  return value.length <= max ? value : `${value.slice(0, max - 1)}…`;
}

function nodeSearchText(node) {
  const cached = searchTextCache.get(node);
  if (cached) return cached;
  const text = node.searchText ?? `${node.name} ${node.path} ${node.type}`.toLowerCase();
  searchTextCache.set(node, text);
  return text;
}

function calculateLayout() {
  const layout = state.model.nodes.map((node) => {
    return {
      node,
      depth: node.layout.depth,
      parent: null,
      x: node.layout.x,
      y: node.layout.y
    };
  });
  state.layout = layout;
  state.layoutById = new Map(layout.map((entry) => [entry.node.id, entry]));
  state.maxColumn = state.model.layout.maxColumn;
  state.layoutColumns = Array.from({ length: state.maxColumn + 1 }, () => []);
  layout.forEach((entry) => state.layoutColumns[entry.depth].push(entry));
  state.flowEdgesByColumn = Array.from({ length: state.maxColumn + 1 }, () => []);
  state.model.edges.forEach((edge) => {
    const source = state.layoutById.get(edge.from);
    const target = state.layoutById.get(edge.to);
    if (!source || !target) return;
    state.flowEdgesByColumn[Math.min(source.depth, target.depth)].push(edge);
  });
  state.edgeRoutes = routeGraphEdges(state.model.edges, state.layoutById, {
    nodeWidth: NODE_WIDTH,
    nodeHeight: NODE_HEIGHT,
    tensorRank: (entry) => multidimensionalRank(entry.node)
  });
  const overviewIds = new Set(state.model.layout.overviewNodeIds);
  state.overviewLayout = layout.filter((entry) => overviewIds.has(entry.node.id));
  state.overviewEdges = state.model.edges.filter((edge) => overviewIds.has(edge.from) && overviewIds.has(edge.to));
  state.flowGroupById = new Map(state.model.groups.map((group) => [group.id, group]));
  state.bounds = state.model.layout.bounds;
}

function detailLevelForScale(scale) {
  return scale < 0.17 ? "overview" : scale < 0.48 ? "compact" : "full";
}

function updateDetailLevel(scale) {
  const detailLevel = detailLevelForScale(scale);
  const changed = detailLevel !== state.detailLevel || elements.svg.dataset.detail !== detailLevel;
  if (!changed) return false;
  state.detailLevel = detailLevel;
  elements.svg.dataset.detail = detailLevel;
  elements.detailLevelHint.textContent = detailLevel === "overview"
    ? "Overview · zoom in for circuits"
    : detailLevel === "compact"
      ? "Compact circuits · zoom in for equations"
      : "Full equations";
  return true;
}

function detailLevelLabel() {
  return state.detailLevel === "overview"
    ? "Overview · zoom in for circuits"
    : state.detailLevel === "compact"
      ? "Compact circuits · zoom in for equations"
      : "Full equations";
}

function updateZoomReadout() {
  const percent = Math.max(1, Math.round(state.transform.scale * 100));
  elements.zoomReadout.textContent = `${percent}%`;
  elements.zoomReadout.setAttribute("aria-label", `Current zoom ${percent} percent; fit graph`);
}

function updateWorldGrid(x, y, scale) {
  // Keep the paper grid attached to model-space while avoiding sub-pixel density
  // at checkpoint-scale overview zooms.
  let worldSpacing = 40;
  while (worldSpacing * scale < 14) worldSpacing *= 5;
  const gridSize = worldSpacing * scale;
  const gridX = ((x % gridSize) + gridSize) % gridSize;
  const gridY = ((y % gridSize) + gridSize) % gridSize;
  elements.viewport.style.setProperty("--graph-grid-size", `${gridSize}px`);
  elements.viewport.style.setProperty("--graph-grid-x", `${gridX}px`);
  elements.viewport.style.setProperty("--graph-grid-y", `${gridY}px`);
}

function updateArrowheadScale(scale) {
  if (state.arrowheadScale === scale) return;
  state.arrowheadScale = scale;
  const markerSize = ARROWHEAD_SCREEN_SIZE / scale;
  elements.arrowheads.forEach((arrowhead) => {
    arrowhead.setAttribute("markerWidth", String(markerSize));
    arrowhead.setAttribute("markerHeight", String(markerSize));
  });
}

function flushCamera() {
  const { x, y, scale } = state.transform;
  elements.camera.setAttribute("transform", `translate(${x} ${y}) scale(${scale})`);
  updateWorldGrid(x, y, scale);
  updateArrowheadScale(scale);
  updateZoomReadout();
  const detailChanged = updateDetailLevel(scale);
  if (state.model) renderFlowScene({ force: detailChanged });
  state.cameraFrame = null;
}

function applyCamera() {
  if (state.cameraFrame !== null) return;
  state.cameraFrame = requestAnimationFrame(flushCamera);
}

function markCameraMoving() {
  elements.svg.classList.add("moving");
  if (state.motionTimer !== null) clearTimeout(state.motionTimer);
  state.motionTimer = setTimeout(() => {
    elements.svg.classList.remove("moving");
    state.motionTimer = null;
  }, 110);
}

function nodeMeta(node) {
  const shape = node.shape ? formatShape(node.shape) : "activation";
  const elements = node.totalElements ?? node.totalParameters ?? 0;
  return elements ? `${shape} · ${formatCount(elements)} elements` : shape;
}

function observedNodeMetric(nodeId) {
  return state.runtime.nodeMetrics.get(nodeId) ?? null;
}

function shortObservedValue(value, signed = false) {
  if (!Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  const formatted = absolute >= 100 ? absolute.toFixed(0) : absolute >= 10 ? absolute.toFixed(1) : absolute.toFixed(2);
  return `${signed && value > 0 ? "+" : signed && value < 0 ? "−" : ""}${formatted}`;
}

function multidimensionalRank(node) {
  return Math.max(
    0,
    ...(node.tensors ?? [])
      .map((tensor) => Array.isArray(tensor.shape) ? tensor.shape.length : 0)
      .filter((rank) => rank >= 2)
  );
}

function circuitFamily(node) {
  return node.circuitFamily ?? "circuit-general";
}

function isResidualStateNode(node) {
  return node.id === "residual_0" || node.id.endsWith("_mlp_residual");
}

function makeTensorStack(entry) {
  const rank = multidimensionalRank(entry.node);
  const depth = tensorStackDepth(rank);
  if (!depth) return null;
  const stack = svgElement("g", {
    class: `graph-node tensor-stack ${entry.node.kind} ${circuitFamily(entry.node)}`,
    transform: `translate(${entry.x} ${entry.y})`,
    "data-node-id": entry.node.id,
    "aria-hidden": "true"
  });
  for (let layer = depth; layer >= 1; layer -= 1) {
    const offset = layer * 6;
    stack.append(svgElement("rect", {
      class: "node-body tensor-stack-layer",
      x: offset,
      y: offset,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      rx: 9
    }));
  }
  return stack;
}

function appendTensorGlyph(group, rank) {
  if (rank < 2) return;
  const glyph = svgElement("g", {
    class: `tensor-glyph${rank > 3 ? " stacked" : ""}`,
    "aria-hidden": "true"
  });
  const title = svgElement("title");
  title.textContent = `${rank}D tensor weights`;
  glyph.append(title);

  if (rank >= 3) {
    const sliceCount = Math.min(4, Math.max(3, rank));
    for (let layer = sliceCount; layer >= 1; layer -= 1) {
      const offset = layer * 1.5;
      glyph.append(svgElement("polygon", {
        class: "tensor-glyph-slice",
        points: `${211 + offset},${48 - offset} ${227 + offset},${48 - offset} ${227 + offset},${62 - offset} ${211 + offset},${62 - offset}`,
        opacity: String(0.08 + (sliceCount - layer) * 0.055)
      }));
    }
  }
  glyph.append(
    svgElement("polygon", { class: "tensor-glyph-top", points: "211,48 217,43 233,43 227,48" }),
    svgElement("polygon", { class: "tensor-glyph-side", points: "227,48 233,43 233,57 227,62" }),
    svgElement("polygon", { class: "tensor-glyph-front", points: "211,48 227,48 227,62 211,62" })
  );
  group.append(glyph);
}

function makeNode(entry) {
  const { node, x, y } = entry;
  const tensorRank = multidimensionalRank(node);
  const observation = observedNodeMetric(node.id);
  const watched = Boolean(watchedNode(node.id));
  const divergent = Number.isFinite(observation?.divergence);
  const causal = Number.isFinite(observation?.causalEffect);
  const group = svgElement("g", {
    class: `graph-node ${node.kind} ${circuitFamily(node)}${isResidualStateNode(node) ? " residual-state-node" : ""}${observation ? " observed-node" : ""}${divergent ? " divergent-node" : ""}${causal ? " causal-node" : ""}${watched ? " watched-node" : ""}${state.selectedId === node.id ? " selected" : ""}`,
    transform: `translate(${x} ${y})`,
    "data-node-id": node.id,
    tabindex: state.selectedId ? (state.selectedId === node.id ? "0" : "-1") : (node.id === "residual_0" ? "0" : "-1"),
    role: "button",
    "aria-pressed": String(state.selectedId === node.id),
    "aria-label": `${node.kind} ${node.name}${watched ? ", on watchlist" : ""}${node.type ? `, ${node.type}` : ""}${node.tensors?.length ? `, ${node.tensors.length} checkpoint tensors` : ""}${observation ? `, observed norm ${shortObservedValue(observation.norm)}, DLA ${shortObservedValue(observation.dla, true)}${Number.isFinite(observation.share) ? `, relative attribution share ${modelMetric(observation.share * 100, 1)} percent` : ""}` : ""}`
  });
  if (divergent) group.style.setProperty("--debug-strength", String(Math.max(0.08, Math.min(1, observation.divergence / Math.max(state.debug.maxDivergence ?? 1, 1e-12)))));

  const body = svgElement("rect", {
    class: "node-body",
    width: NODE_WIDTH,
    height: NODE_HEIGHT,
    rx: 9
  });
  group.append(body);

  const kicker = svgElement("text", { class: "node-kicker", x: 14, y: 17 });
  kicker.textContent = node.type;
  group.append(kicker);

  const title = svgElement("text", { class: "node-title", x: 14, y: 38 });
  title.textContent = truncate(node.name, tensorRank >= 2 ? 24 : 28);
  group.append(title);

  if (node.formula) {
    const formula = svgElement("text", { class: "node-formula", x: 14, y: 57 });
    formula.textContent = truncate(node.formula, tensorRank >= 2 ? 29 : 39);
    group.append(formula);
  }

  const meta = svgElement("text", { class: "node-meta", x: 14, y: node.formula ? 75 : 61 });
  meta.textContent = truncate(nodeMeta(node), tensorRank >= 2 ? 31 : 38);
  group.append(meta);
  appendTensorGlyph(group, tensorRank);

  if (watched) {
    group.append(svgElement("circle", { class: "watched-node-badge", cx: 9, cy: 9, r: 6 }));
    const watchedText = svgElement("text", { class: "watched-node-badge-text", x: 9, y: 12, "text-anchor": "middle" });
    watchedText.textContent = "★";
    group.append(watchedText);
  }

  if (observation) {
    const badgeX = isResidualStateNode(node) ? NODE_WIDTH - 112 : NODE_WIDTH - 89;
    group.append(svgElement("rect", { class: "observed-node-badge", x: badgeX, y: 9, width: 74, height: 18, rx: 4 }));
    const metricText = svgElement("text", { class: "observed-node-badge-text", x: badgeX + 37, y: 21, "text-anchor": "middle" });
    metricText.textContent = Number.isFinite(observation.causalEffect)
      ? `Δ ${shortObservedValue(observation.causalEffect, true)}`
      : Number.isFinite(observation.divergence)
        ? `diff ${shortObservedValue(observation.divergence)}`
      : observation.kind !== "residual" && Number.isFinite(observation.share)
      ? `${observation.share > 0 ? "+" : observation.share < 0 ? "−" : ""}${modelMetric(Math.abs(observation.share) * 100, 1)}%`
      : Number.isFinite(observation.dla)
        ? `DLA ${shortObservedValue(observation.dla, true)}`
      : `‖·‖ ${shortObservedValue(observation.norm)}`;
    group.append(metricText);
  }

  if (isResidualStateNode(node)) {
    group.append(svgElement("rect", {
      class: "residual-ledger-badge",
      x: NODE_WIDTH - 31,
      y: 10,
      width: 18,
      height: 17,
      rx: 8.5
    }));
    const ledgerBadgeText = svgElement("text", {
      class: "residual-ledger-badge-text",
      x: NODE_WIDTH - 22,
      y: 21.5,
      "text-anchor": "middle"
    });
    ledgerBadgeText.textContent = "Σ";
    group.append(ledgerBadgeText);
  }

  if (node.repeat > 1) {
    const repeatWidth = 32 + String(node.repeat).length * 5;
    group.append(
      svgElement("rect", {
        class: "node-repeat-pill",
        x: NODE_WIDTH - repeatWidth - 13,
        y: 10,
        width: repeatWidth,
        height: 17,
        rx: 8.5
      })
    );
    const repeatText = svgElement("text", {
      class: "node-repeat-text",
      x: NODE_WIDTH - repeatWidth / 2 - 13,
      y: 21.5,
      "text-anchor": "middle"
    });
    repeatText.textContent = `× ${node.repeat}`;
    group.append(repeatText);
  }

  return group;
}

function makeFlowEdge(edge, byId, edgeRoutes) {
  const source = byId.get(edge.from);
  const target = byId.get(edge.to);
  const isResidualRail = edge.kind === "residual-stream";
  const circuitKind = edge.circuitKind;
  const group = svgElement("g", {
    class: `flow-edge-group circuit-edge-group-${circuitKind}${isResidualRail ? " residual-rail-group" : ""}`,
    "data-edge-from": edge.from,
    "data-edge-to": edge.to,
    "data-edge-kind": circuitKind
  });
  const title = svgElement("title");
  title.textContent = `${source.node.name} → ${target.node.name}${edge.label ? ` · ${edge.label}` : ""}`;
  const arrowhead = edge.arrowhead;
  const geometry = edgeRoutes.get(edge) ?? routeGraphEdge(
    edge,
    { x: source.x, y: source.y, tensorRank: multidimensionalRank(source.node) },
    { x: target.x, y: target.y, tensorRank: multidimensionalRank(target.node) },
    { nodeWidth: NODE_WIDTH, nodeHeight: NODE_HEIGHT }
  );
  const observedWrite = observedNodeMetric(edge.from);
  const observedRatio = observedWrite && Number.isFinite(observedWrite.share) && state.runtime.maxContributionShare > 0
    ? Math.min(1, Math.abs(observedWrite.share) / state.runtime.maxContributionShare)
    : observedWrite && Number.isFinite(observedWrite.norm) && state.runtime.maxWriteNorm > 0
      ? Math.min(1, observedWrite.norm / state.runtime.maxWriteNorm)
      : null;
  const edgePath = svgElement("path", {
    class: `graph-edge flow-edge circuit-edge-${circuitKind}${isResidualRail ? " residual-rail" : ""}${edge.feedback ? " feedback" : ""}`,
    d: geometry.path,
    "marker-end": `url(#arrowhead-${arrowhead})`
  });
  if (observedRatio !== null && ["attention", "mlp"].includes(circuitKind)) {
    edgePath.classList.add("observed-edge");
    edgePath.style.strokeWidth = String(1.2 + observedRatio * 3.2);
    edgePath.style.opacity = String(0.42 + observedRatio * 0.58);
  }
  group.append(title, edgePath);
  if (edge.label) {
    const label = svgElement("text", {
      class: "edge-label",
      x: geometry.labelPosition.x,
      y: geometry.labelPosition.y,
      "text-anchor": "middle"
    });
    label.textContent = edge.label;
    group.append(label);
  }
  return group;
}

function renderFlowGroups(layoutEntries, { showGroups = true, showResidualLabel = false } = {}) {
  const fragment = document.createDocumentFragment();
  if (showGroups) {
    const visibleGroupIds = new Set(layoutEntries.map((entry) => entry.node.group).filter(Boolean));
    visibleGroupIds.forEach((groupId) => {
      const group = state.flowGroupById.get(groupId);
      if (!group) return;
      const { x, y, width, height } = group.bounds;
      const frameY = y - TRANSFORMER_BLOCK_HEADER_RISE;
      const frameHeight = height + TRANSFORMER_BLOCK_HEADER_RISE;
      const headerX = x + TRANSFORMER_BLOCK_HEADER_INSET;
      const headerY = frameY + 8;
      const headerWidth = Math.min(
        TRANSFORMER_BLOCK_HEADER_WIDTH,
        width - TRANSFORMER_BLOCK_HEADER_INSET * 2
      );
      const container = svgElement("g", {
        class: "transformer-block-container",
        "data-group-id": group.id,
        "aria-hidden": "true"
      });
      const body = svgElement("rect", {
        class: "flow-group-body transformer-block-body",
        x,
        y: frameY,
        width,
        height: frameHeight,
        rx: 13
      });
      const header = svgElement("rect", {
        class: "transformer-block-header",
        x: headerX,
        y: headerY,
        width: headerWidth,
        height: TRANSFORMER_BLOCK_HEADER_HEIGHT,
        rx: 4
      });
      const label = svgElement("text", {
        class: "flow-group-label transformer-block-label",
        x: headerX + 10,
        y: headerY + 15
      });
      label.textContent = `TRANSFORMER BLOCK · ${group.label.toUpperCase()}`;
      container.append(body, header, label);
      group.subgroups.forEach((subgroup) => {
        const bounds = subgroup.bounds;
        const subgroupBody = svgElement("rect", {
          class: `flow-subgroup-body flow-subgroup-${subgroup.kind}`,
          x: bounds.x,
          y: bounds.y,
          width: bounds.width,
          height: bounds.height,
          rx: 7
        });
        const subgroupLabel = svgElement("text", {
          class: `flow-subgroup-label flow-subgroup-label-${subgroup.kind}`,
          x: bounds.x + 8,
          y: bounds.y + 14
        });
        subgroupLabel.textContent = subgroup.label;
        container.append(subgroupBody, subgroupLabel);
      });
      fragment.append(container);
    });
  }

  if (showResidualLabel) {
    const label = svgElement("text", {
      class: "residual-stream-label",
      x: 0,
      y: -43
    });
    label.textContent = `RESIDUAL STREAM LEDGER  ·  h₀ → h₁ → … → h${state.model.groups.length}  ·  Σ ATTENTION + MLP WRITES`;
    fragment.append(label);
  }
  elements.groupLayer.replaceChildren(fragment);
}

function visibleColumnWindow() {
  const stride = NODE_WIDTH + X_GAP;
  const scale = Math.max(state.transform.scale, 0.01);
  const viewportWidth = elements.viewport.clientWidth || 1;
  const worldLeft = -state.transform.x / scale;
  const worldRight = (viewportWidth - state.transform.x) / scale;
  const clampColumn = (column) => Math.max(0, Math.min(state.maxColumn, column));
  return {
    min: clampColumn(Math.floor((worldLeft - NODE_WIDTH) / stride)),
    max: clampColumn(Math.ceil(worldRight / stride))
  };
}

function renderFlowScene({ force = false, all = false } = {}) {
  const overview = !all && state.detailLevel === "overview";
  const visible = overview || all ? { min: 0, max: state.maxColumn } : visibleColumnWindow();
  const current = state.renderedRange;
  if (
    !force &&
    !all &&
    current?.detail === state.detailLevel &&
    current.min <= Math.max(0, visible.min - 1) &&
    current.max >= Math.min(state.maxColumn, visible.max + 1)
  ) return;

  const buffer = overview || all ? 0 : 3;
  const range = {
    min: Math.max(0, visible.min - buffer),
    max: Math.min(state.maxColumn, visible.max + buffer)
  };
  const layoutEntries = overview ? [...state.overviewLayout] : [];
  if (overview && state.searchMatches.size) {
    const overviewIds = new Set(layoutEntries.map((entry) => entry.node.id));
    state.searchMatches.forEach((nodeId) => {
      const entry = state.layoutById.get(nodeId);
      if (entry && !overviewIds.has(nodeId)) layoutEntries.push(entry);
    });
  }
  if (!overview) {
    for (let column = range.min; column <= range.max; column += 1) {
      layoutEntries.push(...(state.layoutColumns[column] ?? []));
    }
  }
  const renderedIds = new Set(layoutEntries.map((entry) => entry.node.id));
  const edges = overview ? state.overviewEdges : [];
  if (!overview) {
    for (let column = range.min; column <= range.max; column += 1) {
      (state.flowEdgesByColumn[column] ?? []).forEach((edge) => {
        if (renderedIds.has(edge.from) && renderedIds.has(edge.to)) edges.push(edge);
      });
    }
  }

  renderFlowGroups(layoutEntries, {
    showGroups: !overview,
    showResidualLabel: overview || all
  });
  const stackFragment = document.createDocumentFragment();
  layoutEntries.forEach((entry) => {
    const stack = makeTensorStack(entry);
    if (stack) stackFragment.append(stack);
  });
  elements.tensorStackLayer.replaceChildren(stackFragment);
  const edgeFragment = document.createDocumentFragment();
  edges.forEach((edge) => edgeFragment.append(makeFlowEdge(edge, state.layoutById, state.edgeRoutes)));
  elements.edgeLayer.replaceChildren(edgeFragment);
  const nodeFragment = document.createDocumentFragment();
  layoutEntries.forEach((entry) => nodeFragment.append(makeNode(entry)));
  elements.nodeLayer.replaceChildren(nodeFragment);
  state.renderedRange = { ...range, detail: state.detailLevel };
  elements.detailLevelHint.textContent = `${detailLevelLabel()} · ${layoutEntries.length} nodes mounted`;

  applySearch();
}

function renderGraph({ fit = false, initial = false } = {}) {
  if (!state.model) return;
  calculateLayout();
  state.searchIndex.clear();
  state.renderedRange = null;

  if (initial) setReadableTransform();
  else if (fit) setFitTransform();
  updateDetailLevel(state.transform.scale);

  renderFlowScene({ force: true });
  applyCamera();
}

function numericLedgerValue(value) {
  if (Number.isFinite(value)) return value;
  return value && Number.isFinite(value.value) ? value.value : null;
}

function formatSignedLedgerValue(value) {
  const numeric = numericLedgerValue(value);
  if (numeric === null) return "—";
  const absolute = Math.abs(numeric);
  const formatted = absolute >= 100
    ? absolute.toFixed(1)
    : absolute >= 10
      ? absolute.toFixed(2)
      : absolute.toFixed(3);
  return `${numeric > 0 ? "+" : numeric < 0 ? "−" : ""}${formatted}`;
}

function formatLedgerMagnitude(value) {
  const numeric = numericLedgerValue(value);
  if (numeric === null) return "—";
  const absolute = Math.abs(numeric);
  return absolute.toFixed(absolute >= 10 ? 2 : 3);
}

function ledgerValueElement(value, label = "Not measured") {
  const numeric = numericLedgerValue(value);
  const element = document.createElement("strong");
  element.className = `residual-ledger-value ${numeric === null ? "unmeasured" : numeric > 0 ? "positive" : numeric < 0 ? "negative" : "neutral"}`;
  element.textContent = formatSignedLedgerValue(value);
  element.title = numeric === null ? label : `Signed direct logit attribution ${formatSignedLedgerValue(value)}`;
  return element;
}

function residualLedgerStateForNode(nodeId) {
  const states = state.model?.residualLedger?.states ?? [];
  return states.find((entry) => (
    entry.id === nodeId || entry.writes?.some((write) => write.sourceNodeId === nodeId || write.targetNodeId === nodeId)
  ));
}

function syncResidualLedgerSelection() {
  const selectedState = residualLedgerStateForNode(state.selectedId);
  elements.residualLedgerList.querySelectorAll(".residual-ledger-row").forEach((button) => {
    const selected = button.dataset.nodeId === selectedState?.id;
    button.classList.toggle("selected", selected);
    if (selected) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  });
}

function renderResidualLedger() {
  const ledger = state.model?.residualLedger;
  elements.residualLedgerList.replaceChildren();
  if (!ledger?.states?.length) {
    elements.residualLedgerPosition.textContent = "Unavailable";
    elements.residualLedgerMetric.textContent = "DLA · unavailable";
    elements.residualLedgerNote.textContent = "This graph does not include residual-stream accounting.";
    return;
  }

  elements.residualLedgerPosition.textContent = `${ledger.position?.label ?? "Position t"} · ${ledger.position?.status ?? "symbolic"}`;
  const metricStatus = String(ledger.metric?.status ?? "not-measured").replaceAll("-", " ");
  elements.residualLedgerMetric.textContent = `DLA · ${metricStatus}`;
  elements.residualLedgerNote.textContent = ledger.measurementNote ?? "";

  const fragment = document.createDocumentFragment();
  ledger.states.forEach((ledgerState) => {
    const item = document.createElement("div");
    item.className = "residual-ledger-item";
    item.setAttribute("role", "listitem");

    const button = document.createElement("button");
    button.className = "residual-ledger-row";
    button.type = "button";
    button.dataset.nodeId = ledgerState.id;
    button.setAttribute("aria-label", `${ledgerState.label}. ${ledgerState.equation}. Direct logit attribution ${formatSignedLedgerValue(ledgerState.directLogitAttribution)}`);

    const heading = document.createElement("div");
    heading.className = "residual-ledger-state";
    const identity = document.createElement("span");
    const layer = document.createElement("small");
    layer.textContent = ledgerState.layer === null ? "Embedding" : `Block ${ledgerState.layer}`;
    const name = document.createElement("b");
    name.textContent = ledgerState.state;
    identity.append(layer, name);
    const stateMetric = document.createElement("span");
    stateMetric.className = "residual-ledger-state-metric";
    const norm = document.createElement("small");
    norm.textContent = `‖${ledgerState.state}‖ ${formatLedgerMagnitude(ledgerState.activationNorm)}`;
    stateMetric.append(norm, ledgerValueElement(ledgerState.directLogitAttribution));
    heading.append(identity, stateMetric);

    const equation = document.createElement("code");
    equation.className = "residual-ledger-equation";
    equation.textContent = ledgerState.equation;

    const writes = document.createElement("div");
    writes.className = "residual-ledger-writes";
    ledgerState.writes?.forEach((write, index) => {
      const writeRow = document.createElement("span");
      writeRow.className = `residual-ledger-write ${write.kind}`;
      const writeIdentity = document.createElement("span");
      const marker = document.createElement("i");
      marker.setAttribute("aria-hidden", "true");
      marker.textContent = index || ledgerState.stage !== "embedding" ? "+" : "=";
      const writeLabel = document.createElement("span");
      writeLabel.textContent = `${write.label} · ${write.symbol}`;
      const writeNorm = document.createElement("small");
      writeNorm.textContent = `‖${write.symbol}‖ ${formatLedgerMagnitude(write.value)}`;
      writeIdentity.append(marker, writeLabel, writeNorm);
      writeRow.append(writeIdentity, ledgerValueElement(write.directLogitAttribution));
      writes.append(writeRow);
    });

    button.append(heading, equation, writes);
    item.append(button);
    fragment.append(item);
  });
  elements.residualLedgerList.append(fragment);
  syncResidualLedgerSelection();
}

function setResidualLedgerOpen(open) {
  state.residualLedgerOpen = Boolean(open) && Boolean(state.model);
  if (state.residualLedgerOpen && state.debug.watchlistOpen) setWatchlistOpen(false);
  elements.residualLedger.hidden = !state.residualLedgerOpen;
  elements.residualLedgerButton.classList.toggle("active", state.residualLedgerOpen);
  elements.residualLedgerButton.setAttribute("aria-expanded", String(state.residualLedgerOpen));
  if (state.residualLedgerOpen) renderResidualLedger();
}

function focusGraphNode(nodeId) {
  const entry = state.layoutById.get(nodeId);
  if (!entry) return;
  const scale = Math.max(0.54, state.transform.scale);
  state.transform.scale = scale;
  state.transform.x = elements.viewport.clientWidth / 2 - (entry.x + NODE_WIDTH / 2) * scale;
  state.transform.y = elements.viewport.clientHeight / 2 - (entry.y + NODE_HEIGHT / 2) * scale;
  state.renderedRange = null;
  applyCamera();
  setTimeout(() => {
    if (!state.model) return;
    renderFlowScene({ force: true });
    selectNode(entry.node);
    const graphNode = [...elements.nodeLayer.querySelectorAll(".graph-node")]
      .find((element) => element.dataset.nodeId === nodeId);
    graphNode?.focus({ preventScroll: true });
  }, 34);
}

function updateSummary() {
  const { stats } = state.model;
  const resolver = state.model.resolver ?? {};
  const checkpointElements = stats.checkpointElements;
  const checkpointTensors = stats.checkpointTensors;
  const topology = state.model.forwardTopology ?? {};
  const validation = state.model.validation ?? {};
  elements.modelTitle.textContent = state.model.name;
  elements.modelType.textContent = state.model.description || state.model.type;
  elements.totalParameters.textContent = formatCount(checkpointElements);
  elements.totalParameters.title = Number.isFinite(checkpointElements)
    ? `${new Intl.NumberFormat().format(checkpointElements)} exact stored elements; ${new Intl.NumberFormat().format(stats.recognizedBufferElements ?? 0)} belong to recognized buffers`
    : "Exact checkpoint element counts are unavailable at this resolver tier.";
  const recognizedBufferElements = stats.recognizedBufferElements ?? 0;
  elements.checkpointBreakdown.textContent = !Number.isFinite(checkpointElements)
    ? `${resolver.label ?? "Partial metadata"} · exact elements unavailable`
    : recognizedBufferElements
    ? `${formatCount(Math.max(0, checkpointElements - recognizedBufferElements))} parameter-like · ${formatCount(recognizedBufferElements)} buffers`
    : "Parameter-like storage · trainability unknown";
  elements.moduleCount.textContent = new Intl.NumberFormat().format(stats.modules);
  elements.unitLabel.textContent = "Operations";
  elements.memoryEstimate.textContent = formatBytes(stats.totalBytes);
  elements.memoryEstimate.title = Number.isFinite(stats.totalBytes)
    ? `${new Intl.NumberFormat().format(stats.totalBytes)} checkpoint bytes reported by ${resolver.label?.toLowerCase() ?? "the resolver"}`
    : "Checkpoint byte size is unavailable.";
  elements.trainablePercent.textContent = Number.isFinite(checkpointTensors) ? new Intl.NumberFormat().format(checkpointTensors) : "—";
  elements.trainablePercent.title = Number.isFinite(checkpointTensors)
    ? `${checkpointTensors} checkpoint tensor names; ${stats.bufferTensors ?? 0} recognized buffers`
    : "Tensor names are unavailable in the configuration scaffold.";
  elements.modelEvidence.hidden = false;
  elements.modelEvidence.classList.toggle("is-scaffold", resolver.tier !== "checkpoint-mapped" || topology.status === "scaffold");
  elements.validationStatus.textContent = resolver.label ?? (validation.status === "verified" ? "Graph internally validated" : "Graph partially validated");
  const topologyLabel = String(topology.residual ?? "topology unresolved").replaceAll("-", " ");
  const positionLabel = String(topology.positionKind ?? "position unresolved").replaceAll("-", " ");
  elements.topologySummary.textContent = `${topologyLabel} · ${positionLabel} · ${topology.confidence ?? "unknown"} confidence`;
  elements.modelEvidence.title = [topology.evidence, validation.limitations].filter(Boolean).join(" — ");
  elements.svg.setAttribute("aria-label", `${state.model.name} autoregressive transformer graph with ${stats.modules} operations and ${Number.isFinite(checkpointTensors) ? checkpointTensors : "unresolved"} checkpoint tensors; ${topologyLabel}`);
}

function loadGraph(graph) {
  const caseModel = state.debug.currentCase?.model;
  const caseMatchesGraph = Boolean(
    caseModel?.modelId === graph.source?.modelId
    && (caseModel?.revision ?? "main") === (graph.source?.revision ?? "main")
  );
  state.model = graph;
  if (!caseMatchesGraph) {
    state.runtime.latestRun = null;
    state.runtime.generationTrace = null;
    state.runtime.causalSweep = null;
  }
  state.runtime.nodeMetrics = new Map();
  state.runtime.maxWriteNorm = 0;
  state.runtime.maxContributionShare = 0;
  elements.runtimeResult.hidden = true;
  elements.modelRunStatus.classList.remove("error");
  state.model.nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  restoreWatchlist(caseMatchesGraph ? state.debug.currentCase?.watchlist ?? null : null);
  state.selectedId = null;
  state.searchIndex.clear();
  state.searchMatches.clear();
  state.searchCursor = -1;
  elements.searchInput.value = "";
  elements.searchCount.textContent = "";
  elements.searchShell.classList.remove("has-query", "no-results");
  elements.inspector.hidden = true;
  elements.emptyState.hidden = true;
  setAppView("workspace", { focus: false, fit: false });
  setGraphControlsEnabled(true);
  elements.runtimeLoadButton.disabled = !state.runtime.reachable || state.runtime.busy;
  updateModelRunReadiness();
  updateSummary();
  renderResidualLedger();
  setResidualLedgerOpen(state.residualLedgerOpen);
  renderGraph({ initial: true });
}

function setReadableTransform() {
  const viewportWidth = elements.viewport.clientWidth;
  const viewportHeight = elements.viewport.clientHeight;
  if (!viewportWidth || !viewportHeight) return false;
  const groupBounds = state.model.groups?.[0]?.bounds;
  const circuitTop = Math.min(0, groupBounds?.y ?? 0);
  const circuitHeight = Math.max(NODE_HEIGHT, (groupBounds?.height ?? state.bounds.height) + circuitTop);
  const availableHeight = Math.max(100, viewportHeight - 54);
  const scale = Math.min(0.72, Math.max(0.54, availableHeight / circuitHeight));
  state.transform = {
    scale,
    x: PAGE_PADDING,
    y: (viewportHeight - circuitHeight * scale) / 2 - circuitTop * scale
  };
  return true;
}

function setFitTransform() {
  const viewportWidth = elements.viewport.clientWidth;
  const viewportHeight = elements.viewport.clientHeight;
  if (!viewportWidth || !viewportHeight) return false;

  const availableWidth = Math.max(100, viewportWidth - PAGE_PADDING * 2);
  const availableHeight = Math.max(100, viewportHeight - PAGE_PADDING * 2);
  const scale = Math.min(
    1.15,
    availableWidth / Math.max(state.bounds.width, 1),
    availableHeight / Math.max(state.bounds.height, 1)
  );

  state.transform = {
    scale,
    x: (viewportWidth - state.bounds.width * scale) / 2,
    y: (viewportHeight - state.bounds.height * scale) / 2
  };
  return true;
}

function fitGraph() {
  if (!state.model || !setFitTransform()) return;
  state.renderedRange = null;
  applyCamera();
}

function zoomBy(factor, center = null) {
  if (!state.model) return;
  const oldScale = state.transform.scale;
  const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, oldScale * factor));
  const rect = elements.svg.getBoundingClientRect();
  const point = center ?? { x: rect.width / 2, y: rect.height / 2 };
  const worldX = (point.x - state.transform.x) / oldScale;
  const worldY = (point.y - state.transform.y) / oldScale;

  state.transform.scale = newScale;
  state.transform.x = point.x - worldX * newScale;
  state.transform.y = point.y - worldY * newScale;
  applyCamera();
}

function applySearch() {
  const query = elements.searchInput.value.trim().toLowerCase();
  const nodes = [...elements.nodeLayer.querySelectorAll(".graph-node")];

  nodes.forEach((node) => {
    const match = query && state.searchMatches.has(node.dataset.nodeId);
    node.classList.toggle("search-match", Boolean(match));
    node.classList.toggle("search-dim", Boolean(query) && !match);
  });
}

function refreshSearch() {
  state.searchFrame = null;
  if (!state.model) return;
  const query = elements.searchInput.value.trim().toLowerCase();
  state.searchCursor = -1;
  state.searchMatches.clear();
  if (query) {
    state.layout.forEach(({ node }) => {
      let searchText = state.searchIndex.get(node.id);
      if (searchText === undefined) {
        searchText = nodeSearchText(node);
        state.searchIndex.set(node.id, searchText);
      }
      if (searchText.includes(query)) state.searchMatches.add(node.id);
    });
  }
  elements.searchCount.textContent = query ? `${state.searchMatches.size} found` : "";
  elements.searchShell.classList.toggle("has-query", Boolean(query));
  elements.searchShell.classList.toggle("no-results", Boolean(query) && state.searchMatches.size === 0);
  renderFlowScene({ force: true });
}

function scheduleSearch() {
  if (state.searchFrame !== null) return;
  state.searchFrame = requestAnimationFrame(refreshSearch);
}

function focusSearchMatch(direction = 1) {
  if (!state.searchMatches.size) return;
  const matches = state.layout.filter(({ node }) => state.searchMatches.has(node.id));
  if (!matches.length) return;
  state.searchCursor = (state.searchCursor + direction + matches.length) % matches.length;
  const entry = matches[state.searchCursor];
  const scale = Math.max(0.54, state.transform.scale);
  state.transform.scale = scale;
  state.transform.x = elements.viewport.clientWidth / 2 - (entry.x + NODE_WIDTH / 2) * scale;
  state.transform.y = elements.viewport.clientHeight / 2 - (entry.y + NODE_HEIGHT / 2) * scale;
  state.renderedRange = null;
  elements.searchCount.textContent = `${state.searchCursor + 1}/${matches.length}`;
  applyCamera();
  setTimeout(() => {
    if (state.model) renderFlowScene({ force: true });
    const match = [...elements.nodeLayer.querySelectorAll(".graph-node")]
      .find((nodeElement) => nodeElement.dataset.nodeId === entry.node.id);
    selectNode(entry.node);
    match?.focus({ preventScroll: true });
  }, 34);
}

function detailRow(term, description) {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = description;
  return [dt, dd];
}

function appendMetadataBlock(parent, title, metadata, { open = false } = {}) {
  if (metadata === null || metadata === undefined) return;
  if (!Array.isArray(metadata) && typeof metadata === "object" && Object.keys(metadata).length === 0) return;
  const details = document.createElement("details");
  details.className = "metadata-block";
  const summary = document.createElement("summary");
  summary.textContent = title;
  const pre = document.createElement("pre");
  pre.textContent = "Expand to format this record…";
  const render = () => {
    if (!details.open || details.dataset.rendered) return;
    pre.textContent = typeof metadata === "string" ? metadata : JSON.stringify(metadata, null, 2);
    details.dataset.rendered = "true";
  };
  details.addEventListener("toggle", render, { passive: true });
  details.append(summary, pre);
  if (open) {
    details.open = true;
    render();
  }
  parent.append(details);
  return details;
}

function setInspectorTab(tab, { focus = false } = {}) {
  state.inspectorTab = tab;
  elements.inspectorTabs.querySelectorAll("[data-inspector-tab]").forEach((button) => {
    const selected = button.dataset.inspectorTab === tab;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  });
  document.querySelectorAll("[data-inspector-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.inspectorPanel !== tab;
  });
  ensureInspectorTabRendered(tab);
}

function inspectorLoadingPlaceholder(label) {
  const loading = document.createElement("div");
  loading.className = "inspector-panel-loading";
  loading.setAttribute("role", "status");
  loading.setAttribute("aria-label", `Preparing ${label}`);
  loading.append(document.createElement("span"), document.createElement("span"), document.createElement("span"));
  return loading;
}

function ensureInspectorTabRendered(tab) {
  if (!state.model || !state.selectedId || !["huggingface", "raw"].includes(tab)) return;
  if (state.inspectorRendered.has(tab)) return;
  const node = state.model.nodeById?.get(state.selectedId)
    ?? state.layout.find((entry) => entry.node.id === state.selectedId)?.node;
  if (!node) return;
  const panel = tab === "huggingface" ? elements.inspectorHuggingFace : elements.inspectorRaw;
  const token = state.inspectorRenderToken;
  state.inspectorRendered.add(tab);
  panel.replaceChildren(inspectorLoadingPlaceholder(tab === "huggingface" ? "Hugging Face metadata" : "raw records"));
  requestAnimationFrame(() => {
    if (token !== state.inspectorRenderToken || node.id !== state.selectedId) return;
    if (tab === "huggingface") renderHuggingFaceDetails(node);
    else renderRawDetails(node);
  });
}

function inspectorSection(parent, title) {
  const section = document.createElement("section");
  section.className = "inspector-section";
  const heading = document.createElement("h4");
  heading.textContent = title;
  section.append(heading);
  parent.append(section);
  return section;
}

function inspectorCard(label, value, detail = "") {
  const card = document.createElement("div");
  card.className = "inspector-card";
  const heading = document.createElement("strong");
  heading.textContent = label;
  const content = document.createElement("span");
  content.textContent = value ?? "—";
  card.append(heading, content);
  if (detail) {
    const small = document.createElement("small");
    small.textContent = detail;
    card.append(small);
  }
  return card;
}

function appendCardGrid(parent, cards) {
  const grid = document.createElement("div");
  grid.className = "inspector-card-grid";
  cards.forEach(([label, value, detail]) => grid.append(inspectorCard(label, value, detail)));
  parent.append(grid);
  return grid;
}

function emptyDetail(parent, message) {
  const paragraph = document.createElement("p");
  paragraph.className = "empty-detail";
  paragraph.textContent = message;
  parent.append(paragraph);
}

function humanDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleString();
}

function tensorSafetensorsRecord(tensor) {
  return tensor.metadata?.safetensors ?? null;
}

function tensorCheckpointRecord(tensor) {
  return tensor.metadata?.checkpoint ?? tensorSafetensorsRecord(tensor);
}

function scaledTensorDimension(value, minimum, maximum) {
  if (!Number.isFinite(value) || value <= 0) return minimum;
  const normalized = Math.min(1, Math.log2(value + 1) / 16);
  return minimum + normalized * (maximum - minimum);
}

function makeTensorPreview(tensor) {
  if (!Array.isArray(tensor.shape) || tensor.shape.length < 2) return null;
  const rank = tensor.shape.length;
  const width = scaledTensorDimension(tensor.shape.at(-1), 27, 38);
  const height = scaledTensorDimension(tensor.shape.at(-2), 24, 35);
  const depth = tensorStackDepth(rank);
  const left = 7;
  const top = 7;

  const wrapper = document.createElement("div");
  wrapper.className = "tensor-preview";
  wrapper.title = `${rank}D tensor · ${formatShape(tensor.shape)}`;
  const svg = svgElement("svg", {
    viewBox: "0 0 66 62",
    role: "img",
    "aria-label": `${rank} dimensional tensor, shape ${formatShape(tensor.shape)}`
  });
  for (let layer = depth; layer >= 1; layer -= 1) {
    const offset = layer * 5;
    svg.append(svgElement("rect", {
      class: "tensor-preview-layer",
      x: left + offset,
      y: top + offset,
      width,
      height,
      rx: 4
    }));
  }
  svg.append(svgElement("rect", {
    class: "tensor-preview-front",
    x: left,
    y: top,
    width,
    height,
    rx: 4
  }));
  const badge = document.createElement("span");
  badge.textContent = `${rank}D`;
  wrapper.append(svg, badge);
  return wrapper;
}

function renderConnectionDetails(node) {
  elements.inspectorConnections.replaceChildren();
  const residualLedger = node.inspector.residualLedger;
  if (residualLedger) {
    const ledgerSection = inspectorSection(elements.inspectorConnections, "Residual accounting");
    const write = residualLedger.write;
    appendCardGrid(ledgerSection, residualLedger.role === "state" ? [
      ["Accumulated state", residualLedger.state, residualLedger.equation],
      ["Activation norm", formatLedgerMagnitude(residualLedger.activationNorm), "Prompt-conditioned · not measured"],
      ["Direct logit attribution", formatSignedLedgerValue(residualLedger.directLogitAttribution), "Target token required · not measured"]
    ] : [
      ["Additive write", write.symbol, `${write.label} into ${residualLedger.state}`],
      ["Accounting equation", residualLedger.equation],
      ["Write magnitude", formatLedgerMagnitude(write.value), "Prompt-conditioned · not measured"],
      ["Direct logit attribution", formatSignedLedgerValue(write.directLogitAttribution), "Target token required · not measured"]
    ]);
    const openLedger = document.createElement("button");
    openLedger.className = "inspector-ledger-link";
    openLedger.type = "button";
    openLedger.textContent = "Open in residual ledger";
    openLedger.addEventListener("click", () => {
      setResidualLedgerOpen(true);
      syncResidualLedgerSelection();
    });
    ledgerSection.append(openLedger);
  }
  if (node.metadata) {
    const metadataSection = inspectorSection(elements.inspectorConnections, "Item metadata");
    appendMetadataBlock(metadataSection, "Complete node metadata", node.metadata, { open: true });
  }
  const group = node.inspector.group;
  if (group) {
    const section = inspectorSection(elements.inspectorConnections, "Decoder block");
    appendCardGrid(section, [
      [group.name, group.description || "Checkpoint-mapped transformer layer", group.repeat > 1 ? `Repeated ${group.repeat} times` : "Exact layer instance"]
    ]);
  }

  const section = inspectorSection(elements.inspectorConnections, "Autoregressive dataflow");
  const list = document.createElement("div");
  list.className = "connection-list";
  node.inspector.connections.forEach((edge) => {
      const item = document.createElement("div");
      item.className = "connection-item";
      const direction = document.createElement("b");
      direction.textContent = edge.direction === "in" ? "Reads" : "Writes";
      const name = document.createElement("span");
      name.textContent = edge.counterpartName;
      const detail = document.createElement("small");
      detail.textContent = [edge.label, edge.kind, edge.feedback ? "recurrent / cache path" : ""].filter(Boolean).join(" · ");
      item.append(direction, name, detail);
      list.append(item);
  });
  if (list.childElementCount) section.append(list);
  else emptyDetail(section, "This item has no explicit graph connections.");

  const findings = node.inspector.findings;
  if (findings.length) {
    const evidenceSection = inspectorSection(elements.inspectorConnections, "Architecture evidence");
    const cards = findings.map((finding) => [
      finding.feature,
      finding.value,
      `${finding.basis} · ${finding.confidence} confidence · ${finding.evidence}`
    ]);
    appendCardGrid(evidenceSection, cards);
  }
}

function renderTensorDetails(node) {
  elements.inspectorTensors.replaceChildren();
  const tensors = [...(node.tensors ?? [])].sort((left, right) => (
    (left.order?.operationIndex ?? Number.MAX_SAFE_INTEGER) - (right.order?.operationIndex ?? Number.MAX_SAFE_INTEGER)
  ));
  elements.inspectorTensorCount.textContent = tensors.length ? String(tensors.length) : "";
  if (!tensors.length) {
    emptyDetail(elements.inspectorTensors, "This operation has no checkpoint tensor assigned to it. Its behavior is represented by the activation formula and configuration evidence.");
    return;
  }

  const tensorSummary = node.inspector.tensorSummary;
  const orderingSummary = tensorSummary.ordering ?? {};
  const summary = document.createElement("section");
  summary.className = "inspector-section";
  appendCardGrid(summary, [
    ["Exact tensors", new Intl.NumberFormat().format(tensors.length)],
    ["Order inference", "Automatic", `${orderingSummary.recognized ?? 0} recognized · ${orderingSummary.pathDerived ?? 0} path-derived`],
    ["Elements", new Intl.NumberFormat().format(tensorSummary.elements)],
    ["Stored bytes", formatBytes(tensorSummary.bytes)],
    ["Dtypes", tensorSummary.dtypes.join(", ")]
  ]);
  elements.inspectorTensors.append(summary);

  const overview = document.createElement("section");
  overview.className = "tensor-order-overview";
  const overviewHeading = document.createElement("div");
  overviewHeading.className = "tensor-order-heading";
  const overviewTitle = document.createElement("strong");
  overviewTitle.textContent = "Automatic operation order";
  const overviewBasis = document.createElement("span");
  overviewBasis.textContent = "Tensor paths · first read → last read";
  overviewHeading.append(overviewTitle, overviewBasis);
  const sequence = document.createElement("div");
  sequence.className = "tensor-order-sequence";
  const visibleSequence = tensors.slice(0, 12);
  visibleSequence.forEach((tensor, index) => {
    const step = document.createElement("span");
    step.className = `tensor-order-step ${tensor.order?.semanticConfidence ?? "path-derived"}`;
    step.title = `Automatically inferred from ${tensor.order?.semanticSource ?? "the tensor module path"}`;
    const number = document.createElement("b");
    number.textContent = String(index + 1).padStart(2, "0");
    const label = document.createElement("span");
    label.textContent = tensor.order?.semanticRole ?? "Model parameter";
    const kind = document.createElement("small");
    kind.textContent = tensor.order?.parameterKind ?? "Parameter";
    step.append(number, label, kind);
    sequence.append(step);
  });
  if (tensors.length > visibleSequence.length) {
    const remaining = document.createElement("span");
    remaining.className = "tensor-order-step remaining";
    remaining.textContent = `+ ${new Intl.NumberFormat().format(tensors.length - visibleSequence.length)} more in sequence`;
    sequence.append(remaining);
  }
  overview.append(overviewHeading, sequence);
  elements.inspectorTensors.append(overview);

  const detailHeading = document.createElement("div");
  detailHeading.className = "tensor-heading tensor-order-detail-heading";
  detailHeading.textContent = "Ordered tensor records";
  elements.inspectorTensors.append(detailHeading);

  const orderList = document.createElement("div");
  orderList.className = "tensor-order-list";
  tensors.forEach((tensor, index) => {
    const row = document.createElement("article");
    row.className = `tensor-order-row ${tensor.order?.semanticConfidence ?? "path-derived"}`;
    const order = tensor.order ?? {};
    const checkpointRecord = tensorCheckpointRecord(tensor);

    const rail = document.createElement("div");
    rail.className = "tensor-order-rail";
    const orderNumber = document.createElement("span");
    orderNumber.textContent = String(index + 1).padStart(2, "0");
    rail.append(orderNumber);

    const preview = tensor.shapeKnown === false ? null : makeTensorPreview(tensor);
    const visual = preview ?? document.createElement("div");
    if (!preview) {
      visual.className = "tensor-linear-preview";
      visual.textContent = tensor.shapeKnown === false ? "?D" : `${Array.isArray(tensor.shape) ? tensor.shape.length : 0}D`;
      visual.setAttribute("aria-label", tensor.shapeKnown === false ? "Tensor shape unavailable" : `${Array.isArray(tensor.shape) ? tensor.shape.length : 0} dimensional tensor`);
    }

    const content = document.createElement("div");
    content.className = "tensor-order-content";
    const roleLine = document.createElement("div");
    roleLine.className = "tensor-order-role";
    const parameterKind = document.createElement("span");
    parameterKind.className = `tensor-parameter-kind ${(order.parameterKind ?? "parameter").toLowerCase()}`;
    parameterKind.textContent = order.parameterKind ?? "Parameter";
    const role = document.createElement("strong");
    role.textContent = order.semanticRole ?? "Model parameter";
    const operationPosition = document.createElement("small");
    operationPosition.textContent = `${order.semanticConfidence === "path-derived" ? "path-derived" : "auto"} · ${index + 1} of ${tensors.length}`;
    operationPosition.title = `Semantic role inferred from ${order.semanticSource ?? "the tensor path"}`;
    roleLine.append(parameterKind, role, operationPosition);

    const path = document.createElement("code");
    path.className = "tensor-order-path";
    path.textContent = tensor.name;

    const facts = document.createElement("div");
    facts.className = "tensor-order-facts";
    [
      tensor.shapeKnown === false ? "shape unavailable" : formatShape(tensor.shape),
      tensor.dtype === "unknown" ? "dtype unavailable" : tensor.dtype,
      tensor.shapeKnown === false ? "element count unavailable" : `${formatCount(tensor.count)} elements`,
      tensor.shapeKnown === false ? "byte offsets unavailable" : formatBytes(tensor.totalBytes)
    ].forEach((value) => {
      const fact = document.createElement("span");
      fact.textContent = value;
      facts.append(fact);
    });
    content.append(roleLine, path, facts);
    appendMetadataBlock(content, "Complete tensor record", {
      path: tensor.path,
      shape: tensor.shape,
      dtype: tensor.dtype,
      elementCount: tensor.count,
      byteEstimate: tensor.totalBytes,
      order: tensor.order,
      metadata: tensor.metadata
    });

    const checkpoint = document.createElement("div");
    checkpoint.className = "tensor-checkpoint-order";
    const checkpointLabel = document.createElement("span");
    checkpointLabel.textContent = "Checkpoint";
    const checkpointIndex = document.createElement("strong");
    checkpointIndex.textContent = Number.isInteger(order.checkpointIndex)
      ? `#${new Intl.NumberFormat().format(order.checkpointIndex + 1)}`
      : "—";
    const checkpointDetail = document.createElement("small");
    checkpointDetail.textContent = [
      Number.isInteger(order.shardIndex) ? `shard ${order.shardIndex + 1}` : checkpointRecord?.file,
      Number.isInteger(order.fileTensorIndex) ? `item ${order.fileTensorIndex + 1}` : "",
      Array.isArray(checkpointRecord?.dataOffsets) ? `@ ${new Intl.NumberFormat().format(checkpointRecord.dataOffsets[0])} B` : ""
    ].filter(Boolean).join(" · ");
    checkpoint.append(checkpointLabel, checkpointIndex, checkpointDetail);

    row.append(rail, visual, content, checkpoint);
    orderList.append(row);
  });
  elements.inspectorTensors.append(orderList);
}

function renderRepositoryFileList(parent, files, artifacts) {
  const list = document.createElement("div");
  list.className = "artifact-list";
  files.forEach((file) => {
    const item = document.createElement("div");
    item.className = "artifact-item";
    const label = document.createElement("b");
    label.textContent = artifacts[file.rfilename] ? "Loaded" : "File";
    const name = document.createElement("span");
    name.textContent = file.rfilename;
    const detail = document.createElement("small");
    detail.textContent = [
      Number.isFinite(file.size) ? formatBytes(file.size) : "size unavailable",
      file.blobId ? `blob ${file.blobId.slice(0, 12)}` : "",
      file.lfs?.sha256 ? `LFS ${String(file.lfs.sha256).slice(0, 12)}` : ""
    ].filter(Boolean).join(" · ");
    item.append(label, name, detail);
    list.append(item);
  });
  parent.append(list);
}

function renderHuggingFaceDetails(node) {
  elements.inspectorHuggingFace.replaceChildren();
  const hf = state.model.huggingFace ?? {};
  const hub = hf.hub ?? {};
  const artifacts = hf.artifacts ?? {};
  const source = state.model.source ?? {};

  const identity = inspectorSection(elements.inspectorHuggingFace, "Hub provenance");
  appendCardGrid(identity, [
    ["Repository", source.modelId ?? state.model.name, hub.author ? `Author: ${hub.author}` : ""],
    ["Commit", source.sha?.slice(0, 16) ?? "—", `Requested revision: ${source.revision ?? "main"}`],
    ["Task", hub.pipeline_tag ?? "—", hub.library_name ?? state.model.type],
    ["Visibility", hub.private ? "Private" : "Public", hub.gated ? `Gated: ${hub.gated}` : "Not gated"],
    ["Downloads", Number.isFinite(hub.downloads) ? new Intl.NumberFormat().format(hub.downloads) : "—", `${hub.likes ?? 0} likes`],
    ["All-time downloads", Number.isFinite(hub.downloadsAllTime) ? new Intl.NumberFormat().format(hub.downloadsAllTime) : "—", `Trending score: ${hub.trendingScore ?? "—"}`],
    ["Repository size", Number.isFinite(hub.usedStorage) ? formatBytes(hub.usedStorage) : "—", `${hub.siblings?.length ?? 0} files`],
    ["Base models", Array.isArray(hub.baseModels) && hub.baseModels.length ? hub.baseModels.join(", ") : "—"],
    ["Security scan", hub.securityRepoStatus?.scansDone ? "Complete" : "Unavailable", `${hub.securityRepoStatus?.filesWithIssues?.length ?? 0} files flagged by Hub`],
    ["Created", humanDate(hub.createdAt)],
    ["Last modified", humanDate(hub.lastModified)]
  ]);
  if (source.url) {
    const link = document.createElement("a");
    link.className = "inspector-link";
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = "Open this exact revision on Hugging Face ↗";
    identity.append(link);
  }

  const configSection = inspectorSection(elements.inspectorHuggingFace, "Configuration for this item");
  const relevantConfig = node.inspector.config;
  if (Object.keys(relevantConfig).length) appendMetadataBlock(configSection, "Relevant resolved fields", relevantConfig, { open: true });
  else emptyDetail(configSection, "No standard configuration fields were mapped specifically to this operation.");

  const files = node.inspector.files;
  const fileSection = inspectorSection(elements.inspectorHuggingFace, "Relevant repository files");
  if (files.length) renderRepositoryFileList(fileSection, files, artifacts);
  else emptyDetail(fileSection, "No repository file is uniquely associated with this activation-only operation.");
  files.forEach((file) => {
    const artifact = artifacts[file.rfilename];
    if (artifact) appendMetadataBlock(fileSection, `${file.rfilename} contents`, artifact.data);
  });

  const artifactSection = inspectorSection(elements.inspectorHuggingFace, "Fetched model metadata");
  const modelCard = artifacts["README.md"];
  if (modelCard) appendMetadataBlock(artifactSection, "README.md model card", modelCard.data);
  Object.entries(artifacts)
    .filter(([file]) => file !== "README.md" && !files.some((candidate) => candidate.rfilename === file))
    .forEach(([file, artifact]) => appendMetadataBlock(artifactSection, file, artifact.data));
  appendMetadataBlock(artifactSection, "Artifact inspection policy and skipped files", hf.artifactInspection);

  const hubSection = inspectorSection(elements.inspectorHuggingFace, "Complete Hub records");
  appendMetadataBlock(hubSection, "Hub model record", hub);
  appendMetadataBlock(hubSection, "All repository file records", hub.siblings);
  appendMetadataBlock(hubSection, "Model card frontmatter", hub.cardData);
  appendMetadataBlock(hubSection, "Transformers integration", hub.transformersInfo);
  appendMetadataBlock(hubSection, "Hub Safetensors summary", hub.safetensors);
  appendMetadataBlock(hubSection, "Hub security scan", hub.securityRepoStatus);
  appendMetadataBlock(hubSection, "Model index and evaluations", hub["model-index"] ?? hub.evalResults);
  appendMetadataBlock(hubSection, "Tags", hub.tags);
  appendMetadataBlock(hubSection, "Widget and inference metadata", {
    widgetData: hub.widgetData,
    inference: hub.inference,
    inferenceProviderMapping: hub.inferenceProviderMapping,
    spaces: hub.spaces
  });
}

function renderRawDetails(node) {
  elements.inspectorRaw.replaceChildren();
  const group = node.inspector.group;
  const connections = node.inspector.connections;
  const itemSection = inspectorSection(elements.inspectorRaw, "Exact graph item");
  appendMetadataBlock(itemSection, "Parsed item record", node, { open: true });
  appendMetadataBlock(itemSection, "Group record", group);
  appendMetadataBlock(itemSection, "Incoming and outgoing edges", connections);

  const modelSection = inspectorSection(elements.inspectorRaw, "Underlying Hugging Face payload");
  appendMetadataBlock(modelSection, "Source and revision", state.model.source);
  appendMetadataBlock(modelSection, "Complete config.json", state.model.config);
  appendMetadataBlock(modelSection, "Resolved decoder config", state.model.resolvedTextConfig);
  appendMetadataBlock(modelSection, "Layer-family inference", state.model.resolvedLayerFamily);
  appendMetadataBlock(modelSection, "Forward-topology contract", state.model.forwardTopology);
  appendMetadataBlock(modelSection, "Graph validation", state.model.validation);
  appendMetadataBlock(modelSection, "All architecture predictions", state.model.architecturePredictions);
  appendMetadataBlock(modelSection, "Checkpoint resolver and metadata record", state.model.checkpoint ?? state.model.safetensors);
  appendMetadataBlock(modelSection, "Complete Hugging Face record", state.model.huggingFace);
}

function selectNode(node) {
  state.selectedId = node.id;
  state.inspectorRenderToken += 1;
  state.inspectorRendered.clear();
  elements.nodeLayer
    .querySelectorAll(".graph-node")
    .forEach((element) => {
      const selected = element.dataset.nodeId === node.id;
      element.classList.toggle("selected", selected);
      element.setAttribute("aria-pressed", String(selected));
      element.setAttribute("tabindex", selected ? "0" : "-1");
    });

  elements.inspectorKind.textContent = node.kind === "root" ? "Model" : node.kind;
  elements.inspectorName.textContent = node.name;
  elements.inspectorDescription.textContent = node.description || node.type || "No description supplied.";
  elements.inspectorDetails.replaceChildren();

  const rows = [
    ["Node ID", node.id],
    ["Path", node.path],
    ["Operation", node.type ?? node.dtype],
    ["Circuit", circuitFamily(node).replace("circuit-", "")],
    ["Checkpoint elements", new Intl.NumberFormat().format(node.totalElements ?? node.totalParameters ?? 0)],
    ["Stored bytes", formatBytes(node.totalBytes)]
  ];

  if (node.kind === "parameter") {
    rows.push(["Shape", formatShape(node.shape)], ["Dtype", node.dtype]);
  }
  if (node.formula) rows.push(["Function", node.formula]);
  if (node.shape && node.kind !== "parameter") rows.push(["Activation", formatShape(node.shape)]);
  if (node.repeat > 1) rows.push(["Repeated", `${node.repeat} times`]);
  const observation = observedNodeMetric(node.id);
  if (observation) {
    rows.push(
      ["Observed norm", modelMetric(observation.norm, 4)],
      ["Target-token DLA", shortObservedValue(observation.dla, true)],
      ["Observed at", state.runtime.latestRun?.positionLabel ?? `position ${state.runtime.latestRun?.position ?? "—"}`]
    );
  }

  rows.forEach(([term, description]) => elements.inspectorDetails.append(...detailRow(term, description)));
  renderConnectionDetails(node);
  renderTensorDetails(node);
  elements.inspectorHuggingFace.replaceChildren();
  elements.inspectorRaw.replaceChildren();
  setInspectorTab(state.inspectorTab);
  elements.copyNodePath.textContent = "Copy path";
  elements.graphAnnouncement.textContent = `${node.name} selected. Detail inspector opened.`;
  elements.inspector.hidden = false;
  syncNodeAnnotationEditor();
  syncResidualLedgerSelection();
}

const EXPORT_STYLES = `
  .graph-edge{fill:none;stroke:#9c9a92;stroke-width:1.05}.circuit-edge-memory{stroke:#c46686;stroke-dasharray:4 4}.circuit-edge-residual{stroke:#788c5d;stroke-width:1.75}.residual-rail{stroke:#63784c;stroke-width:3.2}.circuit-edge-attention{stroke:#6a9bcc}.circuit-edge-mlp{stroke:#cc785c}.flow-group-body{fill:#efebdf;fill-opacity:.46;stroke:#5f5b52;stroke-width:1.25}.transformer-block-header{fill:#faf8f1;stroke:#8b867b;stroke-width:1}.flow-group-label{fill:#555149;font:600 7.5px 'Styrene A',sans-serif;letter-spacing:.65px}.flow-subgroup-body{stroke-dasharray:3 3}.residual-stream-label{fill:#4f603e;font:500 9px 'Styrene A',sans-serif;letter-spacing:1px}.edge-label{fill:#5e5d59;stroke:#faf9f5;stroke-width:5px;font:italic 8px 'Tiempos Text',serif;paint-order:stroke}
  .node-body{stroke-width:1;filter:none}.operation .node-body{fill:#fff;stroke:#b9b6ad}.input .node-body{fill:#dce8e4;stroke:#819f95}.state .node-body{fill:#ebe9f2;stroke:#9c98b2}.output .node-body,.sampler .node-body{fill:#f1e6d0;stroke:#b78b59}.circuit-attention .node-body{fill:#e6eef5;stroke:#6a9bcc}.circuit-mlp .node-body{fill:#f5e8e3;stroke:#cc785c}.circuit-norm .node-body{fill:#efedf4;stroke:#9b96b3}.circuit-residual .node-body{fill:#edf1e8;stroke:#788c5d}.circuit-memory .node-body{fill:#e8f0ed;stroke:#71988a}
  .node-title{fill:#1f1f1e;font:500 11px 'Styrene A',sans-serif}.node-kicker{fill:#65635e;font:500 7px 'Styrene A',sans-serif;letter-spacing:1px}.node-formula{fill:#3d3d3a;font:italic 8.5px 'Tiempos Text',serif}.node-meta{fill:#77756f;font:8px 'Tiempos Text',serif}.tensor-glyph-front{fill:#bcd1ca;stroke:#587c70}.tensor-glyph-top{fill:#e4eeea;fill-opacity:.72;stroke:#587c70}.tensor-glyph-side{fill:#91b5a9;fill-opacity:.68;stroke:#587c70}.tensor-glyph-slice{fill:#6a9bcc;stroke:#4f769d}.residual-ledger-badge{fill:#edf1e8;stroke:#63784c}.residual-ledger-badge-text{fill:#4f603e;font:500 9px 'Styrene A',sans-serif}.collapse-control{display:none}
`;

function exportSvg() {
  if (!state.model) return;
  renderFlowScene({ force: true, all: true });
  const clone = elements.svg.cloneNode(true);
  state.renderedRange = null;
  renderFlowScene({ force: true });
  const width = state.bounds.width + PAGE_PADDING * 2;
  const height = state.bounds.height + PAGE_PADDING * 2;
  clone.setAttribute("xmlns", SVG_NS);
  clone.setAttribute("width", width);
  clone.setAttribute("height", height);
  clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
  clone.setAttribute("style", "background:#f8f9fc");
  clone.querySelector("#camera").setAttribute("transform", `translate(${PAGE_PADDING} ${PAGE_PADDING})`);
  clone.querySelectorAll("[data-arrowhead]").forEach((arrowhead) => {
    arrowhead.setAttribute("markerWidth", String(ARROWHEAD_SCREEN_SIZE));
    arrowhead.setAttribute("markerHeight", String(ARROWHEAD_SCREEN_SIZE));
  });
  clone.querySelector("defs").append(Object.assign(svgElement("style"), { textContent: EXPORT_STYLES }));

  const source = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const safeName = state.model.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  anchor.href = url;
  anchor.download = `${safeName || "model"}-graph.svg`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function renderPredictions(predictions) {
  elements.predictionList.replaceChildren();
  elements.predictionCount.textContent = `${predictions.length} labeled finding${predictions.length === 1 ? "" : "s"}`;
  predictions.forEach((prediction) => {
    const item = document.createElement("div");
    item.className = "prediction-item";
    const heading = document.createElement("div");
    const feature = document.createElement("strong");
    feature.textContent = prediction.feature;
    const badge = document.createElement("span");
    badge.className = `prediction-badge ${prediction.basis}`;
    badge.textContent = prediction.basis === "declared" ? "config" : prediction.confidence;
    heading.append(feature, badge);
    const value = document.createElement("p");
    value.textContent = prediction.value;
    const evidence = document.createElement("small");
    evidence.textContent = prediction.evidence;
    item.append(heading, value, evidence);
    elements.predictionList.append(item);
  });
}

function huggingFaceRequestHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function setHuggingFaceAccountFormOpen(open) {
  const shouldOpen = Boolean(open) && !state.hfAccount;
  elements.hfAccountForm.hidden = !shouldOpen;
  elements.hfAccountToggleButton.setAttribute("aria-expanded", String(shouldOpen));
  elements.hfAccountToggleButton.textContent = state.hfAccount
    ? "Disconnect"
    : shouldOpen
      ? "Close"
      : "Connect";
  if (shouldOpen) requestAnimationFrame(() => elements.hfTokenInput.focus());
}

function renderHuggingFaceAccount(statusMessage = "") {
  const account = state.hfAccount;
  const accountName = account?.name || account?.fullname || "connected account";
  const organisations = Array.isArray(account?.orgs) ? account.orgs : [];
  elements.hfAccount.dataset.state = account ? "connected" : "disconnected";
  elements.hfAccountTitle.textContent = account
    ? account.fullname || `@${accountName}`
    : "Hugging Face account";
  elements.hfAccountDetail.textContent = account
    ? `@${accountName} · ${account.isPro ? "PRO · " : ""}${organisations.length} ${organisations.length === 1 ? "organization" : "organizations"}`
    : "Public models only";
  const avatarUrl = huggingFaceAssetUrl(account?.avatarUrl);
  elements.hfAccountAvatar.hidden = !avatarUrl;
  elements.hfAccountAvatarFallback.hidden = Boolean(avatarUrl);
  elements.settingsButtonAvatar.hidden = !avatarUrl;
  elements.settingsButtonIcon.hidden = Boolean(avatarUrl);
  if (avatarUrl) elements.hfAccountAvatar.src = avatarUrl;
  else elements.hfAccountAvatar.removeAttribute("src");
  if (avatarUrl) elements.settingsButtonAvatar.src = avatarUrl;
  else elements.settingsButtonAvatar.removeAttribute("src");
  elements.hfAccountOrgs.replaceChildren();
  organisations.slice(0, 3).forEach((organisation) => {
    const chip = document.createElement("span");
    chip.className = "hf-account-org";
    chip.textContent = organisation.fullname || organisation.name || "Organization";
    elements.hfAccountOrgs.append(chip);
  });
  if (organisations.length > 3) {
    const more = document.createElement("span");
    more.className = "hf-account-org";
    more.textContent = `+${organisations.length - 3}`;
    elements.hfAccountOrgs.append(more);
  }
  elements.hfAccountProfile.hidden = !account;
  if (account) elements.hfAccountProfileLink.href = `https://huggingface.co/${encodeURIComponent(accountName)}`;
  else elements.hfAccountProfileLink.removeAttribute("href");
  elements.hfAccountStatus.classList.remove("error");
  elements.hfAccountStatus.textContent = statusMessage;
  setHuggingFaceAccountFormOpen(false);
}

function huggingFaceAssetUrl(value) {
  if (typeof value !== "string" || !value.trim()) return "";
  if (value.startsWith("/")) return new URL(value, "https://huggingface.co").href;
  return value.startsWith("https://") ? value : "";
}

async function disconnectHuggingFaceAccount() {
  elements.hfAccount.dataset.state = "loading";
  elements.hfAccountToggleButton.disabled = true;
  elements.hfAccountStatus.classList.remove("error");
  elements.hfAccountStatus.textContent = "Clearing saved session…";
  try {
    const response = await fetch("/api/huggingface/logout", {
      method: "POST",
      credentials: "same-origin"
    });
    if (!response.ok) throw new Error(`Disconnect failed (${response.status})`);
    state.hfAccount = null;
    elements.hfTokenInput.value = "";
    renderHuggingFaceAccount("Disconnected and removed the saved token. Existing graph data remains visible.");
  } catch (error) {
    elements.hfAccount.dataset.state = "error";
    elements.hfAccountStatus.classList.add("error");
    elements.hfAccountStatus.textContent = String(error?.message ?? error);
  } finally {
    elements.hfAccountToggleButton.disabled = false;
  }
}

async function restoreHuggingFaceAccount() {
  state.hfAccountLoading = true;
  elements.hfAccount.dataset.state = "loading";
  elements.hfAccountToggleButton.disabled = true;
  elements.hfAccountDetail.textContent = "Restoring saved session…";
  try {
    const response = await fetch("/api/huggingface/account", { credentials: "same-origin" });
    if (response.status === 401) {
      state.hfAccount = null;
      renderHuggingFaceAccount();
      return;
    }
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error ?? `Session restore failed (${response.status})`);
    state.hfAccount = payload;
    renderHuggingFaceAccount(`Welcome back, ${payload.fullname || `@${payload.name}`}. Your saved session was restored.`);
  } catch (error) {
    state.hfAccount = null;
    renderHuggingFaceAccount();
    elements.hfAccount.dataset.state = "error";
    elements.hfAccountStatus.classList.add("error");
    elements.hfAccountStatus.textContent = String(error?.message ?? error);
  } finally {
    state.hfAccountLoading = false;
    elements.hfAccountToggleButton.disabled = false;
  }
}

async function connectHuggingFaceAccount(event) {
  event.preventDefault();
  const token = elements.hfTokenInput.value.trim();
  if (!token) {
    elements.hfTokenInput.setAttribute("aria-invalid", "true");
    elements.hfAccountStatus.classList.add("error");
    elements.hfAccountStatus.textContent = "Paste a Hugging Face read access token.";
    elements.hfTokenInput.focus();
    return;
  }
  state.hfAccountLoading = true;
  elements.hfAccount.dataset.state = "loading";
  elements.hfAccountConnectButton.disabled = true;
  elements.hfAccountToggleButton.disabled = true;
  elements.hfTokenInput.disabled = true;
  elements.hfImportButton.disabled = true;
  elements.gpt2DevExampleButton.disabled = true;
  elements.hfModelInput.disabled = true;
  elements.hfRevisionInput.disabled = true;
  elements.hfAccountConnectButton.textContent = "Checking…";
  elements.hfAccountStatus.classList.remove("error");
  elements.hfAccountStatus.textContent = "Validating with Hugging Face…";
  try {
    const response = await fetch("/api/huggingface/account", {
      credentials: "same-origin",
      headers: huggingFaceRequestHeaders(token)
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error ?? `Connection failed (${response.status})`);
    state.hfAccount = payload;
    elements.hfTokenInput.value = "";
    elements.hfTokenInput.setAttribute("aria-invalid", "false");
    renderHuggingFaceAccount(`${payload.fullname || `@${payload.name}`} connected. The session is saved for 30 days.`);
  } catch (error) {
    state.hfAccount = null;
    elements.hfTokenInput.value = "";
    elements.hfTokenInput.setAttribute("aria-invalid", "true");
    elements.hfAccount.dataset.state = "error";
    elements.hfAccountStatus.classList.add("error");
    elements.hfAccountStatus.textContent = String(error?.message ?? error);
    elements.hfTokenInput.focus();
  } finally {
    state.hfAccountLoading = false;
    elements.hfAccountConnectButton.disabled = false;
    elements.hfAccountToggleButton.disabled = false;
    elements.hfTokenInput.disabled = false;
    elements.hfImportButton.disabled = false;
    elements.hfModelInput.disabled = false;
    elements.hfRevisionInput.disabled = false;
    elements.hfAccountConnectButton.textContent = "Connect";
  }
}

function setRuntimeMode(mode) {
  const nextMode = mode === "local" ? "local" : "daytona";
  state.runtime.mode = nextMode;
  const local = nextMode === "local";
  elements.runtimeModeLocal.setAttribute("aria-pressed", String(local));
  elements.runtimeModeDaytona.setAttribute("aria-pressed", String(!local));
  elements.runtimeLocalSetup.hidden = !local;
  elements.runtimeDaytonaSetup.hidden = local;
  elements.runtimeConnectButton.textContent = local ? "Connect local worker" : "Start Daytona GPU";
  if (!local) refreshDaytonaRecommendation();
}

function daytonaModelDetails() {
  return {
    modelId: state.model?.source?.modelId ?? "",
    parameterCount: Math.max(0, (state.model?.stats?.checkpointElements ?? state.model?.stats?.totalParameters ?? 0) - (state.model?.stats?.recognizedBufferElements ?? 0)),
    checkpointBytes: state.model?.stats?.totalBytes ?? 0
  };
}

async function refreshDaytonaRecommendation() {
  if (!elements.runtimeGpuRecommendation) return;
  try {
    const recommendation = await runtimeApi("/daytona/recommend", { method: "POST", body: daytonaModelDetails() });
    state.runtime.recommendation = recommendation;
    elements.runtimeGpuRecommendation.textContent = `${recommendation.recommendedGpu} · ${recommendation.gpuMemoryGiB} GiB${recommendation.quantization === "4bit" ? " · 4-bit" : ""}`;
    const estimate = Number.isFinite(recommendation.estimatedPeakBytes)
      ? ` Estimated peak ${formatBytes(recommendation.estimatedPeakBytes)}.`
      : "";
    elements.runtimeGpuReason.textContent = `${recommendation.reason}${estimate}`;
  } catch (error) {
    elements.runtimeGpuRecommendation.textContent = "Recommendation unavailable";
    elements.runtimeGpuReason.textContent = String(error?.message ?? error);
  }
}

async function copyRuntimeCommand() {
  const command = "make -C '/Users/maximiliannicholson/Documents/untitled folder 5' worker";
  try {
    await navigator.clipboard.writeText(command);
    elements.copyRuntimeCommandButton.textContent = "Copied";
    setTimeout(() => { elements.copyRuntimeCommandButton.textContent = "Copy"; }, 1400);
  } catch {
    elements.runtimeStatus.classList.add("error");
    elements.runtimeStatus.textContent = "Could not copy the command. Select it manually instead.";
  }
}

async function runtimeApi(path, { method = "GET", body } = {}) {
  const requestStarted = globalThis.performance?.now?.();
  const response = await fetch(`/api/runtime${path}`, {
    method,
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error ?? `Runtime request failed (${response.status})`);
  const requestEnded = globalThis.performance?.now?.();
  if (path === "/forward" && payload?.performance && Number.isFinite(requestStarted) && Number.isFinite(requestEnded)) {
    payload.performance.clientRoundTripMs = Math.max(0, requestEnded - requestStarted);
  }
  return payload;
}

async function debugApi(path = "", { method = "GET", body } = {}) {
  const response = await fetch(`/api/debug/cases${path}`, {
    method,
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error ?? `Debug-case request failed (${response.status})`);
  return payload;
}

function commaSeparatedValues(value) {
  return String(value ?? "").split(",").map((item) => item.trim()).filter(Boolean);
}

function behaviourMetricSpec() {
  const kind = elements.debugMetricKindSelect.value;
  const defaultNames = {
    target_probability: "Target-token probability",
    logit_difference: "Correct vs incorrect logit difference",
    sequence_loss: "Sequence loss",
    kl_divergence: "Output KL divergence",
    multi_token_score: "Multi-token answer score",
    custom_token_groups: "Custom token-group difference"
  };
  const name = elements.debugMetricNameInput.value.trim() || defaultNames[kind];
  elements.debugMetricSummary.textContent = name;
  return {
    kind,
    name,
    targetToken: elements.runtimeTargetInput.value.trim(),
    answer: elements.debugMetricAnswerInput.value.trim(),
    correctTokens: commaSeparatedValues(elements.debugMetricPositiveInput.value),
    incorrectTokens: commaSeparatedValues(elements.debugMetricNegativeInput.value),
    positiveTokens: commaSeparatedValues(elements.debugMetricPositiveInput.value),
    negativeTokens: commaSeparatedValues(elements.debugMetricNegativeInput.value)
  };
}

function debugRunPayload(prompt) {
  return {
    prompt,
    targetToken: elements.runtimeTargetInput.value.trim(),
    topK: Number(elements.runtimeTopKInput.value) || 10,
    seed: Number(elements.debugSeedInput.value) || 0,
    metric: behaviourMetricSpec(),
    generation: { doSample: false, maxNewTokens: 1 },
    logitLens: { enabled: true, maxStages: 24, topK: 3 },
    expected: elements.debugExpectedInput.value.trim()
  };
}

function compactRun(run) {
  if (!run) return null;
  return {
    ...run,
    attention: (run.attention ?? []).map((layer) => ({
      ...layer,
      heads: (layer.heads ?? []).map(({ lastQuery: _lastQuery, ...head }) => head)
    }))
  };
}

function compactComparison(comparison) {
  if (!comparison) return null;
  return { ...comparison, failure: compactRun(comparison.failure), control: compactRun(comparison.control) };
}

function debugCaseDocument() {
  const current = state.debug.currentCase ?? {};
  const source = state.model?.source ?? {};
  const selected = {
    prompt: elements.runtimePromptInput.value,
    expected: elements.debugExpectedInput.value.trim(),
    latestRun: compactRun(state.runtime.latestRun)
  };
  const reference = {
    prompt: elements.runtimeControlPromptInput.value,
    latestRun: compactRun(state.debug.comparison?.control)
  };
  return {
    ...current,
    name: elements.debugCaseNameInput.value.trim() || `Debug ${(source.modelId ?? elements.hfModelInput.value.trim()) || "case"}`,
    status: current.status ?? "open",
    model: {
      modelId: source.modelId ?? elements.hfModelInput.value.trim(),
      revision: (source.revision ?? elements.hfRevisionInput.value.trim()) || "main",
      commit: source.sha ?? current.model?.commit ?? "",
      architecture: state.model?.architecture ?? current.model?.architecture ?? ""
    },
    selected,
    reference,
    // Retained while local records and worker responses migrate from the old role names.
    failure: selected,
    control: reference,
    chatTemplate: state.runtime.latestRun?.context?.chatTemplateValue ?? current.chatTemplate ?? "raw-text",
    chatTemplateSource: state.runtime.latestRun?.context?.chatTemplate ?? current.chatTemplateSource ?? "raw-text",
    tokenization: state.runtime.latestRun?.context?.tokenizer ?? current.tokenization ?? null,
    software: state.runtime.latestRun?.context?.software ?? current.software ?? null,
    target: elements.runtimeTargetInput.value.trim() || state.runtime.latestRun?.target?.text || "",
    seed: Number(elements.debugSeedInput.value) || 0,
    dtype: state.runtime.worker?.dtype ?? current.dtype ?? "auto",
    device: state.runtime.worker?.device ?? current.device ?? "",
    generation: { doSample: false, maxNewTokens: 1, topK: Number(elements.runtimeTopKInput.value) || 10 },
    expected: { text: elements.debugExpectedInput.value.trim() },
    notes: elements.debugNotesInput.value,
    metric: behaviourMetricSpec(),
    comparison: compactComparison(state.debug.comparison),
    experiments: state.debug.interventions.map((item) => ({ ...item, run: compactRun(item.run) })),
    rootCauseTrace: state.debug.trace,
    benchmarkExamples: state.debug.inbox,
    verification: state.debug.verification,
    runHistory: state.debug.runHistory,
    watchlist: watchlistValues(),
    generationTrace: state.runtime.generationTrace,
    causalSweep: state.runtime.causalSweep
  };
}

function setDebugDirty(dirty = true) {
  state.debug.dirty = dirty;
  elements.debugCaseSaveState.textContent = dirty ? "Unsaved changes" : state.debug.currentCase ? "Saved locally" : "Not saved";
}

function renderDebugCaseLibrary() {
  const previous = state.debug.currentCase?.id ?? elements.debugCaseSelect.value;
  elements.debugCaseSelect.replaceChildren();
  if (!state.debug.cases.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No saved cases";
    elements.debugCaseSelect.append(option);
  } else {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose a saved case";
    elements.debugCaseSelect.append(placeholder);
    state.debug.cases.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = `${item.name}${item.modelId ? ` · ${item.modelId}` : ""}`;
      elements.debugCaseSelect.append(option);
    });
  }
  elements.debugCaseSelect.value = state.debug.cases.some((item) => item.id === previous) ? previous : "";
  const selected = Boolean(elements.debugCaseSelect.value);
  elements.openDebugCaseButton.disabled = !selected;
  elements.deleteDebugCaseButton.disabled = !selected;
}

async function restoreDebugCases() {
  try {
    const payload = await debugApi();
    state.debug.cases = payload.cases ?? [];
    renderDebugCaseLibrary();
    elements.debugCaseLibraryStatus.textContent = state.debug.cases.length
      ? `${state.debug.cases.length} local case${state.debug.cases.length === 1 ? "" : "s"} available.`
      : "Cases are stored locally in SQLite.";
  } catch (error) {
    elements.debugCaseLibraryStatus.textContent = String(error?.message ?? error);
  }
}

function resetDebugCaseEditor() {
  state.debug.currentCase = null;
  state.debug.comparison = null;
  state.debug.interventions = [];
  state.debug.trace = null;
  state.debug.inbox = [];
  state.debug.selectedBenchmarkId = null;
  state.debug.verification = null;
  state.debug.runHistory = [];
  state.runtime.generationTrace = null;
  state.runtime.causalSweep = null;
  elements.debugCaseNameInput.value = "";
  elements.debugExpectedInput.value = "";
  elements.debugNotesInput.value = "";
  elements.runtimePromptInput.value = "";
  elements.runtimeControlPromptInput.value = "";
  elements.runtimeTargetInput.value = "";
  elements.debugMetricKindSelect.value = "target_probability";
  elements.debugMetricNameInput.value = "Target-token probability";
  elements.debugMetricAnswerInput.value = "";
  elements.debugMetricPositiveInput.value = "";
  elements.debugMetricNegativeInput.value = "";
  elements.debugSeedInput.value = "0";
  elements.debugBenchmarkName.value = "";
  elements.debugBenchmarkFilter.value = "all";
  elements.debugComparisonResult.hidden = true;
  elements.debugInterventionResult.hidden = true;
  elements.debugTraceResult.hidden = true;
  elements.debugVerificationResult.hidden = true;
  elements.generationTimelineResult.hidden = true;
  elements.causalSweepResult.hidden = true;
  elements.runDiffResult.hidden = true;
  restoreWatchlist();
  renderRunHistory();
  renderBenchmarkExplorer();
  setDebugDirty(false);
}

function hydrateBenchmarkExamples(values) {
  return (Array.isArray(values) ? values : []).flatMap((source, index) => {
    const [normalised] = normaliseBenchmarkExamples([source], {
      benchmark: "Imported benchmark",
      idFactory: () => source?.id ?? `saved-${index}`
    });
    if (!normalised) return [];
    return [{
      ...normalised,
      ...source,
      benchmark: source.benchmark || normalised.benchmark,
      reference: source.reference ?? normalised.reference,
      exampleId: source.exampleId ?? normalised.exampleId,
      suppliedOutcome: source.suppliedOutcome ?? normalised.suppliedOutcome,
      outcomeSource: source.outcomeSource ?? normalised.outcomeSource,
      cluster: source.cluster ?? normalised.cluster
    }];
  });
}

function hydrateDebugCase(record) {
  state.debug.currentCase = record;
  state.debug.comparison = record.comparison ?? null;
  state.debug.interventions = record.experiments ?? [];
  state.debug.trace = record.rootCauseTrace ?? null;
  state.debug.inbox = hydrateBenchmarkExamples(record.benchmarkExamples ?? record.failureInbox ?? []);
  state.debug.selectedBenchmarkId = null;
  state.debug.verification = record.verification ?? null;
  state.debug.runHistory = Array.isArray(record.runHistory) ? record.runHistory : [];
  state.runtime.generationTrace = record.generationTrace ?? null;
  state.runtime.causalSweep = record.causalSweep ?? null;
  restoreWatchlist(record.watchlist ?? null);
  elements.debugCaseNameInput.value = record.name ?? "";
  const selected = record.selected ?? record.failure ?? {};
  const reference = record.reference ?? record.control ?? {};
  elements.debugExpectedInput.value = record.expected?.text ?? selected.expected ?? "";
  elements.debugNotesInput.value = record.notes ?? "";
  elements.runtimePromptInput.value = selected.prompt ?? "";
  elements.runtimeControlPromptInput.value = reference.prompt ?? "";
  elements.runtimeTargetInput.value = record.target ?? record.metric?.targetToken ?? "";
  elements.debugSeedInput.value = String(record.seed ?? 0);
  elements.debugMetricKindSelect.value = record.metric?.kind ?? "target_probability";
  elements.debugMetricNameInput.value = record.metric?.name ?? "Target-token probability";
  elements.debugMetricAnswerInput.value = record.metric?.answer ?? "";
  elements.debugMetricPositiveInput.value = (record.metric?.positiveTokens ?? record.metric?.correctTokens ?? []).join(", ");
  elements.debugMetricNegativeInput.value = (record.metric?.negativeTokens ?? record.metric?.incorrectTokens ?? []).join(", ");
  elements.debugCaseSelect.value = record.id;
  state.runtime.latestRun = state.debug.comparison?.failure ?? selected.latestRun ?? null;
  if (state.model && state.runtime.latestRun) renderRuntimeResult(state.runtime.latestRun);
  if (state.model && state.debug.comparison) renderComparisonResult(state.debug.comparison);
  if (state.runtime.generationTrace) renderGenerationTimeline(state.runtime.generationTrace);
  else elements.generationTimelineResult.hidden = true;
  renderRunHistory();
  renderBenchmarkExplorer();
  setDebugDirty(false);
}

async function saveDebugCase({ duplicate = false, quiet = false } = {}) {
  const documentValue = debugCaseDocument();
  try {
    const existingId = !duplicate && state.debug.currentCase?.id;
    const saved = await debugApi(existingId ? `/${existingId}` : "", { method: existingId ? "PUT" : "POST", body: documentValue });
    state.debug.currentCase = saved;
    setDebugDirty(false);
    await restoreDebugCases();
    elements.debugCaseSelect.value = saved.id;
    if (!quiet) elements.debugCaseLibraryStatus.textContent = `Saved “${saved.name}” locally.`;
    return saved;
  } catch (error) {
    elements.debugCaseSaveState.textContent = "Save failed";
    elements.debugCaseLibraryStatus.textContent = String(error?.message ?? error);
    throw error;
  }
}

async function openSelectedDebugCase() {
  const caseId = elements.debugCaseSelect.value;
  if (!caseId) return;
  elements.debugCaseLibraryStatus.textContent = "Opening debug case…";
  try {
    const record = await debugApi(`/${caseId}`);
    hydrateDebugCase(record);
    elements.hfModelInput.value = record.model?.modelId ?? "";
    elements.hfRevisionInput.value = record.model?.revision ?? "main";
    const loaded = state.model?.source?.modelId === record.model?.modelId && state.model?.source?.revision === record.model?.revision
      ? true
      : await importHuggingFaceModel();
    if (loaded) {
      if (state.runtime.latestRun) {
        applyRuntimeLedger(state.runtime.latestRun);
        renderRuntimeResult(state.runtime.latestRun);
      }
      if (state.debug.comparison) renderComparisonResult(state.debug.comparison);
      if (state.debug.trace) renderRootCauseTrace(state.debug.trace);
      if (state.debug.verification) renderVerificationResult(state.debug.verification);
      if (state.runtime.causalSweep) renderCausalSweep(state.runtime.causalSweep);
      setModelInterfaceOpen(true);
    }
    elements.debugCaseLibraryStatus.textContent = `Opened “${record.name}”.`;
  } catch (error) {
    elements.debugCaseLibraryStatus.textContent = String(error?.message ?? error);
  }
}

async function deleteSelectedDebugCase() {
  const caseId = elements.debugCaseSelect.value;
  const summary = state.debug.cases.find((item) => item.id === caseId);
  if (!caseId || !summary || !window.confirm(`Delete “${summary.name}” from this computer?`)) return;
  try {
    await debugApi(`/${caseId}`, { method: "DELETE" });
    if (state.debug.currentCase?.id === caseId) resetDebugCaseEditor();
    await restoreDebugCases();
    elements.debugCaseLibraryStatus.textContent = `Deleted “${summary.name}”.`;
  } catch (error) {
    elements.debugCaseLibraryStatus.textContent = String(error?.message ?? error);
  }
}

function setRuntimeBusy(busy, message = "") {
  state.runtime.busy = busy;
  elements.runtimeConnectButton.disabled = busy;
  elements.runtimeDisconnectButton.disabled = busy || !state.runtime.connected;
  elements.runtimeLoadButton.disabled = busy || !state.model || !state.runtime.reachable;
  elements.runtimeForwardButton.disabled = busy || !modelRunReady();
  elements.runtimeCompareButton.disabled = busy || !modelRunReady();
  elements.debugRunInterventionButton.disabled = busy || !selectedInterventionComponent() || !modelRunReady();
  elements.debugRunTraceButton.disabled = busy || !state.debug.comparison || !modelRunReady();
  elements.debugLoadActivationButton.disabled = busy || !elements.debugMicroscopeHook.value || !modelRunReady();
  elements.debugRunInboxButton.disabled = busy || !state.debug.inbox.length || !modelRunReady();
  elements.debugRunVerificationButton.disabled = busy || !state.debug.interventions.length || !modelRunReady();
  elements.runGenerationButton.disabled = busy || !modelRunReady();
  elements.runCausalSweepButton.disabled = busy || !state.runtime.latestRun || !modelRunReady();
  elements.compareRunsButton.disabled = busy || state.debug.runHistory.length < 2;
  elements.runtimePanel.setAttribute("aria-busy", String(busy));
  if (message) elements.runtimeStatus.textContent = message;
  if (busy && message) elements.modelRunStatus.textContent = message;
  updateModelRunReadiness();
}

function renderRuntimeConnection(payload = {}) {
  if (payload.provider === "daytona") {
    setRuntimeMode("daytona");
  } else if (payload.endpoint) {
    setRuntimeMode("local");
    elements.runtimeEndpointInput.value = payload.endpoint;
  }
  state.runtime.connected = Boolean(payload.connected);
  state.runtime.reachable = Boolean(payload.reachable);
  state.runtime.worker = payload.worker ?? null;
  state.runtime.quantization = payload.quantization ?? (payload.provider === "daytona" ? state.runtime.recommendation?.quantization : "none") ?? "none";
  const active = state.runtime.connected && state.runtime.reachable;
  elements.runtimePanel.dataset.state = active ? "connected" : state.runtime.connected ? "unreachable" : "disconnected";
  elements.runtimeBadge.textContent = active ? "Connected" : state.runtime.connected ? "Unavailable" : "Offline";
  elements.runtimeConnectFields.hidden = active;
  elements.runtimeControls.hidden = !active;
  elements.runtimeAccelerator.textContent = payload.worker?.accelerator ?? "—";
  elements.runtimeModel.textContent = payload.worker?.modelLoaded
    ? `${payload.worker.modelId}${payload.worker.dtype ? ` · ${payload.worker.dtype}` : ""}${payload.worker.quantization && payload.worker.quantization !== "none" ? ` · ${payload.worker.quantization}` : ""}`
    : "None loaded";
  elements.runtimeLoadButton.disabled = !state.model || state.runtime.busy;
  elements.runtimeForwardButton.disabled = !payload.worker?.modelLoaded || state.runtime.busy;
  elements.runtimeCompareButton.disabled = !payload.worker?.modelLoaded || state.runtime.busy;
  if (active) {
    elements.runtimeStatus.classList.remove("error");
    elements.runtimeStatus.textContent = payload.worker?.modelLoaded
      ? "Worker restored. Ready for a hooked forward pass."
      : "Worker connected. Load the checkpoint currently open in the graph.";
  } else if (state.runtime.connected) {
    elements.runtimeStatus.classList.add("error");
    elements.runtimeStatus.textContent = payload.error ?? "The saved worker session is not currently reachable.";
  } else {
    elements.runtimeStatus.classList.remove("error");
    elements.runtimeStatus.textContent = "No execution worker connected. Start a Daytona GPU or choose local execution.";
  }
  updateModelRunReadiness();
}

async function restoreRuntimeConnection() {
  try {
    const saved = await runtimeApi("/status");
    renderRuntimeConnection(saved);
  } catch (error) {
    renderRuntimeConnection({ connected: false, reachable: false });
    elements.runtimeStatus.classList.add("error");
    elements.runtimeStatus.textContent = String(error?.message ?? error);
  }
}

async function connectRuntime() {
  if (state.runtime.mode === "daytona") {
    await provisionDaytonaRuntime();
    return;
  }
  const endpoint = elements.runtimeEndpointInput.value.trim();
  const secret = elements.runtimeSecretInput.value.trim();
  if (!endpoint) {
    elements.runtimeStatus.classList.add("error");
    elements.runtimeStatus.textContent = "Start the local worker, then connect again.";
    elements.copyRuntimeCommandButton.focus();
    return;
  }
  setRuntimeBusy(true, "Checking the worker connection…");
  elements.runtimeStatus.classList.remove("error");
  try {
    const payload = await runtimeApi("/connect", { method: "POST", body: { endpoint, secret } });
    elements.runtimeSecretInput.value = "";
    renderRuntimeConnection(payload);
  } catch (error) {
    elements.runtimeStatus.classList.add("error");
    elements.runtimeStatus.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

async function provisionDaytonaRuntime() {
  const apiKey = elements.runtimeDaytonaApiKeyInput.value.trim();
  if (!apiKey) {
    elements.runtimeStatus.classList.add("error");
    elements.runtimeStatus.textContent = "Enter the Daytona API key for the account that should be billed.";
    elements.runtimeDaytonaApiKeyInput.focus();
    return;
  }
  setRuntimeBusy(true, "Provisioning a private Daytona GPU and installing the worker… this can take several minutes.");
  elements.runtimeStatus.classList.remove("error");
  try {
    const payload = await runtimeApi("/daytona/provision", { method: "POST", body: {
      ...daytonaModelDetails(),
      apiKey,
      hfToken: elements.runtimeDaytonaHfTokenInput.value.trim(),
      gpuType: elements.runtimeGpuSelect.value
    } });
    elements.runtimeDaytonaApiKeyInput.value = "";
    elements.runtimeDaytonaHfTokenInput.value = "";
    state.runtime.recommendation = payload.recommendation ?? state.runtime.recommendation;
    renderRuntimeConnection(payload);
  } catch (error) {
    elements.runtimeStatus.classList.add("error");
    elements.runtimeStatus.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

async function disconnectRuntime() {
  setRuntimeBusy(true, "Disconnecting worker…");
  try {
    const payload = await runtimeApi("/disconnect", { method: "POST", body: {} });
    state.runtime.latestRun = null;
    elements.runtimeResult.hidden = true;
    applyRuntimeCapabilityVisibility();
    renderRuntimeConnection({ connected: false, reachable: false });
    if (payload.warning) {
      elements.runtimeStatus.classList.add("error");
      elements.runtimeStatus.textContent = `${payload.warning} The sandbox still has its two-hour automatic deletion backstop.`;
    }
  } catch (error) {
    elements.runtimeStatus.classList.add("error");
    elements.runtimeStatus.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

async function loadRuntimeModel() {
  if (!state.model?.source?.modelId) return;
  setRuntimeBusy(true, `Loading ${state.model.source.modelId} into the worker…`);
  elements.runtimeStatus.classList.remove("error");
  try {
    const payload = await runtimeApi("/load", { method: "POST", body: {
      modelId: state.model.source.modelId,
      revision: state.model.source.sha ?? state.model.source.revision ?? "main",
      dtype: "auto",
      quantization: state.runtime.quantization,
      parameterCount: Math.max(0, (state.model.stats?.checkpointElements ?? state.model.stats?.totalParameters ?? 0) - (state.model.stats?.recognizedBufferElements ?? 0)),
      checkpointBytes: state.model.stats?.totalBytes ?? 0
    } });
    state.runtime.worker = { ...(state.runtime.worker ?? {}), modelLoaded: true, modelId: payload.modelId, revision: payload.revision, dtype: payload.dtype, device: payload.device, quantization: payload.quantization };
    state.runtime.latestRun = null;
    elements.runtimeResult.hidden = true;
    applyRuntimeCapabilityVisibility();
    elements.runtimeModel.textContent = `${payload.modelId} · ${payload.dtype}${payload.quantization && payload.quantization !== "none" ? ` · ${payload.quantization}` : ""}`;
    elements.runtimeStatus.textContent = `${payload.modelId} loaded on ${payload.device}.`;
    updateModelRunReadiness();
  } catch (error) {
    elements.runtimeStatus.classList.add("error");
    elements.runtimeStatus.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

function applyRuntimeLedger(run) {
  const ledger = state.model?.residualLedger;
  if (!ledger?.states?.length || !Array.isArray(run.layers)) return;
  ledger.mode = "observed";
  ledger.position = { label: run.positionLabel ?? `Token ${run.position}`, status: "observed" };
  ledger.metric = { label: "Direct logit attribution", status: "observed", targetToken: run.target?.text ?? run.target?.token ?? null };
  ledger.measurementNote = `Observed in hooked run ${run.runId.slice(0, 8)}. DLA is a component dot product with the selected unembedding row and excludes final normalization.`;
  const firstLayer = run.layers[0];
  const nodeMetrics = new Map();
  if (firstLayer && ledger.states[0]) {
    ledger.states[0].activationNorm = firstLayer.residPre?.norm ?? null;
    ledger.states[0].directLogitAttribution = firstLayer.residPre?.dla ?? null;
    nodeMetrics.set("residual_0", { kind: "residual", norm: firstLayer.residPre?.norm ?? null, dla: firstLayer.residPre?.dla ?? null, layer: null });
  }
  run.layers.forEach((layerRun) => {
    const ledgerState = ledger.states.find((entry) => entry.layer === layerRun.layer);
    if (!ledgerState) return;
    ledgerState.activationNorm = layerRun.residPost?.norm ?? null;
    ledgerState.directLogitAttribution = layerRun.residPost?.dla ?? null;
    const attention = ledgerState.writes?.find((write) => write.kind === "attention");
    const mlp = ledgerState.writes?.find((write) => write.kind === "mlp");
    if (attention && layerRun.attentionWrite) {
      attention.value = layerRun.attentionWrite.norm ?? null;
      attention.directLogitAttribution = layerRun.attentionWrite.dla ?? null;
      nodeMetrics.set(`l${layerRun.layer}_output`, { kind: "attention", norm: layerRun.attentionWrite.norm ?? null, dla: layerRun.attentionWrite.dla ?? null, layer: layerRun.layer });
    }
    if (mlp && layerRun.mlpWrite) {
      mlp.value = layerRun.mlpWrite.norm ?? null;
      mlp.directLogitAttribution = layerRun.mlpWrite.dla ?? null;
      nodeMetrics.set(`l${layerRun.layer}_mlp`, { kind: "mlp", norm: layerRun.mlpWrite.norm ?? null, dla: layerRun.mlpWrite.dla ?? null, layer: layerRun.layer });
    }
    nodeMetrics.set(`l${layerRun.layer}_mlp_residual`, { kind: "residual", norm: layerRun.residPost?.norm ?? null, dla: layerRun.residPost?.dla ?? null, layer: layerRun.layer });
  });
  state.runtime.nodeMetrics = nodeMetrics;
  state.runtime.maxWriteNorm = Math.max(0, ...[...nodeMetrics.values()].filter((entry) => entry.kind !== "residual" && Number.isFinite(entry.norm)).map((entry) => entry.norm));
  (attributionForRun(run).components ?? []).forEach((component) => {
    const metric = nodeMetrics.get(component.nodeId);
    if (metric) metric.share = component.shareOfAbsoluteMass;
  });
  state.runtime.maxContributionShare = Math.max(0, ...[...nodeMetrics.values()].filter((entry) => Number.isFinite(entry.share)).map((entry) => Math.abs(entry.share)));
  renderResidualLedger();
  state.renderedRange = null;
  renderFlowScene({ force: true });
}

function modelMetric(value, digits = 2) {
  return Number.isFinite(value) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value) : "—";
}

function makeChartText(textValue, attributes = {}) {
  const textNode = svgElement("text", attributes);
  textNode.textContent = textValue;
  return textNode;
}

function renderLineChart(container, series, { includeZero = false, xLabel = "layer", xValues = null } = {}) {
  container.replaceChildren();
  const width = 520;
  const height = 176;
  const padding = { left: 38, right: 14, top: 26, bottom: 27 };
  const allValues = series.flatMap((entry) => entry.values).filter(Number.isFinite);
  if (!allValues.length) {
    const empty = document.createElement("p");
    empty.className = "model-chart-empty";
    empty.textContent = "This measurement was not returned by the model.";
    container.append(empty);
    return;
  }
  let minimum = Math.min(...allValues);
  let maximum = Math.max(...allValues);
  if (includeZero) { minimum = Math.min(0, minimum); maximum = Math.max(0, maximum); }
  if (minimum === maximum) { minimum -= 1; maximum += 1; }
  const count = Math.max(2, ...series.map((entry) => entry.values.length));
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const x = (index) => padding.left + (index / (count - 1)) * plotWidth;
  const y = (value) => padding.top + ((maximum - value) / (maximum - minimum)) * plotHeight;
  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": series.map((entry) => entry.label).join(" and ") });
  for (let index = 0; index <= 3; index += 1) {
    const value = maximum - ((maximum - minimum) * index) / 3;
    const gridY = y(value);
    svg.append(svgElement("line", { class: "model-chart-gridline", x1: padding.left, x2: width - padding.right, y1: gridY, y2: gridY }));
    svg.append(makeChartText(modelMetric(value), { class: "model-chart-axis", x: padding.left - 7, y: gridY + 3, "text-anchor": "end" }));
  }
  if (includeZero && minimum < 0 && maximum > 0) svg.append(svgElement("line", { class: "model-chart-zero", x1: padding.left, x2: width - padding.right, y1: y(0), y2: y(0) }));
  series.forEach((entry, seriesIndex) => {
    const validPoints = entry.values.map((value, index) => Number.isFinite(value) ? { value, index } : null).filter(Boolean);
    const path = validPoints.map((point, index) => `${index ? "L" : "M"}${x(point.index).toFixed(2)} ${y(point.value).toFixed(2)}`).join(" ");
    svg.append(svgElement("path", { class: `model-chart-line series-${seriesIndex}`, d: path }));
    if (validPoints.length <= 48) validPoints.forEach((point) => svg.append(svgElement("circle", { class: `model-chart-point series-${seriesIndex}`, cx: x(point.index), cy: y(point.value), r: 2.2 })));
    const legendX = padding.left + seriesIndex * 130;
    svg.append(svgElement("line", { class: `model-chart-legend-line series-${seriesIndex}`, x1: legendX, x2: legendX + 15, y1: 12, y2: 12 }));
    svg.append(makeChartText(entry.label, { class: "model-chart-legend", x: legendX + 20, y: 15 }));
  });
  svg.append(makeChartText(String(xValues?.[0] ?? 0), { class: "model-chart-axis", x: padding.left, y: height - 8, "text-anchor": "middle" }));
  svg.append(makeChartText(String(xValues?.[count - 1] ?? count - 1), { class: "model-chart-axis", x: width - padding.right, y: height - 8, "text-anchor": "middle" }));
  svg.append(makeChartText(xLabel, { class: "model-chart-axis-label", x: width / 2, y: height - 7, "text-anchor": "middle" }));
  container.append(svg);
}

function renderHistogram(container, values, {
  label,
  symmetric = false,
  minimumCount = 2,
  emptyMessage = "At least two finite measurements are required for a distribution.",
} = {}) {
  container.replaceChildren();
  const histogram = buildHistogram(values, { symmetric });
  if (histogram.count < minimumCount || !histogram.domain) {
    const empty = document.createElement("p");
    empty.className = "model-chart-empty";
    empty.textContent = emptyMessage;
    container.append(empty);
    return histogram;
  }

  const width = 520;
  const height = 176;
  const padding = { left: 38, right: 14, top: 18, bottom: 31 };
  const [minimum, maximum] = histogram.domain;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const maximumCount = Math.max(1, ...histogram.bins.map((bin) => bin.count));
  const x = (value) => padding.left + ((value - minimum) / (maximum - minimum)) * plotWidth;
  const y = (count) => padding.top + (1 - count / maximumCount) * plotHeight;
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `${label ?? "Measurement"} histogram with ${histogram.count} observations`,
  });

  const tickCounts = [...new Set([0, Math.ceil(maximumCount / 2), maximumCount])].sort((left, right) => left - right);
  tickCounts.forEach((count) => {
    const gridY = y(count);
    svg.append(svgElement("line", { class: "model-chart-gridline", x1: padding.left, x2: width - padding.right, y1: gridY, y2: gridY }));
    svg.append(makeChartText(String(count), { class: "model-chart-axis", x: padding.left - 7, y: gridY + 3, "text-anchor": "end" }));
  });

  const { q1, median, q3 } = histogram.statistics;
  svg.append(svgElement("rect", {
    class: "model-histogram-iqr",
    x: x(q1),
    y: padding.top,
    width: Math.max(0.5, x(q3) - x(q1)),
    height: plotHeight,
  }));
  const binWidth = plotWidth / histogram.bins.length;
  histogram.bins.forEach((bin, index) => {
    const bar = svgElement("rect", {
      class: `model-histogram-bar${symmetric && (bin.start + bin.end) / 2 < 0 ? " negative" : ""}`,
      x: padding.left + index * binWidth + 1,
      y: y(bin.count),
      width: Math.max(1, binWidth - 2),
      height: Math.max(0, padding.top + plotHeight - y(bin.count)),
    });
    const title = svgElement("title");
    title.textContent = `${modelMetric(bin.start, 4)} to ${modelMetric(bin.end, 4)}: ${bin.count} observation${bin.count === 1 ? "" : "s"}`;
    bar.append(title);
    svg.append(bar);
  });
  if (minimum < 0 && maximum > 0) {
    svg.append(svgElement("line", { class: "model-chart-zero", x1: x(0), x2: x(0), y1: padding.top, y2: padding.top + plotHeight }));
  }
  svg.append(svgElement("line", { class: "model-histogram-median", x1: x(median), x2: x(median), y1: padding.top, y2: padding.top + plotHeight }));
  svg.append(makeChartText(modelMetric(minimum, 3), { class: "model-chart-axis", x: padding.left, y: height - 10, "text-anchor": "start" }));
  if (minimum < 0 && maximum > 0) svg.append(makeChartText("0", { class: "model-chart-axis", x: x(0), y: height - 10, "text-anchor": "middle" }));
  svg.append(makeChartText(modelMetric(maximum, 3), { class: "model-chart-axis", x: width - padding.right, y: height - 10, "text-anchor": "end" }));
  svg.append(makeChartText(label ?? "value", { class: "model-chart-axis-label", x: width / 2, y: height - 8, "text-anchor": "middle" }));

  const stats = document.createElement("div");
  stats.className = "model-histogram-stats";
  [
    ["n", histogram.count],
    ["median", modelMetric(median, 4)],
    ["IQR", `${modelMetric(q1, 4)}–${modelMetric(q3, 4)}`],
    ["mean", modelMetric(histogram.statistics.mean, 4)],
  ].forEach(([name, value]) => {
    const item = document.createElement("span");
    item.textContent = `${name} ${value}`;
    stats.append(item);
  });
  container.append(svg, stats);
  return histogram;
}

function renderMetricDistributions(run) {
  const currentSnapshot = state.debug.runHistory.find((item) => item.id === (run.runId ?? run.generationId));
  const compatibleMetrics = currentSnapshot
    ? state.debug.runHistory
      .filter((item) => item.modelId === currentSnapshot.modelId
        && item.revision === currentSnapshot.revision
        && item.metricIdentity === currentSnapshot.metricIdentity
        && Number.isFinite(item.metric?.value))
      .map((item) => item.metric.value)
    : [];
  const attribution = attributionForRun(run);
  const componentDla = (attribution.components ?? []).map((component) => component.dla);
  const residualNorms = (run.layers ?? []).map((layer) => layer.residPost?.norm);
  elements.metricDistributionScope.textContent = `${componentDla.filter(Number.isFinite).length} components · ${compatibleMetrics.length} compatible run${compatibleMetrics.length === 1 ? "" : "s"}`;
  renderHistogram(elements.metricHistoryHistogram, compatibleMetrics, {
    label: `${run.metric?.name ?? "Behaviour metric"} value`,
    emptyMessage: "Run this exact model revision, target, and metric at least twice to see a comparable distribution.",
  });
  renderHistogram(elements.componentDlaHistogram, componentDla, {
    label: "signed DLA",
    symmetric: true,
    emptyMessage: "At least two target-token component attributions are required.",
  });
  renderHistogram(elements.residualNormHistogram, residualNorms, {
    label: "residual norm",
    emptyMessage: "At least two residual-state norms are required.",
  });
}

function renderInferenceWaterfall(run) {
  elements.inferenceWaterfall.replaceChildren();
  const performanceRecord = run?.performance ?? {};
  const profile = performanceRecord.waterfall;
  const phases = (profile?.phases ?? []).filter((phase) =>
    typeof phase?.label === "string"
    && Number.isFinite(phase.startMs)
    && Number.isFinite(phase.durationMs)
    && phase.startMs >= 0
    && phase.durationMs >= 0
  );
  const derivedTotal = phases.reduce((maximum, phase) => Math.max(maximum, phase.startMs + phase.durationMs), 0);
  const total = Number.isFinite(profile?.totalMs) && profile.totalMs > 0 ? profile.totalMs : derivedTotal;
  if (!phases.length || total <= 0) {
    elements.inferenceWaterfallTotal.textContent = "Unavailable";
    const empty = document.createElement("p");
    empty.className = "model-chart-empty";
    empty.textContent = "This worker did not return a phase-level inference profile.";
    elements.inferenceWaterfall.append(empty);
    elements.inferenceWaterfallNote.textContent = "Restart the bundled worker to enable synchronized inference timings.";
    return;
  }

  const forward = phases.find((phase) => phase.key === "model-forward");
  const roundTrip = performanceRecord.clientRoundTripMs;
  const totalParts = [`${modelMetric(total, 1)} ms worker`];
  if (Number.isFinite(forward?.durationMs)) totalParts.push(`${modelMetric(forward.durationMs, 1)} ms forward`);
  if (Number.isFinite(roundTrip)) totalParts.push(`${modelMetric(roundTrip, 1)} ms round trip`);
  elements.inferenceWaterfallTotal.textContent = totalParts.join(" · ");

  const axis = document.createElement("div");
  axis.className = "inference-waterfall-axis";
  const axisSpacer = document.createElement("span");
  const scale = document.createElement("div");
  const axisMetric = document.createElement("span");
  axisMetric.textContent = "duration · share";
  [0, 0.25, 0.5, 0.75, 1].forEach((fraction) => {
    const tick = document.createElement("i");
    tick.style.setProperty("--waterfall-tick", `${fraction * 100}%`);
    tick.textContent = `${modelMetric(total * fraction, total < 10 ? 1 : 0)} ms`;
    scale.append(tick);
  });
  axis.append(axisSpacer, scale, axisMetric);
  elements.inferenceWaterfall.append(axis);

  phases.forEach((phase) => {
    const row = document.createElement("div");
    row.className = "inference-waterfall-row";
    row.dataset.category = phase.category || "cpu";
    const identity = document.createElement("div");
    const label = document.createElement("strong");
    const category = document.createElement("small");
    label.textContent = phase.label;
    category.textContent = phase.category || "worker";
    identity.append(label, category);
    const track = document.createElement("div");
    track.className = "inference-waterfall-track";
    const bar = document.createElement("i");
    const startPercent = Math.min(100, Math.max(0, phase.startMs / total * 100));
    const durationPercent = Math.min(100 - startPercent, Math.max(0, phase.durationMs / total * 100));
    bar.style.setProperty("--waterfall-start", `${startPercent}%`);
    bar.style.setProperty("--waterfall-duration", `${durationPercent}%`);
    const share = phase.durationMs / total;
    const accessible = `${phase.label}: ${modelMetric(phase.durationMs, 2)} milliseconds, ${modelMetric(share * 100, 1)} percent of worker time.`;
    bar.setAttribute("aria-label", accessible);
    bar.title = `${accessible} ${phase.detail ?? ""}`.trim();
    track.append(bar);
    const metric = document.createElement("div");
    const duration = document.createElement("strong");
    const percent = document.createElement("small");
    duration.textContent = `${modelMetric(phase.durationMs, phase.durationMs < 10 ? 2 : 1)} ms`;
    percent.textContent = `${modelMetric(share * 100, 1)}%`;
    metric.append(duration, percent);
    row.title = phase.detail ?? "";
    row.append(identity, track, metric);
    elements.inferenceWaterfall.append(row);
  });

  const transportNote = Number.isFinite(roundTrip)
    ? ` Browser-observed round trip: ${modelMetric(roundTrip, 1)} ms; transport, proxying, and JSON work are not placed on the worker clock.`
    : "";
  elements.inferenceWaterfallNote.textContent = `${profile.note ?? "Worker-side phase timings."}${transportNote}`;
}

function renderModelRunSummary(run) {
  elements.modelRunSummary.replaceChildren();
  const cacheBytes = run.kvCache?.bytes;
  const topPrediction = run.nextToken?.topK?.[0];
  const deviceMemory = run.performance?.deviceMemory;
  const memoryDetail = Number.isFinite(deviceMemory?.peakAllocatedBytes)
    ? `${formatBytes(deviceMemory.peakAllocatedBytes)} peak device memory`
    : `${deviceMemory?.kind?.toUpperCase?.() ?? "Worker"} execution`;
  const entries = [
    ["Next token", topPrediction?.text || topPrediction?.token || "—", Number.isFinite(topPrediction?.probability) ? `${modelMetric(topPrediction.probability * 100, 1)}% probability` : "Top prediction"],
    ["Target rank", Number.isFinite(run.target?.rank) ? `#${new Intl.NumberFormat().format(run.target.rank)}` : "—", run.target?.text || run.target?.token || "Selected target"],
    ["Prompt", `${run.tokens?.length ?? 0} tokens`, run.positionLabel ?? "Last position"],
    ["KV cache", Number.isFinite(cacheBytes) ? formatBytes(cacheBytes) : "Unavailable", `${run.kvCache?.tensors?.length ?? 0} cached tensors`],
    ["Runtime", Number.isFinite(run.performance?.durationMs) ? `${modelMetric(run.performance.durationMs, 0)} ms` : "—", memoryDetail],
    ["Captures", new Intl.NumberFormat().format(run.hooks?.length ?? 0), "attention + MLP hooks"]
  ];
  entries.forEach(([label, value, detail]) => {
    const item = document.createElement("div");
    const small = document.createElement("small");
    const strong = document.createElement("strong");
    const note = document.createElement("span");
    small.textContent = label;
    strong.textContent = value;
    note.textContent = detail;
    item.append(small, strong, note);
    elements.modelRunSummary.append(item);
  });
}

function renderModelTokens(run) {
  elements.modelTokenStrip.replaceChildren();
  elements.modelTokenContext.textContent = `Position ${run.position ?? "—"} selected`;
  (run.tokens ?? []).forEach((token) => {
    const chip = document.createElement("span");
    if (token.index === run.position) chip.className = "selected";
    const index = document.createElement("small");
    const textNode = document.createElement("strong");
    index.textContent = String(token.index);
    textNode.textContent = token.text || token.token || "∅";
    chip.title = `Token ID ${token.id} · ${token.token}`;
    chip.append(index, textNode);
    elements.modelTokenStrip.append(chip);
  });
}

function renderModelPredictions(run) {
  elements.modelPredictionList.replaceChildren();
  elements.modelPredictionEntropy.textContent = `Entropy ${modelMetric(run.nextToken?.entropy, 3)} nats`;
  const predictions = run.nextToken?.topK ?? [];
  const maximum = Math.max(...predictions.map((entry) => entry.probability ?? 0), 1e-12);
  predictions.forEach((prediction, index) => {
    const row = document.createElement("div");
    const rank = document.createElement("span");
    const token = document.createElement("strong");
    const bar = document.createElement("i");
    const probability = document.createElement("b");
    rank.textContent = String(index + 1).padStart(2, "0");
    token.textContent = prediction.text || prediction.token || "∅";
    bar.style.setProperty("--prediction-width", `${Math.max(1, (prediction.probability / maximum) * 100)}%`);
    probability.textContent = `${modelMetric(prediction.probability * 100, 2)}%`;
    row.title = `Token ${prediction.tokenId} · logit ${modelMetric(prediction.logit, 3)}`;
    row.append(rank, token, bar, probability);
    elements.modelPredictionList.append(row);
  });
}

function attributionForRun(run) {
  if (run.attribution?.components?.length) return run.attribution;
  const components = [];
  const firstLayer = run.layers?.[0];
  if (Number.isFinite(firstLayer?.residPre?.dla)) components.push({ id: "embedding", nodeId: "residual_0", label: "Token embedding / initial residual", kind: "embedding", layer: null, dla: firstLayer.residPre.dla, norm: firstLayer.residPre.norm });
  (run.layers ?? []).forEach((layer) => {
    if (Number.isFinite(layer.attentionWrite?.dla)) components.push({ id: `attention.${layer.layer}`, nodeId: `l${layer.layer}_output`, label: `Layer ${layer.layer} Attention write`, kind: "attention", layer: layer.layer, dla: layer.attentionWrite.dla, norm: layer.attentionWrite.norm });
    if (Number.isFinite(layer.mlpWrite?.dla)) components.push({ id: `mlp.${layer.layer}`, nodeId: `l${layer.layer}_mlp`, label: `Layer ${layer.layer} MLP write`, kind: "mlp", layer: layer.layer, dla: layer.mlpWrite.dla, norm: layer.mlpWrite.norm });
  });
  const absoluteMass = components.reduce((sum, component) => sum + Math.abs(component.dla), 0);
  components.forEach((component) => { component.shareOfAbsoluteMass = absoluteMass ? component.dla / absoluteMass : null; });
  const capturedRawSum = components.reduce((sum, component) => sum + component.dla, 0);
  return {
    method: "raw-unembedding-dot-product",
    targetLogit: run.target?.logit,
    capturedRawSum,
    normalizationAndBiasGap: Number.isFinite(run.target?.logit) ? run.target.logit - capturedRawSum : null,
    positiveTotal: components.reduce((sum, component) => sum + Math.max(0, component.dla), 0),
    negativeTotal: components.reduce((sum, component) => sum + Math.min(0, component.dla), 0),
    absoluteMass,
    components,
    note: "Relative shares compare the signed raw DLA of captured components. They are observational, not causal percentages of the output."
  };
}

function renderModelAttribution(run) {
  const attribution = attributionForRun(run);
  elements.modelAttributionMethod.textContent = attribution.method === "raw-unembedding-dot-product" ? "Raw unembedding DLA" : attribution.method ?? "Direct logit attribution";
  elements.modelAttributionSummary.replaceChildren();
  const entries = [
    ["Target logit", shortObservedValue(attribution.targetLogit, true)],
    ["Captured raw sum", shortObservedValue(attribution.capturedRawSum, true)],
    ["Positive support", shortObservedValue(attribution.positiveTotal, true)],
    ["Negative suppression", shortObservedValue(attribution.negativeTotal, true)],
    ["Norm / bias gap", shortObservedValue(attribution.normalizationAndBiasGap, true)]
  ];
  entries.forEach(([label, value]) => {
    const item = document.createElement("div");
    const small = document.createElement("small");
    const strong = document.createElement("strong");
    small.textContent = label;
    strong.textContent = value;
    item.append(small, strong);
    elements.modelAttributionSummary.append(item);
  });
  elements.modelContributionChart.replaceChildren();
  const components = [...(attribution.components ?? [])].sort((left, right) => Math.abs(right.dla) - Math.abs(left.dla));
  const maxMagnitude = Math.max(...components.map((component) => Math.abs(component.dla)), 1e-12);
  components.forEach((component, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `model-contribution-row ${component.dla >= 0 ? "positive" : "negative"}`;
    const rank = document.createElement("span");
    const identity = document.createElement("strong");
    const track = document.createElement("i");
    const bar = document.createElement("b");
    const value = document.createElement("em");
    const share = document.createElement("small");
    rank.textContent = String(index + 1).padStart(2, "0");
    identity.textContent = component.label;
    bar.style.setProperty("--contribution-width", `${Math.max(1, Math.abs(component.dla) / maxMagnitude * 50)}%`);
    track.append(bar);
    value.textContent = shortObservedValue(component.dla, true);
    share.textContent = Number.isFinite(component.shareOfAbsoluteMass)
      ? `${component.shareOfAbsoluteMass > 0 ? "+" : component.shareOfAbsoluteMass < 0 ? "−" : ""}${modelMetric(Math.abs(component.shareOfAbsoluteMass) * 100, 1)}% of |DLA|`
      : "—";
    row.title = `Focus ${component.label} on the circuit graph`;
    row.append(rank, identity, track, value, share);
    row.addEventListener("click", () => {
      setModelInterfaceOpen(false);
      focusGraphNode(component.nodeId);
    });
    elements.modelContributionChart.append(row);
  });
  if (!components.length) {
    const empty = document.createElement("p");
    empty.className = "model-chart-empty";
    empty.textContent = "No component writes with target-token attribution were captured.";
    elements.modelContributionChart.append(empty);
  }
  const layers = run.layers ?? [];
  renderLineChart(elements.componentDlaChart, [
    { label: "Attention DLA", values: layers.map((layer) => layer.attentionWrite?.dla) },
    { label: "MLP DLA", values: layers.map((layer) => layer.mlpWrite?.dla) }
  ], { includeZero: true });
  elements.modelAttributionNote.textContent = attribution.note ?? "DLA is observational. Validate important components with an intervention such as activation patching or ablation.";
}

function renderAttentionEntropy(run) {
  elements.attentionEntropyMap.replaceChildren();
  const attention = run.attention ?? [];
  if (!attention.length) {
    const note = document.createElement("p");
    note.className = "model-chart-empty";
    note.textContent = "Attention probabilities were not returned by this model implementation.";
    elements.attentionEntropyMap.append(note);
    return;
  }
  const maxEntropy = Math.log(Math.max(2, run.tokens?.length ?? 2));
  attention.forEach((layer) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    label.textContent = `L${layer.layer}`;
    row.append(label);
    (layer.heads ?? []).forEach((head) => {
      const cell = document.createElement("i");
      const sharpness = Math.max(0, Math.min(1, 1 - head.entropy / maxEntropy));
      cell.style.setProperty("--head-strength", String(0.12 + sharpness * 0.88));
      cell.title = `Layer ${layer.layer}, head ${head.head} · entropy ${modelMetric(head.entropy, 3)} · top source ${head.topSourceToken || head.topSourceIndex} (${modelMetric((head.topSourceWeight ?? 0) * 100, 1)}%)`;
      row.append(cell);
    });
    elements.attentionEntropyMap.append(row);
  });
}

function renderHookInventory(run) {
  elements.modelHookTable.replaceChildren();
  const hooks = run.hooks ?? [];
  elements.modelHookCount.textContent = `${hooks.length} component captures · ${run.kvCache?.tensors?.length ?? 0} cache tensors`;
  hooks.slice(0, 64).forEach((hook) => {
    const row = document.createElement("div");
    const order = document.createElement("span");
    const identity = document.createElement("strong");
    const shape = document.createElement("code");
    const norm = document.createElement("b");
    order.textContent = String(hook.order + 1).padStart(2, "0");
    identity.textContent = `L${hook.layer} · ${hook.category}`;
    shape.textContent = `[${(hook.shape ?? []).join(" × ")}]`;
    norm.textContent = `${modelMetric(hook.durationMs, 2)} ms`;
    norm.title = `Activation ${formatBytes(hook.activationBytes ?? 0)}${Number.isFinite(hook.deviceAllocatedBytes) ? ` · device allocated ${formatBytes(hook.deviceAllocatedBytes)}` : ""} · norm ${modelMetric(hook.norm, 3)}`;
    row.title = hook.hookName;
    row.append(order, identity, shape, norm);
    elements.modelHookTable.append(row);
  });
  if (hooks.length > 64) {
    const note = document.createElement("p");
    note.textContent = `${hooks.length - 64} additional captures retained by the worker.`;
    elements.modelHookTable.append(note);
  }
}

function renderLogitLens(lens) {
  elements.logitLensStages.replaceChildren();
  elements.logitLensChart.replaceChildren();
  if (!lens?.available || !lens.stages?.length) {
    elements.logitLensMethod.textContent = lens?.disabled ? "Not requested" : "Unavailable";
    const empty = document.createElement("p");
    empty.className = "model-chart-empty";
    empty.textContent = lens?.evidence?.note ?? "This architecture did not expose a compatible residual-to-vocabulary projection.";
    elements.logitLensChart.append(empty);
    elements.logitLensNote.textContent = lens?.evidence?.note ?? "No logit-lens trace was returned.";
    return;
  }
  elements.logitLensMethod.textContent = lens.method === "normalized-logit-lens" ? "Final-norm lens" : "Raw unembedding lens";
  const probabilityLabel = lens.method === "normalized-logit-lens" ? "Target probability (%)" : "Raw-lens softmax (%)";
  renderLineChart(elements.logitLensChart, [
    { label: probabilityLabel, values: lens.stages.map((stage) => Number.isFinite(stage.targetProbability) ? stage.targetProbability * 100 : null) }
  ], { includeZero: true, xLabel: "sampled residual stage", xValues: lens.stages.map((stage) => stage.index) });
  lens.stages.forEach((stage) => {
    const item = document.createElement("article");
    item.className = "logit-lens-stage";
    const label = document.createElement("strong");
    const prediction = document.createElement("span");
    const detail = document.createElement("small");
    label.textContent = stage.label;
    const top = stage.topK?.[0];
    prediction.textContent = `${top?.text || top?.token || "—"} · ${Number.isFinite(top?.probability) ? `${modelMetric(top.probability * 100, 1)}%` : "—"}`;
    detail.textContent = `target rank ${Number.isFinite(stage.targetRank) ? `#${stage.targetRank}` : "—"} · KL(final‖lens) ${modelMetric(stage.finalToLensKL, 3)}`;
    item.append(label, prediction, detail);
    elements.logitLensStages.append(item);
  });
  elements.logitLensNote.textContent = `${lens.sampled ? `${lens.stages.length} of ${lens.totalStages} stages sampled. ` : ""}${lens.evidence?.note ?? "Logit-lens results are observational."}`;
}

function applyRuntimeCapabilityVisibility(run = null) {
  const capabilities = run?.diagnostics?.capabilities;
  document.querySelectorAll("[data-runtime-capability]").forEach((surface) => {
    const required = String(surface.dataset.runtimeCapability ?? "").split(",").map((value) => value.trim()).filter(Boolean);
    surface.hidden = capabilities ? required.some((capability) => capabilities[capability] !== true) : false;
  });
}

function renderRuntimeResult(run) {
  applyRuntimeCapabilityVisibility(run);
  renderModelRunSummary(run);
  renderInferenceWaterfall(run);
  renderModelTokens(run);
  renderModelPredictions(run);
  renderLogitLens(run.logitLens);
  renderModelAttribution(run);
  renderMetricDistributions(run);
  const layers = run.layers ?? [];
  renderLineChart(elements.residualNormChart, [{ label: "Residual norm", values: layers.map((layer) => layer.residPost?.norm) }]);
  renderLineChart(elements.residualDlaChart, [{ label: "Target DLA", values: layers.map((layer) => layer.residPost?.dla) }], { includeZero: true });
  renderLineChart(elements.residualDeltaChart, [{ label: "Residual update", values: layers.map((layer) => layer.residualDelta?.norm) }]);
  renderLineChart(elements.residualCosineChart, [{ label: "Cosine to final", values: layers.map((layer) => layer.residPost?.cosineToFinal) }]);
  renderLineChart(elements.componentNormChart, [
    { label: "Attention write", values: layers.map((layer) => layer.attentionWrite?.norm) },
    { label: "MLP write", values: layers.map((layer) => layer.mlpWrite?.norm) }
  ]);
  renderAttentionEntropy(run);
  renderHookInventory(run);
  renderDebugDiagnostics(run);
  populateDebugComponentControls(run, state.debug.comparison);
  elements.runtimeResult.hidden = false;
}

function appendDebugSummary(container, label, value, detail = "") {
  const item = document.createElement("div");
  const small = document.createElement("small");
  const strong = document.createElement("strong");
  const note = document.createElement("span");
  small.textContent = label;
  strong.textContent = value;
  note.textContent = detail;
  item.append(small, strong, note);
  container.append(item);
}

function renderDebugDiagnostics(run) {
  elements.debugDiagnostics.replaceChildren();
  const anomalies = run?.diagnostics?.anomalies ?? [];
  const unsupported = run?.diagnostics?.unsupported ?? [];
  elements.debugDiagnosticCount.textContent = anomalies.length
    ? `${anomalies.length} finding${anomalies.length === 1 ? "" : "s"}`
    : unsupported.length ? `${unsupported.length} unsupported` : "No issues";
  anomalies.forEach((anomaly) => {
    const row = document.createElement("div");
    row.className = anomaly.severity ?? "notice";
    const mark = document.createElement("span");
    const message = document.createElement("strong");
    const location = document.createElement("small");
    mark.textContent = anomaly.severity === "error" ? "!" : anomaly.severity === "warning" ? "△" : "i";
    message.textContent = anomaly.message;
    location.textContent = Number.isFinite(anomaly.layer) ? `Layer ${anomaly.layer}${Number.isFinite(anomaly.head) ? ` · head ${anomaly.head}` : ""}` : anomaly.kind;
    row.append(mark, message, location);
    elements.debugDiagnostics.append(row);
  });
  unsupported.forEach((capability) => {
    const row = document.createElement("div");
    row.className = "notice";
    const mark = document.createElement("span");
    const message = document.createElement("strong");
    const location = document.createElement("small");
    mark.textContent = "—";
    message.textContent = `${capability} unavailable`;
    location.textContent = "Architecture capability";
    row.append(mark, message, location);
    elements.debugDiagnostics.append(row);
  });
  if (!anomalies.length && !unsupported.length) {
    const row = document.createElement("div");
    row.className = "notice";
    const mark = document.createElement("span");
    const message = document.createElement("strong");
    const location = document.createElement("small");
    mark.textContent = "✓";
    message.textContent = "No numerical, cache, or hook anomalies were detected.";
    location.textContent = "Heuristic runtime checks";
    row.append(mark, message, location);
    elements.debugDiagnostics.append(row);
  }
}

function populateDebugComponentControls(run, comparison = null) {
  const selected = elements.debugInterventionComponent.value;
  const components = comparison?.divergence?.components?.length
    ? comparison.divergence.components
    : (run?.layers ?? []).flatMap((layer) => [
      { kind: "residual", layer: layer.layer, nodeId: `l${layer.layer}_mlp_residual`, label: `Layer ${layer.layer} residual output` },
      ...(layer.attentionWrite ? [{ kind: "attention", layer: layer.layer, nodeId: `l${layer.layer}_output`, label: `Layer ${layer.layer} attention output` }] : []),
      ...(layer.mlpWrite ? [{ kind: "mlp", layer: layer.layer, nodeId: `l${layer.layer}_mlp`, label: `Layer ${layer.layer} MLP output` }] : [])
    ]);
  elements.debugInterventionComponent.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = components.length ? "Select a component" : "Run or compare a case first";
  elements.debugInterventionComponent.append(placeholder);
  const seen = new Set();
  components.forEach((component) => {
    const key = `${component.kind}:${component.layer}:${component.nodeId}`;
    if (seen.has(key)) return;
    seen.add(key);
    const option = document.createElement("option");
    option.value = key;
    option.textContent = component.label;
    elements.debugInterventionComponent.append(option);
  });
  if ([...elements.debugInterventionComponent.options].some((option) => option.value === selected)) elements.debugInterventionComponent.value = selected;
  elements.debugRunInterventionButton.disabled = !elements.debugInterventionComponent.value || !modelRunReady() || state.runtime.busy;
  elements.debugRunTraceButton.disabled = !comparison || !modelRunReady() || state.runtime.busy;
  elements.debugRunVerificationButton.disabled = !state.debug.interventions.length || !modelRunReady() || state.runtime.busy;

  const previousHook = elements.debugMicroscopeHook.value;
  elements.debugMicroscopeHook.replaceChildren();
  const hookPlaceholder = document.createElement("option");
  hookPlaceholder.value = "";
  hookPlaceholder.textContent = run ? "Select a captured activation" : "Run a case first";
  elements.debugMicroscopeHook.append(hookPlaceholder);
  (run?.layers ?? []).forEach((layer) => {
    const entries = [
      [`resid_pre.${layer.layer}`, `Layer ${layer.layer} residual input`],
      [`resid_post.${layer.layer}`, `Layer ${layer.layer} residual output`],
      ...(layer.attentionWrite ? [[`attention_write.${layer.layer}`, `Layer ${layer.layer} attention output`]] : []),
      ...(layer.attentionHeadOutputs ? [[`attention_head_outputs.${layer.layer}`, `Layer ${layer.layer} pre-projection head outputs`]] : []),
      ...(layer.mlpWrite ? [[`mlp_write.${layer.layer}`, `Layer ${layer.layer} MLP output`]] : []),
      ...((run.attention ?? []).some((item) => item.layer === layer.layer) ? [[`attention_probs.${layer.layer}`, `Layer ${layer.layer} attention probabilities`]] : [])
    ];
    entries.forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      elements.debugMicroscopeHook.append(option);
    });
  });
  if ([...elements.debugMicroscopeHook.options].some((option) => option.value === previousHook)) elements.debugMicroscopeHook.value = previousHook;
  elements.debugLoadActivationButton.disabled = !elements.debugMicroscopeHook.value || !run || !modelRunReady() || state.runtime.busy;
}

function applyComparisonOverlay(comparison) {
  const nodeMetrics = state.runtime.nodeMetrics;
  const components = comparison?.divergence?.components ?? [];
  state.debug.maxDivergence = Math.max(0, ...components.map((item) => Number(item.score) || 0));
  components.forEach((component) => {
    const current = nodeMetrics.get(component.nodeId) ?? { kind: component.kind, layer: component.layer };
    current.divergence = component.score;
    current.comparison = component;
    nodeMetrics.set(component.nodeId, current);
  });
  if (!state.model) return;
  state.renderedRange = null;
  renderFlowScene({ force: true });
}

function renderComparisonResult(comparison) {
  state.debug.comparison = comparison;
  elements.debugComparisonResult.hidden = false;
  elements.debugComparisonMetric.textContent = comparison.metric?.name ?? "Behaviour metric";
  elements.debugComparisonSummary.replaceChildren();
  appendDebugSummary(elements.debugComparisonSummary, "Selected", shortObservedValue(comparison.metric?.failureValue, true), "metric value");
  appendDebugSummary(elements.debugComparisonSummary, "Reference", shortObservedValue(comparison.metric?.controlValue, true), "metric value");
  appendDebugSummary(elements.debugComparisonSummary, "Difference", shortObservedValue(comparison.metric?.failureMinusControl, true), "selected − reference");
  appendDebugSummary(elements.debugComparisonSummary, "Output KL", modelMetric(comparison.divergence?.outputKL, 4), "reference → selected");
  appendDebugSummary(elements.debugComparisonSummary, "First divergence", Number.isFinite(comparison.divergence?.firstMaterialLayer) ? `L${comparison.divergence.firstMaterialLayer}` : "—", "heuristic threshold");
  appendDebugSummary(elements.debugComparisonSummary, "Cache delta", formatBytes(Math.abs(comparison.divergence?.cacheBytesDifference ?? 0)), comparison.divergence?.cacheBytesDifference < 0 ? "selected uses less" : "selected uses more");
  appendDebugSummary(elements.debugComparisonSummary, "Top prediction", `${comparison.outputs?.failureTopPrediction?.text ?? "—"} ↔ ${comparison.outputs?.controlTopPrediction?.text ?? "—"}`, "selected ↔ reference");
  appendDebugSummary(elements.debugComparisonSummary, "Target logit Δ", shortObservedValue(comparison.outputs?.targetLogitDifference, true), "selected − reference");
  elements.debugTokenAlignment.replaceChildren();
  (comparison.tokenAlignment ?? []).forEach((entry) => {
    const chip = document.createElement("span");
    chip.className = entry.status;
    const failureText = entry.failureToken?.text ?? entry.failureToken?.token ?? "∅";
    const controlText = entry.controlToken?.text ?? entry.controlToken?.token ?? "∅";
    chip.textContent = entry.status === "matched" ? failureText : `${failureText} ↔ ${controlText}`;
    chip.title = `Selected ${entry.failureIndex ?? "—"} · reference ${entry.controlIndex ?? "—"}`;
    elements.debugTokenAlignment.append(chip);
  });
  const orderedLayers = [...(comparison.divergence?.layers ?? [])].sort((left, right) => left.layer - right.layer);
  renderLineChart(elements.debugDivergenceChart, [
    { label: "Residual distance", values: orderedLayers.map((layer) => layer.residual?.relativeDistance) },
    { label: "Attention change", values: orderedLayers.map((layer) => layer.attention?.meanAbsoluteDifference) }
  ]);
  renderHistogram(
    elements.debugDivergenceHistogram,
    (comparison.divergence?.components ?? []).map((component) => component.score),
    {
      label: "divergence score",
      emptyMessage: "At least two finite component-divergence scores are required.",
    }
  );
  elements.debugDivergenceList.replaceChildren();
  (comparison.divergence?.components ?? []).slice(0, 30).forEach((component, index) => {
    const row = document.createElement("button");
    row.type = "button";
    const rank = document.createElement("span");
    const label = document.createElement("b");
    const score = document.createElement("em");
    rank.textContent = String(index + 1).padStart(2, "0");
    label.textContent = component.label;
    score.textContent = modelMetric(component.score, 3);
    row.append(rank, label, score);
    row.addEventListener("click", () => {
      setModelInterfaceOpen(false);
      focusGraphNode(component.nodeId);
    });
    elements.debugDivergenceList.append(row);
  });
  (comparison.divergence?.attentionHeads ?? []).slice(0, 12).forEach((head, index) => {
    const row = document.createElement("button");
    row.type = "button";
    const rank = document.createElement("span");
    const label = document.createElement("b");
    const score = document.createElement("em");
    rank.textContent = `H${String(index + 1).padStart(2, "0")}`;
    label.textContent = head.label;
    score.textContent = `L2 ${modelMetric(head.l2Distance, 3)}`;
    row.append(rank, label, score);
    row.addEventListener("click", () => {
      setModelInterfaceOpen(false);
      focusGraphNode(head.nodeId);
    });
    elements.debugDivergenceList.append(row);
  });
  applyComparisonOverlay(comparison);
  populateDebugComponentControls(comparison.failure, comparison);
}

async function runPairedComparison() {
  const selectedPrompt = elements.runtimePromptInput.value;
  const referencePrompt = elements.runtimeControlPromptInput.value;
  if (!selectedPrompt.trim() || !referencePrompt.trim()) {
    elements.modelRunStatus.classList.add("error");
    elements.modelRunStatus.textContent = "Enter both a selected and a reference example.";
    (!selectedPrompt.trim() ? elements.runtimePromptInput : elements.runtimeControlPromptInput).focus();
    return;
  }
  setRuntimeBusy(true, "Running selected and reference traces…");
  elements.modelRunStatus.classList.remove("error");
  try {
    const comparison = await runtimeApi("/compare", { method: "POST", body: {
      failure: { prompt: selectedPrompt },
      control: { prompt: referencePrompt },
      metric: behaviourMetricSpec(),
      targetToken: elements.runtimeTargetInput.value.trim(),
      topK: Number(elements.runtimeTopKInput.value) || 10,
      seed: Number(elements.debugSeedInput.value) || 0,
      generation: { doSample: false, maxNewTokens: 1 },
      logitLens: { enabled: true, maxStages: 24, topK: 3 }
    } });
    state.runtime.latestRun = comparison.failure;
    recordRun(comparison.control, { kind: "reference", label: "Reference trace" });
    recordRun(comparison.failure, { kind: "selected", label: "Selected trace" });
    applyRuntimeLedger(comparison.failure);
    renderRuntimeResult(comparison.failure);
    renderComparisonResult(comparison);
    elements.modelRunStatus.textContent = `Comparison ${comparison.comparisonId.slice(0, 8)} found ${comparison.divergence?.components?.length ?? 0} measurable component differences.`;
    await saveDebugCase({ quiet: true });
  } catch (error) {
    elements.modelRunStatus.classList.add("error");
    elements.modelRunStatus.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

function selectedInterventionComponent() {
  const [kind, layer, nodeId] = elements.debugInterventionComponent.value.split(":");
  return kind && Number.isFinite(Number(layer)) ? { kind, layer: Number(layer), nodeId } : null;
}

async function runDebugIntervention() {
  const component = selectedInterventionComponent();
  const baseRun = state.debug.comparison?.failure ?? state.runtime.latestRun;
  if (!component || !baseRun) return;
  const method = elements.debugInterventionMethod.value;
  const sourceRunId = ["patch", "resample", "steer"].includes(method) ? state.debug.comparison?.control?.runId ?? "" : "";
  if (["patch", "resample", "steer"].includes(method) && !sourceRunId) {
    elements.debugInterventionResult.hidden = false;
    elements.debugInterventionResult.textContent = "This method requires a paired control run.";
    return;
  }
  setRuntimeBusy(true, `Running ${method} intervention…`);
  try {
    const scope = elements.debugInterventionScope.value;
    const result = await runtimeApi("/intervene", { method: "POST", body: {
      baseRunId: baseRun.runId,
      sourceRunId,
      component,
      method,
      scale: Number(elements.debugInterventionScale.value),
      scope,
      // The UI's selected-token scope means the final prompt position. Keep it
      // relative so verification replays it correctly on different-length prompts.
      position: scope === "position" ? -1 : baseRun.position,
      metric: behaviourMetricSpec()
    } });
    state.debug.interventions.push(result);
    recordRun(result.run, { kind: "intervention", label: `${method} · ${result.component?.kind ?? "component"} L${result.component?.layer ?? "—"}` });
    elements.debugInterventionResult.replaceChildren();
    const summary = document.createElement("div");
    summary.className = "debug-summary-grid";
    appendDebugSummary(summary, "Metric effect", shortObservedValue(result.metric?.signedEffect, true), `${result.metric?.name}`);
    appendDebugSummary(summary, "Target logit", shortObservedValue(result.outputEffect?.targetLogit, true), "change");
    appendDebugSummary(summary, "Probability", `${shortObservedValue((result.outputEffect?.targetProbability ?? 0) * 100, true)}%`, "change");
    appendDebugSummary(summary, "Rank", shortObservedValue(result.outputEffect?.targetRank, true), "change");
    appendDebugSummary(summary, "Output KL", modelMetric(result.outputEffect?.distributionKL, 4), "baseline → intervention");
    appendDebugSummary(summary, "Evidence", "Causal", `${result.method} · ${result.scope}`);
    elements.debugInterventionResult.append(summary);
    elements.debugInterventionResult.hidden = false;
    const metric = state.runtime.nodeMetrics.get(component.nodeId) ?? { kind: component.kind, layer: component.layer };
    metric.causalEffect = result.metric?.signedEffect;
    state.runtime.nodeMetrics.set(component.nodeId, metric);
    state.renderedRange = null;
    renderFlowScene({ force: true });
    elements.debugRunVerificationButton.disabled = false;
    await saveDebugCase({ quiet: true });
  } catch (error) {
    elements.debugInterventionResult.hidden = false;
    elements.debugInterventionResult.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

function applyRootCauseOverlay(trace) {
  (trace?.overlay ?? []).forEach((item) => {
    const current = state.runtime.nodeMetrics.get(item.nodeId) ?? {};
    if (item.retained) current.causalEffect = item.acdcEffect;
    current.eapScore = item.eapScore;
    current.circuitRetained = Boolean(item.retained);
    state.runtime.nodeMetrics.set(item.nodeId, current);
  });
  if (!state.model) return;
  state.renderedRange = null;
  renderFlowScene({ force: true });
}

function renderRootCauseTrace(trace) {
  state.debug.trace = trace;
  elements.debugTraceResult.replaceChildren();
  const summary = document.createElement("div");
  summary.className = "debug-summary-grid";
  appendDebugSummary(summary, "EAP candidates", String(trace.candidates?.length ?? 0), "gradient-ranked writes");
  appendDebugSummary(summary, "ACDC circuit", String(trace.retained?.length ?? 0), `|effect| ≥ ${modelMetric(trace.threshold, 4)}`);
  appendDebugSummary(summary, "Fidelity", Number.isFinite(trace.fidelity) ? `${modelMetric(trace.fidelity * 100, 1)}%` : "—", "metric recovery");
  appendDebugSummary(summary, "Stability", Number.isFinite(trace.stability?.score) ? `${modelMetric(trace.stability.score * 100, 1)}%` : "—", "threshold Jaccard");
  const list = document.createElement("div");
  list.className = "debug-ranked-list";
  (trace.candidates ?? []).forEach((candidate, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = candidate.retained ? "retained" : "pruned";
    const rank = document.createElement("span");
    const label = document.createElement("b");
    const effect = document.createElement("em");
    rank.textContent = String(index + 1).padStart(2, "0");
    label.textContent = `${candidate.label}${candidate.retained ? " · retained" : " · pruned"}`;
    effect.textContent = `EAP ${shortObservedValue(candidate.eapScore, true)} · patch ${shortObservedValue(candidate.acdcEffect, true)}`;
    row.append(rank, label, effect);
    row.addEventListener("click", () => {
      setModelInterfaceOpen(false);
      focusGraphNode(candidate.nodeId);
    });
    list.append(row);
  });
  const caveat = document.createElement("p");
  caveat.textContent = trace.evidence?.scope ?? "Circuit evidence is scoped to this paired run and metric.";
  elements.debugTraceResult.append(summary, list, caveat);
  elements.debugTraceResult.hidden = false;
  applyRootCauseOverlay(trace);
}

async function runRootCauseTrace() {
  const comparison = state.debug.comparison;
  if (!comparison?.failure?.runId || !comparison?.control?.runId) return;
  setRuntimeBusy(true, "Running EAP candidate discovery and ACDC validation…");
  try {
    const trace = await runtimeApi("/root-cause", { method: "POST", body: {
      failureRunId: comparison.failure.runId,
      controlRunId: comparison.control.runId,
      metric: behaviourMetricSpec(),
      maxCandidates: Number(elements.debugTraceCandidates.value) || 16,
      threshold: Number(elements.debugTraceThreshold.value) || 0
    } });
    renderRootCauseTrace(trace);
    elements.modelRunStatus.classList.remove("error");
    elements.modelRunStatus.textContent = `Root-cause trace retained ${trace.retained?.length ?? 0} of ${trace.candidates?.length ?? 0} candidates.`;
    await saveDebugCase({ quiet: true });
  } catch (error) {
    elements.debugTraceResult.hidden = false;
    elements.debugTraceResult.textContent = String(error?.message ?? error);
    elements.modelRunStatus.classList.add("error");
    elements.modelRunStatus.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

async function loadDebugActivation() {
  const run = state.runtime.latestRun;
  const hookName = elements.debugMicroscopeHook.value;
  if (!run || !hookName) return;
  setRuntimeBusy(true, "Fetching a bounded activation slice…");
  try {
    const result = await runtimeApi("/activation", { method: "POST", body: {
      runId: run.runId,
      hookName,
      position: Number(elements.debugMicroscopePosition.value),
      head: Number(elements.debugMicroscopeHead.value),
      compareRunId: state.debug.comparison?.control?.runId ?? "",
      limit: Number(elements.debugMicroscopeLimit.value)
    } });
    elements.debugMicroscopeResult.replaceChildren();
    const heading = document.createElement("strong");
    const detail = document.createElement("p");
    heading.textContent = `${result.hookName} · [${result.shape.join(" × ")}]`;
    detail.textContent = `${result.returned} of ${result.total} values · norm ${modelMetric(result.stats?.norm, 3)} · mean ${modelMetric(result.stats?.mean, 4)} · std ${modelMetric(result.stats?.std, 4)}${Number.isFinite(result.directLogitAttribution) ? ` · target DLA ${shortObservedValue(result.directLogitAttribution, true)}` : ""}`;
    const distribution = document.createElement("figure");
    distribution.className = "metric-histogram-card activation-histogram";
    const distributionCaption = document.createElement("figcaption");
    distributionCaption.textContent = `Returned activation-value distribution${result.truncated ? " (bounded prefix slice)" : ""}`;
    const distributionChart = document.createElement("div");
    distributionChart.className = "model-chart model-histogram";
    distribution.append(distributionCaption, distributionChart);
    renderHistogram(distributionChart, result.values ?? [], {
      label: "activation value",
      symmetric: true,
      emptyMessage: "At least two finite activation values are required.",
    });
    const bars = document.createElement("div");
    bars.className = "debug-activation-bars";
    const values = (result.values ?? []).slice(0, 64);
    const maximum = Math.max(...values.filter(Number.isFinite).map(Math.abs), 1e-12);
    values.forEach((value, index) => {
      const bar = document.createElement("i");
      bar.className = value < 0 ? "negative" : "";
      bar.style.height = `${Math.max(2, Math.abs(value ?? 0) / maximum * 68)}px`;
      bar.title = `Feature ${index}: ${modelMetric(value, 5)}`;
      bars.append(bar);
    });
    const featureList = document.createElement("p");
    featureList.textContent = `Top features: ${(result.topFeatures ?? []).slice(0, 8).map((item) => `#${item.index} ${shortObservedValue(item.value, true)}`).join(" · ") || "—"}`;
    const tokenList = document.createElement("p");
    tokenList.textContent = `Top activating tokens: ${(result.topActivatingTokens ?? []).slice(0, 8).map((item) => `${item.position}:${item.token} (${modelMetric(item.activationNorm, 2)})`).join(" · ") || "not applicable"}`;
    const difference = document.createElement("p");
    difference.textContent = result.comparison
      ? `Failure − control: L2 ${modelMetric(result.comparison.l2Distance, 3)} · mean |Δ| ${modelMetric(result.comparison.meanAbsoluteDifference, 4)}`
      : "No aligned control activation is available for this component.";
    const scope = document.createElement("p");
    scope.className = "debugger-science-note";
    scope.textContent = result.truncated
      ? `The histogram covers the first ${result.returned} flattened values requested from a tensor with ${result.total} values. It is a bounded inspection slice, not an estimate of the full activation distribution.`
      : "The histogram covers the complete selected activation vector. Feature values are coordinates from one token-position observation, not independent samples.";
    elements.debugMicroscopeResult.append(heading, detail, distribution, bars, featureList, tokenList, difference, scope);
    elements.debugMicroscopeResult.hidden = false;
  } catch (error) {
    elements.debugMicroscopeResult.hidden = false;
    elements.debugMicroscopeResult.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

function parseCsvRows(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') { field += '"'; index += 1; }
      else quoted = !quoted;
    } else if (character === "," && !quoted) { row.push(field); field = ""; }
    else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(field); field = "";
      if (row.some((item) => item.length)) rows.push(row);
      row = [];
    } else field += character;
  }
  row.push(field);
  if (row.some((item) => item.length)) rows.push(row);
  if (rows.length < 2) return [];
  const headers = rows[0].map((item) => item.trim());
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function renderBenchmarkExplorer() {
  elements.debugInboxList.replaceChildren();
  const counts = state.debug.inbox.reduce((result, item) => {
    result[item.status] = (result[item.status] ?? 0) + 1;
    return result;
  }, {});
  const outcomeSummary = ["passed", "failed", "regressed", "anomaly"]
    .filter((status) => counts[status])
    .map((status) => `${counts[status]} ${status}`)
    .join(" · ");
  elements.debugInboxCount.textContent = `${state.debug.inbox.length} example${state.debug.inbox.length === 1 ? "" : "s"}${outcomeSummary ? ` · ${outcomeSummary}` : ""}`;
  elements.debugRunInboxButton.disabled = !state.debug.inbox.length || !modelRunReady() || state.runtime.busy;
  const filter = elements.debugBenchmarkFilter.value;
  const visibleItems = filter === "all" ? state.debug.inbox : state.debug.inbox.filter((item) => item.status === filter);
  visibleItems.forEach((item) => {
    const index = state.debug.inbox.indexOf(item);
    const row = document.createElement("button");
    row.type = "button";
    row.dataset.status = item.status;
    row.classList.toggle("selected", item.id === state.debug.selectedBenchmarkId);
    const number = document.createElement("span");
    const prompt = document.createElement("strong");
    const status = document.createElement("small");
    number.textContent = String(index + 1).padStart(2, "0");
    prompt.textContent = `${item.benchmark}${item.task ? ` / ${item.task}` : ""} · ${item.prompt}`;
    const scoreValue = Number.isFinite(item.score) ? item.score : item.result?.metric?.value;
    const score = Number.isFinite(scoreValue) ? ` · ${Number.isFinite(item.score) ? "benchmark" : "measured"} score ${modelMetric(scoreValue, 4)}` : "";
    const suspected = item.result?.suspectedComponent?.label ? ` · ${item.result.suspectedComponent.label}` : "";
    status.textContent = `${item.status}${score} · ${item.cluster}${suspected}`;
    status.title = item.outcomeSource ?? "";
    row.append(number, prompt, status);
    row.addEventListener("click", () => selectBenchmarkExample(item));
    elements.debugInboxList.append(row);
  });
  if (!visibleItems.length && state.debug.inbox.length) {
    const empty = document.createElement("p");
    empty.className = "debug-benchmark-empty";
    empty.textContent = `No ${filter} examples in this benchmark.`;
    elements.debugInboxList.append(empty);
  }
}

function selectBenchmarkExample(item, { focus = true } = {}) {
  if (!item) return;
  state.debug.selectedBenchmarkId = item.id;
  elements.runtimePromptInput.value = item.prompt;
  elements.runtimeControlPromptInput.value = item.reference;
  elements.debugExpectedInput.value = item.expected;
  elements.runtimeTargetInput.value = item.target;
  if (item.benchmark && !elements.debugCaseNameInput.value.trim()) elements.debugCaseNameInput.value = `${item.benchmark} · ${item.exampleId}`;
  renderBenchmarkExplorer();
  setDebugDirty();
  if (focus) elements.runtimePromptInput.focus();
}

function gpt2DevelopmentBenchmark() {
  return normaliseBenchmarkExamples(GPT2_DEVELOPMENT_EXAMPLES, {
    benchmark: "GPT-2 UX fixture",
    idFactory: (item) => `gpt2-dev-${item.example_id}`
  }).map((item, index) => ({
    ...item,
    status: GPT2_DEVELOPMENT_EXAMPLES[index].uxOutcome,
    score: GPT2_DEVELOPMENT_EXAMPLES[index].uxScore,
    outcomeSource: "Illustrative development state — run the benchmark to replace it with a GPT-2 measurement",
    cluster: index < 2 ? "output:completion · circuit:unmeasured" : index < 4 ? "output:knowledge · circuit:unmeasured" : "output:language · circuit:unmeasured",
    developmentFixture: true
  }));
}

async function openGpt2DevelopmentExample() {
  elements.gpt2DevExampleButton.disabled = true;
  elements.gpt2DevExampleButton.textContent = "Opening GPT-2…";
  elements.hfModelInput.value = GPT2_DEVELOPMENT_MODEL_ID;
  elements.hfRevisionInput.value = "main";
  try {
    const loaded = await importHuggingFaceModel();
    if (!loaded) return;
    resetDebugCaseEditor();
    elements.debugCaseNameInput.value = "DEV · GPT-2 benchmark investigation";
    elements.debugBenchmarkName.value = "GPT-2 UX fixture";
    elements.debugNotesInput.value = "Development fixture: initial outcomes and scores are illustrative UI states, not GPT-2 measurements. Run the benchmark with a connected worker to replace them with observed results.";
    elements.debugMetricKindSelect.value = "target_probability";
    elements.debugMetricNameInput.value = "Target-token probability";
    state.debug.inbox = gpt2DevelopmentBenchmark();
    selectBenchmarkExample(state.debug.inbox[0], { focus: false });
    setSidebarCollapsed(true);
    setModelInterfaceOpen(true);
    elements.modelRunStatus.textContent = modelRunReady()
      ? "GPT-2 development fixture ready. Run an example or the complete benchmark to replace illustrative states with measurements."
      : "GPT-2 development fixture ready. Load GPT-2 into the connected worker from Settings to collect real measurements.";
  } finally {
    elements.gpt2DevExampleButton.disabled = false;
    elements.gpt2DevExampleButton.textContent = "Open GPT-2 example";
  }
}

async function importBenchmarkExamples() {
  const file = elements.debugInboxFile.files?.[0];
  const dataset = elements.debugInboxDataset.value.trim();
  if (!file && !dataset) {
    elements.debugCaseLibraryStatus.textContent = "Choose a JSONL/CSV file or enter a Hugging Face dataset ID.";
    return;
  }
  try {
    let values;
    let source = file?.name ?? dataset;
    if (file) {
      const text = await file.text();
      if (file.name.toLowerCase().endsWith(".csv")) values = parseCsvRows(text);
      else if (file.name.toLowerCase().endsWith(".jsonl")) values = text.split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line));
      else {
        const parsed = JSON.parse(text);
        values = Array.isArray(parsed) ? parsed : parsed.cases ?? [];
      }
    }
    else {
      const response = await fetch(`/api/huggingface/dataset?${new URLSearchParams({ dataset })}`, { credentials: "same-origin" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error ?? `Dataset import failed (${response.status})`);
      values = payload.rows ?? [];
      source = `${payload.dataset} · ${payload.config}/${payload.split}`;
    }
    const imported = normaliseBenchmarkExamples(values, { benchmark: elements.debugBenchmarkName.value.trim() || source });
    if (!imported.length) throw new Error("No prompt-like text column was found in the imported rows");
    state.debug.inbox.push(...imported);
    renderBenchmarkExplorer();
    setDebugDirty();
    elements.debugCaseLibraryStatus.textContent = `Imported ${imported.length} benchmark example${imported.length === 1 ? "" : "s"} from ${source}.`;
    await saveDebugCase({ quiet: true });
  } catch (error) {
    elements.debugCaseLibraryStatus.textContent = `Could not import benchmark: ${error?.message ?? error}`;
  }
}

function benchmarkCluster(run, comparison = null) {
  const prediction = String(run?.nextToken?.topK?.[0]?.text ?? run?.nextToken?.topK?.[0]?.token ?? "unknown").trim().toLowerCase() || "blank";
  const tokenCount = run?.tokens?.length ?? 0;
  const tokenBand = tokenCount < 32 ? "short" : tokenCount < 128 ? "medium" : "long";
  const divergent = comparison?.divergence?.components?.[0];
  const attributed = [...(run?.attribution?.components ?? [])].sort((left, right) => Math.abs(right.dla ?? 0) - Math.abs(left.dla ?? 0))[0];
  const circuit = divergent ? `L${divergent.layer}-${divergent.kind}` : attributed ? `L${attributed.layer ?? "emb"}-${attributed.kind}` : "unknown-circuit";
  return `output:${prediction.slice(0, 24)} · tokens:${tokenBand} · circuit:${circuit}`;
}

async function runBenchmarkExamples() {
  if (!state.debug.inbox.length || !modelRunReady()) return;
  setRuntimeBusy(true, `Running ${Math.min(100, state.debug.inbox.length)} benchmark examples…`);
  try {
    const limit = Math.min(100, state.debug.inbox.length);
    for (let index = 0; index < limit; index += 1) {
      const item = state.debug.inbox[index];
      const metric = {
        ...behaviourMetricSpec(),
        targetToken: item.target || behaviourMetricSpec().targetToken,
        answer: item.expected || behaviourMetricSpec().answer
      };
      let run;
      let comparison = null;
      if (item.reference) {
        comparison = await runtimeApi("/compare", { method: "POST", body: {
          failure: { prompt: item.prompt }, control: { prompt: item.reference }, metric,
          targetToken: item.target, topK: 5, seed: Number(elements.debugSeedInput.value) || 0
        } });
        run = comparison.failure;
      } else {
        run = await runtimeApi("/forward", { method: "POST", body: {
          ...debugRunPayload(item.prompt), metric, targetToken: item.target, topK: 5
        } });
      }
      const outcome = benchmarkOutcomeForRun(item, run);
      item.status = outcome.status;
      item.outcomeSource = outcome.source;
      if (item.developmentFixture) {
        item.developmentFixture = false;
        item.score = null;
      }
      item.cluster = benchmarkCluster(run, comparison);
      item.result = {
        runId: run.runId,
        topPrediction: String(run.nextToken?.topK?.[0]?.text ?? "").trim(),
        targetProbability: run.target?.probability,
        metric: run.metric,
        suspectedComponent: comparison?.divergence?.components?.[0] ?? null
      };
      elements.runtimeStatus.textContent = `Benchmark ${index + 1}/${limit} · ${item.status}`;
      renderBenchmarkExplorer();
    }
    const clusters = new Set(state.debug.inbox.slice(0, limit).map((item) => item.cluster));
    elements.debugCaseLibraryStatus.textContent = `Ran ${limit} cases and found ${clusters.size} behavioural/internal clusters.`;
    await saveDebugCase({ quiet: true });
  } catch (error) {
    elements.debugCaseLibraryStatus.textContent = `Inbox run stopped: ${error?.message ?? error}`;
  } finally {
    setRuntimeBusy(false);
    renderBenchmarkExplorer();
  }
}

function renderVerificationResult(verification) {
  state.debug.verification = verification;
  elements.debugVerificationResult.replaceChildren();
  const summary = document.createElement("div");
  summary.className = "debug-summary-grid";
  appendDebugSummary(summary, "Outcome", String(verification.status ?? "not run").toUpperCase(), "candidate intervention");
  appendDebugSummary(summary, "Fixed / improved", String(verification.summary?.fixed ?? 0), "cases");
  appendDebugSummary(summary, "Unchanged", String(verification.summary?.unchanged ?? 0), "cases");
  appendDebugSummary(summary, "Regressed", String(verification.summary?.regressed ?? 0), "cases");
  const list = document.createElement("div");
  list.className = "debug-ranked-list";
  (verification.results ?? []).forEach((item, index) => {
    const row = document.createElement("div");
    row.className = item.status;
    const number = document.createElement("span");
    const label = document.createElement("b");
    const effect = document.createElement("em");
    number.textContent = String(index + 1).padStart(2, "0");
    label.textContent = `${item.label} · ${item.status}`;
    effect.textContent = `${shortObservedValue(item.baseline, true)} → ${shortObservedValue(item.candidate, true)} · Δ ${shortObservedValue(item.signedEffect, true)}`;
    row.append(number, label, effect);
    list.append(row);
  });
  elements.debugVerificationResult.append(summary, list);
  elements.debugVerificationResult.hidden = false;
}

async function runFixVerification() {
  const candidate = state.debug.interventions.at(-1);
  if (!candidate) return;
  const guardrails = elements.debugGuardrailsInput.value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
  const metric = behaviourMetricSpec();
  const cases = [
    { role: "failure", label: "Selected example", prompt: elements.runtimePromptInput.value, targetToken: elements.runtimeTargetInput.value.trim(), metric },
    ...guardrails.map((prompt, index) => ({ role: "guardrail", label: `Guardrail ${index + 1}`, prompt, targetToken: elements.runtimeTargetInput.value.trim(), metric }))
  ];
  setRuntimeBusy(true, `Verifying the candidate across ${cases.length} case${cases.length === 1 ? "" : "s"}…`);
  try {
    const verification = await runtimeApi("/verify", { method: "POST", body: {
      candidate: {
        interventionId: candidate.interventionId,
        component: candidate.component,
        method: candidate.method,
        scale: candidate.scale,
        scope: candidate.scope,
        position: candidate.position,
        sourceRunId: candidate.sourceRunId
      },
      cases,
      metric,
      topK: Number(elements.runtimeTopKInput.value) || 10,
      seed: Number(elements.debugSeedInput.value) || 0
    } });
    renderVerificationResult(verification);
    elements.modelRunStatus.textContent = `Verification outcome: ${verification.status}. ${verification.summary?.regressed ?? 0} regression${verification.summary?.regressed === 1 ? "" : "s"}.`;
    await saveDebugCase({ quiet: true });
  } catch (error) {
    elements.debugVerificationResult.hidden = false;
    elements.debugVerificationResult.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

function renderGenerationStepLens(step, container) {
  container.replaceChildren();
  const lens = step.logitLens;
  if (!lens?.available || !lens.stages?.length) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  const heading = document.createElement("div");
  heading.className = "generation-step-lens-heading";
  const title = document.createElement("strong");
  const context = document.createElement("span");
  title.textContent = `Step ${step.index + 1} · ${step.token?.text || step.token?.token || "token"}`;
  context.textContent = `${step.selection} · chosen rank #${step.chosenRank} · entropy ${modelMetric(step.entropy, 3)} nats`;
  heading.append(title, context);
  container.append(heading);
  const chart = document.createElement("div");
  chart.className = "model-chart";
  const probabilityLabel = lens.method === "normalized-logit-lens" ? "Chosen-token probability (%)" : "Raw-lens softmax (%)";
  renderLineChart(chart, [
    { label: probabilityLabel, values: lens.stages.map((stage) => Number.isFinite(stage.targetProbability) ? stage.targetProbability * 100 : null) }
  ], { includeZero: true, xLabel: "sampled residual stage", xValues: lens.stages.map((stage) => stage.index) });
  const stages = document.createElement("div");
  stages.className = "generation-lens-stage-list";
  lens.stages.forEach((stage) => {
    const item = document.createElement("span");
    const top = stage.topK?.[0];
    item.textContent = `${stage.label}: ${top?.text || top?.token || "—"}`;
    item.title = `Chosen-token rank ${stage.targetRank}; KL(final || lens) ${modelMetric(stage.finalToLensKL, 4)}`;
    stages.append(item);
  });
  const caveat = document.createElement("p");
  caveat.className = "debugger-science-note";
  caveat.textContent = lens.evidence?.note ?? "Intermediate vocabulary projections are observational, not causal.";
  container.append(chart, stages, caveat);
}

function renderGenerationTimeline(trace) {
  elements.generationTimelineResult.replaceChildren();
  const summary = document.createElement("div");
  summary.className = "generation-summary";
  const completion = document.createElement("strong");
  const detail = document.createElement("span");
  completion.textContent = trace.completion || "No visible completion";
  detail.textContent = `${trace.steps?.length ?? 0} tokens · ${trace.stopReason?.replaceAll("-", " ") ?? "complete"} · ${modelMetric(trace.performance?.durationMs, 0)} ms`;
  summary.append(completion, detail);
  const timeline = document.createElement("div");
  timeline.className = "generation-token-timeline";
  const lensPanel = document.createElement("div");
  lensPanel.className = "generation-step-lens";
  (trace.steps ?? []).forEach((step, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `generation-token-step${index === 0 ? " selected" : ""}`;
    const order = document.createElement("small");
    const token = document.createElement("strong");
    const probability = document.createElement("span");
    order.textContent = String(index + 1).padStart(2, "0");
    token.textContent = step.token?.text || step.token?.token || "∅";
    probability.textContent = `${modelMetric((step.chosenProbability ?? 0) * 100, 1)}% · #${step.chosenRank ?? "—"}`;
    button.append(order, token, probability);
    button.addEventListener("click", () => {
      timeline.querySelectorAll("button").forEach((item) => item.classList.toggle("selected", item === button));
      renderGenerationStepLens(step, lensPanel);
    });
    timeline.append(button);
  });
  const evidence = document.createElement("p");
  evidence.className = "debugger-science-note";
  evidence.textContent = `${trace.evidence?.note ?? "Autoregressive generation trace."} ${trace.evidence?.sampling ?? ""}`;
  const lensAvailable = (trace.steps ?? []).some((step) => step.logitLens?.available && step.logitLens?.stages?.length);
  elements.generationTimelineResult.append(summary, timeline);
  if (lensAvailable) {
    elements.generationTimelineResult.append(lensPanel);
    const firstLensStep = (trace.steps ?? []).find((step) => step.logitLens?.available && step.logitLens?.stages?.length);
    if (firstLensStep) renderGenerationStepLens(firstLensStep, lensPanel);
  }
  elements.generationTimelineResult.append(evidence);
  elements.generationTimelineResult.hidden = false;
}

async function runGenerationTrace() {
  const prompt = elements.runtimePromptInput.value.trim();
  if (!prompt) {
    elements.generationTimelineResult.hidden = false;
    elements.generationTimelineResult.textContent = "Enter a selected example before tracing generation.";
    return;
  }
  const doSample = elements.generationMode.value === "sample";
  const maximumTokens = Number(elements.generationMaxTokens.value) || 8;
  const lensSupported = state.runtime.latestRun?.diagnostics?.capabilities?.logitLens !== false;
  const maximumStages = Number(elements.generationLensStages.value) || 16;
  if (lensSupported && maximumTokens * maximumStages > 256) {
    elements.generationTimelineResult.hidden = false;
    elements.generationTimelineResult.textContent = "New tokens × lens stages must be 256 or less. Reduce either control to keep the trace bounded.";
    elements.generationLensStages.focus();
    return;
  }
  setRuntimeBusy(true, "Tracing autoregressive generation…");
  try {
    const trace = await runtimeApi("/generate", { method: "POST", body: {
      prompt,
      seed: Number(elements.debugSeedInput.value) || 0,
      topK: Number(elements.runtimeTopKInput.value) || 10,
      generation: {
        maxNewTokens: maximumTokens,
        doSample,
        temperature: Number(elements.generationTemperature.value) || 1,
        topP: Number(elements.generationTopP.value) || 1,
        topK: 0
      },
      logitLens: {
        enabled: lensSupported,
        maxStages: maximumStages,
        topK: 3
      }
    } });
    state.runtime.generationTrace = trace;
    renderGenerationTimeline(trace);
    recordRun(trace, { kind: "generation", label: `Generation · ${trace.steps?.length ?? 0} tokens` });
    elements.modelRunStatus.classList.remove("error");
    elements.modelRunStatus.textContent = `Generation ${trace.generationId.slice(0, 8)} traced ${trace.steps?.length ?? 0} autoregressive decisions.`;
    await saveDebugCase({ quiet: true });
  } catch (error) {
    elements.generationTimelineResult.hidden = false;
    elements.generationTimelineResult.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

function applyCausalSweepOverlay(sweep) {
  const strongest = new Map();
  (sweep?.cells ?? []).forEach((cell) => {
    if (cell.status !== "measured" || !Number.isFinite(cell.signedEffect)) return;
    const current = strongest.get(cell.nodeId);
    if (!current || Math.abs(cell.signedEffect) > Math.abs(current.signedEffect)) strongest.set(cell.nodeId, cell);
  });
  strongest.forEach((cell, nodeId) => {
    const metric = state.runtime.nodeMetrics.get(nodeId) ?? { kind: sweep.kind, layer: cell.layer };
    metric.causalEffect = cell.signedEffect;
    metric.sweepPosition = cell.position;
    state.runtime.nodeMetrics.set(nodeId, metric);
  });
  renderWatchlist();
  if (state.model) {
    state.renderedRange = null;
    renderFlowScene({ force: true });
  }
}

function renderCausalSweep(sweep) {
  elements.causalSweepResult.replaceChildren();
  const summary = document.createElement("div");
  summary.className = "debug-summary-grid";
  appendDebugSummary(summary, "Measured", `${sweep.summary?.measured ?? 0}/${sweep.summary?.requested ?? 0}`, `${sweep.summary?.unsupported ?? 0} unsupported · ${sweep.summary?.errors ?? 0} errors`);
  appendDebugSummary(summary, "Largest |effect|", shortObservedValue(sweep.summary?.maximumAbsoluteEffect?.signedEffect, true), sweep.summary?.maximumAbsoluteEffect ? `L${sweep.summary.maximumAbsoluteEffect.layer} · position ${sweep.summary.maximumAbsoluteEffect.position}` : "No measured cells");
  appendDebugSummary(summary, "Metric", sweep.metric?.name ?? "—", `${sweep.metric?.direction ?? "—"} · baseline ${modelMetric(sweep.metric?.baseline, 4)}`);
  appendDebugSummary(summary, "Evidence", "Causal in-cell", `${sweep.method} · one component-position intervention`);
  elements.causalSweepResult.append(summary);
  const distribution = document.createElement("figure");
  distribution.className = "metric-histogram-card causal-effect-histogram";
  const distributionCaption = document.createElement("figcaption");
  distributionCaption.textContent = "Signed metric-effect distribution across measured cells";
  const distributionChart = document.createElement("div");
  distributionChart.className = "model-chart model-histogram";
  distribution.append(distributionCaption, distributionChart);
  elements.causalSweepResult.append(distribution);
  renderHistogram(
    distributionChart,
    (sweep.cells ?? []).filter((cell) => cell.status === "measured").map((cell) => cell.signedEffect),
    {
      label: "signed metric effect",
      symmetric: true,
      emptyMessage: "At least two measured sweep cells are required for an effect distribution.",
    }
  );
  const maxEffect = Math.max(1e-12, ...(sweep.cells ?? []).filter((cell) => Number.isFinite(cell.signedEffect)).map((cell) => Math.abs(cell.signedEffect)));
  const grid = document.createElement("div");
  grid.className = "causal-sweep-grid";
  grid.style.gridTemplateColumns = `minmax(76px, auto) repeat(${sweep.positions?.length ?? 1}, minmax(58px, 1fr))`;
  const corner = document.createElement("span");
  corner.className = "causal-sweep-axis";
  corner.textContent = "Layer ↓ / token →";
  grid.append(corner);
  (sweep.positions ?? []).forEach((position) => {
    const label = document.createElement("span");
    label.className = "causal-sweep-axis";
    label.textContent = `${position.index} · ${position.token || "∅"}`;
    label.title = `Token position ${position.index}`;
    grid.append(label);
  });
  (sweep.layers ?? []).forEach((layer) => {
    const label = document.createElement("span");
    label.className = "causal-sweep-axis layer";
    label.textContent = `L${layer}`;
    grid.append(label);
    (sweep.positions ?? []).forEach((position) => {
      const cell = (sweep.cells ?? []).find((item) => item.layer === layer && item.position === position.index);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `causal-sweep-cell ${cell?.status ?? "error"}`;
      if (cell?.status === "measured") {
        const strength = (Number.isFinite(cell.improvement) ? cell.improvement : cell.signedEffect) / maxEffect;
        button.style.setProperty("--sweep-strength", String(Math.abs(strength)));
        button.dataset.sign = strength > 0 ? "positive" : strength < 0 ? "negative" : "neutral";
        button.textContent = shortObservedValue(cell.signedEffect, true);
        button.title = `Baseline ${modelMetric(cell.baseline, 5)}; intervened ${modelMetric(cell.intervened, 5)}; signed metric effect ${shortObservedValue(cell.signedEffect, true)}; direction-adjusted improvement ${shortObservedValue(cell.improvement, true)}. Click to inspect ${cell.nodeId}.`;
        button.addEventListener("click", () => {
          setModelInterfaceOpen(false);
          focusGraphNode(cell.nodeId);
        });
      } else {
        button.textContent = cell?.status === "unsupported" ? "N/A" : "!";
        button.title = cell?.error ?? "This sweep cell was not measured.";
        button.disabled = true;
      }
      grid.append(button);
    });
  });
  const note = document.createElement("p");
  note.className = "debugger-science-note";
  note.textContent = `Cell text is signed metric change; colour is direction-adjusted improvement (green) or degradation (red). ${sweep.ui?.layerSampling ?? ""} ${sweep.evidence?.scope ?? "Each measured cell is a local causal comparison."} ${sweep.evidence?.method ?? ""} ${sweep.evidence?.multipleComparisons ?? ""}`;
  elements.causalSweepResult.append(grid, note);
  elements.causalSweepResult.hidden = false;
  applyCausalSweepOverlay(sweep);
}

async function runCausalSweep() {
  const baseRun = state.debug.comparison?.failure ?? state.runtime.latestRun;
  if (!baseRun) return;
  if (elements.debugMetricKindSelect.value === "kl_divergence") {
    elements.causalSweepResult.hidden = false;
    elements.causalSweepResult.textContent = "Causal sweeps require a scalar per-run behaviour metric. Choose a metric other than KL divergence.";
    elements.debugMetricKindSelect.focus();
    return;
  }
  const method = elements.sweepMethod.value;
  const sourceRunId = method === "patch" ? state.debug.comparison?.control?.runId ?? "" : "";
  if (method === "patch" && !sourceRunId) {
    elements.causalSweepResult.hidden = false;
    elements.causalSweepResult.textContent = "Reference patching requires a selected-vs-reference comparison first.";
    return;
  }
  const count = Math.min(Number(elements.sweepPositionCount.value) || 4, baseRun.tokens?.length ?? 1);
  const start = Math.max(0, (baseRun.tokens?.length ?? 1) - count);
  const positions = Array.from({ length: count }, (_, index) => start + index);
  const allLayers = (baseRun.layers ?? []).map((layer) => layer.layer);
  const maximumLayers = Math.max(1, Math.floor(128 / Math.max(1, positions.length)));
  const requestedLayers = allLayers.length <= maximumLayers
    ? allLayers
    : [...new Set(Array.from({ length: maximumLayers }, (_, index) => allLayers[Math.round(index * (allLayers.length - 1) / Math.max(1, maximumLayers - 1))]))];
  setRuntimeBusy(true, `Running ${requestedLayers.length} × ${positions.length} causal interventions…`);
  try {
    const sweep = await runtimeApi("/sweep", { method: "POST", body: {
      baseRunId: baseRun.runId,
      sourceRunId,
      kind: elements.sweepComponentKind.value,
      method,
      scale: Number(elements.sweepScale.value),
      layers: requestedLayers,
      positions,
      metric: behaviourMetricSpec()
    } });
    if (requestedLayers.length < allLayers.length) {
      sweep.ui = { layerSampling: `${requestedLayers.length} of ${allLayers.length} layers were evenly sampled to keep the grid within 128 interventions.` };
    }
    state.runtime.causalSweep = sweep;
    renderCausalSweep(sweep);
    elements.modelRunStatus.classList.remove("error");
    elements.modelRunStatus.textContent = `Sweep ${sweep.sweepId.slice(0, 8)} measured ${sweep.summary?.measured ?? 0} single-component interventions.`;
    setDebugDirty();
    await saveDebugCase({ quiet: true });
  } catch (error) {
    elements.causalSweepResult.hidden = false;
    elements.causalSweepResult.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

function markdownCell(value) {
  return String(value ?? "—").replaceAll("|", "\\|").replaceAll("\n", " ");
}

function exportDebugReport() {
  const record = debugCaseDocument();
  const comparison = state.debug.comparison;
  const trace = state.debug.trace;
  const verification = state.debug.verification;
  const generationTrace = state.runtime.generationTrace;
  const causalSweep = state.runtime.causalSweep;
  const circuitEdges = (trace?.retained ?? []).map((item) => `  ${item.nodeId}["${item.label}"] --> ${item.edge?.to ?? `${item.nodeId}_residual`}`).join("\n");
  const divergenceRows = (comparison?.divergence?.components ?? []).slice(0, 20)
    .map((item) => `| ${markdownCell(item.label)} | ${markdownCell(modelMetric(item.score, 5))} | ${markdownCell(item.relativeDistance)} | ${markdownCell(item.outputContributionDifference ?? item.logitLensDifference)} |`)
    .join("\n");
  const interventionRows = state.debug.interventions.map((item) => `| ${markdownCell(item.component?.kind)} L${markdownCell(item.component?.layer)} | ${markdownCell(item.method)} | ${markdownCell(item.scope)} | ${markdownCell(item.metric?.signedEffect)} | ${markdownCell(item.metric?.improvement)} |`).join("\n");
  const verificationRows = (verification?.results ?? []).map((item) => `| ${markdownCell(item.label)} | ${markdownCell(item.status)} | ${markdownCell(item.baseline)} | ${markdownCell(item.candidate)} | ${markdownCell(item.signedEffect)} |`).join("\n");
  const generationRows = (generationTrace?.steps ?? []).map((step) => `| ${step.index + 1} | ${markdownCell(step.token?.text ?? step.token?.token)} | ${markdownCell(step.selection)} | ${markdownCell(step.chosenProbability)} | ${markdownCell(step.chosenRank)} | ${markdownCell(step.entropy)} |`).join("\n");
  const sweepRows = [...(causalSweep?.cells ?? [])].filter((cell) => cell.status === "measured").sort((left, right) => Math.abs(right.signedEffect) - Math.abs(left.signedEffect)).slice(0, 24).map((cell) => `| L${cell.layer} | ${cell.position} | ${markdownCell(cell.positionToken)} | ${markdownCell(cell.signedEffect)} | ${markdownCell(cell.improvement)} |`).join("\n");
  const watchlistRows = watchlistValues().map((item) => `| ${markdownCell(item.label)} | ${markdownCell(item.nodeId)} | ${markdownCell(item.evidence)} | ${markdownCell(item.note)} |`).join("\n");
  const report = `# ModelDebugger report: ${record.name}\n\nExported: ${new Date().toISOString()}\n\n## Reproduction settings\n\n- Model: \`${record.model.modelId}\` at revision \`${record.model.revision}\` (${record.model.commit || "commit not resolved"})\n- Selected example: ${JSON.stringify(record.selected.prompt)}\n- Reference example: ${JSON.stringify(record.reference.prompt)}\n- Expected behaviour: ${record.expected.text || "Not specified"}\n- Target: ${record.target || "top prediction"}\n- Metric: ${record.metric.name} (\`${record.metric.kind}\`)\n- Seed: ${record.seed}\n- Dtype/device: ${record.dtype} / ${record.device || "worker-selected"}\n- Chat-template source: ${record.chatTemplateSource}\n- Tokenizer: ${record.tokenization ? `${record.tokenization.name} (${record.tokenization.class})` : "not captured"}\n- Software: ${record.software ? `worker ${record.software.worker}, PyTorch ${record.software.torch}, Transformers ${record.software.transformers}` : "not captured"}\n- Generation: \`${JSON.stringify(record.generation)}\`\n\n## Notes\n\n${record.notes || "No researcher notes."}\n\n## Selected vs reference\n\nSelected metric: ${comparison?.metric?.failureValue ?? "not run"}; reference metric: ${comparison?.metric?.controlValue ?? "not run"}; output KL: ${comparison?.divergence?.outputKL ?? "not run"}.\n\n| Component | Divergence score | Relative residual distance | Output contribution delta |\n|---|---:|---:|---:|\n${divergenceRows || "| Not run | — | — | — |"}\n\n## Generation timeline\n\nCompletion: ${JSON.stringify(generationTrace?.completion ?? "not run")}. Method: ${generationTrace?.settings?.doSample ? "sampled" : "greedy"}. Logit-lens projections are observational.\n\n| Step | Token | Selection | Unfiltered probability | Rank | Entropy |\n|---:|---|---|---:|---:|---:|\n${generationRows || "| — | Not run | — | — | — | — |"}\n\n## Causal sweep\n\n${causalSweep ? `${causalSweep.evidence?.scope ?? ""} ${causalSweep.evidence?.multipleComparisons ?? ""}` : "Not run."}\n\n| Layer | Position | Token | Signed metric effect | Direction-adjusted improvement |\n|---:|---:|---|---:|---:|\n${sweepRows || "| — | — | Not run | — | — |"}\n\n## Graph watchlist\n\n| Component | Node | Evidence label | Annotation |\n|---|---|---|---|\n${watchlistRows || "| None | — | — | — |"}\n\n## Candidate circuit\n\nMethod: EAP candidate discovery followed by intervention-backed ACDC pruning. Fidelity: ${trace?.fidelity ?? "not run"}; stability: ${trace?.stability?.score ?? "not run"}.\n\n\`\`\`mermaid\ngraph LR\n${circuitEdges || "  A[Root-cause trace not run]"}\n\`\`\`\n\n## Interventions\n\n| Component | Method | Scope | Signed metric effect | Improvement |\n|---|---|---|---:|---:|\n${interventionRows || "| None | — | — | — | — |"}\n\n## Intervention verification\n\nOverall: **${verification?.status ?? "not run"}**\n\n| Example | Status | Baseline | Candidate | Signed effect |\n|---|---|---:|---:|---:|\n${verificationRows || "| Not run | — | — | — | — |"}\n\n## Evidence and caveats\n\n- Divergence, run diffs, raw DLA, generation traces, and logit-lens projections are observational; they identify changes but do not establish cause.\n- Intervention effects are causal only for the exact model revision, prompt, token position, metric, and replacement distribution recorded here.\n- Causal sweep grids are exploratory multiple-comparison analyses; report the full grid and confirm selected cells on held-out prompts.\n- EAP is a first-order approximation. ACDC candidates in this report are retained only after actual activation patches.\n- Watchlist evidence labels are researcher annotations, not automatically validated claims.\n- Benchmark outcomes retain their supplied scoring criterion when available; inferred first-token checks are labelled as such.\n- Guardrail coverage is limited to the prompts listed in this case. Unsupported architecture features are recorded in run diagnostics.\n\n## Machine-readable case\n\n<details><summary>JSON</summary>\n\n\`\`\`json\n${JSON.stringify(record, null, 2)}\n\`\`\`\n</details>\n`;
  const blob = new Blob([report], { type: "text/markdown;charset=utf-8" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `${record.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "model-debugger-report"}.md`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
}

async function runHookedForwardPass(event) {
  event?.preventDefault();
  const prompt = elements.runtimePromptInput.value;
  if (!prompt.trim()) {
    elements.modelRunStatus.classList.add("error");
    elements.modelRunStatus.textContent = "Enter a prompt for the forward pass.";
    elements.runtimePromptInput.focus();
    return;
  }
  setRuntimeBusy(true, "Running hooks across the model…");
  elements.runtimeStatus.classList.remove("error");
  elements.modelRunStatus.classList.remove("error");
  try {
    const run = await runtimeApi("/forward", { method: "POST", body: debugRunPayload(prompt) });
    state.runtime.latestRun = run;
    recordRun(run, { kind: "forward", label: "Selected forward pass" });
    applyRuntimeLedger(run);
    renderRuntimeResult(run);
    const message = `Run ${run.runId.slice(0, 8)} captured ${run.hooks?.length ?? 0} component hooks across ${run.layers?.length ?? 0} layers.`;
    elements.runtimeStatus.textContent = message;
    elements.modelRunStatus.textContent = `${message} Measured values are now overlaid on the graph.`;
    await saveDebugCase({ quiet: true });
  } catch (error) {
    elements.runtimeStatus.classList.add("error");
    elements.modelRunStatus.classList.add("error");
    elements.runtimeStatus.textContent = String(error?.message ?? error);
    elements.modelRunStatus.textContent = String(error?.message ?? error);
  } finally {
    setRuntimeBusy(false);
  }
}

function setGraphControlsEnabled(enabled) {
  [elements.searchInput, elements.zoomOut, elements.zoomReadout, elements.zoomIn, elements.watchlistButton, elements.residualLedgerButton, elements.fit, elements.export]
    .forEach((control) => { control.disabled = !enabled; });
  elements.runModelButton.disabled = !enabled;
}

function beginImportLoading() {
  state.importController = new AbortController();
  state.importOutcome = "loading";
  elements.hfImport.dataset.loading = "true";
  elements.hfImport.dataset.outcome = "loading";
  elements.hfImportButton.dataset.loading = "true";
  elements.hfImportStatus.dataset.loading = "true";
  elements.graphPanel.classList.add("is-loading");
  elements.graphLoading.classList.remove("is-complete");
  elements.graphLoading.hidden = false;
  elements.hfImport.setAttribute("aria-busy", "true");
  elements.graphPanel.setAttribute("aria-busy", "true");
  elements.hfImportButton.disabled = true;
  elements.hfModelInput.disabled = true;
  elements.hfRevisionInput.disabled = true;
  elements.hfAccountToggleButton.disabled = true;
  elements.hfAccountConnectButton.disabled = true;
  elements.hfTokenInput.disabled = true;
  elements.cancelImportButton.disabled = false;
  elements.cancelImportButton.textContent = "Cancel inspection";
  elements.hfImportStatus.classList.remove("error");
  elements.hfModelInput.setAttribute("aria-invalid", "false");
  elements.statusDot.classList.remove("is-error");
  elements.statusDot.classList.add("is-loading");
  elements.appStatusText.textContent = "Inspecting checkpoint";
  setGraphControlsEnabled(false);
  elements.graphLoadingTitle.textContent = "Loading checkpoint…";
  elements.graphLoadingDetail.textContent = "Fetching repository metadata and checkpoint headers.";
  elements.hfImportStatus.textContent = "Loading checkpoint…";
  elements.hfImportButton.querySelector(".button-label").textContent = "Loading…";
}

async function completeImportLoading(tensorCount, checkpointFileCount, resolver = {}) {
  state.importOutcome = "success";
  elements.hfImport.dataset.outcome = "success";
  elements.cancelImportButton.disabled = true;
  elements.cancelImportButton.textContent = "Inspection complete";
  const exact = resolver.tier === "checkpoint-mapped";
  elements.graphLoadingTitle.textContent = exact ? "Circuit ready" : "Circuit scaffold ready";
  elements.graphLoadingDetail.textContent = exact
    ? `${new Intl.NumberFormat().format(tensorCount)} tensors mapped across ${checkpointFileCount} checkpoint ${checkpointFileCount === 1 ? "file" : "files"}.`
    : `${resolver.label ?? "Partial repository evidence"}; exact tensor shapes and counts remain explicitly unresolved.`;
  elements.appStatusText.textContent = "Circuit ready";
}

function failImportLoading(message) {
  state.importOutcome = "error";
  elements.hfImport.dataset.outcome = "error";
  elements.statusDot.classList.add("is-error");
  elements.appStatusText.textContent = "Inspection failed";
  elements.hfImportStatus.classList.add("error");
  elements.hfImportStatus.textContent = message;
  elements.hfModelInput.setAttribute("aria-invalid", "true");
}

function cancelImportLoading() {
  state.importOutcome = "cancelled";
  elements.hfImport.dataset.outcome = "cancelled";
  elements.hfImportStatus.classList.remove("error");
  elements.hfImportStatus.textContent = "Inspection cancelled. The previous graph is unchanged.";
  elements.statusDot.classList.remove("is-error");
  elements.appStatusText.textContent = state.model ? "Checkpoint circuit viewer" : "Ready for checkpoint";
}

function endImportLoading() {
  delete elements.hfImport.dataset.loading;
  delete elements.hfImportButton.dataset.loading;
  delete elements.hfImportStatus.dataset.loading;
  elements.graphPanel.classList.remove("is-loading");
  elements.graphLoading.hidden = true;
  elements.hfImport.removeAttribute("aria-busy");
  elements.graphPanel.removeAttribute("aria-busy");
  elements.hfImportButton.disabled = false;
  elements.gpt2DevExampleButton.disabled = false;
  elements.hfModelInput.disabled = false;
  elements.hfRevisionInput.disabled = false;
  elements.hfAccountToggleButton.disabled = false;
  elements.hfAccountConnectButton.disabled = false;
  elements.hfTokenInput.disabled = false;
  elements.cancelImportButton.disabled = false;
  elements.hfImportButton.querySelector(".button-label").textContent = "Inspect checkpoint";
  elements.statusDot.classList.remove("is-loading");
  state.importController = null;
  if (state.importOutcome === "success") {
    elements.appStatusText.textContent = state.model ? "Checkpoint circuit viewer" : "Ready for checkpoint";
  } else if (state.importOutcome === "error") {
    elements.hfModelInput.focus();
  }
  setGraphControlsEnabled(Boolean(state.model));
}

async function importHuggingFaceModel(event) {
  event?.preventDefault();
  const modelId = elements.hfModelInput.value.trim();
  if (!modelId) {
    state.importOutcome = "error";
    elements.hfImport.dataset.outcome = "error";
    elements.hfModelInput.setAttribute("aria-invalid", "true");
    elements.hfImportStatus.classList.add("error");
    elements.hfImportStatus.textContent = "Enter a Hugging Face repository ID.";
    elements.hfModelInput.focus();
    return;
  }
  const revision = elements.hfRevisionInput.value.trim() || "main";
  beginImportLoading();
  const controller = state.importController;
  try {
    const query = new URLSearchParams({ model: modelId, revision });
    const response = await fetch(`/api/huggingface?${query}`, {
      signal: controller.signal,
      credentials: "same-origin",
      headers: huggingFaceRequestHeaders()
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error ?? `Import failed (${response.status})`);
    if (!payload.graph) throw new Error("The Python backend did not return a graph");
    loadGraph(payload.graph);
    const resolver = payload.graph.resolver ?? {};
    const tensorCount = payload.graph.stats.checkpointTensors;
    const checkpointFileCount = Number(resolver.checkpointFileCount) || 0;
    elements.hfImportStatus.textContent = resolver.tier === "checkpoint-mapped"
      ? `${tensorCount} tensors · ${checkpointFileCount} Safetensors file${checkpointFileCount === 1 ? "" : "s"}`
      : resolver.tier === "manifest-mapped"
        ? `${tensorCount} tensor names · ${checkpointFileCount} PyTorch checkpoint file${checkpointFileCount === 1 ? "" : "s"} · shapes unavailable`
        : `${resolver.label ?? "Configuration scaffold"} · ${resolver.format ?? "unknown"} weights · exact tensors unavailable`;
    elements.loadedRepository.textContent = payload.graph.source.modelId;
    elements.loadedRevision.textContent = payload.graph.source.revision;
    elements.loadedCommit.textContent = payload.graph.source.sha?.slice(0, 12) ?? "—";
    elements.loadedTensors.textContent = Number.isFinite(tensorCount) ? new Intl.NumberFormat().format(tensorCount) : "Not mapped";
    elements.loadedShards.textContent = checkpointFileCount ? new Intl.NumberFormat().format(checkpointFileCount) : "—";
    elements.checkpointNote.textContent = resolver.tier === "checkpoint-mapped"
      ? "Every Safetensors header field is retained. Select a node to inspect its exact tensors."
      : [resolver.label, ...(resolver.limitations ?? [])].filter(Boolean).join(" ");
    elements.loadedRepositoryLink.href = payload.graph.source.url;
    renderPredictions(payload.graph.architecturePredictions);
    elements.importSummary.hidden = false;
    await completeImportLoading(tensorCount, checkpointFileCount, resolver);
    refreshDaytonaRecommendation();
    if (window.matchMedia?.("(max-width: 900px)")?.matches) setSidebarCollapsed(true);
    return true;
  } catch (error) {
    if (error?.name === "AbortError") cancelImportLoading();
    else failImportLoading(String(error?.message ?? error));
    return false;
  } finally {
    endImportLoading();
  }
}

function setSidebarCollapsed(collapsed) {
  elements.workspace.classList.toggle("sidebar-collapsed", collapsed);
  elements.sidebarToggleButton.setAttribute("aria-expanded", String(!collapsed));
  elements.sidebarToggleButton.title = collapsed ? "Show checkpoint setup" : "Hide checkpoint setup";
  elements.sidebarToggleLabel.textContent = collapsed ? "Show setup" : "Hide setup";
  setTimeout(handleViewportResize, 230);
}

function closeInspector({ restoreFocus = false } = {}) {
  const selectedId = state.selectedId;
  elements.inspector.hidden = true;
  state.selectedId = null;
  state.inspectorRenderToken += 1;
  state.inspectorRendered.clear();
  syncResidualLedgerSelection();
  elements.nodeLayer.querySelectorAll(".selected").forEach((node) => {
    node.classList.remove("selected");
    node.setAttribute("aria-pressed", "false");
  });
  if (restoreFocus && selectedId) {
    const selected = [...elements.nodeLayer.querySelectorAll(".graph-node")]
      .find((node) => node.dataset.nodeId === selectedId);
    selected?.focus({ preventScroll: true });
  }
}

async function copySelectedNodePath() {
  const node = state.model?.nodeById?.get(state.selectedId)
    ?? state.layout.find((entry) => entry.node.id === state.selectedId)?.node;
  if (!node) return;
  const value = node.path || node.id;
  try {
    if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(value);
    else {
      const input = document.createElement("textarea");
      input.value = value;
      input.setAttribute("readonly", "");
      input.className = "sr-only";
      document.body.append(input);
      input.select();
      document.execCommand("copy");
      input.remove();
    }
    elements.copyNodePath.textContent = "Copied";
    elements.inspectorActionStatus.textContent = `Copied ${value}`;
    if (state.copyResetTimer !== null) clearTimeout(state.copyResetTimer);
    state.copyResetTimer = setTimeout(() => {
      elements.copyNodePath.textContent = "Copy path";
      state.copyResetTimer = null;
    }, 1600);
  } catch {
    elements.inspectorActionStatus.textContent = "Could not copy this path.";
  }
}

elements.hfImport.addEventListener("submit", importHuggingFaceModel);
elements.gpt2DevExampleButton.addEventListener("click", openGpt2DevelopmentExample);
elements.debugCaseSelect.addEventListener("change", () => {
  const selected = Boolean(elements.debugCaseSelect.value);
  elements.openDebugCaseButton.disabled = !selected;
  elements.deleteDebugCaseButton.disabled = !selected;
});
elements.newDebugCaseButton.addEventListener("click", () => {
  resetDebugCaseEditor();
  if (state.model) {
    elements.debugCaseNameInput.value = `Debug ${state.model.source?.modelId ?? "case"}`;
    setModelInterfaceOpen(true);
  } else {
    setSidebarCollapsed(false);
    elements.hfModelInput.focus();
    elements.debugCaseLibraryStatus.textContent = "Open a model to start the new debug case.";
  }
});
elements.openDebugCaseButton.addEventListener("click", openSelectedDebugCase);
elements.deleteDebugCaseButton.addEventListener("click", deleteSelectedDebugCase);
elements.saveDebugCaseButton.addEventListener("click", () => saveDebugCase().catch(() => {}));
elements.duplicateDebugCaseButton.addEventListener("click", () => saveDebugCase({ duplicate: true }).catch(() => {}));
[
  elements.debugCaseNameInput,
  elements.debugExpectedInput,
  elements.debugNotesInput,
  elements.runtimePromptInput,
  elements.runtimeControlPromptInput,
  elements.runtimeTargetInput,
  elements.debugMetricNameInput,
  elements.debugMetricAnswerInput,
  elements.debugMetricPositiveInput,
  elements.debugMetricNegativeInput,
  elements.debugSeedInput,
  elements.debugBenchmarkName,
  elements.debugGuardrailsInput
].forEach((input) => input.addEventListener("input", () => setDebugDirty()));
elements.debugMetricKindSelect.addEventListener("change", () => {
  const names = { target_probability: "Target-token probability", logit_difference: "Correct vs incorrect logit difference", sequence_loss: "Sequence loss", kl_divergence: "Output KL divergence", multi_token_score: "Multi-token answer score", custom_token_groups: "Custom token-group difference" };
  elements.debugMetricNameInput.value = names[elements.debugMetricKindSelect.value];
  behaviourMetricSpec();
  setDebugDirty();
});
elements.runtimeCompareButton.addEventListener("click", runPairedComparison);
elements.generationMode.addEventListener("change", () => {
  const sampled = elements.generationMode.value === "sample";
  elements.generationTemperature.disabled = !sampled;
  elements.generationTopP.disabled = !sampled;
  setDebugDirty();
});
elements.runGenerationButton.addEventListener("click", runGenerationTrace);
elements.compareRunsButton.addEventListener("click", renderRunDiff);
elements.sweepMethod.addEventListener("change", () => {
  elements.sweepScale.disabled = elements.sweepMethod.value !== "scale";
  setDebugDirty();
});
elements.runCausalSweepButton.addEventListener("click", runCausalSweep);
elements.debugInterventionComponent.addEventListener("change", () => {
  elements.debugRunInterventionButton.disabled = !selectedInterventionComponent() || !modelRunReady() || state.runtime.busy;
});
elements.debugInterventionMethod.addEventListener("change", () => setDebugDirty());
elements.debugInterventionScale.addEventListener("input", () => setDebugDirty());
elements.debugInterventionScope.addEventListener("change", () => setDebugDirty());
elements.debugRunInterventionButton.addEventListener("click", runDebugIntervention);
elements.debugRunTraceButton.addEventListener("click", runRootCauseTrace);
elements.debugMicroscopeHook.addEventListener("change", () => {
  elements.debugLoadActivationButton.disabled = !elements.debugMicroscopeHook.value || !modelRunReady() || state.runtime.busy;
  elements.debugMicroscopeHead.disabled = !elements.debugMicroscopeHook.value.startsWith("attention_");
});
elements.debugLoadActivationButton.addEventListener("click", loadDebugActivation);
elements.debugBenchmarkFilter.addEventListener("change", renderBenchmarkExplorer);
elements.debugImportInboxButton.addEventListener("click", importBenchmarkExamples);
elements.debugRunInboxButton.addEventListener("click", runBenchmarkExamples);
elements.debugRunVerificationButton.addEventListener("click", runFixVerification);
elements.debugExportReportButton.addEventListener("click", exportDebugReport);
elements.hfAccountForm.addEventListener("submit", connectHuggingFaceAccount);
elements.hfAccountToggleButton.addEventListener("click", () => {
  if (state.hfAccount) disconnectHuggingFaceAccount();
  else setHuggingFaceAccountFormOpen(elements.hfAccountForm.hidden);
});
elements.hfAccountAvatar.addEventListener("error", () => {
  elements.hfAccountAvatar.hidden = true;
  elements.hfAccountAvatarFallback.hidden = false;
});
elements.settingsButtonAvatar.addEventListener("error", () => {
  elements.settingsButtonAvatar.hidden = true;
  elements.settingsButtonIcon.hidden = false;
});
elements.hfTokenInput.addEventListener("input", () => {
  elements.hfTokenInput.setAttribute("aria-invalid", "false");
  elements.hfAccountStatus.classList.remove("error");
  if (elements.hfAccount.dataset.state === "error") elements.hfAccount.dataset.state = "disconnected";
  elements.hfAccountStatus.textContent = "";
});
elements.hfModelInput.addEventListener("input", () => {
  if (elements.hfModelInput.getAttribute("aria-invalid") !== "true") return;
  elements.hfModelInput.setAttribute("aria-invalid", "false");
  elements.hfImportStatus.classList.remove("error");
  delete elements.hfImport.dataset.outcome;
  elements.hfImportStatus.textContent = state.model ? "Ready to inspect another checkpoint." : "Ready to inspect this checkpoint.";
});
elements.cancelImportButton.addEventListener("click", () => {
  if (!state.importController) return;
  elements.cancelImportButton.disabled = true;
  elements.cancelImportButton.textContent = "Cancelling…";
  state.importController.abort();
});
elements.collapseSidebarButton.addEventListener("click", () => setSidebarCollapsed(true));
elements.sidebarToggleButton.addEventListener("click", () => {
  setSidebarCollapsed(!elements.workspace.classList.contains("sidebar-collapsed"));
});
elements.emptyFocusButton.addEventListener("click", () => {
  setSidebarCollapsed(false);
  setTimeout(() => elements.hfModelInput.focus(), 230);
});
elements.searchInput.addEventListener("input", scheduleSearch);
elements.searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    if (state.searchFrame !== null) {
      cancelAnimationFrame(state.searchFrame);
      state.searchFrame = null;
      refreshSearch();
    }
    focusSearchMatch(event.shiftKey ? -1 : 1);
  } else if (event.key === "Escape") {
    elements.searchInput.value = "";
    refreshSearch();
    elements.svg.focus();
  }
});
elements.zoomOut.addEventListener("click", () => zoomBy(0.82));
elements.zoomReadout.addEventListener("click", fitGraph);
elements.zoomIn.addEventListener("click", () => zoomBy(1.22));
elements.watchlistButton.addEventListener("click", () => setWatchlistOpen(!state.debug.watchlistOpen));
elements.closeWatchlistButton.addEventListener("click", () => {
  setWatchlistOpen(false);
  elements.watchlistButton.focus();
});
elements.residualLedgerButton.addEventListener("click", () => setResidualLedgerOpen(!state.residualLedgerOpen));
elements.closeResidualLedger.addEventListener("click", () => {
  setResidualLedgerOpen(false);
  elements.residualLedgerButton.focus();
});
elements.residualLedgerList.addEventListener("click", (event) => {
  const row = event.target.closest(".residual-ledger-row");
  if (row?.dataset.nodeId) focusGraphNode(row.dataset.nodeId);
});
elements.fit.addEventListener("click", fitGraph);
elements.export.addEventListener("click", exportSvg);
elements.inspectorTabs.addEventListener("click", (event) => {
  const tab = event.target.closest("[data-inspector-tab]")?.dataset.inspectorTab;
  if (tab) setInspectorTab(tab);
});
elements.inspectorTabs.addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = [...elements.inspectorTabs.querySelectorAll("[data-inspector-tab]")];
  const current = tabs.indexOf(event.target.closest("[data-inspector-tab]"));
  if (current < 0) return;
  event.preventDefault();
  const next = event.key === "Home"
    ? 0
    : event.key === "End"
      ? tabs.length - 1
      : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  setInspectorTab(tabs[next].dataset.inspectorTab, { focus: true });
});
elements.copyNodePath.addEventListener("click", copySelectedNodePath);
elements.toggleWatchNodeButton.addEventListener("click", () => writeSelectedNodeAnnotation({ toggle: true }));
elements.saveNodeAnnotationButton.addEventListener("click", () => writeSelectedNodeAnnotation());
elements.closeInspector.addEventListener("click", () => closeInspector({ restoreFocus: true }));

document.addEventListener("keydown", (event) => {
  const target = event.target;
  const editable = target instanceof HTMLElement
    && (target.matches("input, textarea, select") || target.isContentEditable);
  const searchShortcut = event.key === "/" || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k");
  if (searchShortcut && state.model && !editable) {
    event.preventDefault();
    elements.searchInput.focus();
    elements.searchInput.select();
  } else if (event.key === "Escape" && state.debug.watchlistOpen) {
    event.preventDefault();
    setWatchlistOpen(false);
    elements.watchlistButton.focus();
  } else if (event.key === "Escape" && !elements.inspector.hidden) {
    event.preventDefault();
    closeInspector({ restoreFocus: true });
  } else if (!editable && state.model && event.key.toLowerCase() === "f") {
    event.preventDefault();
    fitGraph();
  }
});

function modelNodeFromEvent(event) {
  const graphNode = event.target.closest?.(".graph-node");
  if (!graphNode) return null;
  return state.model.nodeById?.get(graphNode.dataset.nodeId)
    ?? state.layout.find((entry) => entry.node.id === graphNode.dataset.nodeId)?.node
    ?? null;
}

elements.nodeLayer.addEventListener("click", (event) => {
  const node = modelNodeFromEvent(event);
  if (!node) return;
  event.stopPropagation();
  selectNode(node);
});

elements.nodeLayer.addEventListener("dblclick", (event) => {
  const node = modelNodeFromEvent(event);
  if (!node?.children?.length) return;
  event.stopPropagation();
  toggleCollapsed(node.id);
});

elements.nodeLayer.addEventListener("keydown", (event) => {
  const node = modelNodeFromEvent(event);
  if (!node) return;
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    selectNode(node);
    return;
  }
  if (event.key === "Home" || event.key === "End") {
    event.preventDefault();
    focusGraphNode(event.key === "Home" ? "residual_0" : "lm_head");
    return;
  }
  if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
  const current = state.layoutById.get(node.id);
  if (!current) return;
  const horizontal = event.key === "ArrowLeft" || event.key === "ArrowRight";
  const direction = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
  const candidates = state.layout.filter((entry) => {
    const primary = horizontal ? entry.x - current.x : entry.y - current.y;
    return entry.node.id !== node.id && Math.sign(primary) === direction;
  });
  const next = candidates.sort((left, right) => {
    const score = (entry) => {
      const primary = Math.abs(horizontal ? entry.x - current.x : entry.y - current.y);
      const secondary = Math.abs(horizontal ? entry.y - current.y : entry.x - current.x);
      return primary + secondary * 0.45;
    };
    return score(left) - score(right);
  })[0];
  if (!next) return;
  event.preventDefault();
  focusGraphNode(next.node.id);
});

elements.svg.addEventListener("wheel", (event) => {
  if (!state.model) return;
  event.preventDefault();
  markCameraMoving();
  if (event.ctrlKey || event.metaKey) {
    const rect = elements.svg.getBoundingClientRect();
    const exponent = Math.max(-0.24, Math.min(0.24, -event.deltaY * 0.0025));
    zoomBy(Math.exp(exponent), {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    });
    return;
  }

  // Trackpads provide both axes. A mouse wheel moves vertically; Shift moves horizontally.
  const horizontal = event.shiftKey && event.deltaX === 0 ? event.deltaY : event.deltaX;
  const vertical = event.shiftKey && event.deltaX === 0 ? 0 : event.deltaY;
  state.transform.x -= horizontal;
  state.transform.y -= vertical;
  applyCamera();
}, { passive: false });

elements.svg.addEventListener("keydown", (event) => {
  if (!state.model) return;
  const panStep = event.shiftKey ? 160 : 64;
  const movement = {
    ArrowLeft: [panStep, 0],
    ArrowRight: [-panStep, 0],
    ArrowUp: [0, panStep],
    ArrowDown: [0, -panStep]
  }[event.key];

  if (movement) {
    event.preventDefault();
    state.transform.x += movement[0];
    state.transform.y += movement[1];
    applyCamera();
  } else if (event.key === "Home") {
    event.preventDefault();
    fitGraph();
  } else if (event.key === "+" || event.key === "=") {
    event.preventDefault();
    zoomBy(1.2);
  } else if (event.key === "-") {
    event.preventDefault();
    zoomBy(0.82);
  }
});

elements.svg.addEventListener("pointerdown", (event) => {
  if (!state.model) return;
  if (event.target.closest(".graph-node")) return;
  elements.svg.setPointerCapture(event.pointerId);
  elements.svg.classList.add("dragging");
  state.drag = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    originX: state.transform.x,
    originY: state.transform.y
  };
});

elements.svg.addEventListener("pointermove", (event) => {
  if (!state.drag || state.drag.pointerId !== event.pointerId) return;
  state.transform.x = state.drag.originX + event.clientX - state.drag.startX;
  state.transform.y = state.drag.originY + event.clientY - state.drag.startY;
  applyCamera();
});

function stopDragging(event) {
  if (!state.drag || state.drag.pointerId !== event.pointerId) return;
  state.drag = null;
  elements.svg.classList.remove("dragging");
}

elements.svg.addEventListener("pointerup", stopDragging);
elements.svg.addEventListener("pointercancel", stopDragging);
elements.svg.addEventListener("click", (event) => {
  if (event.target === elements.svg) closeInspector();
});

function handleViewportResize() {
  const next = {
    width: elements.viewport.clientWidth,
    height: elements.viewport.clientHeight
  };
  const previous = state.viewportSize;
  state.viewportSize = next;
  if (!state.model || !previous || !next.width || !next.height) return;
  state.transform.x += (next.width - previous.width) / 2;
  state.transform.y += (next.height - previous.height) / 2;
  state.renderedRange = null;
  applyCamera();
}

if ("ResizeObserver" in window) {
  new ResizeObserver(handleViewportResize).observe(elements.viewport);
} else {
  window.addEventListener("resize", handleViewportResize);
}
handleViewportResize();

setGraphControlsEnabled(false);
renderRunHistory();
renderWatchlist();
restoreHuggingFaceAccount();
setRuntimeMode("daytona");
restoreRuntimeConnection();
restoreDebugCases();

elements.workspaceLaunchers.forEach((button) => button.addEventListener("click", () => setAppView("workspace")));
elements.appHomeButton.addEventListener("click", () => setAppView("landing"));
elements.tutorialButton.addEventListener("click", openLandingTutorial);
elements.settingsButton.addEventListener("click", () => setSettingsOpen(true));
elements.settingsBackdrop.addEventListener("click", () => setSettingsOpen(false));
elements.closeSettingsButton.addEventListener("click", () => setSettingsOpen(false));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.settingsOpen) {
    event.preventDefault();
    setSettingsOpen(false);
  } else if (event.key === "Escape" && state.modelInterfaceOpen) {
    event.preventDefault();
    setModelInterfaceOpen(false);
    elements.runModelButton.focus();
  }
});
elements.runtimeConnectButton.addEventListener("click", connectRuntime);
elements.runtimeModeLocal.addEventListener("click", () => setRuntimeMode("local"));
elements.runtimeModeDaytona.addEventListener("click", () => setRuntimeMode("daytona"));
elements.copyRuntimeCommandButton.addEventListener("click", copyRuntimeCommand);
elements.runtimeDisconnectButton.addEventListener("click", disconnectRuntime);
elements.runtimeLoadButton.addEventListener("click", loadRuntimeModel);
elements.modelRunForm.addEventListener("submit", runHookedForwardPass);
elements.runModelButton.addEventListener("click", () => setModelInterfaceOpen(!state.modelInterfaceOpen));
elements.closeModelInterfaceButton.addEventListener("click", () => {
  setModelInterfaceOpen(false);
  elements.runModelButton.focus();
});
elements.modelRunSettingsButton.addEventListener("click", () => {
  setModelInterfaceOpen(false);
  setSettingsOpen(true);
});
elements.runtimeEndpointInput.addEventListener("input", () => elements.runtimeStatus.classList.remove("error"));
elements.runtimeSecretInput.addEventListener("input", () => elements.runtimeStatus.classList.remove("error"));
elements.runtimeDaytonaApiKeyInput.addEventListener("input", () => elements.runtimeStatus.classList.remove("error"));
window.addEventListener("hashchange", () => {
  setAppView(appViewFromHash(), { focus: false, updateHistory: false });
});
setAppView(appViewFromHash(), { focus: false, updateHistory: false });
