import { GPT2_DIAGNOSTIC as record } from "./gpt2-diagnostic-data.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function format(value, digits = 3) {
  return Number.isFinite(value)
    ? new Intl.NumberFormat("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value)
    : "—";
}

function signed(value, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  return `${value > 0 ? "+" : value < 0 ? "−" : ""}${format(Math.abs(value), digits)}`;
}

function svgElement(tag, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
  return element;
}

function renderTokens(containerId, tokens, changedIndex) {
  const container = document.querySelector(`#${containerId}`);
  tokens.forEach((token, index) => {
    const chip = document.createElement("span");
    chip.textContent = token;
    chip.classList.toggle("changed", index === changedIndex);
    chip.title = `Token ${index}${index === changedIndex ? " · matched lexical change" : ""}`;
    container.append(chip);
  });
}

function renderDivergenceChart() {
  const container = document.querySelector("#divergenceChart");
  const values = [...record.observation.layerDivergence].sort((left, right) => left.layer - right.layer);
  const width = 900;
  const height = 245;
  const padding = { left: 42, right: 22, top: 27, bottom: 34 };
  const maximum = Math.max(...values.map((item) => item.score)) * 1.08;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const x = (layer) => padding.left + (layer / (values.length - 1)) * plotWidth;
  const y = (value) => padding.top + (1 - value / maximum) * plotHeight;
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Composite paired-trace divergence by GPT-2 layer, rising sharply at layers 9 and 10",
  });

  [0, 0.5, 1, 1.5].forEach((tick) => {
    const lineY = y(tick);
    svg.append(svgElement("line", { class: "demo-chart-gridline", x1: padding.left, x2: width - padding.right, y1: lineY, y2: lineY }));
    const label = svgElement("text", { class: "demo-chart-axis", x: padding.left - 8, y: lineY + 3, "text-anchor": "end" });
    label.textContent = format(tick, 1);
    svg.append(label);
  });

  values.forEach((item) => {
    const label = svgElement("text", { class: "demo-chart-axis", x: x(item.layer), y: height - 10, "text-anchor": "middle" });
    label.textContent = `L${item.layer}`;
    svg.append(label);
  });
  const points = values.map((item) => `${x(item.layer).toFixed(2)} ${y(item.score).toFixed(2)}`);
  const area = `M${x(0)} ${height - padding.bottom} L${points.join(" L")} L${x(values.length - 1)} ${height - padding.bottom} Z`;
  svg.append(svgElement("path", { class: "demo-chart-area", d: area }));
  svg.append(svgElement("path", { class: "demo-chart-line", d: `M${points.join(" L")}` }));

  const materialX = x(record.observation.firstMaterialLayer);
  svg.append(svgElement("line", { class: "demo-chart-marker", x1: materialX, x2: materialX, y1: padding.top, y2: height - padding.bottom }));
  const markerLabel = svgElement("text", { class: "demo-chart-label", x: materialX + 7, y: padding.top + 2 });
  markerLabel.textContent = "first material divergence";
  svg.append(markerLabel);

  values.forEach((item) => {
    const point = svgElement("circle", {
      class: `demo-chart-point${item.layer >= 9 ? " material" : ""}`,
      cx: x(item.layer),
      cy: y(item.score),
      r: item.layer >= 9 ? 4 : 3,
      tabindex: "0",
      role: "img",
      "aria-label": `Layer ${item.layer}, divergence score ${format(item.score, 3)}`,
    });
    const title = svgElement("title");
    title.textContent = `Layer ${item.layer} · score ${format(item.score, 3)} · relative residual distance ${format(item.relativeDistance, 3)}`;
    point.append(title);
    svg.append(point);
  });
  container.append(svg);
}

function predictionColumn(label, predictions) {
  const column = document.createElement("div");
  column.className = "demo-prediction-column";
  const heading = document.createElement("strong");
  heading.textContent = label;
  column.append(heading);
  const maximum = Math.max(...predictions.map((item) => item.probability));
  predictions.forEach((prediction) => {
    const row = document.createElement("div");
    row.className = "demo-prediction-row";
    const token = document.createElement("span");
    const track = document.createElement("i");
    const bar = document.createElement("b");
    const probability = document.createElement("small");
    token.textContent = prediction.text;
    bar.style.setProperty("--bar", `${prediction.probability / maximum * 100}%`);
    probability.textContent = `${format(prediction.probability * 100, 1)}%`;
    track.append(bar);
    row.append(token, track, probability);
    column.append(row);
  });
  return column;
}

function renderPredictions() {
  document.querySelector("#predictionComparison").append(
    predictionColumn("Selected · France", record.observation.selectedTopTokens),
    predictionColumn("Reference · Germany", record.observation.referenceTopTokens),
  );
}

function renderCandidates() {
  const container = document.querySelector("#candidateList");
  record.observation.topComponents.forEach((candidate, index) => {
    const row = document.createElement("div");
    row.className = "demo-candidate-row";
    const rank = document.createElement("span");
    const label = document.createElement("strong");
    const score = document.createElement("small");
    rank.textContent = String(index + 1).padStart(2, "0");
    label.textContent = candidate.label;
    score.textContent = format(candidate.score, 3);
    row.append(rank, label, score);
    container.append(row);
  });
}

function heatmapCell(value, isPeak, label) {
  const cell = document.createElement("div");
  const magnitude = Math.abs(value);
  const strength = Math.min(90, 10 + magnitude / 6.2 * 80);
  cell.className = `demo-heatmap-cell${isPeak ? " peak" : ""}`;
  cell.style.setProperty("--cell-color", value < 0 ? "#ad5038" : "#6f8156");
  cell.style.setProperty("--cell-strength", `${strength}%`);
  cell.style.setProperty("--cell-text", strength > 52 ? "#fff" : "#393631");
  cell.textContent = signed(value, Math.abs(value) >= 1 ? 2 : 3);
  cell.title = `${label} · signed metric effect ${signed(value, 4)} logits`;
  cell.setAttribute("role", "img");
  cell.setAttribute("aria-label", cell.title);
  return cell;
}

function renderCausalHeatmap() {
  const container = document.querySelector("#causalHeatmap");
  const corner = document.createElement("span");
  corner.className = "demo-heatmap-corner";
  corner.textContent = "COMPONENT";
  container.append(corner);
  for (let layer = 0; layer < 12; layer += 1) {
    const header = document.createElement("span");
    header.className = "demo-heatmap-layer";
    header.textContent = `L${layer}`;
    container.append(header);
  }
  Object.entries(record.intervention.sweeps).forEach(([kind, values]) => {
    const label = document.createElement("strong");
    label.className = "demo-heatmap-label";
    label.textContent = kind === "mlp" ? "MLP" : kind[0].toUpperCase() + kind.slice(1);
    container.append(label);
    const maximum = Math.max(...values.map(Math.abs));
    values.forEach((value, layer) => container.append(heatmapCell(value, Math.abs(value) === maximum, `Layer ${layer} ${kind}`)));
  });
}

function renderClaims() {
  [
    ["#supportedClaims", record.diagnosis.supported],
    ["#unsupportedClaims", record.diagnosis.notSupported],
  ].forEach(([selector, claims]) => {
    const list = document.querySelector(selector);
    claims.forEach((claim) => {
      const item = document.createElement("li");
      item.textContent = claim;
      list.append(item);
    });
  });
  document.querySelector("#nextExperiment").textContent = record.diagnosis.nextExperiment;
}

function renderPerformance() {
  const container = document.querySelector("#performanceWaterfall");
  const colours = { cpu: "#9d7a34", transfer: "#557fa9", instrumentation: "#a66550", model: "#ad5038", analysis: "#6f8156", storage: "#77726a" };
  record.performance.phases.forEach((phase) => {
    const row = document.createElement("div");
    row.className = "demo-performance-row";
    const label = document.createElement("strong");
    const track = document.createElement("div");
    const bar = document.createElement("i");
    const duration = document.createElement("small");
    label.textContent = phase.label;
    track.className = "demo-performance-track";
    bar.style.setProperty("--duration", `${phase.durationMs / record.performance.workerTotalMs * 100}%`);
    bar.style.setProperty("--phase-color", colours[phase.category] ?? colours.cpu);
    duration.textContent = `${format(phase.durationMs, 1)} ms`;
    track.append(bar);
    row.append(label, track, duration);
    container.append(row);
  });
  document.querySelector("#performanceNote").textContent = record.performance.note;
}

function reproductionSpec() {
  return {
    modelId: record.model.id,
    revision: record.model.revision,
    selectedPrompt: record.question.selectedPrompt,
    referencePrompt: record.question.referencePrompt,
    metric: record.question.metric,
    seed: record.model.seed,
    dtype: record.model.dtype,
    intervention: {
      component: record.intervention.primaryResult.component,
      nodeId: record.intervention.primaryResult.nodeId,
      method: record.intervention.method,
      position: record.question.interventionPosition,
      source: "aligned reference activation",
    },
  };
}

async function copyReproductionSpec() {
  const status = document.querySelector("#copyStatus");
  try {
    await navigator.clipboard.writeText(JSON.stringify(reproductionSpec(), null, 2));
    status.textContent = "Copied";
  } catch {
    status.textContent = "Clipboard unavailable — expand the JSON record below.";
  }
}

renderTokens("selectedTokens", record.question.selectedTokens, record.question.changedTokenIndex);
renderTokens("referenceTokens", record.question.referenceTokens, record.question.changedTokenIndex);
renderDivergenceChart();
renderPredictions();
renderCandidates();
renderCausalHeatmap();
renderClaims();
renderPerformance();
document.querySelector("#selectedMetricValue").textContent = signed(record.observation.selectedMetric);
document.querySelector("#referenceMetricValue").textContent = signed(record.observation.referenceMetric);
document.querySelector("#metricDifferenceValue").textContent = signed(record.observation.selectedMinusReference);
document.querySelector("#outputKlValue").textContent = format(record.observation.outputKL);
document.querySelector("#provenanceCapturedAt").textContent = new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeStyle: "short" }).format(new Date(record.capturedAt));
document.querySelector("#recordJson").textContent = JSON.stringify(record, null, 2);
document.querySelector("#copyReproductionButton").addEventListener("click", copyReproductionSpec);
