const MAX_RUN_HISTORY = 16;

function finite(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function quantile(sortedValues, probability) {
  if (!sortedValues.length) return null;
  const offset = (sortedValues.length - 1) * probability;
  const lower = Math.floor(offset);
  const fraction = offset - lower;
  const upper = sortedValues[lower + 1];
  return upper === undefined
    ? sortedValues[lower]
    : sortedValues[lower] + fraction * (upper - sortedValues[lower]);
}

export function buildHistogram(values, { maximumBins = 24, minimumBins = 5, symmetric = false } = {}) {
  const finiteValues = (Array.isArray(values) ? values : [])
    .filter((value) => typeof value === "number" && Number.isFinite(value))
    .sort((left, right) => left - right);
  if (!finiteValues.length) {
    return { bins: [], count: 0, domain: null, statistics: null, method: "freedman-diaconis" };
  }

  const count = finiteValues.length;
  const q1 = quantile(finiteValues, 0.25);
  const median = quantile(finiteValues, 0.5);
  const q3 = quantile(finiteValues, 0.75);
  const mean = finiteValues.reduce((sum, value) => sum + value, 0) / count;
  const rawMinimum = finiteValues[0];
  const rawMaximum = finiteValues[count - 1];
  let minimum = rawMinimum;
  let maximum = rawMaximum;
  if (symmetric) {
    const magnitude = Math.max(Math.abs(minimum), Math.abs(maximum));
    minimum = -magnitude;
    maximum = magnitude;
  }
  if (minimum === maximum) {
    const padding = Math.max(Math.abs(minimum) * 0.05, 0.5);
    minimum -= padding;
    maximum += padding;
  }

  const range = maximum - minimum;
  const iqr = q3 - q1;
  const fdWidth = count > 1 && iqr > 0 ? (2 * iqr) / Math.cbrt(count) : 0;
  const sturges = Math.ceil(Math.log2(count) + 1);
  let binCount = fdWidth > 0 ? Math.ceil(range / fdWidth) : sturges;
  const safeMaximum = Math.max(1, Math.min(Math.floor(maximumBins), count));
  const safeMinimum = Math.min(safeMaximum, Math.max(1, Math.floor(minimumBins)));
  binCount = Math.max(safeMinimum, Math.min(safeMaximum, binCount));
  if (symmetric && binCount > 1 && binCount % 2) {
    binCount = binCount < safeMaximum ? binCount + 1 : binCount - 1;
  }
  binCount = Math.max(1, binCount);

  const width = range / binCount;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    start: minimum + index * width,
    end: index === binCount - 1 ? maximum : minimum + (index + 1) * width,
    count: 0,
  }));
  finiteValues.forEach((value) => {
    const index = value === maximum
      ? binCount - 1
      : Math.max(0, Math.min(binCount - 1, Math.floor((value - minimum) / width)));
    bins[index].count += 1;
  });
  return {
    bins,
    count,
    domain: [minimum, maximum],
    statistics: { minimum: rawMinimum, q1, median, mean, q3, maximum: rawMaximum },
    method: fdWidth > 0 ? "freedman-diaconis" : "sturges",
  };
}

function metricIdentity(metric, target) {
  return JSON.stringify({
    kind: metric?.kind ?? null,
    name: metric?.name ?? null,
    targetToken: target?.text ?? target?.token ?? null,
    direction: metric?.direction ?? null,
  });
}

export function createRunSnapshot(run, { kind = "forward", label = "", createdAt = new Date().toISOString() } = {}) {
  const steps = run?.steps ?? [];
  const lastStep = steps.length ? steps[steps.length - 1] : null;
  const identifier = run?.runId ?? run?.generationId ?? run?.interventionId;
  if (!identifier) throw new Error("A run snapshot requires a stable run identifier");
  const metric = run?.metric ?? null;
  const target = run?.target ?? lastStep?.token ?? null;
  return {
    id: String(identifier),
    kind,
    label: label || (kind === "generation" ? `Generation · ${steps.length} tokens` : `Run ${String(identifier).slice(0, 8)}`),
    createdAt,
    modelId: run?.modelId ?? null,
    revision: run?.revision ?? null,
    prompt: run?.prompt ?? "",
    completion: run?.completion ?? "",
    metric: metric ? {
      name: metric.name ?? "Behaviour metric",
      kind: metric.kind ?? null,
      value: finite(metric.value),
      direction: metric.direction ?? null,
    } : null,
    metricIdentity: metricIdentity(metric, target),
    target: target ? {
      text: target.text ?? target.token ?? "",
      tokenId: target.tokenId ?? target.id ?? null,
      probability: finite(target.probability ?? lastStep?.chosenProbability),
      logit: finite(target.logit ?? lastStep?.chosenLogit),
      rank: finite(target.rank ?? lastStep?.chosenRank),
    } : null,
    output: {
      topToken: run?.nextToken?.topK?.[0]?.text ?? lastStep?.token?.text ?? null,
      entropy: finite(run?.nextToken?.entropy ?? lastStep?.entropy),
      generatedTokens: steps.length || null,
    },
    context: {
      seed: run?.context?.seed ?? run?.settings?.seed ?? null,
      device: run?.context?.device ?? null,
      dtype: run?.context?.dtype ?? null,
      chatTemplate: run?.context?.chatTemplate ?? null,
      worker: run?.context?.software?.worker ?? null,
    },
    layers: (run?.layers ?? []).map((layer) => ({
      layer: Number(layer.layer),
      residualNorm: finite(layer.residPost?.norm),
      residualDla: finite(layer.residPost?.dla),
      attentionNorm: finite(layer.attentionWrite?.norm),
      attentionDla: finite(layer.attentionWrite?.dla),
      mlpNorm: finite(layer.mlpWrite?.norm),
      mlpDla: finite(layer.mlpWrite?.dla),
    })),
    evidence: run?.evidence ?? null,
  };
}

export function appendRunHistory(history, snapshot, maximum = MAX_RUN_HISTORY) {
  const values = Array.isArray(history) ? history.filter((item) => item?.id !== snapshot.id) : [];
  values.unshift(snapshot);
  return values.slice(0, Math.max(2, maximum));
}

export function compareRunSnapshots(left, right) {
  if (!left || !right) throw new Error("Choose two run snapshots to compare");
  const sameModel = Boolean(left.modelId && left.modelId === right.modelId);
  const sameRevision = sameModel && left.revision === right.revision;
  const sameTarget = Boolean(
    (left.target?.tokenId != null && left.target.tokenId === right.target?.tokenId)
    || (left.target?.text && left.target.text === right.target?.text)
  );
  const metricComparable = Boolean(
    sameRevision
    && left.metricIdentity === right.metricIdentity
    && Number.isFinite(left.metric?.value)
    && Number.isFinite(right.metric?.value)
  );
  const rightLayers = new Map((right.layers ?? []).map((item) => [item.layer, item]));
  const layers = (left.layers ?? []).flatMap((item) => {
    const other = rightLayers.get(item.layer);
    if (!other) return [];
    return [{
      layer: item.layer,
      residualNormDelta: Number.isFinite(item.residualNorm) && Number.isFinite(other.residualNorm) ? other.residualNorm - item.residualNorm : null,
      residualDlaDelta: Number.isFinite(item.residualDla) && Number.isFinite(other.residualDla) ? other.residualDla - item.residualDla : null,
      attentionDlaDelta: Number.isFinite(item.attentionDla) && Number.isFinite(other.attentionDla) ? other.attentionDla - item.attentionDla : null,
      mlpDlaDelta: Number.isFinite(item.mlpDla) && Number.isFinite(other.mlpDla) ? other.mlpDla - item.mlpDla : null,
    }];
  });
  return {
    left,
    right,
    compatibility: {
      sameModel,
      sameRevision,
      sameTarget,
      metricComparable,
      note: !sameModel
        ? "Runs use different models; numerical differences are descriptive only."
        : !sameRevision
          ? "Runs use different revisions; architecture and scale may differ."
          : !metricComparable
            ? "Metric specifications or targets differ, so no metric delta is reported."
            : "Runs share the same model revision and metric identity.",
    },
    metricDelta: metricComparable ? right.metric.value - left.metric.value : null,
    targetProbabilityDelta: sameRevision && sameTarget && Number.isFinite(left.target?.probability) && Number.isFinite(right.target?.probability)
      ? right.target.probability - left.target.probability
      : null,
    entropyDelta: sameRevision && Number.isFinite(left.output?.entropy) && Number.isFinite(right.output?.entropy)
      ? right.output.entropy - left.output.entropy
      : null,
    layers,
  };
}

export function normaliseWatchlist(values) {
  if (!Array.isArray(values)) return [];
  const seen = new Set();
  return values.flatMap((value) => {
    const nodeId = String(value?.nodeId ?? "").trim();
    if (!nodeId || seen.has(nodeId)) return [];
    seen.add(nodeId);
    const evidence = ["hypothesis", "observation", "causal", "note"].includes(value?.evidence)
      ? value.evidence
      : "hypothesis";
    return [{
      nodeId,
      label: String(value?.label ?? nodeId).slice(0, 180),
      evidence,
      note: String(value?.note ?? "").slice(0, 2000),
      createdAt: String(value?.createdAt ?? new Date().toISOString()),
    }];
  });
}

export function watchlistStorageKey(modelId, revision = "main") {
  return `modeldebugger.watchlist.${encodeURIComponent(String(modelId || "none"))}.${encodeURIComponent(String(revision || "main"))}`;
}

export { MAX_RUN_HISTORY };
