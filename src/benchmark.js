const PASS_OUTCOMES = new Set(["pass", "passed", "correct", "success", "succeeded", "ok"]);
const FAIL_OUTCOMES = new Set(["fail", "failed", "incorrect", "failure"]);

function firstValue(item, names) {
  for (const name of names) {
    const value = item?.[name];
    if (value !== undefined && value !== null && String(value).trim() !== "") return value;
  }
  return "";
}

function firstText(item, names) {
  const value = firstValue(item, names);
  return typeof value === "string" || typeof value === "number" ? String(value).trim() : "";
}

function finiteNumber(value) {
  if (value === "" || value === null || value === undefined || typeof value === "boolean") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function normaliseBenchmarkOutcome(value) {
  if (typeof value === "boolean") return value ? "passed" : "failed";
  const outcome = String(value ?? "").trim().toLowerCase();
  if (!outcome) return "pending";
  if (PASS_OUTCOMES.has(outcome)) return "passed";
  if (FAIL_OUTCOMES.has(outcome)) return "failed";
  if (["regressed", "regression"].includes(outcome)) return "regressed";
  if (["anomaly", "anomalous", "outlier"].includes(outcome)) return "anomaly";
  return outcome.replace(/\s+/g, "-");
}

export function normaliseBenchmarkExamples(values, options = {}) {
  const fallbackBenchmark = String(options.benchmark ?? "").trim();
  const idFactory = options.idFactory ?? (() => crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`);
  return values.filter((item) => item && typeof item === "object").map((item, index) => {
    const prompt = firstText(item, ["prompt", "input", "instruction", "question", "query", "text", "failure"])
      || Object.values(item).find((value) => typeof value === "string" && value.trim())
      || "";
    const rawOutcome = firstValue(item, ["outcome", "status", "passed", "correct", "success"]);
    const normalisedOutcome = rawOutcome === "" ? "pending" : normaliseBenchmarkOutcome(rawOutcome);
    const suppliedOutcome = normalisedOutcome === "pending" ? "" : normalisedOutcome;
    return {
      id: idFactory(item, index),
      benchmark: firstText(item, ["benchmark", "benchmark_name", "suite", "dataset"]) || fallbackBenchmark || "Imported benchmark",
      task: firstText(item, ["task", "task_name", "category", "subset", "config"]),
      split: firstText(item, ["split", "partition"]),
      exampleId: firstText(item, ["example_id", "exampleId", "id", "idx", "index"]) || String(index + 1),
      prompt: String(prompt).trim(),
      reference: firstText(item, ["reference", "reference_prompt", "referencePrompt", "control", "controlPrompt", "clean"]),
      expected: firstText(item, ["expected", "expected_output", "answer", "output", "response", "completion", "label_text"]),
      target: firstText(item, ["target", "target_token", "targetToken"]),
      score: finiteNumber(firstValue(item, ["score", "metric_value", "metricValue", "reward", "result"])),
      threshold: finiteNumber(firstValue(item, ["threshold", "pass_threshold", "passThreshold"])),
      direction: firstText(item, ["direction", "metric_direction", "metricDirection"]).toLowerCase() === "minimize" ? "minimize" : "maximize",
      suppliedOutcome,
      status: suppliedOutcome || "pending",
      outcomeSource: suppliedOutcome ? "benchmark" : "not evaluated",
      cluster: "Untriaged"
    };
  }).filter((item) => item.prompt);
}

export function benchmarkOutcomeForRun(item, run) {
  if (item.suppliedOutcome) {
    return { status: item.suppliedOutcome, source: "benchmark-supplied outcome" };
  }
  const value = finiteNumber(run?.metric?.value);
  if (value !== null && Number.isFinite(item.threshold)) {
    const passed = item.direction === "minimize" ? value <= item.threshold : value >= item.threshold;
    return { status: passed ? "passed" : "failed", source: `${item.direction} threshold ${item.threshold}` };
  }
  const prediction = String(run?.nextToken?.topK?.[0]?.text ?? run?.nextToken?.topK?.[0]?.token ?? "").trim().toLowerCase();
  const expected = String(item.expected ?? "").trim().toLowerCase();
  if (prediction && expected) {
    return {
      status: expected.startsWith(prediction) ? "passed" : "failed",
      source: "first-token expected-output check"
    };
  }
  return { status: "observed", source: "no pass criterion supplied" };
}
