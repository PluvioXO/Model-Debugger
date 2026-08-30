"""Authenticated ModelDebugger worker for local and managed Daytona runtimes.

Model weights and full activation tensors remain in the worker runtime while the
browser receives compact, JSON-safe summaries through the loopback backend.
"""

from __future__ import annotations

import argparse
import difflib
import hmac
import json
import math
import os
import platform
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

WORKER_VERSION = "0.4.0"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_PROMPT_TOKENS = 2048
MAX_STORED_RUNS = 8
MAX_GENERATED_TOKENS = 32
MAX_LOGIT_LENS_STAGES = 48
MAX_SWEEP_CELLS = 128
LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|h|blocks|block|layer)\.(\d+)(?:\.|$)")
BLOCK_PATTERN = re.compile(r"(?:^|\.)(?:layers|h|blocks|block|layer)\.(\d+)$")
ATTENTION_NAMES = {"self_attn", "attn", "attention", "linear_attn"}
MLP_NAMES = {"mlp", "feed_forward", "ffn", "block_sparse_moe", "moe"}
ATTENTION_OUTPUT_PROJECTIONS = {"o_proj", "out_proj", "c_proj", "dense"}


@dataclass
class RunRecord:
    run_id: str
    request: dict[str, Any]
    result: dict[str, Any]
    tensors: dict[str, Any]
    logits: Any
    created_at: float


class InferenceWaterfall:
    """Record contiguous worker phases against one monotonic clock."""

    def __init__(self, clock: Any = time.perf_counter):
        self._clock = clock
        self._origin = float(clock())
        self._cursor = self._origin
        self._phases: list[dict[str, Any]] = []

    def finish(self, key: str, label: str, category: str, detail: str) -> None:
        ended = float(self._clock())
        start_ms = max(0.0, (self._cursor - self._origin) * 1000)
        duration_ms = max(0.0, (ended - self._cursor) * 1000)
        self._phases.append({
            "key": key,
            "label": label,
            "category": category,
            "detail": detail,
            "startMs": start_ms,
            "durationMs": duration_ms,
            "endMs": start_ms + duration_ms,
        })
        self._cursor = ended

    def snapshot(self) -> dict[str, Any]:
        total_ms = max(0.0, (self._cursor - self._origin) * 1000)
        return {
            "clock": "time.perf_counter",
            "scope": "worker",
            "totalMs": total_ms,
            "phases": [
                {
                    **phase,
                    "share": phase["durationMs"] / total_ms if total_ms > 0 else 0.0,
                }
                for phase in self._phases
            ],
            "note": (
                "Worker-side wall time measured with a monotonic clock. Accelerator work is synchronized at phase "
                "boundaries. The model-forward phase includes capture-hook overhead and is not raw serving latency; "
                "browser transport and JSON serialization are outside this timeline."
            ),
        }


@dataclass
class WorkerState:
    secret: str
    lock: threading.RLock = field(default_factory=threading.RLock)
    model: Any = None
    tokenizer: Any = None
    model_id: str = ""
    revision: str = ""
    device: str = ""
    dtype: str = ""
    quantization: str = "none"
    loaded_at: float | None = None
    latest_run_id: str = ""
    latest_tensors: dict[str, Any] = field(default_factory=dict)
    runs: dict[str, RunRecord] = field(default_factory=dict)
    run_order: list[str] = field(default_factory=list)


def _layer_index(name: str) -> int | None:
    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _module_category(name: str) -> str | None:
    if BLOCK_PATTERN.search(name):
        return "residual"
    leaf = name.rsplit(".", 1)[-1]
    if leaf in ATTENTION_NAMES:
        return "attention"
    if leaf in MLP_NAMES:
        return "mlp"
    return None


def _is_attention_output_projection(name: str) -> bool:
    parts = name.split(".")
    if not parts or parts[-1] not in ATTENTION_OUTPUT_PROJECTIONS:
        return False
    parents = set(parts[:-1])
    return bool(parents & ATTENTION_NAMES)


def _first_tensor(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        return None
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _cpu_tensor(tensor: Any) -> Any:
    import torch

    detached = tensor.detach()
    if detached.is_floating_point():
        detached = detached.to(dtype=torch.float16)
    return detached.cpu()


def _position_vector(tensor: Any, position: int = -1) -> Any:
    if tensor.ndim >= 3:
        return tensor[0, position]
    if tensor.ndim == 2:
        return tensor[position]
    return tensor


def _tensor_stats(tensor: Any, position: int = -1) -> dict[str, Any]:
    import torch

    vector = _position_vector(tensor, position).detach().float()
    finite_mask = torch.isfinite(vector)
    finite = vector[finite_mask]
    nan_count = int(torch.isnan(vector).sum().item())
    positive_infinity = int(torch.isposinf(vector).sum().item())
    negative_infinity = int(torch.isneginf(vector).sum().item())
    if finite.numel() == 0:
        return {
            "norm": None,
            "rms": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "maxAbs": None,
            "zeroFraction": None,
            "finiteFraction": 0.0,
            "nanCount": nan_count,
            "positiveInfinityCount": positive_infinity,
            "negativeInfinityCount": negative_infinity,
        }
    return {
        "norm": float(torch.linalg.vector_norm(finite).item()),
        "rms": float(torch.sqrt(torch.mean(finite.square())).item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "maxAbs": float(finite.abs().max().item()),
        "zeroFraction": float(torch.count_nonzero(finite == 0).item() / finite.numel()),
        "finiteFraction": float(finite.numel() / max(1, vector.numel())),
        "nanCount": nan_count,
        "positiveInfinityCount": positive_infinity,
        "negativeInfinityCount": negative_infinity,
    }


def _tensor_summary(tensor: Any, hook_name: str, position: int = -1) -> dict[str, Any]:
    return {
        "hookName": hook_name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        **_tensor_stats(tensor, position),
    }


def _model_input_device(model: Any) -> Any:
    import torch

    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _synchronize_accelerator(torch: Any, device: Any) -> None:
    """Finish queued accelerator work so phase wall times are not under-reported."""
    device_type = str(getattr(device, "type", device)).split(":", 1)[0].lower()
    backend = getattr(torch, device_type, None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _available_accelerator(torch: Any) -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "Apple Metal (MPS)"
    return "cpu", "CPU"


def _worker_memory_capacity(torch: Any, accelerator_type: str) -> dict[str, Any]:
    system_bytes = 0
    try:
        system_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    if accelerator_type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {"kind": "cuda", "availableBytes": int(free_bytes), "totalBytes": int(total_bytes), "systemBytes": system_bytes}
    if accelerator_type == "mps":
        recommended = getattr(torch.mps, "recommended_max_memory", lambda: system_bytes)()
        return {"kind": "mps", "availableBytes": int(recommended or system_bytes), "totalBytes": system_bytes}
    return {"kind": "cpu", "availableBytes": system_bytes, "totalBytes": system_bytes}


def _preflight_model_size(torch: Any, payload: dict[str, Any], accelerator_type: str, quantization: str) -> dict[str, Any]:
    parameters = max(0, int(payload.get("parameterCount", 0) or 0))
    checkpoint_bytes = max(0, int(payload.get("checkpointBytes", 0) or 0))
    if quantization == "4bit" and parameters:
        estimated_bytes = int(parameters * 0.65)
    else:
        estimated_bytes = checkpoint_bytes or parameters * 2
    memory = _worker_memory_capacity(torch, accelerator_type)
    available = int(memory.get("availableBytes", 0))
    if estimated_bytes and available and estimated_bytes * 1.2 > available:
        estimated_gib = estimated_bytes / (1024 ** 3)
        available_gib = available / (1024 ** 3)
        raise ValueError(
            f"This checkpoint needs roughly {estimated_gib:.1f} GiB for weights before activation overhead, "
            f"but the {accelerator_type.upper()} worker exposes about {available_gib:.1f} GiB. "
            "Use a smaller or quantized checkpoint, or connect a worker with more memory."
        )
    return {"estimatedWeightBytes": estimated_bytes or None, "memory": memory}


def _load_model(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = str(payload.get("modelId", "")).strip()
    revision = str(payload.get("revision", "main")).strip() or "main"
    dtype_name = str(payload.get("dtype", "auto"))
    quantization = str(payload.get("quantization", "none"))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?", model_id):
        raise ValueError("Enter a valid Hugging Face model ID")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", revision):
        raise ValueError("Enter a valid model revision")
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError("dtype must be auto, float16, bfloat16, or float32")
    if quantization not in {"none", "4bit"}:
        raise ValueError("quantization must be none or 4bit")

    accelerator_type, _accelerator_name = _available_accelerator(torch)
    preflight = _preflight_model_size(torch, payload, accelerator_type, quantization)
    token = os.environ.get("HF_TOKEN", "").strip() or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        trust_remote_code=False,
    )
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "token": token,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
        "device_map": "auto" if accelerator_type == "cuda" else None,
        "torch_dtype": dtype_map[dtype_name],
        "attn_implementation": "eager",
    }
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise ValueError("4-bit loading requires a CUDA GPU runtime")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if model_kwargs["device_map"] is None:
        del model_kwargs["device_map"]
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except (TypeError, ValueError) as error:
        if "attn" not in str(error).lower():
            raise
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    if accelerator_type == "mps":
        model.to("mps")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    device = _model_input_device(model)
    with state.lock:
        state.model = model
        state.tokenizer = tokenizer
        state.model_id = model_id
        state.revision = revision
        state.device = str(device)
        state.dtype = str(getattr(model, "dtype", dtype_name)).removeprefix("torch.")
        state.quantization = quantization
        state.loaded_at = time.time()
        state.latest_run_id = ""
        state.latest_tensors.clear()
        state.runs.clear()
        state.run_order.clear()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "ok": True,
        "modelId": model_id,
        "revision": revision,
        "device": str(device),
        "dtype": state.dtype,
        "quantization": quantization,
        "parameters": parameters,
        "preflight": preflight,
    }


def _direct_projection(tensor: Any, unembedding: Any) -> float | None:
    import torch

    vector = _position_vector(tensor).detach().float()
    row = unembedding.detach().to(device=vector.device, dtype=torch.float32)
    if vector.ndim != 1 or row.ndim != 1 or vector.numel() != row.numel():
        return None
    return float(torch.dot(vector, row).item())


def _vector_cosine(left: Any, right: Any) -> float | None:
    import torch

    left_vector = _position_vector(left).detach().float().flatten()
    right_vector = _position_vector(right).detach().to(device=left_vector.device, dtype=torch.float32).flatten()
    if left_vector.numel() != right_vector.numel() or left_vector.numel() == 0:
        return None
    left_norm = torch.linalg.vector_norm(left_vector)
    right_norm = torch.linalg.vector_norm(right_vector)
    if left_norm == 0 or right_norm == 0:
        return None
    return float(torch.dot(left_vector, right_vector).div(left_norm * right_norm).item())


def _device_memory(torch: Any) -> dict[str, Any]:
    if torch.cuda.is_available():
        return {
            "kind": "cuda",
            "allocatedBytes": int(torch.cuda.memory_allocated()),
            "reservedBytes": int(torch.cuda.memory_reserved()),
            "peakAllocatedBytes": int(torch.cuda.max_memory_allocated()),
        }
    return {"kind": "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"}


def _attention_summaries(attentions: Any, tokens: list[str], latest: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    result = []
    for layer, attention in enumerate(attentions or []):
        if not torch.is_tensor(attention) or attention.ndim != 4:
            continue
        stored = _cpu_tensor(attention)
        latest[f"attention_probs.{layer}"] = stored
        rows = attention[0, :, -1, :].detach().float()
        heads = []
        for head, row in enumerate(rows):
            probabilities = row.clamp_min(0)
            total = probabilities.sum()
            if total > 0:
                probabilities = probabilities / total
            entropy = float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())
            top_value, top_index = torch.max(probabilities, dim=-1)
            heads.append({
                "head": head,
                "entropy": entropy,
                "topSourceIndex": int(top_index.item()),
                "topSourceToken": tokens[int(top_index.item())] if int(top_index.item()) < len(tokens) else "",
                "topSourceWeight": float(top_value.item()),
                "lastQuery": [float(value) for value in probabilities.cpu().tolist()],
            })
        result.append({"layer": layer, "heads": heads})
    return result


def _cache_summary(cache: Any) -> dict[str, Any]:
    import torch

    if cache is None:
        return {"available": False, "tensors": [], "bytes": 0}
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    tensors: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if torch.is_tensor(value):
            tensors.append({
                "path": path,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "bytes": value.numel() * value.element_size(),
            })
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}" if path else str(index))

    visit(cache, "")
    return {"available": bool(tensors), "tensors": tensors, "bytes": sum(item["bytes"] for item in tensors)}


def _attribution_summary(layers: list[dict[str, Any]], target_logit: float) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    if layers:
        embedding = layers[0].get("residPre") or {}
        if isinstance(embedding.get("dla"), (int, float)):
            components.append({
                "id": "embedding",
                "nodeId": "residual_0",
                "label": "Token embedding / initial residual",
                "kind": "embedding",
                "layer": None,
                "dla": float(embedding["dla"]),
                "norm": embedding.get("norm"),
            })
    for layer in layers:
        layer_index = int(layer["layer"])
        for key, kind, label, node_id in (
            ("attentionWrite", "attention", "Attention write", f"l{layer_index}_output"),
            ("mlpWrite", "mlp", "MLP write", f"l{layer_index}_mlp"),
        ):
            capture = layer.get(key)
            if not isinstance(capture, dict) or not isinstance(capture.get("dla"), (int, float)):
                continue
            components.append({
                "id": f"{kind}.{layer_index}",
                "nodeId": node_id,
                "label": f"Layer {layer_index} {label}",
                "kind": kind,
                "layer": layer_index,
                "dla": float(capture["dla"]),
                "norm": capture.get("norm"),
            })
    absolute_mass = sum(abs(component["dla"]) for component in components)
    for component in components:
        component["shareOfAbsoluteMass"] = component["dla"] / absolute_mass if absolute_mass else None
    raw_sum = sum(component["dla"] for component in components)
    return {
        "method": "raw-unembedding-dot-product",
        "targetLogit": target_logit,
        "capturedRawSum": raw_sum,
        "normalizationAndBiasGap": target_logit - raw_sum,
        "positiveTotal": sum(max(0.0, component["dla"]) for component in components),
        "negativeTotal": sum(min(0.0, component["dla"]) for component in components),
        "absoluteMass": absolute_mass,
        "components": components,
        "note": (
            "Shares are each component's signed fraction of total absolute raw DLA. They compare "
            "captured support and suppression, not causal effect or a literal percentage of the output. "
            "The gap to the emitted target logit includes final normalization, output bias, and uncaptured paths."
        ),
    }


METRIC_KINDS = {
    "target_probability",
    "logit_difference",
    "sequence_loss",
    "kl_divergence",
    "multi_token_score",
    "custom_token_groups",
}


def _normalise_metric(payload: Any, fallback_target: str = "") -> dict[str, Any]:
    value = payload if isinstance(payload, dict) else {}
    kind = str(value.get("kind", "target_probability")).strip().lower().replace("-", "_")
    if kind not in METRIC_KINDS:
        raise ValueError(f"Unsupported behaviour metric: {kind}")
    labels = {
        "target_probability": "Target-token probability",
        "logit_difference": "Correct vs incorrect logit difference",
        "sequence_loss": "Sequence loss",
        "kl_divergence": "Output KL divergence",
        "multi_token_score": "Multi-token answer score",
        "custom_token_groups": "Custom token-group difference",
    }
    name = str(value.get("name", labels[kind])).strip()[:160] or labels[kind]
    target = str(value.get("targetToken", fallback_target)).strip()
    answer = str(value.get("answer", target)).strip()
    correct = value.get("correctTokens", [])
    incorrect = value.get("incorrectTokens", [])
    positive = value.get("positiveTokens", correct)
    negative = value.get("negativeTokens", incorrect)
    return {
        "kind": kind,
        "name": name,
        "targetToken": target,
        "answer": answer,
        "correctTokens": [str(item) for item in correct] if isinstance(correct, list) else [],
        "incorrectTokens": [str(item) for item in incorrect] if isinstance(incorrect, list) else [],
        "positiveTokens": [str(item) for item in positive] if isinstance(positive, list) else [],
        "negativeTokens": [str(item) for item in negative] if isinstance(negative, list) else [],
    }


def _metric_value_identity(specification: dict[str, Any]) -> tuple[Any, ...]:
    """Fields that determine the numerical value, excluding the display name."""
    return (
        specification.get("kind"),
        specification.get("targetToken"),
        specification.get("answer"),
        tuple(specification.get("correctTokens", [])),
        tuple(specification.get("incorrectTokens", [])),
        tuple(specification.get("positiveTokens", [])),
        tuple(specification.get("negativeTokens", [])),
    )


def _require_base_metric(record: Any, specification: dict[str, Any], operation: str) -> None:
    fallback = str(record.result.get("target", {}).get("text", record.request.get("targetToken", "")))
    base_specification = _normalise_metric(record.request.get("metric"), fallback)
    if _metric_value_identity(base_specification) != _metric_value_identity(specification):
        raise ValueError(
            f"{operation} metric differs from the stored base run. Run the selected example again "
            "after changing the behaviour metric so baseline and intervention values remain comparable."
        )


def _token_group_ids(tokenizer: Any, values: list[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        ids = tokenizer.encode(value, add_special_tokens=False)
        if ids:
            token_id = int(ids[-1])
            if token_id not in result:
                result.append(token_id)
    return result


def _metric_from_logits(
    logits: Any,
    tokenizer: Any,
    specification: dict[str, Any],
    *,
    fallback_target_id: int,
    continuation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    kind = specification["kind"]
    probabilities = torch.softmax(logits.detach().float(), dim=-1)
    direction = "maximize"
    value: float | None
    details: dict[str, Any] = {}
    if kind == "target_probability":
        ids = _token_group_ids(tokenizer, [specification["targetToken"]]) if specification["targetToken"] else []
        token_id = ids[-1] if ids else fallback_target_id
        value = float(probabilities[token_id].item())
        details = {"targetTokenId": token_id, "targetToken": tokenizer.decode([token_id])}
    elif kind == "logit_difference":
        correct_ids = _token_group_ids(tokenizer, specification["correctTokens"])
        incorrect_ids = _token_group_ids(tokenizer, specification["incorrectTokens"])
        if not correct_ids or not incorrect_ids:
            raise ValueError("Logit difference requires at least one correct and one incorrect token")
        correct_value = logits[correct_ids].detach().float().mean()
        incorrect_value = logits[incorrect_ids].detach().float().mean()
        value = float((correct_value - incorrect_value).item())
        details = {"correctTokenIds": correct_ids, "incorrectTokenIds": incorrect_ids}
    elif kind == "custom_token_groups":
        positive_ids = _token_group_ids(tokenizer, specification["positiveTokens"])
        negative_ids = _token_group_ids(tokenizer, specification["negativeTokens"])
        if not positive_ids or not negative_ids:
            raise ValueError("Custom token groups require at least one positive and one negative token")
        value = float((logits[positive_ids].detach().float().mean() - logits[negative_ids].detach().float().mean()).item())
        details = {"positiveTokenIds": positive_ids, "negativeTokenIds": negative_ids}
    elif kind in {"sequence_loss", "multi_token_score"}:
        if continuation is None:
            raise ValueError("This metric requires a non-empty answer sequence")
        if kind == "sequence_loss":
            value = float(continuation["loss"])
            direction = "minimize"
        else:
            value = float(continuation["averageLogProbability"])
        details = {key: item for key, item in continuation.items() if key != "loss"}
    else:
        value = None
        direction = "diverge"
        details = {"requiresComparison": True}
    return {
        "name": specification["name"],
        "kind": kind,
        "value": value,
        "direction": direction,
        "details": details,
    }


def _continuation_score(model: Any, tokenizer: Any, prompt_ids: list[int], answer: str, device: Any) -> dict[str, Any]:
    import torch

    answer_ids = [int(item) for item in tokenizer.encode(answer, add_special_tokens=False)]
    if not answer_ids:
        raise ValueError("Sequence metrics require a non-empty answer")
    if len(prompt_ids) + len(answer_ids) > MAX_PROMPT_TOKENS:
        raise ValueError("Prompt and answer exceed the worker token limit")
    full_ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(input_ids=full_ids, use_cache=False, return_dict=True)
    log_probabilities = torch.log_softmax(output.logits[0].detach().float(), dim=-1)
    token_log_probabilities = []
    for offset, token_id in enumerate(answer_ids):
        prediction_position = len(prompt_ids) + offset - 1
        token_log_probabilities.append(float(log_probabilities[prediction_position, token_id].item()))
    average = sum(token_log_probabilities) / len(token_log_probabilities)
    return {
        "answer": answer,
        "answerTokenIds": answer_ids,
        "answerTokens": tokenizer.convert_ids_to_tokens(answer_ids),
        "tokenLogProbabilities": token_log_probabilities,
        "averageLogProbability": average,
        "totalLogProbability": sum(token_log_probabilities),
        "loss": -average,
    }


def _store_run(
    state: WorkerState,
    run_id: str,
    request: dict[str, Any],
    result: dict[str, Any],
    tensors: dict[str, Any],
    logits: Any,
) -> None:
    record = RunRecord(run_id, request, result, tensors, logits, time.time())
    with state.lock:
        state.latest_run_id = run_id
        state.latest_tensors = tensors
        state.runs[run_id] = record
        if run_id in state.run_order:
            state.run_order.remove(run_id)
        state.run_order.append(run_id)
        while len(state.run_order) > MAX_STORED_RUNS:
            expired = state.run_order.pop(0)
            state.runs.pop(expired, None)


def _run_record(state: WorkerState, run_id: str) -> RunRecord:
    with state.lock:
        record = state.runs.get(run_id)
    if record is None:
        raise ValueError("That execution run is no longer available in this worker")
    return record


def _runtime_diagnostics(
    layers: list[dict[str, Any]],
    attention: list[dict[str, Any]],
    cache: dict[str, Any],
    probabilities: Any,
    *,
    token_count: int,
    hook_order: list[dict[str, Any]],
) -> dict[str, Any]:
    anomalies: list[dict[str, Any]] = []
    previous_norm: float | None = None
    for layer in layers:
        layer_index = int(layer["layer"])
        for key, label in (("residPre", "Residual input"), ("residPost", "Residual output"), ("attentionWrite", "Attention write"), ("mlpWrite", "MLP write")):
            summary = layer.get(key)
            if not isinstance(summary, dict):
                continue
            non_finite = int(summary.get("nanCount", 0)) + int(summary.get("positiveInfinityCount", 0)) + int(summary.get("negativeInfinityCount", 0))
            if non_finite:
                anomalies.append({"severity": "error", "kind": "non-finite", "layer": layer_index, "component": key, "message": f"{label} contains {non_finite} NaN or infinite values."})
        norm = layer.get("residPost", {}).get("norm")
        if isinstance(norm, (int, float)) and isinstance(previous_norm, (int, float)) and previous_norm > 0:
            ratio = norm / previous_norm
            if ratio > 8:
                anomalies.append({"severity": "warning", "kind": "exploding-residual", "layer": layer_index, "value": ratio, "message": f"Residual norm increased {ratio:.1f}× at layer {layer_index}."})
            elif ratio < 0.125:
                anomalies.append({"severity": "warning", "kind": "collapsing-residual", "layer": layer_index, "value": ratio, "message": f"Residual norm fell to {ratio:.2f}× at layer {layer_index}."})
        if isinstance(norm, (int, float)):
            previous_norm = norm
    maximum_probability = float(probabilities.max().item())
    if maximum_probability > 0.9999:
        anomalies.append({"severity": "warning", "kind": "saturated-output", "value": maximum_probability, "message": "The output distribution is extremely saturated."})
    maximum_entropy = math.log(max(2, token_count))
    for layer in attention:
        for head in layer.get("heads", []):
            entropy = head.get("entropy")
            if isinstance(entropy, (int, float)) and entropy < min(0.03, maximum_entropy * 0.01):
                anomalies.append({"severity": "notice", "kind": "sharp-attention", "layer": layer["layer"], "head": head["head"], "value": entropy, "message": f"Layer {layer['layer']} head {head['head']} attends almost entirely to one source token."})
    cache_issues = []
    for tensor in cache.get("tensors", []):
        shape = tensor.get("shape", [])
        if len(shape) >= 3 and token_count not in shape:
            cache_issues.append(tensor.get("path", "cache tensor"))
    if cache.get("available") and cache_issues:
        anomalies.append({"severity": "warning", "kind": "cache-shape", "message": f"{len(cache_issues)} cache tensors do not expose the expected sequence length.", "paths": cache_issues[:12]})
    captured_categories = {str(item.get("category")) for item in hook_order}
    if not layers:
        anomalies.append({"severity": "error", "kind": "hook-failure", "message": "No decoder-layer residual states were captured; this architecture is not compatible with the current hook adapter."})
    for category in ("attention", "mlp"):
        captured_layers = {int(item["layer"]) for item in hook_order if item.get("category") == category and isinstance(item.get("layer"), int)}
        if layers and len(captured_layers) < len(layers):
            anomalies.append({
                "severity": "notice",
                "kind": "partial-hook-coverage",
                "component": category,
                "message": f"Captured {category} outputs for {len(captured_layers)} of {len(layers)} decoder layers.",
            })
    durations = [float(item.get("durationMs", 0.0)) for item in hook_order if float(item.get("durationMs", 0.0)) > 0]
    if durations:
        ordered_durations = sorted(durations)
        median_duration = ordered_durations[len(ordered_durations) // 2]
        if median_duration > 0:
            for item in hook_order:
                duration = float(item.get("durationMs", 0.0))
                if duration > max(5.0, median_duration * 4):
                    anomalies.append({
                        "severity": "notice",
                        "kind": "slow-component",
                        "layer": item.get("layer"),
                        "component": item.get("category"),
                        "value": duration,
                        "message": f"{item.get('category', 'Component').title()} hook took {duration:.1f} ms, more than 4× the median capture time.",
                    })
    layer_count = len(layers)
    capabilities = {
        "residualStreams": layer_count > 0,
        "attentionOutputs": "attention" in captured_categories,
        "attentionHeadOutputs": any(isinstance(layer.get("attentionHeadOutputs"), dict) for layer in layers),
        "mlpOutputs": "mlp" in captured_categories,
        "attentionProbabilities": bool(attention),
        "kvCache": bool(cache.get("available")),
        "activationSlices": bool(layers or attention),
        "interventions": bool({"residual", "attention", "mlp"} & captured_categories),
        "rootCauseTrace": bool({"attention", "mlp"} & captured_categories),
    }
    unsupported = [label for key, label in (
        ("attentionOutputs", "attention-output hooks"),
        ("attentionHeadOutputs", "pre-projection attention-head outputs"),
        ("mlpOutputs", "MLP-output hooks"),
        ("attentionProbabilities", "attention probabilities"),
        ("kvCache", "KV-cache inspection"),
    ) if not capabilities[key]]
    return {
        "anomalies": anomalies,
        "capabilities": capabilities,
        "unsupported": unsupported,
        "layerTimings": [
            {
                "layer": item.get("layer"),
                "category": item.get("category"),
                "hookName": item.get("hookName"),
                "durationMs": item.get("durationMs"),
            }
            for item in hook_order
        ],
    }


def _replace_first_tensor(value: Any, replacement: Any) -> Any:
    try:
        import torch
    except ImportError:
        return value
    if torch.is_tensor(value):
        return replacement
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            if _first_tensor(item) is not None:
                items[index] = _replace_first_tensor(item, replacement)
                return tuple(items)
    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            if _first_tensor(item) is not None:
                items[index] = _replace_first_tensor(item, replacement)
                return items
    return value


def _aligned_source_tensor(source: Any, target: Any) -> Any:
    import torch

    source = source.detach().to(device=target.device, dtype=target.dtype)
    if source.shape == target.shape:
        return source
    if source.ndim != target.ndim or source.shape[:-2] != target.shape[:-2] or source.shape[-1] != target.shape[-1]:
        raise ValueError("Source and target activations are not shape-compatible")
    aligned = target.detach().clone()
    count = min(source.shape[-2], target.shape[-2])
    aligned[..., -count:, :] = source[..., -count:, :]
    return aligned


def _intervened_tensor(tensor: Any, specification: dict[str, Any], source: Any | None) -> Any:
    import torch

    method = str(specification.get("method", "zero"))
    position = int(specification.get("position", -1))
    scope = str(specification.get("scope", "position"))
    if method == "zero":
        replacement = torch.zeros_like(tensor)
    elif method == "mean":
        if tensor.ndim < 2:
            replacement = torch.full_like(tensor, tensor.detach().float().mean().to(dtype=tensor.dtype))
        else:
            replacement = tensor.detach().mean(dim=-2, keepdim=True).expand_as(tensor)
    elif method in {"resample", "patch"}:
        if source is None:
            raise ValueError(f"{method.title()} intervention requires a source run")
        replacement = _aligned_source_tensor(source, tensor)
        if method == "resample" and replacement.ndim >= 3 and replacement.shape[-2] > 1:
            permutation = torch.randperm(replacement.shape[-2], device=replacement.device)
            replacement = replacement.index_select(-2, permutation)
    elif method == "scale":
        scale = float(specification.get("scale", 0.0))
        if not math.isfinite(scale) or abs(scale) > 100:
            raise ValueError("Activation scale must be a finite value between -100 and 100")
        replacement = tensor * scale
    elif method == "steer":
        scale = float(specification.get("scale", 1.0))
        direction = specification.get("direction")
        if isinstance(direction, list) and direction:
            direction_tensor = torch.tensor(direction, device=tensor.device, dtype=tensor.dtype)
            if direction_tensor.numel() != tensor.shape[-1]:
                raise ValueError("Steering direction width does not match this component")
            replacement = tensor + scale * direction_tensor
        elif source is not None:
            source_tensor = _aligned_source_tensor(source, tensor)
            replacement = tensor + scale * (source_tensor - tensor.detach())
        else:
            raise ValueError("Steering requires a direction vector or source run")
    else:
        raise ValueError(f"Unsupported intervention method: {method}")
    if scope == "all" or tensor.ndim < 2:
        return replacement
    result = tensor.clone()
    if tensor.ndim >= 3:
        result[:, position] = replacement[:, position]
    else:
        result[position] = replacement[position]
    return result


def _normalise_interventions(payload: Any) -> list[dict[str, Any]]:
    values = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
    result = []
    for value in values:
        kind = str(value.get("kind", "")).strip()
        layer = value.get("layer")
        if kind not in {"residual", "attention", "mlp"} or not isinstance(layer, int) or layer < 0:
            raise ValueError("Each intervention requires a residual, attention, or MLP component and layer")
        result.append({
            **value,
            "kind": kind,
            "layer": layer,
            "method": str(value.get("method", "zero")).strip().lower(),
            "position": int(value.get("position", -1)),
            "scope": str(value.get("scope", "position")).strip().lower(),
        })
    return result


def _prepare_prompt(tokenizer: Any, payload: dict[str, Any]) -> tuple[str, str]:
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        clean_messages = []
        for message in messages:
            if not isinstance(message, dict) or str(message.get("role", "")) not in {"system", "user", "assistant", "tool"}:
                raise ValueError("Chat messages require a supported role and content")
            clean_messages.append({"role": str(message["role"]), "content": str(message.get("content", ""))})
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("This tokenizer does not define a chat template")
        prompt = tokenizer.apply_chat_template(
            clean_messages,
            tokenize=False,
            add_generation_prompt=bool(payload.get("addGenerationPrompt", True)),
        )
        return prompt, "tokenizer-chat-template"
    return str(payload.get("prompt", "")), "raw-text"


def _normalise_generation_settings(payload: Any) -> dict[str, Any]:
    value = payload if isinstance(payload, dict) else {}
    maximum_tokens = int(value.get("maxNewTokens", 8))
    if maximum_tokens < 1 or maximum_tokens > MAX_GENERATED_TOKENS:
        raise ValueError(f"Generation length must be between 1 and {MAX_GENERATED_TOKENS} tokens")
    do_sample = bool(value.get("doSample", False))
    temperature = float(value.get("temperature", 1.0))
    top_p = float(value.get("topP", 1.0))
    top_k = int(value.get("topK", 0))
    if do_sample and (not math.isfinite(temperature) or temperature <= 0 or temperature > 10):
        raise ValueError("Sampling temperature must be greater than 0 and at most 10")
    if not math.isfinite(top_p) or top_p <= 0 or top_p > 1:
        raise ValueError("Top-p must be greater than 0 and at most 1")
    if top_k < 0 or top_k > 500:
        raise ValueError("Sampling top-k must be between 0 and 500")
    return {
        "maxNewTokens": maximum_tokens,
        "doSample": do_sample,
        "temperature": temperature if do_sample else 0.0,
        "topP": top_p,
        "topK": top_k,
    }


def _normalise_lens_settings(payload: Any) -> dict[str, Any]:
    value = payload if isinstance(payload, dict) else {}
    enabled = bool(value.get("enabled", False))
    maximum_stages = int(value.get("maxStages", 24))
    if maximum_stages < 2 or maximum_stages > MAX_LOGIT_LENS_STAGES:
        raise ValueError(f"Logit-lens stages must be between 2 and {MAX_LOGIT_LENS_STAGES}")
    top_k = int(value.get("topK", 3))
    if top_k < 1 or top_k > 10:
        raise ValueError("Logit-lens top-k must be between 1 and 10")
    return {"enabled": enabled, "maxStages": maximum_stages, "topK": top_k}


def _module_at_path(model: Any, path: str) -> Any | None:
    value = model
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _is_normalization_candidate(module: Any) -> bool:
    """Identify normalization-shaped modules by structure, never architecture names."""
    try:
        parameters = list(module.parameters(recurse=False))
        children = list(module.children())
    except (AttributeError, TypeError):
        return False
    if children or not parameters:
        return False
    weight = getattr(module, "weight", None)
    return (
        getattr(weight, "ndim", 0) == 1
        and all(getattr(parameter, "ndim", 2) <= 1 for parameter in parameters)
    )


def _normalization_trace_hook(name: str, calls: dict[str, tuple[Any, Any, Any]]):
    def capture(module: Any, inputs: Any, output: Any) -> None:
        before = _first_tensor(inputs)
        after = _first_tensor(output)
        if before is None or after is None or before.shape != after.shape or before.ndim < 2:
            return
        calls[name] = (module, before.detach(), after.detach())
    return capture


def _register_normalization_trace(model: Any, calls: dict[str, tuple[Any, Any, Any]]) -> list[Any]:
    handles = []
    for name, module in model.named_modules():
        if name and _is_normalization_candidate(module):
            handles.append(module.register_forward_hook(_normalization_trace_hook(name, calls)))
    return handles


def _verified_final_normalization(
    calls: dict[str, tuple[Any, Any, Any]],
    final_hidden_state: Any,
) -> tuple[Any | None, str | None]:
    """Find the actual final norm by matching this run's output, not module paths."""
    import torch

    if final_hidden_state is None:
        return None, None
    reference = _position_vector(final_hidden_state).detach().float()
    matches = []
    with torch.inference_mode():
        for name, (module, _before, after) in calls.items():
            candidate = _position_vector(after).detach().float()
            if candidate.shape != reference.shape:
                continue
            maximum_error = float(torch.max(torch.abs(candidate - reference)).item())
            scale = max(1.0, float(torch.max(torch.abs(reference)).item()))
            if maximum_error <= 1e-5 * scale:
                matches.append((maximum_error, -name.count("."), name, module))
    if not matches:
        return None, None
    _error, _negative_depth, name, module = min(matches, key=lambda item: item[:3])
    return module, name


def _output_projection(model: Any) -> tuple[Any | None, str | None]:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        projection = getter()
        if projection is not None:
            return projection, "get_output_embeddings()"
    return None, None


def _project_output_logits(projection: Any, vector: Any, expected_vocabulary: int) -> Any:
    """Apply a model's output head without assuming it is ``torch.nn.Linear``."""
    import torch
    import torch.nn.functional as functional

    call_error: Exception | None = None
    if callable(projection):
        try:
            projected = _first_tensor(projection(vector))
            if projected is not None and projected.numel() == expected_vocabulary:
                return projected
        except (RuntimeError, ValueError, TypeError, NotImplementedError) as error:
            call_error = error

    # Some tied-weight models expose an Embedding rather than a Linear output
    # module. Its [vocab, hidden] weight is still the exact unembedding matrix.
    weight = getattr(projection, "weight", None)
    if torch.is_tensor(weight) and weight.ndim == 2 and vector.shape[-1] == weight.shape[1]:
        bias = getattr(projection, "bias", None)
        if bias is not None and (not torch.is_tensor(bias) or bias.numel() != weight.shape[0]):
            bias = None
        projected = functional.linear(vector, weight, bias)
        if projected.numel() == expected_vocabulary:
            return projected
    detail = f": {call_error}" if call_error is not None else ""
    raise ValueError(
        f"The output projection did not produce the model's {expected_vocabulary}-token vocabulary{detail}"
    )


def _sample_stage_indices(total: int, maximum: int) -> list[int]:
    if total <= maximum:
        return list(range(total))
    if maximum <= 1:
        return [total - 1]
    values = {round(index * (total - 1) / (maximum - 1)) for index in range(maximum)}
    values.update({0, total - 1})
    return sorted(int(index) for index in values)


def _prediction_summary(logits: Any, tokenizer: Any, target_id: int, top_k: int) -> dict[str, Any]:
    import torch

    vector = logits.detach().float().flatten()
    probabilities = torch.softmax(vector, dim=-1)
    log_probabilities = torch.log_softmax(vector, dim=-1)
    count = min(top_k, probabilities.numel())
    top_probabilities, top_ids = torch.topk(probabilities, k=count)
    predictions = []
    for probability, token_id in zip(top_probabilities.tolist(), top_ids.tolist()):
        token_id = int(token_id)
        predictions.append({
            "tokenId": token_id,
            "token": tokenizer.convert_ids_to_tokens([token_id])[0],
            "text": tokenizer.decode([token_id]),
            "probability": float(probability),
            "logit": float(vector[token_id].item()),
        })
    entropy = float((-(probabilities * log_probabilities)).sum().item())
    target_probability = float(probabilities[target_id].item())
    target_rank = int(torch.count_nonzero(vector > vector[target_id]).item()) + 1
    return {
        "topK": predictions,
        "entropy": entropy,
        "targetProbability": target_probability,
        "targetLogit": float(vector[target_id].item()),
        "targetRank": target_rank,
    }


def _logit_lens_timeline(
    model: Any,
    tokenizer: Any,
    hidden_states: Any,
    final_logits: Any,
    target_id: int,
    settings: dict[str, Any],
    verified_normalization: tuple[Any | None, str | None] = (None, None),
) -> dict[str, Any]:
    import torch

    states = list(hidden_states or [])
    if not states:
        return {
            "available": False,
            "stages": [],
            "evidence": {"kind": "observational", "causal": False, "note": "The model did not return hidden states."},
        }
    normalization, normalization_path = verified_normalization
    if normalization is None:
        return {
            "available": False,
            "stages": [],
            "method": "unavailable",
            "evidence": {
                "kind": "observational",
                "causal": False,
                "note": (
                    "The model's final residual normalization could not be verified from this forward pass. "
                    "The logit lens was omitted instead of applying architecture-specific assumptions."
                ),
            },
        }
    output_projection, projection_path = _output_projection(model)
    if output_projection is None:
        return {
            "available": False,
            "stages": [],
            "evidence": {"kind": "observational", "causal": False, "note": "The model does not expose a usable output projection."},
        }
    selected_indices = _sample_stage_indices(len(states), settings["maxStages"])
    try:
        # Hidden states returned by a forward pass in inference_mode are
        # inference tensors. Every subsequent norm and unembedding operation
        # must remain in inference mode as well; otherwise recent PyTorch builds
        # reject the projection while trying to save those tensors for autograd.
        with torch.inference_mode():
            final_vector = final_logits.detach().float().flatten()
            expected_vocabulary = int(final_vector.numel())
            final_log_probabilities = torch.log_softmax(final_vector, dim=-1)
            final_probabilities = final_log_probabilities.exp()
            stages = []
            last_index = len(states) - 1
            for hidden_index in selected_indices:
                if hidden_index == last_index:
                    lens_logits = final_vector
                    stage_kind = "output"
                    label = "Model output"
                    layer = max(0, last_index - 1)
                else:
                    vector = states[hidden_index][:, -1, :]
                    if normalization is not None:
                        vector = normalization(vector)
                    lens_logits = _project_output_logits(
                        output_projection,
                        vector,
                        expected_vocabulary,
                    ).detach().float().flatten()
                    stage_kind = "embedding" if hidden_index == 0 else "layer"
                    layer = None if hidden_index == 0 else hidden_index - 1
                    label = "Embedding" if hidden_index == 0 else f"Layer {layer}"
                summary = _prediction_summary(lens_logits, tokenizer, target_id, settings["topK"])
                lens_log_probabilities = torch.log_softmax(lens_logits.detach().float().flatten(), dim=-1)
                final_to_lens_kl = float(torch.sum(final_probabilities * (final_log_probabilities - lens_log_probabilities)).item())
                stages.append({
                    "index": hidden_index,
                    "kind": stage_kind,
                    "layer": layer,
                    "label": label,
                    "finalToLensKL": final_to_lens_kl,
                    **summary,
                })
    except Exception as error:
        return {
            "available": False,
            "stages": [],
            "method": "unavailable",
            "normalizationModule": normalization_path,
            "outputProjection": projection_path,
            "evidence": {
                "kind": "observational",
                "causal": False,
                "note": f"Intermediate residual states could not be projected through the model's output head: {error}",
            },
        }
    return {
        "available": True,
        "method": "normalized-logit-lens",
        "normalizationModule": normalization_path,
        "outputProjection": projection_path,
        "target": {
            "tokenId": target_id,
            "token": tokenizer.convert_ids_to_tokens([target_id])[0],
            "text": tokenizer.decode([target_id]),
        },
        "sampled": len(selected_indices) < len(states),
        "totalStages": len(states),
        "stages": stages,
        "evidence": {
            "kind": "observational",
            "causal": False,
            "note": (
                f"The final residual normalization was verified as {normalization_path} from this run, then applied "
                "before the model's public output projection. Lens predictions decode residual states; they do not "
                "show that a layer caused the final token."
            ),
            "klDirection": "KL(final output distribution || lens distribution)",
        },
    }


def _sample_next_token(logits: Any, settings: dict[str, Any]) -> int:
    import torch

    vector = logits.detach().float().flatten()
    if not settings["doSample"]:
        return int(torch.argmax(vector).item())
    filtered = vector / settings["temperature"]
    if settings["topK"]:
        count = min(settings["topK"], filtered.numel())
        threshold = torch.topk(filtered, k=count).values[-1]
        filtered = torch.where(filtered < threshold, torch.full_like(filtered, -torch.inf), filtered)
    if settings["topP"] < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative > settings["topP"]
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered = filtered.clone()
        filtered[sorted_indices[remove]] = -torch.inf
    probabilities = torch.softmax(filtered, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def _forward(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers

    waterfall = InferenceWaterfall()
    with state.lock:
        model = state.model
        tokenizer = state.tokenizer
        model_id = state.model_id
        revision = state.revision
        device_name = state.device
        dtype_name = state.dtype
    if model is None or tokenizer is None:
        raise ValueError("Load a model into the execution worker first")
    prompt, chat_template_source = _prepare_prompt(tokenizer, payload)
    if not prompt:
        raise ValueError("Enter a prompt for the hooked forward pass")
    seed = max(0, min(2**31 - 1, int(payload.get("seed", 0))))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    top_k = max(1, min(50, int(payload.get("topK", 10))))
    add_special_tokens = bool(payload.get("addSpecialTokens", True))
    waterfall.finish(
        "request-preparation",
        "Request preparation",
        "cpu",
        "Prompt/chat-template preparation, validation, random seeds, and run options.",
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    token_count = int(encoded["input_ids"].shape[-1])
    if token_count > MAX_PROMPT_TOKENS:
        raise ValueError(f"Prompt is {token_count} tokens; the worker limit is {MAX_PROMPT_TOKENS}")
    waterfall.finish(
        "tokenization",
        "Tokenization",
        "cpu",
        "Tokenizer encoding and prompt-length validation.",
    )
    input_device = _model_input_device(model)
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    token_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    interventions = _normalise_interventions(payload.get("interventions", []))
    internal_sources = payload.get("_sourceRecords")
    source_records: dict[str, RunRecord] = {}
    for intervention in interventions:
        source_id = str(intervention.get("sourceRunId", payload.get("sourceRunId", "")))
        if source_id:
            source = internal_sources.get(source_id) if isinstance(internal_sources, dict) else None
            source_records[source_id] = source if isinstance(source, RunRecord) else _run_record(state, source_id)
    _synchronize_accelerator(torch, input_device)
    waterfall.finish(
        "input-staging",
        "Input staging",
        "transfer",
        "Tensor transfer to the model input device, token labels, and intervention-source lookup.",
    )

    captures: dict[tuple[int, str], tuple[str, Any]] = {}
    attention_head_captures: dict[int, tuple[str, Any]] = {}
    hook_order: list[dict[str, Any]] = []
    normalization_calls: dict[str, tuple[Any, Any, Any]] = {}
    handles = _register_normalization_trace(model, normalization_calls)
    hook_started: dict[tuple[str, int, str], float] = {}

    def pre_hook_for(name: str, layer: int, category: str):
        def started(_module: Any, _inputs: Any) -> None:
            hook_started[(name, layer, category)] = time.perf_counter()
        return started

    def hook_for(name: str, layer: int, category: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> Any:
            modified = output
            for intervention in interventions:
                if intervention["layer"] != layer or intervention["kind"] != category:
                    continue
                tensor = _first_tensor(modified)
                if tensor is None:
                    continue
                source_id = str(intervention.get("sourceRunId", payload.get("sourceRunId", "")))
                source = None
                if source_id:
                    source_name = (
                        f"resid_post.{layer}" if category == "residual"
                        else f"{category}_write.{layer}"
                    )
                    source = source_records[source_id].tensors.get(source_name)
                    if source is None:
                        raise ValueError(f"Source run did not capture {source_name}")
                modified = _replace_first_tensor(modified, _intervened_tensor(tensor, intervention, source))
            tensor = _first_tensor(modified)
            if tensor is None:
                return modified
            stored = _cpu_tensor(tensor)
            captures[(layer, category)] = (name, stored)
            duration = (time.perf_counter() - hook_started.get((name, layer, category), time.perf_counter())) * 1000
            hook_order.append({
                "order": len(hook_order),
                "layer": layer,
                "category": category,
                "durationMs": max(0.0, duration),
                "activationBytes": int(tensor.numel() * tensor.element_size()),
                "deviceAllocatedBytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None,
                **_tensor_summary(tensor, name),
            })
            return modified
        return capture

    def attention_head_pre_hook(name: str, layer: int):
        def capture(_module: Any, inputs: Any) -> None:
            tensor = _first_tensor(inputs)
            if tensor is None or tensor.ndim < 3:
                return
            attention_head_captures[layer] = (name, _cpu_tensor(tensor))
        return capture

    for name, module in model.named_modules():
        layer = _layer_index(name)
        category = _module_category(name)
        if layer is not None and _is_attention_output_projection(name):
            handles.append(module.register_forward_pre_hook(attention_head_pre_hook(name, layer)))
        if layer is not None and category is not None:
            handles.append(module.register_forward_pre_hook(pre_hook_for(name, layer, category)))
            handles.append(module.register_forward_hook(hook_for(name, layer, category)))
    _synchronize_accelerator(torch, input_device)
    waterfall.finish(
        "instrumentation",
        "Instrumentation setup",
        "instrumentation",
        "Runtime discovery and registration of normalization, residual, attention, MLP, and head hooks.",
    )
    try:
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    _synchronize_accelerator(torch, input_device)
    waterfall.finish(
        "model-forward",
        "Instrumented model forward",
        "model",
        "Causal-LM forward pass with hidden states, attention, KV cache, and activation capture enabled.",
    )

    logits = outputs.logits[0, -1].detach().float()
    metric_specification = _normalise_metric(payload.get("metric"), str(payload.get("targetToken", "")))
    requested_target = str(payload.get("targetToken", metric_specification.get("targetToken", ""))).strip()
    if requested_target:
        target_ids = tokenizer.encode(requested_target, add_special_tokens=False)
        if not target_ids:
            raise ValueError("The requested target token did not encode to a token ID")
        target_id = int(target_ids[-1])
        target_source = "requested-last-subtoken" if len(target_ids) > 1 else "requested"
    else:
        target_id = int(torch.argmax(logits).item())
        target_source = "top-prediction"
    if target_id < 0 or target_id >= logits.numel():
        raise ValueError(
            f"The selected tokenizer ID {target_id} is outside the model output vocabulary of {logits.numel()} tokens"
        )
    target_token = tokenizer.convert_ids_to_tokens([target_id])[0]
    output_projection, _output_projection_path = _output_projection(model)
    output_weight = getattr(output_projection, "weight", None)
    unembedding = (
        output_weight[target_id]
        if torch.is_tensor(output_weight) and output_weight.ndim == 2 and target_id < output_weight.shape[0]
        else None
    )

    probabilities = torch.softmax(logits, dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
    top_predictions = []
    for probability, token_id in zip(top_probabilities.tolist(), top_ids.tolist()):
        token_id = int(token_id)
        top_predictions.append({
            "tokenId": token_id,
            "token": tokenizer.convert_ids_to_tokens([token_id])[0],
            "text": tokenizer.decode([token_id]),
            "probability": float(probability),
            "logit": float(logits[token_id].item()),
        })
    entropy = float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())
    target_probability = float(probabilities[target_id].item())
    target_logit = float(logits[target_id].item())
    target_rank = int(torch.count_nonzero(logits > logits[target_id]).item()) + 1
    logit_margin = float(top_predictions[0]["logit"] - top_predictions[1]["logit"]) if len(top_predictions) > 1 else None

    hidden_states = list(getattr(outputs, "hidden_states", None) or [])
    output_attentions = getattr(outputs, "attentions", None)
    verified_normalization = _verified_final_normalization(
        normalization_calls,
        hidden_states[-1] if hidden_states else None,
    )
    _synchronize_accelerator(torch, input_device)
    waterfall.finish(
        "output-scoring",
        "Output scoring",
        "analysis",
        "Target resolution, top-token probabilities, entropy, rank, and final-normalization verification.",
    )
    latest: dict[str, Any] = {}
    layers = []
    layer_count = max(0, len(hidden_states) - 1)
    final_hidden = _cpu_tensor(hidden_states[-1]) if hidden_states else None
    for layer in range(layer_count):
        pre = _cpu_tensor(hidden_states[layer])
        post = _cpu_tensor(hidden_states[layer + 1])
        latest[f"resid_pre.{layer}"] = pre
        latest[f"resid_post.{layer}"] = post
        attention_capture = captures.get((layer, "attention"))
        mlp_capture = captures.get((layer, "mlp"))
        layer_result: dict[str, Any] = {
            "layer": layer,
            "residPre": {
                **_tensor_summary(pre, f"resid_pre.{layer}"),
                "dla": _direct_projection(pre, unembedding) if unembedding is not None else None,
            },
            "residPost": {
                **_tensor_summary(post, f"resid_post.{layer}"),
                "dla": _direct_projection(post, unembedding) if unembedding is not None else None,
                "cosineToFinal": _vector_cosine(post, final_hidden) if final_hidden is not None else None,
            },
            "residualDelta": _tensor_summary(post.float() - pre.float(), f"residual_delta.{layer}"),
            "attentionWrite": None,
            "mlpWrite": None,
        }
        if attention_capture:
            name, tensor = attention_capture
            hook_name = f"attention_write.{layer}"
            latest[hook_name] = tensor
            layer_result["attentionWrite"] = {
                **_tensor_summary(tensor, name),
                "activationName": hook_name,
                "dla": _direct_projection(tensor, unembedding) if unembedding is not None else None,
            }
        attention_head_capture = attention_head_captures.get(layer)
        layer_result["attentionHeadOutputs"] = None
        if attention_head_capture:
            name, tensor = attention_head_capture
            attention_tensor = output_attentions[layer] if output_attentions and layer < len(output_attentions) else None
            head_count = int(attention_tensor.shape[1]) if torch.is_tensor(attention_tensor) and attention_tensor.ndim == 4 else 0
            if head_count and tensor.shape[-1] % head_count == 0:
                reshaped = tensor.reshape(*tensor.shape[:-1], head_count, tensor.shape[-1] // head_count)
                hook_name = f"attention_head_outputs.{layer}"
                latest[hook_name] = reshaped
                head_vectors = reshaped[0, -1].detach().float()
                layer_result["attentionHeadOutputs"] = {
                    **_tensor_summary(reshaped, name),
                    "activationName": hook_name,
                    "headCount": head_count,
                    "heads": [{"head": head, "norm": float(torch.linalg.vector_norm(vector).item())} for head, vector in enumerate(head_vectors)],
                }
        if mlp_capture:
            name, tensor = mlp_capture
            hook_name = f"mlp_write.{layer}"
            latest[hook_name] = tensor
            layer_result["mlpWrite"] = {
                **_tensor_summary(tensor, name),
                "activationName": hook_name,
                "dla": _direct_projection(tensor, unembedding) if unembedding is not None else None,
            }
        layers.append(layer_result)

    attention = _attention_summaries(output_attentions, tokens, latest)
    attribution = _attribution_summary(layers, target_logit)
    cache = _cache_summary(getattr(outputs, "past_key_values", None))
    _synchronize_accelerator(torch, input_device)
    waterfall.finish(
        "activation-analysis",
        "Activation analysis",
        "analysis",
        "Residual/component summaries, attention statistics, direct projections, and KV-cache inventory.",
    )
    continuation = None
    if metric_specification["kind"] in {"sequence_loss", "multi_token_score"}:
        continuation_handles = []

        def continuation_hook(layer: int, category: str):
            def modify(_module: Any, _inputs: Any, output: Any) -> Any:
                changed = output
                for intervention in interventions:
                    if intervention["layer"] != layer or intervention["kind"] != category:
                        continue
                    tensor = _first_tensor(changed)
                    if tensor is None:
                        continue
                    source_id = str(intervention.get("sourceRunId", payload.get("sourceRunId", "")))
                    source = None
                    if source_id:
                        source_name = f"resid_post.{layer}" if category == "residual" else f"{category}_write.{layer}"
                        source = source_records[source_id].tensors.get(source_name)
                    changed = _replace_first_tensor(changed, _intervened_tensor(tensor, intervention, source))
                return changed
            return modify

        if interventions:
            for name, module in model.named_modules():
                layer = _layer_index(name)
                category = _module_category(name)
                if layer is not None and category is not None:
                    continuation_handles.append(module.register_forward_hook(continuation_hook(layer, category)))
        try:
            continuation = _continuation_score(model, tokenizer, token_ids, metric_specification["answer"], input_device)
        finally:
            for handle in continuation_handles:
                handle.remove()
        _synchronize_accelerator(torch, input_device)
        waterfall.finish(
            "continuation-metric",
            "Continuation metric",
            "model",
            "Additional continuation forward pass required by the selected sequence-level metric.",
        )
    metric = _metric_from_logits(
        logits,
        tokenizer,
        metric_specification,
        fallback_target_id=target_id,
        continuation=continuation,
    )
    diagnostics = _runtime_diagnostics(
        layers,
        attention,
        cache,
        probabilities,
        token_count=token_count,
        hook_order=hook_order,
    )
    _synchronize_accelerator(torch, input_device)
    waterfall.finish(
        "metric-diagnostics",
        "Metric and diagnostics",
        "analysis",
        "Behaviour-metric evaluation, numerical checks, hook coverage, and runtime capability detection.",
    )
    lens_settings = _normalise_lens_settings(payload.get("logitLens"))
    logit_lens = (
        _logit_lens_timeline(
            model,
            tokenizer,
            hidden_states,
            logits,
            target_id,
            lens_settings,
            verified_normalization,
        )
        if lens_settings["enabled"]
        else {"available": False, "stages": [], "disabled": True}
    )
    direct_attribution_available = any(
        isinstance(component.get("dla"), (int, float)) and math.isfinite(float(component["dla"]))
        for component in attribution.get("components", [])
    )
    diagnostics["capabilities"]["directLogitAttribution"] = direct_attribution_available
    diagnostics["capabilities"]["logitLens"] = bool(logit_lens.get("available"))
    root_cause_metric_supported = metric.get("kind") in {
        "target_probability",
        "logit_difference",
        "custom_token_groups",
    }
    diagnostics["capabilities"]["rootCauseTrace"] = bool(
        diagnostics["capabilities"].get("rootCauseTrace") and root_cause_metric_supported
    )
    for supported, label in (
        (direct_attribution_available, "direct logit attribution"),
        (bool(logit_lens.get("available")), "logit-lens projection"),
        (diagnostics["capabilities"]["rootCauseTrace"], "root-cause tracing for the selected metric"),
    ):
        if not supported and label not in diagnostics["unsupported"]:
            diagnostics["unsupported"].append(label)
    if lens_settings["enabled"]:
        _synchronize_accelerator(torch, input_device)
        waterfall.finish(
            "logit-lens" if logit_lens.get("available") else "logit-lens-capability",
            "Logit-lens analysis" if logit_lens.get("available") else "Lens capability check",
            "analysis",
            "Capability-verified residual normalization and vocabulary projection for sampled stages."
            if logit_lens.get("available")
            else "Checked public output projection and runtime-verified final normalization; no compatible lens was exposed.",
        )
    run_id = uuid.uuid4().hex
    evidence_kind = "causal" if interventions else "observational"
    performance: dict[str, Any] = {}
    result = {
        "ok": True,
        "runId": run_id,
        "modelId": model_id,
        "revision": revision,
        "prompt": prompt,
        "position": token_count - 1,
        "positionLabel": f"Token {token_count - 1} · {tokens[-1]}",
        "tokens": [{"index": index, "id": token_id, "token": token, "text": tokenizer.decode([token_id])} for index, (token_id, token) in enumerate(zip(token_ids, tokens))],
        "target": {"tokenId": target_id, "token": target_token, "text": tokenizer.decode([target_id]), "source": target_source, "probability": target_probability, "logit": target_logit, "rank": target_rank},
        "nextToken": {"entropy": entropy, "logitMargin": logit_margin, "topK": top_predictions},
        "metric": metric,
        "layers": layers,
        "attention": attention,
        "attribution": attribution,
        "logitLens": logit_lens,
        "hooks": hook_order,
        "kvCache": cache,
        "diagnostics": diagnostics,
        "interventions": interventions,
        "context": {
            "seed": seed,
            "device": device_name,
            "dtype": dtype_name,
            "chatTemplate": chat_template_source,
            "chatTemplateValue": str(getattr(tokenizer, "chat_template", "") or ""),
            "addSpecialTokens": add_special_tokens,
            "generation": payload.get("generation", {}),
            "tokenizer": {
                "name": str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)),
                "class": tokenizer.__class__.__name__,
                "vocabSize": int(len(tokenizer)),
                "modelMaxLength": int(getattr(tokenizer, "model_max_length", 0) or 0),
                "specialTokenIds": {
                    "bos": getattr(tokenizer, "bos_token_id", None),
                    "eos": getattr(tokenizer, "eos_token_id", None),
                    "pad": getattr(tokenizer, "pad_token_id", None),
                    "unk": getattr(tokenizer, "unk_token_id", None),
                },
            },
            "software": {
                "worker": WORKER_VERSION,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            },
        },
        "performance": performance,
        "evidence": {
            "kind": evidence_kind,
            "causal": bool(interventions),
            "dla": "Raw component dot product with the selected unembedding row; it is not an intervention and does not include the model's final normalization.",
            "attention": "Attention patterns are observational and do not establish causal importance.",
            "intervention": "Intervention effects apply only to this model, input, target, metric, component, and replacement method." if interventions else None,
        },
    }
    stored_request = {
        key: value for key, value in payload.items()
        if key != "direction" and not key.startswith("_")
    }
    device_memory = _device_memory(torch)
    waterfall.finish(
        "result-assembly",
        "Result assembly",
        "cpu",
        "JSON-safe response records, provenance, context, and device-memory statistics.",
    )
    if payload.get("_storeRun", True):
        _store_run(state, run_id, stored_request, result, latest, _cpu_tensor(logits))
        waterfall.finish(
            "run-retention",
            "Run retention",
            "storage",
            "Bounded in-worker activation and run snapshot retained for comparisons and interventions.",
        )
    timing = waterfall.snapshot()
    model_forward = next(
        (phase["durationMs"] for phase in timing["phases"] if phase["key"] == "model-forward"),
        None,
    )
    performance.update({
        "durationMs": timing["totalMs"],
        "modelForwardMs": model_forward,
        "deviceMemory": device_memory,
        "layerTimings": diagnostics.get("layerTimings", []),
        "waterfall": timing,
    })
    return result


def _generation_trace(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers

    with state.lock:
        model = state.model
        tokenizer = state.tokenizer
        model_id = state.model_id
        revision = state.revision
        device_name = state.device
        dtype_name = state.dtype
    if model is None or tokenizer is None:
        raise ValueError("Load a model into the execution worker first")
    prompt, chat_template_source = _prepare_prompt(tokenizer, payload)
    if not prompt:
        raise ValueError("Enter a prompt for the generation trace")
    settings = _normalise_generation_settings(payload.get("generation"))
    lens_settings = _normalise_lens_settings(payload.get("logitLens"))
    if lens_settings["enabled"] and settings["maxNewTokens"] * lens_settings["maxStages"] > 256:
        raise ValueError("Generation length × logit-lens stages is limited to 256; reduce tokens or lens stages")
    seed = max(0, min(2**31 - 1, int(payload.get("seed", 0))))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=bool(payload.get("addSpecialTokens", True)),
    )
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    if len(prompt_ids) + settings["maxNewTokens"] > MAX_PROMPT_TOKENS:
        raise ValueError("Prompt plus requested generation exceeds the worker token limit")
    input_device = _model_input_device(model)
    current_ids = list(prompt_ids)
    generated_ids: list[int] = []
    steps = []
    display_top_k = max(1, min(10, int(payload.get("topK", 5))))
    started = time.perf_counter()
    stop_reason = "max-new-tokens"
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for step_index in range(settings["maxNewTokens"]):
        input_ids = torch.tensor([current_ids], dtype=torch.long, device=input_device)
        attention_mask = torch.ones_like(input_ids)
        normalization_calls: dict[str, tuple[Any, Any, Any]] = {}
        normalization_handles = (
            _register_normalization_trace(model, normalization_calls)
            if lens_settings["enabled"]
            else []
        )
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=lens_settings["enabled"],
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in normalization_handles:
                handle.remove()
        logits = output.logits[0, -1].detach().float()
        token_id = _sample_next_token(logits, settings)
        summary = _prediction_summary(logits, tokenizer, token_id, display_top_k)
        output_hidden_states = getattr(output, "hidden_states", None)
        lens = (
            _logit_lens_timeline(
                model,
                tokenizer,
                output_hidden_states,
                logits,
                token_id,
                lens_settings,
                _verified_final_normalization(
                    normalization_calls,
                    output_hidden_states[-1] if output_hidden_states else None,
                ),
            )
            if lens_settings["enabled"]
            else {"available": False, "stages": [], "disabled": True}
        )
        token_text = tokenizer.decode([token_id])
        steps.append({
            "index": step_index,
            "contextPosition": len(current_ids) - 1,
            "contextTokenCount": len(current_ids),
            "token": {
                "id": token_id,
                "token": tokenizer.convert_ids_to_tokens([token_id])[0],
                "text": token_text,
            },
            "selection": "sampled" if settings["doSample"] else "greedy",
            "chosenProbability": summary["targetProbability"],
            "chosenLogit": summary["targetLogit"],
            "chosenRank": summary["targetRank"],
            "entropy": summary["entropy"],
            "topK": summary["topK"],
            "logitLens": lens,
        })
        current_ids.append(token_id)
        generated_ids.append(token_id)
        if eos_id is not None and token_id == int(eos_id):
            stop_reason = "eos-token"
            break
    return {
        "ok": True,
        "generationId": uuid.uuid4().hex,
        "modelId": model_id,
        "revision": revision,
        "prompt": prompt,
        "promptTokenCount": len(prompt_ids),
        "completion": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "generatedTokenIds": generated_ids,
        "steps": steps,
        "stopReason": stop_reason,
        "settings": {**settings, "seed": seed},
        "context": {
            "device": device_name,
            "dtype": dtype_name,
            "chatTemplate": chat_template_source,
            "tokenizer": {
                "name": str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)),
                "class": tokenizer.__class__.__name__,
            },
            "software": {
                "worker": WORKER_VERSION,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            },
        },
        "performance": {
            "durationMs": (time.perf_counter() - started) * 1000,
            "deviceMemory": _device_memory(torch),
            "fullContextRecomputations": len(steps),
        },
        "evidence": {
            "kind": "observational-generation-trace",
            "causal": False,
            "note": "Each step is an autoregressive model observation recomputed from the full preceding context. Logit-lens stages are observational decodes, not causal interventions.",
            "sampling": "Chosen-token probabilities are from the unfiltered model distribution; top-k/top-p/temperature affect selection only." if settings["doSample"] else "Greedy decoding selects the maximum-logit token at every step.",
        },
    }


def _token_alignment(failure_tokens: list[dict[str, Any]], control_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failure_ids = [int(item["id"]) for item in failure_tokens]
    control_ids = [int(item["id"]) for item in control_tokens]
    matcher = difflib.SequenceMatcher(a=failure_ids, b=control_ids, autojunk=False)
    result: list[dict[str, Any]] = []
    for tag, failure_start, failure_end, control_start, control_end in matcher.get_opcodes():
        width = max(failure_end - failure_start, control_end - control_start)
        for offset in range(width):
            failure_index = failure_start + offset if failure_start + offset < failure_end else None
            control_index = control_start + offset if control_start + offset < control_end else None
            result.append({
                "status": "matched" if tag == "equal" else "changed" if failure_index is not None and control_index is not None else "failure-only" if failure_index is not None else "control-only",
                "failureIndex": failure_index,
                "controlIndex": control_index,
                "failureToken": failure_tokens[failure_index] if failure_index is not None else None,
                "controlToken": control_tokens[control_index] if control_index is not None else None,
            })
    return result


def _activation_difference(failure: Any, control: Any) -> dict[str, Any] | None:
    import torch

    if not torch.is_tensor(failure) or not torch.is_tensor(control):
        return None
    failure_vector = _position_vector(failure).detach().float().flatten()
    control_vector = _position_vector(control).detach().float().flatten()
    if failure_vector.numel() != control_vector.numel() or not failure_vector.numel():
        return None
    difference = failure_vector - control_vector
    failure_norm = torch.linalg.vector_norm(failure_vector)
    control_norm = torch.linalg.vector_norm(control_vector)
    distance = torch.linalg.vector_norm(difference)
    cosine = None
    if failure_norm > 0 and control_norm > 0:
        cosine = float(torch.dot(failure_vector, control_vector).div(failure_norm * control_norm).item())
    return {
        "l2Distance": float(distance.item()),
        "relativeDistance": float(distance.div(control_norm.clamp_min(1e-12)).item()),
        "cosineSimilarity": cosine,
        "cosineDistance": 1.0 - cosine if cosine is not None else None,
        "meanAbsoluteDifference": float(difference.abs().mean().item()),
        "failureNorm": float(failure_norm.item()),
        "controlNorm": float(control_norm.item()),
    }


def _attention_difference(failure: Any, control: Any) -> dict[str, Any] | None:
    import torch

    if not torch.is_tensor(failure) or not torch.is_tensor(control) or failure.ndim != 4 or control.ndim != 4:
        return None
    failure_rows = failure[0, :, -1, :].detach().float()
    control_rows = control[0, :, -1, :].detach().float()
    heads = min(failure_rows.shape[0], control_rows.shape[0])
    positions = min(failure_rows.shape[-1], control_rows.shape[-1])
    if not heads or not positions:
        return None
    failure_rows = failure_rows[:heads, -positions:]
    control_rows = control_rows[:heads, -positions:]
    changes = (failure_rows - control_rows).abs().mean(dim=-1)
    return {
        "meanAbsoluteDifference": float(changes.mean().item()),
        "maximumHeadDifference": float(changes.max().item()),
        "heads": [{"head": index, "meanAbsoluteDifference": float(value)} for index, value in enumerate(changes.tolist())],
    }


def _attention_head_output_difference(failure: Any, control: Any) -> dict[str, Any] | None:
    import torch

    if not torch.is_tensor(failure) or not torch.is_tensor(control) or failure.ndim != 4 or control.ndim != 4:
        return None
    failure_heads = failure[0, -1].detach().float()
    control_heads = control[0, -1].detach().float()
    heads = min(failure_heads.shape[0], control_heads.shape[0])
    width = min(failure_heads.shape[-1], control_heads.shape[-1])
    if not heads or not width:
        return None
    failure_heads = failure_heads[:heads, :width]
    control_heads = control_heads[:heads, :width]
    differences = failure_heads - control_heads
    distances = torch.linalg.vector_norm(differences, dim=-1)
    cosine = torch.nn.functional.cosine_similarity(failure_heads, control_heads, dim=-1)
    ranked = sorted(({
        "head": head,
        "l2Distance": float(distances[head].item()),
        "cosineSimilarity": float(cosine[head].item()),
        "cosineDistance": float(1.0 - cosine[head].item()),
        "meanAbsoluteDifference": float(differences[head].abs().mean().item()),
    } for head in range(heads)), key=lambda item: item["l2Distance"], reverse=True)
    return {
        "meanL2Distance": float(distances.mean().item()),
        "maximumL2Distance": float(distances.max().item()),
        "heads": ranked,
    }


def _distribution_kl(control_logits: Any, failure_logits: Any) -> float:
    import torch

    control = control_logits.detach().float()
    failure = failure_logits.detach().float()
    control_log_probabilities = torch.log_softmax(control, dim=-1)
    failure_log_probabilities = torch.log_softmax(failure, dim=-1)
    control_probabilities = control_log_probabilities.exp()
    return float(torch.sum(control_probabilities * (control_log_probabilities - failure_log_probabilities)).item())


def _compare_runs(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    failure_payload = payload.get("failure")
    control_payload = payload.get("control")
    if not isinstance(failure_payload, dict) or not isinstance(control_payload, dict):
        raise ValueError("Comparison requires failure and control run specifications")
    metric_specification = _normalise_metric(payload.get("metric"), str(payload.get("targetToken", "")))
    shared = {
        "metric": metric_specification,
        "targetToken": str(payload.get("targetToken", metric_specification.get("targetToken", ""))),
        "topK": int(payload.get("topK", 10)),
        "seed": int(payload.get("seed", 0)),
        "generation": payload.get("generation", {}),
    }
    failure = _forward(state, {**shared, **failure_payload, "role": "failure"})
    control = _forward(state, {**shared, **control_payload, "role": "control"})
    failure_record = _run_record(state, failure["runId"])
    control_record = _run_record(state, control["runId"])
    layer_count = min(len(failure.get("layers", [])), len(control.get("layers", [])))
    layers: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    attention_heads: list[dict[str, Any]] = []
    for layer in range(layer_count):
        failure_layer = failure["layers"][layer]
        control_layer = control["layers"][layer]
        residual = _activation_difference(
            failure_record.tensors.get(f"resid_post.{layer}"),
            control_record.tensors.get(f"resid_post.{layer}"),
        ) or {}
        attention_pattern = _attention_difference(
            failure_record.tensors.get(f"attention_probs.{layer}"),
            control_record.tensors.get(f"attention_probs.{layer}"),
        ) or {}
        attention_head_outputs = _attention_head_output_difference(
            failure_record.tensors.get(f"attention_head_outputs.{layer}"),
            control_record.tensors.get(f"attention_head_outputs.{layer}"),
        ) or {}
        attention_heads.extend({
            **head,
            "id": f"attention.{layer}.head.{head['head']}",
            "layer": layer,
            "nodeId": f"l{layer}_output",
            "label": f"Layer {layer} attention head {head['head']} output",
        } for head in attention_head_outputs.get("heads", []))
        logit_lens_delta = None
        failure_dla = failure_layer.get("residPost", {}).get("dla")
        control_dla = control_layer.get("residPost", {}).get("dla")
        if isinstance(failure_dla, (int, float)) and isinstance(control_dla, (int, float)):
            logit_lens_delta = float(failure_dla - control_dla)
        layer_score = (
            float(residual.get("relativeDistance", 0.0))
            + float(residual.get("cosineDistance", 0.0) or 0.0)
            + float(attention_pattern.get("meanAbsoluteDifference", 0.0))
            + min(10.0, float(attention_head_outputs.get("meanL2Distance", 0.0))) * 0.1
            + min(10.0, abs(logit_lens_delta or 0.0)) * 0.1
        )
        layer_result = {
            "layer": layer,
            "nodeId": f"l{layer}_mlp_residual",
            "score": layer_score,
            "residual": residual,
            "attention": attention_pattern,
            "attentionHeadOutputs": attention_head_outputs,
            "logitLensDifference": logit_lens_delta,
        }
        layers.append(layer_result)
        components.append({
            "id": f"residual.{layer}",
            "kind": "residual",
            "layer": layer,
            "nodeId": f"l{layer}_mlp_residual",
            "label": f"Layer {layer} residual output",
            "score": layer_score,
            **residual,
            "logitLensDifference": logit_lens_delta,
        })
        for kind, node_id in (("attention", f"l{layer}_output"), ("mlp", f"l{layer}_mlp")):
            difference = _activation_difference(
                failure_record.tensors.get(f"{kind}_write.{layer}"),
                control_record.tensors.get(f"{kind}_write.{layer}"),
            )
            if difference is None:
                continue
            failure_write = failure_layer.get("attentionWrite" if kind == "attention" else "mlpWrite") or {}
            control_write = control_layer.get("attentionWrite" if kind == "attention" else "mlpWrite") or {}
            contribution_delta = None
            if isinstance(failure_write.get("dla"), (int, float)) and isinstance(control_write.get("dla"), (int, float)):
                contribution_delta = float(failure_write["dla"] - control_write["dla"])
            score = float(difference["relativeDistance"]) + float(difference.get("cosineDistance") or 0.0) + min(10.0, abs(contribution_delta or 0.0)) * 0.1
            components.append({
                "id": f"{kind}.{layer}",
                "kind": kind,
                "layer": layer,
                "nodeId": node_id,
                "label": f"Layer {layer} {kind} output",
                "score": score,
                **difference,
                "outputContributionDifference": contribution_delta,
            })
    components.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    layers.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    attention_heads.sort(key=lambda item: float(item.get("l2Distance", 0.0)), reverse=True)
    kl_divergence = _distribution_kl(control_record.logits, failure_record.logits)
    failure_value = failure.get("metric", {}).get("value")
    control_value = control.get("metric", {}).get("value")
    metric_difference = (
        float(failure_value - control_value)
        if isinstance(failure_value, (int, float)) and isinstance(control_value, (int, float))
        else None
    )
    comparison_metric = {
        "name": metric_specification["name"],
        "kind": metric_specification["kind"],
        "failureValue": failure_value,
        "controlValue": control_value,
        "failureMinusControl": metric_difference,
        "controlToFailureKL": kl_divergence,
        "direction": failure.get("metric", {}).get("direction", "maximize"),
    }
    if metric_specification["kind"] == "kl_divergence":
        comparison_metric.update({"failureValue": kl_divergence, "controlValue": 0.0, "failureMinusControl": kl_divergence, "direction": "minimize"})
    failure_predictions = {int(item["tokenId"]): item for item in failure.get("nextToken", {}).get("topK", [])}
    control_predictions = {int(item["tokenId"]): item for item in control.get("nextToken", {}).get("topK", [])}
    prediction_differences = []
    for token_id in failure_predictions.keys() | control_predictions.keys():
        failure_prediction = failure_predictions.get(token_id, {})
        control_prediction = control_predictions.get(token_id, {})
        prediction_differences.append({
            "tokenId": token_id,
            "text": failure_prediction.get("text", control_prediction.get("text", "")),
            "failureLogit": failure_prediction.get("logit"),
            "controlLogit": control_prediction.get("logit"),
            "logitDifference": float(failure_prediction.get("logit", 0.0) - control_prediction.get("logit", 0.0)),
            "failureProbability": failure_prediction.get("probability", 0.0),
            "controlProbability": control_prediction.get("probability", 0.0),
            "probabilityDifference": float(failure_prediction.get("probability", 0.0) - control_prediction.get("probability", 0.0)),
        })
    prediction_differences.sort(key=lambda item: abs(item["logitDifference"]), reverse=True)
    return {
        "ok": True,
        "comparisonId": uuid.uuid4().hex,
        "modelId": failure["modelId"],
        "revision": failure["revision"],
        "failure": failure,
        "control": control,
        "metric": comparison_metric,
        "outputs": {
            "failureTopPrediction": failure.get("nextToken", {}).get("topK", [None])[0],
            "controlTopPrediction": control.get("nextToken", {}).get("topK", [None])[0],
            "targetLogitDifference": float(failure["target"]["logit"] - control["target"]["logit"]),
            "targetProbabilityDifference": float(failure["target"]["probability"] - control["target"]["probability"]),
            "predictionDifferences": prediction_differences[: max(10, int(shared["topK"]))],
        },
        "tokenAlignment": _token_alignment(failure["tokens"], control["tokens"]),
        "divergence": {
            "layers": layers,
            "components": components,
            "attentionHeads": attention_heads,
            "firstMaterialLayer": next((item["layer"] for item in sorted(layers, key=lambda item: item["layer"]) if item["score"] >= 0.1), None),
            "outputKL": kl_divergence,
            "cacheBytesDifference": int(failure.get("kvCache", {}).get("bytes", 0)) - int(control.get("kvCache", {}).get("bytes", 0)),
        },
        "evidence": {
            "kind": "comparative",
            "causal": False,
            "note": "Divergence identifies differences between these paired runs. It does not establish which difference caused the output change.",
        },
    }


def _intervene(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    base_run_id = str(payload.get("baseRunId", ""))
    base = _run_record(state, base_run_id)
    component = payload.get("component")
    if not isinstance(component, dict):
        raise ValueError("Select a graph component to intervene on")
    metric = _normalise_metric(payload.get("metric", base.request.get("metric")), str(base.result.get("target", {}).get("text", "")))
    _require_base_metric(base, metric, "Intervention")
    intervention = {
        **component,
        "method": str(payload.get("method", "zero")),
        "scale": payload.get("scale", 0.0 if payload.get("method") == "zero" else 1.0),
        "scope": str(payload.get("scope", "position")),
        "position": int(payload.get("position", -1)),
        "sourceRunId": str(payload.get("sourceRunId", "")),
    }
    run_payload = {
        **base.request,
        "prompt": base.result["prompt"],
        "targetToken": base.result.get("target", {}).get("text", ""),
        "metric": metric,
        "interventions": [intervention],
    }
    intervened = _forward(state, run_payload)
    base_value = base.result.get("metric", {}).get("value")
    changed_value = intervened.get("metric", {}).get("value")
    if metric["kind"] == "kl_divergence":
        base_value = 0.0
        changed_value = _distribution_kl(base.logits, _run_record(state, intervened["runId"]).logits)
        intervened["metric"].update({"value": changed_value, "direction": "minimize"})
    effect = float(changed_value - base_value) if isinstance(base_value, (int, float)) and isinstance(changed_value, (int, float)) else None
    direction = intervened.get("metric", {}).get("direction", "maximize")
    improvement = -effect if direction == "minimize" and effect is not None else effect
    return {
        "ok": True,
        "interventionId": uuid.uuid4().hex,
        "baseRunId": base_run_id,
        "sourceRunId": intervention["sourceRunId"] or None,
        "component": component,
        "method": intervention["method"],
        "scope": intervention["scope"],
        "scale": intervention.get("scale"),
        "position": intervention.get("position"),
        "metric": {
            "name": intervened["metric"]["name"],
            "kind": intervened["metric"]["kind"],
            "baseline": base_value,
            "intervened": changed_value,
            "signedEffect": effect,
            "improvement": improvement,
            "direction": direction,
        },
        "outputEffect": {
            "targetLogit": intervened["target"]["logit"] - base.result["target"]["logit"],
            "targetProbability": intervened["target"]["probability"] - base.result["target"]["probability"],
            "targetRank": intervened["target"]["rank"] - base.result["target"]["rank"],
            "distributionKL": _distribution_kl(base.logits, _run_record(state, intervened["runId"]).logits),
        },
        "run": intervened,
        "evidence": {
            "kind": "causal",
            "causal": True,
            "scope": "This effect is causal for the specified intervention under this exact run context; it is not a general component interpretation.",
        },
    }


def _normalise_sweep_axes(payload: Any, *, layer_count: int, token_count: int) -> dict[str, Any]:
    value = payload if isinstance(payload, dict) else {}
    kind = str(value.get("kind", "attention")).strip().lower()
    if kind not in {"residual", "attention", "mlp"}:
        raise ValueError("Sweep component must be residual, attention, or MLP")
    method = str(value.get("method", "zero")).strip().lower()
    if method not in {"zero", "mean", "patch", "scale"}:
        raise ValueError("Sweep method must be zero, mean, patch, or scale")
    raw_layers = value.get("layers", list(range(layer_count)))
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("Sweep requires at least one layer")
    layers = sorted({int(layer) for layer in raw_layers})
    if any(layer < 0 or layer >= layer_count for layer in layers):
        raise ValueError(f"Sweep layers must be between 0 and {max(0, layer_count - 1)}")
    raw_positions = value.get("positions", [-1])
    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError("Sweep requires at least one token position")
    positions = []
    for raw_position in raw_positions:
        position = int(raw_position)
        if position < 0:
            position += token_count
        if position < 0 or position >= token_count:
            raise ValueError(f"Sweep positions must be between 0 and {max(0, token_count - 1)}")
        if position not in positions:
            positions.append(position)
    positions.sort()
    if len(layers) * len(positions) > MAX_SWEEP_CELLS:
        raise ValueError(f"Causal sweeps are limited to {MAX_SWEEP_CELLS} layer × position cells")
    scale = float(value.get("scale", 0.0 if method == "zero" else 1.0))
    if not math.isfinite(scale) or abs(scale) > 100:
        raise ValueError("Sweep scale must be a finite value between -100 and 100")
    return {"kind": kind, "method": method, "layers": layers, "positions": positions, "scale": scale}


def _causal_sweep(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    base_run_id = str(payload.get("baseRunId", ""))
    base = _run_record(state, base_run_id)
    metric = _normalise_metric(payload.get("metric", base.request.get("metric")), str(base.result.get("target", {}).get("text", "")))
    _require_base_metric(base, metric, "Causal sweep")
    if metric["kind"] == "kl_divergence":
        raise ValueError("Causal sweeps require a scalar behaviour metric; choose a metric other than KL divergence")
    layers = base.result.get("layers", [])
    tokens = base.result.get("tokens", [])
    axes = _normalise_sweep_axes(payload, layer_count=len(layers), token_count=len(tokens))
    source_run_id = str(payload.get("sourceRunId", ""))
    if axes["method"] == "patch" and not source_run_id:
        raise ValueError("Patch sweeps require a paired source run")
    source_records: dict[str, RunRecord] = {}
    if source_run_id:
        source_records[source_run_id] = _run_record(state, source_run_id)
    baseline_value = base.result.get("metric", {}).get("value")
    if not isinstance(baseline_value, (int, float)):
        raise ValueError("The base run did not produce a scalar metric value")
    direction = str(base.result.get("metric", {}).get("direction", "maximize"))
    captured = {
        (str(item.get("category")), int(item.get("layer")))
        for item in base.result.get("hooks", [])
        if isinstance(item.get("layer"), int)
    }
    clean_request = {
        key: item for key, item in base.request.items()
        if key not in {"interventions", "sourceRunId"} and not key.startswith("_")
    }
    node_ids = {
        "residual": lambda layer: f"l{layer}_mlp_residual",
        "attention": lambda layer: f"l{layer}_output",
        "mlp": lambda layer: f"l{layer}_mlp",
    }
    cells = []
    started = time.perf_counter()
    for layer in axes["layers"]:
        for position in axes["positions"]:
            cell = {
                "layer": layer,
                "position": position,
                "positionToken": tokens[position].get("text", tokens[position].get("token", "")),
                "nodeId": node_ids[axes["kind"]](layer),
                "status": "measured",
            }
            if (axes["kind"], layer) not in captured:
                cells.append({
                    **cell,
                    "status": "unsupported",
                    "baseline": baseline_value,
                    "intervened": None,
                    "signedEffect": None,
                    "improvement": None,
                    "error": f"Layer {layer} {axes['kind']} output was not captured by this architecture adapter",
                })
                continue
            intervention = {
                "kind": axes["kind"],
                "layer": layer,
                "method": axes["method"],
                "scale": axes["scale"],
                "scope": "position",
                "position": position,
                "sourceRunId": source_run_id,
            }
            try:
                changed = _forward(state, {
                    **clean_request,
                    "prompt": base.result["prompt"],
                    "targetToken": base.result.get("target", {}).get("text", ""),
                    "metric": metric,
                    "interventions": [intervention],
                    "_sourceRecords": source_records,
                    "_storeRun": False,
                    "logitLens": {"enabled": False},
                })
                changed_value = changed.get("metric", {}).get("value")
                if not isinstance(changed_value, (int, float)):
                    raise ValueError("Intervention did not produce a scalar metric")
                signed_effect = float(changed_value - baseline_value)
                improvement = -signed_effect if direction == "minimize" else signed_effect
                cells.append({
                    **cell,
                    "baseline": float(baseline_value),
                    "intervened": float(changed_value),
                    "signedEffect": signed_effect,
                    "improvement": improvement,
                    "topToken": (changed.get("nextToken", {}).get("topK") or [{}])[0].get("text"),
                })
            except ValueError as error:
                cells.append({
                    **cell,
                    "status": "error",
                    "baseline": float(baseline_value),
                    "intervened": None,
                    "signedEffect": None,
                    "improvement": None,
                    "error": str(error),
                })
    measured = [cell for cell in cells if cell["status"] == "measured"]
    maximum = max(measured, key=lambda cell: abs(cell["signedEffect"])) if measured else None
    method_notes = {
        "zero": "The selected activation vector was replaced with zero at one token position.",
        "mean": "The selected activation vector was replaced with that run's mean over token positions; this is not a dataset-mean ablation.",
        "patch": "The selected activation vector was replaced by the aligned vector from the paired source run.",
        "scale": f"The selected activation vector was multiplied by {axes['scale']} at one token position.",
    }
    return {
        "ok": True,
        "sweepId": uuid.uuid4().hex,
        "baseRunId": base_run_id,
        "sourceRunId": source_run_id or None,
        "modelId": base.result.get("modelId"),
        "revision": base.result.get("revision"),
        "kind": axes["kind"],
        "method": axes["method"],
        "scale": axes["scale"],
        "layers": axes["layers"],
        "positions": [{"index": position, "token": tokens[position].get("text", tokens[position].get("token", ""))} for position in axes["positions"]],
        "metric": {
            "name": metric["name"],
            "kind": metric["kind"],
            "baseline": float(baseline_value),
            "direction": direction,
        },
        "cells": cells,
        "summary": {
            "requested": len(cells),
            "measured": len(measured),
            "unsupported": sum(cell["status"] == "unsupported" for cell in cells),
            "errors": sum(cell["status"] == "error" for cell in cells),
            "maximumAbsoluteEffect": maximum,
        },
        "performance": {"durationMs": (time.perf_counter() - started) * 1000},
        "evidence": {
            "kind": "causal-intervention-sweep",
            "causal": True,
            "scope": "Every measured cell compares the same base prompt and scalar metric with one position-scoped intervention. Effects do not establish a general semantic role.",
            "method": method_notes[axes["method"]],
            "multipleComparisons": "The heatmap is exploratory. Confirm selected cells on held-out prompts and report the full tested grid rather than only the maximum.",
        },
    }


def _differentiable_metric(
    logits: Any,
    tokenizer: Any,
    specification: dict[str, Any],
    fallback_target_id: int,
) -> Any:
    """Return the scalar metric tensor used by edge-attribution patching."""
    import torch

    kind = specification["kind"]
    if kind == "target_probability":
        ids = _token_group_ids(tokenizer, [specification["targetToken"]]) if specification["targetToken"] else []
        return torch.softmax(logits.float(), dim=-1)[ids[-1] if ids else fallback_target_id]
    if kind == "logit_difference":
        correct_ids = _token_group_ids(tokenizer, specification["correctTokens"])
        incorrect_ids = _token_group_ids(tokenizer, specification["incorrectTokens"])
        if not correct_ids or not incorrect_ids:
            raise ValueError("Logit difference requires at least one correct and one incorrect token")
        return logits[correct_ids].float().mean() - logits[incorrect_ids].float().mean()
    if kind == "custom_token_groups":
        positive_ids = _token_group_ids(tokenizer, specification["positiveTokens"])
        negative_ids = _token_group_ids(tokenizer, specification["negativeTokens"])
        if not positive_ids or not negative_ids:
            raise ValueError("Custom token groups require at least one positive and one negative token")
        return logits[positive_ids].float().mean() - logits[negative_ids].float().mean()
    raise ValueError(
        "EAP root-cause tracing currently supports target probability, logit difference, and custom token-group metrics"
    )


def _component_identity(kind: str, layer: int) -> dict[str, Any]:
    if kind == "attention":
        return {
            "id": f"attention.{layer}",
            "kind": kind,
            "layer": layer,
            "label": f"Layer {layer} attention output",
            "nodeId": f"l{layer}_output",
            "edge": {"from": f"l{layer}_output", "to": f"l{layer}_attn_residual"},
        }
    return {
        "id": f"mlp.{layer}",
        "kind": kind,
        "layer": layer,
        "label": f"Layer {layer} MLP output",
        "nodeId": f"l{layer}_mlp",
        "edge": {"from": f"l{layer}_mlp", "to": f"l{layer}_mlp_residual"},
    }


def _selection_stability(candidates: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    thresholds = [threshold * 0.5, threshold, threshold * 2.0]
    selections = [
        {str(item["id"]) for item in candidates if abs(float(item.get("acdcEffect", 0.0))) >= value}
        for value in thresholds
    ]

    def jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 1.0

    pairwise = [jaccard(selections[0], selections[1]), jaccard(selections[1], selections[2])]
    return {
        "score": sum(pairwise) / len(pairwise),
        "thresholds": [
            {"threshold": value, "retained": len(selection), "componentIds": sorted(selection)}
            for value, selection in zip(thresholds, selections)
        ],
        "adjacentJaccard": pairwise,
    }


def _root_cause(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    """Run EAP candidate ranking followed by intervention-backed ACDC pruning."""
    import torch

    failure = _run_record(state, str(payload.get("failureRunId", "")))
    control = _run_record(state, str(payload.get("controlRunId", "")))
    metric = _normalise_metric(
        payload.get("metric", failure.request.get("metric")),
        str(failure.result.get("target", {}).get("text", "")),
    )
    _require_base_metric(failure, metric, "Root-cause trace")
    _require_base_metric(control, metric, "Root-cause trace")
    maximum_candidates = max(4, min(64, int(payload.get("maxCandidates", 16))))
    threshold = max(0.0, float(payload.get("threshold", 0.01)))
    failure_value = failure.result.get("metric", {}).get("value")
    control_value = control.result.get("metric", {}).get("value")
    if not isinstance(failure_value, (int, float)) or not isinstance(control_value, (int, float)):
        raise ValueError("Root-cause tracing requires a scalar failing/control metric")

    with state.lock:
        model = state.model
        tokenizer = state.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("Load a model into the execution worker first")
    encoded = tokenizer(
        failure.result["prompt"],
        return_tensors="pt",
        add_special_tokens=bool(failure.result.get("context", {}).get("addSpecialTokens", True)),
    )
    device = _model_input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    captures: dict[tuple[int, str], Any] = {}
    handles = []

    def gradient_hook(layer: int, kind: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = _first_tensor(output)
            if tensor is not None and tensor.requires_grad:
                tensor.retain_grad()
                captures[(layer, kind)] = tensor
            return output
        return capture

    for name, module in model.named_modules():
        layer = _layer_index(name)
        kind = _module_category(name)
        if layer is not None and kind in {"attention", "mlp"}:
            handles.append(module.register_forward_hook(gradient_hook(layer, kind)))
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(**encoded, use_cache=False, return_dict=True)
            logits = output.logits[0, -1]
            scalar = _differentiable_metric(
                logits,
                tokenizer,
                metric,
                int(failure.result.get("target", {}).get("tokenId", torch.argmax(logits).item())),
            )
            scalar.backward()
    finally:
        for handle in handles:
            handle.remove()

    candidates: list[dict[str, Any]] = []
    for (layer, kind), tensor in captures.items():
        gradient = tensor.grad
        source = control.tensors.get(f"{kind}_write.{layer}")
        if gradient is None or source is None:
            continue
        aligned_source = _aligned_source_tensor(source, tensor)
        delta = _position_vector(aligned_source - tensor.detach()).float().flatten()
        gradient_vector = _position_vector(gradient).detach().float().flatten()
        if delta.numel() != gradient_vector.numel() or not delta.numel():
            continue
        score = float(torch.dot(delta, gradient_vector).item())
        if not math.isfinite(score):
            continue
        candidates.append({
            **_component_identity(kind, layer),
            "eapScore": score,
            "absoluteEapScore": abs(score),
        })
    candidates.sort(key=lambda item: item["absoluteEapScore"], reverse=True)
    candidates = candidates[:maximum_candidates]
    if not candidates:
        raise ValueError("This architecture did not expose differentiable attention or MLP outputs for EAP")

    source_records = {control.run_id: control, failure.run_id: failure}
    def evaluate_circuit(active: list[dict[str, Any]]) -> float:
        if not active:
            return float(failure_value)
        interventions = [{
            "kind": item["kind"],
            "layer": item["layer"],
            "method": "patch",
            "scope": "position",
            "position": failure.result.get("position", -1),
            "sourceRunId": control.run_id,
        } for item in active]
        changed = _forward(state, {
            **failure.request,
            "prompt": failure.result["prompt"],
            "metric": metric,
            "interventions": interventions,
            "_sourceRecords": source_records,
            "_storeRun": False,
        })
        changed_value = changed.get("metric", {}).get("value")
        if not isinstance(changed_value, (int, float)):
            raise ValueError("ACDC circuit evaluation did not produce a scalar metric")
        return float(changed_value)

    active = list(candidates)
    circuit_metric = evaluate_circuit(active)
    decisions: dict[str, dict[str, Any]] = {}
    for candidate in reversed(candidates):
        without = [item for item in active if item["id"] != candidate["id"]]
        without_metric = evaluate_circuit(without)
        marginal_effect = float(circuit_metric - without_metric)
        retained_candidate = abs(marginal_effect) >= threshold
        decisions[candidate["id"]] = {
            **candidate,
            "acdcEffect": marginal_effect,
            "patchedMetric": circuit_metric,
            "metricWithoutEdge": without_metric,
            "retained": retained_candidate,
        }
        if not retained_candidate:
            active = without
            circuit_metric = without_metric
    validated = [decisions[item["id"]] for item in candidates]
    retained = [item for item in validated if item["retained"]]
    denominator = float(control_value - failure_value)
    fidelity = (float(circuit_metric) - float(failure_value)) / denominator if abs(denominator) > 1e-12 else None
    stability = _selection_stability(validated, threshold)
    return {
        "ok": True,
        "traceId": uuid.uuid4().hex,
        "method": {"candidateDiscovery": "edge-attribution-patching", "pruning": "activation-patching-acdc"},
        "metric": {
            "name": metric["name"],
            "kind": metric["kind"],
            "failure": failure_value,
            "control": control_value,
            "circuit": circuit_metric,
        },
        "candidates": validated,
        "retained": retained,
        "threshold": threshold,
        "fidelity": fidelity,
        "stability": stability,
        "overlay": [{
            "nodeId": item["nodeId"],
            "edge": item["edge"],
            "eapScore": item["eapScore"],
            "acdcEffect": item["acdcEffect"],
            "retained": item["retained"],
        } for item in validated],
        "evidence": {
            "kind": "causal-circuit",
            "causal": True,
            "candidateMethod": "EAP uses the first-order gradient dot product between the failing activation and its aligned control activation.",
            "validationMethod": "ACDC greedily removes low-ranked residual-write edges when deleting their control-activation patch changes the current circuit metric by less than the configured threshold.",
            "scope": "Fidelity and stability apply only to this model revision, paired input, token position, metric, and patch distribution.",
        },
    }


def _verify(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload.get("candidate")
    cases = payload.get("cases")
    if not isinstance(candidate, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("Verification requires an intervention candidate and at least one case")
    if len(cases) > 32:
        raise ValueError("Verification is limited to 32 cases per run")
    component = candidate.get("component")
    if not isinstance(component, dict):
        raise ValueError("The intervention candidate is missing its component")
    source_run_id = str(candidate.get("sourceRunId", "") or "")
    source_records: dict[str, RunRecord] = {}
    if source_run_id:
        source_records[source_run_id] = _run_record(state, source_run_id)
    intervention = {
        **component,
        "method": str(candidate.get("method", "zero")),
        "scale": candidate.get("scale", 1.0),
        "scope": str(candidate.get("scope", "position")),
        "position": int(candidate.get("position", -1)),
        "sourceRunId": source_run_id,
    }
    tolerance = max(0.0, float(payload.get("tolerance", 1e-5)))
    results = []
    for index, value in enumerate(cases):
        if not isinstance(value, dict) or not str(value.get("prompt", "")).strip():
            raise ValueError("Every verification case requires a prompt")
        metric = _normalise_metric(value.get("metric", payload.get("metric")), str(value.get("targetToken", "")))
        if metric["kind"] == "kl_divergence":
            raise ValueError("KL verification requires an explicit reference distribution; choose another saved behaviour metric")
        run_payload = {
            "prompt": str(value["prompt"]),
            "targetToken": str(value.get("targetToken", metric.get("targetToken", ""))),
            "metric": metric,
            "topK": int(payload.get("topK", 10)),
            "seed": int(payload.get("seed", 0)),
            "_storeRun": False,
        }
        baseline = _forward(state, run_payload)
        fixed = _forward(state, {
            **run_payload,
            "interventions": [intervention],
            "_sourceRecords": source_records,
        })
        baseline_value = baseline.get("metric", {}).get("value")
        fixed_value = fixed.get("metric", {}).get("value")
        if not isinstance(baseline_value, (int, float)) or not isinstance(fixed_value, (int, float)):
            raise ValueError("Verification metric did not produce a scalar value")
        signed_effect = float(fixed_value - baseline_value)
        direction = fixed.get("metric", {}).get("direction", "maximize")
        improvement = -signed_effect if direction == "minimize" else signed_effect
        role = "failure" if index == 0 or str(value.get("role", "")) == "failure" else "guardrail"
        if improvement > tolerance:
            status = "fixed" if role == "failure" else "improved"
        elif improvement < -tolerance:
            status = "regressed"
        else:
            status = "unchanged"
        results.append({
            "index": index,
            "role": role,
            "label": str(value.get("label", f"Case {index + 1}")),
            "prompt": str(value["prompt"]),
            "baseline": baseline_value,
            "candidate": fixed_value,
            "signedEffect": signed_effect,
            "improvement": improvement,
            "status": status,
            "baselineTopToken": baseline.get("nextToken", {}).get("topK", [{}])[0].get("text"),
            "candidateTopToken": fixed.get("nextToken", {}).get("topK", [{}])[0].get("text"),
        })
    failing_status = results[0]["status"]
    guardrail_regressions = sum(item["status"] == "regressed" for item in results[1:])
    overall = "fixed" if failing_status == "fixed" and not guardrail_regressions else "regressed" if guardrail_regressions or failing_status == "regressed" else "unchanged"
    return {
        "ok": True,
        "verificationId": uuid.uuid4().hex,
        "status": overall,
        "candidate": candidate,
        "results": results,
        "summary": {
            "total": len(results),
            "fixed": sum(item["status"] in {"fixed", "improved"} for item in results),
            "unchanged": sum(item["status"] == "unchanged" for item in results),
            "regressed": sum(item["status"] == "regressed" for item in results),
        },
        "evidence": {
            "kind": "causal-regression-check",
            "scope": "Each result compares the same prompt and metric with and without the saved intervention candidate.",
        },
    }


def _activation(state: WorkerState, payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    run_id = str(payload.get("runId", ""))
    hook_name = str(payload.get("hookName", ""))
    position = int(payload.get("position", -1))
    head = max(0, int(payload.get("head", 0)))
    limit = max(1, min(4096, int(payload.get("limit", 512))))
    record = _run_record(state, run_id)
    tensor = record.tensors.get(hook_name)
    if tensor is None or not torch.is_tensor(tensor):
        raise ValueError("Unknown activation hook name")
    if hook_name.startswith("attention_probs.") and tensor.ndim == 4:
        if head >= tensor.shape[1]:
            raise ValueError(f"Head must be between 0 and {tensor.shape[1] - 1}")
        vector = tensor[0, head, position].detach().float().flatten()
    elif hook_name.startswith("attention_head_outputs.") and tensor.ndim == 4:
        if head >= tensor.shape[2]:
            raise ValueError(f"Head must be between 0 and {tensor.shape[2] - 1}")
        vector = tensor[0, position, head].detach().float().flatten()
    else:
        vector = _position_vector(tensor, position).detach().float().flatten()
    values = vector[:limit].tolist()
    top_count = min(32, vector.numel())
    top_values, top_indices = torch.topk(vector.abs(), k=top_count) if top_count else ([], [])
    top_features = [{
        "index": int(index),
        "value": float(vector[int(index)].item()),
        "magnitude": float(magnitude),
    } for magnitude, index in zip(top_values.tolist(), top_indices.tolist())]
    top_tokens: list[dict[str, Any]] = []
    if tensor.ndim == 3 and tensor.shape[0] and tensor.shape[1]:
        token_scores = torch.linalg.vector_norm(tensor[0].detach().float(), dim=-1)
        count = min(12, token_scores.numel())
        score_values, score_indices = torch.topk(token_scores, k=count)
        tokens = record.result.get("tokens", [])
        top_tokens = [{
            "position": int(index),
            "token": tokens[int(index)].get("text", tokens[int(index)].get("token", "")) if int(index) < len(tokens) else "",
            "activationNorm": float(score),
        } for score, index in zip(score_values.tolist(), score_indices.tolist())]
    elif hook_name.startswith("attention_head_outputs.") and tensor.ndim == 4:
        token_scores = torch.linalg.vector_norm(tensor[0, :, head].detach().float(), dim=-1)
        count = min(12, token_scores.numel())
        score_values, score_indices = torch.topk(token_scores, k=count)
        tokens = record.result.get("tokens", [])
        top_tokens = [{
            "position": int(index),
            "token": tokens[int(index)].get("text", tokens[int(index)].get("token", "")) if int(index) < len(tokens) else "",
            "activationNorm": float(score),
        } for score, index in zip(score_values.tolist(), score_indices.tolist())]
    comparison = None
    compare_run_id = str(payload.get("compareRunId", ""))
    if compare_run_id:
        compare_record = _run_record(state, compare_run_id)
        compare_tensor = compare_record.tensors.get(hook_name)
        if compare_tensor is not None and torch.is_tensor(compare_tensor):
            if (
                hook_name.startswith("attention_head_outputs.")
                and compare_tensor.ndim == tensor.ndim == 4
                and compare_tensor.shape[0] == tensor.shape[0]
                and compare_tensor.shape[2:] == tensor.shape[2:]
            ):
                aligned = tensor.detach().clone()
                token_count = min(compare_tensor.shape[1], tensor.shape[1])
                aligned[:, -token_count:] = compare_tensor[:, -token_count:].to(device=tensor.device, dtype=tensor.dtype)
            else:
                aligned = _aligned_source_tensor(compare_tensor, tensor)
            difference_tensor = tensor.detach().float() - aligned.detach().float()
            if hook_name.startswith("attention_probs.") and difference_tensor.ndim == 4:
                difference_vector = difference_tensor[0, head, position].flatten()
            elif hook_name.startswith("attention_head_outputs.") and difference_tensor.ndim == 4:
                difference_vector = difference_tensor[0, position, head].flatten()
            else:
                difference_vector = _position_vector(difference_tensor, position).flatten()
            compare_values = difference_vector[:limit].tolist()
            comparison = {
                "runId": compare_run_id,
                "meanAbsoluteDifference": float(difference_vector.abs().mean().item()),
                "l2Distance": float(torch.linalg.vector_norm(difference_vector).item()),
                "values": [float(value) if math.isfinite(float(value)) else None for value in compare_values],
            }
    contribution = None
    match = re.fullmatch(r"(attention|mlp)_write\.(\d+)", hook_name)
    if match:
        layer = int(match.group(2))
        key = "attentionWrite" if match.group(1) == "attention" else "mlpWrite"
        layers = record.result.get("layers", [])
        if layer < len(layers):
            contribution = layers[layer].get(key, {}).get("dla")
    return {
        "ok": True,
        "runId": run_id,
        "hookName": hook_name,
        "position": position,
        "head": head if hook_name.startswith(("attention_probs.", "attention_head_outputs.")) else None,
        "shape": list(tensor.shape),
        "returned": len(values),
        "total": vector.numel(),
        "truncated": len(values) < vector.numel(),
        "values": [float(value) if math.isfinite(float(value)) else None for value in values],
        "topFeatures": top_features,
        "topActivatingTokens": top_tokens,
        "directLogitAttribution": contribution,
        "comparison": comparison,
        "stats": _tensor_stats(tensor, position),
    }


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: WorkerState):
        self.state = state
        super().__init__(address, WorkerHandler)


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ModelDebuggerWorker"
    sys_version = ""
    server: WorkerServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[worker] {format % args}", flush=True)

    def do_GET(self) -> None:
        if not self._authorized():
            self._send(401, {"error": "Invalid worker session secret"})
            return
        if urlsplit(self.path).path != "/health":
            self._send(404, {"error": "Not found"})
            return
        state = self.server.state
        try:
            import torch

            _accelerator_type, accelerator = _available_accelerator(torch)
        except ImportError:
            _accelerator_type = "cpu"
            accelerator = "Dependencies not installed"
        with state.lock:
            payload = {
                "ok": True,
                "version": WORKER_VERSION,
                "accelerator": accelerator,
                "modelLoaded": state.model is not None,
                "modelId": state.model_id or None,
                "revision": state.revision or None,
                "device": state.device or None,
                "dtype": state.dtype or None,
                "quantization": state.quantization,
                "latestRunId": state.latest_run_id or None,
                "retainedRuns": len(state.runs),
                "memory": _worker_memory_capacity(torch, _accelerator_type),
                "capabilities": {
                    "pairedComparison": True,
                    "behaviourMetrics": sorted(METRIC_KINDS),
                    "interventions": ["zero", "mean", "resample", "patch", "scale", "steer"],
                    "activationSlices": True,
                    "rootCauseTrace": True,
                    "fixVerification": True,
                    "generationTimeline": True,
                    "logitLens": True,
                    "causalSweep": True,
                },
            }
        self._send(200, payload)

    def do_POST(self) -> None:
        if not self._authorized():
            self._send(401, {"error": "Invalid worker session secret"})
            return
        payload = self._read_json()
        if payload is None:
            return
        path = urlsplit(self.path).path
        try:
            if path == "/load":
                result = _load_model(self.server.state, payload)
            elif path == "/forward":
                result = _forward(self.server.state, payload)
            elif path == "/generate":
                result = _generation_trace(self.server.state, payload)
            elif path == "/compare":
                result = _compare_runs(self.server.state, payload)
            elif path == "/intervene":
                result = _intervene(self.server.state, payload)
            elif path == "/sweep":
                result = _causal_sweep(self.server.state, payload)
            elif path == "/root-cause":
                result = _root_cause(self.server.state, payload)
            elif path == "/verify":
                result = _verify(self.server.state, payload)
            elif path == "/activation":
                result = _activation(self.server.state, payload)
            else:
                self._send(404, {"error": "Not found"})
                return
        except Exception as error:
            self._send(400 if isinstance(error, (ValueError, TypeError)) else 500, {"error": str(error)})
            return
        self._send(200, result)

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.state.secret}"
        return hmac.compare_digest(supplied, expected)

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send(413 if length > MAX_REQUEST_BYTES else 400, {"error": "JSON body is missing or too large"})
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, {"error": "Request body must be valid JSON"})
            return None
        if not isinstance(payload, dict):
            self._send(400, {"error": "Request body must be a JSON object"})
            return None
        return payload

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ModelDebugger execution worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--session-file", default="")
    parser.add_argument(
        "--secret",
        default=os.environ.get(
            "MODELDEBUGGER_WORKER_SECRET",
            os.environ.get("REFUSALSCOPE_WORKER_SECRET", ""),
        ),
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        parser.error("--host must be loopback or 0.0.0.0 for a private managed runtime")
    secret = args.secret.strip() or secrets.token_urlsafe(32)
    state = WorkerState(secret=secret)
    server = WorkerServer((args.host, args.port), state)
    session_path = Path(args.session_file).expanduser().resolve() if args.session_file else None
    if session_path is not None:
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(json.dumps({
            "endpoint": f"http://127.0.0.1:{args.port}",
            "secret": secret,
            "pid": os.getpid(),
            "version": WORKER_VERSION,
        }), encoding="utf-8")
        session_path.chmod(0o600)
    print(f"ModelDebugger worker {WORKER_VERSION} listening on {args.host}:{args.port}", flush=True)
    print(f"Worker session secret: {secret}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if session_path is not None:
            try:
                current = json.loads(session_path.read_text(encoding="utf-8"))
                if int(current.get("pid", -1)) == os.getpid():
                    session_path.unlink()
            except (OSError, ValueError, json.JSONDecodeError):
                pass


if __name__ == "__main__":
    main()
