"""Build the browser-ready transformer circuit graph from checkpoint metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import prod
from typing import Iterable

NODE_WIDTH = 244
NODE_HEIGHT = 84
COLUMN_STRIDE = 360
ROW_STRIDE = 138
TENSOR_STACK_OFFSET = 6
TENSOR_STACK_MAX_DEPTH = 3
TENSOR_STACK_MAX_EXTENT = TENSOR_STACK_OFFSET * TENSOR_STACK_MAX_DEPTH

LAYER_COUNT_KEYS = (
    "num_hidden_layers",
    "n_layer",
    "num_layers",
    "n_layers",
    "decoder_layers",
    "num_decoder_layers",
)


class GraphError(ValueError):
    pass


@dataclass(slots=True)
class FamilyResult:
    found: bool = False
    prefix: str = ""
    inferred_count: int = 0
    observed_layers: int = 0
    score: int = 0
    predicted: bool = False


@dataclass(slots=True)
class Architecture:
    attention: str
    position: str
    position_kind: str
    norm: str
    feed_forward: str
    residual_topology: str
    norm_order: str
    topology_confidence: str
    topology_evidence: str
    query_heads: int | None
    kv_heads: int | None
    head_dim: int | None
    output_tied: bool | None
    findings: list[dict]


def configured_layer_count(config: object) -> int:
    if not isinstance(config, dict):
        return 0
    for key in LAYER_COUNT_KEYS:
        value = config.get(key)
        if isinstance(value, (int, float)) and 0 < int(value):
            return int(value)
    return 0


def _nested_configs(config: dict, depth: int = 1) -> Iterable[tuple[str, dict]]:
    if depth > 3:
        return
    for key, value in config.items():
        if isinstance(value, dict):
            yield key, value
            yield from _nested_configs(value, depth + 1)


def resolve_text_config(config: dict) -> dict:
    for key in ("text_config", "language_config", "llm_config", "decoder_config", "language_model_config"):
        candidate = config.get(key)
        if isinstance(candidate, dict) and configured_layer_count(candidate):
            return candidate
    if configured_layer_count(config):
        return config
    candidates = [
        (1 if any(word in key.lower() for word in ("text", "language", "llm", "decoder")) else 0, value)
        for key, value in _nested_configs(config)
        if configured_layer_count(value)
    ]
    return max(candidates, key=lambda item: item[0])[1] if candidates else config


def infer_layer_family(names: list[str], expected_count: int) -> FamilyResult:
    candidates: dict[str, dict] = {}
    for name in names:
        segments = name.split(".")
        for index in range(1, len(segments) - 1):
            if not segments[index].isdigit():
                continue
            prefix = ".".join(segments[:index])
            suffix = ".".join(segments[index + 1 :]).lower()
            candidate = candidates.setdefault(prefix, {"indices": set(), "components": False})
            candidate["indices"].add(int(segments[index]))
            if any(word in suffix for word in ("attn", "attention", "mlp", "feed_forward", "layernorm", "norm")):
                candidate["components"] = True

    result = FamilyResult()
    for prefix, candidate in candidates.items():
        indices = candidate["indices"]
        inferred = max(indices) + 1
        score = len(indices)
        lowered = prefix.lower()
        if expected_count and inferred == expected_count:
            score += 10_000
        if lowered.endswith((".layers", ".layer", ".blocks", ".block", ".h")):
            score += 1_500
        if any(word in lowered for word in ("language", "text", "llm", "decoder", "transformer")):
            score += 1_000
        if candidate["components"]:
            score += 750
        if 0 not in indices:
            score -= 2_000
        if any(word in lowered for word in ("expert", "moe")):
            score -= 4_000
        if any(word in lowered for word in ("vision", "visual", "image", "audio", "speech")):
            score -= 1_000
        if not result.found or score > result.score:
            result = FamilyResult(True, prefix, inferred, len(indices), score)
    return result


def _positive(config_a: dict, config_b: dict, keys: Iterable[str]) -> tuple[int | None, str | None]:
    for config in (config_a, config_b):
        for key in keys:
            value = config.get(key)
            if isinstance(value, (int, float)) and int(value) > 0:
                return int(value), key
    return None, None


def _configured_string(config_a: dict, config_b: dict, keys: Iterable[str], fallback: str = "") -> str:
    for config in (config_a, config_b):
        for key in keys:
            value = config.get(key)
            if isinstance(value, str):
                return value
    return fallback


def _layer_names(names: Iterable[str], prefix: str, layer: int) -> list[str]:
    needle = f"{prefix}.{layer}."
    return [name for name in names if name.startswith(needle)]


def _shape_dimension(tensors: dict, name: str, dimension: int, fallback: int = 0) -> int:
    shape = tensors.get(name, {}).get("shape")
    if isinstance(shape, list) and 0 <= dimension < len(shape) and isinstance(shape[dimension], (int, float)):
        return int(shape[dimension])
    return fallback


def _finding(feature: str, value: str, basis: str, confidence: str, evidence: str) -> dict:
    return {"feature": feature, "value": value, "basis": basis, "confidence": confidence, "evidence": evidence}


def infer_architecture(
    config: dict,
    text_config: dict,
    tensors: dict,
    names: list[str],
    family: FamilyResult,
    layers: int,
    hidden: str,
    vocab: str,
) -> Architecture:
    findings: list[dict] = []
    layer_zero = _layer_names(names, family.prefix, 0)
    attention_heads, attention_key = _positive(text_config, config, ("num_attention_heads", "n_head", "num_heads"))
    kv_heads, kv_key = _positive(text_config, config, ("num_key_value_heads", "num_kv_heads", "n_head_kv"))
    declared_layers, layer_key = _positive(text_config, config, LAYER_COUNT_KEYS)
    declared_hidden, hidden_key = _positive(text_config, config, ("hidden_size", "n_embd", "d_model"))
    declared_vocab, vocab_key = _positive(text_config, config, ("vocab_size", "n_vocab"))

    q_name = next((name for name in layer_zero if name.endswith(("q_proj.weight", "query.weight"))), "")
    k_name = next((name for name in layer_zero if name.endswith(("k_proj.weight", "key.weight"))), "")
    if kv_heads is None and attention_heads and q_name and k_name:
        q_width = _shape_dimension(tensors, q_name, 0)
        k_width = _shape_dimension(tensors, k_name, 0)
        head_width = q_width // attention_heads if attention_heads else 0
        if head_width and k_width % head_width == 0:
            kv_heads = k_width // head_width
            kv_key = f"{k_name} shape"

    findings.append(_finding(
        "Decoder depth",
        f"{layers} layers",
        "declared" if declared_layers else "inferred",
        "high",
        f"config.{layer_key}" if declared_layers else f"{family.prefix}.0…{layers - 1} tensor paths",
    ))
    findings.append(_finding(
        "Hidden width",
        hidden,
        "declared" if declared_hidden else "inferred",
        "high" if declared_hidden else "medium",
        f"config.{hidden_key}" if declared_hidden else "token-embedding tensor shape",
    ))
    findings.append(_finding(
        "Vocabulary",
        f"{vocab} tokens",
        "declared" if declared_vocab else "inferred",
        "high" if declared_vocab else "medium",
        f"config.{vocab_key}" if declared_vocab else "token-embedding tensor shape",
    ))

    attention = "Causal multi-head attention"
    basis, confidence, evidence = "inferred", "medium", "attention projection tensor names"
    if attention_heads:
        attention = f"Multi-head attention · {attention_heads} Q heads"
        basis, confidence, evidence = "declared", "high", f"config.{attention_key}"
    if attention_heads and kv_heads:
        if kv_heads == 1:
            attention = f"Multi-query attention · {attention_heads} Q heads / 1 KV head"
        elif kv_heads < attention_heads:
            attention = f"Grouped-query attention · {attention_heads} Q heads / {kv_heads} KV heads"
        else:
            attention = f"Multi-head attention · {attention_heads} heads"
        if kv_key and "shape" in kv_key:
            basis, evidence = "inferred", f"config.{attention_key} + Q/K projection shapes"
        else:
            basis, evidence = "declared", f"config.{attention_key} + config.{kv_key}"
    findings.append(_finding("Attention", attention, basis, confidence, evidence))

    separate_qkv = all(any(name.endswith(suffix) for name in layer_zero) for suffix in ("q_proj.weight", "k_proj.weight", "v_proj.weight"))
    fused_qkv = any(name.endswith(("c_attn.weight", "query_key_value.weight", "qkv_proj.weight")) for name in layer_zero)
    projection = "Separate Q, K, and V projections" if separate_qkv else "Fused QKV projection" if fused_qkv else "Attention projections"
    findings.append(_finding("Projection layout", projection, "inferred", "high" if separate_qkv or fused_qkv else "low", "layer-0 tensor names"))

    model_type = _configured_string(text_config, config, ("model_type",), "transformer").lower()
    rope_keys = (
        "rope_theta", "rope_scaling", "rotary_dim", "rotary_pct", "rotary_emb_base", "partial_rotary_factor"
    )
    has_rope = any(text_config.get(key) is not None for key in rope_keys) or any(
        word in name.lower() for name in names for word in ("rotary_emb", "inv_freq")
    )
    has_alibi = bool(text_config.get("alibi", config.get("alibi", False))) or any("alibi" in name.lower() for name in names)
    has_relative_bias = any(key in text_config for key in ("relative_attention_num_buckets", "relative_attention_max_distance")) or any(
        "relative_attention_bias" in name.lower() for name in names
    )
    absolute_position_name = next((
        name for name in names
        if name.lower().endswith(("wpe.weight", "embed_positions.weight", "position_embeddings.weight"))
    ), "")
    if has_alibi and not has_rope:
        position_kind = "alibi"
        position, position_basis, position_confidence, position_evidence = "ALiBi attention-logit bias", "declared", "high", "ALiBi config/tensor evidence"
    elif has_relative_bias and not has_rope:
        position_kind = "relative-bias"
        position, position_basis, position_confidence, position_evidence = "Learned relative attention bias", "declared" if any(key in text_config for key in ("relative_attention_num_buckets", "relative_attention_max_distance")) else "inferred", "high", "relative-attention config/tensor evidence"
    elif absolute_position_name and not has_rope:
        position_kind = "absolute"
        position, position_basis, position_confidence, position_evidence = "Learned absolute position embeddings", "checkpoint", "high", absolute_position_name
    elif bool(text_config.get("sinusoidal_pos_embds", False)) and not has_rope:
        position_kind = "absolute-fixed"
        position, position_basis, position_confidence, position_evidence = "Fixed sinusoidal position encoding", "declared", "high", "config.sinusoidal_pos_embds"
    elif has_rope:
        position_kind = "rotary"
        position = "Rotary position embeddings (RoPE)"
        position_basis = "declared" if any(text_config.get(key) is not None for key in rope_keys) else "inferred"
        position_confidence = "high"
        position_evidence = "rotary config fields" if position_basis == "declared" else "rotary tensor names"
    else:
        position_kind = "unresolved"
        position, position_basis, position_confidence, position_evidence = "Position mechanism unresolved", "unresolved", "low", "no position config field or checkpoint tensor matched"
    findings.append(_finding("Position encoding", position, position_basis, position_confidence, position_evidence))

    lowered_zero = [name.lower() for name in layer_zero]
    has_norm_bias = any("layernorm" in name or "layer_norm" in name or ".ln_" in name for name in lowered_zero) and any(name.endswith(".bias") for name in lowered_zero)
    rms_config = "rms_norm_eps" in text_config
    layer_norm_config = any(key in text_config for key in ("layer_norm_eps", "layer_norm_epsilon"))
    norm = "RMSNorm" if rms_config or (not has_norm_bias and not layer_norm_config) else "LayerNorm"
    if rms_config:
        norm_basis, norm_confidence, norm_evidence = "declared", "high", "config.rms_norm_eps"
    elif layer_norm_config:
        norm_basis, norm_confidence, norm_evidence = "declared", "high", "layer-norm config field"
    elif has_norm_bias:
        norm_basis, norm_confidence, norm_evidence = "inferred", "high", "normalization bias tensors"
    else:
        norm_basis, norm_confidence, norm_evidence = "inferred", "medium", "weight-only normalization tensors"
    findings.append(_finding("Normalization", norm, norm_basis, norm_confidence, norm_evidence))

    activation = _configured_string(text_config, config, ("hidden_act", "activation_function"))
    has_gate = any(name.endswith(("gate_proj.weight", "w1.weight")) for name in layer_zero)
    has_experts = any(".experts." in name or ".expert." in name for name in names)
    experts, experts_key = _positive(text_config, config, ("num_local_experts", "num_experts", "n_routed_experts"))
    if has_experts or experts:
        feed_forward = "Sparse mixture-of-experts" + (f" · {experts} experts" if experts else "")
    elif has_gate:
        feed_forward = f"{activation} gated MLP" if activation else "SwiGLU-like gated MLP"
    else:
        feed_forward = f"{activation} feed-forward network" if activation else "dense feed-forward network"
    if experts:
        ff_evidence = f"config.{experts_key}"
    elif has_experts:
        ff_evidence = "expert tensor paths"
    elif has_gate:
        ff_evidence = "gate/up/down projection tensors"
    else:
        ff_evidence = "feed-forward tensor names"
    findings.append(_finding(
        "Feed-forward",
        feed_forward,
        "declared" if activation or experts else "inferred",
        "high" if has_gate or has_experts or activation or experts else "medium",
        ff_evidence,
    ))

    tied_declared = text_config.get("tie_word_embeddings", config.get("tie_word_embeddings"))
    tied = bool(tied_declared) if isinstance(tied_declared, bool) else not any(
        "lm_head.weight" in name or "embed_out.weight" in name for name in names
    )
    findings.append(_finding(
        "Output embedding",
        "Likely tied to token embeddings" if tied else "Separate LM-head weights",
        "declared" if isinstance(tied_declared, bool) else "inferred",
        "high" if isinstance(tied_declared, bool) else "medium",
        "config.tie_word_embeddings" if isinstance(tied_declared, bool) else "no dedicated LM-head tensor" if tied else "dedicated LM-head tensor",
    ))
    explicit_parallel = text_config.get("use_parallel_residual", config.get("use_parallel_residual"))
    parallel_attention = text_config.get("parallel_attn", config.get("parallel_attn"))
    new_decoder_architecture = bool(text_config.get("new_decoder_architecture", config.get("new_decoder_architecture", False)))
    do_norm_before = text_config.get("do_layer_norm_before", config.get("do_layer_norm_before"))
    if model_type in {"gptj", "gpt_j"}:
        residual_topology = "parallel-shared-norm"
        norm_order = "pre"
        topology_confidence, topology_evidence = "high", f"Transformers {model_type} block contract"
    elif explicit_parallel is True:
        residual_topology = "parallel-dual-norm"
        norm_order = "pre"
        topology_confidence, topology_evidence = "high", "config.use_parallel_residual=true"
    elif explicit_parallel is False:
        residual_topology = "sequential-pre-norm"
        norm_order = "pre"
        topology_confidence, topology_evidence = "high", "config.use_parallel_residual=false"
    elif parallel_attention is True or new_decoder_architecture:
        configured_parallel_norms = text_config.get("num_ln_in_parallel_attn", config.get("num_ln_in_parallel_attn", 2))
        parallel_norms = int(configured_parallel_norms) if isinstance(configured_parallel_norms, (int, float)) else 2
        dual_norm = new_decoder_architecture and parallel_norms == 2
        residual_topology = "parallel-dual-norm" if dual_norm else "parallel-shared-norm"
        norm_order = "pre"
        topology_confidence, topology_evidence = "high", "Falcon parallel-attention configuration"
    elif do_norm_before is False:
        residual_topology = "sequential-post-norm"
        norm_order = "post"
        topology_confidence, topology_evidence = "high", "config.do_layer_norm_before=false"
    else:
        residual_topology = "sequential-pre-norm"
        norm_order = "pre"
        known_sequential = model_type in {
            "gpt2", "llama", "mistral", "mixtral", "qwen2", "qwen3", "gemma", "gemma2", "phi", "phi3",
            "bloom", "mpt", "opt", "t5", "mt5", "bart", "pegasus", "deepseek_v2", "deepseek_v3",
        }
        topology_confidence = "high" if do_norm_before is True else "medium" if known_sequential or layer_zero else "low"
        topology_evidence = "config.do_layer_norm_before=true" if do_norm_before is True else f"{model_type} decoder contract and layer tensor paths" if known_sequential else "canonical decoder scaffold; implementation not resolved"
    topology_value = {
        "parallel-dual-norm": "Parallel attention and MLP residual branches · separate pre-norms",
        "parallel-shared-norm": "Parallel attention and MLP residual branches · shared pre-norm",
        "sequential-post-norm": "Sequential residual branches · post-normalization",
        "sequential-pre-norm": "Sequential attention then MLP residual branches · pre-normalization",
    }[residual_topology]
    findings.append(_finding(
        "Residual topology",
        topology_value,
        "declared" if topology_evidence.startswith("config.") or "configuration" in topology_evidence else "inferred",
        topology_confidence,
        topology_evidence,
    ))
    context, context_key = _positive(text_config, config, ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"))
    if context:
        findings.append(_finding("Context window", f"{context} tokens", "declared", "high", f"config.{context_key}"))
    head_dim, _head_dim_key = _positive(text_config, config, ("head_dim", "attention_head_dim", "kv_channels"))
    if head_dim is None and attention_heads and str(hidden).isdigit() and int(hidden) % attention_heads == 0:
        head_dim = int(hidden) // attention_heads
    return Architecture(
        attention=attention,
        position=position,
        position_kind=position_kind,
        norm=norm,
        feed_forward=feed_forward,
        residual_topology=residual_topology,
        norm_order=norm_order,
        topology_confidence=topology_confidence,
        topology_evidence=topology_evidence,
        query_heads=attention_heads,
        kv_heads=kv_heads or attention_heads,
        head_dim=head_dim,
        output_tied=tied,
        findings=findings,
    )


def dtype_bytes(dtype: str) -> int:
    key = dtype.lower().removeprefix("torch.")
    if key.startswith(("float8", "fp8")):
        return 1
    if key in {"bool", "int8", "uint8"}:
        return 1
    if key in {"float16", "half", "bfloat16", "int16", "uint16"}:
        return 2
    if key in {"float64", "double", "int64", "uint64"}:
        return 8
    return 4


def shape_product(shape: object) -> int:
    if not isinstance(shape, list) or any(not isinstance(value, (int, float)) or int(value) < 0 for value in shape):
        return 0
    return prod(int(value) for value in shape)


def _tensor_byte_size(source: dict, count: int, dtype: str) -> tuple[int, str]:
    safetensors = source.get("safetensors") if isinstance(source.get("safetensors"), dict) else {}
    byte_size = safetensors.get("byteSize")
    if isinstance(byte_size, (int, float)) and int(byte_size) >= 0:
        return int(byte_size), "Safetensors data offsets"
    offsets = safetensors.get("dataOffsets")
    if isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(value, (int, float)) for value in offsets):
        start, end = (int(value) for value in offsets)
        if 0 <= start <= end:
            return end - start, "Safetensors data offsets"
    return count * dtype_bytes(dtype), "Element count × normalized dtype width"


def circuit_family(identifier: str, name: str, operation: str) -> str:
    signature = f"{identifier} {name} {operation}".lower()
    if "kv_cache" in signature or "dynamiccache" in signature:
        return "circuit-memory"
    if any(word in signature for word in ("residual", "append generated", "next_step")):
        return "circuit-residual"
    if any(word in signature for word in ("attention", "qkv", "q / k / v", "position encoding", "rope")):
        return "circuit-attention"
    if any(word in signature for word in ("feed-forward", "mlp", "expert")):
        return "circuit-mlp"
    if "norm" in signature:
        return "circuit-norm"
    return "circuit-general"


def _parent_path(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def _first_parent_or(names: list[str], fallback: str) -> str:
    return _parent_path(names[0]) if names else fallback


def _natural_path_key(value: str) -> tuple:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", value.lower())
    )


def _derived_tensor_role(name: str) -> str:
    raw_parts = name.split(".")
    parts = [part for part in raw_parts if part and not part.isdigit()]
    if parts and parts[-1].lower() in {"weight", "bias", "scale", "scales", "zeros"}:
        parts.pop()
    module = parts[-1] if parts else "model"
    label = re.sub(r"[_\-]+", " ", module).strip()
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label).lower()
    expert = next(
        (part for index, part in enumerate(raw_parts) if index and raw_parts[index - 1].lower() in {"expert", "experts"} and part.isdigit()),
        None,
    )
    prefix = f"Expert {expert} " if expert is not None else ""
    return f"{prefix}{label or 'model'} parameter".strip().capitalize()


def _tensor_semantics(name: str) -> tuple[int, str, str, str, str]:
    lowered = name.lower()
    parameter_kind = (
        "Bias" if lowered.endswith(".bias")
        else "Scale" if "norm" in lowered and lowered.endswith(".weight")
        else "Weight" if lowered.endswith(".weight")
        else "Parameter"
    )
    roles = (
        (10, "Token embedding", ("embed_tokens", "word_embeddings", ".wte.")),
        (11, "Absolute position embedding", ("embed_positions", "position_embeddings", ".wpe.")),
        (20, "Input normalization", ("input_layernorm", "attention_norm", ".ln_1.")),
        (30, "Fused QKV projection", ("query_key_value", "qkv_proj", ".c_attn.", "in_proj_weight", "wqkv")),
        (31, "Query projection", ("q_proj", ".query.")),
        (32, "Query normalization", ("q_norm", "query_layernorm")),
        (33, "Key projection", ("k_proj", ".key.")),
        (34, "Key normalization", ("k_norm", "key_layernorm")),
        (35, "Value projection", ("v_proj", ".value.")),
        (36, "Rotary frequency state", ("rotary_emb", "inv_freq")),
        (37, "Relative attention bias", ("relative_attention_bias",)),
        (38, "Causal attention mask", ("masked_bias", "causal_mask", ".attn.bias")),
        (40, "Attention output projection", ("self_attn.o_proj", "attn.c_proj", "attention.dense", "self_attention.dense", "attention.out_proj", "self_attn.out_proj")),
        (50, "Post-attention normalization", ("post_attention_layernorm", "ffn_norm", ".ln_2.")),
        (60, "Router projection", ("router", "block_sparse_moe.gate", "mlp.gate.weight")),
        (70, "Gate projection", ("gate_proj", ".w1.")),
        (80, "MLP expansion projection", ("up_proj", ".w3.", ".c_fc.", "dense_h_to_4h", ".fc1.", ".wi.", ".wi_0.", ".wi_1.", "proj_in")),
        (90, "MLP contraction projection", ("down_proj", ".w2.", "dense_4h_to_h", ".fc2.", ".mlp.c_proj.", ".wo.", "proj_out")),
        (100, "Final normalization", ("final_layernorm", "ln_f", ".norm.")),
        (110, "Vocabulary projection", ("lm_head", "embed_out")),
    )
    match = next(
        ((rank, label) for rank, label, signatures in roles if any(signature in lowered for signature in signatures)),
        None,
    )
    if match:
        return match[0], match[1], parameter_kind, "recognized", "tensor-path alias"
    return 1000, _derived_tensor_role(name), parameter_kind, "path-derived", "module path fallback"


def _tensor_storage_role(name: str, shape: list) -> tuple[str, str]:
    lowered = name.lower()
    known_buffer = lowered.endswith((".masked_bias", ".causal_mask", ".position_ids", ".inv_freq"))
    causal_mask = lowered.endswith(".attn.bias") and len(shape) >= 3
    if known_buffer or causal_mask:
        return "buffer", "recognized persistent model state"
    return "parameter-like", "Checkpoint metadata does not encode requires_grad; classified from the tensor path"


def _tensor_semantic_key(name: str) -> tuple:
    role_rank, _, parameter_kind, _, _ = _tensor_semantics(name)
    kind_rank = {"Weight": 0, "Scale": 0, "Bias": 1, "Parameter": 2}[parameter_kind]
    base_path = name.rsplit(".", 1)[0] if "." in name else name
    return role_rank, _natural_path_key(base_path), kind_rank, _natural_path_key(name)


def _tensor_records(names: list[str], tensors: dict, used: set[str]) -> tuple[list[dict], int, int]:
    records: list[dict] = []
    total_parameters = 0
    total_bytes = 0
    ordered_names = sorted(dict.fromkeys(names), key=_tensor_semantic_key)
    for operation_index, name in enumerate(ordered_names):
        source = tensors.get(name)
        if not isinstance(source, dict):
            continue
        shape_known = isinstance(source.get("shape"), list)
        shape = source.get("shape") if shape_known else []
        dtype = source.get("dtype") if isinstance(source.get("dtype"), str) else "float32"
        count = shape_product(shape) if shape_known else 0
        byte_count, byte_basis = _tensor_byte_size(source, count, dtype) if shape_known else (0, "Unavailable in checkpoint manifest")
        metadata = {}
        if "safetensors" in source:
            metadata["safetensors"] = source["safetensors"]
        if "checkpoint" in source:
            metadata["checkpoint"] = source["checkpoint"]
        _, semantic_role, parameter_kind, semantic_confidence, semantic_source = _tensor_semantics(name)
        storage_role, storage_basis = _tensor_storage_role(name, shape)
        if storage_role == "buffer":
            parameter_kind = "Buffer"
        safetensors = metadata.get("safetensors") if isinstance(metadata.get("safetensors"), dict) else {}
        checkpoint = metadata.get("checkpoint") if isinstance(metadata.get("checkpoint"), dict) else safetensors
        records.append({
            "id": f"parameter:{name}",
            "name": name,
            "kind": "buffer" if storage_role == "buffer" else "parameter-like",
            "path": name,
            "description": "",
            "shape": shape,
            "shapeKnown": shape_known,
            "dtype": dtype,
            "trainable": None,
            "storageRole": storage_role,
            "storageBasis": storage_basis,
            "count": count,
            "metadata": metadata,
            "directParameters": count,
            "totalParameters": count,
            "directElements": count,
            "totalElements": count,
            "trainableParameters": None,
            "totalBytes": byte_count,
            "byteBasis": byte_basis,
            "children": [],
            "order": {
                "operationIndex": operation_index,
                "operationCount": len(ordered_names),
                "semanticRole": semantic_role,
                "parameterKind": parameter_kind,
                "basis": "forward-function",
                "automatic": True,
                "semanticConfidence": semantic_confidence,
                "semanticSource": semantic_source,
                "checkpointIndex": checkpoint.get("checkpointIndex"),
                "shardIndex": checkpoint.get("shardIndex"),
                "fileTensorIndex": checkpoint.get("fileTensorIndex"),
            },
        })
        used.add(name)
        total_parameters += count
        total_bytes += byte_count
    return records, total_parameters, total_bytes


def _add_node(
    nodes: list[dict],
    tensors: dict,
    used: set[str],
    identifier: str,
    name: str,
    kind: str,
    operation: str,
    path: str,
    shape: str,
    formula: str,
    description: str,
    group: str,
    subgroup: str,
    column: int,
    row: int,
    tensor_names: list[str],
    metadata: object = None,
) -> None:
    tensor_records, parameters, byte_count = _tensor_records(tensor_names, tensors, used)
    search_text = f"{identifier} {name} {operation} {path} {description} {formula}"
    for tensor in tensor_records:
        search_text += f" {tensor['name']} {tensor['dtype']}"
    nodes.append({
        "id": identifier,
        "name": name,
        "kind": kind,
        "path": path,
        "type": operation,
        "description": description,
        "formula": formula,
        "shape": shape or None,
        "repeat": 1,
        "group": group or None,
        "subgroup": subgroup or None,
        "circuitFamily": circuit_family(identifier, name, operation),
        "metadata": metadata,
        "position": {"column": column, "row": row},
        "layout": {"x": column * COLUMN_STRIDE, "y": row * ROW_STRIDE, "depth": column},
        "tensors": tensor_records,
        "children": [],
        "directParameters": parameters,
        "totalParameters": parameters,
        "directElements": parameters,
        "totalElements": parameters,
        "trainableParameters": None,
        "totalBytes": byte_count,
        "searchText": search_text.lower(),
    })


def _edge_circuit_kind(source: str, target: str, label: str, kind: str, feedback: bool) -> str:
    signature = f"{source} {target} {label}".lower()
    if feedback or any(word in signature for word in ("kv_cache", "past k", "write k")):
        return "memory"
    if kind in {"residual-stream", "residual"} or "residual" in signature:
        return "residual"
    if any(word in signature for word in ("attention", "qkv", "rope")):
        return "attention"
    if any(word in signature for word in ("mlp", "feed-forward")):
        return "mlp"
    return "signal"


def _add_edge(edges: list[dict], source: str, target: str, label: str, kind: str, feedback: bool = False) -> None:
    circuit_kind = _edge_circuit_kind(source, target, label, kind, feedback)
    edges.append({
        "from": source,
        "to": target,
        "label": label,
        "feedback": feedback,
        "kind": kind,
        "circuitKind": circuit_kind,
        "arrowhead": "residual-rail" if kind == "residual-stream" else circuit_kind,
    })


def _add_group(groups: list[dict], layer: int, model_type: str, predicted: bool, architecture: Architecture) -> None:
    attention_column = layer * 2 + 1
    mlp_column = attention_column + 1
    min_x = attention_column * COLUMN_STRIDE - 18
    max_x = mlp_column * COLUMN_STRIDE + NODE_WIDTH + 18 + TENSOR_STACK_MAX_EXTENT
    max_y = 8 * ROW_STRIDE + NODE_HEIGHT + 17 + TENSOR_STACK_MAX_EXTENT
    subgroup_y = 2 * ROW_STRIDE - 22
    subgroup_max_y = 8 * ROW_STRIDE + NODE_HEIGHT + 10 + TENSOR_STACK_MAX_EXTENT
    qualifier = "Config-declared scaffold" if predicted else "Checkpoint-mapped"
    suffix = "; no indexed tensors were observed for this layer" if predicted else ""
    topology_label = architecture.residual_topology.replace("-", " ")
    groups.append({
        "id": f"layer_{layer}",
        "name": f"Transformer block {layer}",
        "label": f"Layer {layer}",
        "repeat": 1,
        "description": f"{qualifier} {model_type} decoder block using {topology_label} topology ({architecture.topology_confidence} confidence: {architecture.topology_evidence}){suffix}",
        "evidence": {"kind": "structural", "confidence": architecture.topology_confidence, "basis": architecture.topology_evidence, "checkpointMapped": not predicted},
        "bounds": {"x": min_x, "y": -30, "width": max_x - min_x, "height": max_y + 30},
        "subgroups": [
            {"kind": "attention", "label": "Attention", "bounds": {"x": attention_column * COLUMN_STRIDE - 10, "y": subgroup_y, "width": NODE_WIDTH + 20 + TENSOR_STACK_MAX_EXTENT, "height": subgroup_max_y - subgroup_y}},
            {"kind": "mlp", "label": "MLP / feed-forward", "bounds": {"x": mlp_column * COLUMN_STRIDE - 10, "y": subgroup_y, "width": NODE_WIDTH + 20 + TENSOR_STACK_MAX_EXTENT, "height": subgroup_max_y - subgroup_y}},
        ],
    })


def _bucket_layer(names: list[str]) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    buckets = ([], [], [], [], [], [], [], [])
    for original in names:
        name = original.lower()
        if any(word in name for word in ("input_layernorm", "self_attn_layer_norm", "ln_1.", "attention_norm", "pre_attention_layernorm", "ln_attn", "norm_1")):
            buckets[0].append(original)
        elif any(word in name for word in ("q_proj", "k_proj", "v_proj", "c_attn", "query_key_value", "qkv_proj", "in_proj_weight", "wqkv")):
            buckets[1].append(original)
        elif any(word in name for word in ("self_attn.o_proj", "attn.c_proj", "attention.dense", "self_attention.dense", "attention.out_proj", "attn.out_proj", "self_attn.out_proj")):
            buckets[2].append(original)
        elif any(word in name for word in ("post_attention_layernorm", "final_layer_norm", "final_layernorm", "ln_2.", "ffn_norm", "pre_feedforward_layernorm", "ln_mlp", "norm_2")):
            buckets[3].append(original)
        elif any(word in name for word in (
            ".mlp.", ".ffn.", "feed_forward", "feedforward", "gate_proj", "up_proj", "down_proj",
            "block_sparse_moe", ".experts.", ".c_fc.", "dense_h_to_4h", "dense_4h_to_h", ".fc1.", ".fc2.", ".fc_in.", ".fc_out.",
        )):
            buckets[4].append(original)
        elif any(word in name for word in ("rotary_emb", "inv_freq", "relative_attention_bias", "alibi")):
            buckets[5].append(original)
        elif any(word in name for word in ("masked_bias", "causal_mask")) or (name.endswith(".attn.bias") and "c_attn.bias" not in name):
            buckets[6].append(original)
        else:
            buckets[7].append(original)
    return buckets


def _enrich_edges(nodes: list[dict], edges: list[dict]) -> None:
    by_id = {node["id"]: node for node in nodes}
    for edge in edges:
        source = by_id[edge["from"]]["layout"]
        target = by_id[edge["to"]]["layout"]
        sx, sy, tx, ty = source["x"], source["y"], target["x"], target["y"]
        if edge["feedback"]:
            x1, y1 = sx + NODE_WIDTH, sy + NODE_HEIGHT / 2
            x2, y2 = tx + NODE_WIDTH, ty + NODE_HEIGHT / 2
            loop_x = max(x1, x2) + 48
            path = f"M {x1:.3f} {y1:.3f} C {loop_x:.3f} {y1:.3f}, {loop_x:.3f} {y2:.3f}, {x2 + 24:.3f} {y2:.3f} L {x2:.3f} {y2:.3f}"
            label = {"x": loop_x + 4, "y": (y1 + y2) / 2}
        elif abs(tx - sx) >= abs(ty - sy):
            if tx >= sx:
                x1, y1, x2, y2 = sx + NODE_WIDTH, sy + NODE_HEIGHT / 2, tx, ty + NODE_HEIGHT / 2
            else:
                x1, y1, x2, y2 = sx, sy + NODE_HEIGHT / 2, tx + NODE_WIDTH, ty + NODE_HEIGHT / 2
            direction = 1 if x2 >= x1 else -1
            bend = max(28, abs(x2 - x1) * 0.42)
            path = f"M {x1:.3f} {y1:.3f} C {x1 + bend * direction:.3f} {y1:.3f}, {x2 - bend * direction:.3f} {y2:.3f}, {x2:.3f} {y2:.3f}"
            label = {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2 - 5}
        else:
            if ty >= sy:
                x1, y1, x2, y2 = sx + NODE_WIDTH / 2, sy + NODE_HEIGHT, tx + NODE_WIDTH / 2, ty
            else:
                x1, y1, x2, y2 = sx + NODE_WIDTH / 2, sy, tx + NODE_WIDTH / 2, ty + NODE_HEIGHT
            direction = 1 if y2 >= y1 else -1
            bend = max(25, abs(y2 - y1) * 0.42)
            path = f"M {x1:.3f} {y1:.3f} C {x1:.3f} {y1 + bend * direction:.3f}, {x2:.3f} {y2 - bend * direction:.3f}, {x2:.3f} {y2:.3f}"
            label = {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2 - 5}
        edge["path"] = path
        edge["labelPosition"] = label


def _relevant_config(node: dict, text_config: dict, config: dict) -> dict:
    keys = ["model_type", "architectures", "transformers_version", "torch_dtype", "dtype", "hidden_size", "num_hidden_layers", "use_cache", "tie_word_embeddings", "vocab_size", "use_parallel_residual", "parallel_attn", "new_decoder_architecture", "do_layer_norm_before"]
    family_keys = {
        "circuit-attention": ["num_attention_heads", "num_key_value_heads", "head_dim", "attention_bias", "attention_dropout", "sliding_window", "max_window_layers", "rope_theta", "rope_scaling", "max_position_embeddings"],
        "circuit-mlp": ["intermediate_size", "hidden_act", "hidden_activation", "num_experts", "num_local_experts", "num_experts_per_tok", "moe_intermediate_size"],
        "circuit-norm": ["rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon"],
        "circuit-memory": ["sliding_window", "max_position_embeddings"],
        "circuit-residual": ["residual_dropout"],
    }
    keys.extend(family_keys.get(node["circuitFamily"], []))
    signature = f"{node['id']} {node['name']} {node['type']}".lower()
    if any(word in signature for word in ("token", "embed", "vocab")):
        keys.extend(("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"))
    elif any(word in signature for word in ("sample", "distribution", "generation", "next_step")):
        keys.extend(("bos_token_id", "eos_token_id", "pad_token_id"))
    result = {}
    for key in keys:
        if key in result:
            continue
        if key in text_config:
            result[key] = text_config[key]
        elif key in config:
            result[key] = config[key]
    return result


def _relevant_findings(node: dict, findings: list[dict]) -> list[dict]:
    family_words = {
        "circuit-attention": ("attention", "projection", "position", "context"),
        "circuit-memory": ("attention", "projection", "position", "context"),
        "circuit-mlp": ("feed-forward",),
        "circuit-norm": ("normalization",),
        "circuit-residual": ("decoder depth", "hidden width", "residual topology", "normalization"),
    }
    words = family_words.get(node["circuitFamily"], ("decoder depth", "hidden width", "vocabulary", "output embedding"))
    return [finding for finding in findings if any(word in finding["feature"].lower() for word in words)]


def _relevant_files(node: dict, hub: dict) -> list[dict]:
    exact = {
        tensor.get("metadata", {}).get("safetensors", {}).get("file")
        for tensor in node["tensors"]
    } - {None}
    signature = f"{node['id']} {node['name']} {node['type']}".lower()
    selected = []
    for file in hub.get("siblings", []):
        if not isinstance(file, dict):
            continue
        name = str(file.get("rfilename", ""))
        lowered = name.lower()
        matches = name in exact
        if not matches and any(word in signature for word in ("token", "embed", "vocab")):
            matches = any(word in lowered for word in ("tokenizer", "vocab", "merges", "special_tokens", "special.tokens", "chat_template"))
        if not matches and any(word in signature for word in ("sample", "distribution", "generation", "next_step")):
            matches = any(word in lowered for word in ("generation_config", "tokenizer_config", "special_tokens", "special.tokens"))
        if not matches and any(word in signature for word in ("checkpoint", "safetensor")):
            matches = "safetensors" in lowered
        if not matches and any(word in signature for word in ("architecture", "norm", "attention", "mlp", "residual", "position", "rope", "q / k / v")):
            matches = lowered == "config.json"
        if matches:
            selected.append(file)
    return selected


def _enrich_nodes(nodes: list[dict], edges: list[dict], groups: list[dict], config: dict, text_config: dict, hub: dict, findings: list[dict]) -> None:
    by_id = {node["id"]: node for node in nodes}
    groups_by_id = {group["id"]: group for group in groups}
    connections: dict[str, list[dict]] = {node["id"]: [] for node in nodes}
    for edge in edges:
        for identifier, direction, counterpart_id in ((edge["to"], "in", edge["from"]), (edge["from"], "out", edge["to"])):
            counterpart = by_id[counterpart_id]
            connections[identifier].append({
                "direction": direction,
                "counterpartId": counterpart_id,
                "counterpartName": counterpart["name"],
                "label": edge["label"],
                "kind": edge["kind"],
                "feedback": edge["feedback"],
            })
    for node in nodes:
        tensor_dtypes = list(dict.fromkeys(tensor["dtype"] for tensor in node["tensors"]))
        node["inspector"] = {
            "group": groups_by_id.get(node["group"]),
            "connections": connections[node["id"]],
            "config": _relevant_config(node, text_config, config),
            "files": _relevant_files(node, hub),
            "findings": _relevant_findings(node, findings),
            "tensorSummary": {
                "count": len(node["tensors"]),
                "elements": sum(tensor["count"] for tensor in node["tensors"]),
                "bytes": sum(tensor["totalBytes"] for tensor in node["tensors"]),
                "dtypes": tensor_dtypes,
                "ordering": {
                    "automatic": True,
                    "recognized": sum(tensor["order"]["semanticConfidence"] == "recognized" for tensor in node["tensors"]),
                    "pathDerived": sum(tensor["order"]["semanticConfidence"] == "path-derived" for tensor in node["tensors"]),
                    "checkpointLocated": sum(isinstance(tensor["order"].get("checkpointIndex"), int) for tensor in node["tensors"]),
                },
            },
        }


def _build_residual_ledger(layers: int, hidden: str, architecture: Architecture, has_position_input: bool) -> dict:
    """Describe every additive write to the residual stream without inventing run data."""
    initial_equation = "h0 = Etok[x] + Epos[pos]" if has_position_input else "h0 = E[x]"
    initial_writes = [{
        "kind": "embedding",
        "label": "Token embedding",
        "symbol": "Etok[x]" if has_position_input else "E[x]",
        "sourceNodeId": "embedding",
        "targetNodeId": "residual_0",
        "value": None,
        "directLogitAttribution": None,
    }]
    if has_position_input:
        initial_writes.append({
            "kind": "position",
            "label": "Absolute position embedding",
            "symbol": "Epos[pos]",
            "sourceNodeId": "position_embedding",
            "targetNodeId": "residual_0",
            "value": None,
            "directLogitAttribution": None,
        })
    states = [{
        "id": "residual_0",
        "label": "Residual stream h0",
        "stage": "embedding",
        "layer": None,
        "inputState": None,
        "state": "h0",
        "shape": f"[B, T, {hidden}]",
        "equation": initial_equation,
        "writes": initial_writes,
        "activationNorm": None,
        "directLogitAttribution": None,
    }]
    for layer in range(layers):
        if architecture.residual_topology == "sequential-post-norm":
            equation = f"h{layer + 1} = Norm(Norm(h{layer} + a{layer}) + m{layer})"
            attention_target = f"l{layer}_attn_residual"
        else:
            equation = f"h{layer + 1} = h{layer} + a{layer} + m{layer}"
            attention_target = f"l{layer}_attn_residual" if architecture.residual_topology == "sequential-pre-norm" else f"l{layer}_mlp_residual"
        states.append({
            "id": f"l{layer}_mlp_residual",
            "label": f"Residual stream h{layer + 1}",
            "stage": "block-output",
            "layer": layer,
            "inputState": f"h{layer}",
            "state": f"h{layer + 1}",
            "shape": f"[B, T, {hidden}]",
            "equation": equation,
            "writes": [
                {
                    "kind": "attention",
                    "label": "Attention write",
                    "symbol": f"a{layer}",
                    "sourceNodeId": f"l{layer}_output",
                    "targetNodeId": attention_target,
                    "value": None,
                    "directLogitAttribution": None,
                },
                {
                    "kind": "mlp",
                    "label": "MLP write",
                    "symbol": f"m{layer}",
                    "sourceNodeId": f"l{layer}_mlp",
                    "targetNodeId": f"l{layer}_mlp_residual",
                    "value": None,
                    "directLogitAttribution": None,
                },
            ],
            "activationNorm": None,
            "directLogitAttribution": None,
        })
    return {
        "mode": "structural" if architecture.norm_order == "pre" else "local-residual",
        "topology": architecture.residual_topology,
        "position": {"label": "Position t", "status": "symbolic"},
        "metric": {
            "label": "Direct logit attribution",
            "status": "not-measured",
            "targetToken": None,
        },
        "measurementNote": (
            "The block uses post-normalization, so the ledger shows local residual additions; normalization prevents a single globally additive residual decomposition. "
            "Activation norms and direct logit attribution require a prompt-conditioned run."
            if architecture.norm_order == "post" else
            "The mapped block topology exposes additive token/position, attention, and MLP writes. Activation norms, signed write magnitudes, and direct logit attribution require a prompt-conditioned activation run."
        ),
        "states": states,
    }


def _attach_residual_ledger(nodes: list[dict], ledger: dict) -> None:
    by_id = {node["id"]: node for node in nodes}
    for state in ledger["states"]:
        residual_node = by_id.get(state["id"])
        if residual_node:
            residual_node["inspector"]["residualLedger"] = {"role": "state", **state}
        for write in state["writes"]:
            for identifier in {write["sourceNodeId"], write["targetNodeId"]} - {state["id"]}:
                participant = by_id.get(identifier)
                if not participant:
                    continue
                participant["inspector"]["residualLedger"] = {
                    "role": "write",
                    "stateId": state["id"],
                    "state": state["state"],
                    "equation": state["equation"],
                    "write": write,
                }


def _validate_graph(
    nodes: list[dict],
    edges: list[dict],
    groups: list[dict],
    tensors: dict,
    layers: int,
    architecture: Architecture,
    overview_node_ids: list[str],
) -> dict:
    """Fail closed when the generated graph contradicts its own source inventory or topology."""
    checks: list[dict] = []
    failures: list[str] = []

    def verify(identifier: str, label: str, condition: bool, detail: str) -> None:
        if condition:
            checks.append({"id": identifier, "label": label, "status": "verified", "detail": detail})
        else:
            failures.append(f"{label}: {detail}")

    node_ids = [node["id"] for node in nodes]
    node_id_set = set(node_ids)
    verify("unique-nodes", "Unique graph nodes", len(node_ids) == len(node_id_set), "Every node ID must be unique.")

    invalid_edges = [edge for edge in edges if edge["from"] not in node_id_set or edge["to"] not in node_id_set]
    self_edges = [edge for edge in edges if edge["from"] == edge["to"]]
    verify("edge-endpoints", "Edge endpoint integrity", not invalid_edges, f"{len(edges)} edges reference existing nodes.")
    verify("no-self-edges", "No accidental self-edges", not self_edges, "Generated computation edges cannot target their own source.")

    assigned_names = [tensor["name"] for node in nodes for tensor in node.get("tensors", [])]
    assigned_counts = {name: assigned_names.count(name) for name in set(assigned_names)}
    missing_tensors = [name for name in tensors if assigned_counts.get(name, 0) == 0]
    duplicate_tensors = [name for name, count in assigned_counts.items() if count != 1]
    verify(
        "tensor-assignment",
        "Exact tensor assignment",
        not missing_tensors and not duplicate_tensors and len(assigned_names) == len(tensors),
        f"{len(assigned_names)} of {len(tensors)} checkpoint tensors are assigned exactly once.",
    )

    source_elements = sum(
        shape_product(record.get("shape"))
        for record in tensors.values()
        if isinstance(record, dict)
    )
    source_bytes = sum(
        _tensor_byte_size(record, shape_product(record.get("shape")), str(record.get("dtype", "float32")))[0]
        for record in tensors.values() if isinstance(record, dict)
    )
    mapped_elements = sum(tensor["count"] for node in nodes for tensor in node.get("tensors", []))
    mapped_bytes = sum(tensor["totalBytes"] for node in nodes for tensor in node.get("tensors", []))
    verify("element-conservation", "Checkpoint element conservation", mapped_elements == source_elements, f"Mapped {mapped_elements:,} of {source_elements:,} source elements.")
    verify("byte-conservation", "Checkpoint byte conservation", mapped_bytes == source_bytes, f"Mapped {mapped_bytes:,} of {source_bytes:,} source bytes.")

    group_ids = {group["id"] for group in groups}
    expected_group_ids = {f"layer_{layer}" for layer in range(layers)}
    verify("decoder-groups", "Decoder layer coverage", group_ids == expected_group_ids, f"Represented {len(group_ids)} of {layers} decoder layers.")
    verify("overview-nodes", "Overview node integrity", all(identifier in node_id_set for identifier in overview_node_ids), "Every overview node exists in the full graph.")

    position_kind = architecture.position_kind
    layer_position_ids = {f"l{layer}_position" for layer in range(layers)}
    if position_kind in {"absolute", "absolute-fixed"}:
        position_valid = "position_embedding" in node_id_set and not (layer_position_ids & node_id_set)
        position_detail = "Absolute position information is added once before decoder layer 0."
    elif position_kind in {"rotary", "alibi", "relative-bias"}:
        position_valid = layer_position_ids <= node_id_set and "position_embedding" not in node_id_set
        position_detail = f"Every decoder layer represents its {position_kind} attention operation."
    else:
        position_valid = "position_embedding" not in node_id_set and not (layer_position_ids & node_id_set)
        position_detail = "No position operation is invented when the mechanism is unresolved."
    verify("position-topology", "Position topology", position_valid, position_detail)

    edge_pairs = {(edge["from"], edge["to"]) for edge in edges}
    topology_valid = True
    for layer in range(layers):
        previous = "residual_0" if layer == 0 else f"l{layer - 1}_mlp_residual"
        final = f"l{layer}_mlp_residual"
        input_norm = f"l{layer}_input_norm"
        post_norm = f"l{layer}_post_norm"
        qkv = f"l{layer}_qkv"
        output = f"l{layer}_output"
        mlp = f"l{layer}_mlp"
        attn_residual = f"l{layer}_attn_residual"
        if architecture.residual_topology == "sequential-pre-norm":
            required_nodes = {input_norm, post_norm, attn_residual, final}
            required_edges = {
                (previous, input_norm), (input_norm, qkv), (output, attn_residual),
                (previous, attn_residual), (attn_residual, post_norm), (post_norm, mlp),
                (mlp, final), (attn_residual, final),
            }
        elif architecture.residual_topology == "sequential-post-norm":
            required_nodes = {input_norm, attn_residual, final}
            required_edges = {
                (previous, qkv), (output, attn_residual), (previous, attn_residual),
                (attn_residual, input_norm), (input_norm, mlp), (mlp, final), (input_norm, final),
            }
            topology_valid = topology_valid and post_norm not in node_id_set
        elif architecture.residual_topology == "parallel-shared-norm":
            required_nodes = {input_norm, final}
            required_edges = {
                (previous, input_norm), (input_norm, qkv), (input_norm, mlp),
                (output, final), (mlp, final), (previous, final),
            }
            topology_valid = topology_valid and post_norm not in node_id_set and attn_residual not in node_id_set
        else:
            required_nodes = {input_norm, post_norm, final}
            required_edges = {
                (previous, input_norm), (input_norm, qkv), (previous, post_norm),
                (post_norm, mlp), (output, final), (mlp, final), (previous, final),
            }
            topology_valid = topology_valid and attn_residual not in node_id_set
        topology_valid = topology_valid and required_nodes <= node_id_set and required_edges <= edge_pairs
    verify(
        "residual-topology",
        "Residual and normalization topology",
        topology_valid,
        f"All {layers} blocks satisfy the {architecture.residual_topology} circuit contract.",
    )

    if failures:
        raise GraphError("Generated graph validation failed: " + " | ".join(failures))
    return {
        "status": "verified",
        "scope": "Internal structural consistency against config and Safetensors metadata",
        "checks": checks,
        "limitations": "Validation proves inventory conservation and graph-contract consistency, not arbitrary remote model code or prompt-conditioned behavior.",
    }


def build_model_graph(payload: dict) -> dict:
    config = payload.get("config")
    tensors = payload.get("tensors")
    if not isinstance(config, dict):
        raise GraphError("The Hugging Face config is not an object")
    if not isinstance(tensors, dict):
        raise GraphError("The checkpoint inventory is not an object")
    resolver = payload.get("resolver") if isinstance(payload.get("resolver"), dict) else {
        "tier": "checkpoint-mapped",
        "label": "Checkpoint mapped",
        "format": "safetensors",
        "checkpointFiles": payload.get("files", []),
        "checkpointFileCount": len(payload.get("files", [])),
        "tensorNames": True,
        "tensorShapes": True,
        "tensorDtypes": True,
        "limitations": [],
    }
    resolver_tier = str(resolver.get("tier", "configuration-scaffold"))
    checkpoint_mapped = resolver_tier == "checkpoint-mapped"
    model_id = str(payload.get("modelId", ""))
    revision = str(payload.get("revision") or "main")
    sha = str(payload.get("sha") or revision)
    names = list(tensors)
    text_config = resolve_text_config(config)
    expected_layers = configured_layer_count(text_config) or configured_layer_count(config)
    family = infer_layer_family(names, expected_layers)
    layers = expected_layers or family.inferred_count if family.found else expected_layers
    if layers < 1:
        raise GraphError(
            f"Could not identify decoder layers for {model_id}. The config has no recognized layer-count field and the checkpoint tensor names expose no indexed layer path."
        )
    if not family.found:
        family = FamilyResult(True, "predicted_decoder.layers", layers, 0, 0, True)

    embeddings = [name for name in names if name.endswith(("embed_tokens.weight", "tok_embeddings.weight", "wte.weight", "word_embeddings.weight", "shared.weight"))]
    position_embeddings = [name for name in names if name.lower().endswith(("wpe.weight", "embed_positions.weight", "position_embeddings.weight"))]
    family_base = next((family.prefix[: -len(suffix)] for suffix in (".layers", ".blocks", ".block", ".layer", ".h") if family.prefix.endswith(suffix)), family.prefix)
    preferred_embeddings = [name for name in embeddings if name.startswith(family_base)] or embeddings
    preferred_position_embeddings = [name for name in position_embeddings if name.startswith(family_base)] or position_embeddings
    hidden_value, _ = _positive(text_config, config, ("hidden_size", "n_embd", "d_model"))
    vocab_value, _ = _positive(text_config, config, ("vocab_size", "n_vocab"))
    hidden = str(hidden_value or (_shape_dimension(tensors, preferred_embeddings[0], 1, 0) if preferred_embeddings else 0) or "H")
    vocab = str(vocab_value or (_shape_dimension(tensors, preferred_embeddings[0], 0, 0) if preferred_embeddings else 0) or "V")
    model_type = _configured_string(text_config, config, ("model_type",), "transformer")
    architecture = infer_architecture(config, text_config, tensors, names, family, layers, hidden, vocab)
    query_heads = str(architecture.query_heads or "Hq")
    kv_heads = str(architecture.kv_heads or architecture.query_heads or "Hkv")
    head_dim = str(architecture.head_dim or "Dh")
    grouped_kv = bool(
        architecture.query_heads
        and architecture.kv_heads
        and architecture.query_heads != architecture.kv_heads
    )
    qkv_shape = f"Q [B, {query_heads}, T, {head_dim}] · K/V [B, {kv_heads}, T, {head_dim}]"
    attention_shape = f"P [B, {query_heads}, T, Tkv] · y [B, {query_heads}, T, {head_dim}]"
    cache_shape = f"K,V [B, {kv_heads}, Tcache, {head_dim}]"

    nodes: list[dict] = []
    edges: list[dict] = []
    groups: list[dict] = []
    used: set[str] = set()
    _add_node(nodes, tensors, used, "tokens", "Context tokens", "input", "input_ids", "tokens", "[batch, sequence]", "x<ₜ", "", "", "", 0, 1, [])
    has_position_input = architecture.position_kind in {"absolute", "absolute-fixed"}
    embedding_formula = "e_tok = Etok[x<ₜ]" if has_position_input else "h₀ = E[x<ₜ]"
    _add_node(nodes, tensors, used, "embedding", "Token embedding", "operation", "Embedding", _first_parent_or(preferred_embeddings, "model.embed_tokens"), f"[B, T, {hidden}]", embedding_formula, "", "", "", 0, 2, preferred_embeddings)
    _add_edge(edges, "tokens", "embedding", "token ids", "activation")
    if has_position_input:
        position_type = architecture.position
        position_formula = "e_pos = Epos[0…T−1]" if architecture.position_kind == "absolute" else "e_pos = sinusoid(0…T−1)"
        _add_node(nodes, tensors, used, "position_embedding", "Input position encoding", "operation", position_type, _first_parent_or(preferred_position_embeddings, "position_encoding"), f"[B, T, {hidden}]", position_formula, "Added once at the model input; it is not a per-layer Q/K transform.", "", "", 0, 3, preferred_position_embeddings)
        _add_edge(edges, "tokens", "position_embedding", "position indices", "activation")
    initial_formula = "h₀ = e_tok + e_pos" if has_position_input else "h₀ = E[x<ₜ]"
    initial_description = "The hidden-state stream before decoder layer 0. Absolute position embeddings are included here." if has_position_input else "The hidden-state stream before decoder layer 0."
    _add_node(nodes, tensors, used, "residual_0", "Residual stream h₀", "state", "Residual stream", "residual_stream.0", f"[B, T, {hidden}]", initial_formula, initial_description, "", "", 0, 0, [])
    _add_edge(edges, "embedding", "residual_0", "write token embedding", "residual")
    if has_position_input:
        _add_edge(edges, "position_embedding", "residual_0", "write position embedding", "residual")
    previous = "residual_0"

    for layer in range(layers):
        group_id = f"layer_{layer}"
        attention_column = layer * 2 + 1
        mlp_column = attention_column + 1
        layer_names = _layer_names(names, family.prefix, layer)
        _add_group(groups, layer, model_type, not checkpoint_mapped or family.predicted or not layer_names, architecture)
        input_norm, qkv, output_names, post_norm, mlp, layer_position, attention_aux, other = _bucket_layer(layer_names)
        topology = architecture.residual_topology
        position_node = f"l{layer}_position"
        attention_input = f"h{layer}" if topology == "sequential-post-norm" else "u_attn"
        position_formula = {
            "rotary": "Q′=RoPE(Q,pos), K′=RoPE(K,pos)",
            "alibi": f"S = Q{' broadcast_G(K)' if grouped_kv else 'K'}ᵀ/√Dh + BALiBi",
            "relative-bias": f"S = Q{' broadcast_G(K)' if grouped_kv else 'K'}ᵀ/√Dh + Brelative",
        }.get(architecture.position_kind, "")
        position_shape = qkv_shape if architecture.position_kind == "rotary" else f"Bpos [1, {query_heads}, T, Tkv]"
        query_symbol = "Q′" if architecture.position_kind == "rotary" else "Q"
        key_symbol = "K′" if architecture.position_kind == "rotary" else "K"
        logical_key = f"broadcast_G({key_symbol})" if grouped_kv else key_symbol
        logical_value = "broadcast_G(V)" if grouped_kv else "V"
        attention_formula = (
            f"P=softmax({query_symbol}{logical_key}ᵀ/√Dh+Mcausal); y=P{logical_value}"
            if architecture.position_kind not in {"alibi", "relative-bias"}
            else f"P=softmax(S+Mcausal); y=P{logical_value}"
        )
        position_is_layer_operation = architecture.position_kind in {"rotary", "alibi", "relative-bias"}
        common_specs = [
            (f"l{layer}_qkv", "Q / K / V projections", "operation", architecture.attention, _first_parent_or(qkv, f"layer.{layer}.qkv"), qkv_shape, f"Q={attention_input}Wqᵀ; K={attention_input}Wkᵀ; V={attention_input}Wvᵀ", "attention", attention_column, 3, qkv, "Projection shapes preserve separate query and key/value head counts."),
            (f"l{layer}_attention", "Causal self-attention", "operation", architecture.attention, f"layer.{layer}.attention", attention_shape, attention_formula, "attention", attention_column, 5, attention_aux, ("KV heads are logically shared across query-head groups; the cache remains at Hkv rather than being physically expanded. " if grouped_kv else "") + "Attention probabilities are observational; cache state is optional and controlled by use_cache."),
            (f"l{layer}_output", "Attention output projection", "operation", "Linear projection", _first_parent_or(output_names, f"layer.{layer}.output"), f"[B, T, {hidden}]", "a = yWoᵀ", "attention", attention_column, 6, output_names, "Projected attention write before residual addition."),
            (f"l{layer}_mlp", "Feed-forward network", "operation", architecture.feed_forward, _first_parent_or(mlp, f"layer.{layer}.mlp"), f"[B, T, {hidden}]", "m = MLP(u_mlp)", "mlp", mlp_column, 3, mlp, "Feed-forward or routed-expert branch as supported by mapped tensors."),
            (f"l{layer}_kv_cache", "Layer KV cache", "state", "Optional autoregressive KV state", f"layer.{layer}.kv_cache", cache_shape, "Kcache,Vcache ← append(K,V)", "attention", attention_column, 8, [], "Present when generation enables use_cache; cache storage format is implementation-dependent."),
        ]
        for identifier, name, kind, operation, path, shape, formula, subgroup, column, row, tensor_names, description in common_specs:
            _add_node(nodes, tensors, used, identifier, name, kind, operation, path, shape, formula, description, group_id, subgroup, column, row, tensor_names)
        if position_is_layer_operation:
            _add_node(nodes, tensors, used, position_node, "Rotary Q/K transform" if architecture.position_kind == "rotary" else "Attention position bias", "operation", architecture.position, _first_parent_or(layer_position, f"layer.{layer}.position"), position_shape, position_formula, "Applied inside attention; absolute input positions are represented before layer 0 instead.", group_id, "attention", attention_column, 4, layer_position)
        if topology == "sequential-pre-norm":
            _add_node(nodes, tensors, used, f"l{layer}_input_norm", "Attention pre-normalization", "operation", architecture.norm, _first_parent_or(input_norm, f"layer.{layer}.input_norm"), f"[B, T, {hidden}]", f"u_attn = Norm(h{layer})", "Normalization before the attention branch.", group_id, "attention", attention_column, 2, input_norm)
            _add_node(nodes, tensors, used, f"l{layer}_attn_residual", "Post-attention residual state", "operation", "Residual add", f"layer.{layer}.attn_residual", f"[B, T, {hidden}]", f"r_attn = h{layer} + a{layer}", "Local residual state after the attention write.", group_id, "attention", attention_column, 7, [])
            _add_node(nodes, tensors, used, f"l{layer}_post_norm", "MLP pre-normalization", "operation", architecture.norm, _first_parent_or(post_norm, f"layer.{layer}.post_norm"), f"[B, T, {hidden}]", "u_mlp = Norm(r_attn)", "Normalization before the sequential MLP branch.", group_id, "mlp", mlp_column, 2, post_norm)
            _add_node(nodes, tensors, used, f"l{layer}_mlp_residual", f"Residual stream h{layer + 1}", "state", "Residual stream write", f"layer.{layer}.mlp_residual", f"[B, T, {hidden}]", f"h{layer + 1} = h{layer} + a{layer} + m{layer}", "Block-output residual state.", group_id, "", mlp_column, 0, [])
            _add_edge(edges, previous, f"l{layer}_input_norm", f"h{layer}", "activation")
            _add_edge(edges, f"l{layer}_input_norm", f"l{layer}_qkv", "", "activation")
            _add_edge(edges, f"l{layer}_output", f"l{layer}_attn_residual", f"write a{layer}", "residual")
            _add_edge(edges, previous, f"l{layer}_attn_residual", f"residual h{layer}", "residual")
            _add_edge(edges, f"l{layer}_attn_residual", f"l{layer}_post_norm", "", "activation")
            _add_edge(edges, f"l{layer}_post_norm", f"l{layer}_mlp", "", "activation")
            _add_edge(edges, f"l{layer}_mlp", f"l{layer}_mlp_residual", f"write m{layer}", "residual")
            _add_edge(edges, f"l{layer}_attn_residual", f"l{layer}_mlp_residual", "residual r_attn", "residual")
            _add_edge(edges, previous, f"l{layer}_mlp_residual", f"residual rail h{layer} → h{layer + 1}", "residual-stream")
        elif topology == "sequential-post-norm":
            _add_node(nodes, tensors, used, f"l{layer}_attn_residual", "Attention residual add", "operation", "Residual add", f"layer.{layer}.attn_residual", f"[B, T, {hidden}]", f"r_attn = h{layer} + a{layer}", "Local attention residual before post-normalization.", group_id, "attention", attention_column, 7, [])
            _add_node(nodes, tensors, used, f"l{layer}_input_norm", "Post-attention normalization", "operation", architecture.norm, _first_parent_or(input_norm, f"layer.{layer}.attention_norm"), f"[B, T, {hidden}]", "u_mlp = Norm(r_attn)", "Post-normalization architecture; this is not an attention pre-norm.", group_id, "mlp", mlp_column, 2, input_norm)
            _add_node(nodes, tensors, used, f"l{layer}_mlp_residual", f"Hidden state h{layer + 1}", "state", "Residual add + post-normalization", f"layer.{layer}.output", f"[B, T, {hidden}]", f"h{layer + 1} = Norm(u_mlp + m{layer})", "Post-normalization breaks a globally additive residual-stream decomposition.", group_id, "", mlp_column, 0, post_norm)
            _add_edge(edges, previous, f"l{layer}_qkv", f"h{layer}", "activation")
            _add_edge(edges, f"l{layer}_output", f"l{layer}_attn_residual", f"write a{layer}", "residual")
            _add_edge(edges, previous, f"l{layer}_attn_residual", f"residual h{layer}", "residual")
            _add_edge(edges, f"l{layer}_attn_residual", f"l{layer}_input_norm", "", "activation")
            _add_edge(edges, f"l{layer}_input_norm", f"l{layer}_mlp", "", "activation")
            _add_edge(edges, f"l{layer}_mlp", f"l{layer}_mlp_residual", f"write m{layer}; normalize", "residual")
            _add_edge(edges, f"l{layer}_input_norm", f"l{layer}_mlp_residual", "local residual u_mlp", "residual")
        else:
            shared_norm = topology == "parallel-shared-norm"
            _add_node(nodes, tensors, used, f"l{layer}_input_norm", "Shared branch pre-normalization" if shared_norm else "Attention branch pre-normalization", "operation", architecture.norm, _first_parent_or(input_norm, f"layer.{layer}.input_norm"), f"[B, T, {hidden}]", f"u_attn = Norm(h{layer})", "Feeds both branches." if shared_norm else "Feeds the attention branch.", group_id, "attention", attention_column, 2, input_norm)
            if not shared_norm:
                _add_node(nodes, tensors, used, f"l{layer}_post_norm", "MLP branch pre-normalization", "operation", architecture.norm, _first_parent_or(post_norm, f"layer.{layer}.mlp_norm"), f"[B, T, {hidden}]", f"u_mlp = Norm(h{layer})", "Separate normalization of the same block input for the parallel MLP branch.", group_id, "mlp", mlp_column, 2, post_norm)
            _add_node(nodes, tensors, used, f"l{layer}_mlp_residual", f"Residual stream h{layer + 1}", "state", "Parallel residual merge", f"layer.{layer}.residual_merge", f"[B, T, {hidden}]", f"h{layer + 1} = h{layer} + a{layer} + m{layer}", "Attention and MLP writes are computed in parallel from the same block input.", group_id, "", mlp_column, 0, [])
            _add_edge(edges, previous, f"l{layer}_input_norm", f"h{layer}", "activation")
            _add_edge(edges, f"l{layer}_input_norm", f"l{layer}_qkv", "", "activation")
            mlp_norm_node = f"l{layer}_input_norm" if shared_norm else f"l{layer}_post_norm"
            if not shared_norm:
                _add_edge(edges, previous, mlp_norm_node, f"h{layer}", "activation")
            _add_edge(edges, mlp_norm_node, f"l{layer}_mlp", "parallel branch", "activation")
            _add_edge(edges, f"l{layer}_output", f"l{layer}_mlp_residual", f"write a{layer}", "residual")
            _add_edge(edges, f"l{layer}_mlp", f"l{layer}_mlp_residual", f"write m{layer}", "residual")
            _add_edge(edges, previous, f"l{layer}_mlp_residual", f"residual rail h{layer} → h{layer + 1}", "residual-stream")
        if position_is_layer_operation:
            _add_edge(edges, f"l{layer}_qkv", position_node, "", "activation")
            _add_edge(edges, position_node, f"l{layer}_attention", "", "activation")
        else:
            _add_edge(edges, f"l{layer}_qkv", f"l{layer}_attention", "", "activation")
        _add_edge(edges, f"l{layer}_attention", f"l{layer}_output", "", "activation")
        cache_source = position_node if architecture.position_kind == "rotary" else f"l{layer}_qkv"
        _add_edge(edges, cache_source, f"l{layer}_kv_cache", "write K,V", "activation")
        _add_edge(edges, f"l{layer}_kv_cache", f"l{layer}_attention", "past K,V", "activation", True)
        if other:
            _add_node(nodes, tensors, used, f"l{layer}_other", "Unresolved layer tensors", "state", "Unmapped model-specific state", _first_parent_or(other, f"layer.{layer}.other"), "", "", "Preserved without invented execution edges because their forward role was not resolved from checkpoint metadata.", group_id, "mlp", mlp_column, 4, other)
        previous = f"l{layer}_mlp_residual"

    output_column = layers * 2 + 1
    unused = [name for name in names if name not in used]
    final_norm_candidates = [name for name in unused if name.lower().endswith((".norm.weight", ".norm.bias")) or "ln_f." in name.lower() or "final_layernorm" in name.lower() or "final_layer_norm" in name.lower()]
    final_norm_names = [name for name in final_norm_candidates if name.startswith(family_base)] or final_norm_candidates
    used.update(final_norm_names)
    unused = [name for name in names if name not in used]
    head_names = [name for name in unused if "lm_head." in name.lower() or "embed_out." in name.lower()]
    head_operation = "Tied unembedding" if architecture.output_tied else "LM head"
    head_formula_input = "h̄ₜ" if final_norm_names else f"h{layers}ₜ"
    head_formula = f"zₜ = Etokᵀ·{head_formula_input}" if architecture.output_tied else f"zₜ = WU·{head_formula_input}"
    output_specs = []
    if final_norm_names:
        output_specs.append(("final_norm", "Final normalization", "operation", architecture.norm, _first_parent_or(final_norm_names, "model.norm"), f"[B, T, {hidden}]", f"h̄ = Norm(h{layers})", 0, final_norm_names, "Mapped from final-normalization checkpoint tensors.", None))
    output_specs.extend([
        ("lm_head", "Vocabulary projection", "operation", head_operation, _first_parent_or(head_names, "lm_head" if not architecture.output_tied else _first_parent_or(preferred_embeddings, "model.embed_tokens")), f"[B, T, {vocab}]", head_formula, 1, head_names, "Uses the token-embedding matrix by transposition." if architecture.output_tied else "Projects hidden states to vocabulary logits.", {"tiedTo": "embedding"} if architecture.output_tied else None),
        ("distribution", "Next-token distribution", "output", "Softmax", "distribution", f"[B, {vocab}]", "p(xₜ|x<ₜ)=softmax(process(zₜ))", 2, [], "Logit processors may include temperature, penalties, masking, top-k, or top-p depending on generation settings.", None),
        ("sample", "Token selection", "sampler", "Generation strategy", "sample", "", "xₜ = select(p; greedy or sampling strategy)", 3, [], "Selection is deterministic for greedy decoding and stochastic only for sampling strategies.", None),
        ("append", "Append generated token", "output", "Concatenate", "append", "", "x≤ₜ = concat(x<ₜ,xₜ)", 4, [], "", None),
        ("next_step", "Token step t + 1", "state", "Autoregressive recurrence", "next_step", "next decoding step", "decode(x≤ₜ, optional cache) → xₜ₊₁", 5, [], "Generation repeats from the updated token sequence; KV reuse occurs only when caching is enabled.", None),
    ])
    for identifier, name, kind, operation, path, shape, formula, row, tensor_names, description, metadata in output_specs:
        _add_node(nodes, tensors, used, identifier, name, kind, operation, path, shape, formula, description, "", "", output_column, row, tensor_names, metadata)
    remaining = [name for name in names if name not in used]
    if remaining:
        _add_node(nodes, tensors, used, "unmapped_tensors", "Unmapped checkpoint tensors", "state", "Exact weight metadata", "unmapped_tensors", "", "", "Preserved so the graph never silently drops checkpoint tensors.", "", "", output_column, 7, remaining)
    _add_node(nodes, tensors, used, "checkpoint_metadata", "Safetensors checkpoint", "state", "Complete header metadata", "checkpoint_metadata", "", "", "File-level Safetensors metadata retained from every checkpoint shard.", "", "", output_column, 9, [], payload.get("safetensors"))
    inference_metadata = {"disclaimer": "Inferred findings are speculative and should be verified against the model implementation.", "findings": architecture.findings}
    _add_node(nodes, tensors, used, "architecture_inference", "Architecture predictions", "state", "Evidence-based inference", "architecture_inference", "", "", "Checkpoint-derived predictions are explicitly labeled by basis and confidence; they are not treated as ground truth.", "", "", output_column, 11, [], inference_metadata)

    output_input = "final_norm" if final_norm_names else previous
    if final_norm_names:
        _add_edge(edges, previous, "final_norm", f"block output h{layers}", "activation" if architecture.norm_order == "post" else "residual-stream")
    for source, target, label, kind in (
        (output_input, "lm_head", "", "activation"),
        ("lm_head", "distribution", "last-position logits", "activation"),
        ("distribution", "sample", "", "activation"),
        ("sample", "append", "xₜ", "activation"),
        ("append", "next_step", "advance t ← t + 1", "activation"),
    ):
        _add_edge(edges, source, target, label, kind)
    _enrich_edges(nodes, edges)
    hub = payload.get("hub") if isinstance(payload.get("hub"), dict) else {}
    _enrich_nodes(nodes, edges, groups, config, text_config, hub, architecture.findings)
    residual_ledger = _build_residual_ledger(layers, hidden, architecture, has_position_input)
    _attach_residual_ledger(nodes, residual_ledger)

    total_parameters = sum(node["directParameters"] for node in nodes)
    total_bytes = sum(node["totalBytes"] for node in nodes)
    total_tensors = sum(len(node["tensors"]) for node in nodes)
    tensor_records = [tensor for node in nodes for tensor in node["tensors"]]
    buffer_records = [tensor for tensor in tensor_records if tensor.get("storageRole") == "buffer"]
    tensor_ordering = {
        "automatic": True,
        "semanticBasis": "tensor-path aliases with a humanized module-path fallback",
        "checkpointBasis": "natural shard order followed by Safetensors byte offset",
        "recognized": sum(tensor["order"]["semanticConfidence"] == "recognized" for tensor in tensor_records),
        "pathDerived": sum(tensor["order"]["semanticConfidence"] == "path-derived" for tensor in tensor_records),
        "checkpointLocated": sum(isinstance(tensor["order"].get("checkpointIndex"), int) for tensor in tensor_records),
    }
    architectures = config.get("architectures")
    architecture_name = architectures[0] if isinstance(architectures, list) and architectures and isinstance(architectures[0], str) else "AutoModelForCausalLM"
    overview_node_ids = [
        "tokens",
        "embedding",
        *(["position_embedding"] if has_position_input else []),
        "residual_0",
        *(f"l{layer}_mlp_residual" for layer in range(layers)),
        *(["final_norm"] if final_norm_names else []),
        "lm_head",
        "distribution",
    ]
    validation = _validate_graph(nodes, edges, groups, tensors, layers, architecture, overview_node_ids)
    graph = {
        "mode": "flow",
        "name": model_id,
        "type": architecture_name,
        "description": f"{model_type} · {layers} layers · revision {sha[:8]}",
        "source": {"provider": "huggingface", "modelId": model_id, "revision": revision, "sha": payload.get("sha"), "url": f"https://huggingface.co/{model_id}/tree/{sha}"},
        "config": config,
        "resolvedTextConfig": text_config,
        "resolvedLayerFamily": {"prefix": family.prefix, "inferredCount": family.inferred_count, "observedLayers": family.observed_layers, "score": family.score, **({"predicted": True} if family.predicted else {})},
        "architecturePredictions": architecture.findings,
        "forwardTopology": {
            "residual": architecture.residual_topology,
            "normalizationOrder": architecture.norm_order,
            "positionKind": architecture.position_kind,
            "confidence": architecture.topology_confidence,
            "evidence": architecture.topology_evidence,
            "status": "resolved" if architecture.topology_confidence in {"high", "medium"} else "scaffold",
            "disclaimer": "Checkpoint metadata cannot prove arbitrary custom forward code; low-confidence scaffolds and unresolved tensors are labelled rather than treated as exact execution steps.",
        },
        "residualLedger": residual_ledger,
        "tensorOrdering": tensor_ordering,
        "validation": validation,
        "safetensors": payload.get("safetensors"),
        "huggingFace": {"hub": hub, "artifacts": payload.get("artifacts"), "artifactInspection": payload.get("artifactInspection")},
        "groups": groups,
        "nodes": nodes,
        "edges": edges,
        "stats": {"modules": len(nodes), "parameterTensors": total_tensors, "checkpointTensors": total_tensors, "parameterLikeTensors": total_tensors - len(buffer_records), "bufferTensors": len(buffer_records), "maxDepth": 0, "totalParameters": total_parameters, "checkpointElements": total_parameters, "recognizedBufferElements": sum(tensor["count"] for tensor in buffer_records), "trainableParameters": None, "totalBytes": total_bytes, "dtypes": list(dict.fromkeys(str(tensor.get("dtype", "float32")) for tensor in tensors.values() if isinstance(tensor, dict)))},
        "layout": {"nodeWidth": NODE_WIDTH, "nodeHeight": NODE_HEIGHT, "columnStride": COLUMN_STRIDE, "rowStride": ROW_STRIDE, "maxColumn": output_column, "bounds": {"width": output_column * COLUMN_STRIDE + NODE_WIDTH + TENSOR_STACK_MAX_EXTENT, "height": 11 * ROW_STRIDE + NODE_HEIGHT + TENSOR_STACK_MAX_EXTENT}, "overviewNodeIds": overview_node_ids},
    }
    return graph
