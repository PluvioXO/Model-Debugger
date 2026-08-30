"""Managed Daytona GPU runtimes for the ModelDebugger execution worker."""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .http_client import http_request

GIB = 1024**3
DAYTONA_WORKER_PORT = 8765
DAYTONA_RUNTIME_MINUTES = 120
DAYTONA_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
DAYTONA_GPU_MEMORY_GIB = {
    "RTX-4090": 24,
    "RTX-5090": 32,
    "H100": 80,
    "RTX-PRO-6000": 96,
    "H200": 141,
}
DAYTONA_GPU_ORDER = tuple(DAYTONA_GPU_MEMORY_GIB)


class DaytonaError(RuntimeError):
    """Raised when a Daytona runtime cannot be safely provisioned or removed."""


@dataclass(frozen=True, slots=True)
class ProvisionedRuntime:
    endpoint: str
    preview_token: str
    secret: str
    sandbox_id: str
    gpu_type: str
    recommendation: dict[str, Any]


def _non_negative_integer(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def recommend_gpu(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a conservative, single-GPU inference recommendation.

    ModelDebugger captures activations and retains bounded run records, so the
    estimate reserves substantially more than just checkpoint weight storage.
    It is a preflight estimate, not a promise that architecture-specific code or
    unusually long prompts will fit.
    """

    details = payload if isinstance(payload, dict) else {}
    parameter_count = _non_negative_integer(details.get("parameterCount"))
    checkpoint_bytes = _non_negative_integer(details.get("checkpointBytes"))
    model_id = str(details.get("modelId", "")).strip()

    if parameter_count:
        full_weight_bytes = max(checkpoint_bytes, parameter_count * 2)
    else:
        full_weight_bytes = checkpoint_bytes
    # Framework allocations, CUDA kernels, KV cache, and hook captures all need
    # room. Four GiB plus 35% of weights is intentionally conservative for the
    # app's bounded (<=4096 token) diagnostic passes.
    full_peak_bytes = int(full_weight_bytes * 1.35 + 4 * GIB)
    quantized_weight_bytes = int(parameter_count * 0.65) if parameter_count else int(checkpoint_bytes * 0.35)
    quantized_peak_bytes = int(quantized_weight_bytes * 1.35 + 4 * GIB)

    def first_fit(required: int) -> str | None:
        return next(
            (
                gpu
                for gpu in DAYTONA_GPU_ORDER
                if required <= int(DAYTONA_GPU_MEMORY_GIB[gpu] * GIB * 0.85)
            ),
            None,
        )

    quantization = "none"
    estimated_weight_bytes = full_weight_bytes
    estimated_peak_bytes = full_peak_bytes
    recommended = first_fit(full_peak_bytes)
    reason = "Sized for half-precision weights, framework allocations, KV cache, and bounded hook captures."
    if recommended is None and (parameter_count or checkpoint_bytes):
        recommended = first_fit(quantized_peak_bytes)
        if recommended is not None:
            quantization = "4bit"
            estimated_weight_bytes = quantized_weight_bytes
            estimated_peak_bytes = quantized_peak_bytes
            reason = "Half precision exceeds one supported GPU; use 4-bit NF4 weights with BF16 compute and retain activation headroom."
    if recommended is None:
        recommended = DAYTONA_GPU_ORDER[0] if not (parameter_count or checkpoint_bytes) else DAYTONA_GPU_ORDER[-1]
        reason = (
            "Load a checkpoint to calculate a model-aware recommendation."
            if not (parameter_count or checkpoint_bytes)
            else "The estimate exceeds a single H200 even after 4-bit quantization; use a smaller checkpoint or reduce capture scope."
        )

    start_index = DAYTONA_GPU_ORDER.index(recommended)
    preferences = list(DAYTONA_GPU_ORDER[start_index:])
    fits = estimated_peak_bytes <= int(DAYTONA_GPU_MEMORY_GIB[recommended] * GIB * 0.85)
    return {
        "modelId": model_id or None,
        "recommendedGpu": recommended,
        "gpuPreferences": preferences,
        "gpuMemoryGiB": DAYTONA_GPU_MEMORY_GIB[recommended],
        "quantization": quantization,
        "estimatedWeightBytes": estimated_weight_bytes or None,
        "estimatedPeakBytes": estimated_peak_bytes if parameter_count or checkpoint_bytes else None,
        "fitsSingleGpu": fits,
        "confidence": "model-aware" if parameter_count or checkpoint_bytes else "baseline",
        "reason": reason,
        "headroomFraction": 0.15,
    }


def _daytona_sdk() -> tuple[Any, ...]:
    try:
        from daytona import (  # type: ignore[import-not-found]
            CreateSandboxFromImageParams,
            Daytona,
            DaytonaConfig,
            GpuType,
            Resources,
            SessionExecuteRequest,
        )
    except ImportError as error:
        raise DaytonaError(
            "Daytona support is not installed. Run `python -m pip install -e .` and restart ModelDebugger."
        ) from error
    return Daytona, DaytonaConfig, CreateSandboxFromImageParams, Resources, GpuType, SessionExecuteRequest


def _api_key(value: str) -> str:
    key = value.strip() or os.environ.get("DAYTONA_API_KEY", "").strip()
    if not key:
        raise DaytonaError("Enter your Daytona API key in Settings to provision a GPU on your account.")
    if len(key) > 4096 or any(character.isspace() for character in key):
        raise DaytonaError("The Daytona API key format is invalid.")
    return key


def _sandbox_create_params(
    Params: Any,
    Resources: Any,
    GpuType: Any,
    gpu_types: list[str],
    env_vars: dict[str, str],
) -> Any:
    """Build the one allowed Daytona capacity profile: private, ephemeral, and spot."""
    return Params(
        name=f"modeldebugger-{secrets.token_hex(4)}",
        image=DAYTONA_IMAGE,
        public=False,
        ephemeral=True,
        spot=True,
        auto_stop_interval=0,
        ttl_minutes=DAYTONA_RUNTIME_MINUTES,
        env_vars=env_vars,
        labels={"app": "modeldebugger", "managed-by": "modeldebugger"},
        resources=Resources(gpu=1, gpu_type=[GpuType(value) for value in gpu_types]),
    )


def provision_runtime(
    api_key: str,
    payload: dict[str, Any],
    *,
    worker_path: Path,
) -> ProvisionedRuntime:
    """Create, prepare, and health-check a private Daytona GPU sandbox."""

    key = _api_key(api_key)
    recommendation = recommend_gpu(payload)
    requested_gpu = str(payload.get("gpuType", "auto")).strip() or "auto"
    if requested_gpu != "auto" and requested_gpu not in DAYTONA_GPU_MEMORY_GIB:
        raise DaytonaError("Choose a supported Daytona GPU type.")
    gpu_types = [requested_gpu] if requested_gpu != "auto" else recommendation["gpuPreferences"]
    if requested_gpu != "auto" and recommendation["estimatedPeakBytes"]:
        safe_capacity = int(DAYTONA_GPU_MEMORY_GIB[requested_gpu] * GIB * 0.85)
        if int(recommendation["estimatedPeakBytes"]) > safe_capacity:
            raise DaytonaError(
                f"{requested_gpu} does not leave enough activation headroom for this checkpoint. "
                f"Choose {recommendation['recommendedGpu']} or Automatic."
            )
    if not worker_path.is_file():
        raise DaytonaError("The bundled execution worker is missing from this installation.")
    Daytona, DaytonaConfig, Params, Resources, GpuType, SessionExecuteRequest = _daytona_sdk()

    secret = secrets.token_urlsafe(32)
    env_vars = {"MODELDEBUGGER_WORKER_SECRET": secret}
    raw_hf_token = payload.get("hfToken", "")
    hf_token = raw_hf_token.strip() if isinstance(raw_hf_token, str) else ""
    if hf_token:
        if len(hf_token) > 2048 or any(character.isspace() for character in hf_token):
            raise DaytonaError("The optional Hugging Face token format is invalid.")
        env_vars["HF_TOKEN"] = hf_token

    client = Daytona(DaytonaConfig(api_key=key))
    sandbox = None
    try:
        sandbox = client.create(
            _sandbox_create_params(Params, Resources, GpuType, gpu_types, env_vars),
            timeout=180,
        )
        sandbox.fs.upload_file(str(worker_path), "/workspace/modeldebugger_worker.py")
        install = sandbox.process.exec(
            "python -m pip install --disable-pip-version-check "
            "'transformers>=4.46,<5' 'accelerate>=1,<2' 'safetensors>=0.4,<1' 'bitsandbytes>=0.45,<1'",
            cwd="/workspace",
            timeout=1800,
        )
        if int(install.exit_code) != 0:
            tail = str(install.result or "").strip()[-800:]
            raise DaytonaError(f"Daytona could not install the worker dependencies: {tail or 'pip failed'}")

        session_id = "modeldebugger-worker"
        sandbox.process.create_session(session_id)
        started = sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(
                command=f"python /workspace/modeldebugger_worker.py --host 0.0.0.0 --port {DAYTONA_WORKER_PORT}",
                run_async=True,
                suppress_input_echo=True,
            ),
            timeout=30,
        )
        if started.exit_code not in {None, 0}:
            raise DaytonaError(f"The Daytona worker exited during startup: {started.stderr or started.output or 'unknown error'}")

        preview = sandbox.get_preview_link(DAYTONA_WORKER_PORT)
        endpoint = str(preview.url).rstrip("/")
        preview_token = str(preview.token or "")
        if not endpoint.startswith("https://") or not preview_token:
            raise DaytonaError("Daytona did not return a private authenticated preview URL.")

        deadline = time.monotonic() + 120
        last_error = "worker did not answer"
        while time.monotonic() < deadline:
            response = http_request(
                f"{endpoint}/health",
                token=secret,
                headers={"x-daytona-preview-token": preview_token},
                timeout=10,
            )
            if response.status == 200:
                try:
                    health = json.loads(response.body)
                except (ValueError, UnicodeDecodeError):
                    health = None
                if isinstance(health, dict) and health.get("ok") is True:
                    break
                last_error = "worker returned an invalid health response"
                time.sleep(2)
                continue
            last_error = response.error or f"worker returned {response.status}"
            time.sleep(2)
        else:
            raise DaytonaError(f"The Daytona GPU started, but its worker health check failed: {last_error}")

        return ProvisionedRuntime(
            endpoint=endpoint,
            preview_token=preview_token,
            secret=secret,
            sandbox_id=str(sandbox.id),
            gpu_type=str(getattr(sandbox, "gpu_type", "") or (requested_gpu if requested_gpu != "auto" else recommendation["recommendedGpu"])),
            recommendation=recommendation,
        )
    except DaytonaError:
        if sandbox is not None:
            try:
                client.delete(sandbox, timeout=120, wait=True)
            except Exception:
                pass
        raise
    except Exception as error:
        if sandbox is not None:
            try:
                client.delete(sandbox, timeout=120, wait=True)
            except Exception:
                pass
        raise DaytonaError(f"Daytona provisioning failed: {error}") from error


def delete_runtime(api_key: str, sandbox_id: str) -> None:
    """Delete a managed Daytona sandbox immediately to stop credit use."""

    Daytona, DaytonaConfig, *_ = _daytona_sdk()
    try:
        client = Daytona(DaytonaConfig(api_key=_api_key(api_key)))
        sandbox = client.get(sandbox_id)
        client.delete(sandbox, timeout=120, wait=True)
    except DaytonaError:
        raise
    except Exception as error:
        raise DaytonaError(f"Could not delete Daytona sandbox {sandbox_id}: {error}") from error
