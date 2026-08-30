"""Safe Hugging Face repository inspection with capability-aware fallbacks."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote

from .config import (
    HEADER_PROBE_BYTES,
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_FILES,
    MAX_ARTIFACT_TOTAL_BYTES,
    MAX_HEADER_BYTES,
    MAX_PARALLEL_HEADER_REQUESTS,
    MAX_SHARDS,
)
from .http_client import HTTPResponse, http_get

HUB_EXPAND_QUERY = (
    "expand=author&expand=baseModels&expand=cardData&expand=childrenModelCount&expand=config"
    "&expand=createdAt&expand=disabled&expand=downloads&expand=downloadsAllTime&expand=evalResults"
    "&expand=gated&expand=gguf&expand=inference&expand=inferenceProviderMapping&expand=lastModified"
    "&expand=library_name&expand=likes&expand=mask_token&expand=model-index&expand=pipeline_tag"
    "&expand=private&expand=resourceGroup&expand=safetensors&expand=sha&expand=siblings&expand=spaces"
    "&expand=tags&expand=transformersInfo&expand=trendingScore&expand=usedStorage&expand=widgetData"
)


class InspectionError(RuntimeError):
    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class ModelFile:
    name: str
    size: int | None


def _status_name(status: int) -> str:
    return {
        200: "OK",
        206: "Partial Content",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
    }.get(status, "HTTP error")


def fetch_json(url: str, token: str = "") -> object:
    response = HTTPResponse()
    for retry in range(5):
        if retry:
            time.sleep(2 ** (retry - 1))
        response = http_get(url, token=token)
        if response.status != 429:
            break
    if response.error:
        raise InspectionError(response.error, response.status)
    if not 200 <= response.status < 300:
        excerpt = response.text[:180]
        raise InspectionError(
            f"Hugging Face returned {response.status} {_status_name(response.status)}: {excerpt}",
            response.status,
        )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as error:
        raise InspectionError("Hugging Face returned invalid JSON", response.status) from error


def _merge_objects(*objects: object) -> dict:
    merged: dict = {}
    for value in objects:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def repository_files(info: dict) -> list[ModelFile]:
    files: list[ModelFile] = []
    for item in info.get("siblings", []):
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str):
            continue
        size = item.get("size")
        files.append(ModelFile(item["rfilename"], int(size) if isinstance(size, (int, float)) else None))
    return files


def _canonical_safetensors(path: str) -> bool:
    name = PurePosixPath(path).name
    canonical = name == "model.safetensors" or (name.startswith("model-") and name.endswith(".safetensors"))
    return canonical and "optimizer" not in name


def _natural_path_key(value: str) -> tuple:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", value.lower())
    )


def weight_file_names(files: list[ModelFile]) -> list[str]:
    canonical = [file.name for file in files if _canonical_safetensors(file.name)]
    if canonical:
        return sorted(canonical, key=_natural_path_key)
    return sorted(
        [
            file.name for file in files
            if file.name.endswith(".safetensors")
            and "optimizer" not in file.name.lower()
            and "adapter" not in PurePosixPath(file.name).name.lower()
        ],
        key=_natural_path_key,
    )


def _checkpoint_file_inventory(files: list[ModelFile]) -> dict[str, list[str]]:
    inventory = {
        "pytorch": [],
        "adapter": [],
        "gguf": [],
        "tensorflow": [],
        "flax": [],
    }
    ignored = ("optimizer", "training_args", "scheduler", "rng_state")
    for file in files:
        name = PurePosixPath(file.name).name.lower()
        if "adapter" in name and name.endswith((".safetensors", ".bin", ".pt", ".pth")):
            inventory["adapter"].append(file.name)
        elif name.endswith(".gguf"):
            inventory["gguf"].append(file.name)
        elif name.endswith((".h5", ".ckpt")) or name.startswith("tf_model"):
            inventory["tensorflow"].append(file.name)
        elif name.endswith(".msgpack") or name.startswith("flax_model"):
            inventory["flax"].append(file.name)
        elif name.endswith((".bin", ".pt", ".pth")) and not any(word in name for word in ignored):
            inventory["pytorch"].append(file.name)
    return {key: sorted(values, key=_natural_path_key) for key, values in inventory.items()}


def _pytorch_manifest(artifacts: dict) -> tuple[str | None, dict | None]:
    for name, artifact in sorted(artifacts.items(), key=lambda item: _natural_path_key(item[0])):
        data = artifact.get("data") if isinstance(artifact, dict) else None
        if name.lower().endswith(".bin.index.json") and isinstance(data, dict) and isinstance(data.get("weight_map"), dict):
            return name, data
    return None, None


def _known_file_bytes(files: list[ModelFile], names: list[str]) -> int | None:
    selected = [file for file in files if file.name in set(names)]
    if not selected or any(file.size is None for file in selected):
        return None
    return sum(int(file.size or 0) for file in selected)


def normalise_dtype(dtype: str) -> str:
    return {
        "F64": "float64",
        "F32": "float32",
        "F16": "float16",
        "BF16": "bfloat16",
        "F8_E4M3": "float8_e4m3",
        "F8_E4M3FN": "float8_e4m3fn",
        "F8_E5M2": "float8_e5m2",
        "F8_E4M3FNUZ": "float8_e4m3fnuz",
        "F8_E5M2FNUZ": "float8_e5m2fnuz",
        "I64": "int64",
        "I32": "int32",
        "I16": "int16",
        "I8": "int8",
        "U64": "uint64",
        "U32": "uint32",
        "U16": "uint16",
        "U8": "uint8",
        "BOOL": "bool",
    }.get(dtype, dtype.lower())


def _response_file_size(response: HTTPResponse) -> int | None:
    content_range = response.header("Content-Range")
    if "/" not in content_range:
        return None
    value = content_range.rsplit("/", 1)[1]
    try:
        return int(value)
    except ValueError:
        return None


def _weight_url(model_path: str, revision_path: str, file_name: str) -> str:
    return f"https://huggingface.co/{model_path}/resolve/{revision_path}/{quote(file_name, safe='/')}"


def fetch_safetensors_prefix(model_path: str, revision_path: str, file_name: str, token: str) -> HTTPResponse:
    return http_get(
        _weight_url(model_path, revision_path, file_name),
        token=token,
        byte_range=f"0-{HEADER_PROBE_BYTES - 1}",
    )


def _retry_prefix(model_path: str, revision_path: str, file_name: str, token: str, response: HTTPResponse) -> HTTPResponse:
    for retry in range(4):
        if response.status not in {404, 429} and response.status < 500:
            break
        time.sleep(2**retry)
        response = fetch_safetensors_prefix(model_path, revision_path, file_name, token)
    return response


def inspect_safetensors_file(
    model_path: str,
    revision_path: str,
    file_name: str,
    token: str,
    prefix: HTTPResponse,
) -> tuple[dict[str, dict], dict, int]:
    prefix = _retry_prefix(model_path, revision_path, file_name, token, prefix)
    if prefix.error:
        raise InspectionError(prefix.error, prefix.status)
    if prefix.status != 206:
        if prefix.status == 429:
            message = f"Hugging Face rate-limited metadata for {file_name}; connect an account and retry"
        else:
            message = f"Weight host did not honor a metadata range request for {file_name} ({prefix.status})"
        raise InspectionError(message, prefix.status)
    if len(prefix.body) < 8:
        raise InspectionError("Invalid Safetensors header prefix", prefix.status)

    header_length = int.from_bytes(prefix.body[:8], "little", signed=False)
    if not 2 <= header_length <= MAX_HEADER_BYTES:
        raise InspectionError(f"Unreasonable Safetensors header length: {header_length}", prefix.status)
    required = 8 + header_length
    file_size = _response_file_size(prefix)
    if len(prefix.body) >= required:
        header_bytes = prefix.body[8:required]
    else:
        response = http_get(
            _weight_url(model_path, revision_path, file_name),
            token=token,
            byte_range=f"8-{7 + header_length}",
        )
        if response.error:
            raise InspectionError(response.error, response.status)
        if response.status != 206:
            raise InspectionError(
                f"Weight host did not honor a metadata range request for {file_name} ({response.status})",
                response.status,
            )
        if len(response.body) < header_length:
            raise InspectionError("Safetensors header response was truncated", response.status)
        header_bytes = response.body[:header_length]
        file_size = file_size if file_size is not None else _response_file_size(response)

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InspectionError("Invalid Safetensors JSON header", prefix.status) from error
    if not isinstance(header, dict):
        raise InspectionError("Invalid Safetensors JSON header", prefix.status)

    tensors: dict[str, dict] = {}
    tensor_bytes = 0
    for name, raw in header.items():
        if name == "__metadata__" or not isinstance(raw, dict):
            continue
        dtype = raw.get("dtype", "unknown")
        shape = raw.get("shape") if isinstance(raw.get("shape"), list) else []
        offsets = raw.get("data_offsets")
        element_count: int | None = 1
        for dimension in shape:
            if not isinstance(dimension, (int, float)) or int(dimension) < 0:
                element_count = None
                break
            element_count *= int(dimension)
        byte_size: int | None = None
        if isinstance(offsets, list) and len(offsets) == 2:
            start, end = offsets
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and 0 <= int(start) <= int(end):
                byte_size = int(end) - int(start)
                tensor_bytes += byte_size
        tensors[name] = {
            "shape": shape,
            "dtype": normalise_dtype(str(dtype)),
            "safetensors": {
                "file": file_name,
                "dtype": dtype,
                "dataOffsets": offsets if isinstance(offsets, list) else None,
                "byteSize": byte_size,
                "elementCount": element_count,
                "rank": len(shape),
                "raw": raw,
            },
        }

    ordered_tensor_names = sorted(
        tensors,
        key=lambda name: (
            tensors[name]["safetensors"]["dataOffsets"][0]
            if isinstance(tensors[name]["safetensors"]["dataOffsets"], list)
            and tensors[name]["safetensors"]["dataOffsets"]
            and isinstance(tensors[name]["safetensors"]["dataOffsets"][0], (int, float))
            else float("inf"),
            name,
        ),
    )
    for file_tensor_index, name in enumerate(ordered_tensor_names):
        tensors[name]["safetensors"]["fileTensorIndex"] = file_tensor_index

    record = {
        "file": file_name,
        "prefixLength": 8,
        "headerLength": header_length,
        "dataStart": required,
        "fileSize": file_size,
        "etag": prefix.header("ETag") or None,
        "lastModified": prefix.header("Last-Modified") or None,
        "contentType": prefix.header("Content-Type") or None,
        "metadata": header.get("__metadata__"),
        "tensorCount": len(tensors),
        "tensorBytes": tensor_bytes,
        "nonTensorBytes": max(0, file_size - tensor_bytes) if file_size is not None else None,
    }
    return tensors, record, tensor_bytes


def _is_text_metadata_file(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return (
        name == ".gitattributes"
        or name.startswith("license")
        or name.endswith((".json", ".md", ".txt", ".jinja", ".yaml", ".yml", ".toml"))
    )


def _artifact_priority(path: str) -> tuple[int, str]:
    name = PurePosixPath(path).name.lower()
    preferred = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "adapter_config.json",
        "quantization_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "readme.md",
    }
    checkpoint_index = name.endswith((".safetensors.index.json", ".bin.index.json"))
    return (0 if name in preferred or checkpoint_index else 1 if "/" not in path else 2, path)


def inspect_artifacts(
    files: list[ModelFile], model_path: str, revision_path: str, token: str
) -> tuple[dict, dict]:
    candidates = sorted((file for file in files if _is_text_metadata_file(file.name)), key=lambda file: _artifact_priority(file.name))
    selected: list[ModelFile] = []
    skipped: list[dict] = []
    selected_bytes = 0
    for file in candidates:
        reason = ""
        if file.size is None or file.size > MAX_ARTIFACT_BYTES:
            reason = f"larger than {MAX_ARTIFACT_BYTES} bytes"
        elif len(selected) >= MAX_ARTIFACT_FILES:
            reason = f"more than {MAX_ARTIFACT_FILES} metadata files"
        elif selected_bytes + file.size > MAX_ARTIFACT_TOTAL_BYTES:
            reason = f"metadata total would exceed {MAX_ARTIFACT_TOTAL_BYTES} bytes"
        if reason:
            skipped.append({"file": file.name, "size": file.size, "reason": reason})
        else:
            selected.append(file)
            selected_bytes += file.size

    artifacts: dict[str, dict] = {}
    errors: list[dict] = []
    for file in selected:
        response = http_get(_weight_url(model_path, revision_path, file.name), token=token)
        if response.error:
            errors.append({"file": file.name, "error": response.error})
            continue
        if not 200 <= response.status < 300:
            errors.append({"file": file.name, "error": f"Hugging Face returned {response.status}"})
            continue
        if len(response.body) > MAX_ARTIFACT_BYTES:
            errors.append({"file": file.name, "error": "file exceeds the metadata byte limit"})
            continue
        text = response.text
        data: object = text
        format_name = "text"
        if file.name.lower().endswith(".json"):
            try:
                data = json.loads(text)
                format_name = "json"
            except json.JSONDecodeError:
                pass
        artifacts[file.name] = {
            "file": file.name,
            "size": file.size,
            "format": format_name,
            "data": data,
            "response": {
                "etag": response.header("ETag") or None,
                "lastModified": response.header("Last-Modified") or None,
                "contentType": response.header("Content-Type") or None,
            },
        }
    inspection = {
        "fetchedCount": len(artifacts),
        "fetchedBytes": selected_bytes,
        "perFileLimit": MAX_ARTIFACT_BYTES,
        "totalLimit": MAX_ARTIFACT_TOTAL_BYTES,
        "fileLimit": MAX_ARTIFACT_FILES,
        "skipped": skipped,
        "errors": errors,
    }
    return artifacts, inspection


def inspect_account(token: str) -> dict:
    account = fetch_json("https://huggingface.co/api/whoami-v2", token)
    if not isinstance(account, dict):
        raise InspectionError("Hugging Face returned an invalid account response", 502)
    orgs = account.get("orgs") if isinstance(account.get("orgs"), list) else []
    return {
        "name": account.get("name"),
        "fullname": account.get("fullname"),
        "avatarUrl": account.get("avatarUrl"),
        "type": account.get("type"),
        "isPro": account.get("isPro"),
        "orgs": [
            {"name": org.get("name"), "fullname": org.get("fullname"), "avatarUrl": org.get("avatarUrl")}
            for org in orgs
            if isinstance(org, dict)
        ],
    }


def inspect_model(model_id: str, revision: str, token: str = "") -> dict:
    from .graph import build_model_graph

    model_path = quote(model_id, safe="/")
    revision_path = quote(revision, safe="")
    urls = {
        "config": f"https://huggingface.co/{model_path}/resolve/{revision_path}/config.json",
        "blob": f"https://huggingface.co/api/models/{model_path}/revision/{revision_path}?blobs=true",
        "expanded": f"https://huggingface.co/api/models/{model_path}/revision/{revision_path}?{HUB_EXPAND_QUERY}",
        "security": f"https://huggingface.co/api/models/{model_path}/revision/{revision_path}?securityStatus=true",
    }
    try:
        config = fetch_json(urls["config"], token)
    except InspectionError as error:
        raise InspectionError(f"Config request failed: {error}", error.status) from error
    try:
        info_blob = fetch_json(urls["blob"], token)
    except InspectionError as error:
        raise InspectionError(f"Repository file request failed: {error}", error.status) from error
    try:
        info_expanded = fetch_json(urls["expanded"], token)
    except InspectionError as error:
        raise InspectionError(f"Expanded metadata request failed: {error}", error.status) from error
    try:
        info_security = fetch_json(urls["security"], token)
    except InspectionError as error:
        raise InspectionError(f"Security metadata request failed: {error}", error.status) from error

    if not isinstance(config, dict):
        raise InspectionError("The Hugging Face config is not an object", 422)
    info = _merge_objects(info_expanded, info_security, info_blob)
    files = repository_files(info)
    artifacts, artifact_inspection = inspect_artifacts(files, model_path, revision_path, token)
    weights = weight_file_names(files)
    if len(weights) > MAX_SHARDS:
        raise InspectionError(
            f"The checkpoint has {len(weights)} shards; the current inspection limit is {MAX_SHARDS}", 422
        )
    tensors: dict[str, dict] = {}
    file_records: list[dict] = []
    tensor_bytes = 0
    checkpoint_inventory = _checkpoint_file_inventory(files)
    resolver: dict
    if weights:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_HEADER_REQUESTS) as executor:
            prefixes = list(
                executor.map(
                    lambda file_name: fetch_safetensors_prefix(model_path, revision_path, file_name, token),
                    weights,
                )
            )
        checkpoint_tensor_index = 0
        for shard_index, (file_name, prefix) in enumerate(zip(weights, prefixes, strict=True)):
            shard_tensors, record, shard_bytes = inspect_safetensors_file(
                model_path, revision_path, file_name, token, prefix
            )
            for name in sorted(
                shard_tensors,
                key=lambda tensor_name: shard_tensors[tensor_name]["safetensors"]["fileTensorIndex"],
            ):
                safetensors = shard_tensors[name]["safetensors"]
                safetensors["shardIndex"] = shard_index
                safetensors["checkpointIndex"] = checkpoint_tensor_index
                checkpoint_tensor_index += 1
            tensors.update(shard_tensors)
            file_records.append(record)
            tensor_bytes += shard_bytes
        resolver = {
            "tier": "checkpoint-mapped",
            "label": "Checkpoint mapped",
            "format": "safetensors",
            "checkpointFiles": weights,
            "checkpointFileCount": len(weights),
            "tensorNames": True,
            "tensorShapes": True,
            "tensorDtypes": True,
            "safeMetadataOnly": True,
            "weightBytes": _known_file_bytes(files, weights),
            "limitations": [],
        }
    else:
        manifest_name, manifest = _pytorch_manifest(artifacts)
        pytorch_files = checkpoint_inventory["pytorch"]
        manifest_map = manifest.get("weight_map") if isinstance(manifest, dict) else None
        if isinstance(manifest_map, dict):
            manifest_files = sorted(
                {str(file_name) for file_name in manifest_map.values() if isinstance(file_name, str)},
                key=_natural_path_key,
            )
            for checkpoint_index, (name, file_name) in enumerate(manifest_map.items()):
                if not isinstance(name, str) or not isinstance(file_name, str):
                    continue
                tensors[name] = {
                    "shape": None,
                    "dtype": "unknown",
                    "checkpoint": {
                        "format": "pytorch-manifest",
                        "indexFile": manifest_name,
                        "file": file_name,
                        "checkpointIndex": checkpoint_index,
                        "shapeKnown": False,
                        "dtypeKnown": False,
                    },
                }
            metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
            declared_bytes = metadata.get("total_size")
            resolver = {
                "tier": "manifest-mapped",
                "label": "Manifest mapped",
                "format": "pytorch",
                "checkpointFiles": manifest_files or pytorch_files,
                "checkpointFileCount": len(manifest_files or pytorch_files),
                "indexFile": manifest_name,
                "tensorNames": True,
                "tensorShapes": False,
                "tensorDtypes": False,
                "safeMetadataOnly": True,
                "weightBytes": int(declared_bytes) if isinstance(declared_bytes, (int, float)) else _known_file_bytes(files, manifest_files or pytorch_files),
                "limitations": [
                    "The PyTorch index exposes tensor names and shard membership but not tensor shapes, dtypes, or byte offsets.",
                    "Remote pickle checkpoint content was not downloaded or executed during graph inspection.",
                ],
            }
        else:
            format_name, selected_files = next(
                ((name, values) for name, values in checkpoint_inventory.items() if values),
                ("configuration", []),
            )
            resolver = {
                "tier": "configuration-scaffold",
                "label": "Configuration scaffold",
                "format": format_name,
                "checkpointFiles": selected_files,
                "checkpointFileCount": len(selected_files),
                "tensorNames": False,
                "tensorShapes": False,
                "tensorDtypes": False,
                "safeMetadataOnly": True,
                "weightBytes": _known_file_bytes(files, selected_files),
                "limitations": [
                    "The repository does not expose safely range-readable tensor headers or a supported weight index.",
                    "The circuit is derived from configuration only; tensor provenance and exact parameter counts are unavailable.",
                    "Remote checkpoint content and custom repository code were not executed during graph inspection.",
                ],
            }

    payload = {
        "modelId": model_id,
        "revision": revision,
        "sha": info.get("sha"),
        "config": config,
        "hub": info,
        "artifacts": artifacts,
        "artifactInspection": artifact_inspection,
        "tensors": tensors,
        "files": resolver["checkpointFiles"],
        "resolver": resolver,
        "safetensors": {
            "format": "safetensors",
            "files": file_records,
            "fileCount": len(weights),
            "tensorCount": len(tensors),
            "tensorBytes": tensor_bytes,
            "completeHeaderMetadata": True,
        } if weights else None,
    }
    return {"graph": build_model_graph(payload)}
