#!/usr/bin/env python3
"""Validate a downloaded sparse J-space observation artifact.

Uses only the Python standard library. It validates provenance/retention structure,
content addressing, and the bounded smoke-test contract without interpreting the
scientific meaning of token readouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate(
    path: Path, expected_sha256: str | None
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    require(path.is_file(), f"artifact is not a file: {path}", errors)
    if errors:
        return {}, errors

    size = path.stat().st_size
    digest = sha256_file(path)

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"path": str(path), "size_bytes": size, "sha256": digest}, [
            f"invalid UTF-8 JSON: {type(exc).__name__}: {exc}"
        ]

    require(isinstance(artifact, dict), "top-level JSON must be an object", errors)
    if not isinstance(artifact, dict):
        return {"path": str(path), "size_bytes": size, "sha256": digest}, errors

    required_top_level = {
        "created_at_utc",
        "evidence_class",
        "input",
        "instrumentation",
        "lens",
        "measurement",
        "model",
        "retention",
        "run_id",
        "runtime",
        "schema",
        "scope",
    }
    missing = sorted(required_top_level - artifact.keys())
    require(not missing, f"missing top-level fields: {missing}", errors)

    require(
        artifact.get("evidence_class") == "direct_runtime_measurement",
        "evidence_class must be direct_runtime_measurement",
        errors,
    )
    require(
        artifact.get("scope") == "open_loop_observation_only",
        "scope must be open_loop_observation_only",
        errors,
    )
    require(
        isinstance(artifact.get("schema"), str) and bool(artifact.get("schema")),
        "schema must be a non-empty string",
        errors,
    )
    require(
        isinstance(artifact.get("run_id"), str) and bool(artifact.get("run_id")),
        "run_id must be a non-empty string",
        errors,
    )

    for field in (
        "raw_activations_persisted",
        "full_logits_persisted",
        "raw_prompt_persisted",
    ):
        require(
            nested(artifact, "retention", field) is False,
            f"retention.{field} must be false",
            errors,
        )
    require(
        nested(artifact, "input", "raw_prompt_persisted") is False,
        "input.raw_prompt_persisted must be false",
        errors,
    )
    require(
        isinstance(nested(artifact, "input", "sha256"), str)
        and len(nested(artifact, "input", "sha256")) == 64,
        "input.sha256 must be a 64-character digest",
        errors,
    )

    for section, fields in {
        "model": ("repo_id", "revision", "n_layers", "d_model"),
        "lens": (
            "repo_id",
            "revision",
            "filename",
            "sha256",
            "source_layers",
            "d_model",
        ),
        "instrumentation": ("repo", "commit"),
        "runtime": (
            "python",
            "torch",
            "cuda_runtime",
            "gpu_name",
            "gpu_total_vram_gib",
        ),
    }.items():
        value = artifact.get(section)
        require(isinstance(value, dict), f"{section} must be an object", errors)
        if isinstance(value, dict):
            absent = [field for field in fields if field not in value]
            require(not absent, f"{section} missing fields: {absent}", errors)

    lens_hash = nested(artifact, "lens", "sha256")
    require(
        isinstance(lens_hash, str) and len(lens_hash) == 64,
        "lens.sha256 must be a 64-character digest",
        errors,
    )

    selected_layers = nested(artifact, "measurement", "selected_layers")
    jacobian = nested(artifact, "measurement", "jacobian_lens")
    baseline = nested(artifact, "measurement", "logit_lens_baseline")
    repeatability = nested(artifact, "measurement", "repeatability_same_runtime")
    require(
        isinstance(selected_layers, list) and selected_layers,
        "measurement.selected_layers must be a non-empty list",
        errors,
    )
    require(
        isinstance(jacobian, dict) and jacobian,
        "measurement.jacobian_lens must be non-empty",
        errors,
    )
    require(
        isinstance(baseline, dict) and baseline,
        "measurement.logit_lens_baseline must be non-empty",
        errors,
    )
    require(
        isinstance(repeatability, dict),
        "repeatability_same_runtime must be an object",
        errors,
    )
    if (
        isinstance(selected_layers, list)
        and isinstance(jacobian, dict)
        and isinstance(baseline, dict)
    ):
        expected_layer_keys = {str(layer) for layer in selected_layers}
        require(
            set(jacobian) == expected_layer_keys,
            "Jacobian readout layer keys must equal selected_layers",
            errors,
        )
        require(
            set(baseline) == expected_layer_keys,
            "baseline layer keys must equal selected_layers",
            errors,
        )

    if path.name.startswith("jspace_observation_") and path.suffix == ".json":
        prefix = path.stem.removeprefix("jspace_observation_")
        require(
            bool(prefix) and digest.startswith(prefix),
            f"filename digest prefix {prefix!r} does not match SHA-256 {digest}",
            errors,
        )

    if expected_sha256 is not None:
        require(
            digest == expected_sha256.lower(),
            f"SHA-256 mismatch: measured {digest}, expected {expected_sha256.lower()}",
            errors,
        )

    summary = {
        "path": str(path.resolve()),
        "size_bytes": size,
        "sha256": digest,
        "schema": artifact.get("schema"),
        "run_id": artifact.get("run_id"),
        "scope": artifact.get("scope"),
        "model": nested(artifact, "model", "repo_id"),
        "model_revision": nested(artifact, "model", "revision"),
        "lens": nested(artifact, "lens", "repo_id"),
        "lens_revision": nested(artifact, "lens", "revision"),
        "lens_sha256": lens_hash,
        "gpu": nested(artifact, "runtime", "gpu_name"),
        "gpu_total_vram_gib": nested(artifact, "runtime", "gpu_total_vram_gib"),
        "selected_layers": selected_layers,
        "same_topk_token_ids": nested(
            artifact, "measurement", "repeatability_same_runtime", "same_topk_token_ids"
        ),
        "retention": artifact.get("retention"),
        "valid": not errors,
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", type=Path, help="Downloaded J-space observation JSON"
    )
    parser.add_argument("--expected-sha256", help="Runtime-reported SHA-256 to require")
    args = parser.parse_args()

    summary, errors = validate(args.artifact, args.expected_sha256)
    result = {"summary": summary, "errors": errors}
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
