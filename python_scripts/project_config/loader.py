"""
Load SmartSentry AML settings from project_config/*.yaml and project_config/*.json.

Notebooks import this module after ``setup_project_path()`` so the repo root
is on ``sys.path``.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CONFIG_DIR.parent


def setup_project_path() -> Path:
    """Ensure repo root is on sys.path (call from python_scripts notebooks)."""
    root = PROJECT_ROOT
    root_str = str(root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return root


def _read_yaml(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / name

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings() -> dict[str, Any]:
    """Load project_config/settings.yaml."""
    return _read_yaml("settings.yaml")


def load_detector_config() -> dict[str, Any]:
    """Load static detector reference project_config/detector.yaml."""
    return _read_yaml("detector.yaml")


def load_generator_config(
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load synthetic data generator CONFIG.

    Uses ``project_config/generator_config.json`` by default;
    override path via:
        settings['paths']['generator_config_json']
    or env:
        AML_GENERATOR_CONFIG
    """

    settings = settings or load_settings()

    rel = settings.get(
        "paths",
        {},
    ).get(
        "generator_config_json",
        "project_config/generator_config.json",
    )

    path = Path(
        os.environ.get(
            "AML_GENERATOR_CONFIG",
            PROJECT_ROOT / rel,
        )
    )

    if not path.is_file():
        path = PROJECT_ROOT / "project_config" / "generator_config.json"

    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def apply_env_overrides(
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Apply AML_* environment variables over settings['environment']."""

    env_cfg = deepcopy(settings.get("environment", {}))

    mapping = {
        "AML_PIPELINE_MODE": "AML_PIPELINE_MODE",
        "AML_RUN_MODE": "AML_RUN_MODE",
        "AML_DATA_DIR": "AML_DATA_DIR",
        "AML_PHASE1_DIR": "AML_PHASE1_DIR",
        "AML_PHASE2_DIR": "AML_PHASE2_DIR",
        "AML_INPUT_FILE": "AML_INPUT_FILE",
        "AML_INFERENCE_INPUT": "AML_INFERENCE_INPUT",
    }

    for key, env_key in mapping.items():
        val = os.environ.get(env_key)

        if val:
            env_cfg[key] = val

    settings = deepcopy(settings)
    settings["environment"] = env_cfg

    return settings


def resolve_paths(
    settings: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Resolve all path keys to absolute Path objects."""

    settings = apply_env_overrides(
        settings or load_settings()
    )

    env = settings.get("environment", {})
    paths_cfg = settings.get("paths", {})
    root = PROJECT_ROOT

    def _p(key: str, default: str) -> Path:
        override = env.get(
            {
                "outputs_dir": "AML_DATA_DIR",
                "phase1_dir": "AML_PHASE1_DIR",
                "phase2_dir": "AML_PHASE2_DIR",
            }.get(key)
        )

        if override:
            return Path(override).resolve()

        return (
            root / paths_cfg.get(key, default)
        ).resolve()

    return {
        "project_root": root,

        "notebook_dir": (
            root / paths_cfg.get(
                "notebook_dir",
                ".",
            )
        ).resolve(),

        "outputs_dir": _p(
            "outputs_dir",
            "outputs_updated",
        ),

        "phase1_dir": _p(
            "phase1_dir",
            "ml_outputs",
        ),

        "phase2_dir": _p(
            "phase2_dir",
            "phase2_outputs",
        ),

        "executed_notebooks": (
            root / paths_cfg.get(
                "executed_notebooks",
                "outputs_updated/executed_notebooks",
            )
        ).resolve(),

        "executed_inference_notebooks": (
            root / paths_cfg.get(
                "executed_inference_notebooks",
                "outputs_updated/executed_inference_notebooks",
            )
        ).resolve(),

        "generator_config_json": (
            root / paths_cfg.get(
                "generator_config_json",
                "project_config/generator_config.json",
            )
        ).resolve(),
    }


def get_run_mode(
    settings: dict[str, Any] | None = None,
) -> str:
    """Return 'train' or 'predict' from env / settings."""

    settings = apply_env_overrides(
        settings or load_settings()
    )

    mode = (
        os.environ.get("AML_RUN_MODE")
        or settings.get("environment", {}).get("AML_RUN_MODE")
    )

    if not mode or str(mode).lower() == "null":
        mode = settings.get(
            "phase1",
            {},
        ).get(
            "run_mode",
            "train",
        )

    mode = str(mode).lower()

    if mode not in ("train", "predict"):
        raise ValueError(
            f"AML_RUN_MODE must be 'train' or 'predict', got {mode!r}"
        )

    return mode


def get_pipeline_mode(
    settings: dict[str, Any] | None = None,
) -> str:
    """Return 'train' or 'predict' for orchestrator."""

    settings = apply_env_overrides(
        settings or load_settings()
    )

    mode = (
        os.environ.get("AML_PIPELINE_MODE")
        or settings.get("environment", {}).get("AML_PIPELINE_MODE")
    )

    if not mode or str(mode).lower() == "null":
        return "train"

    mode = str(mode).lower()

    if mode in ("yes", "y", "true", "1", "train"):
        return "train"

    if mode in ("no", "n", "false", "0", "predict"):
        return "predict"

    return mode


def get_artifact_path(
    paths: dict[str, Path],
    name: str,
    settings: dict[str, Any] | None = None,
) -> Path:
    """Resolve a named intermediate artifact under outputs_dir."""

    settings = settings or load_settings()

    filename = settings.get(
        "artifacts",
        {},
    ).get(
        name,
        f"{name}.parquet",
    )

    return paths["outputs_dir"] / filename


def save_generator_config(
    config: dict[str, Any],
    settings: dict[str, Any] | None = None,
    paths: dict[str, Path] | None = None,
) -> None:
    """Persist generator CONFIG."""

    settings = settings or load_settings()
    paths = paths or resolve_paths(settings)

    targets = [
        paths["generator_config_json"],
        paths["project_root"]
        / settings.get(
            "paths",
            {},
        ).get(
            "generator_runtime_json",
            "outputs_updated/config.json",
        ),
    ]

    seen: set[str] = set()

    for target in targets:
        target = Path(target).resolve()

        key = str(target)

        if key in seen:
            continue

        seen.add(key)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with target.open(
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(
                config,
                fh,
                indent=2,
                default=str,
            )


def build_detect_config(
    typology_generation_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Create detector config (thresholds/knobs) from generator params.

    The current detector notebook (01__aml_typology_detector.ipynb) expects
    DETECT_CONFIG[<typology_name>] to be a dict containing specific keys.

    This loader maps generator_config.json structure into those keys.
    """

    TG = typology_generation_params or {}

    def _get(section: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        sec = TG.get(section, {})
        if not isinstance(sec, dict):
            return fallback or {}
        return sec

    # Defaults chosen to be consistent with detector logic.
    detect: dict[str, dict[str, Any]] = {
        "structuring": {
            "cash_threshold": 10000.0,
            "amount_floor_pct": 0.80,
            "time_window_days": 3,
            "consolidation_window_days": 3,
            "min_source_accounts": 3,
            "max_links_per_deposit": 3,
        },
        "circular": {
            "time_window_days": 7,
            "min_loop_size": 3,
            "max_loop_size": 6,
            "amount_tolerance_pct": 0.08,
            "min_amount": 80000.0,
            "max_other_txns_between_parties": 999,
            "exclude_hawala_signature": True,
        },
        "funnel": {
            "time_window_days": 5,
            "min_unique_senders": 5,
            "outflow_window_days": 5,
            "outflow_retention_pct": 0.05,
            "min_total_inflow": 50000.0,
        },
        "passthrough": {
            "time_gap_minutes": 60,
            "retention_pct": 0.06,
            "min_amount": 50000.0,
            "min_occurrences": 1,
            "require_different_counterparties": False,
            "max_net_position_ratio": 1.0,
        },
        "layering": {
            "max_chain_hours": 24,
            "min_hops": 4,
            "max_hops": 8,
            "amount_decay_tolerance": 0.05,
            "min_amount": 50000.0,
            "search_limit": 15000,
        },
        "third_party_web": {
            "time_window_days": 10,
            "min_unique_payers": 5,
            "per_payment_amount_range": [10000.0, 100000.0],
            "min_total_inflow": 0.0,
            "channels": ["NEFT", "IMPS", "UPI"],
            "hour_range": [0, 23],
        },
        "money_mule": {
            "min_mules": 3,
            "controller_amount_range": [20000.0, 200000.0],
            "forward_pct_range": [0.85, 0.95],
            "forward_delay_hours": 24,
            "channels": ["IMPS", "UPI", "NEFT"],
            "hour_range": [0, 23],
        },
        "high_risk_corridor": {
            "amount_range": [50000.0, 500000.0],
            "target_countries": ["AE", "PK", "BD"],
            "min_transfers_per_account": 3,
            "time_window_days": 7,
            "channels": ["RTGS", "NEFT", "SWIFT"],
            "hour_range": [0, 23],
        },
        "hawala": {
            "num_parties_range": [3, 4],
            "amount_range": [80000.0, 1200000.0],
            "amount_tolerance_pct": 0.08,
            "time_window_days": 10,
            "channels": ["NEFT", "RTGS", "Branch Cash"],
            "hour_range": [0, 23],
        },
        "charity_abuse": {
            "min_donors": 5,
            "donation_amount_range": [1000.0, 50000.0],
            "donation_window_days": 14,
            "diversion_window_days": 10,
            "diversion_retention_pct": 0.2,
            "donation_channels": ["UPI", "NEFT", "IMPS"],
            "diversion_hour_range": [0, 23],
            "min_total_donation": 20000.0,
        },
    }

    # Map generator -> detector where possible.
    # Structuring
    struct = _get("structuring")
    deposit_range = struct.get("deposit_amount_range", [8000, 9900]) or [8000, 9900]
    detect["structuring"].update(
        {
            "cash_threshold": float(deposit_range[1]),
        }
    )
    detect["structuring"].update({
        "amount_floor_pct": float(struct.get("deposit_amount_floor_pct", 0.80))
        if "deposit_amount_floor_pct" in struct
        else 0.80,
        "time_window_days": 3,
        "consolidation_window_days": float(struct.get("transfer_delay_days_range", [1, 3])[-1]),
        "min_source_accounts": int(struct.get("num_sources_range", [3, 6])[0]),
    })

    # Circular
    circ = _get("circular")
    detect["circular"].update({
        "time_window_days": 7,
        "min_loop_size": int(circ.get("ring_size_range", [3, 5])[0]),
        "max_loop_size": int(circ.get("ring_size_range", [3, 5])[1] if len(circ.get("ring_size_range", [3, 5])) > 1 else 6),
        "amount_tolerance_pct": 0.08,
        "min_amount": float(circ.get("base_amount_range", [50000, 500000])[0]),
        "exclude_hawala_signature": True,
        "max_other_txns_between_parties": 999,
    })

    # Funnel
    funnel = _get("funnel")
    detect["funnel"].update({
        "time_window_days": 5,
        "min_unique_senders": int(funnel.get("num_feeders_range", [15, 50])[0]),
        "outflow_window_days": int(funnel.get("outflow_delay_days_range", [6, 10])[1]),
        "outflow_retention_pct": float(funnel.get("retention_pct", 0.05)),
        "min_total_inflow": float(funnel.get("per_feeder_amount_range", [5000, 30000])[0]) * int(funnel.get("num_feeders_range", [15, 50])[0]),
    })

    # Pass-through
    passw = _get("passthrough")
    detect["passthrough"].update({
        "time_gap_minutes": int(passw.get("time_gap_hours", 1) * 60) if isinstance(passw.get("time_gap_hours", 1), (int, float)) else 60,
        "retention_pct": float(passw.get("retention_pct", 0.06)) if "retention_pct" in passw else 0.06,
        "min_amount": float(passw.get("inflow_amount_range", [200000, 2000000])[0]),
        "min_occurrences": 1,
    })

    # Layering
    lay = _get("layering")
    detect["layering"].update({
        "min_hops": int(lay.get("num_hops_range", [8, 10])[0]) - 4,
        "max_hops": int(lay.get("num_hops_range", [8, 10])[1]),
        "max_chain_hours": 24,
        "amount_decay_tolerance": 0.05,
        "min_amount": float(lay.get("base_amount_range", [100000, 1000000])[0]),
        "search_limit": 15000,
    })

    # Third-party web
    tp = _get("third_party_web")
    if tp:
        detect["third_party_web"].update({
            "time_window_days": 10,
            "min_unique_payers": int(tp.get("num_unrelated_payers_range", [5, 15])[0]),
            "per_payment_amount_range": [float(tp.get("per_payment_amount_range", [10000, 100000])[0]), float(tp.get("per_payment_amount_range", [10000, 100000])[1])],
            "channels": tp.get("payment_channels", detect["third_party_web"]["channels"]),
            "hour_range": tp.get("payment_hour_range", detect["third_party_web"]["hour_range"]),
        })

    # Money mule
    mm = _get("money_mule")
    if mm:
        detect["money_mule"].update({
            "min_mules": int(mm.get("num_mules_range", [5, 20])[0] / 2) if isinstance(mm.get("num_mules_range", [5, 20])[0], (int, float)) else 3,
            "controller_amount_range": [float(mm.get("controller_to_mule_amount_range", [20000, 200000])[0]), float(mm.get("controller_to_mule_amount_range", [20000, 200000])[1])],
            "forward_pct_range": [float(mm.get("mule_forward_pct_range", [0.85, 0.95])[0]), float(mm.get("mule_forward_pct_range", [0.85, 0.95])[1])],
            "forward_delay_hours": int(mm.get("mule_forward_delay_hours_range", [1, 24])[1]),
            "channels": mm.get("channels", detect["money_mule"]["channels"]),
            "hour_range": mm.get("hour_range", detect["money_mule"]["hour_range"]),
        })

    # High-risk corridor
    hrc = _get("high_risk_corridor")
    if hrc:
        detect["high_risk_corridor"].update({
            "amount_range": [float(hrc.get("amount_range", [50000, 500000])[0]), float(hrc.get("amount_range", [50000, 500000])[1])],
            "target_countries": hrc.get("target_countries", detect["high_risk_corridor"]["target_countries"]),
            "min_transfers_per_account": int(hrc.get("frequency_per_account_range", [3, 8])[0]),
            "time_window_days": 7,
            "channels": hrc.get("channels", detect["high_risk_corridor"]["channels"]),
            "hour_range": hrc.get("hour_range", detect["high_risk_corridor"]["hour_range"]),
        })

    # Hawala
    hw = _get("hawala")
    if hw:
        detect["hawala"].update({
            "num_parties_range": [int(hw.get("num_parties_range", [3, 4])[0]), int(hw.get("num_parties_range", [3, 4])[1])],
            "amount_range": [float(hw.get("settlement_amount_range", [100000, 1000000])[0]), float(hw.get("settlement_amount_range", [100000, 1000000])[1])],
            "channels": hw.get("channels", detect["hawala"]["channels"]),
            "hour_range": hw.get("hour_range", detect["hawala"]["hour_range"]),
        })

    # Charity
    ca = _get("charity_abuse")
    if ca:
        detect["charity_abuse"].update({
            "min_donors": int(ca.get("num_donors_range", [10, 40])[0]) if "num_donors_range" in ca else detect["charity_abuse"]["min_donors"],
            "donation_amount_range": [float(ca.get("donation_amount_range", [1000, 50000])[0]), float(ca.get("donation_amount_range", [1000, 50000])[1])],
            "donation_window_days": int(ca.get("donation_spread_days_range", [0, 14])[1]),
            "diversion_window_days": int(ca.get("diversion_delay_days_range", [3, 10])[1]),
            "diversion_retention_pct": 1 - float(ca.get("diversion_pct", 0.8)),
            "donation_channels": ca.get("donation_channels", detect["charity_abuse"]["donation_channels"]),
            "diversion_hour_range": ca.get("diversion_hour_range", detect["charity_abuse"]["diversion_hour_range"]),
            "min_total_donation": float(ca.get("min_total_donation", detect["charity_abuse"]["min_total_donation"])),
        })

    # Detector notebook expects typology names as keys.
    name_map = {
        "structuring": "structuring",
        "circular": "circular",
        "funnel": "funnel",
        "passthrough": "passthrough",
        "layering": "layering",
        "third_party_web": "third_party_web",
        "money_mule": "money_mule",
        "high_risk_corridor": "high_risk_corridor",
        "hawala": "hawala",
        "charity_abuse": "charity_abuse",
    }

    out: dict[str, dict[str, Any]] = {
        "structuring": detect["structuring"],
    }

    # Actually map to human typology names used in the detector.
    return {
        "structuring": detect["structuring"],
        "circular": detect["circular"],
        "funnel": detect["funnel"],
        "passthrough": detect["passthrough"],
        "layering": detect["layering"],
        "third_party_web": detect["third_party_web"],
        "money_mule": detect["money_mule"],
        "high_risk_corridor": detect["high_risk_corridor"],
        "hawala": detect["hawala"],
        "charity_abuse": detect["charity_abuse"],
    }


def _build_pipeline_from_settings(
    settings: dict[str, Any],
    pipe_section: str,
) -> dict[str, Any]:
    """Generic pipeline builder from settings.yaml section."""

    section = settings.get(pipe_section, {})

    pipeline = section.get("pipeline", [])

    # Keep as-is; notebooks/orchestrators consume this structure.
    return {
        "timeout_minutes": section.get("timeout_minutes", 60),
        "stop_on_failure": section.get("stop_on_failure", True),
        "save_executed_notebooks": section.get("save_executed_notebooks", True),
        "nbconvert_timeout_seconds": section.get("nbconvert_timeout_seconds"),
        "pipeline": pipeline,
    }


def build_training_pipeline(
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    orch = settings.get("orchestrator", {})
    # Training pipeline is anchored under orchestrator.pipeline
    return _build_pipeline_from_settings(settings, "orchestrator")


def build_inference_pipeline(
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    return _build_pipeline_from_settings(settings, "inference")


def ensure_notebook_path() -> tuple[
    dict[str, Any],
    dict[str, Path],
]:
    """Standard bootstrap for pipeline notebooks."""

    setup_project_path()

    settings = apply_env_overrides(
        load_settings()
    )

    paths = resolve_paths(settings)

    return settings, paths

