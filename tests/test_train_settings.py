"""Training script settings/resolution tests."""

import argparse

import pytest
import yaml

from src import train


def _args(**overrides):
    base = {
        "phases": None,
        "phase": None,
        "tier": None,
        "resume": None,
        "steps": None,
        "lr": None,
        "lr_min": None,
        "warmup_steps": None,
        "bs": None,
        "save_dir": None,
        "metrics_file": None,
        "no_plots": False,
        "tokenizer": None,
        "data_phase": None,
        "val_data_phase": None,
        "seed": None,
        "config": None,
        "preset": None,
        "output_root": None,
        "run_name": None,
        "save_interval": None,
        "plot_interval": None,
        "val_interval": None,
        "text_sample_interval": None,
        "log_interval": None,
        "snapshot": None,
        "d_dec": None,
        "decoder_layers": None,
        "thalamic_tokens": None,
        "n_heads_decoder": None,
        "compile": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_data_phase_accepts_custom_mixed_case_keys():
    assert train._resolve_data_phase("C", "B-diverse") == "B-diverse"
    assert train._resolve_data_phase("C", "b-diverse") == "B-diverse"
    assert train._resolve_data_phase("C", "longctx") == "longctx"


def test_resolve_data_phase_unknown_raises():
    with pytest.raises(ValueError, match="Unknown data phase"):
        train._resolve_data_phase("C", "nope-not-a-phase")


def test_resolve_settings_reads_data_phase_from_preset(tmp_path):
    cfg_path = tmp_path / "train.yaml"
    payload = {
        "defaults": {
            "data_phase": "B-diverse",
            "val_data_phase": "longctx",
        },
        "presets": {
            "p1": {
                "tier": "a",
            }
        },
        "default_preset": "p1",
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    settings = train.resolve_settings(_args(config=str(cfg_path)))
    assert settings["data_phase"] == "B-diverse"
    assert settings["val_data_phase"] == "longctx"


def test_resolve_settings_cli_overrides_data_phase(tmp_path):
    cfg_path = tmp_path / "train.yaml"
    payload = {
        "presets": {
            "p1": {
                "data_phase": "B-diverse",
                "val_data_phase": "longctx",
            }
        }
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    settings = train.resolve_settings(
        _args(
            config=str(cfg_path),
            preset="p1",
            data_phase="C-synthetic",
            val_data_phase="A",
        )
    )
    assert settings["data_phase"] == "C-synthetic"
    assert settings["val_data_phase"] == "A"

