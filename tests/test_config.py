"""Testes do loader de configuração."""

import pytest

from libras.config import PATHS, load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    for section in ("dataset", "preprocess", "training", "inference", "feedback"):
        assert isinstance(getattr(cfg, section), dict)


def test_keypoint_dim_matches_extractor():
    pytest.importorskip("mediapipe")
    from libras.utils.mediapipe_holistic import TOTAL_DIM

    cfg = load_config("configs/default.yaml")
    assert cfg.preprocess["keypoint_dim"] == TOTAL_DIM


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/nao_existe.yaml")


def test_paths_under_project_root():
    assert PATHS["models"].parent == PATHS["artifacts"]
