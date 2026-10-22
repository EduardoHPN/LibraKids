"""Testes do retreino com landmarks do MediaPipe Tasks: formato sem z, augmentation e scripts/13."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from libras.data.augmentation import augment_training_set, spatial_jitter
from libras.features.keypoints import (
    NO_POSE_Z_TOTAL_DIM, POSE_DIM, TOTAL_DIM, drop_pose_z, get_feature_builder,
    keypoints_from_arrays, keypoints_from_arrays_no_pose_z,
)
from libras.features.landmark_cache import frame_parts, load_landmark_cache, sequence_features

ROOT = Path(__file__).resolve().parents[1]


def _parts(rng):
    pose = rng.random((33, 4))
    pose[11, :2], pose[12, :2] = [0.6, 0.5], [0.4, 0.5]
    return pose, rng.random((478, 3)), rng.random((21, 3)), None


class TestNoPoseZ:
    def test_dimension(self):
        assert NO_POSE_Z_TOTAL_DIM == TOTAL_DIM - 33 == 405

    def test_equals_438_without_pose_z(self):
        parts = _parts(np.random.default_rng(0))
        full = keypoints_from_arrays(*parts)
        no_z = keypoints_from_arrays_no_pose_z(*parts)
        assert no_z.shape == (NO_POSE_Z_TOTAL_DIM,)
        pose_no_z = full[:POSE_DIM].reshape(33, 4)[:, [0, 1, 3]].ravel()
        assert np.allclose(no_z[:99], pose_no_z)       # pose: x, y, vis
        assert np.allclose(no_z[99:], full[POSE_DIM:])  # face e mãos intactas

    def test_drop_pose_z_on_sequences(self):
        seq = np.random.default_rng(1).random((30, TOTAL_DIM))
        assert drop_pose_z(seq).shape == (30, NO_POSE_Z_TOTAL_DIM)

    def test_builder_registered(self):
        assert get_feature_builder(NO_POSE_Z_TOTAL_DIM) is keypoints_from_arrays_no_pose_z


class TestAugmentationFormats:
    @pytest.mark.parametrize("dim, pose_cols", [(TOTAL_DIM, 4), (NO_POSE_Z_TOTAL_DIM, 3)])
    def test_spatial_jitter_only_moves_xy(self, dim, pose_cols):
        seq = np.random.default_rng(2).random((30, dim)).astype(np.float32)
        out = spatial_jitter(seq, max_shift=0.5)
        delta = (out - seq)[:, : 33 * pose_cols].reshape(30, 33, pose_cols)
        assert np.allclose(delta[..., 2:], 0, atol=1e-6)     # z / visibility intactos
        assert not np.allclose(delta[..., :2], 0)
        rest = (out - seq)[:, 33 * pose_cols:].reshape(30, -1, 3)
        assert np.allclose(rest[..., 2], 0, atol=1e-6)       # z de face/mãos intacto

    def test_spatial_jitter_rejects_unknown_dim(self):
        with pytest.raises(ValueError, match="nao suportada"):
            spatial_jitter(np.zeros((30, 1662), dtype=np.float32))

    def test_augment_training_set_405(self):
        X = np.random.default_rng(3).random((4, 30, NO_POSE_Z_TOTAL_DIM)).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]
        X_aug, y_aug = augment_training_set(X, y, multiplier=3)
        assert X_aug.shape == (12, 30, NO_POSE_Z_TOTAL_DIM) and y_aug.shape == (12, 2)


def _write_cache(root: Path, n_frames: int = 12) -> Path:
    """Cache sintético no formato do scripts/11 (2 classes × 1 vídeo, backend tasks)."""
    rng = np.random.default_rng(4)
    for cls in ("Oi", "Sim"):
        pose = rng.random((n_frames, 33, 4)).astype(np.float32)
        pose[:, 11, :2], pose[:, 12, :2] = [0.6, 0.5], [0.4, 0.5]
        right = rng.random((n_frames, 21, 3)).astype(np.float32)
        left = np.full((n_frames, 21, 3), np.nan, dtype=np.float32)
        data = {
            "pose": pose, "face": rng.random((n_frames, 478, 3)).astype(np.float32),
            "left_hand": left, "right_hand": right,
            "pose_present": np.ones(n_frames, bool), "face_present": np.ones(n_frames, bool),
            "left_hand_present": np.zeros(n_frames, bool), "right_hand_present": np.ones(n_frames, bool),
            "fps": np.float32(30), "backend": np.array("tasks"), "mediapipe_version": np.array("0.10.35"),
        }
        (root / "tasks" / cls).mkdir(parents=True)
        np.savez_compressed(root / "tasks" / cls / f"{cls}_01.npz", **data)
    return root


class TestLandmarkCache:
    def test_face_truncated_and_missing_parts(self, tmp_path):
        cache = _write_cache(tmp_path)
        d = load_landmark_cache(next((cache / "tasks").glob("*/*.npz")))
        assert d["face"].shape[1] == 468
        pose, face, left, right = frame_parts(d, 0)
        assert left is None and right is not None and face.shape == (468, 3)
        seq = sequence_features(d, keypoints_from_arrays, 30)
        assert seq.shape == (30, TOTAL_DIM) and np.isfinite(seq).all()


@pytest.mark.parametrize("extra, dim", [([], TOTAL_DIM), (["--no-pose-z"], NO_POSE_Z_TOTAL_DIM)])
def test_build_training_set_script(tmp_path, extra, dim):
    cache = _write_cache(tmp_path / "cache")
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "13_build_training_set.py"),
         "--cache", str(cache), "--out", str(out), *extra],
        capture_output=True, text=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr[-2000:]

    files = sorted(out.glob("*/*.npy"))
    assert [f.parent.name for f in files] == ["Oi", "Sim"]
    arr = np.load(files[0])
    assert arr.shape == (30, dim) and arr.dtype == np.float32

    info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["keypoint_dim"] == dim
    assert info["pose_z"] is (dim == TOTAL_DIM)
    assert info["mediapipe_versions"] == ["0.10.35"]
    assert info["per_class"] == {"Oi": 1, "Sim": 1}
