"""Testes da montagem de features a partir de landmarks brutos (libras.features)."""

from types import SimpleNamespace

import numpy as np
import pytest

from libras.features.keypoints import (
    FACE_SUBSET_INDICES, LEGACY_TOTAL_DIM, POSE_DIM, REGION_SLICES, TOTAL_DIM,
    get_feature_builder, keypoints_from_arrays, keypoints_from_arrays_legacy,
)


def _arrays(rng):
    pose = rng.random((33, 4))
    pose[11, :2] = [0.6, 0.5]  # ombros afastados -> normalização ativa
    pose[12, :2] = [0.4, 0.5]
    return pose, rng.random((468, 3)), rng.random((21, 3)), rng.random((21, 3))


def _as_results(pose, face, left, right):
    """Imita o objeto de resultado do MediaPipe Holistic legado."""
    def lms(arr):
        if arr is None:
            return None
        cols = ("x", "y", "z", "visibility")
        return SimpleNamespace(landmark=[
            SimpleNamespace(**dict(zip(cols, row))) for row in arr
        ])
    return SimpleNamespace(
        pose_landmarks=lms(pose), face_landmarks=lms(face),
        left_hand_landmarks=lms(left), right_hand_landmarks=lms(right),
    )


class TestFaceSubset:
    def test_matches_mediapipe_connections(self):
        mp = pytest.importorskip("mediapipe")
        fm = mp.solutions.face_mesh
        connections = fm.FACEMESH_LIPS | fm.FACEMESH_LEFT_EYEBROW | fm.FACEMESH_RIGHT_EYEBROW
        assert FACE_SUBSET_INDICES == sorted({i for pair in connections for i in pair})


class TestKeypointsFromArrays:
    def test_same_as_extract_keypoints(self):
        """API (landmarks do app) e treino (resultado do MediaPipe) geram o mesmo vetor."""
        pytest.importorskip("mediapipe")
        from libras.utils.mediapipe_holistic import extract_keypoints

        rng = np.random.default_rng(0)
        for parts in (_arrays(rng), (*_arrays(rng)[:2], None, None)):
            expected = extract_keypoints(_as_results(*parts))
            assert np.allclose(keypoints_from_arrays(*parts), expected)

    def test_shape(self):
        out = keypoints_from_arrays(*_arrays(np.random.default_rng(1)))
        assert out.shape == (TOTAL_DIM,)

    def test_uses_only_face_subset(self):
        rng = np.random.default_rng(2)
        pose, face, left, right = _arrays(rng)
        face_changed = face.copy()
        others = np.setdiff1d(np.arange(468), FACE_SUBSET_INDICES)
        face_changed[others] += 10.0
        assert np.allclose(
            keypoints_from_arrays(pose, face, left, right),
            keypoints_from_arrays(pose, face_changed, left, right),
        )

    def test_nothing_detected_is_zeros(self):
        assert np.array_equal(keypoints_from_arrays(None, None, None, None), np.zeros(TOTAL_DIM))


class TestLegacyBuilder:
    def test_full_face_without_normalization(self):
        pose, face, left, right = _arrays(np.random.default_rng(3))
        out = keypoints_from_arrays_legacy(pose, face, left, right)
        assert out.shape == (LEGACY_TOTAL_DIM,)
        assert np.allclose(out, np.concatenate([p.flatten() for p in (pose, face, left, right)]))

    def test_face_from_tasks_truncated_to_468(self):
        pose, face, left, right = _arrays(np.random.default_rng(4))
        face_478 = np.vstack([face, np.ones((10, 3))])
        assert np.allclose(
            keypoints_from_arrays_legacy(pose, face_478, left, right),
            keypoints_from_arrays_legacy(pose, face, left, right),
        )

    def test_missing_parts_are_zeros(self):
        out = keypoints_from_arrays_legacy(None, None, None, None)
        assert np.array_equal(out, np.zeros(LEGACY_TOTAL_DIM))


class TestGetFeatureBuilder:
    def test_current(self):
        assert get_feature_builder(TOTAL_DIM) is keypoints_from_arrays

    def test_legacy(self):
        assert get_feature_builder(LEGACY_TOTAL_DIM) is keypoints_from_arrays_legacy

    def test_unsupported(self):
        with pytest.raises(ValueError, match="nao suportado"):
            get_feature_builder(999)


def test_region_slices_cover_vector():
    assert REGION_SLICES["pose"] == (0, POSE_DIM)
    assert REGION_SLICES["right_hand"][1] == TOTAL_DIM
