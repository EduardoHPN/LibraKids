"""Testes da extração de keypoints e da normalização espacial."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mediapipe")

from libras.utils.mediapipe_holistic import (  # noqa: E402
    FACE_DIM, FACE_SUBSET_INDICES, HAND_DIM, POSE_DIM, REGION_SLICES, TOTAL_DIM,
    extract_keypoints, normalize_spatial,
)


def _lm(x=0.0, y=0.0, z=0.0, visibility=1.0):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _landmarks(n, **kwargs):
    return SimpleNamespace(landmark=[_lm(**kwargs) for _ in range(n)])


def _pose_with_shoulders(left=(0.3, 0.5), right=(0.7, 0.5)):
    pose = _landmarks(33)
    pose.landmark[11] = _lm(*left)
    pose.landmark[12] = _lm(*right)
    return pose


def _results(pose=None, face=None, left=None, right=None):
    return SimpleNamespace(
        pose_landmarks=pose, face_landmarks=face,
        left_hand_landmarks=left, right_hand_landmarks=right,
    )


class TestDimensions:
    def test_face_subset_is_sorted_unique_60(self):
        assert len(FACE_SUBSET_INDICES) == 60
        assert FACE_SUBSET_INDICES == sorted(set(FACE_SUBSET_INDICES))

    def test_dims_consistent(self):
        assert FACE_DIM == 60 * 3
        assert TOTAL_DIM == POSE_DIM + FACE_DIM + 2 * HAND_DIM == 438


class TestExtractKeypoints:
    def test_nothing_detected_is_zeros(self):
        out = extract_keypoints(_results())
        assert out.shape == (TOTAL_DIM,)
        assert np.array_equal(out, np.zeros(TOTAL_DIM))

    def test_full_result_shape_and_finite(self):
        out = extract_keypoints(_results(
            pose=_pose_with_shoulders(),
            face=_landmarks(468, x=0.5, y=0.4),
            left=_landmarks(21, x=0.3, y=0.6),
            right=_landmarks(21, x=0.7, y=0.6),
        ))
        assert out.shape == (TOTAL_DIM,)
        assert np.isfinite(out).all()

    def test_output_is_shoulder_normalized(self):
        out = extract_keypoints(_results(pose=_pose_with_shoulders()))
        l = out[11 * 4: 11 * 4 + 2]
        r = out[12 * 4: 12 * 4 + 2]
        assert np.allclose((l + r) / 2, 0.0, atol=1e-6)
        assert np.isclose(np.hypot(*(r - l)), 1.0)

    def test_missing_hand_is_centered_zeros(self):
        # Mão ausente entra como zeros e passa pela normalização: x vira
        # -cx/scale (cx=0.5, scale=0.4). Documenta o comportamento atual.
        out = extract_keypoints(_results(pose=_pose_with_shoulders()))
        start, end = REGION_SLICES["left_hand"]
        hand = out[start:end].reshape(-1, 3)
        assert np.allclose(hand[:, 0], -0.5 / 0.4)
        assert np.allclose(hand[:, 2], 0.0)

    def test_face_uses_only_subset_indices(self):
        face = _landmarks(468)
        for i, idx in enumerate(FACE_SUBSET_INDICES):
            face.landmark[idx] = _lm(x=0.5 + i * 0.001)
        out = extract_keypoints(_results(pose=_pose_with_shoulders(), face=face))
        start, end = REGION_SLICES["face"]
        xs = out[start:end].reshape(-1, 3)[:, 0]
        assert len(set(np.round(xs, 6))) == 60


class TestNormalizeSpatialInvariance:
    @staticmethod
    def _vec(rng):
        vec = rng.random(TOTAL_DIM)
        vec[11 * 4: 11 * 4 + 2] = [0.3, 0.5]
        vec[12 * 4: 12 * 4 + 2] = [0.7, 0.5]
        return vec

    @staticmethod
    def _transform(vec, dx, dy, k):
        out = vec.copy()
        for start, dim, stride in (
            (0, POSE_DIM, 4), (POSE_DIM, FACE_DIM, 3),
            (POSE_DIM + FACE_DIM, HAND_DIM, 3),
            (POSE_DIM + FACE_DIM + HAND_DIM, HAND_DIM, 3),
        ):
            b = out[start:start + dim].reshape(-1, stride)
            b[:, 0] = b[:, 0] * k + dx
            b[:, 1] = b[:, 1] * k + dy
            b[:, 2] = b[:, 2] * k
            out[start:start + dim] = b.flatten()
        return out

    def test_translation_and_scale_invariant(self):
        vec = self._vec(np.random.default_rng(0))
        moved = self._transform(vec, dx=0.1, dy=-0.2, k=1.7)
        assert np.allclose(normalize_spatial(vec), normalize_spatial(moved), atol=1e-9)

    def test_input_not_mutated(self):
        vec = self._vec(np.random.default_rng(1))
        before = vec.copy()
        normalize_spatial(vec)
        assert np.array_equal(vec, before)

    def test_z_only_scaled(self):
        vec = self._vec(np.random.default_rng(2))
        out = normalize_spatial(vec)
        # Ombros a 0.4 de distância -> z dividido por 0.4, sem translação.
        assert np.isclose(out[2], vec[2] / 0.4)
