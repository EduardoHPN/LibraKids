"""Testes do data augmentation de sequências de keypoints."""

import numpy as np
import pytest

pytest.importorskip("mediapipe")

from libras.data.augmentation import (  # noqa: E402
    add_gaussian_noise, augment_sequence, augment_training_set,
    scale_keypoints, spatial_jitter, time_warp,
)
from libras.utils.mediapipe_holistic import POSE_DIM, TOTAL_DIM  # noqa: E402

FRAMES = 30


@pytest.fixture
def seq():
    return np.random.default_rng(0).random((FRAMES, TOTAL_DIM)).astype(np.float32)


class TestTransforms:
    @pytest.mark.parametrize(
        "fn", [add_gaussian_noise, time_warp, spatial_jitter, scale_keypoints, augment_sequence]
    )
    def test_preserves_shape(self, fn, seq):
        assert fn(seq).shape == (FRAMES, TOTAL_DIM)

    def test_spatial_jitter_keeps_z_and_visibility(self, seq):
        out = spatial_jitter(seq, max_shift=0.5)
        pose_in = seq[:, :POSE_DIM].reshape(FRAMES, -1, 4)
        pose_out = out[:, :POSE_DIM].reshape(FRAMES, -1, 4)
        assert np.allclose(pose_in[..., 2:], pose_out[..., 2:])
        assert not np.allclose(pose_in[..., :2], pose_out[..., :2])

    def test_augment_sequence_does_not_mutate_input(self, seq):
        before = seq.copy()
        augment_sequence(seq, prob=1.0)
        assert np.array_equal(seq, before)


class TestAugmentTrainingSet:
    @pytest.fixture
    def data(self, seq):
        X = np.stack([seq, seq + 1, seq + 2])
        y = np.eye(3, dtype=np.float32)
        return X, y

    def test_size_and_shapes(self, data):
        X, y = data
        X_aug, y_aug = augment_training_set(X, y, multiplier=4)
        assert X_aug.shape == (12, FRAMES, TOTAL_DIM)
        assert y_aug.shape == (12, 3)

    def test_labels_balanced(self, data):
        X, y = data
        _, y_aug = augment_training_set(X, y, multiplier=4)
        assert np.array_equal(y_aug.sum(axis=0), [4, 4, 4])

    def test_deterministic_with_seed(self, data):
        X, y = data
        a, ya = augment_training_set(X, y, multiplier=3, random_seed=7)
        b, yb = augment_training_set(X, y, multiplier=3, random_seed=7)
        assert np.array_equal(a, b)
        assert np.array_equal(ya, yb)
