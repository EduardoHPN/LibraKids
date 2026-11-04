"""Testes do módulo de pré-processamento."""

import numpy as np
import pytest

pytest.importorskip("mediapipe")

from libras.utils.mediapipe_holistic import (  # noqa: E402
    REGION_SLICES, TOTAL_DIM, normalize_sequence, normalize_spatial, parse_class_name,
)


class TestParseClassName:
    def test_simple(self):
        assert parse_class_name("Abacaxi_Articulador1.mp4") == "Abacaxi"

    def test_with_spaces(self):
        assert parse_class_name("À noite toda_Articulador2.mp4") == "À noite toda"

    def test_without_articulador(self):
        assert parse_class_name("OutroFormato.mp4") == "OutroFormato"


class TestNormalizeSequence:
    def test_padding(self):
        frames = np.ones((10, TOTAL_DIM))
        result = normalize_sequence(frames, target=30)
        assert result.shape == (30, TOTAL_DIM)

    def test_truncation(self):
        frames = np.ones((50, TOTAL_DIM))
        result = normalize_sequence(frames, target=30)
        assert result.shape == (30, TOTAL_DIM)

    def test_empty(self):
        frames = np.array([]).reshape(0, TOTAL_DIM)
        result = normalize_sequence(frames, target=30)
        assert result.shape == (30, TOTAL_DIM)
        assert (result == 0).all()


class TestNormalizeSpatial:
    def test_centers_and_scales_shoulders(self):
        vec = np.zeros(TOTAL_DIM)
        vec[11 * 4: 11 * 4 + 2] = [0.3, 0.5]  # ombro esquerdo
        vec[12 * 4: 12 * 4 + 2] = [0.7, 0.5]  # ombro direito
        out = normalize_spatial(vec)

        l = out[11 * 4: 11 * 4 + 2]
        r = out[12 * 4: 12 * 4 + 2]
        assert np.allclose((l + r) / 2, [0.0, 0.0], atol=1e-6)
        assert np.isclose(np.hypot(*(r - l)), 1.0, atol=1e-6)

    def test_visibility_column_untouched(self):
        vec = np.zeros(TOTAL_DIM)
        vec[11 * 4: 11 * 4 + 2] = [0.3, 0.5]
        vec[12 * 4: 12 * 4 + 2] = [0.7, 0.5]
        vec[0 * 4 + 3] = 0.9  # visibility do primeiro landmark de pose
        out = normalize_spatial(vec)
        assert out[0 * 4 + 3] == 0.9

    def test_degenerate_shoulders_returns_unchanged(self):
        vec = np.zeros(TOTAL_DIM)  # ombros coincidentes em (0,0) -> scale=0
        out = normalize_spatial(vec)
        assert np.array_equal(out, vec)


class TestRegionSlices:
    def test_total_dimension(self):
        last_end = max(end for _, end in REGION_SLICES.values())
        assert last_end == TOTAL_DIM

    def test_no_overlap(self):
        slices = sorted(REGION_SLICES.values())
        for (_, end_a), (start_b, _) in zip(slices, slices[1:]):
            assert end_a == start_b
