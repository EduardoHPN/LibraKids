"""
Leitura do cache de landmarks brutos gerado por scripts/11_extract_landmarks.py.

Cada .npz guarda, por frame, os landmarks de pose/face/mãos (NaN onde a parte
não foi detectada) e as máscaras `<parte>_present`. Daqui saem os vetores de
treino (scripts/13) e da validação (scripts/12), montados pelas mesmas
funções de `libras.features.keypoints` que a API usa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from libras.features.keypoints import FACE_MESH_POINTS, FeatureBuilder, normalize_sequence

PARTS = ("pose", "face", "left_hand", "right_hand")


def load_landmark_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Carrega um .npz do cache; a face do Tasks (478 pontos) vira 468, igual à API."""
    d = dict(np.load(path))
    d["face"] = d["face"][:, :FACE_MESH_POINTS]
    return d


def frame_parts(d: dict[str, np.ndarray], t: int) -> tuple:
    """Partes do frame t no formato de keypoints_from_arrays (None se ausente)."""
    return tuple(d[p][t] if d[f"{p}_present"][t] else None for p in PARTS)


def sequence_features(d: dict[str, np.ndarray], builder: FeatureBuilder, num_frames: int) -> np.ndarray:
    """Vetores de todos os frames do vídeo, reamostrados para `num_frames`."""
    frames = np.stack([builder(*frame_parts(d, t)) for t in range(len(d["pose"]))])
    return normalize_sequence(frames, num_frames)
