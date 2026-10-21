"""
Vetor de keypoints a partir de landmarks brutos — fonte única do pré-processamento.

Este módulo NÃO importa cv2 nem mediapipe: é usado pela API (que recebe os
landmarks já extraídos pelo app/navegador) e pelo pipeline de treino (via
`libras.utils.mediapipe_holistic.extract_keypoints`, que delega para cá).

Formatos suportados (escolhidos pelo `keypoint_dim` do metadata do modelo):
  438  — atual: pose + face reduzida (lábios/sobrancelhas) + mãos,
         com normalização espacial (`normalize_spatial`)
  1662 — legado: pose + face completa (468) + mãos, sem normalização
         (formato dos modelos treinados antes de 31/08/2026)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Topologia do MediaPipe Holistic legado (mediapipe==0.10.9).
POSE_POINTS = 33
FACE_MESH_POINTS = 468
HAND_POINTS = 21

# Índices da face mesh mantidos no vetor: lábios + sobrancelhas
# (FACEMESH_LIPS | FACEMESH_LEFT_EYEBROW | FACEMESH_RIGHT_EYEBROW).
# Fixos aqui para não importar mediapipe; tests/test_keypoints.py confere
# que continuam iguais aos derivados das conexões do MediaPipe.
FACE_SUBSET_INDICES: list[int] = [
    0, 13, 14, 17, 37, 39, 40, 46, 52, 53, 55, 61, 63, 65, 66, 70, 78, 80, 81, 82,
    84, 87, 88, 91, 95, 105, 107, 146, 178, 181, 185, 191, 267, 269, 270, 276, 282,
    283, 285, 291, 293, 295, 296, 300, 308, 310, 311, 312, 314, 317, 318, 321, 324,
    334, 336, 375, 402, 405, 409, 415,
]

# Dimensões esperadas
POSE_DIM = POSE_POINTS * 4                      # x, y, z, visibility
FACE_DIM = len(FACE_SUBSET_INDICES) * 3         # x, y, z — só lábios + sobrancelhas
HAND_DIM = HAND_POINTS * 3
TOTAL_DIM = POSE_DIM + FACE_DIM + HAND_DIM + HAND_DIM

LEGACY_FACE_DIM = FACE_MESH_POINTS * 3
LEGACY_TOTAL_DIM = POSE_DIM + LEGACY_FACE_DIM + HAND_DIM + HAND_DIM

# Índices das regiões no vetor de TOTAL_DIM dimensões
REGION_SLICES = {
    "pose":       (0,                              POSE_DIM),
    "face":       (POSE_DIM,                       POSE_DIM + FACE_DIM),
    "left_hand":  (POSE_DIM + FACE_DIM,            POSE_DIM + FACE_DIM + HAND_DIM),
    "right_hand": (POSE_DIM + FACE_DIM + HAND_DIM, TOTAL_DIM),
}


# ── Montagem do vetor ────────────────────────────────────────

def _flat(points: np.ndarray | None, dim: int) -> np.ndarray:
    """Achata (N, C) → (N*C,), ou zeros(dim) se a parte não foi detectada."""
    if points is None:
        return np.zeros(dim)
    return np.asarray(points, dtype=np.float64).reshape(-1)


def keypoints_from_arrays(
    pose: np.ndarray | None,
    face: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
) -> np.ndarray:
    """
    Monta o vetor de TOTAL_DIM (438) dimensões a partir dos landmarks brutos.

    Parameters
    ----------
    pose       : (33, 4) x, y, z, visibility — ou None se não detectada
    face       : (468, 3) malha completa — ou None; só FACE_SUBSET_INDICES é usado
    left_hand  : (21, 3) — ou None
    right_hand : (21, 3) — ou None

    Partes ausentes viram zeros; o resultado passa por `normalize_spatial`.
    """
    face_subset = None if face is None else np.asarray(face)[FACE_SUBSET_INDICES]
    vec = np.concatenate([
        _flat(pose, POSE_DIM),
        _flat(face_subset, FACE_DIM),
        _flat(left_hand, HAND_DIM),
        _flat(right_hand, HAND_DIM),
    ])
    return normalize_spatial(vec)


def keypoints_from_arrays_legacy(
    pose: np.ndarray | None,
    face: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
) -> np.ndarray:
    """
    Monta o vetor legado de LEGACY_TOTAL_DIM (1662) dimensões.

    Face completa (468 pontos) e sem normalização espacial — igual ao
    `extract_keypoints` usado para treinar os modelos anteriores a 31/08/2026.
    """
    face_full = None if face is None else np.asarray(face)[:FACE_MESH_POINTS]
    return np.concatenate([
        _flat(pose, POSE_DIM),
        _flat(face_full, LEGACY_FACE_DIM),
        _flat(left_hand, HAND_DIM),
        _flat(right_hand, HAND_DIM),
    ])


FeatureBuilder = Callable[
    [np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None],
    np.ndarray,
]

FEATURE_BUILDERS: dict[int, FeatureBuilder] = {
    TOTAL_DIM: keypoints_from_arrays,
    LEGACY_TOTAL_DIM: keypoints_from_arrays_legacy,
}


def get_feature_builder(keypoint_dim: int) -> FeatureBuilder:
    """Função de montagem compatível com um modelo de `keypoint_dim` entradas."""
    try:
        return FEATURE_BUILDERS[keypoint_dim]
    except KeyError:
        raise ValueError(
            f"keypoint_dim={keypoint_dim} nao suportado "
            f"(suportados: {sorted(FEATURE_BUILDERS)})"
        ) from None


# ── Normalização espacial ────────────────────────────────────

# Índices dos ombros na topologia padrão do MediaPipe Pose.
_LEFT_SHOULDER_IDX = 11
_RIGHT_SHOULDER_IDX = 12


def normalize_spatial(vec: np.ndarray) -> np.ndarray:
    """
    Centraliza os keypoints no meio-ombro e escala pela largura dos ombros.

    Sem isso, o mesmo sinal gera vetores bem diferentes só por o usuário
    estar mais perto/longe da câmera ou fora do centro do quadro — variância
    que o augmentation (spatial_jitter/scale_keypoints) tenta compensar
    artificialmente, mas que é melhor eliminar na origem. Aplicado a x/y
    (translação + escala) e z (só escala) de pose, face e mãos; a coluna de
    visibility da pose não é espacial e fica intacta.
    """
    l = vec[_LEFT_SHOULDER_IDX * 4: _LEFT_SHOULDER_IDX * 4 + 2]
    r = vec[_RIGHT_SHOULDER_IDX * 4: _RIGHT_SHOULDER_IDX * 4 + 2]
    cx, cy = (l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0
    scale = float(np.hypot(r[0] - l[0], r[1] - l[1]))

    if scale < 1e-6:
        # Ombros não detectados (pose ausente) — vetor já é zeros, mantém.
        return vec

    out = vec.copy()
    for start, dim, stride in (
        (0, POSE_DIM, 4),
        (POSE_DIM, FACE_DIM, 3),
        (POSE_DIM + FACE_DIM, HAND_DIM, 3),
        (POSE_DIM + FACE_DIM + HAND_DIM, HAND_DIM, 3),
    ):
        block = out[start: start + dim].reshape(-1, stride)
        block[:, 0] = (block[:, 0] - cx) / scale
        block[:, 1] = (block[:, 1] - cy) / scale
        block[:, 2] = block[:, 2] / scale
        out[start: start + dim] = block.flatten()
    return out


# ── Normalização temporal ────────────────────────────────────

def normalize_sequence(frames: np.ndarray, target: int) -> np.ndarray:
    """
    Garante exatamente `target` frames por sequência.
    Mais frames: amostragem uniforme. Menos: padding com último frame.
    """
    if len(frames) == 0:
        return np.zeros((target, TOTAL_DIM))
    if len(frames) >= target:
        idx = np.linspace(0, len(frames) - 1, target, dtype=int)
        return frames[idx]
    pad = np.repeat(frames[-1:], target - len(frames), axis=0)
    return np.concatenate([frames, pad], axis=0)
