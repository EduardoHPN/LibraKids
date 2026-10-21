"""
Wrapper sobre o MediaPipe Holistic.

Centraliza:
  • Extração de keypoints (pose + subconjunto de face + mãos)
  • Desenho de landmarks
  • Parsing de nomes de arquivo do V-LIBRASIL

As partes puras (dimensões, subconjunto da face, normalizações e montagem
do vetor) ficam em `libras.features.keypoints`, sem cv2/mediapipe, e são
reexportadas aqui para não quebrar os scripts de treino.

Compatível com mediapipe==0.10.9
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from libras.features.keypoints import (  # noqa: F401 — reexportados
    FACE_DIM,
    FACE_SUBSET_INDICES,
    HAND_DIM,
    POSE_DIM,
    REGION_SLICES,
    TOTAL_DIM,
    keypoints_from_arrays,
    normalize_sequence,
    normalize_spatial,
)

# Handles globais do MediaPipe
_mp_holistic = mp.solutions.holistic
_mp_drawing = mp.solutions.drawing_utils


# ── Detecção ────────────────────────────────────────────────

def detect(image: np.ndarray, model: Any) -> tuple[np.ndarray, Any]:
    """Processa um frame BGR com o modelo MediaPipe Holistic."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def create_holistic(
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
    model_complexity: int = 1,
):
    """Cria instância do Holistic. Use como context manager."""
    return _mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )


# ── Extração de keypoints ────────────────────────────────────

def extract_keypoints(results: Any) -> np.ndarray:
    """
    Extrai vetor de keypoints de um resultado do MediaPipe.

    Composição (TOTAL_DIM):
      pose         33 × 4  = 132
      face (subset) 60 × 3 = 180   — só lábios + sobrancelhas, ver FACE_SUBSET_INDICES
      left_hand    21 × 3  = 63
      right_hand   21 × 3  = 63

    Só converte o resultado do MediaPipe em arrays; a montagem do vetor é
    `keypoints_from_arrays` — a mesma usada pela API com landmarks do app.
    """
    def _points(landmarks: Any, with_visibility: bool = False) -> np.ndarray | None:
        if not landmarks:
            return None
        if with_visibility:
            return np.array([[r.x, r.y, r.z, r.visibility] for r in landmarks.landmark])
        return np.array([[r.x, r.y, r.z] for r in landmarks.landmark])

    return keypoints_from_arrays(
        pose=_points(results.pose_landmarks, with_visibility=True),
        face=_points(results.face_landmarks),
        left_hand=_points(results.left_hand_landmarks),
        right_hand=_points(results.right_hand_landmarks),
    )


# ── Dataset V-LIBRASIL ────────────────────────────────────────

def parse_class_name(filename: str | Path) -> str:
    """
    Extrai a classe do padrão V-LIBRASIL.

    >>> parse_class_name("Abacaxi_Articulador1.mp4")
    'Abacaxi'
    >>> parse_class_name("À noite toda_Articulador2.mp4")
    'À noite toda'
    """
    name = Path(filename).stem
    parts = name.rsplit("_Articulador", 1)
    return parts[0].strip() if len(parts) == 2 else name


# ── Visualização ────────────────────────────────────────────

def draw_landmarks(image: np.ndarray, results: Any) -> None:
    """Desenha todos os landmarks no frame com cores por região."""
    if results.face_landmarks:
        _mp_drawing.draw_landmarks(
            image, results.face_landmarks,
            _mp_holistic.FACEMESH_CONTOURS,
            _mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            _mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        _mp_drawing.draw_landmarks(
            image, results.pose_landmarks, _mp_holistic.POSE_CONNECTIONS,
            _mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            _mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        _mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, _mp_holistic.HAND_CONNECTIONS,
            _mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            _mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        _mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, _mp_holistic.HAND_CONNECTIONS,
            _mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            _mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )
