"""
Validação dos payloads de landmarks enviados pelo app/navegador.

Formato documentado em docs/API_CONTRACT.md. As funções convertem o JSON
em arrays numpy prontos para `libras.features.keypoints` ou levantam
`PayloadError` com uma mensagem clara (a rota responde 400).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from libras.features.keypoints import FACE_MESH_POINTS, HAND_POINTS, POSE_POINTS

# Limites de frames por requisição de /predict_sequence. O servidor
# reamostra para o num_frames do modelo (30); acima de MAX_FRAMES o payload
# fica grande demais para o MAX_CONTENT_LENGTH da API.
MIN_FRAMES = 10
MAX_FRAMES = 60

# O MediaPipe Tasks devolve a malha da face com 478 pontos (468 + 10 da
# íris); os 468 primeiros coincidem com os do Holistic legado do treino.
_FACE_POINTS_ACCEPTED = (FACE_MESH_POINTS, FACE_MESH_POINTS + 10)

_TARGET_LETTER_MAX_LEN = 10

# Landmarks de um frame: (pose, face, left_hand, right_hand), cada um ou None.
FrameLandmarks = tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]


class PayloadError(ValueError):
    """Payload inválido — a mensagem vai para o cliente."""


class NoLandmarksError(PayloadError):
    """Payload bem formado, mas sem nenhuma mão detectada para classificar."""


def _points(
    value: Any,
    where: str,
    n_points: tuple[int, ...],
    n_coords: tuple[int, ...],
) -> np.ndarray | None:
    """Converte uma lista de pontos em array (N, C), validando shape e tipos."""
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(p, list) for p in value):
        raise PayloadError(f"{where} deve ser uma lista de pontos ou null.")
    # Pontos de tamanhos diferentes: checado antes do np.asarray, que só
    # avisa (numpy 1.23) ou levanta ValueError (numpy >= 1.24).
    if len({len(p) for p in value}) > 1:
        raise PayloadError(f"{where} deve conter apenas numeros, em pontos de mesmo tamanho.")

    arr = np.asarray(value)
    # 'b' = booleanos; 'U' = strings; 'O' = aninhamento extra — nada disso
    # é coordenada.
    if arr.dtype.kind not in "iuf":
        raise PayloadError(f"{where} deve conter apenas numeros, em pontos de mesmo tamanho.")
    if arr.ndim != 2 or arr.shape[0] not in n_points or arr.shape[1] not in n_coords:
        raise PayloadError(
            f"{where} deve ter {' ou '.join(map(str, n_points))} pontos com "
            f"{' ou '.join(map(str, n_coords))} coordenadas (recebido: shape {list(arr.shape)})."
        )
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise PayloadError(f"{where} contem valores nao finitos (NaN/Infinity).")
    return arr


def parse_sequence_payload(data: Any) -> list[FrameLandmarks]:
    """Valida o corpo de POST /predict_sequence e devolve os landmarks por frame."""
    if not isinstance(data, dict) or "frames" not in data:
        raise PayloadError("Corpo deve ser um objeto JSON com o campo 'frames'.")

    frames = data["frames"]
    if not isinstance(frames, list):
        raise PayloadError("'frames' deve ser uma lista.")
    if not MIN_FRAMES <= len(frames) <= MAX_FRAMES:
        raise PayloadError(
            f"'frames' deve ter entre {MIN_FRAMES} e {MAX_FRAMES} frames "
            f"(recebido: {len(frames)})."
        )

    parsed: list[FrameLandmarks] = []
    for i, frame in enumerate(frames):
        where = f"frames[{i}]"
        if not isinstance(frame, dict):
            raise PayloadError(f"{where} deve ser um objeto.")

        pose = _points(frame.get("pose"), f"{where}.pose", (POSE_POINTS,), (4,))
        face = _points(frame.get("face"), f"{where}.face", _FACE_POINTS_ACCEPTED, (3,))
        left = _points(frame.get("left_hand"), f"{where}.left_hand", (HAND_POINTS,), (3,))
        right = _points(frame.get("right_hand"), f"{where}.right_hand", (HAND_POINTS,), (3,))

        if face is not None:
            face = face[:FACE_MESH_POINTS]
        parsed.append((pose, face, left, right))

    if all(left is None and right is None for _, _, left, right in parsed):
        raise NoLandmarksError("Nenhuma mao detectada na sequencia.")
    return parsed


def parse_letter_payload(data: Any) -> tuple[np.ndarray | None, str]:
    """
    Valida o corpo de POST /predict_letter_landmarks.

    Retorna (hand, target_letter): hand é (21, 2) com x, y — o modelo de
    letras usa só x/y; um z enviado a mais é descartado — ou None se nenhuma
    mão foi detectada.
    """
    if not isinstance(data, dict):
        raise PayloadError("Corpo deve ser um objeto JSON.")
    if "hand" not in data or "target_letter" not in data:
        raise PayloadError("Campos 'hand' e 'target_letter' sao obrigatorios.")

    target_letter = data["target_letter"]
    if not isinstance(target_letter, str) or not 0 < len(target_letter.strip()) <= _TARGET_LETTER_MAX_LEN:
        raise PayloadError(
            f"'target_letter' deve ser um texto de 1 a {_TARGET_LETTER_MAX_LEN} caracteres."
        )

    hand = _points(data["hand"], "hand", (HAND_POINTS,), (2, 3))
    if hand is not None:
        hand = hand[:, :2]
    return hand, target_letter.strip()
