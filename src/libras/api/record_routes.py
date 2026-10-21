"""Rota de coleta de dados de letras — /record_data.

Grava recortes da mão em disco (data/raw_custom/letters/) para montar o
dataset de letras. Só é registrada quando LIBRAS_ENABLE_RECORD_DATA=1:
o disco do container é efêmero, então em produção ela fica desligada.

É o único módulo da API que ainda roda MediaPipe/OpenCV no servidor — por
isso só é importado quando a rota está habilitada.
"""
from __future__ import annotations

import base64

import cv2
import mediapipe as mp
import numpy as np
from flask import Blueprint, jsonify, request

from libras.config import PATHS

record_bp = Blueprint("record", __name__)

_CAPTURE_DIR = PATHS["data_letters_captured"]

_hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
)


def _decode_image(image_b64: str) -> np.ndarray | None:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _hand_bounding_box(landmarks, w: int, h: int, margin: int = 20) -> tuple[int, int, int, int]:
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)
    return (
        max(0, x_min - margin), max(0, y_min - margin),
        min(w, x_max + margin), min(h, y_max + margin),
    )


@record_bp.route("/record_data", methods=["POST"])
def record_data():
    data = request.get_json()
    if not data or "image" not in data or "label" not in data:
        return jsonify({"success": False, "message": "Dados faltando."}), 400

    try:
        img_bgr = _decode_image(data["image"])
        label_text = data["label"].upper()
        if img_bgr is None:
            return jsonify({"success": False, "message": "Erro ao processar imagem."}), 400

        results = _hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return jsonify({"success": False, "message": "Nenhuma mao detectada."})

        h, w, _ = img_bgr.shape
        x_min, y_min, x_max, y_max = _hand_bounding_box(
            results.multi_hand_landmarks[0].landmark, w, h,
        )
        hand_img = img_bgr[y_min:y_max, x_min:x_max]

        folder_path = _CAPTURE_DIR / label_text
        folder_path.mkdir(parents=True, exist_ok=True)
        file_name = f"{len(list(folder_path.glob('*.png'))) + 1}.png"
        cv2.imwrite(str(folder_path / file_name), hand_img)

        return jsonify({"success": True, "message": f"Imagem salva como {file_name}."})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
