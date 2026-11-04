"""Rotas de reconhecimento de letras estáticas (Keras) — /predict_letter_landmarks.

Stateless: recebe os 21 landmarks da mão já extraídos pelo MediaPipe no
app/navegador (formato em docs/API_CONTRACT.md). Modelo separado do LSTM de
sinais — classifica um único frame (sem dimensão temporal), usando só x/y
dos landmarks, sem normalização.

A antiga /record_data (que grava imagens) fica em record_routes.py, só
habilitada com LIBRAS_ENABLE_RECORD_DATA=1.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import h5py
import joblib
import numpy as np
from flask import Blueprint, jsonify, request
from tensorflow.keras.layers import InputLayer as _InputLayer
from tensorflow.keras.models import load_model as keras_load_model

from libras.api.auth import login_required
from libras.api.payloads import PayloadError, parse_letter_payload
from libras.config import PATHS

letter_bp = Blueprint("letter", __name__)

_MODEL_PATH = PATHS["models_letters"] / "keras_landmarks_model.h5"
_ENCODER_PATH = PATHS["models_letters"] / "label_encoder.pkl"
_CONFIDENCE_THRESHOLD = 0.8

# O modelo de letras aprendeu o formato da mão ESQUERDA na imagem: no teste no
# celular (22/10/2026), a mão esquerda acertava as letras assimétricas (C, L, P,
# Q) sem inverter nada, e a direita só acertava com x invertido. Para servir as
# duas mãos, a direita é espelhada (x → 1 − x) antes de classificar — o espelho
# de uma mão direita tem o formato de uma esquerda. Os clientes seguem enviando
# a imagem NÃO espelhada (contrato da API) e dizem qual mão é em `handedness`;
# sem o campo, assume-se a direita (maioria destra). O modelo de palavras não
# precisa disso: recebe cada mão na sua posição (left_hand / right_hand).
_MODEL_TRAINED_HAND = "left"
_DEFAULT_HANDEDNESS = "right"


class _CompatInputLayer(_InputLayer):
    """Modelo foi salvo com `batch_shape` (formato antigo do Keras);
    a versão local do Keras espera `input_shape`."""

    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs.setdefault("input_shape", batch_shape[1:])
        super().__init__(**kwargs)


def _patch_h5_dtype_policy(src_path) -> str:
    """Copia o .h5 para um temp e remove o campo DTypePolicy do config —
    incompatível entre a versão do Keras usada no treino (Colab) e a local."""

    def _patch(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "DTypePolicy" and "config" in obj:
                return obj["config"].get("name", "float32")
            return {k: _patch(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_patch(i) for i in obj]
        return obj

    tmp = tempfile.mktemp(suffix=".h5")
    shutil.copy2(str(src_path), tmp)
    with h5py.File(tmp, "r+") as f:
        raw = f.attrs.get("model_config")
        if raw is not None:
            cfg_str = raw if isinstance(raw, str) else raw.decode("utf-8")
            f.attrs["model_config"] = json.dumps(_patch(json.loads(cfg_str))).encode("utf-8")
    return tmp


def _load_letter_model():
    if not _MODEL_PATH.exists() or not _ENCODER_PATH.exists():
        if not _MODEL_PATH.exists():
            print(f"Modelo de letras nao encontrado: {_MODEL_PATH}")
        if not _ENCODER_PATH.exists():
            print(f"Encoder nao encontrado: {_ENCODER_PATH}")
        return None, None

    tmp_path = _patch_h5_dtype_policy(_MODEL_PATH)
    try:
        model = keras_load_model(
            tmp_path, custom_objects={"InputLayer": _CompatInputLayer}, compile=False,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    encoder = joblib.load(str(_ENCODER_PATH))
    return model, encoder


# Carregado uma vez por processo — só leitura, sem estado entre requisições.
print("\n--- IA de Letras (Keras) ---")
try:
    _letter_model, _label_encoder = _load_letter_model()
    _letter_classes = _label_encoder.classes_ if _label_encoder is not None else None
    if _letter_model is not None:
        print(f"Modelo de letras carregado: {len(_letter_classes)} classes")
        print(f"   Classes: {list(_letter_classes)}")
    else:
        print("   Rota /predict_letter_landmarks ficara desativada.")
except Exception as e:
    _letter_model = None
    _label_encoder = None
    _letter_classes = None
    print(f"Erro ao carregar modelo de letras: {e}")
    print("   Rota /predict_letter_landmarks ficara desativada.")


def _hand_box(hand: np.ndarray) -> list[float]:
    """Caixa [x_min, y_min, x_max, y_max] da mão, em coordenadas normalizadas (0–1)."""
    x_min, y_min = hand.min(axis=0)
    x_max, y_max = hand.max(axis=0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


@letter_bp.route("/predict_letter_landmarks", methods=["POST"])
@login_required
def predict_letter_landmarks():
    if _letter_model is None:
        return jsonify({"success": False, "message": "Modelo de letras nao carregado."}), 503

    try:
        hand, target_letter, handedness = parse_letter_payload(request.get_json(silent=True))
    except PayloadError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    if hand is None:
        return jsonify({
            "success": True,
            "correct": False,
            "predicted_letter": "Nenhuma mao detectada",
            "target_letter": target_letter,
            "message": "Nenhuma mao detectada.",
        })

    handedness = handedness or _DEFAULT_HANDEDNESS
    model_hand = hand.copy()
    if handedness != _MODEL_TRAINED_HAND:
        model_hand[:, 0] = 1.0 - model_hand[:, 0]
    x = model_hand.flatten().reshape(1, -1).astype(np.float32)
    probas = np.asarray(_letter_model(x, training=False))[0]
    idx = int(np.argmax(probas))
    confidence = float(probas[idx])
    predicted_letter = str(_label_encoder.inverse_transform([idx])[0])

    is_correct = False
    if confidence < _CONFIDENCE_THRESHOLD:
        predicted_letter = "Movimento Invalido"
        message = "Movimento Invalido."
    elif predicted_letter.upper() != target_letter.upper():
        message = f"Gesto incorreto. Voce fez a letra: {predicted_letter}"
    else:
        is_correct = True
        message = "Gesto Correto!"

    return jsonify({
        "success": True,
        "correct": is_correct,
        "predicted_letter": predicted_letter,
        "target_letter": target_letter,
        "box": _hand_box(hand),
        "handedness": handedness,
        "confidence": confidence,
        "message": message,
    })


def is_loaded() -> bool:
    return _letter_model is not None


def num_classes() -> int:
    return len(_letter_classes) if _letter_classes is not None else 0
