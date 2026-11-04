"""Rotas de reconhecimento de sinais (LSTM) — /predict_sequence, /classes.

Stateless: cada requisição traz a sequência inteira de landmarks, extraídos
pelo MediaPipe no app/navegador (formato em docs/API_CONTRACT.md). O servidor
monta o vetor de keypoints com `libras.features.keypoints` — mesma função do
treino —, reamostra para o num_frames do modelo e prediz uma vez. Não há
buffer compartilhado entre usuários, então a API roda com N workers/réplicas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from flask import Blueprint, jsonify, request
from tensorflow.keras.models import load_model

from libras.api.auth import login_required
from libras.api.payloads import NoLandmarksError, PayloadError, parse_sequence_payload
from libras.config import PATHS
from libras.features.keypoints import FeatureBuilder, get_feature_builder, normalize_sequence
from libras.utils.io import load_json

sign_bp = Blueprint("sign", __name__)

_TOP_K = 3
_CONFIDENT_THRESHOLD = 0.25


@dataclass(frozen=True)
class _SignModel:
    """Modelo LSTM carregado uma vez por processo — só leitura."""
    model: object
    classes: list[str]
    num_frames: int
    keypoint_dim: int
    build_features: FeatureBuilder


def _load_sign_model() -> _SignModel | None:
    model_path = PATHS["models"] / "best_model.keras"
    meta_path = PATHS["models"] / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        print(f"Modelo LSTM nao encontrado em {PATHS['models']}")
        return None

    meta = load_json(meta_path)
    model = load_model(model_path)
    # O shape de entrada do próprio modelo decide o formato do vetor
    # (438 atual ou 1662 legado) — não depende do metadata estar em dia.
    _, num_frames, keypoint_dim = model.input_shape
    return _SignModel(
        model=model,
        classes=list(meta["classes"]),
        num_frames=int(num_frames),
        keypoint_dim=int(keypoint_dim),
        build_features=get_feature_builder(int(keypoint_dim)),
    )


print("\n--- IA de Palavras (LSTM) ---")
try:
    _sign = _load_sign_model()
    if _sign is not None:
        print(f"Modelo LSTM carregado: {len(_sign.classes)} classes, "
              f"entrada ({_sign.num_frames}, {_sign.keypoint_dim})")
        print(f"   Classes: {_sign.classes}")
    else:
        print("   Rota /predict_sequence ficara desativada.")
except Exception as e:
    _sign = None
    print(f"Erro ao carregar modelo LSTM: {e}")
    print("   Rota /predict_sequence ficara desativada.")


@sign_bp.route("/predict_sequence", methods=["POST"])
@login_required
def predict_sequence():
    if _sign is None:
        return jsonify({"error": "Modelo LSTM nao carregado."}), 503

    try:
        frames = parse_sequence_payload(request.get_json(silent=True))
    except NoLandmarksError as e:
        return jsonify({"error": str(e)}), 422
    except PayloadError as e:
        return jsonify({"error": str(e)}), 400

    X = np.stack([_sign.build_features(*frame) for frame in frames])
    X = normalize_sequence(X, _sign.num_frames).astype(np.float32)

    # model(...) em vez de model.predict(...): sem overhead de batching
    # para uma única amostra e seguro para chamadas concorrentes.
    probs = np.asarray(_sign.model(X[np.newaxis], training=False))[0]
    top_indices = np.argsort(probs)[::-1][:_TOP_K]

    predictions = [
        {"word": _sign.classes[idx], "confidence": float(probs[idx]), "rank": rank + 1}
        for rank, idx in enumerate(top_indices)
    ]
    top1_confidence = float(probs[top_indices[0]])

    return jsonify({
        "predictions": predictions,
        "top1_word": _sign.classes[top_indices[0]],
        "top1_confidence": top1_confidence,
        "is_confident": top1_confidence >= _CONFIDENT_THRESHOLD,
        "frames_received": len(frames),
    })


@sign_bp.route("/classes", methods=["GET"])
def get_classes():
    classes = _sign.classes if _sign is not None else []
    return jsonify({"classes": classes, "num_classes": len(classes)})


def is_loaded() -> bool:
    return _sign is not None


def num_classes() -> int:
    return len(_sign.classes) if _sign is not None else 0


def keypoint_dim() -> int | None:
    return _sign.keypoint_dim if _sign is not None else None
