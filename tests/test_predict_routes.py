"""Testes das rotas stateless de predição (/predict_sequence, /predict_letter_landmarks)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("flask_migrate")

from libras.api import letter_routes, sign_routes  # noqa: E402
from libras.api.payloads import MAX_FRAMES, MIN_FRAMES  # noqa: E402
from libras.features.keypoints import (  # noqa: E402
    LEGACY_TOTAL_DIM, TOTAL_DIM, keypoints_from_arrays, keypoints_from_arrays_legacy,
)

FIXTURE = Path(__file__).parent / "fixtures" / "predict_sequence.json"


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def frame(payload):
    return payload["frames"][1]


requires_sign_model = pytest.mark.skipif(
    not sign_routes.is_loaded(), reason="artifacts/models/best_model.keras ausente",
)


# ── /predict_sequence ────────────────────────────────────────

@requires_sign_model
class TestPredictSequence:
    def test_fixture_payload(self, client, payload):
        resp = client.post("/predict_sequence", json=payload)
        assert resp.status_code == 200
        body = resp.get_json()

        classes = client.get("/classes").get_json()["classes"]
        preds = body["predictions"]
        assert [p["rank"] for p in preds] == [1, 2, 3]
        confidences = [p["confidence"] for p in preds]
        assert confidences == sorted(confidences, reverse=True)
        assert all(0.0 <= c <= 1.0 for c in confidences)
        assert body["top1_word"] == preds[0]["word"] and body["top1_word"] in classes
        assert body["top1_confidence"] == preds[0]["confidence"]
        assert isinstance(body["is_confident"], bool)
        assert body["frames_received"] == len(payload["frames"])

    def test_is_deterministic(self, client, payload):
        a = client.post("/predict_sequence", json=payload).get_json()
        b = client.post("/predict_sequence", json=payload).get_json()
        assert a == b

    def test_accepts_face_from_mediapipe_tasks(self, client, payload):
        for f in payload["frames"]:
            f["face"] = f["face"] + [[0.5, 0.3, 0.0]] * 10  # 478 pontos (com íris)
        assert client.post("/predict_sequence", json=payload).status_code == 200

    @pytest.mark.parametrize("n", [MIN_FRAMES, MAX_FRAMES])
    def test_frame_limits_accepted(self, client, frame, n):
        resp = client.post("/predict_sequence", json={"frames": [frame] * n})
        assert resp.status_code == 200

    def test_no_hands_is_422(self, client, frame):
        no_hands = {**frame, "left_hand": None, "right_hand": None}
        resp = client.post("/predict_sequence", json={"frames": [no_hands] * MIN_FRAMES})
        assert resp.status_code == 422
        assert "mao" in resp.get_json()["error"]


class TestPredictSequenceInvalid:
    @pytest.mark.parametrize("body, fragment", [
        (None, "frames"),
        ({"frames": "x"}, "lista"),
        ({"frames": []}, "entre"),
    ])
    def test_bad_structure(self, client, body, fragment):
        resp = client.post("/predict_sequence", json=body)
        if not sign_routes.is_loaded():
            assert resp.status_code == 503
            return
        assert resp.status_code == 400
        assert fragment in resp.get_json()["error"]

    @pytest.mark.parametrize("mutate, fragment", [
        (lambda f: f.update(pose=f["pose"][:32]), "frames[0].pose"),
        (lambda f: f.update(face=f["face"] + [[0.5, 0.3, 0.0]] * 2), "frames[0].face"),  # 470
        (lambda f: f.update(right_hand=[["a", "b", "c"]] * 21), "apenas numeros"),
        (lambda f: f.update(right_hand=[[0.1, 0.2]] * 20 + [[0.1, 0.2, 0.3]]), "apenas numeros"),
        (lambda f: f.update(left_hand={"x": 1}), "lista de pontos"),
    ])
    @requires_sign_model
    def test_bad_landmarks(self, client, frame, mutate, fragment):
        frames = [json.loads(json.dumps(frame)) for _ in range(MIN_FRAMES)]
        mutate(frames[0])
        resp = client.post("/predict_sequence", json={"frames": frames})
        assert resp.status_code == 400
        assert fragment in resp.get_json()["error"]

    @requires_sign_model
    @pytest.mark.parametrize("n", [MIN_FRAMES - 1, MAX_FRAMES + 1])
    def test_frame_limits_rejected(self, client, frame, n):
        resp = client.post("/predict_sequence", json={"frames": [frame] * n})
        assert resp.status_code == 400

    @requires_sign_model
    def test_nan_rejected(self, client, frame):
        # JSON com NaN literal (o parser do Python aceita): troca 1 coordenada.
        raw = json.dumps({"frames": [frame] * MIN_FRAMES})
        raw = raw.replace(str(frame["right_hand"][0][0]), "NaN", 1)
        resp = client.post("/predict_sequence", data=raw, content_type="application/json")
        assert resp.status_code == 400
        assert "nao finitos" in resp.get_json()["error"]

    def test_body_too_large_is_413(self, client):
        resp = client.post(
            "/predict_sequence", data=b"x" * (2 * 1024 * 1024 + 1),
            content_type="application/json",
        )
        assert resp.status_code == 413
        assert "MB" in resp.get_json()["error"]

    def test_model_missing_is_503(self, client, payload, monkeypatch):
        monkeypatch.setattr(sign_routes, "_sign", None)
        assert client.post("/predict_sequence", json=payload).status_code == 503


class TestSignModelLoading:
    """O formato do vetor (438 ou 1662) vem do shape de entrada do modelo."""

    @pytest.mark.parametrize("dim, builder", [
        (TOTAL_DIM, keypoints_from_arrays),
        (LEGACY_TOTAL_DIM, keypoints_from_arrays_legacy),
    ])
    def test_builder_follows_model_input(self, tmp_path, monkeypatch, dim, builder):
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Input((30, dim)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        model.save(tmp_path / "best_model.keras")
        (tmp_path / "metadata.json").write_text(json.dumps({"classes": ["A", "B"]}))
        monkeypatch.setitem(sign_routes.PATHS, "models", tmp_path)

        loaded = sign_routes._load_sign_model()
        assert loaded.keypoint_dim == dim
        assert loaded.num_frames == 30
        assert loaded.build_features is builder

    def test_missing_model_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sign_routes.PATHS, "models", tmp_path)
        assert sign_routes._load_sign_model() is None


# ── /predict_letter_landmarks ────────────────────────────────

class _FakeEncoder:
    classes_ = np.array(["A", "B"])

    def inverse_transform(self, idx):
        return self.classes_[idx]


@pytest.fixture
def fake_letter_model(monkeypatch):
    """Substitui o modelo de letras (o .h5 não é versionado) por um falso."""
    probs = {"value": np.array([0.9, 0.1])}
    monkeypatch.setattr(letter_routes, "_letter_model", lambda x, training=False: probs["value"][None])
    monkeypatch.setattr(letter_routes, "_label_encoder", _FakeEncoder())
    monkeypatch.setattr(letter_routes, "_letter_classes", _FakeEncoder.classes_)
    return probs


def _hand(n=21, coords=2):
    rng = np.random.default_rng(0)
    return np.round(rng.uniform(0.3, 0.7, (n, coords)), 4).tolist()


class TestPredictLetterLandmarks:
    def test_correct_letter(self, client, fake_letter_model):
        hand = _hand()
        body = client.post("/predict_letter_landmarks", json={"hand": hand, "target_letter": "a"}).get_json()
        assert body["success"] and body["correct"]
        assert body["predicted_letter"] == "A"
        assert body["confidence"] == pytest.approx(0.9)
        xs, ys = zip(*hand)
        assert body["box"] == pytest.approx([min(xs), min(ys), max(xs), max(ys)])

    def test_wrong_letter(self, client, fake_letter_model):
        body = client.post("/predict_letter_landmarks", json={"hand": _hand(), "target_letter": "B"}).get_json()
        assert body["success"] and not body["correct"]
        assert "Voce fez a letra: A" in body["message"]

    def test_low_confidence(self, client, fake_letter_model):
        fake_letter_model["value"] = np.array([0.6, 0.4])
        body = client.post("/predict_letter_landmarks", json={"hand": _hand(), "target_letter": "A"}).get_json()
        assert body["predicted_letter"] == "Movimento Invalido" and not body["correct"]

    def test_accepts_xyz(self, client, fake_letter_model):
        resp = client.post("/predict_letter_landmarks", json={"hand": _hand(coords=3), "target_letter": "A"})
        assert resp.get_json()["correct"]

    def test_no_hand(self, client, fake_letter_model):
        body = client.post("/predict_letter_landmarks", json={"hand": None, "target_letter": "A"}).get_json()
        assert body["success"] and not body["correct"]
        assert body["predicted_letter"] == "Nenhuma mao detectada"

    @pytest.mark.parametrize("body", [
        {"hand": _hand()},                                   # sem target_letter
        {"hand": _hand(n=20), "target_letter": "A"},         # 20 pontos
        {"hand": _hand(), "target_letter": ""},              # letra vazia
        {"hand": _hand(), "target_letter": 1},               # letra não-texto
    ])
    def test_invalid_payload(self, client, fake_letter_model, body):
        resp = client.post("/predict_letter_landmarks", json=body)
        assert resp.status_code == 400
        assert resp.get_json()["success"] is False

    def test_model_missing_is_503(self, client, monkeypatch):
        monkeypatch.setattr(letter_routes, "_letter_model", None)
        resp = client.post("/predict_letter_landmarks", json={"hand": _hand(), "target_letter": "A"})
        assert resp.status_code == 503


# ── Rotas removidas / desligadas e /health ───────────────────

@pytest.mark.parametrize("path", ["/predict", "/clear", "/predict_letter", "/record_data"])
def test_image_routes_not_registered(client, path):
    assert client.post(path, json={}).status_code == 404


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["record_data_enabled"] is False
    assert "buffer_size" not in body
    if body["lstm_loaded"]:
        assert body["lstm_keypoint_dim"] in (TOTAL_DIM, LEGACY_TOTAL_DIM)


def test_api_does_not_import_mediapipe_or_cv2():
    """Sem /record_data, a API sobe sem MediaPipe/OpenCV (imagem Docker enxuta)."""
    code = (
        "import sys\n"
        "from libras.api import create_app\n"
        "create_app({'SQLALCHEMY_DATABASE_URI': 'sqlite://'})\n"
        "print(sorted(m for m in ('mediapipe', 'cv2') if m in sys.modules))\n"
    )
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("DATABASE_URL", "LIBRAS_ENABLE_RECORD_DATA")}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"
