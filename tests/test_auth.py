"""Testes da API de autenticação e da configuração do banco (create_app)."""

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

# create_app registra as rotas de predição, que importam TensorFlow.
pytest.importorskip("tensorflow")
pytest.importorskip("flask_migrate")

from libras.api import app as app_module  # noqa: E402
from libras.api import create_app, letter_routes, sign_routes  # noqa: E402
from libras.api.app import normalize_database_url  # noqa: E402
from libras.api.extensions import db  # noqa: E402
from libras.api.models import User, UserSession  # noqa: E402

# Fixtures `app` e `client` (SQLite em memória + migrações) em conftest.py.

FIXTURE = Path(__file__).parent / "fixtures" / "predict_sequence.json"
UNAUTHORIZED = {"message": "Nao autenticado."}


def _register(client, username="maria", password="segredo123"):
    return client.post("/register", json={"username": username, "password": password})


def _login(client, username="maria", password="segredo123"):
    return client.post("/login", json={"username": username, "password": password})


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _new_token(client):
    _register(client)
    return _login(client).get_json()["access_token"]


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _expire(app, token):
    """Vence a sessão do token sem esperar o tempo passar."""
    with app.app_context():
        session = UserSession.query.filter_by(token_hash=_token_hash(token)).one()
        session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.session.commit()


def _blank_image_b64():
    """PNG sem mão: /record_data responde 200 sem gravar nada em disco."""
    cv2 = pytest.importorskip("cv2")
    _, png = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))
    return base64.b64encode(png.tobytes()).decode("ascii")


def _sequence_payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _letter_payload():
    return {"hand": [[0.5, 0.5]] * 21, "target_letter": "A", "handedness": "left"}


# Rota, método e corpo de uma chamada que responde 200 com token válido.
_PROTECTED_ROUTES = {
    "/predict_sequence": ("POST", _sequence_payload),
    "/predict_letter_landmarks": ("POST", _letter_payload),
    "/me": ("GET", lambda: None),
    "/logout": ("POST", lambda: None),
}


@pytest.fixture(params=[*_PROTECTED_ROUTES, "/record_data"])
def protected(request):
    """Prepara uma rota protegida e devolve (app, client, função que a chama)."""
    path = request.param
    test_config = {"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://", "BCRYPT_LOG_ROUNDS": 4}
    if path == "/record_data":
        pytest.importorskip("mediapipe")
        method, build_body = "POST", lambda: {"image": _blank_image_b64(), "label": "A"}
        test_config["ENABLE_RECORD_DATA"] = True
    else:
        method, build_body = _PROTECTED_ROUTES[path]
    if path == "/predict_sequence" and not sign_routes.is_loaded():
        pytest.skip("artifacts/models/best_model.keras ausente")
    if path == "/predict_letter_landmarks" and not letter_routes.is_loaded():
        pytest.skip("modelo de letras ausente")

    from flask_migrate import upgrade

    app = create_app(test_config)
    with app.app_context():
        upgrade()
    client = app.test_client()

    def call(headers=None):
        return client.open(path, method=method, json=build_body(), headers=headers)

    return app, client, call


class TestRegister:
    def test_success(self, client):
        resp = _register(client)
        assert resp.status_code == 201

    def test_duplicate_username(self, client):
        _register(client)
        assert _register(client).status_code == 409

    def test_username_too_short(self, client):
        assert _register(client, username="a").status_code == 400

    def test_missing_password(self, client):
        assert _register(client, password="").status_code == 400


class TestLogin:
    def test_returns_token(self, client):
        _register(client)
        resp = _login(client)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["username"] == "maria"
        assert len(body["access_token"]) == 64

    def test_wrong_password(self, client):
        _register(client)
        assert _login(client, password="errada").status_code == 401

    def test_unknown_user(self, client):
        assert _login(client, username="ninguem").status_code == 401


class TestMe:
    def test_with_valid_token(self, client):
        resp = client.get("/me", headers=_bearer(_new_token(client)))
        assert resp.status_code == 200
        assert resp.get_json()["username"] == "maria"


class TestProtectedRoutes:
    """Cada rota protegida nos cinco estados do token, sempre com o mesmo 401."""

    @staticmethod
    def _assert_unauthorized(resp):
        assert resp.status_code == 401
        assert resp.get_json() == UNAUTHORIZED

    def test_without_token(self, protected):
        _, _, call = protected
        self._assert_unauthorized(call())

    @pytest.mark.parametrize("header", [
        "Bearer invalido",
        "Bearer ",
        "Bearer",
        "Token abc",
        "abc",
        "",
    ])
    def test_with_invalid_token(self, protected, header):
        _, _, call = protected
        self._assert_unauthorized(call({"Authorization": header}))

    def test_with_expired_token(self, protected):
        app, client, call = protected
        token = _new_token(client)
        _expire(app, token)
        self._assert_unauthorized(call(_bearer(token)))

    def test_after_logout(self, protected):
        _, client, call = protected
        token = _new_token(client)
        assert client.post("/logout", headers=_bearer(token)).status_code == 200
        self._assert_unauthorized(call(_bearer(token)))

    def test_with_valid_token(self, protected):
        _, client, call = protected
        assert call(_bearer(_new_token(client))).status_code == 200


class TestSessions:
    def test_login_keeps_other_sessions_active(self, client):
        site = _new_token(client)
        app_token = _login(client).get_json()["access_token"]
        assert site != app_token
        assert client.get("/me", headers=_bearer(site)).status_code == 200
        assert client.get("/me", headers=_bearer(app_token)).status_code == 200

    def test_logout_ends_only_current_session(self, client):
        site = _new_token(client)
        app_token = _login(client).get_json()["access_token"]
        assert client.post("/logout", headers=_bearer(site)).status_code == 200
        assert client.get("/me", headers=_bearer(site)).status_code == 401
        assert client.get("/me", headers=_bearer(app_token)).status_code == 200

    def test_stores_only_token_hash(self, app, client):
        token = _new_token(client)
        with app.app_context():
            session = UserSession.query.one()
            assert session.token_hash == _token_hash(token)
            assert session.token_hash != token
            assert not hasattr(User, "token")

    def test_stored_hash_does_not_work_as_token(self, client):
        assert client.get("/me", headers=_bearer(_token_hash(_new_token(client)))).status_code == 401

    def test_ttl_comes_from_config(self, app, client):
        app.config["TOKEN_TTL_HOURS"] = 2
        before = datetime.now(timezone.utc)
        _new_token(client)
        with app.app_context():
            expires_at = UserSession.query.one().expires_at.replace(tzinfo=timezone.utc)
        assert timedelta(hours=2) - timedelta(minutes=1) < expires_at - before <= timedelta(hours=2, minutes=1)

    def test_login_removes_only_own_expired_sessions(self, app, client):
        expired = _new_token(client)
        active = _login(client).get_json()["access_token"]
        _register(client, username="joana")
        other_expired = _login(client, username="joana").get_json()["access_token"]
        _expire(app, expired)
        _expire(app, other_expired)

        _login(client)

        with app.app_context():
            hashes = {s.token_hash for s in UserSession.query.all()}
        assert _token_hash(expired) not in hashes
        assert _token_hash(active) in hashes
        assert _token_hash(other_expired) in hashes

    @pytest.mark.parametrize("path", ["/health", "/classes"])
    def test_public_get_routes_need_no_token(self, client, path):
        assert client.get(path).status_code == 200

    def test_wrong_password_is_not_session_401(self, client):
        _register(client)
        resp = _login(client, password="errada")
        assert resp.status_code == 401
        assert resp.get_json() != UNAUTHORIZED

    def test_browser_can_call_protected_route(self, client):
        """Preflight sem token passa, e o 401 traz CORS para o navegador conseguir lê-lo."""
        origin = {"Origin": "http://localhost:4200"}
        preflight = client.options("/predict_sequence", headers={
            **origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        assert preflight.status_code == 200
        assert "authorization" in preflight.headers["Access-Control-Allow-Headers"].lower()

        resp = client.post("/predict_sequence", json={}, headers=origin)
        assert resp.status_code == 401
        assert resp.headers["Access-Control-Allow-Origin"] == origin["Origin"]


class TestTokenTtlEnv:
    def test_default_is_168_hours(self):
        assert create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://"}).config["TOKEN_TTL_HOURS"] == 168

    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("LIBRAS_TOKEN_TTL_HOURS", "24")
        assert create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://"}).config["TOKEN_TTL_HOURS"] == 24

    @pytest.mark.parametrize("value", ["abc", "0", "-1", "1.5"])
    def test_invalid_value_stops_startup(self, monkeypatch, value):
        monkeypatch.setenv("LIBRAS_TOKEN_TTL_HOURS", value)
        with pytest.raises(ValueError, match="LIBRAS_TOKEN_TTL_HOURS"):
            create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://"})


class TestMigrations:
    PREVIOUS_REVISION = "4dc8d8585ded"

    def test_migrations_match_models(self, app):
        """Falha se um model mudou sem gerar migração (flask db migrate)."""
        from alembic.autogenerate import compare_metadata
        from alembic.migration import MigrationContext

        with app.app_context(), db.engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), db.metadata)
        assert diff == []

    def test_session_migration_keeps_accounts_and_drops_tokens(self, app):
        from flask_migrate import downgrade, upgrade
        from sqlalchemy import inspect, text

        def columns(table):
            return {c["name"] for c in inspect(db.engine).get_columns(table)}

        with app.app_context():
            downgrade(revision=self.PREVIOUS_REVISION)
            assert "token" in columns("user")
            assert not inspect(db.engine).has_table("user_session")
            with db.engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO user (username, password, token) "
                    "VALUES ('antiga', 'hash-da-senha', 'token-antigo')"
                ))

            upgrade()
            assert "token" not in columns("user")
            assert inspect(db.engine).has_table("user_session")
            with db.engine.connect() as conn:
                row = conn.execute(text("SELECT username, password FROM user")).one()
            assert tuple(row) == ("antiga", "hash-da-senha")

            downgrade(revision=self.PREVIOUS_REVISION)
            with db.engine.connect() as conn:
                row = conn.execute(text("SELECT username, password, token FROM user")).one()
            assert tuple(row) == ("antiga", "hash-da-senha", None)


class TestDatabaseUrl:
    @pytest.mark.parametrize("url, expected", [
        ("postgres://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
        ("postgresql://u:p@host:5432/db?sslmode=require",
         "postgresql+psycopg://u:p@host:5432/db?sslmode=require"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("sqlite:///local.db", "sqlite:///local.db"),
    ])
    def test_normalize(self, url, expected):
        assert normalize_database_url(url) == expected

    def test_reads_database_url_from_env(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host:5432/db")
        app = create_app({"TESTING": True})
        assert app.config["SQLALCHEMY_DATABASE_URI"] == "postgresql+psycopg://u:p@host:5432/db"

    def test_sqlite_fallback_without_database_url(self, monkeypatch, tmp_path):
        monkeypatch.setitem(app_module.PATHS, "instance", tmp_path)
        app = create_app({"TESTING": True})
        assert app.config["SQLALCHEMY_DATABASE_URI"] == f"sqlite:///{tmp_path / 'site.db'}"

    def test_engine_options(self, app):
        assert app.config["SQLALCHEMY_ENGINE_OPTIONS"] == {
            "pool_pre_ping": True, "pool_recycle": 300,
        }
