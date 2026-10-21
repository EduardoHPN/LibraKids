"""Testes da API de autenticação e da configuração do banco (create_app)."""

import pytest

# create_app registra as rotas de predição, que importam TensorFlow/MediaPipe.
pytest.importorskip("tensorflow")
pytest.importorskip("mediapipe")
pytest.importorskip("flask_migrate")

from flask_migrate import upgrade  # noqa: E402

from libras.api import app as app_module  # noqa: E402
from libras.api import create_app  # noqa: E402
from libras.api.app import normalize_database_url  # noqa: E402
from libras.api.extensions import db  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Impede que o .env real do desenvolvedor (com o Supabase) vaze para os testes."""
    monkeypatch.setattr(app_module, "_ENV_FILE", tmp_path / "nao_existe.env")
    monkeypatch.delenv("DATABASE_URL", raising=False)


@pytest.fixture
def app():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://"})
    with app.app_context():
        upgrade()  # schema vem das migrações, igual à produção
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _register(client, username="maria", password="segredo123"):
    return client.post("/register", json={"username": username, "password": password})


def _login(client, username="maria", password="segredo123"):
    return client.post("/login", json={"username": username, "password": password})


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
        _register(client)
        token = _login(client).get_json()["access_token"]
        resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.get_json()["username"] == "maria"

    def test_without_token(self, client):
        assert client.get("/me").status_code == 401

    def test_with_invalid_token(self, client):
        resp = client.get("/me", headers={"Authorization": "Bearer invalido"})
        assert resp.status_code == 401


class TestMigrations:
    def test_migrations_match_models(self, app):
        """Falha se um model mudou sem gerar migração (flask db migrate)."""
        from alembic.autogenerate import compare_metadata
        from alembic.migration import MigrationContext

        with app.app_context(), db.engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), db.metadata)
        assert diff == []


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
