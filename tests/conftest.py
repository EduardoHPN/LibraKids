"""Fixtures compartilhadas dos testes da API."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Impede que o .env real do desenvolvedor (com o Supabase) vaze para os testes."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LIBRAS_ENABLE_RECORD_DATA", raising=False)
    monkeypatch.delenv("LIBRAS_TOKEN_TTL_HOURS", raising=False)
    try:
        from libras.api import app as app_module
    except ImportError:  # dependências da API não instaladas
        return
    monkeypatch.setattr(app_module, "_ENV_FILE", tmp_path / "nao_existe.env")


@pytest.fixture
def app():
    """App com SQLite em memória e schema criado pelas migrações (igual à produção)."""
    pytest.importorskip("tensorflow")
    pytest.importorskip("flask_migrate")
    from flask_migrate import upgrade

    from libras.api import create_app

    # BCRYPT_LOG_ROUNDS baixo: o hash com o custo padrão deixaria cada login lento.
    app = create_app({
        "TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://", "BCRYPT_LOG_ROUNDS": 4,
    })
    with app.app_context():
        upgrade()
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_client(app):
    """Cliente com uma sessão aberta: envia `Authorization` em toda requisição."""
    client = app.test_client()
    credentials = {"username": "fixture", "password": "segredo123"}
    client.post("/register", json=credentials)
    token = client.post("/login", json=credentials).get_json()["access_token"]
    client.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token}"
    return client
