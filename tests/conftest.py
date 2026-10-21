"""Fixtures compartilhadas dos testes da API."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Impede que o .env real do desenvolvedor (com o Supabase) vaze para os testes."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LIBRAS_ENABLE_RECORD_DATA", raising=False)
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

    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://"})
    with app.app_context():
        upgrade()
    return app


@pytest.fixture
def client(app):
    return app.test_client()
