"""Factory da API Flask — autenticação + reconhecimento de sinais e letras."""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS
from sqlalchemy.engine import make_url

from libras.api.extensions import bcrypt, db, migrate
from libras.config import PATHS, PROJECT_ROOT

_ENV_FILE = PROJECT_ROOT / ".env"
_DEFAULT_TOKEN_TTL_HOURS = 168  # 7 dias


def normalize_database_url(url: str) -> str:
    """
    Força o driver psycopg 3 em URLs PostgreSQL.

    Provedores (Supabase, Heroku...) entregam `postgres://` ou
    `postgresql://`, que o SQLAlchemy resolveria para o psycopg2 — driver
    que não instalamos. Outras URLs (ex.: `sqlite://`) passam intactas.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def _default_database_uri(app: Flask) -> str:
    """URL do banco a partir de `DATABASE_URL`, com fallback SQLite local."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return normalize_database_url(url)

    # Fallback só para desenvolvimento local — em produção o disco do
    # container é efêmero e os usuários seriam perdidos a cada deploy.
    PATHS["instance"].mkdir(parents=True, exist_ok=True)
    db_path = PATHS["instance"] / "site.db"
    app.logger.warning(
        "DATABASE_URL nao definida — usando SQLite local em %s "
        "(apenas para desenvolvimento).", db_path,
    )
    return f"sqlite:///{db_path}"


def _token_ttl_hours() -> int:
    """Validade do token em horas (`LIBRAS_TOKEN_TTL_HOURS`). Valor inválido impede a API de subir."""
    raw = os.environ.get("LIBRAS_TOKEN_TTL_HOURS", "").strip()
    if not raw:
        return _DEFAULT_TOKEN_TTL_HOURS
    try:
        hours = int(raw)
    except ValueError:
        hours = 0
    if hours <= 0:
        raise ValueError(f"LIBRAS_TOKEN_TTL_HOURS deve ser um inteiro positivo, recebido: {raw!r}")
    return hours


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """
    Cria a aplicação Flask.

    `test_config` sobrescreve qualquer configuração (usado nos testes,
    ex.: `{"SQLALCHEMY_DATABASE_URI": "sqlite://"}` para SQLite em memória).
    """
    # Em produção as variáveis vêm do provedor; o .env é só conveniência
    # local. override=False: variável já definida no ambiente prevalece.
    if _ENV_FILE.is_file():
        load_dotenv(_ENV_FILE, override=False)

    app = Flask(__name__)

    # SECRET_KEY não assina nada sensível hoje (auth usa token opaco com o
    # hash salvo no banco, não sessão de cookie assinada), mas fica
    # configurável via env var em vez de hardcoded no código-fonte.
    app.config["SECRET_KEY"] = os.environ.get("LIBRAS_SECRET_KEY", "dev-only-change-me")
    app.config["TOKEN_TTL_HOURS"] = _token_ttl_hours()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # pool_pre_ping descarta conexões que o pooler do Supabase já fechou;
    # pool_recycle renova antes do timeout de inatividade do servidor.
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 300}
    # Sequências de landmarks (até MAX_FRAMES frames) cabem com folga em 2 MB;
    # acima disso o Flask responde 413 antes de ler o corpo.
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    # /record_data grava em disco — efêmero no container. Desligada por padrão.
    app.config["ENABLE_RECORD_DATA"] = os.environ.get("LIBRAS_ENABLE_RECORD_DATA") == "1"

    if test_config is None or "SQLALCHEMY_DATABASE_URI" not in test_config:
        app.config["SQLALCHEMY_DATABASE_URI"] = _default_database_uri(app)
    if test_config:
        app.config.update(test_config)

    CORS(app, supports_credentials=True)

    db.init_app(app)
    bcrypt.init_app(app)
    # O schema é gerenciado por migrações (pasta migrations/), não por
    # db.create_all(): rode `flask --app run_backend db upgrade`.
    # render_as_batch permite ALTER TABLE também no fallback SQLite.
    migrate.init_app(app, db, directory=str(PROJECT_ROOT / "migrations"), render_as_batch=True)

    from libras.api import models  # noqa: F401 — registra os models no metadata

    db_url = make_url(app.config["SQLALCHEMY_DATABASE_URI"])
    print(f"Banco de dados: {db_url.render_as_string(hide_password=True)}")

    from libras.api.auth import auth_bp
    from libras.api.letter_routes import letter_bp
    from libras.api.sign_routes import sign_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(sign_bp)
    app.register_blueprint(letter_bp)

    if app.config["ENABLE_RECORD_DATA"]:
        # Import tardio: é o único módulo que precisa de MediaPipe/OpenCV.
        from libras.api.record_routes import record_bp

        app.register_blueprint(record_bp)

    @app.errorhandler(413)
    def payload_too_large(_error):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
        return jsonify({"error": f"Corpo da requisicao maior que {limit_mb:g} MB."}), 413

    @app.route("/health", methods=["GET"])
    def health():
        from libras.api import letter_routes, sign_routes

        return jsonify({
            "status": "ok",
            "lstm_loaded": sign_routes.is_loaded(),
            "lstm_classes": sign_routes.num_classes(),
            "lstm_keypoint_dim": sign_routes.keypoint_dim(),
            "letter_loaded": letter_routes.is_loaded(),
            "letter_classes": letter_routes.num_classes(),
            "record_data_enabled": app.config["ENABLE_RECORD_DATA"],
        })

    return app
