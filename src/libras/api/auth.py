"""Rotas de autenticação — registro, login, logout, sessão atual — e `login_required`.

Cada login cria uma sessão própria (`UserSession`), então a mesma conta pode
estar ativa no app e no site ao mesmo tempo. O cliente recebe o token opaco, e o
banco guarda só o hash SHA-256 dele.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Blueprint, current_app, g, jsonify, request

from libras.api.extensions import bcrypt, db
from libras.api.models import User, UserSession

auth_bp = Blueprint("auth", __name__)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _find_active_session() -> tuple[UserSession, User] | None:
    """Sessão válida e usuário do token em `Authorization: Bearer <token>`, ou None."""
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme != "Bearer" or not token:
        return None
    return db.session.execute(
        db.select(UserSession, User)
        .join(User, UserSession.user_id == User.id)
        .where(
            UserSession.token_hash == _hash_token(token),
            UserSession.expires_at > datetime.now(timezone.utc),
        )
    ).first()


def login_required(view):
    """Exige um token válido e preenche `g.current_user` e `g.current_session`.

    O 401 é sempre o mesmo, para não indicar se o token não existe, expirou ou
    já foi encerrado.
    """

    @wraps(view)
    def wrapped(*args, **kwargs):
        active = _find_active_session()
        if active is None:
            return jsonify({"message": "Nao autenticado."}), 401
        g.current_session, g.current_user = active
        return view(*args, **kwargs)

    return wrapped


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"message": "Dados obrigatorios."}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"message": "Usuario e senha obrigatorios."}), 400

    if len(username) < 2 or len(username) > 20:
        return jsonify({"message": "Usuario deve ter entre 2 e 20 caracteres."}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"message": "Este nome de usuario ja esta em uso."}), 409

    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    db.session.add(User(username=username, password=hashed))
    db.session.commit()

    return jsonify({"message": "Conta criada com sucesso!"}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"message": "Dados obrigatorios."}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"message": "Usuario e senha obrigatorios."}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Usuario ou senha incorretos."}), 401

    now = datetime.now(timezone.utc)
    # Limpa as sessões vencidas do usuário aqui mesmo, sem precisar de tarefa agendada.
    UserSession.query.filter(
        UserSession.user_id == user.id, UserSession.expires_at <= now,
    ).delete()

    token = secrets.token_hex(32)
    db.session.add(UserSession(
        user_id=user.id,
        token_hash=_hash_token(token),
        expires_at=now + timedelta(hours=current_app.config["TOKEN_TTL_HOURS"]),
    ))
    db.session.commit()

    return jsonify({
        "access_token": token,
        "username": user.username,
        "message": "Login bem-sucedido!",
    }), 200


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    """Encerra só a sessão deste token. As outras sessões da conta continuam ativas."""
    UserSession.query.filter_by(id=g.current_session.id).delete()
    db.session.commit()
    return jsonify({"message": "Logout realizado."}), 200


@auth_bp.route("/me", methods=["GET"])
@login_required
def me():
    return jsonify({"username": g.current_user.username, "id": g.current_user.id}), 200
