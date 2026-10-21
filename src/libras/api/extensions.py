"""Extensões Flask compartilhadas entre os módulos da API (evita import circular)."""
from __future__ import annotations

from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData

# Nomes explícitos para constraints: o Alembic só consegue alterar/remover
# constraints nomeadas (e o SQLite, em modo batch, exige isso).
_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

db = SQLAlchemy(metadata=MetaData(naming_convention=_NAMING_CONVENTION))
bcrypt = Bcrypt()
migrate = Migrate()
