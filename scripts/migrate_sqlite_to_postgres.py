"""
Copia os usuários do SQLite local (instance/site.db) para o banco de DATABASE_URL.

Pré-requisito: o schema já existe no destino
(`flask --app run_backend db upgrade` com o mesmo DATABASE_URL).

  • Não duplica: usuários cujo `username` já existe no destino são pulados.
  • Os hashes bcrypt são copiados como estão — as senhas continuam valendo.
  • Tokens de login NÃO são copiados: os usuários fazem login de novo.
  • Os ids não são copiados: o destino gera os seus (evita dessincronizar a
    sequence do PostgreSQL).

Uso:
    python scripts/migrate_sqlite_to_postgres.py
    python scripts/migrate_sqlite_to_postgres.py --dry-run
    python scripts/migrate_sqlite_to_postgres.py --sqlite caminho/para/site.db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.engine import make_url

from libras.api.app import normalize_database_url
from libras.api.models import User
from libras.config import PATHS, PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Migra usuários do SQLite para o PostgreSQL")
    parser.add_argument("--sqlite", type=Path, default=PATHS["instance"] / "site.db",
                        help="Arquivo SQLite de origem (padrão: instance/site.db)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra o que seria copiado, sem gravar nada")
    args = parser.parse_args()

    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)

    target_url = os.environ.get("DATABASE_URL", "").strip()
    if not target_url:
        sys.exit("DATABASE_URL nao definida (coloque no .env ou no ambiente).")
    if not args.sqlite.is_file():
        sys.exit(f"SQLite de origem nao encontrado: {args.sqlite}")

    target_url = normalize_database_url(target_url)
    source = create_engine(f"sqlite:///{args.sqlite.resolve()}")
    target = create_engine(target_url, pool_pre_ping=True)
    if source.url == target.url:
        sys.exit("Origem e destino sao o mesmo banco.")

    table = User.__table__
    if not inspect(target).has_table(table.name):
        sys.exit(
            f"Tabela '{table.name}' nao existe no destino. "
            "Rode antes: flask --app run_backend db upgrade"
        )

    print(f"Origem:  {args.sqlite}")
    print(f"Destino: {make_url(target_url).render_as_string(hide_password=True)}")

    with source.connect() as conn:
        rows = conn.execute(
            select(table.c.username, table.c.password).order_by(table.c.id)
        ).all()

    # engine.begin(): tudo numa transação — ou copia todos, ou nenhum.
    with target.begin() as conn:
        existing = set(conn.execute(select(table.c.username)).scalars())
        new_rows = [
            {"username": r.username, "password": r.password}
            for r in rows if r.username not in existing
        ]
        if new_rows and not args.dry_run:
            conn.execute(table.insert(), new_rows)

    skipped = len(rows) - len(new_rows)
    verb = "seriam copiados" if args.dry_run else "copiados"
    print(f"\nUsuarios na origem: {len(rows)}")
    print(f"   {verb}: {len(new_rows)}")
    print(f"   ja existentes (pulados): {skipped}")


if __name__ == "__main__":
    main()
