from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import psycopg
from dotenv import load_dotenv

load_dotenv()


def get_pg_connection(dsn: Optional[str] = None) -> psycopg.Connection:
    resolved_dsn = dsn or os.getenv("DATABASE_URL") or os.getenv("PGDSN")
    return psycopg.connect(resolved_dsn)


def create_schema(sql_path: str | Path = "db/schema.sql", dsn: Optional[str] = None) -> None:
    sql_file = Path(sql_path)
    if not sql_file.exists():
        raise FileNotFoundError(f"Schema file not found: {sql_file}")

    with sql_file.open("r", encoding="utf-8") as handle:
        schema_sql = handle.read()

    with get_pg_connection(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            conn.commit()
