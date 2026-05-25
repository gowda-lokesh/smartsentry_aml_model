"""
db_utils.py  —  shared database helpers for the SmartSentry AML pipeline.

Every notebook does:
    from db_utils import ...

Credentials live in db_config.py — never edit this file to change them.
"""

import io
import json
import joblib
import pandas as pd

from sqlalchemy import create_engine, text

import db_config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Tables that accumulate history (never truncated)
# ─────────────────────────────────────────────────────────────────────────────

APPEND_TABLES = {
    "predictions_output",
    "pipeline_execution_log",
    "run_dashboard",
}


# ─────────────────────────────────────────────────────────────────────────────
# Engine cache
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE = None


# ─────────────────────────────────────────────────────────────────────────────
# Database Engine
# ─────────────────────────────────────────────────────────────────────────────

def get_engine():
    """
    Return pooled SQLAlchemy engine.
    """
    global _ENGINE

    if _ENGINE is None:

        url = (
            f"postgresql+psycopg2://{cfg.DB_USER}:{cfg.DB_PASSWORD}"
            f"@{cfg.DB_HOST}:{cfg.DB_PORT}/{cfg.DB_NAME}"
        )

        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "sslmode": cfg.DB_SSLMODE,
                "options": "-c datestyle=ISO,DMY",
            },
        )

    return _ENGINE


# ─────────────────────────────────────────────────────────────────────────────
# Connection Test
# ─────────────────────────────────────────────────────────────────────────────

def test_connection():
    """
    Print one-line DB connection status.
    """

    eng = get_engine()

    with eng.connect() as conn:

        db = conn.execute(
            text("SELECT current_database()")
        ).scalar()

        ver = conn.execute(
            text("SELECT version()")
        ).scalar()

    print(f"Connected to '{db}' — {ver.split(',')[0]}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Read Table (Chunked Streaming)
# ─────────────────────────────────────────────────────────────────────────────

def read_table(
    name,
    where=None,
    columns=None,
    chunksize=50000
):
    """
    Read a full table safely using chunked streaming.

    Parameters
    ----------
    name : str
        Table name

    where : str
        Optional WHERE clause

    columns : list[str]
        Optional column subset

    chunksize : int
        Rows per chunk
    """

    eng = get_engine()

    cols = ", ".join(columns) if columns else "*"

    sql = f"""
        SELECT {cols}
        FROM {cfg.DB_SCHEMA}.{name}
    """

    if where:
        sql += f" WHERE {where}"

    print(f"Reading {cfg.DB_SCHEMA}.{name}...")

    total_rows = 0
    chunks = []

    with eng.connect().execution_options(
        stream_results=True
    ) as conn:

        for i, chunk in enumerate(

            pd.read_sql(
                text(sql),
                conn,
                chunksize=chunksize
            )

        ):

            chunks.append(chunk)

            total_rows += len(chunk)

            print(
                f"  chunk {i+1:,}: "
                f"{total_rows:,} rows loaded"
            )

    if len(chunks) == 0:
        return pd.DataFrame()

    df = pd.concat(
        chunks,
        ignore_index=True
    )

    print(
        f"Completed: "
        f"{len(df):,} rows x {len(df.columns)} cols"
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Date Coercion
# ─────────────────────────────────────────────────────────────────────────────

_DATE_COLUMNS = {
    "date_of_birth",
    "account_opening_date",
    "inoperative_status_date",
    "cif_creation_date",
    "kyc_update_date",
    "date_of_incorporation",
    "wallet_creation_date",
    "device_first_seen",
    "device_last_seen",
    "timestamp",
    "datestamp",
    "txn_timestamp",
}


def _coerce_dates(df):
    """
    Parse known date columns safely.
    """

    out = df.copy()

    for col in out.columns:

        if col.lower() in _DATE_COLUMNS:

            out[col] = pd.to_datetime(
                out[col],
                dayfirst=True,
                errors="coerce"
            )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Standard Write
# ─────────────────────────────────────────────────────────────────────────────

def write_table(
    df,
    name,
    mode=None,
    chunksize=5000
):
    """
    Write DataFrame to PostgreSQL.
    """

    if mode is None:
        mode = (
            "append"
            if name in APPEND_TABLES
            else "replace"
        )

    eng = get_engine()

    full = f"{cfg.DB_SCHEMA}.{name}"

    with eng.begin() as conn:

        if mode == "replace":

            conn.execute(
                text(f"TRUNCATE TABLE {full}")
            )

        db_cols = pd.read_sql(

            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=:s
                  AND table_name=:t
            """),

            conn,

            params={
                "s": cfg.DB_SCHEMA,
                "t": name
            }

        )["column_name"].tolist()

        keep = [
            c for c in df.columns
            if c in db_cols
        ]

        out = _coerce_dates(
            df[keep].copy()
        )

        out.to_sql(
            name,
            conn,
            schema=cfg.DB_SCHEMA,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunksize
        )

    print(
        f"  write {full}: "
        f"{len(df):,} rows "
        f"(mode={mode}, "
        f"{len(keep)}/{len(df.columns)} cols matched)"
    )

    return len(df)


# ─────────────────────────────────────────────────────────────────────────────
# Fast COPY Bulk Loader
# ─────────────────────────────────────────────────────────────────────────────

def write_table_fast(
    df,
    table_name,
    mode="append"
):
    """
    Fast PostgreSQL bulk loader using COPY.
    """

    import io as _io
    import pandas as _pd

    from psycopg2 import sql as _sql

    eng = get_engine()

    conn = eng.raw_connection()

    cur = conn.cursor()

    try:

        df = df.copy()

        df.columns = [
            c.strip()
            for c in df.columns
        ]

        for col in df.columns:

            if df[col].dtype == object:

                df[col] = df[col].replace(
                    "",
                    None
                )

        df = df.where(
            _pd.notnull(df),
            None
        )

        # Existing table columns
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
              AND table_schema = %s
            """,
            (table_name, cfg.DB_SCHEMA)
        )

        existing = {
            r[0]
            for r in cur.fetchall()
        }

        if not existing:

            raise ValueError(
                f"Table '{cfg.DB_SCHEMA}.{table_name}' not found."
            )

        matched = [
            c for c in df.columns
            if c in existing
        ]

        skipped = [
            c for c in df.columns
            if c not in existing
        ]

        if skipped:

            print(
                f"  NOTE: skipped columns not in table: "
                f"{skipped}"
            )

        if not matched:

            raise ValueError(
                f"No matching columns for table '{table_name}'."
            )

        df = df[matched]

        if mode == "replace":

            cur.execute(

                _sql.SQL(
                    "TRUNCATE TABLE {}.{}"
                ).format(
                    _sql.Identifier(cfg.DB_SCHEMA),
                    _sql.Identifier(table_name)
                )

            )

        # CSV buffer
        buf = _io.StringIO()

        df.to_csv(
            buf,
            index=False,
            header=False,
            na_rep=""
        )

        buf.seek(0)

        # COPY SQL
        copy_sql = _sql.SQL(
            "COPY {}.{} ({}) "
            "FROM STDIN WITH CSV"
        ).format(

            _sql.Identifier(cfg.DB_SCHEMA),

            _sql.Identifier(table_name),

            _sql.SQL(",").join(
                map(_sql.Identifier, df.columns)
            )

        )

        # Bulk insert
        cur.copy_expert(
            copy_sql.as_string(cur),
            buf
        )

        conn.commit()

        print(
            f"  write {cfg.DB_SCHEMA}.{table_name}: "
            f"{len(df):,} rows "
            f"(mode={mode})"
        )

    except Exception as e:

        conn.rollback()

        print(
            f"  ERROR writing {table_name}: {e}"
        )

        raise

    finally:

        cur.close()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_model_registry():

    eng = get_engine()

    with eng.begin() as conn:

        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS
            {cfg.DB_SCHEMA}.model_registry (

                model_name   VARCHAR(64) NOT NULL,
                run_id       VARCHAR(32) NOT NULL,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                model_bytes  BYTEA NOT NULL,
                metrics_json JSONB
            )
        """))

        has_pk = conn.execute(text("""
            SELECT count(*)
            FROM information_schema.table_constraints
            WHERE table_schema=:s
              AND table_name='model_registry'
              AND constraint_type='PRIMARY KEY'
        """), {"s": cfg.DB_SCHEMA}).scalar()

        if not has_pk:

            conn.execute(text(f"""
                ALTER TABLE
                {cfg.DB_SCHEMA}.model_registry

                ADD PRIMARY KEY (
                    model_name,
                    run_id
                )
            """))


# ─────────────────────────────────────────────────────────────────────────────
# Save Model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    model_name,
    run_id,
    bundle,
    metrics=None
):
    """
    Save model bundle into PostgreSQL BYTEA.
    """

    _ensure_model_registry()

    buf = io.BytesIO()

    joblib.dump(
        bundle,
        buf,
        compress=("xz", 3)
    )

    blob = buf.getvalue()

    eng = get_engine()

    with eng.begin() as conn:

        conn.execute(text(f"""
            INSERT INTO {cfg.DB_SCHEMA}.model_registry
            (
                model_name,
                run_id,
                model_bytes,
                metrics_json
            )
            VALUES
            (
                :n,
                :r,
                :b,
                :m
            )

            ON CONFLICT (model_name, run_id)

            DO UPDATE SET

                model_bytes = EXCLUDED.model_bytes,
                metrics_json = EXCLUDED.metrics_json,
                created_at = now()
        """),

        {
            "n": model_name,
            "r": run_id,
            "b": blob,
            "m": json.dumps(metrics or {})
        })

    print(
        f"  model saved: "
        f"{model_name} "
        f"(run {run_id}, "
        f"{len(blob)/1024:.1f} KB)"
    )

    return len(blob)


# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    model_name,
    run_id=None
):
    """
    Load model bundle from PostgreSQL.
    """

    eng = get_engine()

    with eng.connect() as conn:

        if run_id:

            row = conn.execute(text(f"""
                SELECT model_bytes
                FROM {cfg.DB_SCHEMA}.model_registry
                WHERE model_name=:n
                  AND run_id=:r
            """),

            {
                "n": model_name,
                "r": run_id
            }).fetchone()

        else:

            row = conn.execute(text(f"""
                SELECT model_bytes
                FROM {cfg.DB_SCHEMA}.model_registry
                WHERE model_name=:n
                ORDER BY created_at DESC
                LIMIT 1
            """),

            {
                "n": model_name
            }).fetchone()

    if row is None:

        raise FileNotFoundError(
            f"No model found for '{model_name}'"
        )

    bundle = joblib.load(
        io.BytesIO(row[0])
    )

    print(
        f"  model loaded: "
        f"{model_name}"
    )

    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_connection()