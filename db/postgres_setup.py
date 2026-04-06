"""
db/postgres_setup.py
─────────────────────────
Creates all PostgreSQL tables required by the memory agent.

Run once before the first launch::

    python -m db.postgres_setup

Tables created
──────────────
``memory_store``
    One row per stored memory.  The ``pinecone_id`` column holds the Milvus
    vector record ID (the column name is kept for backward compatibility with
    earlier iterations of the project that used Pinecone).

``conflict_log``
    Every write decision produced by :mod:`memory.conflict` — resolution type
    (``UPDATE`` / ``APPEND`` / ``IGNORE``) and a ``consistency_hit`` flag used
    for the accuracy metric in §IV-D of the paper.

``experiment_metrics``
    Per-turn latency and retrieval statistics logged by the
    :func:`graph.nodes.log_metrics` node; used to generate the data in
    §IV-B through §IV-D of the paper.

``langgraph_checkpoints`` (and related tables)
    Created automatically by ``PostgresSaver.setup()`` in
    :mod:`graph.builder` — not managed here.
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config import settings
from rich.console import Console

console = Console()


def get_connection():
    """Return a plain psycopg2 connection (not the LangGraph checkpointer)."""
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


# ─── SQL Definitions ────────────────────────────────────────────────────────

CREATE_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS memory_store (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    session_id      TEXT,
    memory_type     TEXT        NOT NULL,   -- FACTUAL | PREFERENCE | EPHEMERAL
    content         TEXT        NOT NULL,   -- the raw memory string
    pinecone_id     TEXT UNIQUE,            -- reference to the vector record
    importance      FLOAT       DEFAULT 0.5,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,            -- NULL = never expires
    last_accessed   TIMESTAMPTZ DEFAULT NOW(),
    access_count    INT         DEFAULT 0
);
"""

CREATE_CONFLICT_LOG = """
CREATE TABLE IF NOT EXISTS conflict_log (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    old_memory_id   INT REFERENCES memory_store(id) ON DELETE SET NULL,
    new_content     TEXT,
    resolution      TEXT,       -- UPDATED | APPENDED | IGNORED
    consistency_hit BOOLEAN,    -- did the agent correctly detect intent change?
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_EXPERIMENT_LOG = """
CREATE TABLE IF NOT EXISTS experiment_metrics (
    id                  SERIAL PRIMARY KEY,
    run_id              TEXT,
    user_id             TEXT,
    query               TEXT,
    threshold_used      FLOAT,
    memories_retrieved  INT,
    memory_fetch_ms     FLOAT,
    llm_generation_ms   FLOAT,
    total_latency_ms    FLOAT,
    hallucination_flag  BOOLEAN,
    instruction_follow  BOOLEAN,
    index_type          TEXT,
    model_key           TEXT,        -- which model was used (e.g. gpt-4o-mini)
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_user    ON memory_store (user_id);",
    "CREATE INDEX IF NOT EXISTS idx_memory_type    ON memory_store (memory_type);",
    "CREATE INDEX IF NOT EXISTS idx_memory_expires ON memory_store (expires_at);",
    "CREATE INDEX IF NOT EXISTS idx_experiment_run ON experiment_metrics (run_id);",
]


def create_database_if_missing():
    """
    Connect to the default 'postgres' db and create our target db if absent.
    Required for a brand-new PostgreSQL installation.
    """
    try:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname="postgres",
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (settings.POSTGRES_DB,)
        )
        if not cur.fetchone():
            cur.execute(f"CREATE DATABASE {settings.POSTGRES_DB}")
            console.print(f"[green]✓ Database '{settings.POSTGRES_DB}' created.[/green]")
        else:
            console.print(f"[cyan]ℹ Database '{settings.POSTGRES_DB}' already exists.[/cyan]")
        conn.close()
    except Exception as e:
        console.print(f"[yellow]⚠ Could not auto-create DB: {e}[/yellow]")


def run_migrations():
    """Create all tables and indexes."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_MEMORY_TABLE)
            console.print("[green]✓ Table: memory_store[/green]")

            cur.execute(CREATE_CONFLICT_LOG)
            console.print("[green]✓ Table: conflict_log[/green]")

            cur.execute(CREATE_EXPERIMENT_LOG)
            console.print("[green]✓ Table: experiment_metrics[/green]")

            for idx_sql in CREATE_INDEXES:
                cur.execute(idx_sql)
            console.print("[green]✓ Indexes created[/green]")

        conn.commit()
        console.print("\n[bold green]✅ All migrations applied.[/bold green]")
    except Exception as e:
        conn.rollback()
        console.print(f"[bold red]✗ Migration failed: {e}[/bold red]")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    console.rule("[bold blue]PostgreSQL Setup[/bold blue]")
    create_database_if_missing()
    run_migrations()
