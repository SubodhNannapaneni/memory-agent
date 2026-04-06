"""
memory/decay.py
────────────────
The Memory Forgetting Mechanism — TTL pruning and semantic deduplication.

Two complementary strategies are applied in sequence by
:func:`run_decay_cycle`:

  1. **TTL Pruning** — hard-deletes rows from PostgreSQL (and their
     corresponding Milvus vectors) whose ``expires_at`` timestamp has passed.
     TTL values are set at write time by the gatekeeper:
     ``FACTUAL`` → 365 days, ``PREFERENCE`` → 90 days, ``EPHEMERAL`` → 1 day.

  2. **Semantic Pruning** — sends the most recent ``batch_size`` memories to
     an LLM (``temperature=0``) which identifies redundant or superseded pairs
     and returns their IDs for deletion.  This catches logical obsolescence
     that TTL cannot — e.g. an old preference overridden by a new one.

Usage::

    # Run from the command line (suitable for a nightly cron job):
    python -m memory.decay --user_id <uid>

    # Or call programmatically from a scheduler:
    from memory.decay import run_decay_cycle
    run_decay_cycle(user_id)
"""
import argparse
import json
from datetime import datetime, timezone

from openai import OpenAI
from config import settings
from db.postgres_setup import get_connection
from db.vector_store import bulk_delete_memories
from rich.console import Console
from rich.table import Table

console = Console()
_llm = OpenAI(api_key=settings.OPENAI_API_KEY)


# ─── 1. TTL Pruning ──────────────────────────────────────────────────────────

def ttl_prune(user_id: str) -> list[str]:
    """
    Delete all memory rows whose ``expires_at`` timestamp is in the past.

    Removes the rows from PostgreSQL first, then deletes the corresponding
    vectors from Milvus via :func:`db.vector_store.bulk_delete_memories`.

    Returns
    -------
    list[str]
        The vector record IDs (``pinecone_id`` column) that were removed,
        useful for confirming Milvus cleanup.
    """
    now = datetime.now(tz=timezone.utc)
    conn = get_connection()
    deleted_pinecone_ids = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM memory_store
                WHERE user_id = %s
                  AND expires_at IS NOT NULL
                  AND expires_at < %s
                RETURNING pinecone_id
                """,
                (user_id, now),
            )
            rows = cur.fetchall()
            deleted_pinecone_ids = [r[0] for r in rows if r[0]]
        conn.commit()
    finally:
        conn.close()

    if deleted_pinecone_ids:
        console.print(
            f"[yellow]🗑  TTL pruned {len(deleted_pinecone_ids)} expired memories "
            f"for user '{user_id}'.[/yellow]"
        )
        bulk_delete_memories(deleted_pinecone_ids, user_id)
    else:
        console.print("[dim]  TTL: no expired memories found.[/dim]")

    return deleted_pinecone_ids


# ─── 2. Semantic Pruning ─────────────────────────────────────────────────────

SEMANTIC_PRUNE_PROMPT = """You are a memory auditor for an AI assistant.

Below are the stored memories for a user. Identify pairs or groups that are:
  (a) REDUNDANT  – essentially duplicates
  (b) SUPERSEDED – an older memory contradicts a newer one
      (e.g., "User likes Python" and "User now prefers Rust → Python entry is superseded")

For each memory that should be deleted, return its `id`.

Respond with a JSON object:
{{
  "to_delete": [<id>, <id>, ...],
  "reasoning": "<brief explanation>"
}}

Respond ONLY with valid JSON. No markdown.

Memories:
{memories_json}
"""


def semantic_prune(user_id: str, batch_size: int = 50) -> list[int]:
    """
    Ask the LLM to identify redundant / superseded memories and delete them.
    Returns list of PostgreSQL row IDs that were removed.
    """
    conn = get_connection()

    # Fetch the most recent `batch_size` memories for this user
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, memory_type, content, created_at
                FROM memory_store
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, batch_size),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        console.print("[dim]  Semantic pruning: no memories to review.[/dim]")
        return []

    memories_json = json.dumps(
        [
            {
                "id": r[0],
                "type": r[1],
                "content": r[2],
                "created_at": str(r[3]),
            }
            for r in rows
        ],
        indent=2,
    )

    prompt = SEMANTIC_PRUNE_PROMPT.format(memories_json=memories_json)
    resp = _llm.chat.completions.create(
        model=settings.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        console.print("[red]✗ Semantic pruning: LLM returned invalid JSON.[/red]")
        return []

    to_delete_ids = result.get("to_delete", [])
    reasoning = result.get("reasoning", "")

    if not to_delete_ids:
        console.print("[dim]  Semantic pruning: no redundant memories found.[/dim]")
        return []

    console.print(
        f"[yellow]🧠 Semantic pruning will delete {len(to_delete_ids)} memories.[/yellow]"
    )
    console.print(f"   Reason: {reasoning}")

    # Fetch pinecone_ids before deleting
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pinecone_id FROM memory_store WHERE id = ANY(%s)",
                (to_delete_ids,),
            )
            pinecone_ids = [r[0] for r in cur.fetchall() if r[0]]

            cur.execute(
                "DELETE FROM memory_store WHERE id = ANY(%s)", (to_delete_ids,)
            )
        conn.commit()
    finally:
        conn.close()

    bulk_delete_memories(pinecone_ids, user_id)
    return to_delete_ids


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_decay_cycle(user_id: str):
    """Run both TTL and semantic pruning for a given user."""
    console.rule(f"[bold]Memory Decay Cycle – user={user_id}[/bold]")

    ttl_deleted = ttl_prune(user_id)
    semantic_deleted = semantic_prune(user_id)

    table = Table(title="Decay Summary")
    table.add_column("Strategy", style="cyan")
    table.add_column("Deleted", justify="right", style="red")
    table.add_row("TTL Pruning", str(len(ttl_deleted)))
    table.add_row("Semantic Pruning", str(len(semantic_deleted)))
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory decay cycle.")
    parser.add_argument("--user_id", required=True, help="Target user ID")
    args = parser.parse_args()
    run_decay_cycle(args.user_id)
