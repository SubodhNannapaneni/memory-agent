"""
memory/gatekeeper.py
─────────────────────
The Memory Gatekeeper — classification and initial storage.

This module sits between every raw conversation turn and the persistent
memory stores.  Before any write occurs it answers three questions:

  1. Is this worth storing long-term at all?
  2. Which category is it?  (``FACTUAL`` / ``PREFERENCE`` / ``EPHEMERAL``)
  3. What importance weight should it carry?

An LLM call with structured output (``response_format={"type": "json_object"}``
at ``temperature=0``) performs the classification.  On a positive decision the
module writes to two stores:

  * **Milvus** — the embedding vector via :func:`db.vector_store.upsert_memory`
  * **PostgreSQL** — the metadata row via :func:`db.postgres_setup.get_connection`

See §III-C of the accompanying paper for the gatekeeper design rationale.
"""
import json
from datetime import datetime, timedelta, timezone

import psycopg2
from openai import OpenAI
from config import settings
from db.postgres_setup import get_connection
from db.vector_store import upsert_memory
from rich.console import Console

console = Console()
_llm = OpenAI(api_key=settings.OPENAI_API_KEY)


# ─── Classification prompt ───────────────────────────────────────────────────

CLASSIFY_PROMPT = """You are a memory classification agent.

Given a user message, decide:
1. `should_save` (bool): Is this information worth storing long-term?
   - Save: facts about the user, preferences, goals, important context.
   - Do NOT save: greetings, small talk, one-off questions with no personal info.

2. `memory_type` (str): One of:
   - "FACTUAL"    → verifiable facts (name, job, location, skills)
   - "PREFERENCE" → likes, dislikes, style choices, recurring desires
   - "EPHEMERAL"  → temporary context (current mood, today's task) – short TTL

3. `importance` (float 0.0–1.0): How important is this for future interactions?

4. `distilled_content` (str): A clean, third-person memory sentence.
   Example: "User prefers dark mode in their IDE."

Respond ONLY with valid JSON. No markdown fences.

User message:
{message}
"""


def classify_memory(user_message: str) -> dict:
    """
    Call the LLM to classify whether and how to save a user message.

    Returns a dict with keys:
      should_save, memory_type, importance, distilled_content
    """
    prompt = CLASSIFY_PROMPT.format(message=user_message)
    resp = _llm.chat.completions.create(
        model=settings.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: don't save if classification fails
        return {"should_save": False}


# ─── Storage helpers ─────────────────────────────────────────────────────────

def _compute_expiry(memory_type: str) -> datetime | None:
    ttl_days = settings.MEMORY_TYPES.get(memory_type, {}).get("ttl_days", 30)
    if ttl_days >= 365:
        return None  # treat as "never expires"
    return datetime.now(tz=timezone.utc) + timedelta(days=ttl_days)


def save_to_postgres(
    user_id: str,
    session_id: str,
    memory_type: str,
    content: str,
    importance: float,
    pinecone_id: str,
) -> int:
    """Insert a memory metadata row; return the new row id."""
    expires_at = _compute_expiry(memory_type)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_store
                    (user_id, session_id, memory_type, content,
                     pinecone_id, importance, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, session_id, memory_type, content,
                 pinecone_id, importance, expires_at),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        return row_id
    finally:
        conn.close()


# ─── Main Gatekeeper Function ─────────────────────────────────────────────────

def gatekeeper_node(
    user_message: str,
    user_id: str,
    session_id: str,
    index_strategy: str = "PARTITIONED",
) -> dict:
    """
    The Memory Filter / Gatekeeper node.

    Call this after each user message to conditionally persist memory.

    Returns a summary dict describing what was stored (or why it was skipped).
    """
    console.print(f"\n[bold cyan]🔍 Gatekeeper evaluating:[/bold cyan] {user_message[:80]}…")

    # Step 1: LLM classification
    classification = classify_memory(user_message)

    if not classification.get("should_save", False):
        console.print("[dim]  ↳ Classified as ephemeral chatter. Skipping.[/dim]")
        return {"saved": False, "reason": "ephemeral chatter"}

    memory_type = classification.get("memory_type", "EPHEMERAL")
    importance = float(classification.get("importance", 0.5))
    content = classification.get("distilled_content", user_message)

    console.print(
        f"  ↳ [green]Saving[/green] as [bold]{memory_type}[/bold] "
        f"(importance={importance:.2f}): {content[:60]}…"
    )

    # Step 2: Upsert embedding to Milvus
    pinecone_id, embed_ms = upsert_memory(
        content=content,
        user_id=user_id,
        memory_type=memory_type,
        importance=importance,
        index_strategy=index_strategy,
    )
    console.print(f"  ↳ Milvus upsert done in {embed_ms:.1f} ms  (id={pinecone_id[:8]}…)")

    # Step 3: Save metadata to PostgreSQL
    pg_id = save_to_postgres(
        user_id=user_id,
        session_id=session_id,
        memory_type=memory_type,
        content=content,
        importance=importance,
        pinecone_id=pinecone_id,
    )
    console.print(f"  ↳ PostgreSQL row id={pg_id}")

    return {
        "saved": True,
        "pg_id": pg_id,
        "pinecone_id": pinecone_id,
        "memory_type": memory_type,
        "importance": importance,
        "content": content,
        "embed_ms": embed_ms,
    }
