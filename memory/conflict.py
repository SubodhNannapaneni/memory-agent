"""
memory/conflict.py
───────────────────
Conflict-aware memory write pipeline (UPDATE / APPEND / IGNORE).

Before a new memory is committed, this module runs a four-step pipeline:

  1. **Classify** — the gatekeeper LLM decides whether the user's message
     is worth storing and distils it into a clean third-person sentence.
  2. **Retrieve candidates** — Milvus is queried for semantically similar
     existing memories using a high similarity threshold
     (``CONFLICT_THRESHOLD = 0.82``) to avoid false positives.
  3. **Detect conflict** — a second LLM call (``temperature=0``) decides:

       * ``UPDATE``  — the new memory supersedes an existing one.
       * ``APPEND``  — the new memory is genuinely additive.
       * ``IGNORE``  — the new memory is a near-duplicate.

  4. **Execute** — the decision is applied to both Milvus and PostgreSQL;
     every decision is logged to ``conflict_log`` with a ``consistency_hit``
     flag for the research metric reported in §IV-D of the paper.

Public entry point: :func:`resolve_and_store`.
"""
import json

import psycopg2
from openai import OpenAI
from config import settings
from db.postgres_setup import get_connection
from db.vector_store import query_similar_memories, upsert_memory, delete_memory
from memory.gatekeeper import save_to_postgres, classify_memory
from rich.console import Console

console = Console()
_llm = OpenAI(api_key=settings.OPENAI_API_KEY)

# High-threshold for conflict detection: we only want very similar matches
CONFLICT_THRESHOLD = 0.82


CONFLICT_DETECT_PROMPT = """You are a memory conflict detector.

A user AI assistant has a NEW memory candidate and a set of EXISTING memories
that are semantically similar. Decide:

  - "UPDATE"  → the new memory REPLACES or supersedes an existing one
                 (user changed their mind, preference evolved, fact corrected).
  - "APPEND"  → the new memory is genuinely NEW and should be added alongside existing ones.
  - "IGNORE"  → the new memory is essentially a duplicate; don't store it.

You must also identify WHICH existing memory ID to update/delete (if UPDATE).

New memory:
{new_content}

Existing similar memories:
{existing_json}

Respond with valid JSON ONLY:
{{
  "resolution": "UPDATE" | "APPEND" | "IGNORE",
  "target_id": <existing_pg_id or null>,
  "target_pinecone_id": "<pinecone_id or null>",
  "reasoning": "<one sentence>",
  "consistency_hit": true | false   // true = agent correctly detected intent change
}}
"""


def detect_conflict(
    new_content: str,
    similar_memories: list[dict],
    pg_id_map: dict,          # pinecone_id → postgres_id
) -> dict:
    """
    Use the LLM to decide if new_content conflicts with any similar_memories.

    pg_id_map lets the LLM reference PostgreSQL IDs (needed for deletion).
    """
    if not similar_memories:
        return {
            "resolution": "APPEND",
            "target_id": None,
            "target_pinecone_id": None,
            "reasoning": "No similar memories found.",
            "consistency_hit": False,
        }

    # Enrich similar_memories with postgres id for the LLM
    enriched = []
    for m in similar_memories:
        enriched.append(
            {
                "pinecone_id": m["id"],
                "pg_id": pg_id_map.get(m["id"], "unknown"),
                "content": m["content"],
                "type": m["memory_type"],
                "score": m["score"],
            }
        )

    prompt = CONFLICT_DETECT_PROMPT.format(
        new_content=new_content,
        existing_json=json.dumps(enriched, indent=2),
    )

    resp = _llm.chat.completions.create(
        model=settings.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {
            "resolution": "APPEND",
            "target_id": None,
            "target_pinecone_id": None,
            "reasoning": "JSON parse error – defaulting to APPEND.",
            "consistency_hit": False,
        }


def _fetch_pg_id_map(pinecone_ids: list[str]) -> dict:
    """Look up PostgreSQL row IDs for a list of Milvus vector record IDs."""
    if not pinecone_ids:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pinecone_id, id FROM memory_store WHERE pinecone_id = ANY(%s)",
                (pinecone_ids,),
            )
            return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        conn.close()


def _log_conflict(
    user_id: str,
    old_memory_id: int | None,
    new_content: str,
    resolution: str,
    consistency_hit: bool,
):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conflict_log
                    (user_id, old_memory_id, new_content, resolution, consistency_hit)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, old_memory_id, new_content, resolution, consistency_hit),
            )
        conn.commit()
    finally:
        conn.close()


def _delete_old_memory(pg_id: int, pinecone_id: str, user_id: str):
    """Hard-delete a superseded memory from both stores."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_store WHERE id = %s", (pg_id,))
        conn.commit()
    finally:
        conn.close()
    delete_memory(pinecone_id, user_id)


# ─── Public API ───────────────────────────────────────────────────────────────

def resolve_and_store(
    user_message: str,
    user_id: str,
    session_id: str,
    index_strategy: str = "PARTITIONED",
) -> dict:
    """
    Conflict-aware memory write.

    Pipeline:
      classify → query similar → detect conflict → UPDATE | APPEND | IGNORE
    """
    # 1. Classify the raw message
    classification = classify_memory(user_message)
    if not classification.get("should_save", False):
        return {"saved": False, "reason": "ephemeral chatter"}

    memory_type = classification.get("memory_type", "EPHEMERAL")
    importance = float(classification.get("importance", 0.5))
    new_content = classification.get("distilled_content", user_message)

    console.print(f"\n[bold magenta]⚡ Conflict check:[/bold magenta] {new_content[:70]}…")

    # 2. Find semantically similar existing memories
    similar, _ = query_similar_memories(
        query=new_content,
        user_id=user_id,
        top_k=5,
        threshold=CONFLICT_THRESHOLD,
        index_strategy=index_strategy,
    )

    # 3. Build pinecone_id → pg_id map
    pinecone_ids = [m["id"] for m in similar]
    pg_id_map = _fetch_pg_id_map(pinecone_ids)

    # 4. Ask LLM to resolve the conflict
    decision = detect_conflict(new_content, similar, pg_id_map)
    resolution = decision.get("resolution", "APPEND")
    old_pg_id = decision.get("target_id")
    old_pinecone_id = decision.get("target_pinecone_id")
    consistency_hit = decision.get("consistency_hit", False)
    reasoning = decision.get("reasoning", "")

    console.print(
        f"  ↳ Resolution: [bold]{resolution}[/bold]  |  {reasoning}"
    )

    # 5. Execute the resolution
    if resolution == "IGNORE":
        _log_conflict(user_id, old_pg_id, new_content, resolution, consistency_hit)
        return {"saved": False, "reason": "duplicate – IGNORED", "resolution": resolution}

    if resolution == "UPDATE" and old_pg_id and old_pinecone_id:
        console.print(f"  ↳ [yellow]Deleting superseded memory pg_id={old_pg_id}[/yellow]")
        _delete_old_memory(old_pg_id, old_pinecone_id, user_id)

    # APPEND (or UPDATE after deletion): write the new memory
    pinecone_id, embed_ms = upsert_memory(
        content=new_content,
        user_id=user_id,
        memory_type=memory_type,
        importance=importance,
        index_strategy=index_strategy,
    )
    pg_id = save_to_postgres(
        user_id=user_id,
        session_id=session_id,
        memory_type=memory_type,
        content=new_content,
        importance=importance,
        pinecone_id=pinecone_id,
    )

    _log_conflict(user_id, old_pg_id, new_content, resolution, consistency_hit)

    console.print(f"  ↳ [green]Stored[/green] pg_id={pg_id}, embed_ms={embed_ms:.1f}")

    return {
        "saved": True,
        "resolution": resolution,
        "pg_id": pg_id,
        "pinecone_id": pinecone_id,
        "memory_type": memory_type,
        "importance": importance,
        "consistency_hit": consistency_hit,
        "embed_ms": embed_ms,
    }
