"""
db/vector_store.py
───────────────────
Milvus vector store for long-term semantic memory.

This module owns all Milvus CRUD — embedding, upsert, similarity search, and
deletion.  Two index strategies are supported and compared empirically in the
accompanying paper (§IV-C):

  FLAT        → IVF_FLAT index, all users in a single collection, filtered by
                ``user_id`` at query time.
  PARTITIONED → one Milvus partition per ``user_id``; scopes the ANN search to
                that partition, reducing p95 tail latency by ~28 %.

Embeddings are generated via OpenAI ``text-embedding-3-large`` with MRL
truncation to 1536 dimensions (50 % of the native 3072-dim output).  The
truncation halves storage and search cost while retaining ≈96 % of full-dim
retrieval quality — see §III-B of the paper for the rationale.

Free local alternative
───────────────────────
To avoid the OpenAI embedding API entirely, swap ``embed_text()`` for a local
SentenceTransformer model::

    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
    def embed_text(text: str) -> list[float]:
        return _st_model.encode(text).tolist()
    # Also set EMBED_DIM = 384 in config.py

Start Milvus locally (Docker)::

    docker run --name milvus-standalone \\
        -p 19530:19530 -p 9091:9091 \\
        milvusdb/milvus:latest standalone
"""
import time
import uuid
from typing import Optional

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from openai import OpenAI
from config import settings
from rich.console import Console

console = Console()

# ── Singleton clients ─────────────────────────────────────────────────────────
_openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ── Schema constants ──────────────────────────────────────────────────────────
_COLLECTION   = settings.MILVUS_COLLECTION
_DIM          = settings.EMBED_DIM
_MAX_STR      = 1024    # max chars for string fields
_INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type":  "IVF_FLAT",
    "params":      {"nlist": 128},
}
_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"nprobe": 16}}


# ── Connect & Collection setup ────────────────────────────────────────────────

def _connect():
    """Establish connection to the local Milvus instance."""
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=str(settings.MILVUS_PORT),
    )


def _get_or_create_collection() -> Collection:
    """
    Create the Milvus collection + index if it doesn't exist yet.
    Returns a Collection object ready for insert/search.
    """
    _connect()

    if utility.has_collection(_COLLECTION):
        col = Collection(_COLLECTION)
        col.load()
        return col

    # Define schema
    fields = [
        FieldSchema(name="id",          dtype=DataType.VARCHAR, max_length=64,   is_primary=True),
        FieldSchema(name="user_id",     dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="content",     dtype=DataType.VARCHAR, max_length=_MAX_STR),
        FieldSchema(name="importance",  dtype=DataType.FLOAT),
        FieldSchema(name="embedding",   dtype=DataType.FLOAT_VECTOR, dim=_DIM),
    ]
    schema = CollectionSchema(fields, description="Long-term agent memory")
    col = Collection(name=_COLLECTION, schema=schema)

    # Create COSINE index on the embedding field
    col.create_index(field_name="embedding", index_params=_INDEX_PARAMS)
    col.load()

    console.print(f"[green]✓ Milvus collection '{_COLLECTION}' created & indexed.[/green]")
    return col


# ── Embedding helper ──────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Return a 1536-dimensional embedding vector for *text*.

    Uses ``text-embedding-3-large`` with Matryoshka Representation Learning
    (MRL) truncation via the OpenAI ``dimensions`` parameter.  The native
    model produces 3072-dim vectors; truncating to 1536 halves storage and
    ANN-search cost while retaining ≈96 % of semantic quality (see config.py
    for the EMBED_MODEL / EMBED_DIM settings).

    Free local alternative — replace this function body with::

        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
        return _st_model.encode(text).tolist()
        # Also set EMBED_DIM = 384 in config.py
    """
    resp = _openai_client.embeddings.create(
        model=settings.EMBED_MODEL,
        input=text,
        dimensions=settings.EMBED_DIM,   # MRL truncation: large-model quality at 1536-dim
    )
    return resp.data[0].embedding


# ── Core CRUD ─────────────────────────────────────────────────────────────────

def upsert_memory(
    content: str,
    user_id: str,
    memory_type: str,
    importance: float = 0.5,
    pinecone_id: Optional[str] = None,   # legacy parameter name — used as the Milvus record ID
    index_strategy: str = "PARTITIONED",
) -> tuple[str, float]:
    """
    Embed *content* and upsert it into Milvus.

    Parameters
    ----------
    content:
        The raw memory string to embed and store.
    user_id:
        Scopes the record to a specific user.
    memory_type:
        One of ``"FACTUAL"``, ``"PREFERENCE"``, or ``"EPHEMERAL"``.
    importance:
        Float in [0, 1] used for future retrieval ranking.
    pinecone_id:
        Optional caller-supplied record ID (kept for backward compatibility
        with the PostgreSQL schema column name ``pinecone_id``).
        When *None*, a new UUID is generated.
    index_strategy:
        ``"PARTITIONED"`` (default, recommended) or ``"FLAT"``.

    Returns
    -------
    tuple[str, float]
        ``(record_id, embed_time_ms)`` — the vector record ID and the time
        taken to generate the embedding.
    """
    record_id = pinecone_id or str(uuid.uuid4())
    col = _get_or_create_collection()

    # Ensure partition exists for PARTITIONED strategy
    partition_name = None
    if index_strategy == "PARTITIONED":
        safe_name = f"u_{user_id[:32].replace('-', '_')}"   # Milvus partition name rules
        if not col.has_partition(safe_name):
            col.create_partition(safe_name)
        partition_name = safe_name

    t0 = time.perf_counter()
    vector = embed_text(content)
    embed_ms = (time.perf_counter() - t0) * 1000

    data = [
        [record_id],
        [user_id],
        [memory_type],
        [content[:_MAX_STR]],
        [importance],
        [vector],
    ]

    # Delete existing record with same id before inserting (upsert behaviour)
    try:
        col.delete(expr=f'id == "{record_id}"')
    except Exception:
        pass

    insert_kwargs = {"data": data}
    if partition_name:
        insert_kwargs["partition_name"] = partition_name

    col.insert(**insert_kwargs)
    col.flush()

    return record_id, embed_ms


def query_similar_memories(
    query: str,
    user_id: str,
    top_k: int = None,
    threshold: float = None,
    index_strategy: str = "PARTITIONED",
) -> tuple[list[dict], float]:
    """
    Retrieve semantically similar memories for a given user.

    Parameters
    ----------
    query:
        The natural-language question or statement to embed and search against.
    user_id:
        Restricts results to memories belonging to this user.
    top_k:
        Maximum number of candidates to retrieve from Milvus before threshold
        filtering.  Defaults to ``settings.MAX_MEMORIES_PER_QUERY``.
    threshold:
        Minimum cosine similarity score for a result to be included.
        Defaults to ``settings.MEMORY_SIMILARITY_THRESHOLD``.
    index_strategy:
        ``"PARTITIONED"`` (default) or ``"FLAT"``.

    Returns
    -------
    tuple[list[dict], float]
        A list of memory dicts (keys: ``id``, ``score``, ``content``,
        ``memory_type``, ``importance``) and the total fetch time in ms.
    """
    top_k     = top_k     or settings.MAX_MEMORIES_PER_QUERY
    threshold = threshold or settings.MEMORY_SIMILARITY_THRESHOLD
    col       = _get_or_create_collection()

    # Choose search scope
    partition_names = None
    if index_strategy == "PARTITIONED":
        safe_name = f"u_{user_id[:32].replace('-', '_')}"
        if col.has_partition(safe_name):
            partition_names = [safe_name]
        else:
            return [], 0.0   # user has no memories yet

    t0 = time.perf_counter()
    query_vector = embed_text(query)

    search_kwargs: dict = {
        "data":          [query_vector],
        "anns_field":    "embedding",
        "param":         _SEARCH_PARAMS,
        "limit":         top_k,
        "output_fields": ["id", "user_id", "content", "memory_type", "importance"],
        "expr":          f'user_id == "{user_id}"',   # safety filter for FLAT mode
    }
    if partition_names:
        search_kwargs["partition_names"] = partition_names

    results  = col.search(**search_kwargs)
    fetch_ms = (time.perf_counter() - t0) * 1000

    memories = []
    for hits in results:
        for hit in hits:
            # Milvus COSINE returns distance in [0,1]; convert to similarity
            score = 1.0 - hit.distance   # 1 = identical, 0 = orthogonal
            if score >= threshold:
                memories.append(
                    {
                        "id":          hit.entity.get("id"),
                        "score":       round(score, 4),
                        "content":     hit.entity.get("content", ""),
                        "memory_type": hit.entity.get("memory_type", "UNKNOWN"),
                        "importance":  hit.entity.get("importance", 0.5),
                    }
                )

    return memories, fetch_ms


def delete_memory(
    record_id: str,
    user_id: str,
    index_strategy: str = "PARTITIONED",
):
    """Remove a single memory vector from Milvus."""
    col = _get_or_create_collection()
    col.delete(expr=f'id == "{record_id}"')
    col.flush()


def bulk_delete_memories(
    record_ids: list[str],
    user_id: str,
    index_strategy: str = "PARTITIONED",
):
    """Batch-delete expired/pruned memories."""
    if not record_ids:
        return
    col  = _get_or_create_collection()
    ids_str = ", ".join(f'"{rid}"' for rid in record_ids)
    col.delete(expr=f"id in [{ids_str}]")
    col.flush()
    console.print(f"[yellow]🗑  Milvus: deleted {len(record_ids)} vectors.[/yellow]")
