"""
graph/nodes.py
─────────────────────────
All four LangGraph node functions for the memory agent pipeline.

Pipeline order (wired in ``graph/builder.py``):

  1. :func:`retrieve_memory`  — semantic search in Milvus
  2. :func:`llm_respond`      — memory-augmented LLM call
  3. :func:`conflict_write`   — conflict-aware memory persistence
  4. :func:`log_metrics`      — latency row written to PostgreSQL

The LLM is fully provider-agnostic.  Pass any key from ``config.MODELS``
into ``AgentState.model_key`` and the correct LangChain client is
instantiated automatically via :func:`_get_llm`.

Supported providers: ``openai`` | ``anthropic`` | ``gemini`` | ``groq``
"""
import time
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import settings
from db.postgres_setup import get_connection
from db.vector_store import query_similar_memories
from memory.conflict import resolve_and_store
from graph.state import AgentState
from rich.console import Console

console = Console()


# ── Provider factory ──────────────────────────────────────────────────────────

def _get_llm(model_key: str):
    """
    Return the correct LangChain chat model for the given model key.
    model_key must be one of the keys in config.MODELS.
    """
    model_cfg = settings.MODELS.get(model_key)
    if not model_cfg:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available: {list(settings.MODELS.keys())}"
        )

    provider  = model_cfg["provider"]
    model_id  = model_cfg["model_id"]

    # Per-model temperature — o-series reasoning models require temperature=1
    temperature = model_cfg.get("temperature", 0.7)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_id,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_id,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
        )

    elif provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to .env to enable Groq models."
            )
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_id,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )

    raise ValueError(f"Unknown provider '{provider}'")


# ── Node 1: retrieve_memory ───────────────────────────────────────────────────

def retrieve_memory(state: AgentState) -> dict:
    """Fetch semantically relevant memories from Milvus."""
    last_msg   = state["messages"][-1]
    query_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    threshold      = state.get("threshold_used", settings.MEMORY_SIMILARITY_THRESHOLD)
    index_strategy = state.get("index_strategy", "PARTITIONED")

    memories, fetch_ms = query_similar_memories(
        query=query_text,
        user_id=state["user_id"],
        threshold=threshold,
        index_strategy=index_strategy,
    )

    console.print(
        f"\n[cyan]🧠 Milvus: {len(memories)} memories retrieved[/cyan] "
        f"(threshold={threshold}, {fetch_ms:.1f} ms)"
    )

    return {
        "retrieved_memories": memories,
        "memory_fetch_ms":    fetch_ms,
    }


# ── Node 2: llm_respond ───────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a personalized AI assistant with long-term memory.

The following memories about this user are relevant to the current conversation.
Use them to personalise your response. Do NOT hallucinate facts not in these memories.

--- MEMORIES ---
{memory_block}
--- END MEMORIES ---

Current date: {date}
"""

def llm_respond(state: AgentState) -> dict:
    """
    Build a memory-augmented prompt and call the configured LLM.
    The model is selected from state['model_key'] — defaults to config DEFAULT_MODEL.
    """
    memories  = state.get("retrieved_memories", [])
    model_key = state.get("model_key", settings.DEFAULT_MODEL)

    if memories:
        memory_block = "\n".join(
            f"• [{m['memory_type']}] (score={m['score']}) {m['content']}"
            for m in memories
        )
    else:
        memory_block = "No relevant memories found for this query."

    from datetime import date
    system_msg = SystemMessage(
        content=SYSTEM_TEMPLATE.format(
            memory_block=memory_block,
            date=date.today().isoformat(),
        )
    )

    full_messages = [system_msg] + list(state["messages"])

    llm = _get_llm(model_key)
    provider = settings.MODELS[model_key]["provider"]

    console.print(f"[bold]🤖 Calling[/bold] [magenta]{model_key}[/magenta] ({provider})…")

    t0       = time.perf_counter()
    response = llm.invoke(full_messages)
    llm_ms   = (time.perf_counter() - t0) * 1000

    console.print(f"   ↳ Response in [green]{llm_ms:.1f} ms[/green]")

    return {
        "messages":          [response],
        "llm_generation_ms": llm_ms,
    }


# ── Node 3: conflict_write ────────────────────────────────────────────────────

def conflict_write(state: AgentState) -> dict:
    """Persist any new memory from the user's latest message via conflict detection."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if not last_human:
        return {"last_memory_write": {"saved": False, "reason": "no human message"}}

    result = resolve_and_store(
        user_message=last_human.content,
        user_id=state["user_id"],
        session_id=state["session_id"],
        index_strategy=state.get("index_strategy", "PARTITIONED"),
    )
    return {"last_memory_write": result}


# ── Node 4: log_metrics ───────────────────────────────────────────────────────

def log_metrics(state: AgentState) -> dict:
    """Write a latency row to experiment_metrics for analysis."""
    fetch_ms = state.get("memory_fetch_ms", 0.0)
    llm_ms   = state.get("llm_generation_ms", 0.0)
    total_ms = fetch_ms + llm_ms

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query_text = last_human.content[:500] if last_human else ""
    model_key  = state.get("model_key", settings.DEFAULT_MODEL)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiment_metrics
                  (run_id, user_id, query, threshold_used,
                   memories_retrieved, memory_fetch_ms, llm_generation_ms,
                   total_latency_ms, index_type, model_key)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    state.get("run_id", str(uuid.uuid4())[:8]),
                    state["user_id"],
                    query_text,
                    state.get("threshold_used", settings.MEMORY_SIMILARITY_THRESHOLD),
                    len(state.get("retrieved_memories", [])),
                    round(fetch_ms, 2),
                    round(llm_ms,   2),
                    round(total_ms, 2),
                    state.get("index_strategy", "PARTITIONED"),
                    model_key,
                ),
            )
        conn.commit()
    except Exception:
        # model_key column may not exist on old schemas — add it gracefully
        conn.rollback()
    finally:
        conn.close()

    console.print(
        f"[dim]📊 fetch={fetch_ms:.1f}ms  llm={llm_ms:.1f}ms  "
        f"total={total_ms:.1f}ms  model={model_key}[/dim]"
    )
