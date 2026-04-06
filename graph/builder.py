"""
graph/builder.py
─────────────────────────
Assembles and compiles the LangGraph ``StateGraph``.

The compiled graph includes:
  * **PostgresSaver** — durable checkpoint storage backed by the same
    PostgreSQL instance used for memory metadata.  Full conversation history
    is preserved across process restarts.
  * **4 nodes** in a linear pipeline::

        retrieve_memory → llm_respond → conflict_write → log_metrics

Public API
──────────
:func:`build_graph`  — build and compile the graph (call once at startup)
:func:`run_turn`     — convenience wrapper to invoke the graph for one turn
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg.rows import dict_row

from config import settings
from graph.state import AgentState
from graph.nodes import retrieve_memory, llm_respond, conflict_write, log_metrics

# Module-level persistent connection — stays open for the lifetime of the process
_pg_conn: psycopg.Connection | None = None


def _get_checkpointer() -> PostgresSaver:
    """
    Return a PostgresSaver backed by a persistent psycopg3 connection.
    Opens the connection once and reuses it across build_graph() calls.
    """
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg.connect(
            settings.postgres_dsn,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
    saver = PostgresSaver(_pg_conn)
    saver.setup()   # creates langgraph checkpoint tables if not present (idempotent)
    return saver


def build_graph() -> StateGraph:
    """
    Build and compile the memory-augmented agent graph.

    The PostgresSaver uses the same Postgres DB as the rest of the app,
    but stores checkpoints in a separate set of tables it manages itself
    (langgraph_checkpoints, langgraph_writes, etc.).
    """
    # ── Checkpointer (persistent conversation state) ─────────────────────
    checkpointer = _get_checkpointer()

    # ── Graph definition ─────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("llm_respond", llm_respond)
    graph.add_node("conflict_write", conflict_write)
    graph.add_node("log_metrics", log_metrics)

    # Wire edges (linear pipeline for now)
    graph.add_edge(START, "retrieve_memory")
    graph.add_edge("retrieve_memory", "llm_respond")
    graph.add_edge("llm_respond", "conflict_write")
    graph.add_edge("conflict_write", "log_metrics")
    graph.add_edge("log_metrics", END)

    return graph.compile(checkpointer=checkpointer)


# ── Convenience: run a single turn ───────────────────────────────────────────

def run_turn(
    graph,
    user_message: str,
    user_id: str,
    session_id: str,
    run_id: str = "default",
    threshold: float = None,
    index_strategy: str = "PARTITIONED",
    model_key: str = None,
) -> dict:
    """
    Helper to invoke the graph for one conversation turn.
    Returns the final state dict.
    """
    from langchain_core.messages import HumanMessage
    from config import settings as cfg

    initial_state: AgentState = {
        "messages":          [HumanMessage(content=user_message)],
        "user_id":           user_id,
        "session_id":        session_id,
        "retrieved_memories":[],
        "memory_fetch_ms":   0.0,
        "index_strategy":    index_strategy,
        "last_memory_write": {},
        "llm_generation_ms": 0.0,
        "run_id":            run_id,
        "threshold_used":    threshold or cfg.MEMORY_SIMILARITY_THRESHOLD,
        "model_key":         model_key or cfg.DEFAULT_MODEL,
        "next_action":       "",
    }

    config = {"configurable": {"thread_id": session_id}}
    return graph.invoke(initial_state, config=config)
