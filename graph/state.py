"""
graph/state.py
─────────────────────────
Shared state definition for the LangGraph agent.

``AgentState`` is the single TypedDict that flows through every node in the
graph.  Each node receives the current state and returns a *partial* dict
containing only the fields it modified; LangGraph merges the partial updates
back into the state automatically.

All nodes must be compatible with this schema.  Adding a new field here is
the correct way to thread new data between nodes.
"""
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]   # full chat history (HumanMessage / AIMessage)
    user_id: str
    session_id: str

    # ── Memory context ────────────────────────────────────────────────────
    retrieved_memories: list[dict]            # memories fetched for this turn
    memory_fetch_ms: float                    # latency of Milvus retrieval
    index_strategy: str                       # "FLAT" | "PARTITIONED"

    # ── Gatekeeper / conflict output ──────────────────────────────────────
    last_memory_write: dict                   # result from gatekeeper/conflict node

    # ── Experiment metrics ────────────────────────────────────────────────
    llm_generation_ms: float
    run_id: str                               # tie a batch of turns together
    threshold_used: float

    # ── Model selection ───────────────────────────────────────────────────
    model_key: str                            # key from config.MODELS e.g. "gpt-4o-mini"

    # ── Routing ───────────────────────────────────────────────────────────
    next_action: str                          # used by conditional edges
