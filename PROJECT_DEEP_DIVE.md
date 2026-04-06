# Memory-Agent: Complete Project Deep Dive
**A production-grade LangGraph agent with persistent long-term memory — full architecture, every file explained, data strategy, and practical evaluation.**

---

## Table of Contents

1. [Core Objective](#1-core-objective)
2. [Why This Problem Matters](#2-why-this-problem-matters)
3. [System Architecture](#3-system-architecture)
4. [Infrastructure Stack](#4-infrastructure-stack)
5. [Every File Explained](#5-every-file-explained)
   - [config.py](#51-configpy)
   - [main.py](#52-mainpy)
   - [graph/state.py](#53-graphstatepy)
   - [graph/nodes.py](#54-graphnodespy)
   - [graph/builder.py](#55-graphbuilderpy)
   - [memory/gatekeeper.py](#56-memorygatekeeperpy)
   - [memory/conflict.py](#57-memoryconflictpy)
   - [memory/decay.py](#58-memorydecaypy)
   - [db/postgres_setup.py](#59-dbpostgres_setuppy)
   - [db/vector_store.py](#510-dbvector_storepy)
   - [experiments/data_loader.py](#511-experimentsdata_loaderpy)
   - [experiments/benchmarks.py](#512-experimentsbenchmarkspy)
   - [experiments/model_comparison.py](#513-experimentsmodel_comparisonpy)
   - [experiments/run_research.py](#514-experimentsrun_researchpy)
6. [Complete Data Flow — Turn by Turn](#6-complete-data-flow--turn-by-turn)
7. [The Data — What, Why, and How Practical](#7-the-data--what-why-and-how-practical)
   - [PersonaChat Dataset](#71-personachat)
   - [DailyDialog Dataset](#72-dailydialog)
   - [MSC (Multi-Session Chat)](#73-msc-multi-session-chat)
   - [LoCoMo Dataset](#74-locomo)
   - [Synthetic Seed Data](#75-synthetic-seed-data)
   - [Is the Data Real-Time?](#76-is-the-data-real-time)
8. [Embedding Design Decisions](#8-embedding-design-decisions)
9. [PostgreSQL Schema — Every Table](#9-postgresql-schema--every-table)
10. [Key Design Decisions and Trade-offs](#10-key-design-decisions-and-trade-offs)
11. [LLM Provider Abstraction](#11-llm-provider-abstraction)
12. [Experiment Results at a Glance](#12-experiment-results-at-a-glance)

---

## 1. Core Objective

> **Build a conversational AI agent that genuinely remembers users across sessions — and rigorously measure whether that memory is accurate, efficient, and consistent across different LLM providers.**

Most chatbots are stateless: every conversation starts from zero. This project solves that by giving an LLM agent a **two-layer persistent memory store** (semantic vectors in Milvus + structured metadata in PostgreSQL) and connecting it via a LangGraph pipeline that automatically:

- **Saves** relevant facts and preferences from every user message
- **Retrieves** the right memories when they are relevant to a new question
- **Detects and resolves conflicts** when a user's beliefs change
- **Prunes expired or superseded** memories over time

The research dimension goes further: the project runs five controlled experiments to quantify *how well* this memory system actually works, producing publication-ready data comparing 12 LLMs across OpenAI, Anthropic, and Google Gemini families.

**In one sentence**: It is a scientific testbed for long-term memory in LLM agents, built entirely in Python, running locally on Docker.

---

## 2. Why This Problem Matters

| Pain point | How this project addresses it |
|---|---|
| LLMs forget everything between sessions | Persistent dual-store memory (PostgreSQL + Milvus) |
| Generic chatbots give one-size-fits-all answers | Memory-augmented prompts personalize every response |
| Popular memory systems (MemGPT, etc.) are black boxes | Every component is open, inspectable, and measurable |
| Hard to compare LLMs on memory tasks | Standardised 12-model benchmark with identical test harness |
| Memory gets stale or contradictory | Conflict detection + TTL decay keeps memory accurate |

---

## 3. System Architecture

```
User message
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LangGraph Graph                             │
│                                                                     │
│  ┌─────────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │ retrieve_memory │ →  │  llm_respond  │ →  │ conflict_write   │  │
│  │                 │    │               │    │                  │  │
│  │ Milvus cosine   │    │ Inject memory │    │ Gatekeeper:      │  │
│  │ similarity      │    │ block into    │    │ classify → save  │  │
│  │ search          │    │ system prompt │    │ or update/ignore │  │
│  └─────────────────┘    └───────────────┘    └──────────────────┘  │
│           ↑                     ↑                      ↓            │
│     Milvus vector          LLM API call          Conflict check     │
│     store (HNSW)           (any provider)        → Milvus upsert   │
│                                                  → Postgres insert  │
│                                    ┌─────────────────┐             │
│                                    │  log_metrics    │             │
│                                    │ Postgres insert │             │
│                                    │ latency row     │             │
│                                    └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
AI response (memory-augmented, personalized)
```

**LangGraph** provides the execution graph — a linear pipeline of four nodes where each node reads from and writes to a shared `AgentState` dictionary. The graph is compiled once with a `PostgresSaver` checkpointer that persists the full conversation history across sessions.

---

## 4. Infrastructure Stack

| Component | Technology | Why chosen |
|---|---|---|
| Conversation graph | **LangGraph** (StateGraph) | Declarative DAG, built-in checkpointing, hot-swappable nodes |
| LLM abstraction | **LangChain** chat models | Single interface for OpenAI, Anthropic, Gemini, Groq |
| Semantic memory | **Milvus** (Docker) | Free, local, HNSW index, partition support — no cloud cost |
| Structured metadata | **PostgreSQL** (Docker) | Reliable, queryable, TTL expiry, conflict logging |
| Conversation history | **PostgresSaver** (LangGraph) | Same Postgres DB, managed checkpoint tables |
| Embeddings | **OpenAI `text-embedding-3-large`** | 1536-dim MRL, best quality/cost for research |
| Dependency injection | **`config.py` singleton** | All parameters in one file, zero magic |
| Output/display | **Rich** (tables, panels) | Terminal-readable experiment output |

All services run locally in Docker; no cloud infrastructure is required beyond API keys for LLM providers.

---

## 5. Every File Explained

### 5.1 `config.py`

**What it does**: The single source of truth for the entire project. Contains every tunable parameter — API keys (loaded from `.env`), all model definitions, database connection strings, and memory behaviour knobs.

**Key sections**:

```python
# Every model in the registry — 24 total across 4 providers
MODELS: dict = {
    "gpt-4.1-mini": {
        "provider": "openai",
        "model_id": "gpt-4.1-mini",
        "cost_per_1k_in": 0.0004,
        "cost_per_1k_out": 0.0016,
    },
    # ... 23 more models
}
```

The `ACTIVE_MODELS` property automatically filters out any model whose API key is absent — so if you only have OpenAI credentials, Gemini and Groq models silently disappear from the available list.

```python
# Memory threshold — the most important tuning knob in the whole project
MEMORY_SIMILARITY_THRESHOLD: float = 0.75   # cosine similarity cutoff

# Three memory types with different TTLs
MEMORY_TYPES = {
    "FACTUAL":   {"ttl_days": 365, "weight": 1.0},    # facts last 1 year
    "PREFERENCE":{"ttl_days": 90,  "weight": 0.85},   # preferences last 3 months
    "EPHEMERAL": {"ttl_days": 1,   "weight": 0.3},    # temporary context expires next day
}
```

**Why a singleton class instead of a flat dict**: All settings are co-located in one inspectable object. Properties like `postgres_dsn` and `ACTIVE_MODELS` compute themselves from the raw values, eliminating scattered string formatting across multiple files.

---

### 5.2 `main.py`

**What it does**: The interactive chat entry point. Starts a REPL loop where the user converses with the memory-augmented agent.

**Notable features**:
- Accepts `--model`, `--user_id`, `--threshold`, `--index` flags at startup
- Mid-session model switching with `switch <model_key>` command
- `decay` command triggers memory pruning
- `compare` command launches the model comparison benchmark inline

**Code snippet — the REPL loop core**:
```python
while True:
    user_input = Prompt.ask("[bold green]You[/bold green]")
    if user_input.lower() == "quit":
        break
    result = run_turn(
        graph, user_input, user_id, session_id,
        threshold=threshold,
        model_key=model_key,
    )
    ai_msg = result["messages"][-1]
    console.print(Panel(ai_msg.content, title="Assistant"))
```

---

### 5.3 `graph/state.py`

**What it does**: Defines `AgentState` — the TypedDict that flows through all four LangGraph nodes. Every piece of data produced by one node and needed by another lives here.

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # full chat history — add_messages 
                                              # ensures messages are APPENDED, not replaced
    user_id: str
    session_id: str
    retrieved_memories: list[dict]           # Milvus results for this turn
    memory_fetch_ms: float                   # retrieval latency measurement
    index_strategy: str                      # "FLAT" | "PARTITIONED"
    last_memory_write: dict                  # gatekeeper/conflict output
    llm_generation_ms: float                 # LLM call latency
    run_id: str                              # experiment batch ID
    threshold_used: float                    # override for experiments
    model_key: str                           # which LLM to use this turn
    next_action: str                         # routing hook (unused in linear mode)
```

The `add_messages` annotation is critical — LangGraph uses it to merge incoming message lists rather than replace the prior list, preserving the full conversation history.

---

### 5.4 `graph/nodes.py`

**What it does**: Implements the four executable nodes of the graph. This is the heart of the runtime pipeline.

#### `_get_llm(model_key)` — Provider Factory
The most important helper. Takes a model key string (e.g., `"gemini-3-flash"`) and returns the appropriate LangChain chat model, instantiated lazily on each call:

```python
if provider == "openai":
    return ChatOpenAI(model=model_id, api_key=..., temperature=temperature)
elif provider == "anthropic":
    return ChatAnthropic(model=model_id, api_key=..., temperature=temperature)
elif provider == "gemini":
    return ChatGoogleGenerativeAI(model=model_id, google_api_key=...)
elif provider == "groq":
    return ChatGroq(model=model_id, api_key=...)
```

OpenAI `o3`/`o4-mini` reasoning models require `temperature=1` (locked by the API) — handled via a per-model `temperature` override in `config.py`.

#### `retrieve_memory(state)` — Node 1
Extracts the last user message, calls `query_similar_memories()` in Milvus, and returns the list of relevant memories plus fetch latency. Nothing is written here.

#### `llm_respond(state)` — Node 2
Builds the memory-augmented system prompt and invokes the LLM:

```python
SYSTEM_TEMPLATE = """You are a personalized AI assistant with long-term memory.

The following memories about this user are relevant to the current conversation.
Use them to personalise your response. Do NOT hallucinate facts not in these memories.

--- MEMORIES ---
{memory_block}
--- END MEMORIES ---

Current date: {date}
"""
```

The `memory_block` is a formatted list of all retrieved memories with their type and score. If no memories are found, it explicitly tells the LLM so — preventing hallucination of non-existent context.

#### `conflict_write(state)` — Node 3
Finds the last human message and calls `resolve_and_store()` to run the full gatekeeper + conflict detection cycle before writing the new memory.

#### `log_metrics(state)` — Node 4
Writes a structured row to `experiment_metrics` in PostgreSQL — capturing fetch latency, LLM latency, total latency, query text, and model key for every single turn. This is what powers the research experiments.

---

### 5.5 `graph/builder.py`

**What it does**: Assembles the LangGraph `StateGraph`, registers the four nodes, wires them in linear order, attaches the `PostgresSaver` checkpointer, and compiles it.

```python
graph.add_edge(START, "retrieve_memory")
graph.add_edge("retrieve_memory", "llm_respond")
graph.add_edge("llm_respond", "conflict_write")
graph.add_edge("conflict_write", "log_metrics")
graph.add_edge("log_metrics", END)
```

The `PostgresSaver` stores the full LangGraph checkpoint (all state including message history) in dedicated tables it manages itself (`langgraph_checkpoints`, `langgraph_writes`). This is what makes conversations persistent across process restarts.

The `run_turn()` helper wraps `graph.invoke()`, constructing the initial `AgentState` with sensible defaults and the `thread_id` config needed by the checkpointer to resume a specific conversation.

---

### 5.6 `memory/gatekeeper.py`

**What it does**: The decision filter that determines whether a user's message contains information worth storing, classifies it, and persists it to both stores.

**The classification prompt** asks the LLM three things and gets back structured JSON:

```
1. should_save (bool)     — is this worth storing?
2. memory_type (str)      — FACTUAL | PREFERENCE | EPHEMERAL
3. importance (float 0–1) — how significant is this for future turns?
4. distilled_content (str)— clean third-person memory sentence
```

**Example transformation**:
```
User message: "btw I switched to vim last week and I love it"
↓ gatekeeper
{
  "should_save": true,
  "memory_type": "PREFERENCE",
  "importance": 0.75,
  "distilled_content": "User recently switched to Vim as their primary editor and enjoys it."
}
```

The LLM runs at `temperature=0` with `json_object` response format enforced — eliminating malformed JSON. The `_compute_expiry()` function sets the `expires_at` timestamp based on memory type: FACTUAL gets 365 days (treated as never-expiring), PREFERENCE gets 90 days, EPHEMERAL gets 1 day.

---

### 5.7 `memory/conflict.py`

**What it does**: The most sophisticated module. Before storing a new memory, it checks if any existing memory is semantically similar enough to conflict or be superseded.

**Pipeline**:
1. Run the new distilled memory through `classify_memory()` (gatekeeper)
2. Query Milvus for the top-5 most similar existing memories at `threshold=0.82` (strict)
3. Build a `pinecone_id → postgres_id` map by querying PostgreSQL
4. Ask the LLM to decide: `UPDATE` | `APPEND` | `IGNORE`
5. If `UPDATE`: hard-delete the old memory from both Milvus and PostgreSQL, then write the new one
6. If `APPEND`: write the new memory alongside existing ones
7. If `IGNORE`: skip — it is a duplicate
8. Log the decision to `conflict_log` with a `consistency_hit` flag

**The conflict detection prompt** gives the LLM the new memory alongside the top-5 most similar existing memories (enriched with their Postgres IDs) and asks for a resolution with reasoning:

```json
{
  "resolution": "UPDATE",
  "target_id": 42,
  "target_pinecone_id": "abc-123",
  "reasoning": "New preference (Vim) supersedes old preference (VS Code).",
  "consistency_hit": true
}
```

**The critical threshold**: `CONFLICT_THRESHOLD = 0.82` — deliberately strict. The conflict module only surfaces candidates it is highly confident are related. The research (§3.0) reveals this is actually too strict for belief-change detection, and 0.35 is the recommended fix.

---

### 5.8 `memory/decay.py`

**What it does**: Implements the forgetting mechanism — two strategies for pruning stale memories.

#### TTL Pruning
Hard SQL delete:
```sql
DELETE FROM memory_store
WHERE user_id = %s
  AND expires_at IS NOT NULL
  AND expires_at < NOW()
RETURNING pinecone_id
```
Returns the deleted Milvus IDs so they can be bulk-deleted from the vector store too.

#### Semantic Pruning
An LLM reviews the 50 most recent memories and identifies redundant or superseded ones, returning a list of IDs to delete. This catches cases TTL misses — e.g., an EPHEMERAL memory that was re-saved before expiry, or two FACTUAL memories that contradict each other.

**When to run**: Designed as a nightly cron job (`python -m memory.decay --user_id=<uid>`). In the interactive chat, it is triggered manually with the `decay` command.

---

### 5.9 `db/postgres_setup.py`

**What it does**: Creates the database, all tables, and all indexes. Run once at setup.

**Three custom tables**:

| Table | Purpose |
|---|---|
| `memory_store` | One row per stored memory — content, type, importance, TTL, access tracking |
| `conflict_log` | Every memory write decision — resolution, consistency hit flag, old/new content |
| `experiment_metrics` | Every conversation turn — latencies, model key, threshold, retrieval count |

Plus the LangGraph checkpointer manages its own tables (`langgraph_checkpoints`, `langgraph_writes`) automatically.

The `get_connection()` function returns a plain `psycopg2` connection and is used throughout the codebase. The LangGraph `PostgresSaver` uses a separate `psycopg3` connection (different driver, required by LangGraph).

---

### 5.10 `db/vector_store.py`

**What it does**: All Milvus operations — connect, create collection, embed text, upsert, query, and delete.

**Collection schema**:
```
id           VARCHAR(64)   — UUID, primary key
user_id      VARCHAR(128)  — user isolation
memory_type  VARCHAR(32)   — FACTUAL | PREFERENCE | EPHEMERAL
content      VARCHAR(1024) — the memory text
importance   FLOAT         — 0.0–1.0 score
embedding    FLOAT_VECTOR  — 1536-dim OpenAI embedding
```

**Index**: `IVF_FLAT` with COSINE metric, `nlist=128`, `nprobe=16`. IVF_FLAT is the simplest Milvus index — not approximate like HNSW but exact within each bucket, suitable for research where reproducibility matters more than speed at massive scale.

**Two index strategies**:

| Strategy | How it works | When to use |
|---|---|---|
| `FLAT` | All users in the default partition. Filter by `user_id` at query time using an expression. | Simpler, works for prototyping |
| `PARTITIONED` | One Milvus partition per user (`u_<user_id>`). Search is scoped to that partition. | Faster for large user pools, better isolation |

**COSINE conversion**: Milvus returns `distance` for COSINE which is `1 - similarity`. The code converts: `score = 1.0 - hit.distance` so a score of 1.0 means identical and 0.0 means orthogonal.

**`embed_text()`** — the embedding function. Uses `text-embedding-3-large` with MRL (Matryoshka Representation Learning) truncation to 1536 dimensions. The comment in the code explicitly documents the free alternative:
```python
# FREE ALTERNATIVE: Use sentence-transformers locally
# from sentence_transformers import SentenceTransformer
# _st_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
```

---

### 5.11 `experiments/data_loader.py`

**What it does**: Loads four public research datasets into the system for experiments. Also contains 40 synthetic PersonaChat-style persona profiles as a built-in fallback.

**The storage map** the file documents itself explicitly:
```
PersonaChat → memory_store (FACTUAL/PREFERENCE) + Milvus PARTITIONED → §2.1 + §3.0
DailyDialog → memory_store (EPHEMERAL, short TTL) + Milvus FLAT+PARTITIONED → §2.2
MSC         → memory_store (session-1) + conflict_log (session-2 deltas) → §3.0
LoCoMo      → memory_store + locomo_probes.csv (ground-truth QA pairs) → §2.1
```

**Key transformation** — PersonaChat facts are in first person; the loader converts them to third person:
```python
"I prefer dark mode in all interfaces."
→ "User prefers dark mode in all interfaces."
```

Rule-based `FACTUAL` vs `PREFERENCE` classification avoids an LLM call for every record, keeping bulk loading fast and deterministic.

---

### 5.12 `experiments/benchmarks.py`

**What it does**: The original Phase-2 experiment runner for the three infrastructure benchmarks. Contains 10 canonical `SEED_MEMORIES` used across all experiments and three experiment functions.

**Experiment 1** (`threshold_accuracy_experiment`): Sweeps five thresholds [0.5, 0.65, 0.75, 0.80, 0.90], measures accuracy on 5 test queries.

**Experiment 2** (`volume_impact_experiment`): Creates three test users (0, 50, 500 memories), queries all of them, compares retrieval count, latency, and estimates "Lost in the Middle" risk.

**Experiment 3** (`latency_analysis_experiment`): Same user, same query, 10+ runs each for FLAT and PARTITIONED strategies. Measures average, min, and max fetch time.

This file was the research foundation; `run_research.py` is its more rigorous successor.

---

### 5.13 `experiments/model_comparison.py`

**What it does**: The cross-provider LLM benchmark. Seeds 8 specific memories for a test user, then runs 5 recall probe questions against each model N times.

**The 8 seeded memories**:
```
User's primary programming language is Python.
User prefers dark mode in all tools.
User is building a LangGraph-based memory agent.
User works remotely from Bangalore, India.
User dislikes verbose documentation.
User's favourite framework for ML is PyTorch.
User prefers concise, bullet-point answers.
User has 8 years of software engineering experience.
```

**The 5 recall probes** (query + expected keyword):
```
"What programming language do I mainly use?"  → expected: "python"
"What city am I working from?"                → expected: "bangalore"
"How many years of experience do I have?"     → expected: "8"
"What is my current project about?"           → expected: "langgraph"
"Which ML framework do I prefer?"             → expected: "pytorch"
```

**Recall check**: Simple case-insensitive substring match — `expected_fact.lower() in response_text.lower()`. This is intentionally lenient: the model just needs to mention the right word anywhere in its response.

**Gemini 3 fix** (added this session): `response.content` for Gemini 3+ returns a list of content parts, not a string. The code handles both:
```python
raw = response.content
response_text = (
    "".join(p if isinstance(p, str) else p.get("text", "") for p in raw)
    if isinstance(raw, list)
    else str(raw)
)
```

---

### 5.14 `experiments/run_research.py`

**What it does**: The master experiment runner. Re-implements all five experiments with full statistical rigor — Cohen's d, p-values, F1, p95 latency — and produces publication-quality CSV + Rich terminal tables.

Run any individual section or all at once:
```bash
python -m experiments.run_research              # all five sections
python -m experiments.run_research --section 2.1
python -m experiments.run_research --section 4.0 --model_runs 2
```

It is the only file that should be used to regenerate research results. `benchmarks.py` is the older draft.

---

## 6. Complete Data Flow — Turn by Turn

Here is exactly what happens when a user sends "What IDE should I use for Python?":

```
Step 1: RETRIEVE MEMORY (Node 1)
─────────────────────────────────
• Extract query: "What IDE should I use for Python?"
• Embed it via OpenAI text-embedding-3-large → 1536-dim float vector
• Search Milvus collection 'agent_memory':
    - PARTITIONED mode: scope to partition u_<user_id>
    - IVF_FLAT COSINE search, nprobe=16, top_k=10
    - Filter: score >= MEMORY_SIMILARITY_THRESHOLD (default 0.75)
• Returns e.g.:
    [
      {"content": "User's primary IDE is VS Code.", "score": 0.82, "type": "FACTUAL"},
      {"content": "User prefers dark mode in all tools.", "score": 0.76, "type": "PREFERENCE"},
    ]
• Writes: retrieved_memories, memory_fetch_ms → AgentState

Step 2: LLM RESPOND (Node 2)
──────────────────────────────
• Build system prompt:
    "You are a personalized AI assistant...
     MEMORIES:
     • [FACTUAL] (score=0.82) User's primary IDE is VS Code.
     • [PREFERENCE] (score=0.76) User prefers dark mode in all tools.
     Current date: 2026-04-02"
• Call LLM (e.g. gemini-3-flash): [system_msg] + [human: "What IDE should I use for Python?"]
• Returns: "Based on your history, you already use VS Code which is excellent for Python..."
• Writes: messages (AI response appended), llm_generation_ms → AgentState

Step 3: CONFLICT WRITE (Node 3)
─────────────────────────────────
• Find last human message: "What IDE should I use for Python?"
• Call gatekeeper → classify_memory():
    LLM decides: {"should_save": false, "reason": "one-off question, no personal info"}
• No memory written.
• (If user had said "I just switched to PyCharm", gatekeeper would save it,
   then conflict detection would find the VS Code memory and decide UPDATE vs APPEND)

Step 4: LOG METRICS (Node 4)
──────────────────────────────
• Insert row into experiment_metrics:
    run_id, user_id, query, threshold=0.75,
    memories_retrieved=2, fetch_ms=312.4, llm_ms=2765.3,
    total_ms=3077.7, index_type="PARTITIONED", model_key="gemini-3-flash"
```

---

## 7. The Data — What, Why, and How Practical

### 7.1 PersonaChat

**Source**: Facebook AI Research, 2018. Available on HuggingFace as `AlekseyKorshuk/persona-chat`.

**What it is**: 164,356 turns of crowdsourced dialogue between pairs of humans, each assigned a 4-5 item "persona" (e.g., "I am a nurse. I love hiking. I play guitar."). These personas are ground-truth facts about a simulated person.

**Format in this project**:
- Each persona fact → one `memory_store` row (FACTUAL or PREFERENCE)
- Each persona → one virtual `user_id`
- Facts converted from first-person ("I am a nurse") to third-person ("User is a nurse") via rule-based replacement

**Why selected**:
- The only widely-used benchmark of *real human-written personal facts* suitable for memory seeding
- Facts are diverse, cover realistic life domains (job, hobbies, food preferences, location)
- First-party ground truth: you know exactly what you stored → you know exactly what should be retrieved
- Used in **§2.1** (threshold recall accuracy): seed persona facts → probe with paraphrased queries → measure if the right memory is recalled

**How practical**: Very practical. PersonaChat facts read like real things people say in conversation. The limitation is that they feel slightly generic/crowdsourced — highly specific professional details are underrepresented.

---

### 7.2 DailyDialog

**Source**: Li et al., 2017. Available on HuggingFace as `daily_dialog`.

**What it is**: 13,118 human dialogues covering everyday life topics (relationships, work, hobbies, current events). Unlike PersonaChat, these are not persona-driven — they are open-ended conversations.

**Format in this project**:
- Selected utterances that contain personal statements (classifiable as EPHEMERAL)
- Set `expires_at = NOW() + 1 day` to model temporary context
- Loaded into the FLAT and PARTITIONED collections for volume testing

**Why selected**:
- Provides realistic *ephemeral* content (today's mood, current task, immediate plans)
- Volume testing needed large corpora — DailyDialog's 13k dialogues provide enough content to fill the 50-memory and 500-memory conditions
- Tests the TTL decay module: EPHEMERAL memories expire in 1 day, DailyDialog content is ideal for this since it represents temporary context

**How practical**: Moderate. The content is natural but not specific to technical users. The volume testing experiment cared about *quantity* more than *semantic richness*, making DailyDialog appropriate.

---

### 7.3 MSC (Multi-Session Chat)

**Source**: Xu et al., Meta AI, 2021. Available on HuggingFace as `multi_session_chat`.

**What it is**: Dialogues between the same two people across multiple sessions separated by time, where participants' preferences and life facts evolve between sessions (e.g., "I used to like Python, now I prefer Rust").

**Format in this project**:
- **Session 1** facts → `memory_store` (the "old belief")
- **Session 2** introduced contradictions → `conflict_log` — the system must decide UPDATE vs APPEND vs IGNORE
- The `consistency_hit` column records whether the agent correctly identified the intent change

**Why selected**: MSC is the *only* publicly available benchmark that tests *temporal belief evolution* — exactly what the conflict resolution module handles. It provides:
- Real examples of people changing their mind across time
- Verified pairs of old belief + new belief
- Multiple session structure that mirrors the agent's use case

**How practical**: Very high research value. MSC is directly modelling the hardest real-world scenario: users who change jobs, change tools, change locations — exactly the cases where a naive "always append" strategy fails.

**Caveat**: MSC is occasionally unavailable on HuggingFace or has access restrictions. The `data_loader.py` has a synthetic fallback that generates 20 controlled belief-change scenarios used as the §3.0 test set.

---

### 7.4 LoCoMo

**Source**: Maharana et al., 2023. Long-Context Conversation Modelling. Available on HuggingFace.

**What it is**: 30 extended conversation histories (hundreds of turns each) with accompanying ground-truth QA pairs — questions about facts mentioned in the conversation and their correct answers.

**Format in this project**:
- Conversation turns → `memory_store` + Milvus (the agent's memory bank)
- QA pairs → `locomo_probes.csv` (25 rows: `user_id, question, expected_answer`)
- At experiment time: issue each question against the user's memory bank → check if the correct answer appears in retrieved memories

**Why selected**:
- Only benchmark that provides **end-to-end ground truth** for memory retrieval: you know what was stored (the conversation) and you know what the right answer is (the QA pairs)
- Long contexts (hundreds of turns) stress-test the retrieval system in a way short dialogs cannot
- 30 distinct "users" provides statistical significance for threshold sweep experiments
- QA evaluation is binary and deterministic — no subjective judgment needed

**Sample probes** (from `locomo_probes.csv`):
```
user_id,question,expected_answer
locomo_user_0,"What programming language does Eric prefer?","Python"
locomo_user_1,"Where does the user work?","San Francisco"
locomo_user_2,"What is the user's favorite hobby?","hiking"
```

**How practical**: Extremely practical for research — the closest thing to a real long-running personal assistant scenario. The limitation is that LoCoMo conversations were written for the benchmark, not scraped from actual assistant logs, so real-world deployment may surface distribution shift.

---

### 7.5 Synthetic Seed Data

Both `benchmarks.py` and `model_comparison.py` use hand-crafted synthetic seed data for controlled experiments:

**`benchmarks.py` — 10 canonical memories**:
```python
SEED_MEMORIES = [
    ("User is a senior Python developer with 8 years of experience.", "FACTUAL", 0.9),
    ("User prefers dark mode in all their tools.", "PREFERENCE", 0.7),
    ("User is currently learning Rust.", "FACTUAL", 0.8),
    ("User works remotely from Bangalore, India.", "FACTUAL", 0.85),
    ("User's current project involves LangGraph.", "FACTUAL", 0.95),
    # ...
]
```

**`model_comparison.py` — 8 recall memories + 5 probe questions**:
Hand-designed so each probe maps to exactly one memory, with a single-word expected answer (`"python"`, `"bangalore"`, `"8"`, `"langgraph"`, `"pytorch"`). This makes recall evaluation fully deterministic and human-auditable.

**Why synthetic data was used for §4.0**: Cross-provider model comparison requires a *controlled* test. If memories were drawn from a real dataset, variance in embedding quality or content format could confound the model comparison signal. Synthetic data with known probes isolates the variable of interest: which LLM best uses injected memory context.

---

### 7.6 Is the Data Real-Time?

**No — the data is not real-time.** All four source datasets are static offline corpora.

| Dataset | Recency | Last Updated |
|---|---|---|
| PersonaChat | Static research dataset | 2018 |
| DailyDialog | Static research dataset | 2017 |
| MSC | Static research dataset | 2021 |
| LoCoMo | Static research dataset | 2023 |

**What IS real-time** in this system:
- The LLM API calls (all models are the current live versions, evaluated April 2026)
- The memory writing pipeline (when you use `main.py` interactively, everything you say is stored live)
- The model IDs and prices in `config.py` (verified against live APIs April 2026)
- The Gemini 3 model family (first benchmark of `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-3.1-pro-preview` family, added this week)

**Production scenario** (if deployed as a real product): Every user interaction would generate real-time memories. The datasets are only used for experiments — in production, the gatekeeper processes real user messages with no dataset dependency at all.

---

## 8. Embedding Design Decisions

| Decision | Choice | Alternative | Reason |
|---|---|---|---|
| Embedding model | `text-embedding-3-large` (OpenAI) | `all-MiniLM-L6-v2` (local) | Better recall on diverse personal facts; MRL allows dimension reduction |
| Dimensions | 1536 | 384 or 3072 | MRL sweet spot — large-model quality at manageable storage cost |
| Provider for embeddings | OpenAI only | Per-provider embeddings | Consistent vector space across all LLMs; enables fair comparison |
| Similarity metric | COSINE | Euclidean, Dot product | COSINE is scale-invariant — best for variable-length text embeddings |
| Index type | IVF_FLAT | HNSW | Exact recall within buckets; reproducible for research |
| Batch rate limit | 20 records then 1s sleep | No limit | Stays within OpenAI Tier 1 embedding RPM limits during bulk load |

A built-in free alternative is documented in `vector_store.py` comments: swap `embed_text()` to use `sentence-transformers/all-MiniLM-L6-v2` locally (384-dim, no API key needed). Update `EMBED_DIM=384` in `config.py` and rebuild the collection.

---

## 9. PostgreSQL Schema — Every Table

### `memory_store` — The Core

```sql
CREATE TABLE memory_store (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT NOT NULL,               -- identifies which user this memory belongs to
    session_id    TEXT,                         -- which chat session created it
    memory_type   TEXT NOT NULL,               -- FACTUAL | PREFERENCE | EPHEMERAL
    content       TEXT NOT NULL,               -- the distilled memory sentence
    pinecone_id   TEXT UNIQUE,                 -- the Milvus vector record ID (column kept as "pinecone_id" for legacy reasons)
    importance    FLOAT DEFAULT 0.5,           -- LLM-assigned 0–1 score
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    expires_at    TIMESTAMPTZ,                 -- NULL = never expires (FACTUAL)
    last_accessed TIMESTAMPTZ DEFAULT NOW(),   -- updated on each retrieval hit
    access_count  INT DEFAULT 0               -- how many times this memory was retrieved
);
```

### `conflict_log` — Every Write Decision

```sql
CREATE TABLE conflict_log (
    id             SERIAL PRIMARY KEY,
    user_id        TEXT NOT NULL,
    old_memory_id  INT REFERENCES memory_store(id) ON DELETE SET NULL,
    new_content    TEXT,                        -- the incoming memory candidate
    resolution     TEXT,                        -- UPDATE | APPEND | IGNORE
    consistency_hit BOOLEAN,                   -- did the agent correctly identify intent change?
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
```

### `experiment_metrics` — Every Turn's Performance

```sql
CREATE TABLE experiment_metrics (
    id                 SERIAL PRIMARY KEY,
    run_id             TEXT,                    -- batch identifier for experiment grouping
    user_id            TEXT,
    query              TEXT,                    -- the user's question (first 500 chars)
    threshold_used     FLOAT,                   -- what cosine threshold was active
    memories_retrieved INT,                     -- how many memories were returned
    memory_fetch_ms    FLOAT,                   -- Milvus query latency
    llm_generation_ms  FLOAT,                   -- LLM API call latency
    total_latency_ms   FLOAT,                   -- sum of both
    hallucination_flag BOOLEAN,                 -- manual flag (unused in automated runs)
    instruction_follow BOOLEAN,                 -- manual flag (unused in automated runs)
    index_type         TEXT,                    -- FLAT | PARTITIONED
    model_key          TEXT,                    -- e.g. "gemini-3-flash"
    created_at         TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 10. Key Design Decisions and Trade-offs

### Decision 1: Milvus over Pinecone
**Chosen**: Milvus (Docker, local, free)  
**Rejected**: Pinecone (cloud, $70+/month for production tier)  
**Trade-off**: Milvus requires Docker setup and manual management vs Pinecone's managed service. For research purposes, local = reproducible = better. Note: `conflict.py` and comments still reference "Pinecone" in places — legacy from the initial design before the switch.

### Decision 2: Dual-store architecture (Milvus + PostgreSQL)
**Why both?**  
- Milvus: semantic similarity search (the "what is related?" question)  
- PostgreSQL: structured queries — sort by recency, filter by type, expiry-based deletion, conflict logging  
Neither database alone can do both jobs efficiently.

### Decision 3: LangGraph over raw Python
**Why LangGraph?**  
- Built-in graph checkpointing (conversation persistence) out of the box  
- Clean separation of concerns — each node does one thing  
- Easy to add conditional edges for future routing (e.g., "if no memories found, trigger a broader search")  
**Trade-off**: Adds dependency and some boilerplate vs a plain Python class.

### Decision 4: Provider-agnostic embedding
All providers (OpenAI, Anthropic, Gemini, Groq) use OpenAI embeddings for memory storage and retrieval. This means:  
- All memories live in the same vector space regardless of which LLM was used to write them  
- Fair comparison across providers in §4.0 (no embedding advantage)  
- Single API key required for the embedding side regardless of which LLM you pick

### Decision 5: temperature=0 for gatekeeper and conflict modules
Deterministic outputs are critical when classifying what to save and how to resolve conflicts. Non-zero temperature would cause the same message to be classified differently across runs, making experiments non-reproducible.

---

## 11. LLM Provider Abstraction

The `_get_llm()` factory function in `graph/nodes.py` provides total LLM portability. Adding a new provider requires exactly two steps:

1. Add an entry to `MODELS` in `config.py`:
```python
"my-new-model": {
    "provider": "newprovider",
    "model_id": "actual-api-model-id",
    "cost_per_1k_in": 0.001,
    "cost_per_1k_out": 0.004,
}
```

2. Add a branch in `_get_llm()`:
```python
elif provider == "newprovider":
    from langchain_newprovider import ChatNewProvider
    return ChatNewProvider(model=model_id, api_key=settings.NEW_API_KEY)
```

That is the only change required. Every experiment, every benchmark, and the interactive chat immediately support the new model.

---

## 12. Experiment Results at a Glance

All results are in `experiments/results/`. Full analysis in `RESEARCH_ANALYSIS.md`.

### §2.1 — Threshold vs Recall (25 LoCoMo probes)
Best recall: **88%** at threshold=0.40. Optimal balance: **0.50** (76% recall, manageable noise). Cliff point at 0.60 (drops to 40%). Zero recall above 0.90.

### §2.2 — Volume Impact
50 memories → relevance score 0.838. 500 memories → 0.709 (−15% quality at same retrieval count). Both sit at ~1,600 estimated context tokens — MEDIUM or HIGH LitM risk.

### §2.3 — FLAT vs PARTITIONED Latency (30 runs each)
PARTITIONED: 332ms mean, 510ms p95. FLAT: 360ms mean, 707ms p95. PARTITIONED is 1.08× faster overall, **40% better worst-case**.

### §3.0 — Conflict Detection Accuracy (20 scenarios)
Overall: **25%** accuracy. UPDATE scenarios: **0%** (all 10 failed — cosine similarity too low to surface opposing beliefs). APPEND: **100%** (trivial case). IGNORE: **0%** (paraphrases not recognized as duplicates).

### §4.0 — 12-Model Benchmark
| Winner | Why |
|---|---|
| Highest recall | `gpt-5` (70%) — but 12s avg, $0.01/1k |
| Best speed/recall | `gemini-3-flash` (60%, 2765ms, $0.00030/1k) |
| Best budget | `gemini-2.5-flash-lite` (50%, 899ms, $0.00010/1k) |
| Worst value | `gpt-5.4` (20% recall, $0.015/1k — worst model in the test) |
| Universal failure | "city" and "experience" probes failed for all 12 models — retrieval threshold issue, not model quality |
