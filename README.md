# LangGraph Memory Agent

> **Research implementation** for the paper:  
> *"Persistent Memory for Conversational AI Agents: Architecture, Benchmarks, and Operational Thresholds"*  
> Subodh Kumar Nannapaneni — April 2026  
> 📄 **[Read the paper (IEEE_PAPER.md)](IEEE_PAPER.md)** · 🔗 Preprint link TBD

A production-grade, open-source memory layer for LLM-based conversational agents.
Stores, retrieves, and manages long-term user memories using **PostgreSQL** (metadata)
and **Milvus** (vector similarity search) — fully local, no cloud vector DB required.
Supports **12 LLMs** across four providers with a single hot-swappable backend.

---

## What This Implements

| Component | Description |
|---|---|
| **Gatekeeper** | LLM-based classifier decides what to store (FACTUAL / PREFERENCE / EPHEMERAL) |
| **Vector retrieval** | Milvus IVF_FLAT with COSINE similarity, PARTITIONED index strategy |
| **Conflict resolution** | UPDATE / APPEND / IGNORE pipeline before every write |
| **Memory decay** | TTL pruning + LLM-based semantic deduplication |
| **Checkpointing** | Full conversation history via LangGraph PostgresSaver |
| **Benchmarks** | Threshold sweep, volume scaling, FLAT vs PARTITIONED latency, 12-model recall |

### Key findings from the paper

- Cosine similarity threshold τ is the **dominant system variable** — moving τ from 0.75 → 0.50 increases Recall@10 by 56 pp (more than the range across all 12 tested LLMs)
- **PARTITIONED** index reduces p95 tail latency by 28% and worst-case latency by 40% vs FLAT
- Across 75 probes per model (n=75, 95% CI ±11 pp): GPT-5 leads recall at 49.3%; GPT-4.1-mini and Gemini-3-Flash tie at 44.0% — statistically indistinguishable from GPT-5
- **Claude-Sonnet-4-6 is the weakest model for memory tasks** (17.3% recall), despite appearing competitive in small-sample pilots

---

## Project Structure

```
memory-agent/
├── config.py                    # All settings: models, Milvus, thresholds
├── main.py                      # Interactive chat UI (model-switchable)
├── requirements.txt
├── .env.example  →  copy to .env
│
├── db/
│   ├── postgres_setup.py        # DB schema + migrations (run once)
│   └── vector_store.py          # Milvus CRUD — embed, upsert, search, delete
│
├── memory/
│   ├── gatekeeper.py            # Classify & store incoming user messages
│   ├── decay.py                 # TTL pruning + semantic deduplication
│   └── conflict.py              # UPDATE / APPEND / IGNORE write pipeline
│
├── graph/
│   ├── state.py                 # AgentState TypedDict
│   ├── nodes.py                 # 4 pipeline nodes (retrieve → respond → write → log)
│   └── builder.py               # Compile graph with PostgresSaver checkpointer
│
├── experiments/
│   ├── benchmarks.py            # §IV-A–C: threshold, volume, latency experiments
│   ├── model_comparison.py      # §IV-E: 12-model recall benchmark (n=75 probes)
│   └── results/                 # CSV output used in the paper
│
└── IEEE_PAPER.md                # Full research paper
```

---

## Quick Start

### 1. Start Docker services

```bash
# PostgreSQL
docker run --name pg-memory \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 -d postgres

# Milvus (fully local — no cloud account needed)
docker run --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest standalone
```

### 2. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env — only add keys for providers you want to use:
#   OPENAI_API_KEY     → https://platform.openai.com/api-keys
#   ANTHROPIC_API_KEY  → https://console.anthropic.com
#   GOOGLE_API_KEY     → https://aistudio.google.com/app/apikey  (free tier available)
#   GROQ_API_KEY       → https://console.groq.com/keys           (free)
```

### 4. Create database tables (once)

```bash
python -m db.postgres_setup
```

### 5. Start chatting

```bash
python main.py                              # uses gpt-4.1-mini by default
python main.py --model gpt-5
python main.py --model gemini-3-flash
python main.py --model llama3-70b          # free via Groq
```

### 6. Run the multi-model benchmark (reproduces §IV-E)

```bash
python -m experiments.model_comparison
# or test a subset:
python -m experiments.model_comparison --models gpt-4.1-mini gemini-3-flash claude-haiku-4-5 --runs 3
```

### 7. Run retrieval benchmarks (reproduces §IV-A through §IV-C)

```bash
python -m experiments.benchmarks
```

---

## Available Models (verified April 2026)

| Key | Provider | Avg Recall (n=75) | Avg Latency | Cost /1k in | Notes |
|---|---|---|---|---|---|
| `gpt-5` | OpenAI | **49.3%** | 13,632 ms | $0.01000 | Highest recall; high latency |
| `gpt-4.1-mini` | OpenAI | **44.0%** | 1,515 ms | $0.00040 | Best latency/recall ratio |
| `gemini-3-flash` | Google | **44.0%** | 3,910 ms | $0.00030 | Tied with gpt-4.1-mini |
| `gemini-2.5-pro` | Google | 40.0% | 9,094 ms | $0.00125 | |
| `gemini-2.5-flash-lite` | Google | 40.0% | 1,021 ms | $0.00010 | Lowest cost |
| `gemini-3-pro` | Google | 38.7% | 7,146 ms | $0.00200 | |
| `gemini-3.1-flash-lite` | Google | 37.3% | 1,818 ms | $0.00010 | Lowest cost |
| `gemini-3.1-pro` | Google | 33.3% | 7,595 ms | $0.00200 | |
| `gpt-5.4` | OpenAI | 25.3% | 2,324 ms | $0.01500 | Worst cost/recall value |
| `claude-haiku-4-5` | Anthropic | 25.3% | 1,840 ms | $0.00080 | |
| `gemini-2.5-flash` | Google | 25.3% | 2,589 ms | $0.00030 | |
| `claude-sonnet-4-6` | Anthropic | **17.3%** | 3,257 ms | $0.00300 | Lowest recall |
| `llama3-70b` | Groq | — | — | free | Groq free tier |
| `mixtral-8x7b` | Groq | — | — | free | Groq free tier |
| `o3`, `o4-mini` | OpenAI | — | — | varies | Reasoning models |

> Recall numbers are from the §IV-E benchmark (75 probes/model, 95% CI ±11 pp).
> Models without a recall figure were not included in the §IV-E run.

---

## In-Session Commands

| Command | What it does |
|---|---|
| `switch <model-key>` | Hot-swap the LLM without restarting |
| `decay` | Run TTL + semantic pruning for the current user |
| `compare` | Run the full multi-model benchmark inline |
| `quit` / `exit` | End the session |

---

## Embedding Model

All experiments use **OpenAI `text-embedding-3-large`** truncated to **1536 dimensions**
via Matryoshka Representation Learning (MRL).  The native model outputs 3072 dimensions;
50% truncation halves Milvus storage and ANN-search cost while retaining ~96% of
full-dimensional retrieval quality.

To switch to a free local embedding model, replace `embed_text()` in
`db/vector_store.py` with a SentenceTransformer call and update `EMBED_DIM=384`
in `config.py`.

---

## Recommended Architecture (from paper §V-D)

| Use case | Recommended model | Threshold τ |
|---|---|---|
| Real-time chat | `gpt-4.1-mini` | 0.50 |
| Cost-critical / high volume | `gemini-2.5-flash-lite` | 0.50 |
| Offline / batch processing | `gpt-5` | 0.50 |
| Avoid for memory tasks | `claude-sonnet-4-6` | — |

> τ = 0.50 is the empirically optimal threshold from §IV-A.
> The system default (0.75) reduces Recall@10 from 76% to 20%.

---

## Citation

If you use this codebase or the benchmark data in your research, please cite:

```
Subodh Kumar Nannapaneni (2026). Persistent Memory for Conversational AI Agents:
Architecture, Benchmarks, and Operational Thresholds.
GitHub: https://github.com/SubodhNannapaneni/memory-agent
```

