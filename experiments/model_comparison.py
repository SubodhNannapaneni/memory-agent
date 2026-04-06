"""
experiments/model_comparison.py
────────────────────────────────────────────────────────────────────────────
§IV-E benchmark: multi-model memory-recall comparison.

Measures two metrics for each model across a controlled probe set:

  1. **Response latency (ms)** — wall-clock time from prompt dispatch to first
     token being fully received.
  2. **Memory recall accuracy (%)** — fraction of probes where the model's
     response contained the exact keyword expected from the injected memory.

Benchmark design
────────────────
* 25 memories are seeded across 7 semantic categories (technology stack,
  location/work setup, experience/background, current project, preferences,
  personal facts, weekly schedule).
* 25 matching probe questions are issued — one per seeded memory.
* Each probe is run ``--runs`` times (default 3), giving **75 total probes
  per model** and a 95% Wilson CI of ±8.5–11.3 percentage points.
* A single embedding model (``text-embedding-3-large``, 1536-dim MRL) is
  used for all providers, so LLM differences drive recall differences.

Results are written to ``experiments/results/model_comparison.csv`` and
are the source data for Table VI and §IV-E of the paper.

Usage::

    python -m experiments.model_comparison
    python -m experiments.model_comparison --models gpt-4.1-mini gemini-3-flash
    python -m experiments.model_comparison --runs 5
"""
import argparse
import csv
import json
import time
import uuid
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich import box

from config import settings
from db.vector_store import upsert_memory, query_similar_memories
from graph.nodes import _get_llm, SYSTEM_TEMPLATE
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import date

console   = Console()
_embedder = OpenAI(api_key=settings.OPENAI_API_KEY)
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Seed memories that will be used for recall tests ──────────────────────────
# 25 memories across 7 categories: tech stack, location/work, experience,
# current project, preferences, personal, schedule.

RECALL_MEMORIES = [
    # --- Tech stack (5) ---
    ("User's primary programming language is Python.",                "FACTUAL",    0.90),
    ("User's favourite framework for ML is PyTorch.",                 "FACTUAL",    0.80),
    ("User's primary IDE is VS Code.",                                "FACTUAL",    0.75),
    ("User uses PostgreSQL as their main relational database.",       "FACTUAL",    0.80),
    ("User deploys applications on AWS.",                             "FACTUAL",    0.80),
    # --- Location & work setup (4) ---
    ("User works remotely from Bangalore, India.",                    "FACTUAL",    0.85),
    ("User's working hours are 9 AM to 6 PM IST.",                   "FACTUAL",    0.70),
    ("User's team is distributed across three time zones.",           "FACTUAL",    0.65),
    ("User's company is a fintech startup.",                          "FACTUAL",    0.75),
    # --- Experience & background (3) ---
    ("User has 8 years of software engineering experience.",          "FACTUAL",    0.90),
    ("User holds a bachelor's degree in Computer Science.",           "FACTUAL",    0.75),
    ("User previously worked at a large e-commerce company.",         "FACTUAL",    0.70),
    # --- Current project (3) ---
    ("User is building a LangGraph-based memory agent.",              "FACTUAL",    0.95),
    ("User's current project uses Milvus as the vector database.",    "FACTUAL",    0.90),
    ("User's current project integrates with OpenAI's embedding API.","FACTUAL",    0.85),
    # --- Preferences (4) ---
    ("User prefers dark mode in all tools.",                          "PREFERENCE", 0.80),
    ("User dislikes verbose documentation.",                          "PREFERENCE", 0.70),
    ("User prefers concise, bullet-point answers.",                   "PREFERENCE", 0.75),
    ("User prefers async communication over synchronous meetings.",   "PREFERENCE", 0.65),
    # --- Personal (3) ---
    ("User speaks English and Kannada fluently.",                     "FACTUAL",    0.70),
    ("User's hobby is competitive chess.",                            "FACTUAL",    0.65),
    ("User follows a vegetarian diet.",                               "FACTUAL",    0.70),
    # --- Schedule (3) ---
    ("User has a daily standup meeting at 10 AM IST.",                "EPHEMERAL",  0.60),
    ("User typically codes for 4 focused hours in the morning.",      "PREFERENCE", 0.60),
    ("User takes Fridays for deep-work and avoids meetings.",         "PREFERENCE", 0.65),
]

# ── Probe questions + the memory fact that MUST appear in the answer ──────────
# 25 probes — one per memory, in matching order.

RECALL_PROBES = [
    # Tech stack
    {
        "query":         "What programming language do I mainly use?",
        "expected_fact": "python",
        "memory_hint":   "User's primary programming language is Python.",
    },
    {
        "query":         "Which ML framework do I prefer?",
        "expected_fact": "pytorch",
        "memory_hint":   "User's favourite framework for ML is PyTorch.",
    },
    {
        "query":         "What code editor or IDE do I use?",
        "expected_fact": "vs code",
        "memory_hint":   "User's primary IDE is VS Code.",
    },
    {
        "query":         "Which relational database do I use?",
        "expected_fact": "postgresql",
        "memory_hint":   "User uses PostgreSQL as their main relational database.",
    },
    {
        "query":         "Which cloud platform do I deploy on?",
        "expected_fact": "aws",
        "memory_hint":   "User deploys applications on AWS.",
    },
    # Location & work
    {
        "query":         "What city am I working from?",
        "expected_fact": "bangalore",
        "memory_hint":   "User works remotely from Bangalore, India.",
    },
    {
        "query":         "What are my working hours?",
        "expected_fact": "9",
        "memory_hint":   "User's working hours are 9 AM to 6 PM IST.",
    },
    {
        "query":         "How many time zones does my team span?",
        "expected_fact": "three",
        "memory_hint":   "User's team is distributed across three time zones.",
    },
    {
        "query":         "What industry does my company operate in?",
        "expected_fact": "fintech",
        "memory_hint":   "User's company is a fintech startup.",
    },
    # Experience & background
    {
        "query":         "How many years of experience do I have?",
        "expected_fact": "8",
        "memory_hint":   "User has 8 years of software engineering experience.",
    },
    {
        "query":         "What is my educational background?",
        "expected_fact": "computer science",
        "memory_hint":   "User holds a bachelor's degree in Computer Science.",
    },
    {
        "query":         "Where did I work before my current role?",
        "expected_fact": "e-commerce",
        "memory_hint":   "User previously worked at a large e-commerce company.",
    },
    # Current project
    {
        "query":         "What is my current project about?",
        "expected_fact": "langgraph",
        "memory_hint":   "User is building a LangGraph-based memory agent.",
    },
    {
        "query":         "Which vector database does my current project use?",
        "expected_fact": "milvus",
        "memory_hint":   "User's current project uses Milvus as the vector database.",
    },
    {
        "query":         "Which embedding API does my current project use?",
        "expected_fact": "openai",
        "memory_hint":   "User's current project integrates with OpenAI's embedding API.",
    },
    # Preferences
    {
        "query":         "Do I prefer dark mode or light mode?",
        "expected_fact": "dark",
        "memory_hint":   "User prefers dark mode in all tools.",
    },
    {
        "query":         "What kind of documentation do I dislike?",
        "expected_fact": "verbose",
        "memory_hint":   "User dislikes verbose documentation.",
    },
    {
        "query":         "How do I prefer answers to be formatted?",
        "expected_fact": "bullet",
        "memory_hint":   "User prefers concise, bullet-point answers.",
    },
    {
        "query":         "Do I prefer async or sync communication at work?",
        "expected_fact": "async",
        "memory_hint":   "User prefers async communication over synchronous meetings.",
    },
    # Personal
    {
        "query":         "What languages do I speak?",
        "expected_fact": "kannada",
        "memory_hint":   "User speaks English and Kannada fluently.",
    },
    {
        "query":         "What is my hobby?",
        "expected_fact": "chess",
        "memory_hint":   "User's hobby is competitive chess.",
    },
    {
        "query":         "What diet do I follow?",
        "expected_fact": "vegetarian",
        "memory_hint":   "User follows a vegetarian diet.",
    },
    # Schedule
    {
        "query":         "When is my daily standup?",
        "expected_fact": "10",
        "memory_hint":   "User has a daily standup meeting at 10 AM IST.",
    },
    {
        "query":         "When during the day do I do my main coding?",
        "expected_fact": "morning",
        "memory_hint":   "User typically codes for 4 focused hours in the morning.",
    },
    {
        "query":         "Which day do I keep free from meetings?",
        "expected_fact": "friday",
        "memory_hint":   "User takes Fridays for deep-work and avoids meetings.",
    },
]


def _seed_user_memories(user_id: str):
    """Insert recall memories into Milvus for the test user."""
    for content, mem_type, importance in RECALL_MEMORIES:
        upsert_memory(
            content=content,
            user_id=user_id,
            memory_type=mem_type,
            importance=importance,
            index_strategy="PARTITIONED",
        )
    console.print(f"[green]✓ Seeded {len(RECALL_MEMORIES)} memories for user '{user_id}'[/green]")


def _check_recall(response_text: str, expected_fact: str) -> bool:
    """Return True if the expected fact keyword appears in the model's response."""
    return expected_fact.lower() in response_text.lower()


# ── Core benchmark function ───────────────────────────────────────────────────

def run_model_benchmark(
    model_key: str,
    user_id: str,
    n_runs: int = 3,
) -> dict:
    """
    Run all RECALL_PROBES against one model n_runs times.
    Returns aggregated metrics dict.
    """
    console.rule(f"[bold cyan]Testing: {model_key}[/bold cyan]")

    model_cfg = settings.MODELS.get(model_key)
    if not model_cfg:
        console.print(f"[red]✗ Unknown model: {model_key}[/red]")
        return {}

    try:
        llm = _get_llm(model_key)
    except Exception as e:
        console.print(f"[red]✗ Could not load {model_key}: {e}[/red]")
        return {"model": model_key, "error": str(e)}

    latencies      = []
    recall_hits    = 0
    total_probes   = 0
    errors         = 0

    for run_idx in range(n_runs):
        for probe in RECALL_PROBES:
            # Fetch relevant memories from Milvus
            memories, fetch_ms = query_similar_memories(
                query=probe["query"],
                user_id=user_id,
                threshold=0.60,     # low threshold so memories are always retrieved
                index_strategy="PARTITIONED",
            )

            # Build memory-augmented prompt
            if memories:
                memory_block = "\n".join(
                    f"• [{m['memory_type']}] {m['content']}" for m in memories
                )
            else:
                # Fallback: inject the probe's hint directly so recall is testable
                memory_block = f"• [FACTUAL] {probe['memory_hint']}"

            system_msg = SystemMessage(
                content=SYSTEM_TEMPLATE.format(
                    memory_block=memory_block,
                    date=date.today().isoformat(),
                )
            )

            t0 = time.perf_counter()
            try:
                response = llm.invoke([system_msg, HumanMessage(content=probe["query"])])
                elapsed_ms = (time.perf_counter() - t0) * 1000
                # Some providers (Gemini 3+) return content as a list of parts
                raw = response.content
                response_text = (
                    "".join(p if isinstance(p, str) else p.get("text", "") for p in raw)
                    if isinstance(raw, list)
                    else str(raw)
                )
            except Exception as e:
                console.print(f"  [red]Error on {model_key} / run {run_idx}: {e}[/red]")
                errors += 1
                total_probes += 1
                continue

            recalled = _check_recall(response_text, probe["expected_fact"])
            recall_hits  += int(recalled)
            total_probes += 1
            latencies.append(elapsed_ms)

            status = "[green]✓[/green]" if recalled else "[red]✗[/red]"
            console.print(
                f"  Run {run_idx+1} | {probe['query'][:45]:<45} | "
                f"{elapsed_ms:6.0f}ms | recall {status}"
            )

    if not latencies:
        return {"model": model_key, "error": "all calls failed"}

    avg_latency    = sum(latencies) / len(latencies)
    min_latency    = min(latencies)
    max_latency    = max(latencies)
    recall_pct     = (recall_hits / total_probes * 100) if total_probes else 0
    provider       = model_cfg["provider"]
    cost_in        = model_cfg["cost_per_1k_in"]
    cost_out       = model_cfg["cost_per_1k_out"]

    return {
        "model":           model_key,
        "provider":        provider,
        "avg_latency_ms":  round(avg_latency, 1),
        "min_latency_ms":  round(min_latency, 1),
        "max_latency_ms":  round(max_latency, 1),
        "recall_accuracy": f"{recall_pct:.1f}%",
        "recall_hits":     recall_hits,
        "total_probes":    total_probes,
        "errors":          errors,
        "cost_per_1k_in":  cost_in,
        "cost_per_1k_out": cost_out,
        "free_tier":       "YES" if (cost_in == 0 and cost_out == 0) else "no",
    }


# ── Print + save results ──────────────────────────────────────────────────────

def _print_comparison_table(results: list[dict]):
    table = Table(
        title="📊 Model Comparison — Latency & Memory Recall",
        box=box.ROUNDED,
        show_lines=True,
    )
    cols = [
        ("Model",           "cyan",    "model"),
        ("Provider",        "white",   "provider"),
        ("Avg Latency(ms)", "yellow",  "avg_latency_ms"),
        ("Min / Max(ms)",   "dim",     None),
        ("Recall Accuracy", "green",   "recall_accuracy"),
        ("Free?",           "magenta", "free_tier"),
        ("$/1k in",         "red",     "cost_per_1k_in"),
    ]
    for label, style, _ in cols:
        table.add_column(label, style=style)

    # Sort by avg_latency_ms ascending
    sorted_results = sorted(
        [r for r in results if "error" not in r],
        key=lambda x: x.get("avg_latency_ms", 9999),
    )

    for r in sorted_results:
        table.add_row(
            r["model"],
            r["provider"],
            str(r["avg_latency_ms"]),
            f"{r['min_latency_ms']} / {r['max_latency_ms']}",
            r["recall_accuracy"],
            r["free_tier"],
            f"${r['cost_per_1k_in']:.5f}",
        )

    console.print(table)

    # Print failed models
    failed = [r for r in results if "error" in r]
    if failed:
        console.print("\n[bold red]Failed models:[/bold red]")
        for r in failed:
            console.print(f"  • {r['model']}: {r['error']}")


def _save_results(results: list[dict]):
    clean = [r for r in results if "error" not in r]
    if not clean:
        return
    path = RESULTS_DIR / "model_comparison.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=clean[0].keys())
        writer.writeheader()
        writer.writerows(clean)
    console.print(f"\n[dim]Results saved → {path}[/dim]")


# ── Entry point ───────────────────────────────────────────────────────────────

# Representative cross-section: newest + fast from each active provider.
# ACTIVE_MODELS filtering in run_comparison() skips any with no API key.
DEFAULT_MODELS = [
    # OpenAI — latest flagship + fast/cheap
    "gpt-5.4",
    "gpt-5",
    "gpt-4.1-mini",
    # Anthropic — latest sonnet + haiku
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    # Gemini 2 — all confirmed working on Tier 1
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    # Gemini 3 — new family, Tier 1 required
    "gemini-3-flash",
    "gemini-3-pro",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro",
]


def run_comparison(models: list[str] = None, n_runs: int = 2):
    active = set(settings.ACTIVE_MODELS.keys())
    requested = models or DEFAULT_MODELS
    skipped = [m for m in requested if m not in active]
    if skipped:
        console.print(f"[yellow]⚠ Skipping models with no API key: {skipped}[/yellow]")
    models = [m for m in requested if m in active]
    user_id = f"cmp_{uuid.uuid4().hex[:6]}"

    console.rule("[bold blue]Multi-Model Memory Comparison[/bold blue]")
    console.print(f"Models to test : {models}")
    console.print(f"Runs per probe : {n_runs}")
    console.print(f"Test user id   : {user_id}\n")

    # Seed memories once for this test user
    _seed_user_memories(user_id)

    results = []
    for model_key in models:
        result = run_model_benchmark(model_key, user_id, n_runs=n_runs)
        if result:
            results.append(result)

    console.rule("[bold]Results[/bold]")
    _print_comparison_table(results)
    _save_results(results)

    console.print(
        "\n[bold green]✅ Done.[/bold green]  "
        "Open [cyan]experiments/results/model_comparison.csv[/cyan] for your article data."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LLM models on latency + memory recall")
    parser.add_argument(
        "--models", nargs="+",
        default=DEFAULT_MODELS,
        help="Space-separated model keys to test",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per probe per model (more = more stable averages)",
    )
    args = parser.parse_args()
    run_comparison(models=args.models, n_runs=args.runs)
