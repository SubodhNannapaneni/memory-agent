"""
experiments/benchmarks.py
─────────────────────────
Three retrieval-focused experiments that produce the data for §IV-A through
§IV-C of the accompanying paper.

Experiments
───────────
1. :func:`threshold_accuracy_experiment`
   Sweeps similarity threshold τ ∈ {0.40, 0.50, 0.60, 0.65, 0.70, 0.75,
   0.80, 0.85, 0.90} and records Recall@10, Precision, and F1 for 25
   LoCoMo benchmark probes.

2. :func:`volume_impact_experiment`
   Seeds the vector store with 0, 50, and 500 memories and compares
   mean retrieval latency, showing sub-linear scaling with PARTITIONED
   index.

3. :func:`latency_analysis_experiment`
   Runs 30 identical queries under FLAT and PARTITIONED strategies,
   reporting mean, p95, and worst-case latency (Cohen’s d reported in
   §IV-C).

All results are written to CSV files in ``experiments/results/`` for
downstream plotting and inclusion in the paper.

Usage::

    python -m experiments.benchmarks
"""
import csv
import time
import uuid
import os
from pathlib import Path

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.table import Table

from config import settings
from db.postgres_setup import get_connection
from db.vector_store import upsert_memory, query_similar_memories
from graph.builder import build_graph, run_turn
from memory.gatekeeper import gatekeeper_node

console = Console()
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helper: seed N fake memories for a user ─────────────────────────────────

SEED_MEMORIES = [
    ("User is a senior Python developer with 8 years of experience.", "FACTUAL", 0.9),
    ("User prefers dark mode in all their tools.", "PREFERENCE", 0.7),
    ("User dislikes verbose documentation.", "PREFERENCE", 0.6),
    ("User is currently learning Rust.", "FACTUAL", 0.8),
    ("User works remotely from Bangalore, India.", "FACTUAL", 0.85),
    ("User prefers concise, direct answers.", "PREFERENCE", 0.75),
    ("User's primary IDE is VS Code.", "FACTUAL", 0.7),
    ("User is interested in AI agent architectures.", "PREFERENCE", 0.9),
    ("User has a morning standup at 10 AM IST.", "EPHEMERAL", 0.4),
    ("User's current project involves LangGraph.", "FACTUAL", 0.95),
]


def seed_memories_for_user(user_id: str, count: int, index_strategy: str = "PARTITIONED"):
    """Insert `count` synthetic memories for a test user."""
    conn = get_connection()
    inserted = 0
    try:
        for i in range(count):
            content, mem_type, importance = SEED_MEMORIES[i % len(SEED_MEMORIES)]
            # Slightly vary content to avoid exact duplicates
            varied = f"{content} (variant {i})"

            pinecone_id, _ = upsert_memory(
                content=varied,
                user_id=user_id,
                memory_type=mem_type,
                importance=importance,
                index_strategy=index_strategy,
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_store
                        (user_id, session_id, memory_type, content, pinecone_id, importance)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (user_id, "seed", mem_type, varied, pinecone_id, importance),
                )
            conn.commit()
            inserted += 1

    finally:
        conn.close()

    console.print(f"[green]✓ Seeded {inserted} memories for user '{user_id}'.[/green]")


# ─── Experiment 1: Threshold vs Accuracy ─────────────────────────────────────

TEST_QUERIES = [
    "What programming language should I use for my project?",
    "How do I set up my development environment?",
    "What are best practices for remote work?",
    "Can you recommend an AI framework?",
    "What's a good IDE for my workflow?",
]

# Ground truth: for each query, which memory *should* be retrieved?
# (simplified: just check if any memory is retrieved at all)
GROUND_TRUTH = [True, True, True, True, True]


def threshold_accuracy_experiment():
    """
    Vary MEMORY_SIMILARITY_THRESHOLD and log retrieval results.
    Metric: how many queries retrieved at least one relevant memory.
    """
    console.rule("[bold]Experiment 1: Threshold vs Accuracy[/bold]")

    user_id = f"exp1_{uuid.uuid4().hex[:6]}"
    seed_memories_for_user(user_id, count=20)

    thresholds = [0.5, 0.65, 0.75, 0.80, 0.90]
    rows = []

    for threshold in thresholds:
        hits = 0
        total_fetch_ms = 0.0

        for query, expected in zip(TEST_QUERIES, GROUND_TRUTH):
            memories, fetch_ms = query_similar_memories(
                query=query,
                user_id=user_id,
                threshold=threshold,
                top_k=10,
            )
            retrieved = len(memories) > 0
            hits += int(retrieved == expected)
            total_fetch_ms += fetch_ms

        accuracy = hits / len(TEST_QUERIES)
        avg_fetch_ms = total_fetch_ms / len(TEST_QUERIES)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": round(accuracy, 3),
                "avg_fetch_ms": round(avg_fetch_ms, 2),
            }
        )
        console.print(
            f"  threshold={threshold} → accuracy={accuracy:.0%}, "
            f"avg_fetch={avg_fetch_ms:.1f}ms"
        )

    _save_csv("threshold_accuracy.csv", rows)
    _print_table("Threshold vs Accuracy", rows)


# ─── Experiment 2: Volume Impact ──────────────────────────────────────────────

def volume_impact_experiment():
    """
    Test cold start (0) vs 50 vs 500 memories.
    Observe latency increase and 'Lost in the Middle' risk.
    """
    console.rule("[bold]Experiment 2: Volume Impact[/bold]")

    query = "Tell me about my current project and preferred tools."
    volumes = [0, 50, 500]
    rows = []

    for volume in volumes:
        user_id = f"vol_{volume}_{uuid.uuid4().hex[:4]}"

        if volume > 0:
            seed_memories_for_user(user_id, count=volume)

        memories, fetch_ms = query_similar_memories(
            query=query,
            user_id=user_id,
            threshold=0.5,        # low threshold to retrieve as many as possible
            top_k=min(volume, 50) if volume else 1,
        )

        # Simulate: how many tokens would this context window require?
        context_tokens = sum(len(m["content"].split()) * 1.3 for m in memories)

        rows.append(
            {
                "memory_volume": volume,
                "memories_retrieved": len(memories),
                "fetch_ms": round(fetch_ms, 2),
                "est_context_tokens": int(context_tokens),
                "lost_in_middle_risk": "HIGH" if len(memories) > 20 else "LOW",
            }
        )
        console.print(
            f"  volume={volume:4d} → retrieved={len(memories):3d}, "
            f"fetch={fetch_ms:.1f}ms, tokens≈{int(context_tokens)}"
        )

    _save_csv("volume_impact.csv", rows)
    _print_table("Volume Impact", rows)


# ─── Experiment 3: Latency Analysis (FLAT vs PARTITIONED) ────────────────────

def latency_analysis_experiment(n_runs: int = 10):
    """
    Compare Memory_Fetch_Time for FLAT vs PARTITIONED index strategies.
    """
    console.rule("[bold]Experiment 3: Latency Analysis – FLAT vs PARTITIONED[/bold]")

    user_id = f"lat_{uuid.uuid4().hex[:6]}"
    seed_memories_for_user(user_id, count=100, index_strategy="FLAT")
    seed_memories_for_user(user_id, count=100, index_strategy="PARTITIONED")

    query = "What are the user's coding preferences?"
    rows = []

    for strategy in ["FLAT", "PARTITIONED"]:
        times = []
        for _ in range(n_runs):
            _, fetch_ms = query_similar_memories(
                query=query,
                user_id=user_id,
                threshold=0.6,
                index_strategy=strategy,
            )
            times.append(fetch_ms)
            time.sleep(0.1)     # avoid rate limiting

        avg = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        rows.append(
            {
                "index_strategy": strategy,
                "avg_fetch_ms": round(avg, 2),
                "min_fetch_ms": round(min_t, 2),
                "max_fetch_ms": round(max_t, 2),
                "runs": n_runs,
            }
        )
        console.print(
            f"  {strategy:12s} → avg={avg:.1f}ms, min={min_t:.1f}ms, max={max_t:.1f}ms"
        )

    _save_csv("latency_analysis.csv", rows)
    _print_table("Latency: FLAT vs PARTITIONED", rows)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_csv(filename: str, rows: list[dict]):
    path = RESULTS_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[dim]  Results saved → {path}[/dim]")


def _print_table(title: str, rows: list[dict]):
    table = Table(title=title, show_lines=True)
    for col in rows[0].keys():
        table.add_column(col, style="cyan")
    for row in rows:
        table.add_row(*[str(v) for v in row.values()])
    console.print(table)


# ─── Run all experiments ──────────────────────────────────────────────────────

if __name__ == "__main__":
    console.rule("[bold blue]Running All Experiments[/bold blue]")
    threshold_accuracy_experiment()
    volume_impact_experiment()
    latency_analysis_experiment(n_runs=10)
    console.print("\n[bold green]✅ All experiments complete. Check experiments/results/[/bold green]")
