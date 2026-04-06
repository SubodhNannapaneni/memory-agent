"""
experiments/run_research.py
────────────────────────────
Master research experiment runner — all three phases from Objective.txt.

Produces publication-ready CSV + Rich tables for every experiment.

§ 2.1  Threshold vs Recall Accuracy   (locomo_probes.csv × 9 thresholds)
§ 2.2  Volume Impact / Lost-in-Middle (cold / vol_50 / vol_500)
§ 2.3  Latency: FLAT vs PARTITIONED   (30-run statistical benchmark)
§ 3.0  Conflict Detection Consistency (20 controlled belief-update scenarios)
§ 4.0  Multi-Model Benchmark          (cross-provider recall accuracy)

Usage:
    python -m experiments.run_research              # all sections
    python -m experiments.run_research --section 2.1
    python -m experiments.run_research --section 2.2
    python -m experiments.run_research --section 2.3
    python -m experiments.run_research --section 3.0
    python -m experiments.run_research --section 4.0
    python -m experiments.run_research --section 2.3 --runs 50
"""
import argparse
import csv
import json
import math
import statistics
import time
import uuid
from datetime import date
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                            TextColumn, TimeElapsedColumn)
from rich.table import Table

from config import settings
from db.postgres_setup import get_connection
from db.vector_store import query_similar_memories, upsert_memory

console     = Console()
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── CSV helper ───────────────────────────────────────────────────────────────

def _save_csv(filename: str, rows: list[dict]):
    if not rows:
        return
    path = RESULTS_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"  [dim]→ saved: {path}[/dim]")


def _print_table(title: str, rows: list[dict], highlight_col: str = None):
    if not rows:
        return
    table = Table(title=title, box=box.ROUNDED, show_lines=True, title_style="bold cyan")
    for col in rows[0].keys():
        style = "bold green" if col == highlight_col else "white"
        table.add_column(str(col), style=style)
    for row in rows:
        table.add_row(*[str(v) for v in row.values()])
    console.print(table)


# ═════════════════════════════════════════════════════════════════════════════
# § 2.1  THRESHOLD vs RECALL ACCURACY
# ─────────────────────────────────────────────────────────────────────────────
# Dataset : locomo_probes.csv (25 ground-truth QA pairs, 5 locomo_user_*)
# Vary    : similarity threshold from 0.40 → 0.90
# Metrics : Recall@10, Precision, F1, avg memories retrieved, avg fetch ms
#
# Research insight:
#   Low threshold  → high recall but noisy context (hallucination risk)
#   High threshold → low recall, too strict (misses useful memories)
#   Optimal sweet-spot is typically 0.65–0.75 for diverse conversational data
# ═════════════════════════════════════════════════════════════════════════════

THRESHOLDS = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def section_2_1() -> list[dict]:
    console.rule("[bold cyan]§ 2.1  Threshold vs Recall Accuracy[/bold cyan]")

    probe_csv = RESULTS_DIR / "locomo_probes.csv"
    if not probe_csv.exists():
        console.print("[red]✗ locomo_probes.csv not found. Run: python -m experiments.data_loader[/red]")
        return []

    probes = []
    with open(probe_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            probes.append(row)

    console.print(f"  Probes loaded: {len(probes)} QA pairs across "
                  f"{len(set(p['user_id'] for p in probes))} users")

    rows = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Threshold sweep…", total=len(THRESHOLDS))

        for threshold in THRESHOLDS:
            hits = 0
            precision_sum = 0.0
            fetch_times   = []
            ret_counts    = []

            for probe in probes:
                user_id  = probe["user_id"]
                question = probe["question"]
                expected = probe["expected_answer"].lower()
                # keywords: words longer than 3 chars are meaningful signals
                keywords = [w for w in expected.split() if len(w) > 3]

                memories, fetch_ms = query_similar_memories(
                    query=question,
                    user_id=user_id,
                    threshold=threshold,
                    top_k=10,
                    index_strategy="PARTITIONED",
                )
                fetch_times.append(fetch_ms)
                ret_counts.append(len(memories))

                # Recall: does any retrieved memory contain the expected answer?
                recalled = any(
                    any(kw in m["content"].lower() for kw in keywords)
                    for m in memories
                ) if memories and keywords else False
                hits += int(recalled)

                # Precision: fraction of retrieved memories that are relevant
                if memories and keywords:
                    rel = sum(
                        1 for m in memories
                        if any(kw in m["content"].lower() for kw in keywords)
                    )
                    precision_sum += rel / len(memories)
                # if no memories retrieved, precision contribution is 0

            n       = len(probes)
            recall  = hits / n
            prec    = precision_sum / n
            f1      = (2 * recall * prec / (recall + prec)) if (recall + prec) > 0 else 0.0
            avg_ret = statistics.mean(ret_counts)
            avg_ms  = statistics.mean(fetch_times)

            row = {
                "threshold":     threshold,
                "recall_at_10":  round(recall, 3),
                "precision":     round(prec, 3),
                "f1_score":      round(f1, 3),
                "avg_retrieved": round(avg_ret, 1),
                "avg_fetch_ms":  round(avg_ms, 1),
                "n_probes":      n,
            }
            rows.append(row)
            console.print(
                f"  t={threshold:.2f} | recall={recall:.0%}  "
                f"prec={prec:.0%}  F1={f1:.3f} | "
                f"retrieved={avg_ret:.1f}  fetch={avg_ms:.0f}ms"
            )
            progress.advance(task)

    _save_csv("research_2_1_threshold_recall.csv", rows)
    _print_table("§ 2.1  Threshold vs Recall Accuracy", rows, highlight_col="f1_score")
    best = max(rows, key=lambda r: r["f1_score"])
    console.print(f"  [bold green]↑ Optimal threshold = {best['threshold']}  "
                  f"(F1={best['f1_score']}, recall={best['recall_at_10']:.0%})[/bold green]")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# § 2.2  VOLUME IMPACT — "Lost in the Middle"
# ─────────────────────────────────────────────────────────────────────────────
# Three pre-built users: cold_start (0), vol_50 (50 memories), vol_500 (500)
# Run 5 probe queries per condition at threshold 0.60 (permissive)
#
# Research insight:
#   cold_start → no context, agent must hallucinate or admit ignorance
#   vol_50     → ideal range: enough signal, manageable context
#   vol_500    → retrieval quality degrades; top-k window polluted with near-misses
#   → "Lost in the Middle" syndrome quantified via avg_relevance_score decay
# ═════════════════════════════════════════════════════════════════════════════

VOLUME_QUERIES = [
    "What are my communication preferences and personal style?",
    "Describe my professional background and technical skills.",
    "What hobbies or personal interests do I have?",
    "What are my daily routines and work habits?",
    "What topics do I frequently talk about or care about?",
]


def section_2_2() -> list[dict]:
    console.rule("[bold cyan]§ 2.2  Volume Impact — 'Lost in the Middle'[/bold cyan]")

    conditions = [
        {"label": "cold_start", "user_id": f"cold_{uuid.uuid4().hex[:6]}", "n_total": 0},
        {"label": "vol_50",     "user_id": "vol_50",                       "n_total": 50},
        {"label": "vol_500",    "user_id": "vol_500",                      "n_total": 500},
    ]

    rows = []
    for cond in conditions:
        user_id = cond["user_id"]
        fetch_times, ret_counts, scores = [], [], []

        for query in VOLUME_QUERIES:
            memories, fetch_ms = query_similar_memories(
                query=query,
                user_id=user_id,
                threshold=0.60,
                top_k=50,
                index_strategy="PARTITIONED",
            )
            fetch_times.append(fetch_ms)
            ret_counts.append(len(memories))
            if memories:
                scores.append(statistics.mean(m["score"] for m in memories))

        avg_fetch = statistics.mean(fetch_times)
        avg_ret   = statistics.mean(ret_counts)
        avg_score = statistics.mean(scores) if scores else 0.0
        # Estimated context tokens (avg ~25 words/memory × 1.3 token/word)
        est_tokens = int(avg_ret * 25 * 1.3)
        # Lost-in-the-Middle risk heuristic
        if avg_ret > 20 and avg_score < 0.72:
            litm_risk = "HIGH"
        elif avg_ret > 10 or (avg_ret > 5 and avg_score < 0.70):
            litm_risk = "MEDIUM"
        else:
            litm_risk = "LOW"

        row = {
            "condition":          cond["label"],
            "total_memories":     cond["n_total"],
            "avg_retrieved":      round(avg_ret, 1),
            "avg_relevance_score":round(avg_score, 3),
            "avg_fetch_ms":       round(avg_fetch, 1),
            "est_context_tokens": est_tokens,
            "litm_risk":          litm_risk,
        }
        rows.append(row)
        console.print(
            f"  {cond['label']:<12} | retrieved={avg_ret:.1f}  "
            f"relevance={avg_score:.3f}  fetch={avg_fetch:.0f}ms  "
            f"tokens≈{est_tokens}  LitM={litm_risk}"
        )

    _save_csv("research_2_2_volume_impact.csv", rows)
    _print_table("§ 2.2  Volume Impact (LitM Risk)", rows, highlight_col="litm_risk")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# § 2.3  LATENCY — FLAT vs PARTITIONED INDEX
# ─────────────────────────────────────────────────────────────────────────────
# Same user (vol_500, 500 memories) queried two ways:
#   FLAT        → scans all ~2000+ vectors with user_id filter (no partition scope)
#   PARTITIONED → scans only vol_500's partition (~500 vectors)
# N_RUNS queries each; report mean ± std, p50, p95, min, max
#
# Research insight:
#   PARTITIONED reduces search scope by ~4–5x vs FLAT on this dataset.
#   At production scale (millions of memories), the gap widens dramatically.
#   The tradeoff is slightly more complex collection management.
# ═════════════════════════════════════════════════════════════════════════════

LATENCY_QUERIES = [
    "What are my communication preferences and personal style?",
    "Tell me about the user's technical background.",
    "What topics does this person care about?",
    "Describe the user's work style and habits.",
    "What does the user enjoy doing outside work?",
]


def section_2_3(n_runs: int = 30) -> list[dict]:
    console.rule("[bold cyan]§ 2.3  Latency: FLAT vs PARTITIONED Index[/bold cyan]")
    console.print(f"  User: vol_500 (500 memories)  |  Runs: {n_runs} per strategy")

    user_id = "vol_500"
    rows    = []

    for strategy in ["FLAT", "PARTITIONED"]:
        times = []
        query_cycle = LATENCY_QUERIES * (n_runs // len(LATENCY_QUERIES) + 1)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TextColumn("{task.completed}/{task.total}"),
                      TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task(
                f"  {strategy} ({n_runs} queries)…", total=n_runs)
            for i in range(n_runs):
                query = query_cycle[i % len(LATENCY_QUERIES)]
                _, fetch_ms = query_similar_memories(
                    query=query,
                    user_id=user_id,
                    threshold=0.60,
                    top_k=10,
                    index_strategy=strategy,
                )
                times.append(fetch_ms)
                time.sleep(0.04)   # avoid hammering Milvus
                progress.advance(task)

        s = sorted(times)
        mean = statistics.mean(times)
        std  = statistics.stdev(times) if len(times) > 1 else 0.0
        p50  = s[len(s) // 2]
        p95  = s[int(len(s) * 0.95)]
        row = {
            "strategy":   strategy,
            "mean_ms":    round(mean, 2),
            "std_ms":     round(std, 2),
            "p50_ms":     round(p50, 2),
            "p95_ms":     round(p95, 2),
            "min_ms":     round(min(times), 2),
            "max_ms":     round(max(times), 2),
            "n_runs":     n_runs,
        }
        rows.append(row)
        console.print(
            f"  {strategy:12s} | mean={mean:.1f}ms  std={std:.1f}  "
            f"p50={p50:.1f}  p95={p95:.1f}  min={min(times):.1f}  max={max(times):.1f}"
        )

    if len(rows) == 2:
        flat_mean = rows[0]["mean_ms"]
        part_mean = rows[1]["mean_ms"]
        speedup   = flat_mean / part_mean if part_mean > 0 else 1.0
        # Cohen's d for effect size
        pooled_std = math.sqrt((rows[0]["std_ms"]**2 + rows[1]["std_ms"]**2) / 2)
        cohens_d   = abs(flat_mean - part_mean) / pooled_std if pooled_std > 0 else 0.0
        console.print(
            f"\n  [bold green]PARTITIONED speedup: {speedup:.2f}×  "
            f"(Cohen's d = {cohens_d:.2f} — "
            f"{'large' if cohens_d > 0.8 else 'medium' if cohens_d > 0.5 else 'small'} effect)[/bold green]"
        )
        rows.append({
            "strategy":  "SPEEDUP",
            "mean_ms":   round(speedup, 2),
            "std_ms":    "—",
            "p50_ms":    "—",
            "p95_ms":    "—",
            "min_ms":    "—",
            "max_ms":    "—",
            "n_runs":    "cohens_d=" + str(round(cohens_d, 2)),
        })

    _save_csv("research_2_3_latency.csv", rows)
    _print_table("§ 2.3  FLAT vs PARTITIONED Latency (ms)", rows, highlight_col="p95_ms")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# § 3.0  CONFLICT DETECTION — CONSISTENCY SCORE
# ─────────────────────────────────────────────────────────────────────────────
# 20 controlled belief-evolution scenarios (10 UPDATE + 5 APPEND + 5 IGNORE).
# Each: load old belief → feed contradicting new belief → measure resolution.
#
# Consistency Score = correct_resolutions / total
# Breakdown by type: UPDATE accuracy, APPEND accuracy, IGNORE accuracy
#
# Research insight:
#   UPDATE accuracy shows ability to detect intent change (key for adaptive agents)
#   IGNORE accuracy shows deduplication capability
#   APPEND accuracy shows correct discrimination of genuinely new info
# ═════════════════════════════════════════════════════════════════════════════

# (old_belief, new_belief, expected_resolution, category)
CONFLICT_SCENARIOS = [
    # ── UPDATE scenarios (user's belief / preference changed) ──────────────
    ("User prefers Python as their primary programming language.",
     "User has switched to Rust as their primary programming language.",
     "UPDATE", "language_switch"),
    ("User prefers dark mode in all interfaces.",
     "User now prefers light mode after experiencing eye strain.",
     "UPDATE", "ui_preference"),
    ("User works from home in Bangalore, India.",
     "User has relocated to Amsterdam, Netherlands.",
     "UPDATE", "location_change"),
    ("User's primary IDE is VS Code.",
     "User switched to Neovim for all development work.",
     "UPDATE", "tooling_change"),
    ("User prefers concise, bullet-pointed summaries.",
     "User now prefers detailed, narrative-style explanations.",
     "UPDATE", "communication_style"),
    ("User is currently learning Rust.",
     "User has mastered Rust and is now focusing on Zig.",
     "UPDATE", "skills_progression"),
    ("User works at a fintech startup.",
     "User changed jobs and now works at a climate tech company.",
     "UPDATE", "career_change"),
    ("User's favorite ML framework is PyTorch.",
     "User has switched from PyTorch to JAX for all ML work.",
     "UPDATE", "framework_switch"),
    ("User dislikes meetings and prefers async communication.",
     "User now values face-to-face meetings for complex problems.",
     "UPDATE", "work_style"),
    ("User follows a vegetarian diet.",
     "User has adopted a fully vegan diet, cutting out all dairy.",
     "UPDATE", "lifestyle_change"),
    # ── APPEND scenarios (genuinely new, unrelated info) ───────────────────
    ("User has 8 years of software engineering experience.",
     "User recently obtained a machine learning certification from Coursera.",
     "APPEND", "new_credential"),
    ("User works remotely from Amsterdam.",
     "User volunteers at a local food bank every other Saturday.",
     "APPEND", "new_activity"),
    ("User's primary IDE is Neovim.",
     "User is writing a technical blog about software engineering.",
     "APPEND", "new_project"),
    ("User works at a climate tech company.",
     "User has two golden retriever dogs named Max and Bella.",
     "APPEND", "personal_fact"),
    ("User prefers JAX for ML work.",
     "User is learning Spanish for an upcoming trip to Barcelona.",
     "APPEND", "unrelated_new"),
    # ── IGNORE scenarios (semantic duplicate — should not add noise) ────────
    ("User prefers dark mode in all interfaces.",
     "User has dark mode enabled in all their development tools.",
     "IGNORE", "near_duplicate"),
    ("User is a software engineer with 8 years of experience.",
     "User is an experienced software developer with about 8 years under their belt.",
     "IGNORE", "paraphrase"),
    ("User works at a climate tech company.",
     "User's employer operates in the climate technology sector.",
     "IGNORE", "rephrasing"),
    ("User follows a vegan diet.",
     "User is vegan and does not consume any animal products.",
     "IGNORE", "expansion_of_same"),
    ("User uses Neovim as their primary code editor.",
     "User's preferred text editor for all coding is Neovim.",
     "IGNORE", "synonym_rewrite"),
]


def section_3_0() -> list[dict]:
    console.rule("[bold cyan]§ 3.0  Conflict Detection — Consistency Score[/bold cyan]")

    from memory.conflict import detect_conflict, _fetch_pg_id_map

    user_id = f"cexp_{uuid.uuid4().hex[:8]}"
    conn    = get_connection()
    detail  = []
    correct = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Running conflict scenarios…", total=len(CONFLICT_SCENARIOS))

        for old_belief, new_belief, expected, category in CONFLICT_SCENARIOS:
            # 1. Store the old belief
            milvus_id, _ = upsert_memory(
                content=old_belief,
                user_id=user_id,
                memory_type="FACTUAL",
                importance=0.80,
                index_strategy="PARTITIONED",
            )
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_store "
                    "(user_id, session_id, memory_type, content, pinecone_id, importance) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (user_id, "conflict_exp", "FACTUAL", old_belief, milvus_id, 0.80),
                )
            conn.commit()

            # 2. Query similar memories for the new belief
            similar, _ = query_similar_memories(
                query=new_belief,
                user_id=user_id,
                threshold=0.72,
                top_k=5,
                index_strategy="PARTITIONED",
            )

            # 3. Fetch PostgreSQL IDs
            pg_id_map = _fetch_pg_id_map([m["id"] for m in similar])

            # 4. Run LLM conflict detector
            result_obj  = detect_conflict(new_belief, similar, pg_id_map)
            actual      = result_obj.get("resolution", "APPEND")
            is_correct  = (actual == expected)
            if is_correct:
                correct += 1

            detail.append({
                "category":        category,
                "expected":        expected,
                "actual":          actual,
                "correct":         is_correct,
                "consistency_hit": result_obj.get("consistency_hit", False),
                "reasoning":       result_obj.get("reasoning", "")[:100],
                "old_belief":      old_belief[:60],
                "new_belief":      new_belief[:60],
            })

            icon = "[green]✓[/green]" if is_correct else "[red]✗[/red]"
            console.print(
                f"  {icon} [{category:<20}] expected={expected:<6}  got={actual:<6}"
            )
            time.sleep(0.4)   # rate-limit LLM calls
            progress.advance(task)

    conn.close()

    # ── Aggregate by resolution type ────────────────────────────────────
    by_type: dict[str, dict] = {
        "UPDATE": {"correct": 0, "total": 0},
        "APPEND": {"correct": 0, "total": 0},
        "IGNORE": {"correct": 0, "total": 0},
    }
    for r in detail:
        bt = by_type[r["expected"]]
        bt["total"] += 1
        if r["correct"]:
            bt["correct"] += 1

    total             = len(CONFLICT_SCENARIOS)
    consistency_score = correct / total

    console.print(
        f"\n  [bold]Overall Consistency Score: "
        f"[green]{consistency_score:.1%}[/green]  ({correct}/{total})[/bold]"
    )
    for rtype, c in by_type.items():
        pct = c["correct"] / c["total"] if c["total"] else 0
        bar = "█" * int(pct * 20)
        console.print(f"    {rtype:<6}: {pct:.0%}  {bar}  ({c['correct']}/{c['total']})")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    detail_path = RESULTS_DIR / "research_3_0_conflict_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail[0].keys())
        writer.writeheader()
        writer.writerows(detail)
    console.print(f"  [dim]→ detail log: {detail_path}[/dim]")

    summary_rows = [
        {
            "resolution_type": "OVERALL",
            "accuracy":        round(consistency_score, 3),
            "correct":         correct,
            "total":           total,
            "notes":           "all 20 scenarios",
        }
    ]
    for rtype, c in by_type.items():
        pct = c["correct"] / c["total"] if c["total"] else 0
        summary_rows.append({
            "resolution_type": rtype,
            "accuracy":        round(pct, 3),
            "correct":         c["correct"],
            "total":           c["total"],
            "notes":           {
                "UPDATE": "10 belief-change scenarios",
                "APPEND": "5 genuinely new-info scenarios",
                "IGNORE": "5 semantic-duplicate scenarios",
            }[rtype],
        })

    _save_csv("research_3_0_conflict_summary.csv", summary_rows)
    _print_table("§ 3.0  Conflict Detection Accuracy", summary_rows, highlight_col="accuracy")
    return summary_rows


# ═════════════════════════════════════════════════════════════════════════════
# § 4.0  MULTI-MODEL BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
# Delegates to experiments/model_comparison.py infrastructure.
# Runs cross-provider recall accuracy test (OpenAI, Anthropic, Gemini).
# ═════════════════════════════════════════════════════════════════════════════

def section_4_0(n_runs: int = 2):
    console.rule("[bold cyan]§ 4.0  Multi-Model Benchmark[/bold cyan]")
    from experiments.model_comparison import run_comparison
    run_comparison(n_runs=n_runs)


# ═════════════════════════════════════════════════════════════════════════════
# Final summary table (paper-ready)
# ═════════════════════════════════════════════════════════════════════════════

def _print_final_summary(results: dict):
    console.print("\n")
    console.rule("[bold green]Research Summary — Paper-Ready Metrics[/bold green]")

    table = Table(
        title=f"Memory-Augmented LLM Agent — Experimental Results  ({date.today()})",
        box=box.DOUBLE_EDGE, show_lines=True, title_style="bold",
    )
    table.add_column("Phase",    style="bold cyan",  width=7)
    table.add_column("Metric",   style="white",      width=38)
    table.add_column("Result",   style="bold green", width=16, justify="right")
    table.add_column("Research Interpretation", style="dim", width=55)

    # ── § 2.1 ─────────────────────────────────────────────────────────────
    r21 = results.get("2.1", [])
    if r21:
        best   = max(r21, key=lambda r: r["f1_score"])
        t_05   = next((r for r in r21 if r["threshold"] == 0.50), None)
        t_09   = next((r for r in r21 if r["threshold"] == 0.90), None)
        table.add_row(
            "§2.1", "Optimal threshold (max F1)",
            f"t={best['threshold']}  F1={best['f1_score']}",
            "Sweet-spot balancing recall and precision",
        )
        if t_05:
            table.add_row(
                "§2.1", f"Recall @ threshold 0.50",
                f"{t_05['recall_at_10']:.0%}  prec={t_05['precision']:.0%}",
                "High recall but noisy; hallucination risk ↑",
            )
        if t_09:
            table.add_row(
                "§2.1", f"Recall @ threshold 0.90",
                f"{t_09['recall_at_10']:.0%}  prec={t_09['precision']:.0%}",
                "Too strict; misses relevant memories",
            )
        table.add_row(
            "§2.1", "Avg fetch time (best threshold)",
            f"{best['avg_fetch_ms']:.0f} ms",
            "Retrieval overhead at peak quality",
        )

    # ── § 2.2 ─────────────────────────────────────────────────────────────
    r22 = results.get("2.2", [])
    if r22:
        v0   = next((r for r in r22 if r["condition"] == "cold_start"), {})
        v50  = next((r for r in r22 if r["condition"] == "vol_50"),     {})
        v500 = next((r for r in r22 if r["condition"] == "vol_500"),    {})
        if v0:
            table.add_row(
                "§2.2", "Cold-start latency (0 memories)",
                f"{v0.get('avg_fetch_ms', '—')} ms",
                "Baseline — no memory context",
            )
        if v50:
            table.add_row(
                "§2.2", "vol_50: retrieved / relevance / LitM risk",
                f"{v50.get('avg_retrieved','—')} / "
                f"{v50.get('avg_relevance_score','—')} / "
                f"{v50.get('litm_risk','—')}",
                "Ideal operating range for most agents",
            )
        if v500:
            table.add_row(
                "§2.2", "vol_500: retrieved / relevance / LitM risk",
                f"{v500.get('avg_retrieved','—')} / "
                f"{v500.get('avg_relevance_score','—')} / "
                f"{v500.get('litm_risk','—')}",
                "'Lost in the Middle' risk evident at scale",
            )

    # ── § 2.3 ─────────────────────────────────────────────────────────────
    r23 = results.get("2.3", [])
    if r23:
        flat_r = next((r for r in r23 if r["strategy"] == "FLAT"),        {})
        part_r = next((r for r in r23 if r["strategy"] == "PARTITIONED"), {})
        spd_r  = next((r for r in r23 if r["strategy"] == "SPEEDUP"),     {})
        if flat_r:
            table.add_row(
                "§2.3", "FLAT index: mean / p95 fetch",
                f"{flat_r.get('mean_ms','—')} / {flat_r.get('p95_ms','—')} ms",
                "Full-collection scan with user_id filter",
            )
        if part_r:
            table.add_row(
                "§2.3", "PARTITIONED index: mean / p95 fetch",
                f"{part_r.get('mean_ms','—')} / {part_r.get('p95_ms','—')} ms",
                "Partition-scoped search — faster at scale",
            )
        if spd_r:
            table.add_row(
                "§2.3", "Speedup (PARTITIONED vs FLAT)",
                f"{spd_r.get('mean_ms','—')}×",
                f"Effect size: {spd_r.get('n_runs','—')}",
            )

    # ── § 3.0 ─────────────────────────────────────────────────────────────
    r30 = results.get("3.0", [])
    if r30:
        overall = next((r for r in r30 if r["resolution_type"] == "OVERALL"), {})
        upd_r   = next((r for r in r30 if r["resolution_type"] == "UPDATE"),  {})
        ign_r   = next((r for r in r30 if r["resolution_type"] == "IGNORE"),  {})
        app_r   = next((r for r in r30 if r["resolution_type"] == "APPEND"),  {})
        if overall:
            table.add_row(
                "§3.0", "Overall consistency score",
                f"{float(overall['accuracy']):.1%}  ({overall['correct']}/{overall['total']})",
                "Fraction of conflict scenarios resolved correctly",
            )
        for lbl, rec, note in [
            ("UPDATE accuracy", upd_r, "Correctly detected belief changes"),
            ("APPEND accuracy", app_r, "Correctly identified new-info additions"),
            ("IGNORE accuracy", ign_r, "Correctly rejected semantic duplicates"),
        ]:
            if rec:
                table.add_row(
                    "§3.0", lbl,
                    f"{float(rec['accuracy']):.1%}  ({rec['correct']}/{rec['total']})",
                    note,
                )

    console.print(table)
    console.print(
        "\n[bold green]✅  All results saved in experiments/results/[/bold green]\n"
        "[dim]CSVs: research_2_1_threshold_recall.csv, research_2_2_volume_impact.csv,\n"
        "      research_2_3_latency.csv, research_3_0_conflict_summary.csv,\n"
        "      research_3_0_conflict_detail.csv, model_comparison.csv[/dim]"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Agent Research Runner")
    parser.add_argument(
        "--section", default="all",
        choices=["2.1", "2.2", "2.3", "3.0", "4.0", "all"],
        help="Which experiment section to run (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of runs for § 2.3 latency experiment (default: 30)",
    )
    parser.add_argument(
        "--model_runs", type=int, default=2,
        help="Runs per probe for § 4.0 multi-model benchmark (default: 2)",
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Memory-Agent Research Experiment Runner[/bold cyan]\n"
        "Phases: §2.1 Threshold | §2.2 Volume | §2.3 Latency | "
        "§3.0 Conflict | §4.0 Multi-Model\n"
        f"Section : {args.section}   "
        f"Latency runs: {args.runs}   "
        f"Model runs: {args.model_runs}\n"
        f"Date    : {date.today().isoformat()}",
        border_style="cyan",
    ))

    all_results: dict = {}

    if args.section in ("2.1", "all"):
        all_results["2.1"] = section_2_1()

    if args.section in ("2.2", "all"):
        all_results["2.2"] = section_2_2()

    if args.section in ("2.3", "all"):
        all_results["2.3"] = section_2_3(n_runs=args.runs)

    if args.section in ("3.0", "all"):
        all_results["3.0"] = section_3_0()

    if args.section in ("4.0", "all"):
        section_4_0(n_runs=args.model_runs)

    if all_results:
        _print_final_summary(all_results)
