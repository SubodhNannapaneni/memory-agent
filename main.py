"""
main.py
─────────────────────────
Interactive memory-augmented chat agent.

Pick any model registered in ``config.MODELS`` at startup or switch mid-session
with the ``switch <model-key>`` command.  Every turn runs the full LangGraph
pipeline: Milvus retrieval → LLM response → conflict-aware memory write →
PostgreSQL metrics log.

Usage examples::

    python main.py                                        # default model
    python main.py --model gpt-5
    python main.py --model claude-sonnet-4-6
    python main.py --model gemini-3-flash --user_id alice
    python main.py --model llama3-70b --threshold 0.5

In-session commands
───────────────────
``switch <model-key>``  — hot-swap the LLM without restarting
``decay``               — run TTL + semantic pruning for the current user
``compare``             — run the full multi-model benchmark inline
``quit`` / ``exit``     — end the session
"""
import argparse
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from graph.builder import build_graph, run_turn
from config import settings

console = Console()


def start_chat(
    user_id: str,
    session_id: str,
    threshold: float,
    index_strategy: str,
    model_key: str,
):
    provider = settings.MODELS.get(model_key, {}).get("provider", "unknown")
    console.print(
        Panel.fit(
            f"[bold cyan]Memory Agent[/bold cyan]\n"
            f"model    = [magenta]{model_key}[/magenta]  ({provider})\n"
            f"user     = {user_id}  |  session = {session_id}\n"
            f"threshold= {threshold}  |  index = {index_strategy}",
            border_style="cyan",
        )
    )

    # Show all available models for reference
    console.print("[dim]Available models (type 'switch <key>' to change mid-session):[/dim]")
    for key, cfg in settings.ACTIVE_MODELS.items():
        free = " [green](free)[/green]" if cfg["cost_per_1k_in"] == 0 else ""
        console.print(f"  [dim]{key:<22} {cfg['provider']}{free}[/dim]")
    console.print("[dim]Type 'quit' to exit | 'decay' to prune memories | 'compare' to run model comparison[/dim]\n")

    graph     = build_graph()
    run_id    = uuid.uuid4().hex[:8]
    cur_model = model_key

    while True:
        user_input = Prompt.ask(f"[bold green]You[/bold green]")

        if user_input.lower() in ("quit", "exit"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.lower() == "decay":
            from memory.decay import run_decay_cycle
            run_decay_cycle(user_id)
            continue

        if user_input.lower() == "compare":
            from experiments.model_comparison import run_comparison
            run_comparison(n_runs=2)
            continue

        # Switch model mid-session: switch gpt-4o
        if user_input.lower().startswith("switch "):
            new_model = user_input.split(" ", 1)[1].strip()
            if new_model in settings.MODELS:
                cur_model = new_model
                console.print(f"[yellow]⇄ Switched to model: {cur_model}[/yellow]")
            else:
                console.print(f"[red]Unknown model '{new_model}'. Available: {list(settings.MODELS.keys())}[/red]")
            continue

        # Run one turn through the graph
        final_state = run_turn(
            graph=graph,
            user_message=user_input,
            user_id=user_id,
            session_id=session_id,
            run_id=run_id,
            threshold=threshold,
            index_strategy=index_strategy,
            model_key=cur_model,
        )

        last_ai = final_state["messages"][-1]
        console.print(
            Panel(
                last_ai.content,
                title=f"[bold blue]{cur_model}[/bold blue]",
                border_style="blue",
            )
        )

        fetch_ms    = final_state.get("memory_fetch_ms", 0)
        llm_ms      = final_state.get("llm_generation_ms", 0)
        mem_count   = len(final_state.get("retrieved_memories", []))
        write_result= final_state.get("last_memory_write", {})
        console.print(
            f"[dim]  ⏱ fetch={fetch_ms:.0f}ms  llm={llm_ms:.0f}ms  "
            f"memories_used={mem_count}  "
            f"write={write_result.get('resolution', 'NONE')}[/dim]\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Memory Agent")
    parser.add_argument("--user_id",    default=f"user_{uuid.uuid4().hex[:6]}")
    parser.add_argument("--session_id", default=f"sess_{uuid.uuid4().hex[:6]}")
    parser.add_argument(
        "--model", default=settings.DEFAULT_MODEL,
        choices=list(settings.ACTIVE_MODELS.keys()),
        help="Which LLM to use",
    )
    parser.add_argument(
        "--threshold", type=float,
        default=settings.MEMORY_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "--index", default="PARTITIONED",
        choices=["FLAT", "PARTITIONED"],
    )
    args = parser.parse_args()

    start_chat(
        user_id=args.user_id,
        session_id=args.session_id,
        threshold=args.threshold,
        index_strategy=args.index,
        model_key=args.model,
    )
