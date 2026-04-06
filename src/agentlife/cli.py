"""CLI entry point for AgentLife."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="agentlife")
def cli():
    """AgentLife — Peek inside your AI agents."""
    pass


@cli.command()
@click.option("--port", "-p", default=8777, help="Port to serve on")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
def ui(port: int, host: str):
    """Launch the AgentLife web dashboard."""
    import uvicorn

    console.print(f"\n  [bold magenta]AgentLife[/] dashboard starting...\n")
    console.print(f"  [dim]Open in browser:[/] [bold cyan]http://{host}:{port}[/]\n")

    uvicorn.run(
        "agentlife.server.app:app",
        host=host,
        port=port,
        log_level="warning",
    )


@cli.command()
@click.option("--limit", "-n", default=20, help="Number of sessions to show")
def sessions(limit: int):
    """List recent traced sessions."""
    import asyncio
    from agentlife.store import Store

    async def _list():
        store = Store()
        return await store.list_sessions(limit=limit)

    results = asyncio.run(_list())

    if not results:
        console.print("[dim]No sessions found.[/]")
        return

    table = Table(title="Recent Sessions", show_lines=False)
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Spans", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Time")

    for s in results:
        status_style = {"ok": "green", "error": "red", "running": "yellow"}.get(s.status.value, "")
        dur = f"{s.total_duration_ms:.0f}ms" if s.total_duration_ms else "-"
        cost = f"${s.total_cost:.4f}" if s.total_cost else "$0"
        table.add_row(
            s.id, s.name,
            f"[{status_style}]{s.status.value}[/{status_style}]",
            str(s.span_count), str(s.total_tokens), cost, dur,
            s.started_at[:19] if s.started_at else "-",
        )

    console.print(table)


@cli.command()
@click.confirmation_option(prompt="Delete ALL trace data?")
def clear():
    """Delete all stored sessions and spans."""
    import asyncio
    from agentlife.store import Store

    asyncio.run(Store().clear_all())
    console.print("[green]All trace data cleared.[/]")


if __name__ == "__main__":
    cli()
