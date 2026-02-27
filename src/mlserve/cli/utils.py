"""CLI utility helpers — Rich console output, spinners, and formatting."""

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

API_BASE = "http://localhost:8000"


def api_url(path: str) -> str:
    """Build full API URL."""
    return f"{API_BASE}{path}"


def check_server_running() -> bool:
    """Check if the Control Plane API server is reachable."""
    try:
        resp = httpx.get(api_url("/health"), timeout=3.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def print_success(message: str) -> None:
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_info(message: str) -> None:
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_deploy_result(result: dict) -> None:
    """Pretty-print a deployment result."""
    console.print()
    print_success(f'Model "{result["name"]}" deployed successfully (v{result["version"]})')
    console.print()
    console.print(f"  [bold]Endpoint:[/bold]  {result.get('endpoint_url', 'N/A')}")
    console.print(f"  [bold]Status:[/bold]    {result.get('status', 'N/A')}")
    console.print(f"  [bold]Framework:[/bold] {result.get('framework', 'N/A')}")
    console.print(f"  [bold]Image:[/bold]     {result.get('container_image', 'N/A')}")
    console.print(f"  [bold]Port:[/bold]      {result.get('port', 'N/A')}")
    console.print()

    endpoint = result.get("endpoint_url")
    if endpoint:
        console.print("[dim]Test it:[/dim]")
        console.print(f'  curl -X POST {endpoint} \\')
        console.print('    -H "Content-Type: application/json" \\')
        console.print('    -d \'{"instances": [[1.0, 2.0, 3.0]]}\'')
        console.print()


def print_deployments_table(deployments: list[dict]) -> None:
    """Print a Rich table of deployments."""
    if not deployments:
        print_info("No deployments found.")
        return

    table = Table(title="MLServe Deployments", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Version", justify="center")
    table.add_column("Framework", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Endpoint")
    table.add_column("Port", justify="center")

    for d in deployments:
        status = d.get("status", "unknown")
        status_style = {
            "running": "[green]running[/green]",
            "building": "[yellow]building[/yellow]",
            "deploying": "[yellow]deploying[/yellow]",
            "failed": "[red]failed[/red]",
            "stopped": "[dim]stopped[/dim]",
            "pending": "[dim]pending[/dim]",
        }.get(status, status)

        table.add_row(
            d.get("name", ""),
            f"v{d['version']}" if d.get("version") else "-",
            d.get("framework", "-"),
            status_style,
            d.get("endpoint_url", "-"),
            str(d.get("port", "-")),
        )

    console.print(table)


def print_deployment_detail(detail: dict) -> None:
    """Print detailed deployment info."""
    console.print()
    console.print(f"[bold]{detail['name']}[/bold]")
    console.print(f"  Version:    v{detail.get('version', '?')}")
    console.print(f"  Framework:  {detail.get('framework', '?')}")
    console.print(f"  Status:     {detail.get('status', '?')}")
    console.print(f"  Endpoint:   {detail.get('endpoint_url', 'N/A')}")
    console.print(f"  Image:      {detail.get('container_image', 'N/A')}")
    console.print(f"  Port:       {detail.get('port', 'N/A')}")
    console.print(f"  Created:    {detail.get('created_at', 'N/A')}")

    container = detail.get("container")
    if container:
        console.print()
        console.print("  [bold]Container:[/bold]")
        console.print(f"    ID:       {container.get('container_id', 'N/A')}")
        console.print(f"    Status:   {container.get('status', 'N/A')}")
        console.print(f"    Health:   {container.get('health', 'N/A')}")

    events = detail.get("events", [])
    if events:
        console.print()
        console.print("  [bold]Events:[/bold]")
        for e in events:
            ts = e.get('timestamp', '')
            etype = e.get('type', '')
            msg = e.get('message', '')
            console.print(f"    [{ts}] {etype}: {msg}")

    console.print()
