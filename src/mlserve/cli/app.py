"""
MLServe CLI — the main entry point for the mlserve command.

Commands:
  deploy    Deploy an ML model as a REST API
  list      List all deployments
  status    Show status of a deployment
  delete    Delete a deployment
  logs      View logs of a deployed model
  server    Start/stop the Control Plane API server
"""

import subprocess
import sys
from pathlib import Path

import httpx
import typer

from mlserve.cli.utils import (
    api_url,
    check_server_running,
    console,
    print_deploy_result,
    print_deployment_detail,
    print_deployments_table,
    print_error,
    print_info,
    print_success,
    print_warning,
)

app = typer.Typer(
    name="mlserve",
    help="Give me a model, I'll give you a production API.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

server_app = typer.Typer(help="Manage the MLServe Control Plane server.")
app.add_typer(server_app, name="server")


# --- Deploy ---


@app.command()
def deploy(
    model_path: str = typer.Argument(..., help="Path to the model file (.pkl, .joblib, .onnx)"),
    name: str = typer.Option(..., "--name", "-n", help="Deployment name (e.g., fraud-detector)"),
    framework: str | None = typer.Option(
        None, "--framework", "-f", help="ML framework (auto-detected if omitted)"
    ),
    replicas: int = typer.Option(1, "--replicas", "-r", help="Number of replicas", min=1, max=10),
):
    """Deploy an ML model as a REST API."""
    path = Path(model_path)
    if not path.exists():
        print_error(f"File not found: {model_path}")
        raise typer.Exit(1)

    if not check_server_running():
        print_error("MLServe server is not running. Start it with: mlserve server start")
        raise typer.Exit(1)

    with console.status(f"[bold blue]Deploying model '{name}'...", spinner="dots"):
        try:
            with open(path, "rb") as f:
                files = {"file": (path.name, f, "application/octet-stream")}
                data = {"name": name, "replicas": str(replicas)}
                if framework:
                    data["framework"] = framework

                resp = httpx.post(
                    api_url("/api/v1/models/deploy"),
                    files=files,
                    data=data,
                    timeout=300.0,  # Build + deploy can take a while
                )

            if resp.status_code == 200:
                print_deploy_result(resp.json())
            else:
                error = resp.json().get("detail", resp.text)
                print_error(f"Deployment failed: {error}")
                raise typer.Exit(1)

        except httpx.ConnectError:
            print_error("Cannot connect to MLServe server. Is it running?")
            raise typer.Exit(1)
        except httpx.TimeoutException:
            print_error("Request timed out. The build may still be in progress.")
            print_info("Check status with: mlserve status {name}")
            raise typer.Exit(1)


# --- List ---


@app.command("list")
def list_deployments():
    """List all model deployments."""
    if not check_server_running():
        print_error("MLServe server is not running. Start it with: mlserve server start")
        raise typer.Exit(1)

    try:
        resp = httpx.get(api_url("/api/v1/deployments"), timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            print_deployments_table(data.get("deployments", []))
        else:
            print_error(f"Failed to list deployments: {resp.text}")
    except httpx.ConnectError:
        print_error("Cannot connect to MLServe server.")
        raise typer.Exit(1)


# --- Status ---


@app.command()
def status(
    name: str = typer.Argument(..., help="Deployment name"),
):
    """Show detailed status of a deployment."""
    if not check_server_running():
        print_error("MLServe server is not running. Start it with: mlserve server start")
        raise typer.Exit(1)

    try:
        resp = httpx.get(api_url(f"/api/v1/deployments/{name}"), timeout=10.0)
        if resp.status_code == 200:
            print_deployment_detail(resp.json())
        elif resp.status_code == 404:
            print_error(f"Deployment '{name}' not found.")
            raise typer.Exit(1)
        else:
            print_error(f"Failed: {resp.text}")
            raise typer.Exit(1)
    except httpx.ConnectError:
        print_error("Cannot connect to MLServe server.")
        raise typer.Exit(1)


# --- Delete ---


@app.command()
def delete(
    name: str = typer.Argument(..., help="Deployment name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a deployment and stop its container."""
    if not check_server_running():
        print_error("MLServe server is not running. Start it with: mlserve server start")
        raise typer.Exit(1)

    if not yes:
        confirm = typer.confirm(f"Delete deployment '{name}'?")
        if not confirm:
            print_info("Cancelled.")
            raise typer.Exit(0)

    try:
        resp = httpx.delete(api_url(f"/api/v1/deployments/{name}"), timeout=30.0)
        if resp.status_code == 200:
            print_success(f"Deployment '{name}' deleted.")
        elif resp.status_code == 404:
            print_error(f"Deployment '{name}' not found.")
            raise typer.Exit(1)
        else:
            print_error(f"Failed: {resp.text}")
            raise typer.Exit(1)
    except httpx.ConnectError:
        print_error("Cannot connect to MLServe server.")
        raise typer.Exit(1)


# --- Logs ---


@app.command()
def logs(
    name: str = typer.Argument(..., help="Deployment name"),
    tail: int = typer.Option(50, "--tail", "-t", help="Number of lines to show"),
):
    """View logs of a deployed model container."""
    try:
        import docker
        import docker.errors

        client = docker.from_env()
        container_name = f"mlserve-{name}"
        container = client.containers.get(container_name)
        output = container.logs(tail=tail, timestamps=True).decode("utf-8", errors="replace")
        console.print(output)
    except docker.errors.NotFound:
        print_error(f"Container 'mlserve-{name}' not found.")
        raise typer.Exit(1)
    except docker.errors.DockerException as e:
        print_error(f"Docker error: {e}")
        raise typer.Exit(1)


# --- Server management ---


@server_app.command("start")
def server_start(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the MLServe Control Plane API server."""
    if check_server_running():
        print_warning("Server is already running.")
        return

    print_info(f"Starting MLServe server on {host}:{port}")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "mlserve.api.app:app",
        "--host", host,
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print_info("\nServer stopped.")


@server_app.command("stop")
def server_stop():
    """Stop the MLServe server (if running in background)."""
    print_info("To stop the server, press Ctrl+C in the terminal where it's running,")
    print_info("or find the process with: ps aux | grep uvicorn")


# --- Entry point ---


def main():
    """CLI entry point (called from pyproject.toml scripts)."""
    app()
