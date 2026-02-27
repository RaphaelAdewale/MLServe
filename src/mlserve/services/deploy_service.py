"""
Deploy Service — manages model containers via Docker.

Handles running, stopping, health checking, and listing deployed model containers.
Uses Traefik labels for automatic HTTP routing.
"""

import logging
import time
from dataclasses import dataclass

import docker
import docker.errors

from mlserve.core.config import settings
from mlserve.core.exceptions import DeploymentError

logger = logging.getLogger("mlserve.services.deploy")

CONTAINER_PREFIX = "mlserve-"
MANAGED_LABEL = "mlserve.managed"


@dataclass
class ContainerInfo:
    """Information about a running model container."""

    name: str
    container_id: str
    status: str
    health: str
    image: str
    port: int | None
    model_name: str
    framework: str
    version: str


class DeployService:
    """Manages Docker containers for model serving."""

    def __init__(self):
        try:
            self._client = docker.from_env()
            self._client.ping()
        except docker.errors.DockerException as e:
            raise DeploymentError(
                f"Cannot connect to Docker daemon. Is Docker running? Error: {e}"
            )

    def ensure_network(self) -> None:
        """Create the mlserve Docker network if it doesn't exist."""
        try:
            self._client.networks.get(settings.docker_network)
            logger.debug(f"Network '{settings.docker_network}' already exists")
        except docker.errors.NotFound:
            self._client.networks.create(settings.docker_network, driver="bridge")
            logger.info(f"Created Docker network: {settings.docker_network}")

    def deploy(
        self,
        name: str,
        image_tag: str,
        framework: str,
        version: int,
        port: int,
    ) -> str:
        """
        Deploy a model container.

        Args:
            name: Model/deployment name (e.g., "fraud-detector").
            image_tag: Full Docker image tag.
            framework: ML framework name.
            version: Model version number.
            port: Host port to map container:8080 to.

        Returns:
            The container ID.

        Raises:
            DeploymentError: If the container cannot be started.
        """
        container_name = f"{CONTAINER_PREFIX}{name}"

        # Remove existing container with same name if it exists
        self._remove_if_exists(container_name)

        # Ensure the network exists
        self.ensure_network()

        logger.info(f"Deploying {name} (image={image_tag}, port={port})")

        try:
            container = self._client.containers.run(
                image=image_tag,
                name=container_name,
                detach=True,
                environment={
                    "MODEL_NAME": name,
                    # MODEL_PATH is baked into the image via Dockerfile ENV
                    "MODEL_VERSION": str(version),
                    "FRAMEWORK": framework,
                },
                labels={
                    MANAGED_LABEL: "true",
                    "mlserve.model": name,
                    "mlserve.version": str(version),
                    "mlserve.framework": framework,
                    # Traefik auto-discovery labels
                    "traefik.enable": "true",
                    f"traefik.http.routers.{name}.rule": f"PathPrefix(`/models/{name}`)",
                    f"traefik.http.services.{name}.loadbalancer.server.port": "8080",
                    # Strip the /models/{name} prefix before forwarding to container
                    f"traefik.http.middlewares.{name}-strip.stripprefix.prefixes":
                        f"/models/{name}",
                    f"traefik.http.routers.{name}.middlewares": f"{name}-strip",
                },
                ports={"8080/tcp": port},
                network=settings.docker_network,
                restart_policy={"Name": "unless-stopped"},
            )

            logger.info(f"Container started: {container.short_id} ({container_name})")
            return container.id

        except docker.errors.APIError as e:
            raise DeploymentError(f"Failed to start container '{container_name}': {e}") from e

    def teardown(self, name: str) -> None:
        """Stop and remove a model container."""
        container_name = f"{CONTAINER_PREFIX}{name}"
        try:
            container = self._client.containers.get(container_name)
            container.stop(timeout=10)
            container.remove()
            logger.info(f"Removed container: {container_name}")
        except docker.errors.NotFound:
            logger.warning(f"Container '{container_name}' not found, nothing to remove")
        except docker.errors.APIError as e:
            raise DeploymentError(f"Failed to remove container '{container_name}': {e}") from e

    def get_status(self, name: str) -> ContainerInfo:
        """Get the status of a deployed model container."""
        container_name = f"{CONTAINER_PREFIX}{name}"
        try:
            container = self._client.containers.get(container_name)
            container.reload()
            return self._container_to_info(container)
        except docker.errors.NotFound:
            raise DeploymentError(f"Container '{container_name}' not found")

    def list_deployments(self) -> list[ContainerInfo]:
        """List all MLServe-managed containers."""
        containers = self._client.containers.list(
            all=True,
            filters={"label": f"{MANAGED_LABEL}=true"},
        )
        return [self._container_to_info(c) for c in containers]

    def get_logs(self, name: str, tail: int = 100) -> str:
        """Get recent logs from a model container."""
        container_name = f"{CONTAINER_PREFIX}{name}"
        try:
            container = self._client.containers.get(container_name)
            return container.logs(tail=tail, timestamps=True).decode("utf-8", errors="replace")
        except docker.errors.NotFound:
            raise DeploymentError(f"Container '{container_name}' not found")

    def wait_for_healthy(self, name: str, timeout: int = 120, interval: int = 2) -> bool:
        """
        Block until a container's health check passes or timeout is reached.

        Args:
            name: Model/deployment name.
            timeout: Maximum seconds to wait.
            interval: Seconds between health check polls.

        Returns:
            True if the container is healthy.

        Raises:
            DeploymentError: If the container becomes unhealthy or exits.
        """
        container_name = f"{CONTAINER_PREFIX}{name}"
        deadline = time.time() + timeout

        logger.info(f"Waiting for {container_name} to become healthy (timeout={timeout}s)")

        while time.time() < deadline:
            try:
                container = self._client.containers.get(container_name)
                container.reload()
            except docker.errors.NotFound:
                raise DeploymentError(f"Container '{container_name}' disappeared")

            status = container.status
            if status in ("exited", "dead"):
                logs = container.logs(tail=20).decode("utf-8", errors="replace")
                raise DeploymentError(
                    f"Container '{container_name}' exited unexpectedly.\nLogs:\n{logs}"
                )

            health = container.attrs.get("State", {}).get("Health", {})
            health_status = health.get("Status", "no_healthcheck")

            if health_status == "healthy":
                logger.info(f"Container {container_name} is healthy")
                return True

            if health_status == "unhealthy":
                log_entries = health.get("Log", [])
                last_log = log_entries[-1] if log_entries else {}
                raise DeploymentError(
                    f"Container '{container_name}' is unhealthy. "
                    f"Last check output: {last_log.get('Output', 'N/A')}"
                )

            time.sleep(interval)

        raise DeploymentError(
            f"Container '{container_name}' did not become healthy within {timeout}s"
        )

    def find_available_port(self, start: int = 9001) -> int:
        """
        Find an available host port starting from `start`.

        Checks existing MLServe containers to avoid conflicts.
        """
        used_ports = set()
        for info in self.list_deployments():
            if info.port is not None:
                used_ports.add(info.port)

        port = start
        while port in used_ports:
            port += 1

        return port

    def _remove_if_exists(self, container_name: str) -> None:
        """Remove a container if it already exists (any state)."""
        try:
            container = self._client.containers.get(container_name)
            container.stop(timeout=5)
            container.remove(force=True)
            logger.info(f"Removed existing container: {container_name}")
        except docker.errors.NotFound:
            pass

    def _container_to_info(self, container) -> ContainerInfo:
        """Convert a Docker container object to ContainerInfo."""
        labels = container.labels
        health = container.attrs.get("State", {}).get("Health", {})

        # Extract host port
        port = None
        ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        if "8080/tcp" in ports and ports["8080/tcp"]:
            port = int(ports["8080/tcp"][0]["HostPort"])

        return ContainerInfo(
            name=labels.get("mlserve.model", container.name),
            container_id=container.short_id,
            status=container.status,
            health=health.get("Status", "no_healthcheck"),
            image=container.image.tags[0] if container.image.tags else "unknown",
            port=port,
            model_name=labels.get("mlserve.model", "unknown"),
            framework=labels.get("mlserve.framework", "unknown"),
            version=labels.get("mlserve.version", "unknown"),
        )
