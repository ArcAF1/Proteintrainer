from __future__ import annotations
"""Simple Neo4j helper with `wait_for_neo4j` convenience.

The goal is *only* to block until a local Neo4j bolt endpoint answers so that
other modules (graph ingestion / RAG) don't fail on start-up.
"""
from typing import Optional, Callable
import time
import socket
from neo4j import GraphDatabase, Driver
from .config import settings


def wait_for_neo4j(uri: str | None = None, timeout: int = 60, notifier: Optional[Callable[[str], None]] = None) -> None:
    """Block until a Neo4j bolt endpoint accepts TCP connections.

    Parameters
    ----------
    uri:
        bolt URI; defaults to settings.neo4j_uri
    timeout:
        seconds to wait before raising RuntimeError
    notifier:
        optional callback to receive status messages (print by default)
    """

    uri = uri or settings.neo4j_uri
    host, port_str = uri.replace("bolt://", "").split(":")
    port = int(port_str)

    def _say(msg: str) -> None:
        if notifier:
            notifier(msg)
        else:
            print(msg)

    _say(f"[wait_for_neo4j] Waiting for Neo4j @ {host}:{port} …")
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            try:
                sock.connect((host, port))
                _say("[wait_for_neo4j] Bolt port is open – continuing …")
                return
            except OSError:
                _say("[wait_for_neo4j] Neo4j not ready yet …")
                time.sleep(2)
    raise RuntimeError(f"Neo4j did not become ready within {timeout} s → aborting.")


class BiomedicalKnowledgeGraph:
    """Thin convenience wrapper around the Neo4j `Driver`."""

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        wait_for_neo4j(uri)
        self._driver: Driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
            connection_timeout=settings.neo4j_connection_timeout,
        )

    @property
    def driver(self) -> Driver:  # noqa: D401
        """Return the underlying neo4j Driver."""
        return self._driver

    def close(self) -> None:
        self._driver.close() 