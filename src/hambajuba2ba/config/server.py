"""Server configuration."""
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Server and network settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False  # Never True in production
