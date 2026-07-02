"""Application configuration for the obstetric emergency decision support system."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Centralized application settings."""

    app_name: str = "Obstetric Emergency Decision Support"
    version: str = "1.0.0"
    debug: bool = False
    company_name: str = "TriangleZ"
    footer_text: str = "© 2026 TriangleZ. All Rights Reserved."
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key")


CONFIG = AppConfig()


def get_config() -> AppConfig:
    """Return the active application configuration."""
    return CONFIG
