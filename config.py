"""Application configuration for the obstetric emergency decision support system."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Centralized application settings."""

    app_name: str = "Obstetric Emergency Decision Support"
    version: str = "2.0.0"
    debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    company_name: str = "TriangleZ"
    footer_text: str = "© 2026 TriangleZ. All Rights Reserved."
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key")
    engine_type: str = os.getenv("RISK_ENGINE", "model")
    model_path: str = os.getenv("MODEL_PATH", "models/model.pkl")
    audit_log_file: str = os.getenv("AUDIT_LOG_FILE", "logs/audit.log")


CONFIG = AppConfig()


def get_config() -> AppConfig:
    """Return the active application configuration."""
    return CONFIG
