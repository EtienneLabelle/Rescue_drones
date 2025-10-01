"""Datasets and generators for non-IID client distributions and drift."""

from .partitioners import seed_for_client, maybe_apply_drift

__all__ = ["seed_for_client", "maybe_apply_drift"]
