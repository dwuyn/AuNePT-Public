"""
src/pipeline
───────────
Evidence-driven pentest pipeline.

    recon -> evidence normalization -> CVE source collection ->
    candidate collection -> deterministic queue -> policy preflight ->
    method execution -> independent oracle -> cleanup

This package implements the improved research pipeline described in the
implementation handoff. The original PoC-only workflow remains archived under
the ``baseline-poc-only-v1`` git tag; nothing here mutates the legacy graph.
"""

from __future__ import annotations

__all__: list[str] = []
