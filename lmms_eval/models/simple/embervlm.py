"""
EmberVLM adapter entrypoint for lmms-eval.

This module exists to satisfy the default lmms-eval import path:
`lmms_eval.models.simple.embervlm.EmberVLM`.

It re-exports the EmberVLM adapter defined in tinyllava.py.
"""

from lmms_eval.models.simple.tinyllava import EmberVLM  # noqa: F401

__all__ = ["EmberVLM"]
