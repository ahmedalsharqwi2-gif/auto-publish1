"""Shared deterministic Arabic speech preflight and resilience primitives."""
from .normalize import PreparedText, prepare_text
from .gates import GateError, provider_preflight, text_gate, tts_smoke_test
from .provider_pool import ProviderPool

__all__ = ["PreparedText", "prepare_text", "GateError", "text_gate", "provider_preflight", "tts_smoke_test", "ProviderPool"]
