from .base import Cancelled, Engine, RunContext, Upstream
from .fake import FakeEngine
from .llamacpp import LlamaCppEngine
from .ollama import OllamaEngine
from .vllm import VllmEngine

_ENGINES: dict[str, Engine] = {
    e.name: e for e in (LlamaCppEngine(), VllmEngine(), OllamaEngine(), FakeEngine())
}


def get_engine(name: str) -> Engine:
    try:
        return _ENGINES[name]
    except KeyError:
        raise ValueError(f"unknown engine: {name}") from None


def all_engines() -> list[Engine]:
    return list(_ENGINES.values())


__all__ = ["Cancelled", "Engine", "RunContext", "Upstream", "get_engine", "all_engines"]
