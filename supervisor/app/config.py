from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Bearer token for /api and /v1. Required: the supervisor refuses to start
    # without it, same contract as the old single-model entrypoint.
    api_key: str

    host: str = "0.0.0.0"
    port: int = 8000

    # Model files live under models_dir/<recipe-id>/ and are deleted on stop.
    models_dir: Path = Path("/workspace/models")

    # Recipes baked into the image, plus recipes pushed at runtime via the API
    # (so a new model can be tried without an image rebuild or a commit).
    recipes_dir: Path = REPO_ROOT / "recipes"
    local_recipes_dir: Path = Path("/workspace/recipes")
    # Optional "owner/repo": re-fetch recipes/ from GitHub at startup and on
    # "Refresh recipes". github_token is only needed for a private repo.
    recipes_repo: str | None = None
    recipes_ref: str = "main"
    github_token: str | None = None

    llama_server_bin: str = "/opt/llama.cpp/bin/llama-server"
    vllm_bin: str = "vllm"
    ollama_bin: str = "ollama"
    ollama_port: int = 11434

    # First port handed to per-model engine processes (bound to 127.0.0.1).
    engine_base_port: int = 9001

    # Set by RunPod on every pod; used to build the public proxy URL.
    runpod_pod_id: str | None = None
    # Explicit override for the public base URL (e.g. behind another proxy).
    public_url: str | None = None

    # Dev mode enables the "fake" engine + dev-recipes/ so the whole stack
    # (UI, gateway, lifecycle) can be exercised locally without a GPU.
    mp_dev: bool = False

    static_dir: Path = REPO_ROOT / "frontend" / "dist"

    def public_base_url(self, fallback: str) -> str:
        if self.public_url:
            return self.public_url.rstrip("/")
        if self.runpod_pod_id:
            return f"https://{self.runpod_pod_id}-{self.port}.proxy.runpod.net"
        return fallback.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
