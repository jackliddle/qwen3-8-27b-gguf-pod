"""Recipe = one deployable model. YAML files under recipes/ are the "plugins"."""

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

log = logging.getLogger(__name__)

EngineName = Literal["llamacpp", "vllm", "ollama", "comfyui", "fake"]
Capability = Literal["chat", "vision", "reasoning", "tools", "embeddings", "txt2img", "img2img"]
IMAGE_ENGINES = {"comfyui"}


class ParamSpec(BaseModel):
    type: Literal["int", "float", "str", "bool", "enum", "multi"]
    default: Any = None
    values: list[str] | None = None  # for enum / multi
    description: str | None = None
    # CLI flag this maps to. Optional when the engine knows the param name
    # (e.g. ctx_size); engines that don't use flags (ollama) ignore it.
    flag: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "ParamSpec":
        if self.type in ("enum", "multi") and not self.values:
            raise ValueError(f"{self.type} params need 'values'")
        if self.default is not None:
            self.default = self.coerce(self.default)
        return self

    def coerce(self, value: Any) -> Any:
        """Coerce a UI/CLI value (often a string) to this param's type."""
        if value is None:
            return None
        match self.type:
            case "int":
                return int(value)
            case "float":
                return float(value)
            case "bool":
                if isinstance(value, str):
                    if value.lower() in ("1", "true", "yes", "on"):
                        return True
                    if value.lower() in ("0", "false", "no", "off"):
                        return False
                    raise ValueError(f"not a bool: {value!r}")
                return bool(value)
            case "enum":
                value = str(value)
                if value not in (self.values or []):
                    raise ValueError(f"{value!r} not in {self.values}")
                return value
            case "multi":
                items = [v.strip() for v in value.split(",") if v.strip()] if isinstance(value, str) else list(value)
                bad = [v for v in items if v not in (self.values or [])]
                if bad:
                    raise ValueError(f"{bad} not in {self.values}")
                return items
            case _:
                return str(value)


class Source(BaseModel):
    hf_repo: str | None = None
    revision: str | None = None
    # llamacpp: files to fetch (exact names or globs); the first match that
    # isn't the mmproj is the model passed to --model (split GGUFs: list the
    # -00001-of- shard first, llama.cpp loads the rest).
    # vllm: allow_patterns for the snapshot; empty = whole repo.
    files: list[str] = Field(default_factory=list)
    mmproj: str | None = None
    # ollama: library tag ("llama3.2:1b") or "hf.co/<repo>:<quant>".
    ollama_model: str | None = None
    # comfyui: individual files, each linked into ComfyUI's models/<dir>/.
    comfy_files: list["ComfyFile"] = Field(default_factory=list)


class ComfyFile(BaseModel):
    repo: str
    file: str  # path inside the repo
    dir: str  # ComfyUI model folder: diffusion_models, text_encoders, vae, loras, ...
    revision: str | None = None


class LoraSpec(BaseModel):
    repo: str
    file: str
    strength: float = 1.0
    # Appended to the prompt when the LoRA is applied (style LoRAs need it).
    trigger: str | None = None
    revision: str | None = None


class InputTarget(BaseModel):
    node: str
    input: str


class LoraInsert(BaseModel):
    model_node: str  # node whose MODEL output (slot 0) gets the LoRA chain
    clip_node: str | None = None  # if set, LoRAs also patch this CLIP (slot 0)


class Recipe(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str
    description: str | None = None
    engine: EngineName
    source: Source
    served_name: str
    vram_gb: float = 0
    disk_gb: float | None = None
    capabilities: list[Capability] = Field(default_factory=lambda: ["chat"])
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # Seconds to wait for the engine to report healthy after launch.
    start_timeout: int = 1800
    # comfyui only: API-format workflow (inline, or a path relative to the
    # recipe file, resolved at load time), how request fields map onto it,
    # optional LoRA catalog, and warm-up overrides run before "ready".
    workflow: dict[str, Any] | str | None = None
    inputs: dict[str, list[InputTarget]] = Field(default_factory=dict)
    loras: dict[str, LoraSpec] = Field(default_factory=dict)
    lora_insert: LoraInsert | None = None
    warmup: dict[str, Any] = Field(default_factory=dict)
    origin: str = "builtin"  # builtin | remote | local | dev (set by loader)

    @property
    def kind(self) -> str:
        return "image" if self.engine in IMAGE_ENGINES else "llm"

    def resolve_params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        overrides = overrides or {}
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise ValueError(f"unknown params for {self.id}: {sorted(unknown)}")
        out: dict[str, Any] = {}
        for name, spec in self.params.items():
            try:
                out[name] = spec.coerce(overrides[name]) if name in overrides else spec.default
            except (TypeError, ValueError) as e:
                raise ValueError(f"param {name}: {e}") from e
        return out


def parse_recipe(text: str, origin: str, base_dir: Path | None = None, files: dict[str, str] | None = None) -> Recipe:
    """Parse + validate a recipe. A string `workflow:` is resolved against
    `files` (remote tarball contents) or `base_dir` (the recipe's directory)."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("recipe must be a YAML mapping")
    data["origin"] = origin
    wf = data.get("workflow")
    if isinstance(wf, str):
        if files is not None and wf in files:
            data["workflow"] = json.loads(files[wf])
        elif base_dir is not None and (base_dir / wf).is_file():
            data["workflow"] = json.loads((base_dir / wf).read_text())
        else:
            raise ValueError(f"workflow file not found: {wf}")
    recipe = Recipe.model_validate(data)
    from .engines import get_engine  # local import: engines import this module

    get_engine(recipe.engine).validate(recipe)
    return recipe


def _load_dir(path: Path, origin: str, base_dir: Path | None = None) -> dict[str, Recipe]:
    out: dict[str, Recipe] = {}
    if not path.is_dir():
        return out
    for f in sorted(path.glob("*.y*ml")):
        try:
            r = parse_recipe(f.read_text(), origin, base_dir=base_dir or path)
        except (ValidationError, ValueError, yaml.YAMLError) as e:
            log.warning("skipping invalid recipe %s: %s", f, e)
            continue
        out[r.id] = r
    return out


async def fetch_remote_recipes(repo: str, ref: str, token: str | None) -> dict[str, Recipe]:
    """Pull recipes/*.yaml out of the repo tarball (works for private repos with a token)."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{repo}/tarball/{ref}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as c:
        resp = await c.get(url, headers=headers)
        resp.raise_for_status()
    # <owner>-<repo>-<sha>/recipes/<path>: collect everything under recipes/
    # first so recipes can reference workflows/*.json from the same tarball.
    files: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = m.name.split("/")
            if len(parts) >= 3 and parts[1] == "recipes" and m.isfile():
                fh = tar.extractfile(m)
                if fh is not None:
                    files["/".join(parts[2:])] = fh.read().decode()
    out: dict[str, Recipe] = {}
    for name, text in files.items():
        if "/" in name or not name.endswith((".yaml", ".yml")):
            continue
        try:
            r = parse_recipe(text, "remote", files=files)
        except (ValidationError, ValueError, yaml.YAMLError) as e:
            log.warning("skipping invalid remote recipe %s: %s", name, e)
            continue
        out[r.id] = r
    return out


class RecipeStore:
    def __init__(self, builtin_dir: Path, local_dir: Path, dev_dir: Path | None = None):
        self.builtin_dir = builtin_dir
        self.local_dir = local_dir
        self.dev_dir = dev_dir
        self._remote: dict[str, Recipe] = {}
        self.remote_error: str | None = None
        self.recipes: dict[str, Recipe] = {}
        self.reload()

    def reload(self) -> None:
        # Later layers win: builtin < remote (repo main) < local (pushed at runtime).
        merged: dict[str, Recipe] = {}
        if self.dev_dir:
            merged.update(_load_dir(self.dev_dir, "dev"))
        merged.update(_load_dir(self.builtin_dir, "builtin"))
        merged.update(self._remote)
        # Pushed recipes may reference the image's built-in workflows/ files.
        merged.update(_load_dir(self.local_dir, "local", base_dir=self.builtin_dir))
        self.recipes = merged

    async def refresh_remote(self, repo: str | None, ref: str, token: str | None) -> None:
        if repo:
            try:
                self._remote = await fetch_remote_recipes(repo, ref, token)
                self.remote_error = None
            except Exception as e:  # network/auth errors shouldn't take the supervisor down
                log.warning("remote recipe fetch failed: %s", e)
                self.remote_error = str(e)
        self.reload()

    def get(self, recipe_id: str) -> Recipe:
        try:
            return self.recipes[recipe_id]
        except KeyError:
            raise KeyError(f"no such recipe: {recipe_id}") from None

    def save_local(self, text: str) -> Recipe:
        recipe = parse_recipe(text, "local", base_dir=self.builtin_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        (self.local_dir / f"{recipe.id}.yaml").write_text(text)
        self.reload()
        return self.recipes[recipe.id]

    def delete_local(self, recipe_id: str) -> bool:
        path = self.local_dir / f"{recipe_id}.yaml"
        if not path.exists():
            return False
        path.unlink()
        self.reload()
        return True
