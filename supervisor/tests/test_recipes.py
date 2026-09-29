from pathlib import Path

import pytest

from app.config import Settings
from app.engines import get_engine
from app.recipes import RecipeStore, parse_recipe

from .conftest import REPO


class Ctx:
    """Minimal RunContext for command() snapshots."""

    def __init__(self, recipe, model_dir: Path, overrides=None):
        self.recipe = recipe
        self.params = recipe.resolve_params(overrides)
        self.model_dir = model_dir
        self.port = 9001
        self.settings = Settings(api_key="x")

    def log(self, line): ...
    def set_progress(self, done, total): ...
    def check_cancelled(self): ...


def test_all_repo_recipes_valid():
    files = sorted((REPO / "recipes").glob("*.yaml"))
    assert files
    for f in files:
        r = parse_recipe(f.read_text(), "builtin", base_dir=f.parent)
        assert r.id == f.stem, f"{f.name}: id should match filename"


def test_store_layers_local_over_builtin(tmp_path):
    local = tmp_path / "local"
    store = RecipeStore(REPO / "recipes", local)
    assert store.get("qwen3-0.6b-vllm").origin == "builtin"
    text = (REPO / "recipes" / "qwen3-0.6b-vllm.yaml").read_text().replace("vram_gb: 6", "vram_gb: 7")
    store.save_local(text)
    assert store.get("qwen3-0.6b-vllm").origin == "local"
    assert store.get("qwen3-0.6b-vllm").vram_gb == 7
    assert store.delete_local("qwen3-0.6b-vllm")
    assert store.get("qwen3-0.6b-vllm").origin == "builtin"


def test_param_resolution():
    r = parse_recipe((REPO / "recipes" / "qwen3.8-27b-hauhau-q8.yaml").read_text(), "builtin")
    assert r.resolve_params() == {"ctx_size": 32768, "parallel": 1, "reasoning_effort": "medium"}
    assert r.resolve_params({"ctx_size": "65536"})["ctx_size"] == 65536
    with pytest.raises(ValueError, match="unknown params"):
        r.resolve_params({"nope": 1})
    with pytest.raises(ValueError, match="reasoning_effort"):
        r.resolve_params({"reasoning_effort": "extreme"})


def test_llamacpp_command_matches_original_entrypoint(tmp_path):
    r = parse_recipe((REPO / "recipes" / "qwen3.8-27b-hauhau-q8.yaml").read_text(), "builtin")
    (tmp_path / r.source.files[0]).touch()
    (tmp_path / r.source.mmproj).touch()
    cmd = get_engine("llamacpp").command(Ctx(r, tmp_path))
    assert cmd == [
        "/opt/llama.cpp/bin/llama-server",
        "--model", str(tmp_path / "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"),
        "--host", "127.0.0.1",
        "--port", "9001",
        "--alias", "qwen3.8-27b",
        "--mmproj", str(tmp_path / "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"),
        "--n-gpu-layers", "999",
        "--ctx-size", "32768",
        "--parallel", "1",
        "--reasoning-effort", "medium",
        "--flash-attn", "on",
        "--jinja",
        "--reasoning", "on",
        "--reasoning-format", "deepseek",
        "--reasoning-preserve",
        "--spec-type", "draft-mtp",
    ]


def test_llamacpp_glob_picks_first_shard_not_mmproj(tmp_path):
    text = """
id: split
name: split
engine: llamacpp
source: {hf_repo: a/b, files: ["Q4/*.gguf"], mmproj: "Q4/mmproj*.gguf"}
served_name: split
"""
    r = parse_recipe(text, "local")
    (tmp_path / "Q4").mkdir()
    for n in ("m-00002-of-00002.gguf", "m-00001-of-00002.gguf", "mmproj-f16.gguf"):
        (tmp_path / "Q4" / n).touch()
    cmd = get_engine("llamacpp").command(Ctx(r, tmp_path))
    assert cmd[cmd.index("--model") + 1].endswith("Q4/m-00001-of-00002.gguf")
    assert cmd[cmd.index("--mmproj") + 1].endswith("Q4/mmproj-f16.gguf")


def test_vllm_command(tmp_path):
    r = parse_recipe((REPO / "recipes" / "qwen3-0.6b-vllm.yaml").read_text(), "builtin")
    cmd = get_engine("vllm").command(Ctx(r, tmp_path, {"ctx_size": 4096}))
    assert cmd[:4] == ["vllm", "serve", str(tmp_path), "--host"]
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.15"
    assert cmd[cmd.index("--served-model-name") + 1] == "qwen3-0.6b"
    assert cmd[-2:] == ["--tool-call-parser", "hermes"]


def test_bool_flags_and_unknown_param_rejected():
    ok = """
id: b
name: b
engine: vllm
source: {hf_repo: a/b}
served_name: b
params:
  enforce_eager: {type: bool, default: true}
  custom: {type: int, default: 3, flag: --custom}
"""
    r = parse_recipe(ok, "local")
    assert get_engine("vllm").flag_args(r, r.resolve_params()) == ["--enforce-eager", "--custom", "3"]
    assert get_engine("vllm").flag_args(r, r.resolve_params({"enforce_eager": "false"})) == ["--custom", "3"]
    with pytest.raises(ValueError, match="no 'flag'"):
        parse_recipe(ok.replace("flag: --custom", ""), "local")


def test_engine_source_validation():
    with pytest.raises(ValueError, match="ollama_model"):
        parse_recipe("id: o\nname: o\nengine: ollama\nsource: {}\nserved_name: o\n", "local")
    with pytest.raises(ValueError):
        parse_recipe("id: Bad Id\nname: o\nengine: ollama\nsource: {ollama_model: x}\nserved_name: o\n", "local")
