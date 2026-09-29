from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.comfy_workflow import (
    build,
    check_against_object_info,
    insert_loras,
    prefix_model_names,
    render,
    set_output_prefix,
)
from app.images import ImageRequest, LoraUse, build_values
from app.recipes import InputTarget, Recipe, RecipeStore, parse_recipe

from .conftest import REPO, SUPERVISOR

WF = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "m.safetensors"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "te.safetensors"}},
    "3": {"class_type": "ModelSampling", "inputs": {"model": ["1", 0]}},
    "4": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 5, "steps": 8}},
    "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "x"}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "a", "images": ["4", 0]}},
}


@pytest.fixture(scope="module")
def store():
    return RecipeStore(REPO / "recipes", Path("/nonexistent"), SUPERVISOR / "dev-recipes")


def test_render_sets_mapped_inputs_and_skips_none():
    inputs = {"seed": [InputTarget(node="4", input="seed")], "steps": [InputTarget(node="4", input="steps")]}
    out = render(WF, inputs, {"seed": 42, "steps": None, "unmapped": 1})
    assert out["4"]["inputs"]["seed"] == 42
    assert out["4"]["inputs"]["steps"] == 8  # None keeps the workflow's value
    assert WF["4"]["inputs"]["seed"] == 5  # input not mutated
    with pytest.raises(KeyError, match="missing node"):
        render(WF, {"x": [InputTarget(node="nope", input="a")]}, {"x": 1})


def test_insert_loras_chains_and_rewires_every_consumer():
    out = insert_loras(WF, "1", None, [("a.safetensors", 0.8), ("b.safetensors", 0.5)])
    assert out["mp_lora_0"]["inputs"] == {"lora_name": "a.safetensors", "strength_model": 0.8, "model": ["1", 0]}
    assert out["mp_lora_1"]["inputs"]["model"] == ["mp_lora_0", 0]
    assert out["3"]["inputs"]["model"] == ["mp_lora_1", 0]
    assert out["4"]["inputs"]["model"] == ["mp_lora_1", 0]
    assert insert_loras(WF, "1", None, []) == WF


def test_insert_loras_with_clip():
    out = insert_loras(WF, "1", "2", [("a.safetensors", 1.0)])
    assert out["mp_lora_0"]["class_type"] == "LoraLoader"
    assert out["5"]["inputs"]["clip"] == ["mp_lora_0", 1]
    assert out["4"]["inputs"]["model"] == ["mp_lora_0", 0]


def test_output_prefix_and_object_info_check():
    assert set_output_prefix(WF, "mp/x")["9"]["inputs"]["filename_prefix"] == "mp/x"
    info = {
        "UNETLoader": {"input": {"required": {"unet_name": [["m.safetensors"], {}]}}},
        # newer COMBO spelling
        "CLIPLoader": {"input": {"required": {"clip_name": ["COMBO", {"options": ["other.safetensors"]}]}}},
        "ModelSampling": {}, "KSampler": {}, "CLIPTextEncode": {}, "SaveImage": {},
    }
    errors = check_against_object_info(WF, info)
    assert errors == ["node 2 (CLIPLoader): clip_name='te.safetensors' not available"]
    del info["KSampler"]
    assert any("unknown class 'KSampler'" in e for e in check_against_object_info(WF, info))


def test_qwen_recipe_build_prefixes_names_and_stacks_user_lora_under_lightning(store):
    r = store.get("qwen-image-2512")
    assert r.kind == "image"
    r = Recipe.model_validate({**r.model_dump(), "loras": {"x": {"repo": "a/b", "file": "loras/x.safetensors"}}})
    wf = build(r, {"prompt": "hi", "fast": True, "steps": 6, "width": 512}, [("x", 0.7)])
    assert wf["238:226"]["inputs"]["unet_name"] == "qwen-image-2512/qwen_image_2512_fp8_e4m3fn.safetensors"
    assert wf["238:221"]["inputs"]["lora_name"] == "qwen-image-2512/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
    assert wf["mp_lora_0"]["inputs"]["lora_name"] == "qwen-image-2512/x.safetensors"
    # both consumers of the UNET (Lightning LoRA + the model switch) now read the user LoRA
    assert wf["238:221"]["inputs"]["model"] == ["mp_lora_0", 0]
    assert wf["238:233"]["inputs"]["on_false"] == ["mp_lora_0", 0]
    assert wf["238:229"]["inputs"]["value"] is True
    assert wf["238:224"]["inputs"]["value"] == wf["238:225"]["inputs"]["value"] == 6
    assert wf["238:227"]["inputs"]["text"] == "hi"


def test_all_image_recipes_reference_valid_nodes(store):
    image = [r for r in store.recipes.values() if r.kind == "image"]
    assert {"qwen-image-2512", "krea-2-turbo", "krea-2-raw", "qwen-image-edit-2511"} <= {r.id for r in image}
    for r in image:
        loras = [(n, 1.0) for n in r.loras][:2]
        build(r, {"prompt": "p", "seed": 1, "image": "in.png"}, loras)  # raises on any bad reference


def test_validation_errors():
    base = (SUPERVISOR / "dev-recipes" / "fake-image.yaml").read_text()
    with pytest.raises(ValueError, match="not in workflow"):
        parse_recipe(base.replace('prompt: [{node: "6"', 'prompt: [{node: "66"'), "local")
    with pytest.raises(ValueError, match="not in the loras catalog"):
        parse_recipe(base.replace("values: [inkwash, neon]", "values: [inkwash, nope]"), "local")
    with pytest.raises(ValueError, match="need inputs.image"):
        parse_recipe(base.replace("capabilities: [txt2img]", "capabilities: [img2img]"), "local")
    with pytest.raises(ValueError, match="multi params need"):
        parse_recipe(base.replace("loras: {type: multi, values: [inkwash, neon], default: [inkwash]}", "loras: {type: multi}"), "local")


def test_build_values_loras_triggers_params(store):
    r = store.get("fake-image")
    dep = SimpleNamespace(recipe=r, params=r.resolve_params())  # deployed with [inkwash]
    values, loras = build_values(dep, ImageRequest(model="fake-image", prompt="a cat", size="500x333", loras=[LoraUse(name="inkwash")]))
    assert loras == [("inkwash", 0.8)]
    assert values["prompt"] == "a cat, ink wash style"
    assert (values["width"], values["height"]) == (496, 336)  # rounded to multiples of 16
    assert isinstance(values["seed"], int)
    with pytest.raises(HTTPException, match="wasn't selected at deploy"):
        build_values(dep, ImageRequest(model="fake-image", prompt="x", loras=[LoraUse(name="neon")]))
    with pytest.raises(HTTPException, match="unknown LoRA"):
        build_values(dep, ImageRequest(model="fake-image", prompt="x", loras=[LoraUse(name="zzz")]))
    with pytest.raises(HTTPException, match="unknown param"):
        build_values(dep, ImageRequest(model="fake-image", prompt="x", params={"bogus": 1}))
