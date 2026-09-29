"""Pure functions for filling in / rewriting ComfyUI API-format workflows.

An API-format workflow is {node_id: {"class_type": str, "inputs": {name: value | [node_id, slot]}}}.
"""

import copy
import posixpath
from typing import Any

from .recipes import InputTarget, Recipe

LORA_NODE_PREFIX = "mp_lora_"


def model_ref(recipe_id: str, filename: str) -> str:
    """Name ComfyUI lists a recipe's file under: files are linked into
    models/<dir>/<recipe-id>/, so two recipes can ship same-named files."""
    return f"{recipe_id}/{posixpath.basename(filename)}"


def prefix_model_names(workflow: dict, recipe: Recipe) -> dict:
    """Point loader inputs at this recipe's linked files.

    Workflows are written with plain filenames (as exported from ComfyUI);
    any string input equal to the basename of one of the recipe's files or
    LoRAs is rewritten to "<recipe-id>/<basename>".
    """
    names = {posixpath.basename(f.file) for f in recipe.source.comfy_files}
    names |= {posixpath.basename(lo.file) for lo in recipe.loras.values()}
    wf = copy.deepcopy(workflow)
    for node in wf.values():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, str) and val in names:
                node["inputs"][key] = model_ref(recipe.id, val)
    return wf


def render(workflow: dict, inputs: dict[str, list[InputTarget]], values: dict[str, Any]) -> dict:
    """Set every mapped input whose value is not None (None = keep the workflow's own value)."""
    wf = copy.deepcopy(workflow)
    for field, targets in inputs.items():
        value = values.get(field)
        if value is None:
            continue
        for t in targets:
            if t.node not in wf:
                raise KeyError(f"input {field!r} targets missing node {t.node!r}")
            wf[t.node].setdefault("inputs", {})[t.input] = value
    return wf


def insert_loras(
    workflow: dict,
    model_node: str,
    clip_node: str | None,
    loras: list[tuple[str, float]],
) -> dict:
    """Splice a chain of LoRA loaders after `model_node` (and `clip_node`).

    Every input that consumed [model_node, 0] (or [clip_node, 0]) is rewired
    to the end of the chain, so this works on any workflow without knowing
    its shape.
    """
    wf = copy.deepcopy(workflow)
    if not loras:
        return wf
    for n in (model_node, clip_node):
        if n is not None and n not in wf:
            raise KeyError(f"lora_insert node {n!r} not in workflow")

    model_refs: list[tuple[str, str]] = []
    clip_refs: list[tuple[str, str]] = []
    for nid, node in wf.items():
        for key, val in node.get("inputs", {}).items():
            if val == [model_node, 0]:
                model_refs.append((nid, key))
            elif clip_node is not None and val == [clip_node, 0]:
                clip_refs.append((nid, key))

    prev_model: list = [model_node, 0]
    prev_clip: list | None = [clip_node, 0] if clip_node else None
    for i, (name, strength) in enumerate(loras):
        nid = f"{LORA_NODE_PREFIX}{i}"
        if prev_clip is not None:
            wf[nid] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": name,
                    "strength_model": strength,
                    "strength_clip": strength,
                    "model": prev_model,
                    "clip": prev_clip,
                },
            }
            prev_clip = [nid, 1]
        else:
            wf[nid] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": name, "strength_model": strength, "model": prev_model},
            }
        prev_model = [nid, 0]

    for nid, key in model_refs:
        wf[nid]["inputs"][key] = prev_model
    for nid, key in clip_refs:
        wf[nid]["inputs"][key] = prev_clip
    return wf


def set_output_prefix(workflow: dict, prefix: str) -> dict:
    """Route SaveImage output to a known subfolder/prefix so results can be found."""
    wf = copy.deepcopy(workflow)
    for node in wf.values():
        if node.get("class_type") == "SaveImage":
            node.setdefault("inputs", {})["filename_prefix"] = prefix
    return wf


def _combo_options(spec: Any) -> list | None:
    """Options for a COMBO input spec, in either object_info format:
    [[opt, ...], {...}] (classic) or ["COMBO", {"options": [...]}] (newer)."""
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def check_against_object_info(workflow: dict, object_info: dict) -> list[str]:
    """Problems that would make ComfyUI reject the prompt: unknown node
    classes, or a model filename it doesn't list (not linked / wrong name)."""
    errors = []
    for nid, node in workflow.items():
        cls = node.get("class_type")
        info = object_info.get(cls)
        if info is None:
            errors.append(f"node {nid}: unknown class {cls!r} (ComfyUI too old, or needs a custom node)")
            continue
        specs = {**info.get("input", {}).get("required", {}), **info.get("input", {}).get("optional", {})}
        for key, val in node.get("inputs", {}).items():
            if not isinstance(val, str):
                continue
            options = _combo_options(specs.get(key))
            if options is not None and key.endswith("_name") and val not in options:
                errors.append(f"node {nid} ({cls}): {key}={val!r} not available")
    return errors


def build(recipe: Recipe, values: dict[str, Any], loras: list[tuple[str, float]] | None = None) -> dict:
    """Recipe workflow → ready-to-submit prompt: link names, LoRA chain, request values."""
    if not isinstance(recipe.workflow, dict):
        raise ValueError(f"{recipe.id} has no workflow")
    wf = prefix_model_names(recipe.workflow, recipe)
    if loras:
        if recipe.lora_insert is None:
            raise ValueError(f"{recipe.id} doesn't support LoRAs (no lora_insert)")
        chain = [(model_ref(recipe.id, recipe.loras[name].file), strength) for name, strength in loras]
        wf = insert_loras(wf, recipe.lora_insert.model_node, recipe.lora_insert.clip_node, chain)
    return render(wf, recipe.inputs, values)
