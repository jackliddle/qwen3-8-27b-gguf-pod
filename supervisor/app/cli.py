"""`mp` — command-line client for the supervisor API (for humans, scripts and agents).

Config: MP_URL (default http://localhost:8000) and MP_API_KEY (or API_KEY).
Every command exits non-zero on failure so it can gate scripts.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

from .smoke import run_image_smoke, run_smoke

TERMINAL = {"ready", "failed", "stopped"}


def _client() -> httpx.Client:
    url = os.environ.get("MP_URL", "http://localhost:8000").rstrip("/")
    key = os.environ.get("MP_API_KEY") or os.environ.get("API_KEY")
    if not key:
        sys.exit("set MP_API_KEY (or API_KEY)")
    return httpx.Client(base_url=url, headers={"Authorization": f"Bearer {key}"}, timeout=120)


def _key() -> str:
    return os.environ.get("MP_API_KEY") or os.environ.get("API_KEY") or ""


def _call(c: httpx.Client, method: str, path: str, **kw) -> dict | list:
    r = c.request(method, path, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        sys.exit(f"error: HTTP {r.status_code}: {detail}")
    return r.json()


def _gb(n: float | None) -> str:
    return "?" if n is None else f"{n / 1e9:.1f}GB"


def _progress(d: dict) -> str:
    p = d.get("progress")
    if not p or not p.get("total_bytes"):
        return ""
    return f" {p['done_bytes'] / p['total_bytes'] * 100:5.1f}% of {_gb(p['total_bytes'])}"


def cmd_status(c, a):
    s = _call(c, "GET", "/api/status")
    if a.json:
        return print(json.dumps(s, indent=2))
    print(f"UI:        {s['public_url']}")
    print(f"OpenAI:    {s['openai_base_url']}")
    for g in s["gpus"]:
        print(f"GPU {g['index']}:     {g['name']}  {g['memory_used_mib'] / 1024:.1f}/{g['memory_total_mib'] / 1024:.1f}GB used")
    if not s["gpus"]:
        print("GPU:       none detected")
    print(f"Disk:      {_gb(s['disk']['free_bytes'])} free of {_gb(s['disk']['total_bytes'])} ({s['disk']['path']})")
    if s.get("recipes_remote_error"):
        print(f"Recipes:   remote fetch failed: {s['recipes_remote_error']}")


def cmd_recipes(c, a):
    rs = _call(c, "GET", "/api/recipes")
    if a.json:
        return print(json.dumps(rs, indent=2))
    for r in rs:
        status = r["deployment_status"] or "-"
        print(f"{r['id']:<32} {r['engine']:<9} {r['vram_gb']:>5g}GB  {status:<11} {r['origin']:<8} {r['name']}")


def cmd_recipe_push(c, a):
    r = _call(c, "POST", "/api/recipes", content=Path(a.file).read_text(), headers={"Content-Type": "text/plain"})
    print(f"pushed {r['id']} ({r['engine']})")


def cmd_recipe_rm(c, a):
    _call(c, "DELETE", f"/api/recipes/{a.id}")
    print(f"deleted local recipe {a.id}")


def cmd_refresh(c, a):
    r = _call(c, "POST", "/api/recipes/refresh")
    print(f"{r['count']} recipes" + (f" (remote error: {r['remote_error']})" if r["remote_error"] else ""))


def _parse_sets(sets: list[str]) -> dict:
    out = {}
    for s in sets:
        k, sep, v = s.partition("=")
        if not sep:
            sys.exit(f"--set expects key=value, got {s!r}")
        out[k] = v
    return out


def _wait(c, dep_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while True:
        d = _call(c, "GET", f"/api/deployments/{dep_id}")
        line = f"{d['status']}{_progress(d)}"
        if line != last:
            print(f"  {dep_id}: {line}", file=sys.stderr)
            last = line
        if d["status"] in TERMINAL:
            return d
        if time.monotonic() > deadline:
            sys.exit(f"timed out after {timeout:.0f}s waiting for {dep_id} (status {d['status']})")
        time.sleep(3)


def cmd_deploy(c, a):
    d = _call(c, "POST", "/api/deployments", json={"recipe_id": a.id, "params": _parse_sets(a.set), "force": a.force})
    print(f"deploying {d['id']} with {json.dumps(d['params'])}")
    if a.wait:
        d = _wait(c, a.id, a.timeout)
        if d["status"] != "ready":
            _dump_tail(c, a.id, 40)
            sys.exit(f"{a.id} {d['status']}: {d.get('error')}")
        _print_endpoint(d)


def cmd_wait(c, a):
    d = _wait(c, a.id, a.timeout)
    if d["status"] != "ready":
        sys.exit(f"{a.id} {d['status']}: {d.get('error')}")


def cmd_ls(c, a):
    ds = _call(c, "GET", "/api/deployments")
    if a.json:
        return print(json.dumps(ds, indent=2))
    if not ds:
        print("no deployments")
    for d in ds:
        err = f"  ({d['error']})" if d.get("error") else ""
        print(f"{d['id']:<32} {d['engine']:<9} {d['status']}{_progress(d)}  model={d['served_name']}{err}")


def cmd_stop(c, a):
    d = _call(c, "POST", f"/api/deployments/{a.id}/stop", params={"delete_files": not a.keep_files})
    print(f"{d['id']}: {d['status']}" + (" (files kept)" if a.keep_files else " (files deleted)"))


def cmd_restart(c, a):
    d = _call(c, "POST", f"/api/deployments/{a.id}/restart", params={"force": a.force})
    print(f"restarting {d['id']}")
    if a.wait:
        d = _wait(c, a.id, a.timeout)
        if d["status"] != "ready":
            sys.exit(f"{a.id} {d['status']}: {d.get('error')}")


def _dump_tail(c, dep_id: str, n: int) -> None:
    for e in _call(c, "GET", f"/api/deployments/{dep_id}/logs")[-n:]:
        print(e["line"], file=sys.stderr)


def cmd_logs(c, a):
    since = 0
    entries = _call(c, "GET", f"/api/deployments/{a.id}/logs")
    for e in entries[-a.tail :] if a.tail else entries:
        print(e["line"])
    since = entries[-1]["seq"] if entries else 0
    while a.follow:
        time.sleep(2)
        for e in _call(c, "GET", f"/api/deployments/{a.id}/logs", params={"since": since}):
            print(e["line"], flush=True)
            since = e["seq"]


def _print_endpoint(d: dict) -> None:
    ep = d["endpoint"]
    print(
        f"""
== {d['name']} ==
OPENAI_BASE_URL={ep['base_url']}
OPENAI_API_KEY={_key()}
MODEL={ep['model']}

{_example(d)}"""
    )


def _example(d: dict) -> str:
    ep = d["endpoint"]
    if d.get("kind") != "image":
        return f"""curl {ep['base_url']}/chat/completions \\
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \\
  -d '{{"model": "{ep['model']}", "messages": [{{"role": "user", "content": "Hello"}}]}}'"""
    if "txt2img" in d["capabilities"]:
        return f"""curl {ep['base_url']}/images/generations \\
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \\
  -d '{{"model": "{ep['model']}", "prompt": "a red fox in the snow", "size": "1024x1024", "response_format": "url"}}'

# slow jobs (>~90s through the RunPod proxy): mp gen {ep['model']} "a red fox in the snow" -o fox.png"""
    return f"""curl {ep['base_url']}/images/edits \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -F model={ep['model']} -F image=@input.png -F prompt="make it night time" -F response_format=url"""


def cmd_endpoints(c, a):
    ds = [d for d in _call(c, "GET", "/api/deployments") if d["status"] == "ready" and (not a.id or d["id"] == a.id)]
    if not ds:
        sys.exit("no ready deployments")
    for d in ds:
        _print_endpoint(d)


def cmd_test(c, a):
    d = _call(c, "GET", f"/api/deployments/{a.id}")
    if d["status"] != "ready":
        sys.exit(f"{a.id} is {d['status']}, not ready")
    caps = a.caps.split(",") if a.caps else d["capabilities"]
    base = d["endpoint"]["base_url"] if a.public else f"{c.base_url}/v1"
    if d.get("kind") == "image":
        loras = list(d["params"].get("loras") or [])
        results = run_image_smoke(base, _key(), d["served_name"], caps, loras, a.save, a.size)
    else:
        results = run_smoke(base, _key(), d["served_name"], caps, a.max_tokens)
    if a.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for r in results:
            print(f"{'PASS' if r.ok else 'FAIL'}  {r.name:<10} {r.seconds:6.2f}s  {r.detail}")
    if not all(r.ok for r in results):
        sys.exit(1)


def cmd_gen(c, a):
    """One image via the async job API (no proxy timeout), saved to --out."""
    import base64

    body = {"model": a.model, "prompt": a.prompt, "size": a.size, "n": a.n}
    for key in ("negative_prompt", "seed", "steps", "cfg"):
        if getattr(a, key) is not None:
            body[key] = getattr(a, key)
    if a.lora:
        body["loras"] = [{"name": n, **({"strength": float(st)} if st else {})} for n, _, st in (x.partition(":") for x in a.lora)]
    if a.set:
        body["params"] = _parse_sets(a.set)
    if a.image:
        body["image_b64"] = base64.b64encode(Path(a.image).read_bytes()).decode()
    t0 = time.monotonic()
    job = _call(c, "POST", "/api/images/jobs", json=body)
    while job["status"] not in ("done", "error"):
        time.sleep(1)
        job = _call(c, "GET", f"/api/images/jobs/{job['job_id']}")
    if job["status"] == "error":
        sys.exit(f"generation failed: {job['error']}")
    out = Path(a.out)
    for i, img in enumerate(job["images"]):
        path = out if len(job["images"]) == 1 else out.with_name(f"{out.stem}-{i + 1}{out.suffix}")
        path.write_bytes(httpx.get(img["url"], timeout=60).raise_for_status().content)
        print(path)
    print(f"seed {job['seed']}, {time.monotonic() - t0:.1f}s", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(prog="mp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="GPU/disk/URLs")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("recipes", help="list recipes")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_recipes)

    s = sub.add_parser("push", help="add/replace a runtime recipe from a YAML file")
    s.add_argument("file")
    s.set_defaults(fn=cmd_recipe_push)

    s = sub.add_parser("rm-recipe", help="delete a runtime (pushed) recipe")
    s.add_argument("id")
    s.set_defaults(fn=cmd_recipe_rm)

    s = sub.add_parser("refresh", help="re-fetch recipes from the GitHub repo")
    s.set_defaults(fn=cmd_refresh)

    s = sub.add_parser("deploy", help="deploy a recipe")
    s.add_argument("id")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a param")
    s.add_argument("--force", action="store_true", help="deploy even if VRAM looks insufficient")
    s.add_argument("--wait", action="store_true", help="block until ready/failed")
    s.add_argument("--timeout", type=float, default=3600)
    s.set_defaults(fn=cmd_deploy)

    s = sub.add_parser("wait", help="block until a deployment is ready/failed")
    s.add_argument("id")
    s.add_argument("--timeout", type=float, default=3600)
    s.set_defaults(fn=cmd_wait)

    s = sub.add_parser("ls", help="list deployments")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_ls)

    s = sub.add_parser("stop", help="stop a deployment (deletes model files unless --keep-files)")
    s.add_argument("id")
    s.add_argument("--keep-files", action="store_true")
    s.set_defaults(fn=cmd_stop)

    s = sub.add_parser("restart", help="restart with the same params (keeps files)")
    s.add_argument("id")
    s.add_argument("--force", action="store_true")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--timeout", type=float, default=3600)
    s.set_defaults(fn=cmd_restart)

    s = sub.add_parser("logs", help="show deployment logs")
    s.add_argument("id")
    s.add_argument("-f", "--follow", action="store_true")
    s.add_argument("-n", "--tail", type=int, default=0)
    s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("endpoints", help="copy-pasteable connection details for ready models")
    s.add_argument("id", nargs="?")
    s.set_defaults(fn=cmd_endpoints)

    s = sub.add_parser("test", help="run the smoke suite against a ready deployment")
    s.add_argument("id")
    s.add_argument("--caps", help="comma-separated capabilities to test (default: the recipe's)")
    s.add_argument("--max-tokens", type=int, default=512)
    s.add_argument("--public", action="store_true", help="go through the public proxy URL instead of MP_URL")
    s.add_argument("--json", action="store_true")
    s.add_argument("--save", metavar="DIR", help="image models: save generated images here")
    s.add_argument("--size", type=int, default=512, help="image models: test image size")
    s.set_defaults(fn=cmd_test)

    s = sub.add_parser("gen", help="generate an image (image models)")
    s.add_argument("model", help="served_name of a ready image deployment")
    s.add_argument("prompt")
    s.add_argument("-o", "--out", default="out.png")
    s.add_argument("--size", default="1024x1024")
    s.add_argument("-n", type=int, default=1)
    s.add_argument("--negative-prompt", dest="negative_prompt")
    s.add_argument("--seed", type=int)
    s.add_argument("--steps", type=int)
    s.add_argument("--cfg", type=float)
    s.add_argument("--lora", action="append", default=[], metavar="NAME[:STRENGTH]")
    s.add_argument("--image", help="input image for edit models")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="recipe param override, e.g. fast=true")
    s.set_defaults(fn=cmd_gen)

    a = p.parse_args()
    with _client() as c:
        a.fn(c, a)


if __name__ == "__main__":
    main()
