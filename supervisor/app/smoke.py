"""Smoke tests against the /v1 gateway, driven by a recipe's capabilities."""

import base64
import json
import struct
import time
import zlib
from dataclasses import asdict, dataclass

import httpx


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _red_png(size: int = 32) -> str:
    """A solid red PNG as a data URL, built by hand (no Pillow dependency)."""
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _message(resp: dict) -> dict:
    return resp["choices"][0]["message"]


def _reasoning(msg: dict) -> str:
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def run_smoke(base_url: str, api_key: str, model: str, capabilities: list[str], max_tokens: int = 256) -> list[Result]:
    base_url = base_url.rstrip("/")
    client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=300)
    results: list[Result] = []

    def check(name: str, fn) -> None:
        t0 = time.monotonic()
        try:
            detail = fn()
            results.append(Result(name, True, detail, time.monotonic() - t0))
        except Exception as e:
            results.append(Result(name, False, f"{type(e).__name__}: {e}", time.monotonic() - t0))

    def chat(messages: list, **extra) -> dict:
        r = client.post("/chat/completions", json={"model": model, "messages": messages, "max_tokens": max_tokens, **extra})
        if r.status_code != 200:
            raise AssertionError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def t_models() -> str:
        ids = [m["id"] for m in client.get("/models").raise_for_status().json()["data"]]
        assert model in ids, f"{model} not in {ids}"
        return f"listed: {ids}"

    def t_chat() -> str:
        t0 = time.monotonic()
        resp = chat([{"role": "user", "content": "Reply with exactly: pong"}])
        dt = time.monotonic() - t0
        msg = _message(resp)
        content = msg.get("content") or ""
        assert content.strip() or _reasoning(msg), "empty response"
        toks = (resp.get("usage") or {}).get("completion_tokens")
        rate = f", {toks / dt:.1f} tok/s" if toks else ""
        return f"{content.strip()[:80]!r} ({toks} tokens in {dt:.2f}s{rate})"

    def t_stream() -> str:
        t0 = time.monotonic()
        ttft = None
        chunks = 0
        done = False
        body = {"model": model, "messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": max_tokens, "stream": True}
        with client.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise AssertionError(f"HTTP {r.status_code}: {r.read()[:300]!r}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                choices = json.loads(data).get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                if delta.get("content") or _reasoning(delta):
                    chunks += 1
                    if ttft is None:
                        ttft = time.monotonic() - t0
        assert chunks > 0, "no content chunks"
        assert done, "stream ended without [DONE]"
        return f"{chunks} chunks, TTFT {ttft:.2f}s, total {time.monotonic() - t0:.2f}s"

    def t_vision() -> str:
        resp = chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this image? Answer in one word."},
                        {"type": "image_url", "image_url": {"url": _red_png()}},
                    ],
                }
            ]
        )
        msg = _message(resp)
        content = (msg.get("content") or "").strip()
        assert content or _reasoning(msg), "empty response"
        hint = "" if "red" in content.lower() else " (didn't say 'red')"
        return f"{content[:80]!r}{hint}"

    def t_tools() -> str:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ]
        resp = chat([{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}], tools=tools)
        calls = _message(resp).get("tool_calls") or []
        assert calls, f"no tool_calls; content={_message(resp).get('content')!r}"
        fn = calls[0]["function"]
        return f"{fn['name']}({fn['arguments']})"

    def t_reasoning() -> str:
        msg = _message(chat([{"role": "user", "content": "Is 91 prime? Think briefly, then answer."}]))
        r = _reasoning(msg)
        assert r, "no reasoning_content/reasoning field in response"
        return f"{len(r)} chars of reasoning; answer {(msg.get('content') or '').strip()[:60]!r}"

    check("models", t_models)
    check("chat", t_chat)
    check("stream", t_stream)
    if "vision" in capabilities:
        check("vision", t_vision)
    if "tools" in capabilities:
        check("tools", t_tools)
    if "reasoning" in capabilities:
        check("reasoning", t_reasoning)
    client.close()
    return results


def run_image_smoke(
    base_url: str,
    api_key: str,
    model: str,
    capabilities: list[str],
    loras: list[str],
    save_dir: str | None = None,
    size: int = 512,
) -> list[Result]:
    """Image checks: OpenAI endpoint, async job API, a LoRA, and edits — as the recipe supports."""
    from pathlib import Path

    from .pngutil import png_size, solid_png

    base_url = base_url.rstrip("/")
    root = base_url.removesuffix("/v1")
    client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=900)
    results: list[Result] = []
    out = Path(save_dir) if save_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    def check(name: str, fn) -> None:
        t0 = time.monotonic()
        try:
            detail = fn()
            results.append(Result(name, True, detail, time.monotonic() - t0))
        except Exception as e:
            results.append(Result(name, False, f"{type(e).__name__}: {e}", time.monotonic() - t0))

    def keep(name: str, png: bytes, want: tuple[int, int] | None) -> str:
        got = png_size(png)
        if want:
            assert got == want, f"got {got[0]}x{got[1]}, wanted {want[0]}x{want[1]}"
        path = ""
        if out:
            (out / f"{model}-{name}.png").write_bytes(png)
            path = f" → {out / f'{model}-{name}.png'}"
        return f"{got[0]}x{got[1]} PNG{path}"

    def openai(name: str, path: str, **kw) -> bytes:
        r = client.post(base_url + path, **kw)
        if r.status_code != 200:
            raise AssertionError(f"HTTP {r.status_code}: {r.text[:300]}")
        return base64.b64decode(r.json()["data"][0]["b64_json"])

    def job(body: dict) -> bytes:
        r = client.post(root + "/api/images/jobs", json=body)
        if r.status_code != 200:
            raise AssertionError(f"HTTP {r.status_code}: {r.text[:300]}")
        job_id = r.json()["job_id"]
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            j = client.get(f"{root}/api/images/jobs/{job_id}").json()
            if j["status"] == "done":
                return client.get(j["images"][0]["url"]).raise_for_status().content
            if j["status"] == "error":
                raise AssertionError(j["error"])
            time.sleep(1)
        raise AssertionError("job not done after 900s")

    prompt = "a red fox sitting in fresh snow, golden hour, detailed photo"
    sz = f"{size}x{size}"
    if "txt2img" in capabilities:
        check("generate", lambda: keep("generate", openai("generate", "/images/generations", json={"model": model, "prompt": prompt, "size": sz}), (size, size)))
        check("job", lambda: keep("job", job({"model": model, "prompt": "a lighthouse on a cliff at night, stormy sea", "size": sz}), (size, size)))
        if loras:
            check(
                f"lora:{loras[0]}",
                lambda: keep("lora", openai("lora", "/images/generations", json={"model": model, "prompt": prompt, "size": sz, "loras": [{"name": loras[0]}]}), (size, size)),
            )
    if "img2img" in capabilities:
        src = solid_png(size, size, (200, 30, 30))
        check(
            "edit",
            lambda: keep(
                "edit",
                openai("edit", "/images/edits", files={"image": ("in.png", src, "image/png")}, data={"model": model, "prompt": "make the whole image deep blue"}),
                None,
            ),
        )
        check(
            "edit-job",
            lambda: keep("edit-job", job({"model": model, "prompt": "add a white star in the centre", "image_b64": base64.b64encode(src).decode()}), None),
        )
    client.close()
    return results
