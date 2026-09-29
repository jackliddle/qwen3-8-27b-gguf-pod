"""Image stack end to end against app.fake_comfy (no GPU)."""

import base64
import socket
import time

import httpx
import pytest

from app.pngutil import png_size, solid_png

from .test_e2e_fake import wait_status


def pixel(png: bytes) -> tuple[int, int, int]:
    import zlib

    # first pixel of the first row of our uncompressed-filter solid PNGs
    idat = png[png.index(b"IDAT") + 4 :]
    raw = zlib.decompressobj().decompress(idat)
    return tuple(raw[1:4])


@pytest.fixture
def image_deps(client):
    for rid in ("fake-image", "fake-edit"):
        client.post("/api/deployments", json={"recipe_id": rid}).raise_for_status()
    for rid in ("fake-image", "fake-edit"):
        d = wait_status(client, rid)
        assert d["status"] == "ready", d
    yield
    for rid in ("fake-image", "fake-edit"):
        client.post(f"/api/deployments/{rid}/stop")
        client.delete(f"/api/deployments/{rid}")


def test_generations_b64_url_batch_and_models(client, server, image_deps):
    r = client.post("/v1/images/generations", json={"model": "fake-image", "prompt": "a cat", "size": "640x480"})
    assert r.status_code == 200, r.text
    body = r.json()
    png = base64.b64decode(body["data"][0]["b64_json"])
    assert png_size(png) == (640, 480)
    assert isinstance(body["seed"], int)

    r = client.post(
        "/v1/images/generations",
        json={"model": "fake-image", "prompt": "a cat", "n": 2, "size": "256x256", "response_format": "url", "loras": [{"name": "inkwash"}]},
    )
    urls = [d["url"] for d in r.json()["data"]]
    assert len(urls) == 2
    # result files are fetchable without the API key (unguessable names)
    for u in urls:
        got = httpx.get(u.replace("http://testserver", server["base"]))
        assert got.status_code == 200 and png_size(got.content) == (256, 256)

    models = {m["id"]: m["type"] for m in client.get("/v1/models").json()["data"]}
    assert models == {"fake-image": "image", "fake-edit": "image"}


def test_errors(client, server, image_deps):
    assert client.post("/v1/images/generations", json={"model": "nope", "prompt": "x"}).status_code == 404
    r = client.post("/v1/images/generations", json={"model": "fake-image", "prompt": "x", "loras": [{"name": "neon"}]})
    assert r.status_code == 500 and "wasn't selected at deploy" in r.json()["error"]["message"]
    # chat endpoint refuses image models
    r = client.post("/v1/chat/completions", json={"model": "fake-image", "messages": []})
    assert r.status_code == 400 and r.json()["error"]["code"] == "wrong_endpoint"
    # txt2img model given an image / img2img model without one
    img = base64.b64encode(solid_png(64, 64)).decode()
    assert client.post("/api/images/jobs", json={"model": "fake-image", "prompt": "x", "image_b64": img}).status_code == 400
    assert client.post("/v1/images/generations", json={"model": "fake-edit", "prompt": "x"}).status_code == 400
    assert httpx.get(server["base"] + "/api/images/files/_/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert httpx.post(server["base"] + "/api/images/jobs", json={"model": "fake-image", "prompt": "x"}).status_code == 401


def test_async_job_and_edits(client, image_deps):
    job = client.post("/api/images/jobs", json={"model": "fake-image", "prompt": "a dog", "size": "320x320"}).json()
    for _ in range(100):
        j = client.get(f"/api/images/jobs/{job['job_id']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    assert len(j["images"]) == 1
    assert any(x["job_id"] == job["job_id"] for x in client.get("/api/images/jobs").json())

    # OpenAI multipart edit: output keeps the input size, fake "edit" paints it blue
    r = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", solid_png(200, 120, (0, 255, 0)), "image/png")},
        data={"model": "fake-edit", "prompt": "make it blue"},
    )
    assert r.status_code == 200, r.text
    out = base64.b64decode(r.json()["data"][0]["b64_json"])
    assert png_size(out) == (200, 120) and pixel(out) == (0, 0, 255)

    # same through the job API with image_b64 (data: URL tolerated)
    img = "data:image/png;base64," + base64.b64encode(solid_png(96, 64)).decode()
    job = client.post("/api/images/jobs", json={"model": "fake-edit", "prompt": "blue", "image_b64": img}).json()
    for _ in range(100):
        j = client.get(f"/api/images/jobs/{job['job_id']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j


def test_stop_cleans_links_and_files_then_daemon_exits(client, server):
    comfy = server["tmp"] / "comfy" / "models"
    client.post("/api/deployments", json={"recipe_id": "fake-image"}).raise_for_status()
    assert wait_status(client, "fake-image")["status"] == "ready"
    assert (comfy / "loras" / "fake-image" / "inkwash.safetensors").is_symlink()
    assert (comfy / "diffusion_models" / "fake-image" / "krea2_turbo_fp8_scaled.safetensors").is_symlink()
    assert not (comfy / "loras" / "fake-image" / "neon.safetensors").exists()  # not selected

    # an LLM alongside, to check both kinds coexist in /v1/models
    client.post("/api/deployments", json={"recipe_id": "fake-echo", "params": {"fake_download_mb": 0}}).raise_for_status()
    assert wait_status(client, "fake-echo")["status"] == "ready"
    types = {m["id"]: m["type"] for m in client.get("/v1/models").json()["data"]}
    assert types == {"fake-image": "image", "fake-echo": "llm"}
    client.post("/api/deployments/fake-echo/stop")
    client.delete("/api/deployments/fake-echo")

    client.post("/api/deployments/fake-image/stop").raise_for_status()
    assert not (comfy / "loras" / "fake-image").exists()
    assert not (server["tmp"] / "models" / "fake-image").exists()
    client.delete("/api/deployments/fake-image")


def test_comfy_daemon_stops_with_last_image_model(client, server):
    port = server["comfy_port"]
    client.post("/api/deployments", json={"recipe_id": "fake-image"}).raise_for_status()
    assert wait_status(client, "fake-image")["status"] == "ready"
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) == 0
    client.post("/api/deployments/fake-image/stop").raise_for_status()
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0
    client.delete("/api/deployments/fake-image")
