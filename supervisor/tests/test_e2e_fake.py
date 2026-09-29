"""Full stack (API, lifecycle, gateway, smoke suite) against the fake engine."""

import time

import httpx
import pytest

from app.smoke import run_smoke

from .conftest import API_KEY


def wait_status(client, dep_id, statuses=("ready", "failed", "stopped"), timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = client.get(f"/api/deployments/{dep_id}").json()
        if d["status"] in statuses:
            return d
        time.sleep(0.2)
    raise AssertionError(f"{dep_id} stuck in {d['status']}")


@pytest.fixture
def cleanup(client):
    yield
    for d in client.get("/api/deployments").json():
        if d["status"] not in ("stopped", "failed"):
            client.post(f"/api/deployments/{d['id']}/stop")
        client.delete(f"/api/deployments/{d['id']}")


def test_auth_required(server):
    assert httpx.get(server["base"] + "/api/recipes").status_code == 401
    assert httpx.get(server["base"] + "/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert httpx.get(server["base"] + "/api/recipes", params={"key": API_KEY}).status_code == 200


def test_recipes_include_dev_and_builtin(client):
    ids = {r["id"] for r in client.get("/api/recipes").json()}
    assert {"fake-echo", "fake-echo-2", "qwen3.8-27b-hauhau-q8", "qwen3-0.6b-vllm", "llama3.2-1b-ollama"} <= ids


def test_deploy_chat_stream_stop(client, server, cleanup):
    r = client.post("/api/deployments", json={"recipe_id": "fake-echo", "params": {"fake_download_mb": 10}})
    assert r.status_code == 200, r.text
    assert r.json()["params"] == {"fake_download_mb": 10, "fail_start": False}
    d = wait_status(client, "fake-echo")
    assert d["status"] == "ready", d
    assert d["progress"]["done_bytes"] == d["progress"]["total_bytes"] == 10_000_000
    model_dir = server["tmp"] / "models" / "fake-echo"
    assert model_dir.exists()

    # duplicate deploy rejected
    assert client.post("/api/deployments", json={"recipe_id": "fake-echo"}).status_code == 409

    models = client.get("/v1/models").json()["data"]
    assert [m["id"] for m in models] == ["fake-echo"]

    resp = client.post("/v1/chat/completions", json={"model": "fake-echo", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.json()["choices"][0]["message"]["content"] == "echo(fake-echo): hi"

    with client.stream(
        "POST", "/v1/chat/completions", json={"model": "fake-echo", "stream": True, "messages": [{"role": "user", "content": "a b c"}]}
    ) as s:
        lines = [l for l in s.iter_lines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]" and len(lines) > 3

    results = run_smoke(server["base"] + "/v1", API_KEY, "fake-echo", ["chat", "vision", "tools", "reasoning"])
    assert all(r.ok for r in results), [r for r in results if not r.ok]
    assert {r.name for r in results} == {"models", "chat", "stream", "vision", "tools", "reasoning"}

    logs = client.get("/api/deployments/fake-echo/logs").json()
    assert any("[status] ready" in e["line"] for e in logs)

    d = client.post("/api/deployments/fake-echo/stop").json()
    assert d["status"] == "stopped"
    assert not model_dir.exists(), "stop should delete model files"
    assert client.get("/v1/models").json()["data"] == []


def test_two_models_routed_by_name(client, cleanup):
    for rid in ("fake-echo", "fake-echo-2"):
        client.post("/api/deployments", json={"recipe_id": rid, "params": {"fake_download_mb": 0}}).raise_for_status()
    for rid in ("fake-echo", "fake-echo-2"):
        assert wait_status(client, rid)["status"] == "ready"
    for name in ("fake-echo", "fake-echo-2"):
        resp = client.post("/v1/chat/completions", json={"model": name, "messages": [{"role": "user", "content": "x"}]})
        assert resp.json()["choices"][0]["message"]["content"] == f"echo({name}): x"
    missing = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"


def test_stop_during_download_cancels(client, server, cleanup):
    client.post("/api/deployments", json={"recipe_id": "fake-echo", "params": {"fake_download_mb": 100}}).raise_for_status()
    wait_status(client, "fake-echo", ("downloading",))
    d = client.post("/api/deployments/fake-echo/stop").json()
    assert d["status"] == "stopped"
    assert not (server["tmp"] / "models" / "fake-echo").exists()


def test_start_failure_reported_and_restartable(client, cleanup):
    client.post("/api/deployments", json={"recipe_id": "fake-echo", "params": {"fake_download_mb": 0, "fail_start": True}}).raise_for_status()
    d = wait_status(client, "fake-echo")
    assert d["status"] == "failed"
    assert "exited with code 3" in d["error"]
    # A failed deployment can be redeployed with different params.
    client.post("/api/deployments", json={"recipe_id": "fake-echo", "params": {"fake_download_mb": 0}}).raise_for_status()
    assert wait_status(client, "fake-echo")["status"] == "ready"
    client.post("/api/deployments/fake-echo/restart").raise_for_status()
    assert wait_status(client, "fake-echo")["status"] == "ready"


def test_push_recipe_and_bad_params(client, cleanup):
    text = open(__file__.replace("test_e2e_fake.py", "../dev-recipes/fake-echo.yaml")).read()
    text = text.replace("id: fake-echo", "id: pushed-echo").replace("served_name: fake-echo", "served_name: pushed")
    r = client.post("/api/recipes", content=text, headers={"Content-Type": "text/plain"})
    assert r.status_code == 200 and r.json()["origin"] == "local"
    bad = client.post("/api/deployments", json={"recipe_id": "pushed-echo", "params": {"bogus": 1}})
    assert bad.status_code == 400 and "unknown params" in bad.json()["detail"]
    assert client.post("/api/recipes", content="id: x\nengine: nope", headers={"Content-Type": "text/plain"}).status_code == 422
    assert client.delete("/api/recipes/pushed-echo").status_code == 200


def test_sse_log_stream(client, server, cleanup):
    client.post("/api/deployments", json={"recipe_id": "fake-echo", "params": {"fake_download_mb": 0}}).raise_for_status()
    wait_status(client, "fake-echo")
    seen = []
    with httpx.stream("GET", server["base"] + "/api/deployments/fake-echo/logs/stream", params={"key": API_KEY}, timeout=10) as s:
        for line in s.iter_lines():
            if line.startswith("data:"):
                seen.append(line)
            if any("[status] ready" in l for l in seen):
                break
    assert seen
