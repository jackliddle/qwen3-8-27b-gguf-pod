import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

SUPERVISOR = Path(__file__).resolve().parents[1]
REPO = SUPERVISOR.parent
API_KEY = "test-key"

# Unit tests import app.config; make sure Settings() can be built.
os.environ.setdefault("API_KEY", API_KEY)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    """A real supervisor process in dev mode (fake engine, no GPU)."""
    tmp = tmp_path_factory.mktemp("mp")
    port = _free_port()
    env = {
        **os.environ,
        "API_KEY": API_KEY,
        "MP_DEV": "1",
        "PORT": str(port),
        "MODELS_DIR": str(tmp / "models"),
        "LOCAL_RECIPES_DIR": str(tmp / "recipes"),
        "RECIPES_DIR": str(REPO / "recipes"),
        "STATIC_DIR": str(tmp / "no-ui"),
        "ENGINE_BASE_PORT": str(_free_port()),
        "RUNPOD_POD_ID": "",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=SUPERVISOR,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/api/health").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("supervisor did not start")
    yield {"base": base, "tmp": tmp}
    proc.terminate()
    proc.wait(timeout=30)


@pytest.fixture
def client(server):
    with httpx.Client(base_url=server["base"], headers={"Authorization": f"Bearer {API_KEY}"}, timeout=30) as c:
        yield c
