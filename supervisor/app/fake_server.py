"""Fake OpenAI-compatible engine for dev/tests (engine: fake). Echoes the last user message."""

import argparse
import asyncio
import json
import sys
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


def build_app(model: str) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": model, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        last = body["messages"][-1]["content"]
        if isinstance(last, list):  # vision-style content parts
            n_img = sum(1 for p in last if p.get("type") == "image_url")
            last = " ".join(p.get("text", "") for p in last if p.get("type") == "text") + f" [{n_img} image(s)]"
        text = f"echo({body.get('model')}): {last}"
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        tool_calls = None
        if body.get("tools"):
            fn = body["tools"][0]["function"]["name"]
            tool_calls = [{"id": "call_1", "type": "function", "function": {"name": fn, "arguments": "{}"}}]

        if not body.get("stream"):
            msg = {"role": "assistant", "content": None if tool_calls else text, "reasoning_content": "thinking..."}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return {
                "id": cid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model"),
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": len(text.split()), "total_tokens": 5 + len(text.split())},
            }

        async def gen():
            for word in text.split(" "):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": body.get("model"),
                    "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fail", action="store_true", help="exit immediately (tests startup failure)")
    a = p.parse_args()
    if a.fail:
        print("fake engine: failing on purpose", flush=True)
        sys.exit(3)
    uvicorn.run(build_app(a.model), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
