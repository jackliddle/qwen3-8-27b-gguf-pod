# model-pod: one image, three engines (vLLM, llama.cpp, Ollama) + supervisor/web UI.
#
# CUDA: vLLM v0.30.0 ships CUDA 13.0; llama.cpp's server-cuda13 image is built
# against 13.3. llama.cpp gets its own copy of the CUDA libs it links against
# (in /opt/llama.cpp/cuda, only on its LD_LIBRARY_PATH via the wrapper) so the
# two never fight over cuBLAS versions. Ollama bundles its own libs in
# /usr/lib/ollama. Pods need a host driver that supports CUDA 13.x.

ARG VLLM_TAG=v0.30.0-ubuntu2404
ARG LLAMACPP_TAG=server-cuda13-b10524
ARG OLLAMA_TAG=0.35.0

FROM ghcr.io/ggml-org/llama.cpp:${LLAMACPP_TAG} AS llamacpp
RUN mkdir -p /out/cuda && \
    cp -a /usr/local/cuda/lib64/libcudart.so.13* /usr/local/cuda/lib64/libcublas.so.13* \
          /usr/local/cuda/lib64/libcublasLt.so.13* /out/cuda/

FROM ollama/ollama:${OLLAMA_TAG} AS ollama

FROM node:22-slim AS ui
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM vllm/vllm-openai:${VLLM_TAG}

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 libcurl4 libssl3 openssh-server curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /run/sshd

# llama.cpp: binaries + ggml backends (loaded from the executable's dir) + its CUDA libs.
COPY --from=llamacpp /app/ /opt/llama.cpp/app/
COPY --from=llamacpp /out/cuda/ /opt/llama.cpp/cuda/
RUN mkdir -p /opt/llama.cpp/bin && \
    printf '#!/bin/sh\nexport LD_LIBRARY_PATH=/opt/llama.cpp/app:/opt/llama.cpp/cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\nexec /opt/llama.cpp/app/llama-server "$@"\n' \
      > /opt/llama.cpp/bin/llama-server && \
    chmod +x /opt/llama.cpp/bin/llama-server && \
    ! LD_LIBRARY_PATH=/opt/llama.cpp/app:/opt/llama.cpp/cuda ldd /opt/llama.cpp/app/llama-server | grep -E "not found|version .* not found"

# Ollama
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama

# Supervisor in its own venv so its deps never collide with vLLM's.
WORKDIR /opt/model-pod
COPY supervisor/pyproject.toml supervisor/poetry.lock supervisor/README.md ./supervisor/
COPY supervisor/app ./supervisor/app
RUN uv venv /opt/supervisor-venv --python 3.12 && \
    env -u UV_OVERRIDE VIRTUAL_ENV=/opt/supervisor-venv uv pip install --no-cache -e ./supervisor
COPY supervisor/dev-recipes ./supervisor/dev-recipes
COPY recipes ./recipes
COPY --from=ui /src/frontend/dist ./frontend/dist
COPY start.sh /start.sh
RUN chmod +x /start.sh && ln -s /opt/supervisor-venv/bin/mp /usr/local/bin/mp

ENV MODELS_DIR=/workspace/models \
    LOCAL_RECIPES_DIR=/workspace/recipes \
    RECIPES_DIR=/opt/model-pod/recipes \
    STATIC_DIR=/opt/model-pod/frontend/dist \
    LLAMA_SERVER_BIN=/opt/llama.cpp/bin/llama-server \
    HF_HOME=/workspace/.hf \
    HF_XET_HIGH_PERFORMANCE=1 \
    MP_URL=http://localhost:8000

EXPOSE 8000 22
ENTRYPOINT ["/start.sh"]
