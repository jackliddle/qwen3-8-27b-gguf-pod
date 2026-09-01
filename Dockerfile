FROM ghcr.io/ggml-org/llama.cpp:server-cuda13-b10524

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --break-system-packages "huggingface_hub[cli,hf_transfer]"

ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 22

ENTRYPOINT ["/entrypoint.sh"]
