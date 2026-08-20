FROM ghcr.io/ggml-org/llama.cpp:server-cuda13-b10524

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 22

ENTRYPOINT ["/entrypoint.sh"]
