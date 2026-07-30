# SPDX-License-Identifier: Apache-2.0
# Multi-stage Dockerfile for fusion-mlx
# Stage 1: builder — install deps
# Stage 2: runtime — minimal image

FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY fusion_mlx/ fusion_mlx/

RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim AS runtime

LABEL maintainer="fusion-mlx"
LABEL description="MLX inference server with OpenAI-compatible API"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -m -s /bin/bash fusion
USER fusion

WORKDIR /home/fusion

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

ENTRYPOINT ["python", "-m", "fusion_mlx"]
