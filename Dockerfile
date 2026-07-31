# SPDX-License-Identifier: Apache-2.0
# Multi-stage Dockerfile for fusion-mlx
#
# NOTE: MLX requires Apple Silicon (M1+) hardware. This image is useful for
# development consistency and CI on macOS Docker Desktop. It will NOT provide
# GPU acceleration on non-Apple hardware.
#
# Usage:
#   docker build -t fusion-mlx .
#   docker run -p 8897:8897 -v ~/.fusion-mlx:/home/fusion/.fusion-mlx fusion-mlx

FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY fusion_mlx/ fusion_mlx/

RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim AS runtime

LABEL maintainer="fusion-mlx"
LABEL description="MLX inference server with OpenAI/Anthropic/Ollama-compatible API"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -m -s /bin/bash fusion
USER fusion

WORKDIR /home/fusion

ENV HOST=0.0.0.0
ENV PORT=8897

EXPOSE 8897

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8897/health || exit 1

ENTRYPOINT ["fusion-mlx", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8897"]
