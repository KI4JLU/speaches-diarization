# speaches-diarization — custom arm64 (GB10/sbsa) Speaches image, diarization-only.
#
# The official arm64 Speaches image (ghcr.io/speaches-ai/speaches:latest-cuda)
# ships without torch and pyannote (verified 2026-07-14/15) — the diarization
# code is missing entirely. This image builds Speaches from source at a pinned
# commit so that pyannote + CUDA torch (sbsa wheels from PyPI) are included.
#
# Intended for NVIDIA DGX Spark (GB10, aarch64, unified memory), but the build
# is not arch-specific: on x86_64 the same lockfile resolves the amd64 wheels.
#
# Endpoints HAWKI depends on (must survive upstream bumps — see patches/):
#   POST /v1/audio/diarization        (incl. known_speaker_names[]/references[])
#   POST /v1/audio/speech/timestamps  (silero VAD)

ARG BASE_IMAGE=nvidia/cuda:12.6.3-base-ubuntu24.04
# hadolint ignore=DL3006
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/KI4JLU/speaches-diarization"
LABEL org.opencontainers.image.description="Speaches (pinned commit) with pyannote diarization for aarch64/GB10 — diarization-only deployment"
LABEL org.opencontainers.image.licenses="MIT"

# ffmpeg: audio decoding for uploaded files; git: fetching the pinned source.
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git

# Non-root user, same convention as upstream (uid 1000 = default ubuntu user).
RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu || true
USER ubuntu
ENV HOME=/home/ubuntu \
    PATH=/home/ubuntu/.local/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv \
    UV_PYTHON_CACHE_DIR=/home/ubuntu/.cache/uv/python
WORKDIR $HOME/speaches

COPY --chown=ubuntu --from=ghcr.io/astral-sh/uv:0.10 /uv /bin/uv

# Pin the exact upstream commit. The worker's production instance reports
# version 0.8.3 but actually runs a master build (main.py hardcodes the
# version string); this commit's endpoint schemas match the worker's live
# openapi.json (verified 2026-07-15).
ARG SPEACHES_REPO=https://github.com/speaches-ai/speaches.git
ARG SPEACHES_COMMIT=993994f7984bf3fe9655b267448328cf66fccb42
RUN git init -q . && \
    git remote add origin ${SPEACHES_REPO} && \
    git fetch -q --depth 1 origin ${SPEACHES_COMMIT} && \
    git checkout -q FETCH_HEAD

# Local patches (see patches/*.patch for rationale):
#   0001: serialize diarization runs (DIARIZATION_MAX_CONCURRENCY, default 1)
#   0002: pass num_speakers/min_speakers/max_speakers through to pyannote
COPY --chown=ubuntu patches/ /tmp/patches/
RUN git apply --stat --apply /tmp/patches/0001-serialize-diarization-jobs.patch \
                             /tmp/patches/0002-speaker-count-constraints.patch

# Dependencies exactly as locked upstream (torch 2.8.0 + pyannote-audio 4.0.4).
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --compile-bytecode --no-dev

# The upstream lockfile resolves the CPU-only torch build on aarch64
# (verified in the built image: torch reports "2.8.0+cpu", CUDA unavailable).
# Swap torch/torchaudio for the CUDA sbsa wheels — same version, cu129 build,
# which supports Blackwell/GB10 (sm_121).
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv pip install --python $HOME/speaches/.venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu129 \
      "torch==2.8.0+cu129" "torchaudio==2.8.0+cu129"

# Pre-create the HF cache dir so a root-owned volume mount doesn't break writes.
RUN mkdir -p $HOME/.cache/huggingface/hub

# Register pip-installed CUDA library directories with ldconfig (same as
# upstream) so native libs outside the venv loader path can be found.
USER root
RUN find /home/ubuntu/speaches/.venv -maxdepth 7 -path "*/nvidia/*/lib" -type d \
    > /etc/ld.so.conf.d/venv-nvidia.conf && ldconfig
USER ubuntu

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV PATH="$HOME/speaches/.venv/bin:$PATH"

# Telemetry off (same as upstream image).
ENV DO_NOT_TRACK=1
ENV GRADIO_ANALYTICS_ENABLED="False"
ENV DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PYANNOTE_METRICS_ENABLED=0

# Diarization-only deployment defaults (all overridable at runtime):
# no Gradio UI, download the pyannote pipeline at startup, one GPU job at a
# time (the spark shares unified memory with the realtime STT workload).
ENV ENABLE_UI=false
ENV PRELOAD_MODELS='["pyannote/speaker-diarization-community-1"]'
ENV DIARIZATION_MAX_CONCURRENCY=1

EXPOSE 8000
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
