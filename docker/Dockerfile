# Generic base image for the local backend. Python environments are created at run time
# on the mounted storage root by jobs/setup_envs.sh, so this image stays small and reusable.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/work/uv-python \
    UV_CACHE_DIR=/work/uv-cache

WORKDIR /workspace
CMD ["bash"]
