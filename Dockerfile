FROM python:3.13-slim

USER root

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/.cache/pip pip install ipython ipdb memray
RUN apt-get update && apt-get install --no-install-recommends -y \
	libvips-tools \
    libvips \
    libvips-dev \
    build-essential

RUN mkdir -p /app
COPY src/ /app/src/

COPY pyproject.toml setup.py LICENSE.txt README.rst /app/

RUN --mount=type=cache,target=/.cache/pip cd /app && pip install '.[dev,test]'

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Download pytorch model weights
COPY ./docker/docker_download_weights.py docker_download_weights.py
RUN python3 docker_download_weights.py


