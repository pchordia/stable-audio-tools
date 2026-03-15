FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    ninja-build \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/stable-audio-tools

COPY . /opt/stable-audio-tools

COPY docker/gpu-runtime/runpod-baseline-requirements.txt /tmp/runpod-baseline-requirements.txt

RUN python -m venv "$VIRTUAL_ENV" \
    && "$VIRTUAL_ENV/bin/pip" install --upgrade pip 'setuptools<81' wheel packaging ninja \
    && "$VIRTUAL_ENV/bin/pip" install -e . \
    && "$VIRTUAL_ENV/bin/pip" install -r /tmp/runpod-baseline-requirements.txt \
    && "$VIRTUAL_ENV/bin/pip" install --no-build-isolation flash-attn==2.7.3

RUN python scripts/validate_flash_attn.py

CMD ["/bin/bash"]
