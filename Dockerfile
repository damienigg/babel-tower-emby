FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user. UID/GID 568 matches the `apps` user on TrueNAS Scale
# 24.10+, so on TrueNAS the container can read/write its datasets without any
# ACL fiddling. Override at deploy time with `user: "<UID>:<GID>"` (or via the
# PUID/PGID env vars in docker-compose.yml) if your host uses a different UID.
RUN groupadd --gid 568 app && useradd --create-home --uid 568 --gid 568 --shell /bin/bash app

WORKDIR /app

# CPU-only torch (~200 MB). Required by the NLLB fallback path so the
# zero-setup default provider (nllb) works on the CPU image too — not just
# on the openvino-flavored one. Pulled from the CPU-only PyPI mirror to skip
# the CUDA wheels.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    'fastapi>=0.115' \
    'uvicorn[standard]>=0.32' \
    'pydantic>=2.9' \
    'pydantic-settings>=2.6' \
    'faster-whisper>=1.1' \
    'anthropic>=0.69' \
    'openai>=1.55' \
    'httpx>=0.27' \
    'jinja2>=3.1' \
    'python-multipart>=0.0.9' \
    'transformers>=4.45' \
    'sentencepiece>=0.2'

COPY --chown=app:app app/ ./app/

# Pre-create /cache so the empty image directory is owned by `app`. A bind
# mount on top of /cache will inherit the host directory's ownership at
# runtime — see README's TrueNAS perms section.
RUN mkdir -p /cache && chown -R app:app /cache /app

ENV BABEL_CACHE_DIR=/cache \
    HF_HOME=/cache/hf \
    HOME=/home/app
VOLUME ["/cache"]

USER app
EXPOSE 8765
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
