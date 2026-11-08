FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/requirements-api.lock deploy/requirements-cpu.lock deploy/requirements-inference-cpu.lock /app/deploy/
RUN pip install --no-cache-dir \
    -r /app/deploy/requirements-api.lock \
    -r /app/deploy/requirements-inference-cpu.lock

COPY . /app

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin litterlog \
    && mkdir -p /var/lib/litterlog /models /run/litterlog \
    && chown -R 1000:1000 /var/lib/litterlog

USER 1000:1000
EXPOSE 8443

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "/app/deploy/healthcheck.py"]

ENTRYPOINT ["python", "-u", "/app/deploy/retained_logs.py"]
CMD ["api"]
