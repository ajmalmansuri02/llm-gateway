# Override to use a registry mirror, e.g. --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim
ARG PYTHON_IMAGE=python:3.12-slim

# ---- build stage: resolve and install dependencies into a virtualenv ----
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# ---- runtime stage: only the virtualenv and config, running as non-root ----
FROM ${PYTHON_IMAGE} AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GATEWAY_CONFIG_PATH=/app/config/gateway.yaml

RUN groupadd --system --gid 10001 gateway \
 && useradd --system --uid 10001 --gid gateway --no-create-home --shell /usr/sbin/nologin gateway

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=gateway:gateway config ./config

USER gateway
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "llm_gateway.app:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
