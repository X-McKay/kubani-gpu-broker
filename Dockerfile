FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
RUN useradd --uid 10001 --create-home broker
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src
ENV PATH="/app/.venv/bin:$PATH" \
    CONFIG_PATH=/etc/gpu-broker/config.yaml
USER broker
EXPOSE 8080
CMD ["uvicorn", "--factory", "kubani_gpu_broker:create_app", "--host", "0.0.0.0", "--port", "8080"]
