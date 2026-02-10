FROM python:3.14-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        openssh-client \
        rsync \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY imputify/ imputify/
COPY experiments/ experiments/
RUN uv sync --frozen --no-dev

VOLUME /root/.runpod
VOLUME /root/.ssh
VOLUME /app/results

ENTRYPOINT ["uv", "run", "python", "-m", "experiments.cli"]
