FROM python:3.13.5-slim@sha256:4c2cf9917bd1cbacc5e9b07320025bdb7cdf2df7b0ceaccb55e9dd7e30987419

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/packages:/workspace

WORKDIR /workspace
RUN apt-get update \
    && apt-get install --yes --no-install-recommends jq xz-utils \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY packages packages
COPY schemas schemas
COPY services services
COPY research research
COPY scripts scripts
COPY tests tests
COPY pyproject.toml .

RUN useradd --create-home --uid 10003 tester && chown -R tester:tester /workspace
USER tester
CMD ["pytest"]
