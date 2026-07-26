FROM python:3.13.5-slim@sha256:4c2cf9917bd1cbacc5e9b07320025bdb7cdf2df7b0ceaccb55e9dd7e30987419

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/packages:/app/services/orchestrator

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY packages /app/packages
COPY schemas /app/schemas
COPY services/orchestrator /app/services/orchestrator

RUN useradd --create-home --uid 10001 equinox && chown -R equinox:equinox /app
USER equinox

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--app-dir", "/app/services/orchestrator", "--host", "0.0.0.0", "--port", "8080"]
