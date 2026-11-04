# check=skip=JSONArgsRecommended
# O CMD fica em formato shell de propósito: é o que expande PORT e WEB_CONCURRENCY.

# ── Build: instala as dependências de runtime num venv ───────
FROM python:3.11-slim AS build

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --disable-pip-version-check -r /tmp/requirements.txt

# ── Final: venv, código e modelos, rodando com usuário não-root ──
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3

RUN useradd --create-home --uid 10001 libras

WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY run_backend.py ./
COPY migrations/ migrations/
COPY src/ src/
COPY artifacts/models/best_model.keras artifacts/models/metadata.json artifacts/models/
COPY artifacts/models/letters/ artifacts/models/letters/

USER libras

EXPOSE 8080

# start-period: cada worker carrega o TensorFlow e os dois modelos ao subir.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/health', timeout=4)"

# exec: o gunicorn vira o PID 1 e recebe o SIGTERM do docker stop.
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers ${WEB_CONCURRENCY:-1} --threads 4 --timeout 120 run_backend:app
