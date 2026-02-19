# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
  apt-get update \
  && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr

COPY pyproject.toml README.md /app/

ARG INSTALL_DOCLING=false

RUN INSTALL_DOCLING="$INSTALL_DOCLING" python -c "import os, tomllib; from pathlib import Path; data = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8')); extras = data['project'].get('optional-dependencies', {}); include_docling = os.getenv('INSTALL_DOCLING', 'false').strip().lower() in {'1', 'true', 'yes', 'on'}; deps = [*data['project'].get('dependencies', []), *extras.get('ui', []), *extras.get('ai-lite', []), *(extras.get('ai-docling', []) if include_docling else [])]; seen = set(); ordered = [dep for dep in deps if not (dep in seen or seen.add(dep))]; Path('/tmp/requirements-ui-runtime.txt').write_text('\\n'.join(ordered) + '\\n', encoding='utf-8')"

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
  pip install -r /tmp/requirements-ui-runtime.txt

COPY src /app/src
COPY frontend /app/frontend

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
