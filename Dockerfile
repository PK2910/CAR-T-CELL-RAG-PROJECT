# The RAG API in one box (README, Docker). Build and run:
#   python src/slim_index.py                    # once: copy only the live index to data/docker/qdrant
#   docker build -t cart-rag .
#   docker run --env-file .env -p 8000:8000 -v cart-rag-state:/app/state cart-rag
# The API key is passed in when the box starts (--env-file) and is never copied into it.
# The logbook, cache and usage counts live in /app/state, a volume, so they survive restarts and rebuilds.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    RAG_STATE_DIR=/app/state \
    RAG_DOCS=0

WORKDIR /app

# PyTorch's CPU-only build (the default Linux build includes GPU libraries: several GB more), then the rest at the
# versions in requirements.txt (used as constraints; the spaCy model line is a URL, which constraints cannot hold)
COPY requirements.txt requirements-api.txt ./
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && grep -v "^en_core_web_sm" requirements.txt > /tmp/constraints.txt \
 && pip install -r requirements-api.txt -c /tmp/constraints.txt

# The search model, downloaded now so the box starts without internet access to Hugging Face
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

# Code, the chunks, the index's build record, and the slim index (only the live collection)
COPY src/ src/
COPY api/ api/
COPY data/processed/chunks.json data/processed/index_info.json data/processed/
COPY data/docker/qdrant/ data/qdrant/

# Run as an ordinary user, not root
RUN useradd --create-home app && mkdir -p /app/state && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
