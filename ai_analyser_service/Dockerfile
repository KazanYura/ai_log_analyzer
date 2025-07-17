FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install poetry

COPY pyproject.toml poetry.lock* README.md ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root

COPY src/ ./src/

RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    model_name = 'microsoft/DialoGPT-medium'; \
    print(f'Downloading DialoGPT model: {model_name}'); \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForCausalLM.from_pretrained(model_name); \
    print('DialoGPT model cached successfully')" && \
    python -c "try: \
        from sentence_transformers import SentenceTransformer; \
        print('Downloading sentence-transformers model for RAG embeddings'); \
        model = SentenceTransformer('all-MiniLM-L6-v2'); \
        print('Sentence-transformers model cached successfully'); \
    except ImportError: \
        print('sentence-transformers not available, will use fallback similarity'); \
    except Exception as e: \
        print(f'Warning: Could not cache sentence-transformers model: {e}'); \
        print('Will use fallback similarity method')" && \
    python -c "from transformers import AutoTokenizer; \
        print('Downloading BERT tokenizer for fallback similarity'); \
        AutoTokenizer.from_pretrained('bert-base-uncased'); \
        print('BERT tokenizer cached successfully')"

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]