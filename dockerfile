
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py rag_agent.py ./
COPY books/ ./books/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
