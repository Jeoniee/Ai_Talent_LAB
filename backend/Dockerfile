# backend/Dockerfile
FROM python:3.11-slim
#FROM ubuntu:latest
LABEL authors="jeoniee"

ENTRYPOINT ["top", "-b"]

WORKDIR /app
COPY ./app /app

RUN pip install --no-cache-dir fastapi uvicorn langchain openai chromadb

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]