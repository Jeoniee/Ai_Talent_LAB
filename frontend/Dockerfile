# frontend/Dockerfile
FROM python:3.11-slim
#FROM ubuntu:latest
LABEL authors="jeoniee"

#ENTRYPOINT ["top", "-b"]

WORKDIR /app
COPY ./ /app

RUN pip install --no-cache-dir streamlit requests

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]