# Dockerfile — builds the semantic cache proxy as a container image.
# Using Docker here is entirely OPTIONAL. If you just want to run the
# project locally in VS Code, skip this file and follow the README's
# "Run without Docker" instructions instead.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
