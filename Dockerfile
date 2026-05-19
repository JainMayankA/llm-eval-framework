FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN touch evaluator/__init__.py agents/__init__.py api/__init__.py tests/__init__.py

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
