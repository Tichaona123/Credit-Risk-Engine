FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models if not present
RUN if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then \
    echo "Models directory empty, will train on first run"; \
    fi

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
