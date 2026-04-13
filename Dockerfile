FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

RUN chmod +x startup.sh

# Azure Container Apps / App Service expects $PORT (default 8501 for Streamlit)
EXPOSE 8501

CMD ["./startup.sh"]
