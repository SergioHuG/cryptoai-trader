FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Configure Poetry — no virtual env inside container
RUN poetry config virtualenvs.create false

# Copy dependency files first (layer caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-root --no-interaction

# Copy application code
COPY . .

# Default command
CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
