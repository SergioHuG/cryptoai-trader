#!/bin/bash
# CryptoAI Trader — First-time local setup
# Run: bash scripts/setup.sh

set -e

echo "🚀 Setting up CryptoAI Trader development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Check Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✅ Poetry: $(poetry --version)"

# Install dependencies
echo "📦 Installing Python dependencies..."
poetry install

# Copy .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Created .env from .env.example — fill in your credentials before running"
fi

# Start Docker services
echo "🐳 Starting Docker services (PostgreSQL + Redis)..."
docker compose up -d postgres redis

# Wait for services
echo "⏳ Waiting for services to be ready..."
sleep 5

# Run database migrations
echo "🗄️  Running database migrations..."
poetry run alembic upgrade head

# Run tests to verify setup
echo "🧪 Running tests to verify setup..."
poetry run pytest tests/ -v --no-cov

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your Kraken API keys and Telegram credentials"
echo "  2. Run the full stack: docker compose up"
echo "  3. Start development: poetry run uvicorn api.main:app --reload"
