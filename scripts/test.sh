#!/bin/bash
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  python:3.11-slim \
  bash -c "pip install pytest pytest-asyncio pytest-cov pytest-mock sqlalchemy 'python-telegram-bot==21.3' ccxt httpx pandas numpy tables  pandas numpy tables scikit-learn -q && python -m pytest  -q && python -m pytest tests/ tests/research/ tests/test_import_boundary.py -v"