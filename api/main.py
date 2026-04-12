"""
FastAPI Application Entrypoint — api/main.py
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CryptoAI Trader",
    description="AI-driven cryptocurrency trading system with human-in-the-loop approval",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """System health check — used by Docker and monitoring."""
    return {"status": "healthy", "version": "0.1.0", "mode": "paper"}


@app.get("/")
async def root():
    return {"message": "CryptoAI Trader API", "docs": "/docs"}
