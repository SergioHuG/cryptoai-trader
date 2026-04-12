"""
Database Session Management — database/session.py

Provides a SQLAlchemy engine and session factory.
All database access goes through get_session() context manager.
"""
import os
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base

logger = logging.getLogger(__name__)


def get_engine(database_url: str | None = None):
    """Create SQLAlchemy engine from environment or provided URL."""
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError(
            "DATABASE_URL must be set in environment variables."
        )
    return create_engine(url, echo=False, pool_pre_ping=True)


def create_tables(database_url: str | None = None) -> None:
    """Create all tables. Used in tests and initial setup."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    logger.info("Database tables created.")


def drop_tables(database_url: str | None = None) -> None:
    """Drop all tables. Used in tests only — never in production."""
    engine = get_engine(database_url)
    Base.metadata.drop_all(engine)
    logger.info("Database tables dropped.")


@contextmanager
def get_session(database_url: str | None = None) -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        with get_session() as session:
            session.add(record)
            session.commit()
    """
    engine = get_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
