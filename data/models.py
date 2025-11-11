"""
Data Models
-----------
SQLAlchemy ORM models for Hugo's data layer.

TODO: Implement with SQLAlchemy 2.0 async
"""

from datetime import datetime
from typing import Optional


# TODO: Install SQLAlchemy: pip install sqlalchemy[asyncio]
# from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
# from sqlalchemy.ext.declarative import declarative_base
# from pgvector.sqlalchemy import Vector

# Base = declarative_base()


class Message:
    """
    Message model for conversation history.

    TODO: Implement as SQLAlchemy model
    """
    # id = Column(Integer, primary_key=True)
    # session_id = Column(String, nullable=False, index=True)
    # timestamp = Column(DateTime, nullable=False, default=datetime.now)
    # role = Column(String, nullable=False)
    # content = Column(Text, nullable=False)
    # embedding_vector = Column(Vector(384))
    # importance = Column(Float, default=0.5)
    # metadata = Column(JSON)
    pass


class Session:
    """
    Session model.

    TODO: Implement as SQLAlchemy model
    """
    pass


class Reflection:
    """
    Reflection model.

    TODO: Implement as SQLAlchemy model
    """
    pass


class Skill:
    """
    Skill model.

    TODO: Implement as SQLAlchemy model
    """
    pass


class Event:
    """
    Event log model.

    TODO: Implement as SQLAlchemy model
    """
    pass
