from sqlalchemy import BigInteger, Integer
from sqlalchemy.orm import DeclarativeBase


BIGINT_PK = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass

