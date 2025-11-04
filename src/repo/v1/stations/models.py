from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from src.db.session_v1 import Base


class Station(Base):
    """
    Model for storing station information.
    """

    __tablename__ = "stations"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Station information
    code = Column(String(50), nullable=False, unique=True)
    name = Column(String(255), nullable=False)

    # Relationships
    estimates = relationship(
        "Estimate", back_populates="station", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Station(id={self.id}, code='{self.code}', name='{self.name}')>"
