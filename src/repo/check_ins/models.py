from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.db.session import Base


class CheckIn(Base):
    """
    Model for storing check-in events.
    """

    __tablename__ = "check_ins"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Event information
    timestamp = Column(DateTime, nullable=False, default=func.now())
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False)
    card_id = Column(String(50), nullable=False)

    # Relationship to station
    station = relationship("Station", back_populates="check_ins")

    def __repr__(self):
        return f"<CheckIn(id={self.id}, station_id={self.station_id}, card_id='{self.card_id}', timestamp='{self.timestamp}')>"
