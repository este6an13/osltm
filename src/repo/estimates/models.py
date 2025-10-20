from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from src.db.config import Base


class Estimate(Base):
    """
    Model for storing lambda estimates.
    """

    __tablename__ = "estimates"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Time dimensions
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    week_of_month = Column(Integer, nullable=False)  # 1, 2, 3, 4, 5
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 6=Sunday
    time = Column(Integer, nullable=False)  # e.g., 400 → 04:00, 2300 → 23:00

    # Holiday indicator
    is_holiday = Column(Boolean, nullable=False, default=False)

    # Foreign key to station
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False)

    # Lambda estimates
    estimated_lambda_in = Column(Float, nullable=True)
    estimated_lambda_out = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())

    # Relationships
    station = relationship("Station", back_populates="estimates")

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint(
            "station_id",
            "year",
            "month",
            "week_of_month",
            "day_of_week",
            "time",
            "is_holiday",
            name="uq_estimate_station_time",
        ),
        Index("ix_estimates_station_time", "station_id", "time"),
        Index("ix_estimates_day_of_week", "day_of_week"),
    )

    def __repr__(self):
        return (
            f"<Estimate(id={self.id}, station_id={self.station_id}, time={self.time})>"
        )
