from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from src.db.config import Base


class Counts15Min(Base):
    """
    Model for storing aggregated in/out counts and variance for 15-minute windows.
    Variance is computed across 1-minute sub-intervals within each window.
    """

    __tablename__ = "counts_15min"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Time dimensions
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    day = Column(Integer, nullable=False)  # 1–31
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 6=Sunday
    time = Column(Integer, nullable=False)  # e.g., 400 → 04:00, 2300 → 23:00

    # Date type classification
    date_type = Column(String(20), nullable=False)  # WD, SA, SU, HO

    # Foreign key to station
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False)

    # Aggregated counts and variance
    count_in = Column(Integer, nullable=True)
    count_out = Column(Integer, nullable=True)
    variance_in_1min = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())

    # Relationships
    station = relationship("Station", back_populates="counts_15min")

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint(
            "station_id",
            "year",
            "month",
            "day",
            "day_of_week",
            "time",
            "date_type",
            name="uq_counts15min_station_time",
        ),
        Index("ix_counts15min_station_time", "station_id", "time"),
        Index("ix_counts15min_day_of_week", "day_of_week"),
    )

    def __repr__(self):
        return (
            f"<Counts15Min(id={self.id}, station_id={self.station_id}, "
            f"date={self.year}-{self.month:02d}-{self.day:02d}, time={self.time}, "
            f"date_type={self.date_type}, count_in={self.count_in}, count_out={self.count_out})>"
        )
