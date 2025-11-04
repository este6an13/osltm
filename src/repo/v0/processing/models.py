from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.db.config import Base


class ProcessedDailyCheckInsFile(Base):
    """
    Model for tracking daily check-in CSV files.
    Each file may have multiple processing types (check-ins, estimates, etc.).
    """

    __tablename__ = "processed_daily_check_ins_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(50), unique=True, nullable=False)  # e.g., "20251014"
    created_at = Column(DateTime, nullable=False, default=func.now())

    # Relationship with per-process statuses
    processes = relationship(
        "DailyCheckInsProcessingStatus",
        back_populates="file",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<ProcessedDailyCheckInsFile(filename='{self.filename}')>"


class DailyCheckInsProcessingStatus(Base):
    """
    Tracks whether a specific processing type has been applied to a file.
    """

    __tablename__ = "daily_check_ins_processing_statuses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(
        Integer,
        ForeignKey("processed_daily_check_ins_files.id", ondelete="CASCADE"),
        nullable=False,
    )
    process_type = Column(String(50), nullable=False)  # e.g. 'check_ins', 'estimates'
    processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime, default=None)

    file = relationship("ProcessedDailyCheckInsFile", back_populates="processes")

    __table_args__ = (
        UniqueConstraint("file_id", "process_type", name="uq_file_process_type"),
    )

    def __repr__(self):
        return (
            f"<DailyCheckInsProcessingStatus(file_id={self.file_id}, "
            f"process_type='{self.process_type}', processed={self.processed})>"
        )


class ProcessedDailyCheckOutsFile(Base):
    """
    Model for tracking daily check-out CSV files.
    Each file may have multiple processing types (check-outs, estimates, etc.).
    """

    __tablename__ = "processed_daily_check_outs_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(50), unique=True, nullable=False)  # e.g., "20251014"
    created_at = Column(DateTime, nullable=False, default=func.now())

    # Relationship with per-process statuses
    processes = relationship(
        "DailyCheckOutsProcessingStatus",
        back_populates="file",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<ProcessedDailyCheckOutsFile(filename='{self.filename}')>"


class DailyCheckOutsProcessingStatus(Base):
    """
    Tracks whether a specific processing type has been applied to a check-out file.
    """

    __tablename__ = "daily_check_outs_processing_statuses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(
        Integer,
        ForeignKey("processed_daily_check_outs_files.id", ondelete="CASCADE"),
        nullable=False,
    )
    process_type = Column(String(50), nullable=False)  # e.g. 'check_outs', 'estimates'
    processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime, default=None)

    file = relationship("ProcessedDailyCheckOutsFile", back_populates="processes")

    __table_args__ = (
        UniqueConstraint(
            "file_id", "process_type", name="uq_checkout_file_process_type"
        ),
    )

    def __repr__(self):
        return (
            f"<DailyCheckOutsProcessingStatus(file_id={self.file_id}, "
            f"process_type='{self.process_type}', processed={self.processed})>"
        )
