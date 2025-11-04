from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    func,
)

from src.db.config import Base


class ProcessedFile(Base):
    """
    General model for tracking processed files of any type.

    Example:
        filename: "20251014"
        process_type: "check_ins" | "estimates" | "aggregations" | etc.
    """

    __tablename__ = "processed_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(100), nullable=False)
    process_type = Column(String(50), nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime, default=None)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("filename", "process_type", name="uq_processed_file_type"),
    )

    def __repr__(self):
        return (
            f"<ProcessedFile(filename='{self.filename}', "
            f"process_type='{self.process_type}', processed={self.processed})>"
        )
