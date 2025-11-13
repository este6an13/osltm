from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from src.repo.v2.processing.models import ProcessedFile


class ProcessedFileRepository:
    """Repository for operations on processed files of any type."""

    def __init__(self, db: Session):
        self.db = db

    def get_or_create(self, filename: str, process_type: str) -> ProcessedFile:
        """
        Get or create a processed file record.
        """
        record = (
            self.db.query(ProcessedFile)
            .filter_by(filename=filename, process_type=process_type)
            .first()
        )

        if not record:
            record = ProcessedFile(filename=filename, process_type=process_type)
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)

        return record

    def mark_processed(self, filename: str, process_type: str) -> None:
        """
        Mark a given file and process type as processed.
        Creates the record if it doesn't exist.
        """
        record = self.get_or_create(filename, process_type)
        record.processed = True
        record.processed_at = func.now()

        self.db.commit()
        self.db.refresh(record)

    def is_processed(self, filename: str, process_type: str) -> bool:
        """
        Check if a specific file and process type have been processed.
        """
        record = (
            self.db.query(ProcessedFile)
            .filter_by(filename=filename, process_type=process_type)
            .first()
        )
        return bool(record and record.processed)

    def get_all(self, filename: str) -> list[ProcessedFile]:
        """
        Return all process records for a given filename.
        """
        return (
            self.db.query(ProcessedFile)
            .filter_by(filename=filename)
            .order_by(ProcessedFile.process_type)
            .all()
        )

    def delete(self, filename: str, process_type: str | None = None) -> None:
        """
        Delete a processed record (optionally for a specific process type).
        """
        query = self.db.query(ProcessedFile).filter_by(filename=filename)
        if process_type:
            query = query.filter_by(process_type=process_type)

        query.delete(synchronize_session=False)
        self.db.commit()
