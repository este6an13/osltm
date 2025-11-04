from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from src.repo.v0.processing.models import (
    DailyCheckInsProcessingStatus,
    DailyCheckOutsProcessingStatus,
    ProcessedDailyCheckInsFile,
    ProcessedDailyCheckOutsFile,
)


class ProcessedDailyCheckInsFileRepository:
    """Repository for operations on processed daily check-in files."""

    def __init__(self, db: Session):
        self.db = db

    # --- File-level operations ---

    def get_or_create_file(self, filename: str) -> ProcessedDailyCheckInsFile:
        """Get or create a file record."""
        record = (
            self.db.query(ProcessedDailyCheckInsFile)
            .filter_by(filename=filename)
            .first()
        )
        if not record:
            record = ProcessedDailyCheckInsFile(filename=filename)
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
        return record

    # --- Process-level operations ---

    def mark_processed(self, filename: str, process_type: str) -> None:
        """
        Mark a given process type as processed for a file.
        """
        file_record = self.get_or_create_file(filename)

        process = (
            self.db.query(DailyCheckInsProcessingStatus)
            .filter_by(file_id=file_record.id, process_type=process_type)
            .first()
        )

        if not process:
            process = DailyCheckInsProcessingStatus(
                file_id=file_record.id,
                process_type=process_type,
                processed=True,
                processed_at=func.now(),
            )
            self.db.add(process)
        else:
            process.processed = True
            process.processed_at = func.now()

        self.db.commit()

    def is_processed(self, filename: str, process_type: str) -> bool:
        """
        Check if a specific processing type has been done for a file.
        """
        file_record = (
            self.db.query(ProcessedDailyCheckInsFile)
            .filter_by(filename=filename)
            .first()
        )
        if not file_record:
            return False

        process = (
            self.db.query(DailyCheckInsProcessingStatus)
            .filter_by(file_id=file_record.id, process_type=process_type)
            .first()
        )
        return bool(process and process.processed)

    def get_all_processes(self, filename: str):
        """Return all process statuses for a given file."""
        file_record = (
            self.db.query(ProcessedDailyCheckInsFile)
            .filter_by(filename=filename)
            .first()
        )
        return file_record.processes if file_record else []


class ProcessedDailyCheckOutsFileRepository:
    """Repository for operations on processed daily check-out files."""

    def __init__(self, db: Session):
        self.db = db

    # --- File-level operations ---

    def get_or_create_file(self, filename: str) -> ProcessedDailyCheckOutsFile:
        """Get or create a check-out file record."""
        record = (
            self.db.query(ProcessedDailyCheckOutsFile)
            .filter_by(filename=filename)
            .first()
        )
        if not record:
            record = ProcessedDailyCheckOutsFile(filename=filename)
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
        return record

    # --- Process-level operations ---

    def mark_processed(self, filename: str, process_type: str) -> None:
        """
        Mark a given process type as processed for a check-out file.
        """
        file_record = self.get_or_create_file(filename)

        process = (
            self.db.query(DailyCheckOutsProcessingStatus)
            .filter_by(file_id=file_record.id, process_type=process_type)
            .first()
        )

        if not process:
            process = DailyCheckOutsProcessingStatus(
                file_id=file_record.id,
                process_type=process_type,
                processed=True,
                processed_at=func.now(),
            )
            self.db.add(process)
        else:
            process.processed = True
            process.processed_at = func.now()

        self.db.commit()

    def is_processed(self, filename: str, process_type: str) -> bool:
        """
        Check if a specific processing type has been done for a check-out file.
        """
        file_record = (
            self.db.query(ProcessedDailyCheckOutsFile)
            .filter_by(filename=filename)
            .first()
        )
        if not file_record:
            return False

        process = (
            self.db.query(DailyCheckOutsProcessingStatus)
            .filter_by(file_id=file_record.id, process_type=process_type)
            .first()
        )
        return bool(process and process.processed)

    def get_all_processes(self, filename: str):
        """Return all process statuses for a given check-out file."""
        file_record = (
            self.db.query(ProcessedDailyCheckOutsFile)
            .filter_by(filename=filename)
            .first()
        )
        return file_record.processes if file_record else []
