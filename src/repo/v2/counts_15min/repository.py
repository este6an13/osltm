from typing import List, Optional

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from src.repo.v2.counts_15min.models import Counts15Min


class Counts15MinRepository:
    """Repository for managing 15-minute aggregated in/out counts and variances."""

    def __init__(self, db: Session):
        self.db = db
        self.model = Counts15Min

    def create_count_record(
        self,
        year: int,
        month: int,
        day: int,
        day_of_week: int,
        time: int,
        date_type: str,
        station_id: int,
        count_in: Optional[int] = None,
        count_out: Optional[int] = None,
        variance_in_1min: Optional[float] = None,
    ) -> Counts15Min:
        """
        Create a single 15-minute count record for a given station and time.
        """
        record = self.model(
            year=year,
            month=month,
            day=day,
            day_of_week=day_of_week,
            time=time,
            date_type=date_type,
            station_id=station_id,
            count_in=count_in,
            count_out=count_out,
            variance_in_1min=variance_in_1min,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_count_record(self, **filters) -> Optional[Counts15Min]:
        """
        Retrieve a single count record matching the given filters.
        """
        return self.db.query(self.model).filter_by(**filters).first()

    def exists(self, **filters) -> bool:
        """
        Check whether a count record exists for the given filters.
        """
        return self.db.query(self.model).filter_by(**filters).first() is not None

    def update_counts(
        self, count_in: Optional[int] = None, count_out: Optional[int] = None, **filters
    ):
        """
        Update count_in and/or count_out fields for a given record.
        """
        record = self.db.query(self.model).filter_by(**filters).first()
        if record:
            if count_in is not None:
                record.count_in = count_in
            if count_out is not None:
                record.count_out = count_out
            self.db.commit()

    def update_variance(self, variance_in_1min: float, **filters):
        """
        Update variance_in_1min for a given record.
        """
        record = self.db.query(self.model).filter_by(**filters).first()
        if record:
            record.variance_in_1min = variance_in_1min
            self.db.commit()

    def bulk_upsert_counts(self, records: List[dict], batch_size: int = 800):
        """
        Efficiently bulk-insert or update count rows for 15-minute windows.

        Args:
            records: List of dicts matching Counts15Min columns.
            batch_size: Number of rows per DB chunk (safe for SQLite).
        """
        if not records:
            return

        total = len(records)
        for i in range(0, total, batch_size):
            chunk = records[i : i + batch_size]

            stmt = insert(self.model).values(chunk)

            # Determine which fields to update based on what's provided
            set_fields = {}
            if "count_in" in chunk[0]:
                set_fields["count_in"] = stmt.excluded.count_in
            if "count_out" in chunk[0]:
                set_fields["count_out"] = stmt.excluded.count_out
            if "variance_in_1min" in chunk[0]:
                set_fields["variance_in_1min"] = stmt.excluded.variance_in_1min

            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    "year",
                    "month",
                    "day",
                    "day_of_week",
                    "time",
                    "date_type",
                    "station_id",
                ],
                set_=set_fields,
            )

            self.db.execute(stmt)

        self.db.commit()
