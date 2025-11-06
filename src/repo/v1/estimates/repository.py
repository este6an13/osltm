from typing import List, Optional

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from src.repo.v1.estimates.models import Estimate


class EstimateRepository:
    """Repository for managing λ_in and λ_out estimates efficiently."""

    def __init__(self, db: Session):
        self.db = db
        self.model = Estimate

    def create_estimate(
        self,
        year: int,
        month: int,
        day: int,
        day_of_week: int,
        time: int,
        date_type: str,
        station_id: int,
        estimated_lambda_in: Optional[float] = None,
        estimated_lambda_out: Optional[float] = None,
    ) -> Estimate:
        estimate = self.model(
            year=year,
            month=month,
            day=day,
            day_of_week=day_of_week,
            time=time,
            date_type=date_type,
            station_id=station_id,
            estimated_lambda_in=estimated_lambda_in,
            estimated_lambda_out=estimated_lambda_out,
        )
        self.db.add(estimate)
        self.db.commit()
        self.db.refresh(estimate)
        return estimate

    def get_estimate(self, **filters) -> Optional[Estimate]:
        return self.db.query(self.model).filter_by(**filters).first()

    def exists(self, **filters) -> bool:
        return self.db.query(self.model).filter_by(**filters).first() is not None

    def update_estimate_lambda_in(self, lambda_in: float, **filters):
        estimate = self.db.query(self.model).filter_by(**filters).first()
        if estimate:
            estimate.estimated_lambda_in = lambda_in
            self.db.commit()

    def update_estimate_lambda_out(self, lambda_out: float, **filters):
        estimate = self.db.query(self.model).filter_by(**filters).first()
        if estimate:
            estimate.estimated_lambda_out = lambda_out
            self.db.commit()

    def bulk_upsert_estimates(self, estimates: List[dict], batch_size: int = 800):
        """
        Efficiently bulk-insert or update Estimate rows.
        Handles both estimated_lambda_in and estimated_lambda_out fields.

        Args:
            estimates: List of dicts with Estimate columns.
            batch_size: Number of rows per DB chunk (safe for SQLite).
        """
        if not estimates:
            return

        total = len(estimates)
        for i in range(0, total, batch_size):
            chunk = estimates[i : i + batch_size]

            stmt = insert(self.model).values(chunk)

            # Determine which fields to update based on presence in first dict
            set_fields = {}
            if "estimated_lambda_in" in chunk[0]:
                set_fields["estimated_lambda_in"] = stmt.excluded.estimated_lambda_in
            if "estimated_lambda_out" in chunk[0]:
                set_fields["estimated_lambda_out"] = stmt.excluded.estimated_lambda_out

            # Perform ON CONFLICT DO UPDATE on the unique key
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
