from sqlalchemy import and_
from sqlalchemy.orm import Session

from src.repo.estimates.models import Estimate


class EstimateRepository:
    """Repository for operations on lambda estimates."""

    def __init__(self, db: Session):
        self.db = db

    def create_estimate(
        self,
        *,
        year: int,
        month: int,
        week_of_month: int,
        day_of_week: int,
        time: int,
        is_holiday: bool,
        station_id: int,
        estimated_lambda_in: float | None = None,
        estimated_lambda_out: float | None = None,
    ) -> Estimate:
        """
        Create a new estimate record.
        """
        if self.exists(
            year=year,
            month=month,
            week_of_month=week_of_month,
            day_of_week=day_of_week,
            time=time,
            is_holiday=is_holiday,
            station_id=station_id,
        ):
            raise ValueError(
                "Estimate record already exists for this time and station."
            )

        estimate = Estimate(
            year=year,
            month=month,
            week_of_month=week_of_month,
            day_of_week=day_of_week,
            time=time,
            is_holiday=is_holiday,
            station_id=station_id,
            estimated_lambda_in=estimated_lambda_in,
            estimated_lambda_out=estimated_lambda_out,
        )

        self.db.add(estimate)
        self.db.commit()
        self.db.refresh(estimate)
        return estimate

    def exists(
        self,
        *,
        year: int,
        month: int,
        week_of_month: int,
        day_of_week: int,
        time: int,
        is_holiday: bool,
        station_id: int,
    ) -> bool:
        """Check if an estimate record already exists for the given parameters."""
        record = (
            self.db.query(Estimate)
            .filter(
                and_(
                    Estimate.year == year,
                    Estimate.month == month,
                    Estimate.week_of_month == week_of_month,
                    Estimate.day_of_week == day_of_week,
                    Estimate.time == time,
                    Estimate.is_holiday == is_holiday,
                    Estimate.station_id == station_id,
                )
            )
            .first()
        )
        return record is not None

    def update_estimate_lambda_in(
        self,
        *,
        year: int,
        month: int,
        week_of_month: int,
        day_of_week: int,
        time: int,
        is_holiday: bool,
        station_id: int,
        lambda_in: float,
    ) -> None:
        """Update the estimated_lambda_in value if the record exists."""
        record = (
            self.db.query(Estimate)
            .filter(
                and_(
                    Estimate.year == year,
                    Estimate.month == month,
                    Estimate.week_of_month == week_of_month,
                    Estimate.day_of_week == day_of_week,
                    Estimate.time == time,
                    Estimate.is_holiday == is_holiday,
                    Estimate.station_id == station_id,
                )
            )
            .first()
        )

        if record:
            record.estimated_lambda_in = lambda_in
            self.db.commit()
            self.db.refresh(record)

    def update_estimate_lambda_out(
        self,
        *,
        year: int,
        month: int,
        week_of_month: int,
        day_of_week: int,
        time: int,
        is_holiday: bool,
        station_id: int,
        lambda_out: float,
    ) -> None:
        """Update the estimated_lambda_out value if the record exists."""
        record = (
            self.db.query(Estimate)
            .filter(
                and_(
                    Estimate.year == year,
                    Estimate.month == month,
                    Estimate.week_of_month == week_of_month,
                    Estimate.day_of_week == day_of_week,
                    Estimate.time == time,
                    Estimate.is_holiday == is_holiday,
                    Estimate.station_id == station_id,
                )
            )
            .first()
        )

        if record:
            record.estimated_lambda_out = lambda_out
            self.db.commit()
            self.db.refresh(record)
