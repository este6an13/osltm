from sqlalchemy.orm import Session

from src.repo.check_ins.models import CheckIn
from src.repo.stations.models import Station


class CheckInRepository:
    """Repository for check-in operations."""

    def __init__(self, db: Session):
        self.db = db

    def create_check_in(self, station_id: int, card_id: str, timestamp) -> CheckIn:
        """
        Create a new check-in event.

        Args:
            station_id: ID of the station
            card_id: Card identifier
            timestamp: Datetime of the check-in event

        Returns:
            The created CheckIn object

        Raises:
            ValueError: If station_id doesn't exist
        """
        # Check if station exists
        station = self.db.query(Station).filter(Station.id == station_id).first()
        if not station:
            raise ValueError(f"Station with ID {station_id} does not exist")

        # Create new check-in
        check_in = CheckIn(
            station_id=station_id,
            card_id=card_id,
            timestamp=timestamp,
        )

        self.db.add(check_in)
        self.db.commit()
        self.db.refresh(check_in)

        return check_in

    def get_check_ins_by_station(self, station_id: int):
        """
        Get all check-ins for a specific station.

        Args:
            station_id: Station ID

        Returns:
            List of CheckIn objects for the station
        """
        return self.db.query(CheckIn).filter(CheckIn.station_id == station_id).all()

    def get_check_ins_by_card_id(self, card_id: str):
        """
        Get all check-ins for a specific ID card.

        Args:
            card_id: card ID identifier

        Returns:
            List of CheckIn objects for the ID card
        """
        return self.db.query(CheckIn).filter(CheckIn.card_id == card_id).all()
