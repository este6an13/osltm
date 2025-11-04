from typing import Optional

from sqlalchemy.orm import Session

from src.repo.v1.stations.models import Station


class StationRepository:
    """Repository for station operations."""

    def __init__(self, db: Session):
        self.db = db

    def create_station(self, code: str, name: str) -> Station:
        """
        Create a new station in the database.

        Args:
            code: Station code (must be unique)
            name: Station name

        Returns:
            The created Station object

        Raises:
            ValueError: If station code already exists
        """
        # Check if station with this code already exists
        existing_station = self.db.query(Station).filter(Station.code == code).first()
        if existing_station:
            raise ValueError(f"Station with code '{code}' already exists")

        # Create new station
        station = Station(code=code, name=name)
        self.db.add(station)
        self.db.commit()
        self.db.refresh(station)

        return station

    def get_station_by_code(self, code: str) -> Optional[Station]:
        """
        Get a station by its code.

        Args:
            code: Station code

        Returns:
            Station object if found, None otherwise
        """
        return self.db.query(Station).filter(Station.code == code).first()

    def get_station_by_id(self, station_id: int) -> Optional[Station]:
        """
        Get a station by its ID.

        Args:
            station_id: Station ID

        Returns:
            Station object if found, None otherwise
        """
        return self.db.query(Station).filter(Station.id == station_id).first()

    def get_all_stations(self):
        """
        Get all stations.

        Returns:
            List of all Station objects
        """
        return self.db.query(Station).all()
