import os

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from src.db.config import Base

# Import models so they register with Base
from src.repo.check_ins.models import CheckIn
from src.repo.estimates.models import Estimate
from src.repo.processing.models import (
    DailyCheckInsProcessingStatus,
    DailyCheckOutsProcessingStatus,
    ProcessedDailyCheckInsFile,
    ProcessedDailyCheckOutsFile,
)
from src.repo.stations.models import Station
from src.utils.logging import logger

print(
    Station,
    CheckIn,
    Estimate,
    ProcessedDailyCheckInsFile,
    ProcessedDailyCheckOutsFile,
    DailyCheckInsProcessingStatus,
    DailyCheckOutsProcessingStatus,
)

DATABASE_FILE = os.getenv("DATABASE_FILE", "osltm.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

engine = create_engine(DATABASE_URL)
SessionLocal = scoped_session(sessionmaker(autocommit=False, bind=engine))

# Create tables based on the models
Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_database():
    """Resets the database by dropping and recreating all tables."""
    logger.info("Starting database reset...")
    # Create a new session
    session = SessionLocal()
    try:
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped.")

        # Clear cached metadata to avoid any inconsistencies
        Base.metadata.clear()

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("All tables created successfully.")

    except Exception as e:
        logger.error("Error while resetting the database: %s", str(e))
    finally:
        session.close()
