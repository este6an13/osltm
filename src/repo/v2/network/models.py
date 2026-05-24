from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, UniqueConstraint
from sqlalchemy.sql import func

from src.db.config import Base


class NetworkNode(Base):
    """
    Model for storing transmilenio network nodes (stations).
    """

    __tablename__ = "network_nodes"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Node information
    station_id = Column(String(50), nullable=False, unique=True)
    name = Column(String(255), nullable=False)
    trazado = Column(String(50), nullable=False)
    troncal = Column(String(100), nullable=False)
    tipo = Column(Integer, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    flagged = Column(Boolean, default=False)
    flag_reason = Column(String(255), default="")
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<NetworkNode(id={self.id}, station_id='{self.station_id}', name='{self.name}')>"


class NetworkEdge(Base):
    """
    Model for storing transmilenio network edges (connections).
    """

    __tablename__ = "network_edges"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Edge information
    source = Column(String(50), nullable=False)
    target = Column(String(50), nullable=False)
    distance_m = Column(Float, nullable=False)
    trazado = Column(String(50), nullable=True)
    edge_type = Column(String(20), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    # Constraints
    __table_args__ = (
        UniqueConstraint('source', 'target', 'edge_type', name='uq_network_edge'),
    )

    def __repr__(self):
        return f"<NetworkEdge(id={self.id}, source='{self.source}', target='{self.target}', type='{self.edge_type}')>"
