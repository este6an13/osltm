from sqlalchemy.orm import Session

from src.repo.v2.network.models import NetworkNode, NetworkEdge
from src.utils.logging import logger


class NetworkRepository:
    """Repository for managing transmilenio network nodes and edges."""

    def __init__(self, session: Session):
        self.session = session

    def clear_all(self):
        """Delete all network nodes and edges."""
        logger.info("Clearing existing network nodes and edges")
        self.session.query(NetworkEdge).delete()
        self.session.query(NetworkNode).delete()
        self.session.commit()

    def bulk_insert_nodes(self, nodes: list[dict]):
        """Insert network nodes from a list of dicts."""
        logger.info("Bulk inserting %d network nodes", len(nodes))
        self.session.bulk_insert_mappings(NetworkNode, nodes)
        self.session.commit()

    def bulk_insert_edges(self, edges: list[dict]):
        """Insert network edges from a list of dicts."""
        logger.info("Bulk inserting %d network edges", len(edges))
        self.session.bulk_insert_mappings(NetworkEdge, edges)
        self.session.commit()

    def replace_all(self, nodes: list[dict], edges: list[dict]):
        """Full replace: clear existing data and insert new nodes and edges."""
        self.clear_all()
        self.bulk_insert_nodes(nodes)
        self.bulk_insert_edges(edges)

    def get_node_count(self) -> int:
        """Get total number of network nodes."""
        return self.session.query(NetworkNode).count()

    def get_edge_count(self) -> int:
        """Get total number of network edges."""
        return self.session.query(NetworkEdge).count()

    def has_data(self) -> bool:
        """Check if network data exists in the database."""
        return self.get_node_count() > 0

    def get_all_nodes(self) -> list[NetworkNode]:
        """Retrieve all network nodes."""
        return self.session.query(NetworkNode).all()

    def get_all_edges(self) -> list[NetworkEdge]:
        """Retrieve all network edges."""
        return self.session.query(NetworkEdge).all()
