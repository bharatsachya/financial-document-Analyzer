"""Graph Repository for Neo4j Factual Data access.

Provides async interface to Neo4j for querying client fact data.
Uses the comprehensive Cypher query that traverses the complete client graph:
Client -> RiskProfile, Goals, IncomeSource, Liability, Account -> Asset

CRITICAL ERROR HANDLING:
If Neo4j is unreachable, catches neo4j.exceptions.ServiceUnavailable
and returns a safe fallback dictionary structure so the pipeline doesn't crash.
"""

import logging
from typing import Any

import neo4j
from neo4j import AsyncDriver, AsyncGraphDatabase

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Safe Fallback Data
# =============================================================================

_SAFE_FALLBACK_FACT_FIND: dict[str, Any] = {
    "client_id": "unavailable",
    "risk_profile": {
        "id": "fallback",
        "profile_type": "Not Available",
        "risk_tolerance": "Unknown",
        "investment_horizon": "Unknown",
    },
    "goals": [],
    "income_sources": [],
    "liabilities": [],
    "accounts": [],
    "assets": [],
    "_fallback": True,
    "_error": "Graph database unavailable",
}


class GraphRepository:
    """Repository for Neo4j graph database operations.

    Manages async connection to Neo4j and provides methods
    for querying client fact data with proper error handling.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        driver: AsyncDriver | None = None,
    ) -> None:
        """Initialize GraphRepository.

        Args:
            settings: Optional application settings. Defaults to get_settings().
            driver: Optional pre-configured async Neo4j driver.
        """
        self._settings = settings or get_settings()
        self._driver: AsyncDriver | None = driver
        self._verified: bool = False

    async def _get_driver(self) -> AsyncDriver:
        """Get or create the async Neo4j driver.

        Lazily initializes the driver on first access.

        Returns:
            The async Neo4j driver.

        Raises:
            RuntimeError: If driver cannot be created.
        """
        if self._driver is not None:
            return self._driver

        try:
            self._driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(
                    self._settings.neo4j_user,
                    self._settings.neo4j_password,
                ),
            )
            self._verified = False
            logger.info(
                f"Neo4j driver created: {self._settings.neo4j_uri}"
            )
            return self._driver

        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}", exc_info=True)
            raise RuntimeError(f"Cannot connect to Neo4j: {e}") from e

    async def _verify_connection(self) -> bool:
        """Verify that Neo4j connection is alive.

        Returns:
            True if connection is verified, False otherwise.
        """
        if self._verified:
            return True

        driver = await self._get_driver()

        try:
            async with driver.session(database=self._settings.neo4j_database) as session:
                # Simple verification query
                result = await session.run("RETURN 1 as n")
                await result.consume()
            self._verified = True
            logger.info("Neo4j connection verified")
            return True

        except neo4j.exceptions.ServiceUnavailable:
            logger.warning("Neo4j service unavailable during verification")
            return False
        except Exception as e:
            logger.error(f"Neo4j verification failed: {e}", exc_info=True)
            return False

    async def get_client_fact_find(self, client_id: str) -> dict[str, Any]:
        """Retrieve complete client fact graph data.

        Comprehensive Cypher query that traverses:
        Client -> RiskProfile
        Client -> Goals
        Client -> IncomeSource
        Client -> Liability
        Client -> Account -> Asset

        Args:
            client_id: The client identifier to look up.

        Returns:
            Dict with client fact data structure:
            {
                "client_id": str,
                "risk_profile": dict | None,
                "goals": list[dict],
                "income_sources": list[dict],
                "liabilities": list[dict],
                "accounts": list[dict],
                "assets": list[dict],
                "_fallback": bool,  # True if data is fallback
                "_error": str | None,  # Error message if fallback
            }

            Returns safe fallback dict if Neo4j is unreachable.

        Example:
            >>> repo = GraphRepository()
            >>> facts = await repo.get_client_fact_find("client_123")
            >>> print(facts["risk_profile"]["profile_type"])
        """
        try:
            # Verify connection first
            if not await self._verify_connection():
                logger.warning(
                    f"Neo4j unavailable for client {client_id}, using fallback"
                )
                return _SAFE_FALLBACK_FACT_FIND.copy()

            driver = await self._get_driver()

            # Comprehensive Cypher query for client fact find
            query = """
            MATCH (c:Client {id: $client_id})
            OPTIONAL MATCH (c)-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
            OPTIONAL MATCH (c)-[:HAS_GOAL]->(g:Goal)
            OPTIONAL MATCH (c)-[:HAS_INCOME_SOURCE]->(inc:IncomeSource)
            OPTIONAL MATCH (c)-[:HAS_LIABILITY]->(l:Liability)
            OPTIONAL MATCH (c)-[:HAS_ACCOUNT]->(a:Account)
            OPTIONAL MATCH (a)-[:CONTAINS]->(ass:Asset)
            RETURN
                c.id as client_id,
                c.name as client_name,
                {
                    id: rp.id,
                    profile_type: rp.profile_type,
                    risk_tolerance: rp.risk_tolerance,
                    investment_horizon: rp.investment_horizon
                } as risk_profile,
                collect({
                    id: g.id,
                    name: g.name,
                    target_amount: g.target_amount,
                    target_date: g.target_date,
                    priority: g.priority
                }) as goals,
                collect({
                    id: inc.id,
                    source_type: inc.source_type,
                    amount: inc.amount,
                    frequency: inc.frequency
                }) as income_sources,
                collect({
                    id: l.id,
                    liability_type: l.liability_type,
                    amount: l.amount,
                    interest_rate: l.interest_rate,
                    maturity_date: l.maturity_date
                }) as liabilities,
                collect({
                    id: a.id,
                    account_type: a.account_type,
                    account_number: a.account_number,
                    balance: a.balance,
                    currency: a.currency
                }) as accounts,
                collect({
                    id: ass.id,
                    asset_type: ass.asset_type,
                    name: ass.name,
                    value: ass.value,
                    quantity: ass.quantity
                }) as assets
            """

            async with driver.session(
                database=self._settings.neo4j_database
            ) as session:
                result = await session.run(query, client_id=client_id)
                record = await result.single()

                if record is None:
                    logger.info(f"Client {client_id} not found in Neo4j")
                    return {
                        "client_id": client_id,
                        "risk_profile": None,
                        "goals": [],
                        "income_sources": [],
                        "liabilities": [],
                        "accounts": [],
                        "assets": [],
                        "_fallback": False,
                        "_error": "Client not found",
                    }

                return {
                    "client_id": record["client_id"],
                    "client_name": record.get("client_name", ""),
                    "risk_profile": record["risk_profile"],
                    "goals": record["goals"],
                    "income_sources": record["income_sources"],
                    "liabilities": record["liabilities"],
                    "accounts": record["accounts"],
                    "assets": record["assets"],
                    "_fallback": False,
                    "_error": None,
                }

        except neo4j.exceptions.ServiceUnavailable as e:
            logger.error(
                f"Neo4j service unavailable for client {client_id}: {e}",
                exc_info=True,
            )
            return _SAFE_FALLBACK_FACT_FIND.copy()

        except neo4j.exceptions.AuthError as e:
            logger.error(f"Neo4j authentication error: {e}", exc_info=True)
            return {
                **_SAFE_FALLBACK_FACT_FIND,
                "_error": "Authentication failed",
            }

        except neo4j.exceptions.DriverError as e:
            logger.error(f"Neo4j driver error for client {client_id}: {e}", exc_info=True)
            return _SAFE_FALLBACK_FACT_FIND.copy()

        except Exception as e:
            logger.error(
                f"Unexpected error getting client fact find {client_id}: {e}",
                exc_info=True,
            )
            return _SAFE_FALLBACK_FACT_FIND.copy()

    async def get_client_risk_profile(
        self, client_id: str
    ) -> dict[str, Any] | None:
        """Get client risk profile only.

        Args:
            client_id: The client identifier.

        Returns:
            Risk profile dict or None if not found/unavailable.
        """
        if not await self._verify_connection():
            return None

        driver = await self._get_driver()

        query = """
        MATCH (c:Client {id: $client_id})-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
        RETURN {
            id: rp.id,
            profile_type: rp.profile_type,
            risk_tolerance: rp.risk_tolerance,
            investment_horizon: rp.investment_horizon
        } as risk_profile
        """

        try:
            async with driver.session(
                database=self._settings.neo4j_database
            ) as session:
                result = await session.run(query, client_id=client_id)
                record = await result.single()

                if record:
                    logger.info(f"Retrieved risk profile for client {client_id}")
                    return record["risk_profile"]

                logger.info(f"No risk profile found for client {client_id}")
                return None

        except neo4j.exceptions.ServiceUnavailable as e:
            logger.error(
                f"Neo4j unavailable getting risk profile for {client_id}: {e}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(f"Error getting risk profile: {e}", exc_info=True)
            return None

    async def get_client_goals(self, client_id: str) -> list[dict[str, Any]]:
        """Get client goals.

        Args:
            client_id: The client identifier.

        Returns:
            List of goal dicts, empty if not found/unavailable.
        """
        if not await self._verify_connection():
            return []

        driver = await self._get_driver()

        query = """
        MATCH (c:Client {id: $client_id})-[:HAS_GOAL]->(g:Goal)
        RETURN {
            id: g.id,
            name: g.name,
            target_amount: g.target_amount,
            target_date: g.target_date,
            priority: g.priority,
            status: g.status
        } as goal
        ORDER BY g.priority DESC
        """

        try:
            async with driver.session(
                database=self._settings.neo4j_database
            ) as session:
                result = await session.run(query, client_id=client_id)
                records = await result.data()

                goals = [record["goal"] for record in records]
                logger.info(f"Retrieved {len(goals)} goals for client {client_id}")
                return goals

        except neo4j.exceptions.ServiceUnavailable as e:
            logger.error(
                f"Neo4j unavailable getting goals for {client_id}: {e}",
                exc_info=True,
            )
            return []
        except Exception as e:
            logger.error(f"Error getting goals: {e}", exc_info=True)
            return []

    async def close(self) -> None:
        """Close the Neo4j driver.

        Should be called when shutting down the application
        or when the repository is no longer needed.
        """
        if self._driver is not None:
            try:
                await self._driver.close()
                self._driver = None
                self._verified = False
                logger.info("Neo4j driver closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}", exc_info=True)


# =============================================================================
# Global Instance for Convenience
# =============================================================================

_global_repository: GraphRepository | None = None


async def get_graph_repository() -> GraphRepository:
    """Get or create the global GraphRepository instance.

    Returns:
        The singleton GraphRepository instance.
    """
    global _global_repository
    if _global_repository is None:
        _global_repository = GraphRepository()
    return _global_repository
