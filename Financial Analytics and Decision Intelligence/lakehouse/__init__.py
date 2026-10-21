"""
FLAME-AI: Medallion Lakehouse Engine for Adaptive Financial Analytics
Bronze (Raw) -> Silver (Cleaned/Enriched) -> Gold (Analytics Cubes & DuckDB)
"""

from .bronze_ingestion import BronzeIngestionEngine
from .silver_transformation import SilverTransformationEngine
from .gold_aggregations import GoldAggregationEngine
from .query_engine import LakehouseQueryEngine

__all__ = [
    "BronzeIngestionEngine",
    "SilverTransformationEngine",
    "GoldAggregationEngine",
    "LakehouseQueryEngine"
]
