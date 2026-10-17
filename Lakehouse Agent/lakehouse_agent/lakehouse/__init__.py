"""
Lakehouse Module implementing Medallion Architecture (Bronze -> Silver -> Gold)
using DuckDB and Parquet storage.
"""
from .engine import LakehouseEngine
from .bronze import BronzeLayer
from .silver import SilverLayer
from .gold import GoldLayer
from .data_generator import FinancialDataGenerator

__all__ = [
    "LakehouseEngine",
    "BronzeLayer",
    "SilverLayer",
    "GoldLayer",
    "FinancialDataGenerator",
]
