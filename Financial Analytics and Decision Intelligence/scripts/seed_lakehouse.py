"""
Automated Lakehouse Seeder: Executes Bronze Ingestion -> Silver Transformation -> Gold Aggregations.
Ensures full reproducible datasets are loaded into the Medallion Lakehouse.
"""

import sys
import os
import time
import yaml

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lakehouse.bronze_ingestion import BronzeIngestionEngine
from lakehouse.silver_transformation import SilverTransformationEngine
from lakehouse.gold_aggregations import GoldAggregationEngine
from lakehouse.query_engine import LakehouseQueryEngine


def seed_lakehouse(force_synthetic: bool = False):
    print("=" * 80)
    print("FLAME-AI LAKEHOUSE SEEDER: MEDALLION PIPELINE EXECUTION")
    print("=" * 80)

    start_time = time.time()

    # Load config
    config_path = "configs/system_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tickers = config["universe"]["tickers"] + config["universe"]["benchmarks"] + config["universe"]["macro_tickers"]
    start_date = config["universe"]["start_date"]
    end_date = config["universe"]["end_date"]

    # 1. Bronze Layer
    print("\n[PHASE 1] Executing Bronze Raw Ingestion...")
    bronze = BronzeIngestionEngine()
    bronze.ingest_market_data(tickers, start_date=start_date, end_date=end_date, force_synthetic=force_synthetic)
    bronze.ingest_unstructured_filings()
    print("[PHASE 1] Bronze Ingestion complete.")

    # 2. Silver Layer
    print("\n[PHASE 2] Executing Silver Cleansing & Feature Engineering...")
    silver = SilverTransformationEngine()
    silver.transform_market_data()
    silver.transform_unstructured_filings()
    print("[PHASE 2] Silver Transformation complete.")

    # 3. Gold Layer
    print("\n[PHASE 3] Executing Gold Aggregation & DuckDB Catalog Registration...")
    gold = GoldAggregationEngine()
    gold.build_gold_analytics_cubes()
    print("[PHASE 3] Gold Aggregation complete.")

    # 4. Verify Catalog
    print("\n[PHASE 4] Verifying Lakehouse Catalog via DuckDB Query Engine...")
    query_engine = LakehouseQueryEngine()
    summary = query_engine.get_lakehouse_catalog_summary()
    print("\nLAKEHOUSE CATALOG SUMMARY:")
    for layer, details in summary.items():
        print(f"  * {layer.upper()}: {details}")

    elapsed = time.time() - start_time
    print(f"\n>>> MEDALLION LAKEHOUSE PIPELINE SEEDING SUCCESSFUL! (Elapsed: {elapsed:.2f}s) <<<")
    query_engine.close()


if __name__ == "__main__":
    force_synth = "--synthetic" in sys.argv
    seed_lakehouse(force_synthetic=force_synth)
