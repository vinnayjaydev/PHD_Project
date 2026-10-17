"""
Command-Line Interface (CLI) for Lakehouse Agent System
Supports batch runs, pipeline refreshes, SQL queries, and recommendation generation.
"""
import sys
import argparse
import json
from lakehouse_agent.lakehouse.engine import LakehouseEngine
from lakehouse_agent.lakehouse.bronze import BronzeLayer
from lakehouse_agent.lakehouse.silver import SilverLayer
from lakehouse_agent.lakehouse.gold import GoldLayer
from lakehouse_agent.vector_store.store import FinancialVectorStore
from lakehouse_agent.agents.orchestrator import FinancialAdvisorOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Lakehouse Agentic Financial Recommendation CLI"
    )
    parser.add_argument("--init", action="store_true", help="Initialize and ingest Lakehouse medallion tables")
    parser.add_argument("--investor", type=str, help="Generate recommendations for investor ID (e.g. INV001)")
    parser.add_argument("--query", type=str, help="Execute a SQL query on the Lakehouse")
    parser.add_argument("--list-tables", action="store_true", help="List all tables in the DuckDB Lakehouse")

    args = parser.parse_args()

    engine = LakehouseEngine()
    bronze = BronzeLayer(engine)
    silver = SilverLayer(engine)
    gold = GoldLayer(engine)
    vector_store = FinancialVectorStore()

    if args.init:
        print("[*] Running Lakehouse Medallion ETL Pipeline...")
        b_res = bronze.ingest_all()
        print("  -> Bronze Layer:", b_res["status"])
        s_res = silver.process_all()
        print("  -> Silver Layer:", s_res["status"])
        g_res = gold.process_all()
        print("  -> Gold Layer:", g_res["status"])
        v_count = vector_store.sync_from_lakehouse(engine)
        print(f"  -> Vector Store: Indexed {v_count} chunks.")
        print("[+] Lakehouse initialized successfully.")
        return

    if args.list_tables:
        tables = engine.list_tables()
        print(f"\n--- Lakehouse Tables ({len(tables)}) ---")
        for t in tables:
            print(f" - {t['table_name']} ({t['table_type']})")
        return

    if args.query:
        print(f"\n--- Executing SQL: {args.query} ---")
        df = engine.query_df(args.query)
        print(df.to_string())
        return

    if args.investor:
        print(f"\n[*] Orchestrating Multi-Agent Financial Recommendation for {args.investor}...")
        vector_store.sync_from_lakehouse(engine)
        orchestrator = FinancialAdvisorOrchestrator(engine=engine, vector_store=vector_store)
        report = orchestrator.run_advisory_pipeline(args.investor)

        print("\n" + "=" * 70)
        print(f"RECOMMENDATION REPORT: {report['recommendation_id']}")
        print(f"Investor: {report['investor_name']} (Risk Tier: {report['risk_tier']})")
        print(f"Primary Goal: {report['primary_goal']}")
        print(f"Compliance Status: {report['compliance']['status']}")
        print("=" * 70)
        print("\n[Target Strategic Asset Allocation]")
        for ac, weight in report['target_allocation'].items():
            print(f"  * {ac}: {weight}%")

        print("\n[Rebalancing Trade Orders]")
        for o in report['orders']:
            print(f"  - {o['action']} {o['shares']} shares of {o['ticker']} @ ${o['price']} (${o['estimated_amount']:,.2f}) -> {o['reason']}")

        print("\n[Retrieved Evidence & Citations]")
        for c in report['rag_citations']:
            print(f"  * {c}")

        print("\n[Executive Advisory Summary]")
        print(report['executive_summary'])
        print("=" * 70)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
