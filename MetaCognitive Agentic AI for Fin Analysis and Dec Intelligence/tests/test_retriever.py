"""Unit tests for the Hybrid Financial Retriever."""

from metacog_finance.core.models import DocumentChunk, DocumentSection
from metacog_finance.rag.hybrid_retriever import HybridFinancialRetriever


def test_hybrid_retriever_keyword_and_semantic():
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            ticker="AAPL",
            fiscal_year=2023,
            section=DocumentSection.ITEM_7_MDNA,
            content="Total net sales were $383,285 million in fiscal 2023. Gross margin was 44.1%.",
            is_tabular=False,
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            ticker="AAPL",
            fiscal_year=2023,
            section=DocumentSection.ITEM_1A_RISK_FACTORS,
            content="Global supply chain disruption and semiconductor availability represent material risks.",
            is_tabular=False,
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            ticker="AAPL",
            fiscal_year=2023,
            section=DocumentSection.ITEM_8_FINANCIAL_STATEMENTS,
            content="| Revenue | $383,285.0M |\n| Net Income | $96,995.0M |",
            is_tabular=True,
        ),
    ]

    retriever = HybridFinancialRetriever(chunks)

    # 1. Query for sales and margin
    results = retriever.retrieve("net sales gross margin", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_1"

    # 2. Query with tabular filter
    tab_results = retriever.retrieve("revenue net income", top_k=1, require_tabular=True)
    assert len(tab_results) == 1
    assert tab_results[0].chunk_id == "chunk_3"

    # 3. Query with section filter
    risk_results = retriever.retrieve("supply chain disruption", top_k=1, section_filter=DocumentSection.ITEM_1A_RISK_FACTORS)
    assert len(risk_results) == 1
    assert risk_results[0].chunk_id == "chunk_2"
