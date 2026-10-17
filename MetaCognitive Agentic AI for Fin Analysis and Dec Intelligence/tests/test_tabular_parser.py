"""Unit tests for financial tabular parsing and formatting."""

from metacog_finance.rag.tabular_parser import FinancialTableParser, parse_accounting_number


def test_parse_accounting_number():
    assert parse_accounting_number("123.45") == 123.45
    assert parse_accounting_number("$1,234.56") == 1234.56
    assert parse_accounting_number("(500.0)") == -500.0
    assert parse_accounting_number("-250.5") == -250.5
    assert parse_accounting_number("15.5%") == 0.155
    assert parse_accounting_number("—") == 0.0
    assert parse_accounting_number(None) == 0.0


def test_markdown_table_conversion():
    data = {"revenue": 383285.0, "net_income": 96995.0}
    md = FinancialTableParser.dict_to_markdown_table(data, "Sample")
    assert "| Revenue | $383,285.0M |" in md
    assert "| Net Income | $96,995.0M |" in md


def test_extract_tables_from_markdown():
    text = """
Some narrative context...

| Item | Amount |
| :--- | :--- |
| Sales | $100M |
| Expenses | $80M |

Another paragraph.
"""
    dfs = FinancialTableParser.extract_tables_from_markdown(text)
    assert len(dfs) == 1
    assert dfs[0].shape == (2, 2)
    assert dfs[0].iloc[0]["Item"] == "Sales"
