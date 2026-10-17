"""Tabular Parser for SEC Financial Statements and Disclosures.

Parses tabular financial disclosure data, preserves markdown formatting for LLMs,
and normalizes accounting parentheses and units for deterministic calculations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import pandas as pd


def parse_accounting_number(val: Any) -> float:
    """Parses accounting strings like '(11,245)', '$383,285', '12.5%' into float."""
    if isinstance(val, (int, float)):
        return float(val)
    if val is None:
        return 0.0

    s = str(val).strip()
    if not s or s == "-" or s == "—" or s == "N/A":
        return 0.0

    # Remove dollar signs and commas
    s = s.replace("$", "").replace(",", "").strip()

    # Handle percentage
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()

    # Handle parentheses as negative: (123.45) -> -123.45
    is_neg = False
    if s.startswith("(") and s.endswith(")"):
        is_neg = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        is_neg = True
        s = s[1:].strip()

    try:
        num = float(s)
        if is_neg:
            num = -num
        if is_pct:
            num = num / 100.0
        return num
    except ValueError:
        return 0.0


class FinancialTableParser:
    """Extracts, formats, and structures financial tables from raw text or JSON."""

    @staticmethod
    def dataframe_to_markdown(df: pd.DataFrame, title: Optional[str] = None) -> str:
        """Converts a pandas DataFrame into a formatted markdown table."""
        header = f"### {title}\n\n" if title else ""
        return header + df.to_markdown(index=False)

    @staticmethod
    def dict_to_markdown_table(data: Dict[str, Any], title: str = "Financial Items") -> str:
        """Converts a key-value dictionary of financial items into a 2-column Markdown table."""
        rows = [f"| Metric / Line Item | Reported Amount ($M) |", "| :--- | :--- |"]
        for k, v in data.items():
            if isinstance(v, (int, float)):
                val_str = f"${v:,.1f}M" if abs(v) >= 1.0 else f"{v:.4f}"
            else:
                val_str = str(v)
            clean_name = k.replace("_", " ").title()
            rows.append(f"| {clean_name} | {val_str} |")
        return f"### {title}\n" + "\n".join(rows)

    @staticmethod
    def extract_tables_from_markdown(text: str) -> List[pd.DataFrame]:
        """Extracts markdown tables from text blocks into a list of DataFrames."""
        tables = []
        lines = [line.strip() for line in text.replace("\r", "").split("\n")]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("|") and line.endswith("|") and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.startswith("|") and next_line.endswith("|") and "-" in next_line:
                    headers = [col.strip() for col in line.split("|")[1:-1]]
                    data_rows = []
                    i += 2
                    while i < len(lines) and lines[i].startswith("|") and lines[i].endswith("|"):
                        cols = [col.strip() for col in lines[i].split("|")[1:-1]]
                        if len(cols) == len(headers):
                            data_rows.append(cols)
                        i += 1
                    if data_rows and headers:
                        tables.append(pd.DataFrame(data_rows, columns=headers))
                    continue
            i += 1
        return tables
