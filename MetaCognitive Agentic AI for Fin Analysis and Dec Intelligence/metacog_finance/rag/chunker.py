"""Financial Section and Tabular Chunking Engine.

Splits financial documents along regulatory SEC Item boundaries, ensures tables are kept intact,
and attaches rich metadata (section, fiscal year, doc type, tabular flag).
"""

from __future__ import annotations

import re
from typing import List
from metacog_finance.core.models import DocumentChunk, DocumentSection


class FinancialChunker:
    """Structure-aware chunker tailored for SEC 10-K, 10-Q, and earnings transcripts."""

    def __init__(self, target_chunk_size: int = 1200, overlap_size: int = 200):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size

    def chunk_filing(
        self,
        text: str,
        ticker: str,
        fiscal_year: int,
        doc_type: str = "10-K",
    ) -> List[DocumentChunk]:
        """Splits full SEC filing into structured chunks preserving regulatory sections."""
        chunks: List[DocumentChunk] = []

        # Regex for identifying SEC Form 10-K sections
        section_patterns = [
            (r"(?:ITEM\s+1\.\s+BUSINESS)([\s\S]*?)(?=ITEM\s+1A\.|$)", DocumentSection.ITEM_1_BUSINESS),
            (r"(?:ITEM\s+1A\.\s+RISK\s+FACTORS)([\s\S]*?)(?=ITEM\s+1B\.|ITEM\s+2\.|$)", DocumentSection.ITEM_1A_RISK_FACTORS),
            (r"(?:ITEM\s+7\.\s+MANAGEMENT['’]S\s+DISCUSSION)([\s\S]*?)(?=ITEM\s+7A\.|ITEM\s+8\.|$)", DocumentSection.ITEM_7_MDNA),
            (r"(?:ITEM\s+8\.\s+FINANCIAL\s+STATEMENTS)([\s\S]*?)(?=ITEM\s+9\.|$)", DocumentSection.ITEM_8_FINANCIAL_STATEMENTS),
            (r"(?:NOTES\s+TO\s+(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS)([\s\S]*?)(?=ITEM\s+9\.|$)", DocumentSection.FOOTNOTES),
        ]

        found_any_section = False
        for pattern, sec_enum in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                found_any_section = True
                sec_text = match.group(1).strip()
                sub_chunks = self._subdivide_text(
                    sec_text, ticker, fiscal_year, doc_type, sec_enum, start_idx=len(chunks)
                )
                chunks.extend(sub_chunks)

        # Fallback if text does not use standard 10-K ITEM headers (e.g. general earnings transcript)
        if not found_any_section or len(chunks) == 0:
            sub_chunks = self._subdivide_text(
                text, ticker, fiscal_year, doc_type, DocumentSection.GENERAL, start_idx=0
            )
            chunks.extend(sub_chunks)

        return chunks

    def _subdivide_text(
        self,
        text: str,
        ticker: str,
        fiscal_year: int,
        doc_type: str,
        section: DocumentSection,
        start_idx: int,
    ) -> List[DocumentChunk]:
        """Subdivides a section into overlapping chunks, keeping markdown tables intact."""
        chunks: List[DocumentChunk] = []
        
        # Split by double newline to preserve paragraph and table blocks
        paragraphs = text.split("\n\n")
        current_chunk_parts: List[str] = []
        current_length = 0
        chunk_counter = start_idx

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            is_table = ("|" in para) and ("\n|" in para or "-|-" in para)

            # If adding this paragraph exceeds target size and we already have content
            if current_length + para_len > self.target_chunk_size and current_chunk_parts:
                combined_content = "\n\n".join(current_chunk_parts)
                has_table = any("|" in p and "\n|" in p for p in current_chunk_parts)
                chunk_counter += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{ticker}_{fiscal_year}_{doc_type}_chunk_{chunk_counter}",
                        ticker=ticker,
                        doc_type=doc_type,
                        fiscal_year=fiscal_year,
                        section=section,
                        content=combined_content,
                        is_tabular=has_table,
                        metadata={"word_count": len(combined_content.split())},
                    )
                )
                # Keep last paragraph for overlap if it wasn't a large table
                if not is_table and len(current_chunk_parts[-1]) < self.overlap_size:
                    current_chunk_parts = [current_chunk_parts[-1], para]
                    current_length = len(current_chunk_parts[0]) + para_len
                else:
                    current_chunk_parts = [para]
                    current_length = para_len
            else:
                current_chunk_parts.append(para)
                current_length += para_len

        if current_chunk_parts:
            combined_content = "\n\n".join(current_chunk_parts)
            has_table = any("|" in p and "\n|" in p for p in current_chunk_parts)
            chunk_counter += 1
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{ticker}_{fiscal_year}_{doc_type}_chunk_{chunk_counter}",
                    ticker=ticker,
                    doc_type=doc_type,
                    fiscal_year=fiscal_year,
                    section=section,
                    content=combined_content,
                    is_tabular=has_table,
                    metadata={"word_count": len(combined_content.split())},
                )
            )

        return chunks
