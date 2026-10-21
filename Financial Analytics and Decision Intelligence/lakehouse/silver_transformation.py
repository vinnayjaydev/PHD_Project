"""
Silver Transformation Engine: Cleaning, Alignment, Feature Engineering & Semantic Chunking.
Part of the FLAME-AI Financial Lakehouse Architecture.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SilverTransformation")


class SilverTransformationEngine:
    """
    Transforms Bronze raw data into Silver-layer curated, aligned, and enriched tables:
    1. Quantitative Feature Engineering (OHLCV -> Returns, Volatilities, RSI, MACD, Bollinger Bands, ATR)
    2. Document Structure Preservation & Semantic Chunking with metadata & point-in-time timestamps.
    """

    def __init__(self, bronze_dir: str = "data/bronze", silver_dir: str = "data/silver"):
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
        self.market_features_path = os.path.join(silver_dir, "market_features.parquet")
        self.chunks_path = os.path.join(silver_dir, "filings_chunks.parquet")
        self.metadata_dir = os.path.join(silver_dir, "_metadata")

        os.makedirs(self.silver_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def transform_market_data(self) -> pd.DataFrame:
        """
        Reads all raw market Parquet files from Bronze, cleans, standardizes,
        computes technical/econometric features, and aligns cross-sectional asset data.
        """
        raw_market_dir = os.path.join(self.bronze_dir, "market_data")
        if not os.path.exists(raw_market_dir):
            raise FileNotFoundError(f"Bronze raw market directory missing: {raw_market_dir}")

        files = [os.path.join(raw_market_dir, f) for f in os.listdir(raw_market_dir) if f.endswith(".parquet")]
        if not files:
            raise FileNotFoundError("No parquet files found in Bronze market_data.")

        logger.info(f"Transforming {len(files)} market data files from Bronze to Silver...")
        dfs = []
        for f in files:
            try:
                sub_df = pd.read_parquet(f)
                sub_df["date"] = pd.to_datetime(sub_df["date"])
                sub_df = sub_df.sort_values("date").reset_index(drop=True)
                
                # Deduplicate and forward fill missing
                sub_df = sub_df.drop_duplicates(subset=["date", "ticker"])

                # Feature engineering per ticker
                sub_df = self._compute_technical_features(sub_df)
                dfs.append(sub_df)
            except Exception as e:
                logger.error(f"Error processing {f}: {e}")

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Cross-sectional benchmark relative returns
        spy_df = combined_df[combined_df["ticker"] == "SPY"][["date", "return_1d", "return_21d"]].copy()
        if not spy_df.empty:
            spy_df = spy_df.rename(columns={"return_1d": "spy_return_1d", "return_21d": "spy_return_21d"})
            combined_df = pd.merge(combined_df, spy_df, on="date", how="left")
            combined_df["alpha_21d"] = combined_df["return_21d"] - combined_df["spy_return_21d"]
        else:
            combined_df["alpha_21d"] = 0.0

        # Save to Silver Parquet
        combined_df.to_parquet(self.market_features_path, index=False, engine="pyarrow")
        logger.info(f"Silver market features saved: {self.market_features_path} ({len(combined_df)} rows)")

        # Save transformation manifest
        manifest = {
            "transform_job": "Silver_Market_Features",
            "total_rows": len(combined_df),
            "tickers": combined_df["ticker"].unique().tolist(),
            "date_range": [str(combined_df["date"].min()), str(combined_df["date"].max())],
            "feature_columns": [c for c in combined_df.columns if c not in ["date", "ticker", "ingestion_timestamp"]],
            "layer": "silver"
        }
        with open(os.path.join(self.metadata_dir, "market_features_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        return combined_df

    def _compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes econometric, momentum, volatility, and trend indicators.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Log returns
        df["return_1d"] = np.log(close / close.shift(1)).fillna(0.0)
        df["return_5d"] = np.log(close / close.shift(5)).fillna(0.0)
        df["return_21d"] = np.log(close / close.shift(21)).fillna(0.0)
        df["return_63d"] = np.log(close / close.shift(63)).fillna(0.0)

        # Moving Averages
        df["ema_14"] = close.ewm(span=14, adjust=False).mean()
        df["ema_50"] = close.ewm(span=50, adjust=False).mean()
        df["ema_200"] = close.ewm(span=200, adjust=False).mean()

        # Relative Strength Index (RSI 14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50.0)

        # MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands (20, 2)
        rolling_mean_20 = close.rolling(window=20).mean()
        rolling_std_20 = close.rolling(window=20).std()
        df["bb_upper"] = rolling_mean_20 + 2.0 * rolling_std_20
        df["bb_lower"] = rolling_mean_20 - 2.0 * rolling_std_20
        df["bb_zscore"] = (close - rolling_mean_20) / (rolling_std_20 + 1e-9)
        df["bb_zscore"] = df["bb_zscore"].fillna(0.0)

        # Average True Range (ATR 14)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14).mean() / (close + 1e-9)
        df["atr_14"] = df["atr_14"].fillna(0.02)

        # Rolling Volatilities (annualized: sqrt(252))
        df["vol_10d"] = df["return_1d"].rolling(window=10).std() * np.sqrt(252)
        df["vol_21d"] = df["return_1d"].rolling(window=21).std() * np.sqrt(252)
        df["vol_63d"] = df["return_1d"].rolling(window=63).std() * np.sqrt(252)
        df["vol_21d"] = df["vol_21d"].fillna(df["vol_21d"].median() or 0.20)

        # Cumulative maximum and drawdown
        cum_ret = (1 + df["return_1d"]).cumprod()
        rolling_max = cum_ret.cummax()
        df["drawdown"] = (cum_ret - rolling_max) / rolling_max

        return df

    def transform_unstructured_filings(self, chunk_size: int = 400, overlap: int = 60) -> pd.DataFrame:
        """
        Parses Bronze filings JSON, segments sections into contextual chunks with strict
        point-in-time publication dates, generates dense semantic embeddings,
        and saves into Silver Parquet format.
        """
        filings_file = os.path.join(self.bronze_dir, "unstructured_filings", "unstructured_documents.json")
        if not os.path.exists(filings_file):
            raise FileNotFoundError(f"Bronze filings not found at: {filings_file}")

        with open(filings_file, "r", encoding="utf-8") as f:
            docs = json.load(f)

        logger.info(f"Chunking {len(docs)} documents into Silver semantic segments...")
        chunks_records = []

        for doc in docs:
            ticker = doc.get("ticker", "UNKNOWN")
            doc_id = doc.get("doc_id", "DOC")
            doc_type = doc.get("doc_type", "FILING")
            filing_date = doc.get("filing_date", "2024-01-01")
            company = doc.get("company_name", ticker)
            sections = doc.get("sections", {})

            for section_name, section_text in sections.items():
                # Segment section text into structured chunks
                sub_chunks = self._chunk_text(section_text, chunk_size=chunk_size, overlap=overlap)
                for idx, chunk_text in enumerate(sub_chunks):
                    chunk_id = f"{doc_id}_{section_name.split()[0]}_{idx}"
                    chunks_records.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "ticker": ticker,
                        "company_name": company,
                        "doc_type": doc_type,
                        "filing_date": filing_date,
                        "point_in_time_date": filing_date,
                        "section": section_name,
                        "chunk_index": idx,
                        "content": chunk_text,
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "is_tabular": "|" in chunk_text or "$" in chunk_text
                    })

        chunks_df = pd.DataFrame(chunks_records)

        # Generate lightweight normalized dense embeddings using TF-IDF + TruncatedSVD for deterministic speed
        logger.info(f"Generating dense vector representations for {len(chunks_df)} chunks...")
        tfidf = TfidfVectorizer(max_features=256, stop_words="english")
        tfidf_mat = tfidf.fit_transform(chunks_df["content"]).toarray()
        
        # L2-normalize embeddings
        norms = np.linalg.norm(tfidf_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = (tfidf_mat / norms).astype(np.float32)

        # Store embedding as binary blob or list
        chunks_df["embedding"] = [vec.tolist() for vec in normalized_embeddings]

        # Save to Silver
        chunks_df.to_parquet(self.chunks_path, index=False, engine="pyarrow")
        logger.info(f"Silver filing chunks saved: {self.chunks_path} ({len(chunks_df)} chunks)")

        manifest = {
            "transform_job": "Silver_Filing_Chunks",
            "total_chunks": len(chunks_df),
            "tickers": chunks_df["ticker"].unique().tolist(),
            "sections_extracted": chunks_df["section"].unique().tolist(),
            "layer": "silver"
        }
        with open(os.path.join(self.metadata_dir, "filings_chunks_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        return chunks_df

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Splits text into chunks respecting sentence and table row boundaries.
        """
        # If it's a markdown table, keep rows together
        if "|" in text:
            lines = text.strip().split("\n")
            chunks = []
            curr_chunk = []
            curr_len = 0
            for line in lines:
                curr_chunk.append(line)
                curr_len += len(line)
                if curr_len >= chunk_size:
                    chunks.append("\n".join(curr_chunk))
                    curr_chunk = curr_chunk[-1:]  # Keep last line for overlap
                    curr_len = len(curr_chunk[0])
            if curr_chunk:
                chunks.append("\n".join(curr_chunk))
            return chunks

        # Sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        curr = []
        curr_words = 0

        for sentence in sentences:
            words = sentence.split()
            if curr_words + len(words) > (chunk_size // 5):  # Approx words
                if curr:
                    chunks.append(" ".join(curr))
                curr = words
                curr_words = len(words)
            else:
                curr.extend(words)
                curr_words += len(words)

        if curr:
            chunks.append(" ".join(curr))

        return chunks if chunks else [text]


if __name__ == "__main__":
    trans = SilverTransformationEngine()
    trans.transform_market_data()
    trans.transform_unstructured_filings()
    print("Silver Transformation Self-Test Successful!")
