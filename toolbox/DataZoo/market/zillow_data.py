"""Zillow ZIP-level ZHVI / ZHVF data loaders."""

from __future__ import annotations

import pandas as pd


class ZillowZipData:
    """Download and normalize Zillow ZIP Code ZHVI and ZHVF CSVs (All Homes)."""

    ZHVI_URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/" "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    ZHVF_URL = "https://files.zillowstatic.com/research/public_csvs/zhvf_growth/" "Zip_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"

    @staticmethod
    def _normalize_zip_frame(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["RegionID"] = df["RegionID"].astype(str)
        df["RegionName"] = df["RegionName"].astype(str).str.zfill(5)
        return df

    @staticmethod
    def load_zhvi() -> pd.DataFrame:
        """Smoothed, seasonally adjusted ZIP ZHVI — All Homes, mid-tier."""
        return ZillowZipData._normalize_zip_frame(pd.read_csv(ZillowZipData.ZHVI_URL, low_memory=False))

    @staticmethod
    def load_zhvf() -> pd.DataFrame:
        """ZIP ZHVF: All Homes mid-tier, smoothed & seasonally adjusted growth forecasts."""
        return ZillowZipData._normalize_zip_frame(pd.read_csv(ZillowZipData.ZHVF_URL, low_memory=False))
