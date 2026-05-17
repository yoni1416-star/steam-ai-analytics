"""
Stage 5C — Steam Catalog Enrichment for Live Review Intelligence.

Loads `steam_catalog_real.csv` (read-only) and joins fetched live reviews by App ID.
Does not alter Market Analytics mode or the catalog file on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Canonical columns after normalization (aligned with Market games schema where useful)
CATALOG_READ_COLUMNS = [
    "AppID",
    "Name",
    "Genres",
    "Price",
    "Positive",
    "Negative",
    "Peak CCU",
    "Release date",
]

RAW_TO_CANONICAL = {
    "AppID": "app_id",
    "Name": "game_name",
    "Genres": "genre",
    "Price": "price",
    "Positive": "positive_reviews",
    "Negative": "negative_reviews",
    "Peak CCU": "peak_players",
    "Release date": "release_date",
}


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parent / "steam_catalog_real.csv"


def _price_category(price: float) -> str:
    if price <= 0:
        return "Free"
    if price < 10:
        return "Low"
    if price < 30:
        return "Medium"
    return "High"


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    return out


def normalize_steam_catalog_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Rename / coerce types; caller supplies columns already subset."""
    df = _sanitize_columns(df)
    out = df.rename(columns={k: v for k, v in RAW_TO_CANONICAL.items() if k in df.columns})
    if "app_id" in out.columns:
        # Catalog exports sometimes quote App IDs or embed separators — coerce robustly
        aid = (
            out["app_id"]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
        out["app_id"] = pd.to_numeric(aid, errors="coerce")
        out.loc[~np.isfinite(out["app_id"]), "app_id"] = np.nan
        out = out.dropna(subset=["app_id"])
        out["app_id"] = out["app_id"].astype(np.int64)
    if "price" in out.columns:
        out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0)
    for c in ("positive_reviews", "negative_reviews", "peak_players"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(lower=0)
    if "release_date" in out.columns:
        out["release_date"] = pd.to_datetime(out["release_date"], errors="coerce")
    if "genre" in out.columns:
        out["genre"] = out["genre"].astype(str).str.strip()
        out.loc[out["genre"].isin(("", "nan", "None")), "genre"] = pd.NA
    out["total_reviews"] = out["positive_reviews"] + out["negative_reviews"]
    out["rating_percent"] = out.apply(
        lambda r: r["positive_reviews"] / r["total_reviews"] if r["total_reviews"] > 0 else pd.NA,
        axis=1,
    )
    out["price_category"] = out["price"].apply(_price_category)
    return out.drop_duplicates(subset=["app_id"], keep="first")


def load_steam_catalog_real(path: Path | None = None) -> pd.DataFrame:
    """
    Load and normalize the bundled catalog if the file exists.
    Uses column subset + chunked read for large files.
    """
    p = path or default_catalog_path()
    if not p.is_file():
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    try:
        # Peek header for column presence (handles minor header drift)
        head = _sanitize_columns(pd.read_csv(p, nrows=0, encoding="utf-8-sig", index_col=False))
        avail = set(head.columns.str.strip())
        usecols = [c for c in CATALOG_READ_COLUMNS if c in avail]
        if not usecols or "AppID" not in usecols:
            return pd.DataFrame()

        for chunk in pd.read_csv(
            p,
            encoding="utf-8-sig",
            usecols=usecols,
            chunksize=50_000,
            low_memory=False,
            index_col=False,
        ):
            chunks.append(normalize_steam_catalog_chunk(_sanitize_columns(chunk)))
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError):
        return pd.DataFrame()

    if not chunks:
        return pd.DataFrame()
    full = pd.concat(chunks, ignore_index=True)
    return full.drop_duplicates(subset=["app_id"], keep="first")


def lookup_catalog_profile(catalog: pd.DataFrame, app_id: int) -> pd.Series | None:
    """Single-row slice for Game Profile card."""
    if catalog.empty or "app_id" not in catalog.columns:
        return None
    hit = catalog.loc[catalog["app_id"] == int(app_id)]
    if hit.empty:
        return None
    return hit.iloc[0]


def live_catalog_context_sentence(profile: pd.Series) -> str:
    """One-line analyst framing for insights (Stage 5C)."""
    genre_raw = str(profile.get("genre") or "Unknown")
    genre_short = genre_raw.split(",")[0].strip() if genre_raw else "Unknown"
    pc = str(profile.get("price_category") or "Unknown")
    peak = int(profile.get("peak_players") or 0)
    if peak >= 200_000:
        peak_phrase = "very high peak concurrent usage"
    elif peak >= 50_000:
        peak_phrase = "high peak player activity"
    elif peak >= 10_000:
        peak_phrase = "strong concurrent audience"
    elif peak > 0:
        peak_phrase = "moderate peak traffic"
    else:
        peak_phrase = "limited peak CCU in catalog snapshot"

    return (
        f"Live sentiment is being analyzed for a **{pc}** **{genre_short}** title with **{peak_phrase}** "
        f"(catalog snapshot)."
    )


def merge_steam_catalog_into_live_reviews(
    cleaned_reviews: pd.DataFrame,
    steam_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join bundled catalog metadata onto cleaned live rows by `app_id`."""
    if cleaned_reviews.empty or steam_catalog.empty:
        out = cleaned_reviews.copy()
        out["steam_catalog_match"] = False
        return out

    meta_cols = [
        c
        for c in (
            "app_id",
            "game_name",
            "genre",
            "price",
            "price_category",
            "rating_percent",
            "positive_reviews",
            "negative_reviews",
            "peak_players",
            "release_date",
            "total_reviews",
        )
        if c in steam_catalog.columns
    ]
    sc = steam_catalog[meta_cols].drop_duplicates("app_id", keep="first")

    # `game_name` exists on both sides → left keeps name; catalog title becomes `game_name_steamcat`
    out = cleaned_reviews.merge(sc, on="app_id", how="left", suffixes=("", "_steamcat"))
    if "game_name_steamcat" in out.columns:
        out["catalog_game_name"] = out["game_name_steamcat"]
        out.drop(columns=["game_name_steamcat"], inplace=True)

    out["steam_catalog_match"] = out["genre"].notna() if "genre" in out.columns else False
    return out


def optional_csv_catalog_overlay(cleaned_after_steam: pd.DataFrame, optional_games: pd.DataFrame, merge_fn: Any) -> pd.DataFrame:
    """
    Optional games CSV join by game name — fills gaps only so Steam catalog metadata wins when present.

    `merge_fn` is `merge_reviews_with_games_catalog` from review_sentiment.
    """
    if optional_games is None or optional_games.empty:
        return cleaned_after_steam
    steam_match_flag = cleaned_after_steam["steam_catalog_match"].copy()
    merged = merge_fn(cleaned_after_steam, optional_games)
    merged["steam_catalog_match"] = steam_match_flag.values
    if "genre_catalog" in merged.columns:
        merged["genre"] = merged["genre"].fillna(merged["genre_catalog"])
    if "price_catalog" in merged.columns:
        merged["price"] = merged["price"].fillna(merged["price_catalog"])
    if "price_category_catalog" in merged.columns:
        merged["price_category"] = merged["price_category"].fillna(merged["price_category_catalog"])
    if "rating_percent_catalog" in merged.columns:
        merged["rating_percent"] = merged["rating_percent"].fillna(merged["rating_percent_catalog"])
    merged["catalog_match"] = merged["genre"].notna()
    return merged
