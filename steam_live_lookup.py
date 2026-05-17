"""
Stage 5A — Steam Live Lookup (public Store endpoints, no API key).

Fetches recent reviews for an App ID with pagination (up to 5,000), local CSV cache,
Streamlit caching, and alignment with shared sentiment helpers.
"""

from __future__ import annotations

import html
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

from review_sentiment import (
    NEGATIVE_THEME_KEYWORDS,
    POSITIVE_THEME_KEYWORDS,
    REV_CHART_HEIGHT,
    REV_CHART_MARGIN,
    clean_reviews_df,
    detect_risk_signals,
    figure_sentiment_pie,
    keyword_hit_counts,
    merge_reviews_with_games_catalog,
    synthesize_review_insights,
    top_keyword_themes,
)
from steam_catalog_enrichment import (
    live_catalog_context_sentence,
    lookup_catalog_profile,
    merge_steam_catalog_into_live_reviews,
    optional_csv_catalog_overlay,
)
from game_discovery import record_recent_game, render_game_discovery_section, render_selection_message
from player_pain_intelligence import render_player_pain_intelligence
from steam_live_executive import render_executive_decision_layer

REVIEW_LIMIT_OPTIONS = (100, 500, 1000, 2500, 5000)
DEFAULT_LIVE_REVIEW_LIMIT = 100
BATCH_SIZE = 100
REVIEWS_DATA_DIR = Path(__file__).resolve().parent / "data" / "reviews"

STEAM_REVIEWS_BASE_URL = "https://store.steampowered.com/appreviews/{app_id}"
STEAM_APPDETAILS_URL = "https://store.steampowered.com/api/appdetails?appids={app_id}"

PAGE_FETCH_MAX_RETRIES = 3
PAGE_RETRY_BACKOFF_SEC = 0.6

STEAM_REVIEW_DATA_COLUMNS = (
    "review_text",
    "voted_up",
    "timestamp_created",
    "playtime_forever",
    "language",
    "votes_up",
    "weighted_vote_score",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
)

# Steam occasionally filters bare Python clients
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


def reviews_csv_path(app_id: int, limit: int) -> Path:
    """Local cache path: data/reviews/steam_reviews_{app_id}_{limit}.csv"""
    return REVIEWS_DATA_DIR / f"steam_reviews_{app_id}_{limit}.csv"


def ensure_reviews_data_dir() -> Path:
    REVIEWS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return REVIEWS_DATA_DIR


def fetch_app_display_name(app_id: int, timeout: float = 15.0) -> str:
    """Resolve storefront title; fallback keeps UX usable when appdetails fails."""
    url = STEAM_APPDETAILS_URL.format(app_id=app_id)
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        entry = payload.get(str(app_id), {})
        if entry.get("success") and isinstance(entry.get("data"), dict):
            name = entry["data"].get("name")
            if name:
                return str(name).strip()
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass
    return f"Steam App {app_id}"


@dataclass
class SteamFetchStats:
    """Pagination outcome for UI messaging (does not affect analytics)."""

    requested: int = 0
    fetched: int = 0
    pages_fetched: int = 0
    steam_exhausted: bool = False
    stop_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "fetched": self.fetched,
            "pages_fetched": self.pages_fetched,
            "steam_exhausted": self.steam_exhausted,
            "stop_reason": self.stop_reason,
        }


def _steam_reviews_query_params(*, cursor: str, num_per_page: int) -> dict[str, str | int]:
    return {
        "json": 1,
        "filter": "recent",
        "language": "all",
        "num_per_page": max(1, min(int(num_per_page), BATCH_SIZE)),
        "cursor": cursor,
    }


def fetch_steam_reviews(app_id: int, timeout: float = 25.0) -> dict[str, Any]:
    """
    GET first page of public appreviews JSON (legacy single-request helper).
    Prefer `fetch_steam_reviews_batched` for larger pulls.
    """
    return _fetch_steam_reviews_page(app_id, "*", num_per_page=min(20, BATCH_SIZE), timeout=timeout)


def _fetch_steam_reviews_page(
    app_id: int,
    cursor: str,
    *,
    num_per_page: int = BATCH_SIZE,
    timeout: float = 25.0,
) -> dict[str, Any]:
    """Single appreviews page; cursor is passed via params so it is URL-encoded correctly."""
    url = STEAM_REVIEWS_BASE_URL.format(app_id=app_id)
    params = _steam_reviews_query_params(cursor=cursor, num_per_page=num_per_page)
    resp = requests.get(url, params=params, headers=_HTTP_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _cursor_key(cursor: str | None) -> str:
    return str(cursor or "").strip()


def _extract_next_cursor(payload: dict[str, Any]) -> str:
    """Steam may return the next cursor at top level or under query_summary."""
    raw = payload.get("cursor")
    if raw is None:
        qs = payload.get("query_summary")
        if isinstance(qs, dict):
            raw = qs.get("cursor")
    return _cursor_key(raw)


def _strip_review_html(raw: str) -> str:
    """Light cleanup for Steam's HTML-ish review bodies."""
    if not raw:
        return ""
    t = html.unescape(str(raw))
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    return t.strip()


def _review_dict_from_steam(rev: dict[str, Any], *, app_id: int, game_name: str) -> dict[str, Any]:
    voted_up = bool(rev.get("voted_up"))
    row: dict[str, Any] = {
        "game_name": game_name,
        "review_text": _strip_review_html(rev.get("review") or ""),
        "sentiment": "positive" if voted_up else "negative",
        "source": "steam_live",
        "app_id": app_id,
        "voted_up": voted_up,
        "timestamp_created": rev.get("timestamp_created"),
        "playtime_forever": rev.get("author", {}).get("playtime_forever")
        if isinstance(rev.get("author"), dict)
        else rev.get("playtime_forever"),
        "language": rev.get("language"),
        "votes_up": rev.get("votes_up"),
        "weighted_vote_score": rev.get("weighted_vote_score"),
        "steam_purchase": rev.get("steam_purchase"),
        "received_for_free": rev.get("received_for_free"),
        "written_during_early_access": rev.get("written_during_early_access"),
    }
    return row


def steam_reviews_to_dataframe(
    data: dict[str, Any],
    app_id: int,
    game_name: str | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Flatten Steam `reviews` array into the project's canonical textual-review shape.
    """
    title = game_name if game_name else fetch_app_display_name(app_id)
    rows: list[dict[str, Any]] = []
    for rev in data.get("reviews") or []:
        rows.append(_review_dict_from_steam(rev, app_id=app_id, game_name=title))
        if limit is not None and len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def ensure_live_review_schema(df: pd.DataFrame, app_id: int | None = None) -> pd.DataFrame:
    """Normalize CSV/API rows for KPIs, charts, and NLP helpers."""
    if df.empty:
        return df
    out = df.copy()
    if "voted_up" in out.columns:
        vu = out["voted_up"]
        if vu.dtype == object:
            vu = vu.map(lambda x: str(x).strip().lower() in ("true", "1", "yes"))
        out["voted_up"] = vu.fillna(False).astype(bool)
        if "sentiment" not in out.columns or out["sentiment"].isna().all():
            out["sentiment"] = out["voted_up"].map(lambda v: "positive" if v else "negative")
    if app_id is not None and "app_id" not in out.columns:
        out["app_id"] = app_id
    if "source" not in out.columns:
        out["source"] = "steam_live"
    return out


def _fetch_steam_reviews_page_with_retry(
    app_id: int,
    cursor: str,
    *,
    page_num: int,
    num_per_page: int,
) -> dict[str, Any] | None:
    """Retry transient empty/failed page responses before giving up on a cursor."""
    last_exc: Exception | None = None
    for attempt in range(1, PAGE_FETCH_MAX_RETRIES + 1):
        try:
            payload = _fetch_steam_reviews_page(app_id, cursor, num_per_page=num_per_page)
            success = payload.get("success")
            batch = payload.get("reviews")
            if success in (1, True, "1") and batch is not None:
                if batch or attempt == PAGE_FETCH_MAX_RETRIES:
                    return payload
                logger.warning(
                    "Steam page %s empty reviews (attempt %s/%s) cursor=%s",
                    page_num,
                    attempt,
                    PAGE_FETCH_MAX_RETRIES,
                    _cursor_preview(cursor),
                )
            else:
                logger.warning(
                    "Steam page %s bad payload success=%s reviews=%s (attempt %s/%s)",
                    page_num,
                    success,
                    "missing" if batch is None else f"len={len(batch)}",
                    attempt,
                    PAGE_FETCH_MAX_RETRIES,
                )
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            logger.warning(
                "Steam page %s request failed (attempt %s/%s): %s",
                page_num,
                attempt,
                PAGE_FETCH_MAX_RETRIES,
                exc,
            )
        if attempt < PAGE_FETCH_MAX_RETRIES:
            time.sleep(PAGE_RETRY_BACKOFF_SEC * attempt)
    if last_exc is not None:
        raise last_exc
    return None


def _cursor_preview(cursor: str, max_len: int = 48) -> str:
    c = _cursor_key(cursor)
    if len(c) <= max_len:
        return c or "(empty)"
    return f"{c[:max_len]}…"


def fetch_steam_reviews_batched(
    app_id: int,
    limit: int,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    batch_pause_sec: float = 0.25,
) -> tuple[pd.DataFrame | None, str | None, SteamFetchStats]:
    """
    Paginated Steam appreviews fetch until `limit` rows or Steam has no more pages.

    Returns (dataframe or None, error_message or None, fetch_stats).
    """
    limit = max(1, min(int(limit), REVIEW_LIMIT_OPTIONS[-1]))
    stats = SteamFetchStats(requested=limit)
    title = fetch_app_display_name(app_id)
    collected: list[dict[str, Any]] = []
    cursor = "*"
    seen_cursors: set[str] = {_cursor_key("*")}
    max_pages = max((limit + BATCH_SIZE - 1) // BATCH_SIZE + 10, 20)
    page_num = 0

    try:
        while len(collected) < limit and page_num < max_pages:
            page_num += 1
            page_size = min(BATCH_SIZE, limit - len(collected))
            current_cursor = cursor

            logger.info(
                "Steam reviews page=%s batch_size=%s total_fetched=%s cursor=%s",
                page_num,
                page_size,
                len(collected),
                _cursor_preview(current_cursor),
            )

            payload = _fetch_steam_reviews_page_with_retry(
                app_id,
                current_cursor,
                page_num=page_num,
                num_per_page=page_size,
            )
            if payload is None:
                stats.stop_reason = "empty_response_after_retries"
                logger.warning("Steam page %s: no usable payload after retries", page_num)
                break

            success = payload.get("success")
            if success not in (1, True, "1"):
                if not collected:
                    return None, (
                        "Steam reported no data for this App ID (invalid ID, delisted app, or Store refusal). "
                        "Confirm the ID on the Steam Store."
                    ), stats
                stats.stop_reason = "steam_success_false"
                stats.steam_exhausted = True
                break

            batch = payload.get("reviews")
            if batch is None:
                if not collected:
                    return None, "Unexpected response shape from Steam (missing `reviews` field).", stats
                stats.stop_reason = "missing_reviews_field"
                stats.steam_exhausted = True
                break

            count_before = len(collected)
            for rev in batch:
                collected.append(_review_dict_from_steam(rev, app_id=app_id, game_name=title))
                if len(collected) >= limit:
                    break
            added = len(collected) - count_before

            next_cursor = _extract_next_cursor(payload)
            logger.info(
                "Steam reviews page=%s done batch_returned=%s added=%s total_fetched=%s next_cursor=%s",
                page_num,
                len(batch),
                added,
                len(collected),
                _cursor_preview(next_cursor),
            )

            if on_progress is not None:
                on_progress(len(collected), limit)

            stats.pages_fetched = page_num

            if len(collected) >= limit:
                stats.stop_reason = "limit_reached"
                break

            if not batch:
                stats.stop_reason = "empty_batch"
                stats.steam_exhausted = True
                logger.info("Steam returned empty review batch — no more pages")
                break

            if not next_cursor or next_cursor == "@end":
                stats.stop_reason = "no_next_cursor"
                stats.steam_exhausted = True
                logger.info("Steam returned no next cursor — catalog exhausted")
                break

            if next_cursor in seen_cursors and added == 0:
                stats.stop_reason = "duplicate_cursor_no_progress"
                stats.steam_exhausted = True
                logger.warning(
                    "Duplicate cursor with zero new reviews (cursor=%s) — stopping",
                    _cursor_preview(next_cursor),
                )
                break

            if next_cursor in seen_cursors and added > 0:
                logger.warning(
                    "Duplicate cursor but %s new reviews — advancing once (cursor=%s)",
                    added,
                    _cursor_preview(next_cursor),
                )

            seen_cursors.add(next_cursor)
            cursor = next_cursor
            time.sleep(batch_pause_sec)

        else:
            stats.stop_reason = "max_pages_guard"
            logger.warning("Stopped at max_pages=%s (fetched=%s, requested=%s)", max_pages, len(collected), limit)

    except requests.Timeout:
        if not collected:
            return None, "Steam Store did not respond in time — try again shortly.", stats
        stats.stop_reason = "timeout_partial"
    except requests.ConnectionError:
        if not collected:
            return None, "Network connection failed — check your internet connection.", stats
        stats.stop_reason = "connection_partial"
    except requests.HTTPError as exc:
        if not collected:
            code = exc.response.status_code if exc.response is not None else "unknown"
            return None, f"HTTP error from Steam ({code}).", stats
        stats.stop_reason = "http_partial"
    except requests.RequestException as exc:
        if not collected:
            return None, f"Request failed: {exc}", stats
        stats.stop_reason = "request_partial"
    except ValueError:
        if not collected:
            return None, "Could not parse JSON response from Steam.", stats
        stats.stop_reason = "json_partial"

    stats.fetched = len(collected)
    if stats.fetched >= limit:
        stats.stop_reason = stats.stop_reason or "limit_reached"
    elif not stats.stop_reason:
        stats.stop_reason = "steam_exhausted"
        stats.steam_exhausted = True

    if not collected:
        return pd.DataFrame(), None, stats

    return ensure_live_review_schema(pd.DataFrame(collected[:limit]), app_id=app_id), None, stats


def load_reviews_from_csv(app_id: int, limit: int) -> pd.DataFrame | None:
    path = reviews_csv_path(app_id, limit)
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        return ensure_live_review_schema(df, app_id=app_id)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def save_reviews_to_csv(df: pd.DataFrame, app_id: int, limit: int) -> Path:
    ensure_reviews_data_dir()
    path = reviews_csv_path(app_id, limit)
    export = df.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def format_steam_fetch_result_message(stats: SteamFetchStats) -> str:
    """User-facing summary when fetched count may be below the selected limit."""
    msg = f"Steam returned {stats.fetched:,} out of {stats.requested:,} requested reviews."
    if stats.fetched < stats.requested and stats.steam_exhausted:
        msg += " Steam has no additional recent reviews available for this App ID."
    return msg


def _cached_steam_api_fetch_impl(app_id: int, limit: int) -> pd.DataFrame:
    """Uncached body used by Streamlit cache wrapper."""
    df, err, _stats = fetch_steam_reviews_batched(app_id, limit)
    if err or df is None:
        return pd.DataFrame()
    return df


def get_cached_steam_api_fetch():
    import streamlit as st

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_steam_api_fetch(app_id: int, limit: int) -> pd.DataFrame:
        return _cached_steam_api_fetch_impl(app_id, limit)

    return _cached_steam_api_fetch


def get_cached_reviews_csv_load():
    import streamlit as st

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_load_reviews_csv(app_id: int, limit: int, file_mtime: float) -> pd.DataFrame:
        _ = file_mtime
        loaded = load_reviews_from_csv(app_id, limit)
        return loaded if loaded is not None else pd.DataFrame()

    return _cached_load_reviews_csv


def parse_app_id_input(raw: str) -> tuple[int | None, str | None]:
    """Validate numeric Steam App ID (positive integer)."""
    s = (raw or "").strip()
    if not s:
        return None, "Enter a numeric Steam App ID."
    if not s.isdigit():
        return None, "App ID must contain digits only (e.g. 730, 570, 105600)."
    n = int(s)
    if n <= 0:
        return None, "App ID must be a positive integer."
    return n, None


def fetch_live_reviews_pipeline(
    app_id: int,
    limit: int = 20,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Full fetch + dataframe build with normalized error strings for the UI.

    Returns (dataframe or None on hard failure, error_message or None).
    Empty dataframe means success but zero reviews.
    """
    try:
        if limit <= 20:
            payload = fetch_steam_reviews(app_id)
            success = payload.get("success")
            if success not in (1, True, "1"):
                return None, (
                    "Steam reported no data for this App ID (invalid ID, delisted app, or Store refusal). "
                    "Confirm the ID on the Steam Store."
                )
            reviews_list = payload.get("reviews")
            if reviews_list is None:
                return None, "Unexpected response shape from Steam (missing `reviews` field)."
            title = fetch_app_display_name(app_id)
            df = steam_reviews_to_dataframe(payload, app_id, game_name=title, limit=limit)
            return ensure_live_review_schema(df, app_id=app_id), None
        df, err, _stats = fetch_steam_reviews_batched(app_id, limit)
        return df, err
    except requests.Timeout:
        return None, "Steam Store did not respond in time — try again shortly."
    except requests.ConnectionError:
        return None, "Network connection failed — check your internet connection."
    except requests.HTTPError as exc:
        return None, f"HTTP error from Steam ({exc.response.status_code if exc.response else 'unknown'})."
    except requests.RequestException as exc:
        return None, f"Request failed: {exc}"
    except ValueError:
        return None, "Could not parse JSON response from Steam."


def resolve_live_reviews_dataset(
    app_id: int,
    limit: int,
    *,
    force_steam: bool = False,
    progress_bar: Any | None = None,
    progress_caption: Any | None = None,
) -> tuple[pd.DataFrame | None, str | None, str, SteamFetchStats | None]:
    """
    Load from local CSV when present unless `force_steam`.
    Otherwise paginate from Steam (with optional progress widgets), persist CSV, and use cache_data.

    Returns (dataframe, error_message, source_label, fetch_stats).
    source_label is one of: disk, steam, cache, empty.
    """
    import streamlit as st

    limit = int(limit)
    path = reviews_csv_path(app_id, limit)
    cached_csv_load = get_cached_reviews_csv_load()
    cached_api_fetch = get_cached_steam_api_fetch()

    if not force_steam and path.is_file():
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        df = cached_csv_load(app_id, limit, mtime)
        if df is not None and not df.empty:
            n = len(df)
            stats = SteamFetchStats(
                requested=limit,
                fetched=n,
                pages_fetched=0,
                steam_exhausted=n < limit,
                stop_reason="disk_cache",
            )
            return ensure_live_review_schema(df, app_id=app_id), None, "disk", stats
        if df is not None and df.empty:
            return df, None, "disk", SteamFetchStats(requested=limit, stop_reason="disk_cache_empty")

    if not force_steam:
        df_memo = cached_api_fetch(app_id, limit)
        if not df_memo.empty:
            try:
                save_reviews_to_csv(df_memo, app_id, limit)
            except OSError:
                pass
            n = len(df_memo)
            stats = SteamFetchStats(
                requested=limit,
                fetched=n,
                pages_fetched=0,
                steam_exhausted=n < limit,
                stop_reason="memory_cache",
            )
            return ensure_live_review_schema(df_memo, app_id=app_id), None, "cache", stats

    def _on_progress(done: int, target: int) -> None:
        if progress_bar is not None:
            progress_bar.progress(min(1.0, done / max(target, 1)))
        if progress_caption is not None:
            progress_caption.caption(f"Fetched **{done:,}** of **{target:,}** reviews from Steam…")

    if progress_bar is not None:
        progress_bar.progress(0.0)
    df, err, fetch_stats = fetch_steam_reviews_batched(app_id, limit, on_progress=_on_progress)
    if err:
        if progress_bar is not None:
            progress_bar.empty()
        if progress_caption is not None:
            progress_caption.empty()
        return None, err, "empty", fetch_stats

    if df is None:
        if progress_bar is not None:
            progress_bar.empty()
        if progress_caption is not None:
            progress_caption.empty()
        return None, "Steam returned no review data.", "empty", fetch_stats

    if progress_bar is not None:
        progress_bar.progress(1.0)
        progress_bar.empty()
    if progress_caption is not None:
        progress_caption.empty()

    try:
        save_reviews_to_csv(df, app_id, limit)
    except OSError as exc:
        st.warning(f"Reviews fetched but could not be saved locally ({exc}).")

    mtime = path.stat().st_mtime if path.is_file() else time.time()
    cached_csv_load.clear()
    _ = cached_csv_load(app_id, limit, mtime)

    return df, None, "steam", fetch_stats


def enrich_live_with_catalog(df: pd.DataFrame, games_catalog: pd.DataFrame) -> pd.DataFrame:
    """Reuse CSV catalog joins when `game_name` aligns with uploaded games (optional overlay only)."""
    if df.empty:
        return df
    cleaned = clean_reviews_df(df)
    return merge_reviews_with_games_catalog(cleaned, games_catalog)


def build_live_enriched_for_insights(
    df_live: pd.DataFrame,
    steam_catalog_real: pd.DataFrame | None,
    optional_games_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stage 5C priority: bundled `steam_catalog_real` by App ID, then optional CSV name match for gaps.

    Does not touch Market Analytics dataframes.
    """
    if df_live.empty:
        return df_live
    cleaned = clean_reviews_df(df_live)
    sc = steam_catalog_real if steam_catalog_real is not None else pd.DataFrame()
    merged_steam = merge_steam_catalog_into_live_reviews(cleaned, sc)
    return optional_csv_catalog_overlay(merged_steam, optional_games_catalog, merge_reviews_with_games_catalog)


def _finalize_live_fig(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=16, color="#dfe6ea")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,30,42,0.55)",
        font=dict(color="#b8c6d1", size=12),
        height=REV_CHART_HEIGHT,
        margin=REV_CHART_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0),
    )
    return fig


def figure_live_reviews_by_language(df: pd.DataFrame, top_k: int = 10) -> go.Figure:
    if df.empty or "language" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No language data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_live_fig(fig, "Reviews by language")
    vc = df["language"].fillna("unknown").astype(str).value_counts().head(top_k)
    fig = px.bar(x=vc.index, y=vc.values, labels={"x": "Language", "y": "Reviews"})
    return _finalize_live_fig(fig, "Reviews by language")


def figure_live_reviews_over_time(df: pd.DataFrame) -> go.Figure:
    if df.empty or "timestamp_created" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No timestamp data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_live_fig(fig, "Reviews over time")
    ts = pd.to_numeric(df["timestamp_created"], errors="coerce")
    valid = df.loc[ts.notna()].copy()
    if valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid timestamps", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_live_fig(fig, "Reviews over time")
    valid["review_date"] = pd.to_datetime(valid["timestamp_created"], unit="s", utc=True).dt.date
    daily = valid.groupby("review_date", as_index=False).size().rename(columns={"size": "reviews"})
    fig = px.line(daily, x="review_date", y="reviews", markers=True)
    return _finalize_live_fig(fig, "Reviews over time")


def figure_live_playtime_by_sentiment(df: pd.DataFrame) -> go.Figure:
    if df.empty or "playtime_forever" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No playtime data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_live_fig(fig, "Average playtime by sentiment")
    work = df.copy()
    if "sentiment" not in work.columns and "voted_up" in work.columns:
        work["sentiment"] = work["voted_up"].map(lambda v: "positive" if bool(v) else "negative")
    work["playtime_hours"] = pd.to_numeric(work["playtime_forever"], errors="coerce") / 60.0
    agg = work.groupby("sentiment", as_index=False)["playtime_hours"].mean()
    if agg.empty or agg["playtime_hours"].isna().all():
        fig = go.Figure()
        fig.add_annotation(text="No playtime values", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_live_fig(fig, "Average playtime by sentiment")
    fig = px.bar(agg, x="sentiment", y="playtime_hours", labels={"playtime_hours": "Avg hours played"})
    return _finalize_live_fig(fig, "Average playtime by sentiment (hours)")


def compute_live_review_kpis(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {
            "total": 0,
            "pos_pct": 0.0,
            "neg_pct": 0.0,
            "avg_playtime_h": 0.0,
            "top_languages": "—",
            "steam_purchase_pct": 0.0,
            "early_access_pct": 0.0,
            "helpful_top": pd.DataFrame(),
        }
    if "sentiment" in df.columns:
        pos_n = int((df["sentiment"].astype(str).str.lower() == "positive").sum())
    elif "voted_up" in df.columns:
        pos_n = int(df["voted_up"].astype(bool).sum())
    else:
        pos_n = 0
    neg_n = n - pos_n
    play = pd.to_numeric(df.get("playtime_forever"), errors="coerce")
    avg_play_h = float(play.mean() / 60.0) if play.notna().any() else 0.0

    langs = "—"
    if "language" in df.columns:
        top = df["language"].fillna("unknown").astype(str).value_counts().head(3)
        langs = ", ".join(f"{k} ({v})" for k, v in top.items())

    steam_pct = 0.0
    if "steam_purchase" in df.columns:
        sp = df["steam_purchase"]
        if sp.dtype == object:
            steam_pct = 100.0 * sp.astype(str).str.lower().isin(("true", "1", "yes")).mean()
        else:
            steam_pct = 100.0 * sp.fillna(False).astype(bool).mean()

    ea_pct = 0.0
    if "written_during_early_access" in df.columns:
        ea = df["written_during_early_access"]
        if ea.dtype == object:
            ea_pct = 100.0 * ea.astype(str).str.lower().isin(("true", "1", "yes")).mean()
        else:
            ea_pct = 100.0 * ea.fillna(False).astype(bool).mean()

    helpful = pd.DataFrame()
    if "votes_up" in df.columns:
        cols = [c for c in ["review_text", "votes_up", "sentiment", "language"] if c in df.columns]
        helpful = df.sort_values("votes_up", ascending=False).head(5)[cols]

    return {
        "total": n,
        "pos_pct": 100.0 * pos_n / n,
        "neg_pct": 100.0 * neg_n / n,
        "avg_playtime_h": avg_play_h,
        "top_languages": langs,
        "steam_purchase_pct": steam_pct,
        "early_access_pct": ea_pct,
        "helpful_top": helpful,
    }


def render_live_review_kpis(df: pd.DataFrame) -> None:
    import streamlit as st

    k = compute_live_review_kpis(df)
    r1 = st.columns(4)
    r2 = st.columns(4)
    with r1[0]:
        st.metric("Total reviews fetched", f"{k['total']:,}")
    with r1[1]:
        st.metric("Positive reviews %", f"{k['pos_pct']:.1f}%")
    with r1[2]:
        st.metric("Negative reviews %", f"{k['neg_pct']:.1f}%")
    with r1[3]:
        st.metric("Average playtime", f"{k['avg_playtime_h']:.1f} h")
    with r2[0]:
        st.metric("Top languages", k["top_languages"])
    with r2[1]:
        st.metric("Steam purchase %", f"{k['steam_purchase_pct']:.1f}%")
    with r2[2]:
        st.metric("Early access reviews %", f"{k['early_access_pct']:.1f}%")
    with r2[3]:
        st.metric("Sample depth", f"{k['total']:,} rows")

    if not k["helpful_top"].empty:
        st.markdown("##### Most helpful reviews")
        st.dataframe(k["helpful_top"], use_container_width=True, hide_index=True, height=220)


def render_live_review_charts(df: pd.DataFrame) -> None:
    import streamlit as st

    from dashboard_ux import render_chart_block

    chart_df = df.copy()
    if "sentiment_polarity" not in chart_df.columns:
        chart_df["sentiment_polarity"] = chart_df.get("sentiment", pd.Series(dtype=str)).astype(str).str.lower()
    k = compute_live_review_kpis(df)
    pie_insight = (
        f"**{k['neg_pct']:.0f}% negative** in this pull — treat as a warning lens."
        if k["neg_pct"] >= 40
        else f"**{k['pos_pct']:.0f}% positive** — satisfaction tone is favorable in this sample."
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_chart_block(
            figure_sentiment_pie(chart_df),
            question="What is the sentiment mix in this fetch?",
            insight=pie_insight,
        )
        render_chart_block(
            figure_live_reviews_over_time(df),
            question="How did review volume trend over time?",
            insight="Spikes often align with patches, sales, or controversy windows.",
        )
    with c2:
        render_chart_block(
            figure_live_reviews_by_language(df),
            question="Which languages dominate the sample?",
            insight=f"Top languages: **{k['top_languages']}** — localize triage if negatives cluster elsewhere.",
        )
        render_chart_block(
            figure_live_playtime_by_sentiment(df),
            question="Does playtime differ by sentiment?",
            insight=f"Average reporter playtime **{k['avg_playtime_h']:.1f} h** — veterans often drive nuanced negatives.",
        )


def render_ai_executive_summary(df: pd.DataFrame, enriched: pd.DataFrame) -> None:
    """Rule-based executive narrative: satisfaction, risk, engagement, and product signals."""
    import streamlit as st

    k = compute_live_review_kpis(df)
    n = k["total"]
    if n == 0:
        return

    st.markdown("##### Executive summary (AI-style)")
    st.caption(
        "Automated interpretation of this review sample — satisfaction, risk, engagement, and product cues "
        "(no external LLM; derived from review metrics and lexicon helpers)."
    )

    paragraphs: list[str] = []

    if k["pos_pct"] >= 75:
        paragraphs.append(
            f"**Player satisfaction** looks strong in this {n:,}-review sample ({k['pos_pct']:.0f}% positive). "
            "The tone supports continued investment in content and community programs."
        )
    elif k["pos_pct"] >= 55:
        paragraphs.append(
            f"**Player satisfaction** is moderately positive ({k['pos_pct']:.0f}% positive, {k['neg_pct']:.0f}% negative). "
            "Monitor recurring complaint themes before they widen into refund or review-bomb risk."
        )
    else:
        paragraphs.append(
            f"**Player satisfaction** is under pressure ({k['neg_pct']:.0f}% negative in this pull). "
            "Treat this as an early-warning lens and validate whether issues are systemic or sample-specific."
        )

    if k["avg_playtime_h"] >= 40:
        paragraphs.append(
            f"**Engagement depth** is high — average reporter playtime is about **{k['avg_playtime_h']:.0f} hours**, "
            "suggesting invested players rather than drive-by impressions."
        )
    elif k["avg_playtime_h"] >= 10:
        paragraphs.append(
            f"**Engagement depth** is moderate (~**{k['avg_playtime_h']:.0f} h** average playtime). "
            "Negative reviews from long-time players often flag balance, endgame, or live-ops fatigue."
        )
    else:
        paragraphs.append(
            f"**Engagement depth** appears shallow in this sample (~**{k['avg_playtime_h']:.1f} h** average). "
            "Critiques may reflect onboarding, first-session friction, or refund-window experiences."
        )

    risks = detect_risk_signals(enriched) if not enriched.empty else []
    if risks:
        paragraphs.append(
            "**Risk & reputation:** " + " ".join(risks[:2])
            + (" …" if len(risks) > 2 else "")
        )
    elif k["neg_pct"] >= 40:
        paragraphs.append(
            "**Risk & reputation:** Elevated negative share without dominant lexicon banners — "
            "still worth a targeted pass on performance, monetization, and multiplayer health."
        )
    else:
        paragraphs.append(
            "**Risk & reputation:** No major lexicon risk banners fired in this sample; continue periodic pulls to confirm stability."
        )

    if k["steam_purchase_pct"] < 70 and n >= 50:
        paragraphs.append(
            f"**Business signal:** Only **{k['steam_purchase_pct']:.0f}%** of reviewers report a Steam purchase — "
            "key gift, bundle, or free-weekend cohorts may be skewing sentiment."
        )
    elif k["early_access_pct"] >= 15:
        paragraphs.append(
            f"**Product signal:** **{k['early_access_pct']:.0f}%** of reviews were written during early access — "
            "compare tone with current build quality before roadmap commitments."
        )
    else:
        paragraphs.append(
            f"**Business signal:** Purchase mix ({k['steam_purchase_pct']:.0f}% Steam purchase) and language spread "
            f"({k['top_languages']}) inform localization and positioning decisions."
        )

    for para in paragraphs[:4]:
        st.markdown(para)


def render_game_profile_card(profile: pd.Series) -> None:
    """Steam catalog snapshot for the fetched App ID."""
    import streamlit as st

    rd = profile.get("release_date")
    rd_disp = "—"
    if pd.notna(rd):
        try:
            rd_disp = pd.Timestamp(rd).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            rd_disp = str(rd)

    st.markdown("##### Game Profile")
    st.caption("Metadata from **steam_catalog_real.csv** (matched by App ID).")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Game name", str(profile.get("game_name", "—"))[:80])
        st.metric("Genre", str(profile.get("genre", "—"))[:60])
        st.metric("Price", f"${float(profile.get('price', 0) or 0):.2f}")
    with g2:
        st.metric("Price category", str(profile.get("price_category", "—")))
        st.metric("Peak players (CCU)", f"{int(profile.get('peak_players', 0) or 0):,}")
        st.metric("Positive reviews", f"{int(profile.get('positive_reviews', 0) or 0):,}")
    with g3:
        st.metric("Negative reviews", f"{int(profile.get('negative_reviews', 0) or 0):,}")
        st.metric("Release date", rd_disp)


def render_live_keyword_and_ai_snippets(
    enriched: pd.DataFrame,
    *,
    catalog_profile: pd.Series | None = None,
) -> None:
    """Keyword tables + shared insight/risk helpers (subset of Review Sentiment Intelligence)."""
    import streamlit as st

    if enriched.empty:
        return

    if catalog_profile is not None:
        st.markdown("##### Catalog-informed snapshot")
        st.info(live_catalog_context_sentence(catalog_profile))

    pos_map = keyword_hit_counts(enriched["review_text"], POSITIVE_THEME_KEYWORDS)
    neg_map = keyword_hit_counts(enriched["review_text"], NEGATIVE_THEME_KEYWORDS)

    st.markdown("##### Live keyword snapshot")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Top positive keywords (lexicon)")
        st.dataframe(top_keyword_themes(pos_map), use_container_width=True, hide_index=True, height=200)
    with c2:
        st.caption("Top negative keywords (lexicon)")
        st.dataframe(top_keyword_themes(neg_map), use_container_width=True, hide_index=True, height=200)

    st.markdown("##### AI-generated review insights (live fetch)")
    st.caption(
        "Same rule engine as Review Sentiment Intelligence — rows enriched with **steam_catalog_real** metadata "
        "when the App ID matches."
    )
    try:
        for line in synthesize_review_insights(enriched):
            st.markdown(f"- {line}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Insight generator skipped: {exc}")

    st.markdown("##### Risk signals from live reviews")
    try:
        risks = detect_risk_signals(enriched)
        if risks:
            for line in risks:
                st.markdown(f"- {line}")
        else:
            st.markdown("- No lexical risk banners exceeded thresholds for this sample.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Risk detection skipped: {exc}")


def render_steam_live_lookup_panel(
    games_catalog: pd.DataFrame,
    *,
    steam_catalog_real: pd.DataFrame | None = None,
    panel_title: str | None = "Steam Live Lookup",
    lead_divider: bool = True,
) -> None:
    """Streamlit UI: App ID input, fetch, KPIs, sentiment viz, sample table, reuse NLP helpers."""
    import streamlit as st

    if "steam_live_df" not in st.session_state:
        st.session_state["steam_live_df"] = None
    if "steam_live_last_error" not in st.session_state:
        st.session_state["steam_live_last_error"] = None
    if "steam_live_last_source" not in st.session_state:
        st.session_state["steam_live_last_source"] = None
    if "steam_live_review_limit" not in st.session_state:
        st.session_state["steam_live_review_limit"] = DEFAULT_LIVE_REVIEW_LIMIT

    if lead_divider:
        st.markdown("---")
    if panel_title:
        st.subheader(panel_title)
    st.caption(
        "Fetch up to **5,000** recent Steam Store reviews (paginated batches, no API key). "
        "Results are cached locally under `data/reviews/` and in-session for one hour. "
        "Examples: **730** Counter-Strike 2, **570** Dota 2, **105600** Terraria."
    )

    render_game_discovery_section()
    render_selection_message()

    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        st.text_input(
            "Steam App ID",
            placeholder="e.g. 730",
            key="steam_live_app_id_field",
            help="Numeric App ID from the store URL: store.steampowered.com/app/<id>/…",
        )
    with ctrl2:
        st.selectbox(
            "Review limit",
            options=list(REVIEW_LIMIT_OPTIONS),
            key="steam_live_review_limit",
            help="Maximum reviews to pull from Steam (batched requests of 100).",
        )

    err_box = st.session_state.get("steam_live_last_error")
    if err_box:
        st.warning(err_box)

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        fetch_clicked = st.button("Fetch Steam Data", key="steam_live_fetch_btn")
    with col_b:
        refresh_clicked = st.button("Refresh from Steam", key="steam_live_refresh_btn")
    with col_c:
        st.caption("Uses paginated `store.steampowered.com/appreviews/{id}?json=1` · saved as CSV per App ID + limit")

    if fetch_clicked or refresh_clicked:
        app_raw = str(st.session_state.get("steam_live_app_id_field", "") or "")
        limit = int(st.session_state.get("steam_live_review_limit", DEFAULT_LIVE_REVIEW_LIMIT))
        aid, verr = parse_app_id_input(app_raw)
        st.session_state["steam_live_last_error"] = None
        if verr:
            st.session_state["steam_live_last_error"] = verr
            st.session_state["steam_live_df"] = None
            st.session_state["steam_live_last_source"] = None
            st.rerun()

        force_steam = bool(refresh_clicked)
        progress = st.progress(0.0)
        caption_slot = st.empty()
        try:
            df_live, fetch_err, source, fetch_stats = resolve_live_reviews_dataset(
                aid,
                limit,
                force_steam=force_steam,
                progress_bar=progress,
                progress_caption=caption_slot,
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state["steam_live_last_error"] = f"Could not load reviews ({exc})."
            st.session_state["steam_live_df"] = None
            st.session_state["steam_live_last_source"] = None
            st.session_state["steam_live_fetch_stats"] = None
            st.rerun()

        if fetch_err:
            st.session_state["steam_live_last_error"] = fetch_err
            st.session_state["steam_live_df"] = None
            st.session_state["steam_live_last_source"] = None
            st.session_state["steam_live_fetch_stats"] = None
        else:
            st.session_state["steam_live_df"] = df_live
            st.session_state["steam_live_last_error"] = None
            st.session_state["steam_live_last_source"] = source
            st.session_state["steam_live_fetch_stats"] = (
                fetch_stats.as_dict() if fetch_stats is not None else None
            )
            game_title = (
                str(df_live["game_name"].iloc[0])
                if isinstance(df_live, pd.DataFrame)
                and not df_live.empty
                and "game_name" in df_live.columns
                else f"Steam App {aid}"
            )
            record_recent_game(game_title, aid)
            if fetch_stats is not None:
                st.session_state["steam_live_info"] = format_steam_fetch_result_message(fetch_stats)
                if source == "disk" and not force_steam:
                    path = reviews_csv_path(aid, limit)
                    st.session_state["steam_live_info"] += f" Loaded from `{path.name}`."
                elif source == "steam":
                    st.session_state["steam_live_info"] += " Saved to `data/reviews/`."
            elif source == "disk" and not force_steam:
                path = reviews_csv_path(aid, limit)
                st.session_state["steam_live_info"] = f"Loaded cached reviews from `{path.name}`."
            elif source == "steam":
                st.session_state["steam_live_info"] = (
                    f"Fetched {len(df_live):,} reviews from Steam and saved to `data/reviews/`."
                )
        st.rerun()

    info_msg = st.session_state.pop("steam_live_info", None)
    if info_msg:
        st.info(info_msg)

    df_live = st.session_state.get("steam_live_df")
    if df_live is None:
        return
    if isinstance(df_live, pd.DataFrame) and df_live.empty:
        if not st.session_state.get("steam_live_last_error"):
            st.info("Steam returned **success** but **no reviews** in this pull — try another App ID or retry later.")
        return

    df_live = ensure_live_review_schema(df_live)
    aid = int(df_live["app_id"].iloc[0]) if "app_id" in df_live.columns else 0
    limit = int(st.session_state.get("steam_live_review_limit", len(df_live)))
    csv_path = reviews_csv_path(aid, limit)
    if csv_path.is_file():
        st.caption(f"Local cache: `{csv_path.relative_to(Path(__file__).resolve().parent)}`")

    stats_blob = st.session_state.get("steam_live_fetch_stats")
    if isinstance(stats_blob, dict) and stats_blob.get("requested"):
        try:
            fetch_stats = SteamFetchStats(
                requested=int(stats_blob.get("requested", limit)),
                fetched=int(stats_blob.get("fetched", len(df_live))),
                pages_fetched=int(stats_blob.get("pages_fetched", 0)),
                steam_exhausted=bool(stats_blob.get("steam_exhausted", False)),
                stop_reason=str(stats_blob.get("stop_reason", "")),
            )
            st.info(format_steam_fetch_result_message(fetch_stats))
        except (TypeError, ValueError):
            pass

    sc_df = steam_catalog_real if steam_catalog_real is not None else pd.DataFrame()
    catalog_profile = lookup_catalog_profile(sc_df, aid) if not sc_df.empty else None
    enriched = build_live_enriched_for_insights(df_live, steam_catalog_real, games_catalog)

    from dashboard_ux import close_section_header, render_section_header

    render_section_header(
        "Executive summary",
        "Live review snapshot",
        f"**{len(df_live):,}** reviews · App ID **{aid}** · expand sections for sentiment, pain, and decision intelligence.",
    )
    render_live_review_kpis(df_live)
    close_section_header()

    with st.expander("Sentiment analysis", expanded=True):
        render_live_review_charts(df_live)

    with st.expander("Executive intelligence", expanded=True):
        render_ai_executive_summary(df_live, enriched)

    csv_mtime = csv_path.stat().st_mtime if csv_path.is_file() else 0.0
    with st.expander("Pain analytics", expanded=True):
        render_player_pain_intelligence(df_live, app_id=aid, limit=limit, csv_mtime=csv_mtime)

    with st.expander("Game profile & competitive context", expanded=False):
        if catalog_profile is not None:
            render_game_profile_card(catalog_profile)
        elif sc_df.empty:
            st.info(
                "**steam_catalog_real.csv** was not found or could not be read — App ID enrichment and Game Profile "
                "are unavailable. Place the catalog next to the app or verify the file path."
            )
        else:
            st.warning("No catalog match found for this App ID.")

    with st.expander("Decision layer & review drill-down", expanded=False):
        render_executive_decision_layer(df_live, enriched, catalog_profile)

        sample_cols = [c for c in ["game_name", "review_text", "sentiment", "source", "app_id"] if c in df_live.columns]
        extra = [c for c in STEAM_REVIEW_DATA_COLUMNS if c in df_live.columns and c not in sample_cols]
        show_cols = sample_cols + extra[:6]

        st.markdown("**Sample reviews**")
        st.dataframe(df_live[show_cols].head(20), use_container_width=True, height=280)

        render_live_keyword_and_ai_snippets(
            enriched,
            catalog_profile=catalog_profile if catalog_profile is not None else None,
        )
