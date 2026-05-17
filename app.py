"""
Steam AI Analytics Platform – MVP
Streamlit dashboard for Steam games CSV analysis (cleaning, KPIs, charts, insights).
"""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from review_sentiment import (
    EXPECTED_REVIEW_COLS,
    clean_reviews_df,
    merge_reviews_with_games_catalog,
    render_review_sentiment_intelligence,
    scope_reviews_to_filtered_games,
)
from steam_catalog_enrichment import default_catalog_path, load_steam_catalog_real
from dashboard_ux import (
    close_section_header,
    finalize_dashboard_chart,
    inject_dashboard_ux_css,
    insight_for_genre_chart,
    insight_for_peak_chart,
    insight_for_price_chart,
    insight_for_rating_chart,
    market_kpi_summary,
    render_chart_block,
    render_insight,
    render_section_header,
)
from steam_live_lookup import render_steam_live_lookup_panel

# -----------------------------------------------------------------------------
# Page & theme (Steam-inspired dark storefront)
# -----------------------------------------------------------------------------

PAGE_TITLE = "Steam AI Analytics Platform – MVP"

# Shared Plotly appearance
PLOT_MARGIN = dict(l=40, r=20, t=52, b=44)
CHART_HEIGHT = 360


def inject_dark_theme_css() -> None:
    """Steam / gaming storefront inspired dark dashboard styles."""
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(1200px 600px at 10% -10%, #1b2838 0%, transparent 55%),
                    radial-gradient(900px 500px at 100% 0%, #2a475e 0%, transparent 50%),
                    linear-gradient(180deg, #0e141b 0%, #0a0e13 60%, #0e141b 100%);
                color: #c7d5e0;
            }
            [data-testid="stHeader"] {
                background: rgba(27, 40, 56, 0.92);
                border-bottom: 1px solid #2a475e;
                backdrop-filter: blur(10px);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #16202d 0%, #171a21 100%);
                border-right: 1px solid #2a475e;
            }
            [data-testid="stSidebar"] > div:first-child { padding-top: 1.35rem; }
            div[data-testid="stMarkdownContainer"] p,
            div[data-testid="stMarkdownContainer"] li { color: #c7d5e0; }
            .sidebar-brand {
                font-size: 1.05rem;
                font-weight: 700;
                color: #66c0f4;
                letter-spacing: 0.02em;
                margin-bottom: 0.25rem;
            }
            .sidebar-sub {
                font-size: 0.85rem;
                color: #8f98a0;
                margin-bottom: 1rem;
                line-height: 1.45;
            }
            div.kpi-card {
                background: linear-gradient(160deg, #1f2e3f 0%, #1b2838 55%, #151b24 100%);
                border: 1px solid #2a475e;
                border-radius: 12px;
                padding: 1.15rem 1.35rem;
                margin-bottom: 0.85rem;
                box-shadow: 0 12px 40px rgba(0,0,0,0.45);
            }
            div.kpi-card .kpi-label {
                font-size: 0.74rem;
                color: #8f98a0;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                margin-bottom: 0.35rem;
            }
            div.kpi-card .kpi-value {
                font-size: 1.45rem;
                font-weight: 700;
                color: #66c0f4;
            }
            h1 {
                font-weight: 700 !important;
                letter-spacing: -0.02em;
                color: #dfe6ea !important;
                margin-bottom: 0.35rem !important;
            }
            h2 {
                font-size: 1.35rem !important;
                margin: 2rem 0 0.75rem 0 !important;
                color: #e5eef5 !important;
                border-bottom: 1px solid #2a475e !important;
                padding-bottom: 0.4rem !important;
            }
            h3 {
                font-size: 1.08rem !important;
                margin: 1.25rem 0 0.5rem !important;
                color: #b8c6d1 !important;
            }
            hr {
                border-color: #2a475e !important;
                margin: 2rem 0 !important;
            }
            div.section-spacer { height: 1.1rem; }
            [data-testid="stPlotlyChart"] { margin-bottom: 0.35rem; }
            h2.dash-section-title {
                font-size: 1.22rem !important;
                border-bottom: none !important;
                margin: 0 0 0.35rem 0 !important;
                padding-bottom: 0 !important;
            }
            [data-testid="stDataFrame"] {
                border: 1px solid #2a475e;
                border-radius: 10px;
                overflow: hidden;
            }
            [data-testid="stSidebar"] .stMultiSelect span,
            [data-testid="stSidebar"] label { color: #c7d5e0; }
            div.platform-shell {
                padding: 0.25rem 0 1rem 0;
                margin-bottom: 0.5rem;
                border-bottom: 1px solid #2a475e;
            }
            div.platform-shell-title {
                font-size: 1.82rem;
                font-weight: 700;
                color: #f0f6fc;
                letter-spacing: -0.03em;
                line-height: 1.2;
            }
            div.platform-shell-tagline {
                color: #8f98a0;
                font-size: 0.88rem;
                margin-top: 0.35rem;
                line-height: 1.45;
            }
            div.module-banner {
                color: #b8c6d1;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.09em;
                margin-bottom: 0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card_html(label: str, value: str) -> str:
    """Single KPI card markup."""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>'
    )


# -----------------------------------------------------------------------------
# Data contract & cleaning
# -----------------------------------------------------------------------------

EXPECTED_COLS = [
    "game_name",
    "genre",
    "price",
    "positive_reviews",
    "negative_reviews",
    "peak_players",
    "release_date",
]

PRICE_CATEGORY_ORDER = ["Free", "Low", "Medium", "High"]

# Platform UX — dual module picker (session_state key MUST remain stable across reruns)
PLATFORM_RADIO_KEY = "steam_ai_platform_mode_radio_labels"
LABEL_MARKET = "Market Analytics Dashboard"
LABEL_LIVE = "Steam Live Review Intelligence"
LABEL_COMPARE = "Comparative Intelligence"
MODE_MARKET = "market"
MODE_LIVE = "live"
MODE_COMPARE = "compare"


def _strip_currency(s: pd.Series) -> pd.Series:
    """Remove currency symbols / commas before numeric coercion."""
    if s.dtype == object:
        return (
            s.astype(str)
            .str.replace(r"[$€£,]", "", regex=True)
            .str.replace(r"\s+", "", regex=True)
            .str.strip()
        )
    return s


def clean_steam_games_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names, handle missing values, and coerce types.
    Extra columns from the CSV are preserved; expected columns are ensured.
    """
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df["game_name"] = df["game_name"].astype(str).str.strip()
    df["game_name"] = df["game_name"].replace({"nan": pd.NA, "": pd.NA})
    df["genre"] = df["genre"].astype(str).str.strip()
    df["genre"] = df["genre"].replace({"nan": pd.NA, "": pd.NA, "None": pd.NA})
    df["genre"] = df["genre"].fillna("Unknown")

    df["price"] = pd.to_numeric(_strip_currency(df["price"]), errors="coerce")
    df["positive_reviews"] = pd.to_numeric(df["positive_reviews"], errors="coerce")
    df["negative_reviews"] = pd.to_numeric(df["negative_reviews"], errors="coerce")
    df["peak_players"] = pd.to_numeric(df["peak_players"], errors="coerce")

    df["price"] = df["price"].fillna(0)
    df["positive_reviews"] = df["positive_reviews"].fillna(0).clip(lower=0)
    df["negative_reviews"] = df["negative_reviews"].fillna(0).clip(lower=0)
    df["peak_players"] = df["peak_players"].fillna(0).clip(lower=0)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    df = df[df["game_name"].notna()].reset_index(drop=True)

    df["total_reviews"] = df["positive_reviews"] + df["negative_reviews"]
    df["rating_percent"] = df.apply(
        lambda r: r["positive_reviews"] / r["total_reviews"] if r["total_reviews"] > 0 else pd.NA,
        axis=1,
    )
    df["popularity_score"] = df["rating_percent"].fillna(0) * df["peak_players"]

    def price_category(price: float) -> str:
        if price <= 0:
            return "Free"
        if price < 10:
            return "Low"
        if price < 30:
            return "Medium"
        return "High"

    df["price_category"] = df["price"].apply(price_category)

    return df


def empty_games_catalog() -> pd.DataFrame:
    """Empty-but-valid games schema for Live-module enrichment joins without uploading CSV."""
    return clean_steam_games_df(pd.DataFrame(columns=EXPECTED_COLS))


@st.cache_data(show_spinner="Loading Steam catalog…")
def load_cached_steam_catalog(_mtime: float) -> pd.DataFrame:
    """Read-only bundle used only by Steam Live Review Intelligence (Stage 5C)."""
    return load_steam_catalog_real()


# -----------------------------------------------------------------------------
# Filtering — sidebar selections drive the filtered view everywhere
# -----------------------------------------------------------------------------


def year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    """Min/max release years for slider; sensible defaults when dates are sparse."""
    ys = df["release_year"].dropna()
    if ys.empty:
        return 2000, 2026
    y_min = int(ys.min())
    y_max = int(ys.max())
    if y_min == y_max:
        return y_min, y_max + 1
    return y_min, y_max


def render_filter_controls(cleaned: pd.DataFrame) -> dict:
    """
    Sidebar filter panel. Empty multiselect means no restriction (show all values).
    """
    st.sidebar.markdown('<p class="sidebar-brand">Filters</p>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p class="sidebar-sub">Narrow the view. Clear a multiselect to show all options.</p>',
        unsafe_allow_html=True,
    )

    genres_sorted = sorted(g for g in cleaned["genre"].dropna().unique())
    genre_sel = st.sidebar.multiselect(
        "Genre",
        options=genres_sorted,
        default=genres_sorted,
        help="Remove selections to include every genre again.",
    )

    price_sel = st.sidebar.multiselect(
        "Price category",
        options=PRICE_CATEGORY_ORDER,
        default=PRICE_CATEGORY_ORDER,
    )

    y_lo, y_hi = year_bounds(cleaned)
    year_range = st.sidebar.slider(
        "Release year range",
        min_value=y_lo,
        max_value=y_hi,
        value=(y_lo, y_hi),
    )

    include_unknown_year = st.sidebar.checkbox(
        "Include games with unknown release year",
        value=True,
    )

    search_q = st.sidebar.text_input(
        "Search game name",
        value="",
        placeholder="Partial name…",
    )
    return {
        "genres": genre_sel,
        "price_categories": price_sel,
        "year_range": year_range,
        "include_unknown_year": include_unknown_year,
        "search": search_q,
    }


def apply_game_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply sidebar rules; empty genre/price selections => no filter on that dimension."""
    out = df.copy()

    genres = list(cfg["genres"])
    if genres:
        out = out[out["genre"].isin(genres)]

    prices = list(cfg["price_categories"])
    if prices:
        out = out[out["price_category"].isin(prices)]

    y0, y1 = cfg["year_range"]
    ry = out["release_year"]
    in_range = ry.notna() & (ry >= y0) & (ry <= y1)
    if cfg["include_unknown_year"]:
        out = out[in_range | ry.isna()]
    else:
        out = out[in_range]

    q = (cfg.get("search") or "").strip()
    if q:
        out = out[out["game_name"].astype(str).str.contains(q, case=False, regex=False, na=False)]

    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# KPIs & charts
# -----------------------------------------------------------------------------


def compute_kpis(df: pd.DataFrame) -> dict:
    """Aggregate KPIs (expects filtered frame)."""
    total_games = len(df)
    rated = df[df["total_reviews"] > 0]
    avg_rating = rated["rating_percent"].mean() if len(rated) else float("nan")
    avg_price = df["price"].mean() if len(df) else float("nan")
    total_peak = int(df["peak_players"].sum()) if len(df) else 0
    top_genre = ""
    if len(df) and df["genre"].notna().any():
        top_genre = df["genre"].value_counts().index[0]

    return {
        "total_games": total_games,
        "avg_rating": avg_rating,
        "avg_price": avg_price,
        "total_peak": total_peak,
        "top_genre": top_genre,
    }


def plotly_dark_template() -> str:
    return "plotly_dark"


def _chart_layout(fig: go.Figure, title: str) -> go.Figure:
    return finalize_dashboard_chart(fig, title, height=CHART_HEIGHT)


def empty_chart_message(title: str, message: str = "No data for current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{message}</b>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#8f98a0"),
    )
    return _chart_layout(fig, title)


def genre_game_counts(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_chart_message("Top genres by number of games")
    counts = df["genre"].value_counts().reset_index()
    counts.columns = ["genre", "games"]
    fig = px.bar(
        counts.head(15),
        x="games",
        y="genre",
        orientation="h",
        title="",
        labels={"games": "Games", "genre": "Genre"},
        color="games",
        color_continuous_scale=["#1b2838", "#66c0f4"],
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False)
    fig.update_coloraxes(colorbar=dict(title="", tickfont=dict(size=11)))
    return _chart_layout(fig, "Top genres by number of games")


def top_games_peak(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_chart_message("Top 10 games by peak players")
    sub = df.nlargest(10, "peak_players")[["game_name", "peak_players", "genre"]].copy()
    fig = px.bar(
        sub,
        x="peak_players",
        y="game_name",
        orientation="h",
        color="peak_players",
        title="",
        labels={"peak_players": "Peak players", "game_name": "Game"},
        color_continuous_scale=["#1b4d2e", "#5ba32b"],
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False)
    fig.update_coloraxes(colorbar=dict(title="", tickfont=dict(size=11)))
    return _chart_layout(fig, "Top 10 games by peak players")


def avg_rating_by_genre(df: pd.DataFrame) -> go.Figure:
    g = (
        df[df["total_reviews"] > 0]
        .groupby("genre", as_index=False)["rating_percent"]
        .mean()
        .sort_values("rating_percent", ascending=False)
    )
    if g.empty:
        return empty_chart_message("Average rating by genre")
    fig = px.bar(
        g.head(15),
        x="genre",
        y="rating_percent",
        title="",
        labels={"genre": "Genre", "rating_percent": "Avg. positive ratio"},
        color="rating_percent",
        color_continuous_scale=["#3c1e50", "#c36bff"],
    )
    fig.update_traces(marker_line_width=0)
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=-32)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_coloraxes(colorbar=dict(title="", tickfont=dict(size=11)))
    return _chart_layout(fig, "Average rating by genre")


def price_category_distribution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_chart_message("Price category distribution")
    dist = df["price_category"].value_counts().reindex(PRICE_CATEGORY_ORDER).fillna(0).astype(int)
    if dist.sum() == 0:
        return empty_chart_message("Price category distribution")
    palette = ["#8f98a0", "#66c0f4", "#e5a54b", "#cd5447"]
    fig = px.pie(
        names=dist.index,
        values=dist.values,
        title="",
        hole=0.45,
        color_discrete_sequence=palette,
    )
    fig.update_traces(textposition="outside", textinfo="percent", textfont=dict(size=10))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.08, x=0))
    return _chart_layout(fig, "Price category distribution")


# -----------------------------------------------------------------------------
# Automated insights & top performers
# -----------------------------------------------------------------------------


def enhanced_insights(df: pd.DataFrame) -> list[str]:
    """Rule-based narratives on the current filtered subset."""
    if df.empty:
        return ["No rows match the current filters — loosen genre, price, year, or search."]

    insights: list[str] = []

    vc = df["genre"].value_counts(normalize=True)
    if len(vc):
        g, p = vc.index[0], vc.iloc[0]
        insights.append(f"Dominant genre in this selection: **{g}** ({p:.1%} of titles).")

    by_peak = df.groupby("genre")["peak_players"].sum().sort_values(ascending=False)
    if len(by_peak) and by_peak.iloc[0] > 0:
        insights.append(
            f"Highest engagement genre (sum of peak players): **{by_peak.index[0]}** "
            f"({int(by_peak.iloc[0]):,} combined peak)."
        )

    rated = df[df["total_reviews"] > 0]

    if len(rated) >= 1:
        q80 = rated["rating_percent"].quantile(0.8)
        pool = rated[rated["rating_percent"] >= q80]
        if len(pool):
            cheap = pool.loc[pool["price"].idxmin()]
            insights.append(
                f"Cheapest high-rated pick: among top sentiment (≥{q80:.0%} positive share), "
                f"**{cheap['game_name']}** is **${cheap['price']:.2f}**."
            )
        hi = rated.loc[rated["rating_percent"].idxmax()]
        insights.append(
            f"Highest-rated in view: **{hi['game_name']}** ({hi['rating_percent']:.1%} positive, "
            f"{int(hi['total_reviews']):,} reviews)."
        )

    peak_sum = df["peak_players"].sum()
    if peak_sum > 0:
        topn = df.nlargest(max(1, min(5, len(df))), "peak_players")
        share = topn["peak_players"].sum() / peak_sum
        insights.append(
            f"Player activity: top {len(topn)} games by peaks account for **{share:.1%}** "
            f"of summed peak players — check if hits dominate your slice."
        )
        med_peak = df["peak_players"].median()
        insights.append(
            f"Typical concurrent interest: median peak **{med_peak:,.0f}** players per title "
            f"(mean **{df['peak_players'].mean():,.0f}**)."
        )

    paid = df[df["price"] > 0]
    free_share = (df["price"] <= 0).mean()
    if len(paid):
        insights.append(
            f"Pricing: **{free_share:.1%}** free ($0); average paid title **${paid['price'].mean():.2f}** "
            f"across {len(paid)} paid games."
        )
    else:
        insights.append(f"Pricing: **{free_share:.1%}** free; no paid titles in this slice.")

    if len(rated) >= 2:
        gavg = rated.groupby("genre")["rating_percent"].mean().sort_values(ascending=False)
        insights.append(
            f"Strongest average sentiment by genre: **{gavg.index[0]}** (~{gavg.iloc[0]:.1%} positive)."
        )

    if peak_sum > 0:
        gp = df.groupby("genre")["popularity_score"].mean().sort_values(ascending=False)
        insights.append(
            f"Quality × reach: **{gp.index[0]}** leads on average rating-weighted peak score."
        )

    return insights


# -----------------------------------------------------------------------------
# Intelligence layer — deterministic “AI” briefing (no external APIs)
# -----------------------------------------------------------------------------


def _safe_min_max_norm(series: pd.Series, fill_na: float = 0.0) -> pd.Series:
    """Map numeric series to roughly [0, 1]; constant or empty → uniform 1.0 for weighting stability."""
    s = pd.to_numeric(series, errors="coerce").fillna(fill_na)
    if len(s) == 0:
        return s
    lo, hi = s.min(), s.max()
    if hi - lo <= 0:
        return pd.Series(1.0, index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def _price_attractiveness(prices: pd.Series) -> pd.Series:
    """
    Higher score = easier entry price for players (rules-based).
    Free-to-play anchored at ceiling; diminishing returns above ~$40.
    """
    p = pd.to_numeric(prices, errors="coerce").fillna(0).clip(lower=0)
    # Free/max appeal; paid declines smoothly
    return np.where(p <= 0, 1.0, 1.0 / (1.0 + np.power(p / 24.0, 1.15)))


def add_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blend sentiment, reach, review evidence, and price appeal into a 0–100 index.
    Normalization is within the current filtered slice so it reads as relative opportunity.
    """
    out = df.copy()
    if out.empty:
        out["opportunity_score"] = pd.Series(dtype=float)
        return out

    rating = out["rating_percent"].fillna(0).clip(0, 1)
    log_peaks = np.log1p(out["peak_players"].astype(float))
    log_revs = np.log1p(out["total_reviews"].astype(float))
    price_attr = pd.Series(_price_attractiveness(out["price"]), index=out.index)

    w_rating, w_peaks, w_revs, w_price = 0.35, 0.30, 0.20, 0.15
    comp = (
        w_rating * _safe_min_max_norm(rating)
        + w_peaks * _safe_min_max_norm(log_peaks)
        + w_revs * _safe_min_max_norm(log_revs)
        + w_price * _safe_min_max_norm(price_attr)
    )
    out["opportunity_score"] = (100 * comp).round(1)
    return out


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}" if pd.notna(x) else "n/a"


def _fmt_num(n: int) -> str:
    return f"{int(n):,}"


def build_executive_ai_summary(filtered: pd.DataFrame) -> dict[str, str]:
    """Six short analyst-style paragraphs derived from the filtered slice."""
    if filtered.empty:
        return {
            "market_overview": "No titles match the active filters; broaden scope to generate a market read.",
            "strongest_genre_signal": "Insufficient data.",
            "player_engagement_observation": "Insufficient data.",
            "pricing_observation": "Insufficient data.",
            "main_opportunity": "Re-open filters to surface portfolio or competitive whitespace.",
            "main_risk": "Selection is empty — validation of catalog coverage is not possible.",
        }

    n = len(filtered)
    rated = filtered[filtered["total_reviews"] > 0]
    peak_sum = int(filtered["peak_players"].sum())
    free_share = (filtered["price"] <= 0).mean()
    paid = filtered[filtered["price"] > 0]
    years = filtered["release_year"].dropna()
    year_span = ""
    if len(years) >= 1:
        year_span = f" Release years span **{int(years.min())}–{int(years.max())}** in this slice."

    market_overview = (
        f"The selection contains **{_fmt_num(n)}** titles with **{_fmt_num(peak_sum)}** summed peak players. "
        f"Approximately **{free_share:.0%}** are free-to-play (list price $0)."
        f"{year_span}"
    )

    # Strongest genre: blend count share and mean popularity_score among genres with ≥1 title
    top_by_count = filtered["genre"].value_counts()
    share_genre = top_by_count.index[0] if len(top_by_count) else "—"
    share_pct = top_by_count.iloc[0] / n if len(top_by_count) else 0
    ps = filtered.groupby("genre")["popularity_score"].mean().sort_values(ascending=False)
    blend_genre = ps.index[0] if len(ps) and peak_sum > 0 else share_genre
    strongest_genre_signal = (
        f"**{share_genre}** accounts for the largest catalog share (**{share_pct:.0%}** of games). "
        f"On a quality×reach basis, **{blend_genre}** currently exhibits the strongest aggregate signal."
    )

    if peak_sum > 0:
        med_p = filtered["peak_players"].median()
        top3 = filtered.nlargest(3, "peak_players")["peak_players"].sum() / peak_sum
        player_engagement_observation = (
            f"Engagement is **{'concentrated' if top3 > 0.55 else 'more evenly spread'}**: "
            f"the top three peak-player titles represent **{top3:.0%}** of summed peaks. "
            f"Median peak concurrency is **{_fmt_num(int(med_p))}** players per title."
        )
    else:
        player_engagement_observation = (
            "Peak-player fields are zero across the selection; treat engagement metrics as unavailable until data is populated."
        )

    if len(paid):
        mode_pc = filtered["price_category"].value_counts().idxmax()
        pricing_observation = (
            f"Among paid SKUs, mean list price is **${paid['price'].mean():.2f}** (median **${paid['price'].median():.2f}**). "
            f"The most common price band in this slice is **{mode_pc}**."
        )
    else:
        pricing_observation = (
            f"The slice is entirely free-to-play at list price; monetization mix (DLC, IAP) is not inferred from this dataset."
        )

    # Opportunity: genre with strong rating but smaller catalog vs average
    main_opportunity = (
        "Consider doubling down on genres that pair **above-median sentiment** with **below-median title depth** — "
        "a practical whitespace heuristic for catalog or partnership prioritization."
    )
    if len(rated) >= 3 and len(filtered["genre"].unique()) > 1:
        g_stats = (
            rated.groupby("genre")
            .agg(avg_rating=("rating_percent", "mean"), titles=("game_name", "count"))
            .reset_index()
        )
        med_r = g_stats["avg_rating"].median()
        med_t = g_stats["titles"].median()
        cand = g_stats[(g_stats["avg_rating"] >= med_r) & (g_stats["titles"] <= med_t)]
        if not cand.empty:
            pick = cand.sort_values(["avg_rating", "titles"], ascending=[False, True]).iloc[0]
            main_opportunity = (
                f"**{pick['genre']}** shows **{_fmt_pct(float(pick['avg_rating']))}** average positive share across "
                f"**{int(pick['titles'])}** titles — sentiment is healthy relative to peers while supply remains lean, "
                f"suggesting room to invest or benchmark without immediate saturation."
            )

    # Risk stacking: prioritize high-traffic softness, then premium misalignment, else generic data caveat
    risk_parts: list[str] = []
    if len(filtered["peak_players"]) and peak_sum > 0 and filtered["peak_players"].max() > 0:
        q75p = filtered["peak_players"].quantile(0.75)
        hot_low = filtered[
            (filtered["peak_players"] >= q75p)
            & filtered["rating_percent"].notna()
            & (filtered["total_reviews"] >= 20)
        ]
        if len(hot_low):
            worst = hot_low.nsmallest(1, "rating_percent").iloc[0]
            if float(worst["rating_percent"]) < 0.55:
                risk_parts.append(
                    f"**{worst['game_name']}** concentrates peak traffic (**{_fmt_num(int(worst['peak_players']))}**) "
                    f"while sentiment sits at **{_fmt_pct(float(worst['rating_percent']))}** — product and comms risk merit review."
                )
    hi_price = filtered[filtered["price_category"].isin(["High", "Medium"])]
    weak_rated = hi_price[(hi_price["total_reviews"] >= 15) & hi_price["rating_percent"].notna()]
    if len(weak_rated):
        wr = weak_rated.nsmallest(1, "rating_percent").iloc[0]
        if float(wr["rating_percent"]) < 0.62 and wr["price"] > 0:
            risk_parts.append(
                f"**{wr['game_name']}** is priced at **${wr['price']:.2f}** ({wr['price_category']}) with "
                f"**{_fmt_pct(float(wr['rating_percent']))}** positive share — stress-test value proof for premium buyers."
            )
    default_risk = (
        "Watch for **thin review coverage** on marquee SKUs: directional sentiment can swing with small sample sizes; "
        "refresh inputs before major roadmap bets."
    )
    main_risk = risk_parts[0] if risk_parts else default_risk
    if len(risk_parts) >= 2:
        main_risk = f"{risk_parts[0]} {risk_parts[1]}"

    return {
        "market_overview": market_overview,
        "strongest_genre_signal": strongest_genre_signal,
        "player_engagement_observation": player_engagement_observation,
        "pricing_observation": pricing_observation,
        "main_opportunity": main_opportunity,
        "main_risk": main_risk,
    }


def build_smart_recommendations(filtered: pd.DataFrame, scored: pd.DataFrame) -> list[str]:
    """4–6 concise, product-analyst style actions derived from rules on the filtered set."""
    if filtered.empty:
        return ["Expand filters to at least one title before operational recommendations can be generated."]

    recs: list[str] = []
    rated = filtered[filtered["total_reviews"] > 0]

    # Genre focus: highest mean popularity_score with ≥2 games
    if len(filtered) >= 2:
        gsize = filtered.groupby("genre").size()
        eligible = gsize[gsize >= 2].index
        if len(eligible):
            best = (
                filtered[filtered["genre"].isin(eligible)]
                .groupby("genre")["popularity_score"]
                .mean()
                .sort_values(ascending=False)
            )
            g = best.index[0]
            recs.append(
                f"Allocate analytical bandwidth to **{g}**: it leads eligible genres on **quality×reach** while "
                f"maintaining multi-title depth (lower false positives from single-game spikes)."
            )

    # Benchmark title
    if not scored.empty and scored["opportunity_score"].notna().any():
        bench = scored.loc[scored["opportunity_score"].idxmax()]
        recs.append(
            f"Use **{bench['game_name']}** as a composite benchmark (Opportunity Score **{bench['opportunity_score']:.1f}** / 100) "
            f"when comparing packaging, pricing, and live-ops expectations within **{bench['genre']}**."
        )

    # Best-performing price category by average rating among titles with reviews
    if len(rated) >= 4:
        pr = (
            rated.groupby("price_category")["rating_percent"]
            .mean()
            .reindex(PRICE_CATEGORY_ORDER)
            .dropna()
        )
        if len(pr):
            best_pc = pr.idxmax()
            recs.append(
                f"Position case studies around the **{best_pc}** price band: it posts the **highest mean sentiment** "
                f"in the current selection, useful for pitch decks and genre-specific pricing guardrails."
            )

    # High rating, low peaks (hidden demand / under-distributed)
    if len(rated) >= 5:
        rp75 = rated["rating_percent"].quantile(0.75)
        pk50 = rated["peak_players"].quantile(0.50)
        gems = rated[(rated["rating_percent"] >= rp75) & (rated["peak_players"] <= pk50)]
        gems = gems.sort_values(["rating_percent", "total_reviews"], ascending=[False, False])
        if len(gems):
            g0 = gems.iloc[0]
            recs.append(
                f"Prioritize discovery experiments for **{g0['game_name']}** — **{_fmt_pct(float(g0['rating_percent']))}** "
                f"positive share with **{_fmt_num(int(g0['peak_players']))}** peak players suggests favorable word-of-mouth "
                f"that has not yet converted to broad reach."
            )

    # High peaks, weaker sentiment
    if len(rated) >= 5:
        pk75 = rated["peak_players"].quantile(0.75)
        rt50 = rated["rating_percent"].quantile(0.50)
        stressed = rated[(rated["peak_players"] >= pk75) & (rated["rating_percent"] <= rt50)]
        if len(stressed):
            s0 = stressed.sort_values("peak_players", ascending=False).iloc[0]
            recs.append(
                f"Schedule a live-service diagnostic on **{s0['game_name']}** — traffic is in the top quartile "
                f"(**{_fmt_num(int(s0['peak_players']))}** peaks) while sentiment trails the median, indicating potential friction."
            )

    # Catalog gap: few titles in a high-sentiment genre
    if len(rated) >= 6:
        gavg = rated.groupby("genre").agg(avg=("rating_percent", "mean"), n=("game_name", "count")).reset_index()
        gavg = gavg[gavg["n"] >= 2]
        if len(gavg) >= 2:
            hi = gavg["avg"].quantile(0.65)
            lo_n = gavg["n"].quantile(0.35)
            gap = gavg[(gavg["avg"] >= hi) & (gavg["n"] <= lo_n)]
            if not gap.empty:
                gg = gap.sort_values("avg", ascending=False).iloc[0]
                recs.append(
                    f"Evaluate **{gg['genre']}** for selective publishing: mean sentiment is **{_fmt_pct(float(gg['avg']))}** "
                    f"yet only **{int(gg['n'])}** qualifying titles — a measured increase in supply could capture unmet demand."
                )

    # Trim to 6 max, ensure at least 4 when data rich
    if len(recs) < 4 and len(filtered) >= 3:
        vc = filtered["genre"].value_counts()
        recs.append(
            f"Maintain a rolling **genre share dashboard** — **{vc.index[0]}** currently leads at **{vc.iloc[0]}** titles; "
            f"rebalance research as share drifts beyond **{vc.iloc[0] / len(filtered):.0%}** of the portfolio view."
        )

    return recs[:6]


def build_anomaly_signals(filtered: pd.DataFrame) -> list[str]:
    """Narrative anomaly lines; examples reference concrete titles when rules fire."""
    if filtered.empty:
        return ["No anomalies to evaluate — the filtered set is empty."]

    lines: list[str] = []
    rated = filtered[filtered["total_reviews"] > 0]
    if len(rated) < 3:
        lines.append(
            "Review depth is limited in this slice; cross-check statistically driven signals before stakeholder circulation."
        )

    if len(rated) >= 5:
        pk75, rt40 = rated["peak_players"].quantile(0.75), rated["rating_percent"].quantile(0.40)
        a = rated[(rated["peak_players"] >= pk75) & (rated["rating_percent"] <= rt40)]
        if len(a):
            row = a.nlargest(1, "peak_players").iloc[0]
            lines.append(
                f"**Traffic–sentiment gap:** **{row['game_name']}** peaks at **{_fmt_num(int(row['peak_players']))}** players "
                f"but positive share is only **{_fmt_pct(float(row['rating_percent']))}** — investigate reviews and update cadence."
            )

        rp75, pk40 = rated["rating_percent"].quantile(0.75), rated["peak_players"].quantile(0.40)
        b = rated[(rated["rating_percent"] >= rp75) & (rated["peak_players"] <= pk40)]
        if len(b):
            row = b.nlargest(1, "rating_percent").iloc[0]
            lines.append(
                f"**Under-distributed quality:** **{row['game_name']}** holds **{_fmt_pct(float(row['rating_percent']))}** "
                f"positive share with **{_fmt_num(int(row['peak_players']))}** peaks — distribution may lag product–market fit."
            )

        prem = rated[(rated["price_category"].isin(["High", "Medium"])) & (rated["price"] > 0)]
        if len(prem) >= 3:
            pmed = prem["rating_percent"].quantile(0.35)
            c = prem[(prem["total_reviews"] >= 12) & (prem["rating_percent"] <= pmed)]
            if len(c):
                row = c.nlargest(1, "price").iloc[0]
                lines.append(
                    f"**Premium pricing pressure:** **{row['game_name']}** (**${row['price']:.2f}**, {row['price_category']}) "
                    f"sits at **{_fmt_pct(float(row['rating_percent']))}** positive — validate feature promise vs. price point."
                )

    # Free + strong engagement
    free = filtered[filtered["price"] <= 0]
    if len(free) >= 3:
        fq = free["peak_players"].quantile(0.80)
        d = free[free["peak_players"] >= fq]
        if len(d):
            row = d.nlargest(1, "peak_players").iloc[0]
            lines.append(
                f"**Free-to-play gravity well:** **{row['game_name']}** (**{_fmt_num(int(row['peak_players']))}** peaks) "
                f"anchors engagement for F2P — benchmark beats, events, and cohort retention against this outlier."
            )

    # Genre: strong rating, limited supply vs peers
    g_stats = (
        rated.groupby("genre")
        .agg(avg_rating=("rating_percent", "mean"), titles=("game_name", "count"))
        .reset_index()
    )
    g_stats = g_stats[g_stats["titles"] >= 2]
    if len(g_stats) >= 3:
        ar_hi = g_stats["avg_rating"].quantile(0.70)
        ct_lo = g_stats["titles"].quantile(0.40)
        e = g_stats[(g_stats["avg_rating"] >= ar_hi) & (g_stats["titles"] <= ct_lo)]
        if len(e):
            gg = e.sort_values("avg_rating", ascending=False).iloc[0]
            lines.append(
                f"**Supply-constrained quality cluster:** **{gg['genre']}** averages **{_fmt_pct(float(gg['avg_rating']))}** "
                f"positive across **{int(gg['titles'])}** titles — fewer SKUs than peers with comparable sentiment strength."
            )

    if not lines:
        lines.append(
            "No extreme statistical contrasts detected under current thresholds; the selection appears relatively consistent."
        )
    return lines


def render_executive_ai_summary_block(parts: dict[str, str]) -> None:
    """Two-column analyst brief; uses Markdown so emphasis renders correctly."""
    st.caption("Rule-based synthesis on the filtered selection — no external model calls.")

    left = [
        ("market_overview", "Market overview"),
        ("strongest_genre_signal", "Strongest genre signal"),
        ("player_engagement_observation", "Player engagement"),
    ]
    right = [
        ("pricing_observation", "Pricing"),
        ("main_opportunity", "Main opportunity"),
        ("main_risk", "Main risk"),
    ]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        for key, title in left:
            st.markdown(f"**{title}**")
            st.markdown(parts[key])
    with c2:
        for key, title in right:
            st.markdown(f"**{title}**")
            st.markdown(parts[key])


def render_executive_through_opportunity_sections(filtered: pd.DataFrame, scored: pd.DataFrame) -> None:
    """
    Executive AI Summary, Smart Recommendations, Signals & Anomalies, Top 10 Opportunity Games.
    Placed **after** Charts and Top Performers (pre–Stage‑4 storefront layout).
    """
    render_executive_ai_summary_block(build_executive_ai_summary(filtered))

    st.markdown("##### Smart recommendations")
    st.caption("Deterministic guidance — swap for LLM-backed synthesis when APIs are enabled.")
    for r in build_smart_recommendations(filtered, scored):
        st.markdown(f"- {r}")

    st.markdown("##### Signals & anomalies")
    st.caption("Heuristic flags; tune thresholds as your dataset grows.")
    anomalies = build_anomaly_signals(filtered)
    for s in anomalies:
        st.markdown(f"- {s}")
    if anomalies:
        render_insight(
            "Negative or divergent signals above deserve a closer read — they often precede review or CCU inflection points.",
            variant="warn",
        )

    st.markdown("##### Top 10 opportunity games")
    st.caption("Ranked by composite Opportunity Score (0–100) within the current filters.")
    if scored.empty or scored["opportunity_score"].isna().all():
        st.info("Opportunity scores require at least one row in the filtered selection.")
    else:
        top10 = scored.nlargest(10, "opportunity_score")[
            [
                "game_name",
                "genre",
                "opportunity_score",
                "rating_percent",
                "peak_players",
                "total_reviews",
                "price",
                "price_category",
            ]
        ].copy()
        st.dataframe(format_display_df(top10), use_container_width=True, height=380)


def top_rated_games(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    r = df[df["total_reviews"] > 0].copy()
    if r.empty:
        return r
    r = r.sort_values(["rating_percent", "total_reviews"], ascending=[False, False]).head(n)
    return r[
        ["game_name", "genre", "rating_percent", "total_reviews", "peak_players", "price", "price_category"]
    ]


def top_peak_games(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if df.empty:
        return df
    t = df.nlargest(n, "peak_players")
    return t[["game_name", "genre", "peak_players", "rating_percent", "price", "price_category"]]


def top_underrated_budget_games(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Free/low price tier, solid review volume, favor high sentiment then smaller peaks — “hidden gems.”
    """
    low = df[df["price_category"].isin(["Free", "Low"])].copy()
    low = low[low["total_reviews"] >= 10]
    if low.empty:
        low = df[df["price_category"].isin(["Free", "Low"])].copy()
        low = low[low["total_reviews"] > 0]
    if low.empty:
        return low.iloc[0:0]
    low = low.sort_values(
        by=["rating_percent", "peak_players", "total_reviews"],
        ascending=[False, True, False],
    ).head(n)
    return low[
        ["game_name", "genre", "price", "price_category", "rating_percent", "total_reviews", "peak_players"]
    ]


def format_display_df(table: pd.DataFrame) -> pd.DataFrame:
    """Readable formatting for on-screen tables (copy)."""
    if table.empty:
        return table
    out = table.copy()
    if "rating_percent" in out.columns:
        out["rating_percent"] = out["rating_percent"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    if "price" in out.columns:
        out["price"] = out["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
    if "opportunity_score" in out.columns:
        out["opportunity_score"] = out["opportunity_score"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
        )
    for col in ("peak_players", "total_reviews"):
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    return out


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="filtered_games", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def resolve_platform_mode() -> str:
    sel = st.session_state.get(PLATFORM_RADIO_KEY, LABEL_MARKET)
    if sel == LABEL_LIVE:
        return MODE_LIVE
    if sel == LABEL_COMPARE:
        return MODE_COMPARE
    return MODE_MARKET


def render_platform_shell() -> str:
    """Persistent branding + horizontal module picker; honors Steam-themed styling."""
    if PLATFORM_RADIO_KEY not in st.session_state:
        st.session_state[PLATFORM_RADIO_KEY] = LABEL_MARKET
    st.markdown(
        '<div class="platform-shell">'
        '<div class="platform-shell-title">Steam AI Analytics Platform</div>'
        '<div class="platform-shell-tagline">Portfolio CSV analytics • Live Store intelligence • Comparative mode</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.radio(
        "Platform mode",
        options=[LABEL_MARKET, LABEL_LIVE, LABEL_COMPARE],
        horizontal=True,
        key=PLATFORM_RADIO_KEY,
    )
    return resolve_platform_mode()


def render_sidebar_market_intro() -> None:
    """Market module sidebar — mirrors legacy onboarding copy."""
    st.sidebar.markdown('<p class="sidebar-brand">Market Analytics</p>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p class="sidebar-sub">CSV-backed modeling • Filters propagate KPIs, charts, AI layers, export, and reviews CSV.</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Expected columns — games CSV")
    for c in EXPECTED_COLS:
        st.sidebar.code(c)
    with st.sidebar.expander("Reviews CSV columns (optional)"):
        st.caption("Second upload — e.g. `steam_reviews_sample.csv`")
        for c in EXPECTED_REVIEW_COLS:
            st.code(c)


def render_sidebar_live_intro() -> None:
    """Live module sidebar — scoped guidance only."""
    st.sidebar.markdown('<p class="sidebar-brand">Live Intelligence</p>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p class="sidebar-sub">Public Store appreviews • **steam_catalog_real.csv** enriches by App ID when present.</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="module-banner">Module isolation</div>', unsafe_allow_html=True)
    st.sidebar.caption(
        "Live fetch output never feeds Market KPIs or CSV charts. Optional CSV upload only overlays gaps after Steam catalog join."
    )


def render_dashboard_body(
    filtered: pd.DataFrame,
    cleaned_full: pd.DataFrame,
    *,
    reviews_enriched: pd.DataFrame | None = None,
    reviews_load_error: str | None = None,
    reviews_file_selected: bool = False,
) -> None:
    """
    Market module only: KPI → Charts → Top Performers → Executive AI … Top 10 Opportunity → Automated Insights →
    Dataset → Excel → optional Review Sentiment. Steam Live probes render exclusively under Live Intelligence mode.
    """
    if len(filtered) < len(cleaned_full):
        st.caption(f"Showing **{len(filtered):,}** of **{len(cleaned_full):,}** games after filters.")
    else:
        st.caption(f"Showing all **{len(filtered):,}** games.")

    scored = add_opportunity_score(filtered.copy())

    kpis = compute_kpis(filtered)

    render_section_header(
        "Executive summary",
        "Portfolio snapshot",
        "Headline metrics for your current filter — expand sections below for charts, AI briefs, and exports.",
    )
    kpi_specs = [
        ("Total games", f"{kpis['total_games']:,}"),
        ("Average rating", f"{kpis['avg_rating']:.1%}" if pd.notna(kpis["avg_rating"]) else "—"),
        ("Average price", f"${kpis['avg_price']:.2f}" if pd.notna(kpis["avg_price"]) else "—"),
        ("Total peak players", f"{kpis['total_peak']:,}"),
        ("Top genre", kpis["top_genre"] or "—"),
    ]
    cols = st.columns(5)
    for i, spec in enumerate(kpi_specs):
        with cols[i]:
            st.markdown(kpi_card_html(*spec), unsafe_allow_html=True)
    render_insight(market_kpi_summary(kpis))
    close_section_header()

    with st.expander("Market metrics", expanded=True):
        g1, g2 = st.columns(2, gap="large")
        with g1:
            render_chart_block(
                genre_game_counts(filtered),
                question="Which genres dominate the catalog?",
                insight=insight_for_genre_chart(filtered),
            )
        with g2:
            render_chart_block(
                top_games_peak(filtered),
                question="Which titles drive peak engagement?",
                insight=insight_for_peak_chart(filtered),
            )
        g3, g4 = st.columns(2, gap="large")
        with g3:
            render_chart_block(
                avg_rating_by_genre(filtered),
                question="Where is sentiment strongest by genre?",
                insight=insight_for_rating_chart(filtered),
            )
        with g4:
            render_chart_block(
                price_category_distribution(filtered),
                question="How is the portfolio split across price tiers?",
                insight=insight_for_price_chart(filtered),
            )

    with st.expander("Top performers", expanded=True):
        p1, p2, p3 = st.columns(3, gap="large")
        with p1:
            st.markdown("##### Highest rated")
            st.caption("Titles with review-backed positive share.")
            st.dataframe(format_display_df(top_rated_games(filtered)), use_container_width=True, height=240)
        with p2:
            st.markdown("##### Peak players")
            st.caption("Largest concurrent player peaks in scope.")
            st.dataframe(format_display_df(top_peak_games(filtered)), use_container_width=True, height=240)
        with p3:
            st.markdown("##### Underrated low-price")
            st.caption("Free/Low price with strong sentiment vs. modest peaks.")
            st.dataframe(
                format_display_df(top_underrated_budget_games(filtered)),
                use_container_width=True,
                height=260,
            )

    with st.expander("Executive intelligence · AI brief & opportunity", expanded=True):
        try:
            render_executive_through_opportunity_sections(filtered, scored)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Executive / opportunity sections could not render ({exc}).")

    with st.expander("Automated portfolio insights", expanded=False):
        for line in enhanced_insights(filtered):
            st.markdown(f"- {line}")

    with st.expander("Dataset & export", expanded=False):
        st.caption("Includes **opportunity_score** (0–100) for the active filter context.")
        st.dataframe(scored, use_container_width=True, height=380)
        st.download_button(
            label="Download filtered data as Excel",
            data=dataframe_to_excel_bytes(scored),
            file_name="steam_games_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=filtered.empty,
            key="download_filtered_games_excel",
        )

    if reviews_file_selected:
        with st.expander("Review sentiment intelligence", expanded=True):
            reviews_scoped = (
                scope_reviews_to_filtered_games(reviews_enriched, filtered)
                if reviews_enriched is not None
                else pd.DataFrame()
            )
            try:
                render_review_sentiment_intelligence(
                    reviews_enriched,
                    reviews_scoped,
                    filtered,
                    reviews_load_error,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Review Sentiment Intelligence hit an error ({exc}); games analytics above are unaffected.")


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_dark_theme_css()
    inject_dashboard_ux_css()

    active_mode = render_platform_shell()
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    if active_mode == MODE_MARKET:
        render_sidebar_market_intro()

        render_section_header(
            "Market analytics",
            "Portfolio workspace",
            "Filters steer KPIs, charts, AI summaries, opportunity scoring, export, and optional textual reviews.",
        )
        close_section_header()

        uploaded = st.file_uploader(
            "Upload games CSV",
            type=["csv"],
            key="market_module_games_csv",
            help="CSV should include the columns listed in the sidebar.",
        )
        uploaded_reviews = st.file_uploader(
            "Upload reviews CSV (optional)",
            type=["csv"],
            key="steam_reviews_csv",
            help="Use `steam_reviews_sample.csv` shape: game_name, review_text, sentiment.",
        )

        if uploaded is None:
            st.info("Upload a games CSV to load the Market Analytics Dashboard.")
            sample = pd.DataFrame(
                {
                    "game_name": ["Demo RPG", "Demo FPS"],
                    "genre": ["RPG", "Action"],
                    "price": ["19.99", "59.99"],
                    "positive_reviews": [1200, 800],
                    "negative_reviews": [150, 400],
                    "peak_players": [50000, 120000],
                    "release_date": ["2020-01-15", "2021-06-01"],
                }
            )
            st.markdown("**Sample structure** (first rows):")
            st.dataframe(sample, use_container_width=True)
            return

        try:
            uploaded.seek(0)
            raw_df = pd.read_csv(uploaded, encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001 — user uploads can be malformed
            st.error(f"Could not read CSV: {e}")
            return

        cleaned_full = clean_steam_games_df(raw_df)

        reviews_enriched: pd.DataFrame | None = None
        reviews_load_error: str | None = None
        reviews_file_selected = uploaded_reviews is not None

        if uploaded_reviews is not None:
            try:
                uploaded_reviews.seek(0)
                raw_rev = pd.read_csv(uploaded_reviews, encoding="utf-8-sig")
                cleaned_rev = clean_reviews_df(raw_rev)
                if cleaned_rev.empty:
                    reviews_load_error = (
                        "No valid review rows after cleaning. Confirm headers `game_name`, `review_text`, `sentiment` "
                        "(BOM / UTF-8 from Excel are supported)."
                    )
                else:
                    reviews_enriched = merge_reviews_with_games_catalog(cleaned_rev, cleaned_full)
            except Exception as e:  # noqa: BLE001 — optional upload may be malformed
                reviews_load_error = str(e)

        st.sidebar.markdown("---")
        filter_cfg = render_filter_controls(cleaned_full)
        filtered = apply_game_filters(cleaned_full, filter_cfg)

        render_dashboard_body(
            filtered,
            cleaned_full,
            reviews_enriched=reviews_enriched,
            reviews_load_error=reviews_load_error,
            reviews_file_selected=reviews_file_selected,
        )
        return

    if active_mode == MODE_COMPARE:
        from comparative_intelligence import render_compare_sidebar, render_comparative_intelligence_panel

        render_compare_sidebar()

        render_section_header(
            "Competitive intelligence",
            "Dual-title comparison",
            "Two App IDs, two live review pulls, and steam_catalog_real.csv enrichment — isolated from Market and single-game Live state.",
        )
        close_section_header()

        cmp_catalog_upload = st.file_uploader(
            "Optional games catalog CSV (same schema as Market — overlays gaps after Steam catalog join only)",
            type=["csv"],
            key="cmp_module_optional_catalog_csv",
            help="Optional. Does not affect Market dashboards. Used only for Comparative Intelligence joins.",
        )
        cmp_catalog_df = empty_games_catalog()
        if cmp_catalog_upload is not None:
            try:
                cmp_catalog_upload.seek(0)
                cmp_catalog_df = clean_steam_games_df(pd.read_csv(cmp_catalog_upload, encoding="utf-8-sig"))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Optional catalog could not be read ({exc}); proceeding without CSV overlay.")
                cmp_catalog_df = empty_games_catalog()

        cat_path_cmp = default_catalog_path()
        catalog_mtime_cmp = cat_path_cmp.stat().st_mtime if cat_path_cmp.is_file() else -1.0
        steam_catalog_cmp = load_cached_steam_catalog(catalog_mtime_cmp)

        try:
            render_comparative_intelligence_panel(cmp_catalog_df, steam_catalog_cmp)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Comparative Intelligence failed ({exc}). Switch modes or retry; Market and Live modules are unchanged.")
        return

    # --- Steam Live Review Intelligence (standalone module) ---
    render_sidebar_live_intro()

    render_section_header(
        "Review intelligence",
        "Live Store probe",
        "Fetch up to 5,000 recent Steam reviews (paginated, cached locally) — KPIs, sentiment, pain analytics, "
        "executive summary, and lexicon themes from shared NLP helpers.",
    )
    close_section_header()

    live_catalog_upload = st.file_uploader(
        "Optional games catalog CSV (genre / price enrichment for fetched rows only)",
        type=["csv"],
        key="live_module_optional_catalog_csv",
        help="Same schema as Market games CSV. Never merges into Market dashboards.",
    )
    catalog_df = empty_games_catalog()
    if live_catalog_upload is not None:
        try:
            live_catalog_upload.seek(0)
            catalog_df = clean_steam_games_df(pd.read_csv(live_catalog_upload, encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Optional catalog could not be read ({exc}); proceeding without enrichment joins.")
            catalog_df = empty_games_catalog()

    cat_path = default_catalog_path()
    catalog_mtime = cat_path.stat().st_mtime if cat_path.is_file() else -1.0
    steam_catalog_bundle = load_cached_steam_catalog(catalog_mtime)

    try:
        render_steam_live_lookup_panel(
            catalog_df,
            steam_catalog_real=steam_catalog_bundle,
            panel_title=None,
            lead_divider=False,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Steam Live Review Intelligence failed ({exc}).")


if __name__ == "__main__":
    main()
