"""
Review Sentiment Intelligence — rule-based NLP-style signals (no external APIs).
Joins textual reviews to the games catalog where game_name matches after normalization.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_ux import finalize_dashboard_chart, render_chart_block, render_insight

# Keyword lexicons for theme / risk detection (word-boundary scans)
POSITIVE_THEME_KEYWORDS = [
    "amazing",
    "fantastic",
    "great",
    "excellent",
    "beautiful",
    "replayability",
    "storytelling",
]
NEGATIVE_THEME_KEYWORDS = [
    "bugs",
    "unstable",
    "toxic",
    "crashes",
    "optimization",
    "performance",
    "difficult",
]

EXPECTED_REVIEW_COLS = ["game_name", "review_text", "sentiment"]

# Plot styling aligned with storefront dashboard
REV_CHART_MARGIN = dict(l=40, r=20, t=52, b=44)
REV_CHART_HEIGHT = 360


def _norm_game_name(val: Any) -> str:
    """Stable join key vs. catalog (strip whitespace, unify case)."""
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def _finalize_rev_fig(fig: go.Figure, title: str) -> go.Figure:
    return finalize_dashboard_chart(fig, title, height=REV_CHART_HEIGHT)


def _empty_rev_fig(title: str, message: str) -> go.Figure:
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
    return _finalize_rev_fig(fig, title)


def normalize_sentiment_label(raw: Any) -> str:
    """Map heterogeneous labels onto {positive, negative, neutral}."""
    if pd.isna(raw):
        return "neutral"
    s = str(raw).strip().lower()
    if s in {"positive", "pos", "+", "1", "true"}:
        return "positive"
    if s in {"negative", "neg", "-", "0", "false"}:
        return "negative"
    return "neutral"


def _sanitize_header(name: Any) -> str:
    """Strip BOM / whitespace — Windows editors often prepend \\ufeff to the first CSV column."""
    s = str(name).strip().lower().replace(" ", "_")
    return s.lstrip("\ufeff").strip()


def clean_reviews_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns and strip content; tolerant of extra columns."""
    df = raw.copy()
    df.columns = [_sanitize_header(c) for c in df.columns]
    for c in EXPECTED_REVIEW_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df["game_name"] = df["game_name"].astype(str).str.strip()
    df.loc[df["game_name"].str.lower().isin(("nan", "none", "")), "game_name"] = pd.NA
    df["review_text"] = df["review_text"].astype(str)
    df["sentiment_polarity"] = df["sentiment"].map(normalize_sentiment_label)

    df = df.dropna(subset=["game_name"]).reset_index(drop=True)
    df["game_norm"] = df["game_name"].map(_norm_game_name)
    return df


def merge_reviews_with_games_catalog(
    reviews_clean: pd.DataFrame,
    games_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join textual reviews onto the **full** games catalog once.
    Exposes genre / pricing for thematic analysis without duplicating ingestion logic.
    """
    gg = games_catalog.copy()
    gg["game_norm"] = gg["game_name"].map(_norm_game_name)
    keep = ["game_norm", "genre", "price", "price_category", "rating_percent"]
    use = gg[[c for c in keep if c in gg.columns]].drop_duplicates("game_norm", keep="last")

    out = reviews_clean.merge(use, how="left", on="game_norm", suffixes=("", "_catalog"))
    out["catalog_match"] = out["genre"].notna()
    return out


def scope_reviews_to_filtered_games(enriched_reviews: pd.DataFrame, filtered_games: pd.DataFrame) -> pd.DataFrame:
    """Retain only reviews referencing titles present in the current dashboard filter."""
    fg = {_norm_game_name(x) for x in filtered_games["game_name"].dropna()}
    return enriched_reviews[enriched_reviews["game_norm"].isin(fg)].reset_index(drop=True)


def keyword_hit_counts(series: pd.Series, keywords: list[str]) -> dict[str, int]:
    """Count reviews whose text mentions each keyword (case-insensitive, word-ish boundaries)."""
    if not keywords:
        return {}
    text = series.astype(str).str.lower()
    out: dict[str, int] = {}
    for kw in keywords:
        pat = rf"(?<![a-z]){re.escape(kw)}(?![a-z])"
        out[kw] = int(text.str.contains(pat, regex=True, na=False).sum())
    return out


def review_kpis(scope: pd.DataFrame) -> dict[str, Any]:
    """Totals and headline ratios used in KPI tiles."""
    n = len(scope)
    if n == 0:
        return {
            "total": 0,
            "pos_pct": float("nan"),
            "neg_pct": float("nan"),
            "most_discussed": "—",
        }
    vc = scope["sentiment_polarity"].value_counts()
    pos_n = int(vc.get("positive", 0))
    neg_n = int(vc.get("negative", 0))
    return {
        "total": n,
        "pos_pct": 100.0 * pos_n / n,
        "neg_pct": 100.0 * neg_n / n,
        "most_discussed": scope["game_name"].value_counts().index[0],
    }


def figure_sentiment_pie(scope: pd.DataFrame) -> go.Figure:
    if scope.empty:
        return _empty_rev_fig("Sentiment composition", "No reviews for current filters.")
    vc = scope["sentiment_polarity"].value_counts().reindex(["positive", "negative", "neutral"]).fillna(0)
    vc = vc[vc > 0]
    if vc.sum() == 0:
        return _empty_rev_fig("Sentiment composition", "No polarity labels populated.")
    colors = {"positive": "#3fb950", "negative": "#f85149", "neutral": "#8f98a0"}
    fig = px.pie(
        names=vc.index,
        values=vc.values,
        title="",
        hole=0.38,
        color=vc.index,
        color_discrete_map=colors,
    )
    fig.update_traces(textinfo="percent+label")
    return _finalize_rev_fig(fig, "Positive vs Negative (labeled reviews)")


def figure_sentiment_by_game(scope: pd.DataFrame, top_k: int = 12) -> go.Figure:
    if scope.empty:
        return _empty_rev_fig("Sentiment by game", "No scoped reviews.")

    vc = scope["game_name"].value_counts().head(top_k).index.tolist()
    sub = scope[scope["game_name"].isin(vc)]
    tally = (
        sub.groupby(["game_name", "sentiment_polarity"], observed=False).size().reset_index(name="reviews")
    )
    order = sorted(vc)
    fig = px.bar(
        tally,
        x="game_name",
        y="reviews",
        color="sentiment_polarity",
        category_orders={"game_name": order},
        labels={"reviews": "# reviews", "game_name": "Game", "sentiment_polarity": "Sentiment"},
        color_discrete_map={"positive": "#3fb950", "negative": "#f85149", "neutral": "#8f98a0"},
        title="",
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickangle=28)
    return _finalize_rev_fig(fig, f"Sentiment mix — top {len(vc)} discussed titles")


def figure_most_reviewed_games(scope: pd.DataFrame, head: int = 12) -> go.Figure:
    if scope.empty:
        return _empty_rev_fig("Most reviewed titles", "No scoped reviews.")
    cts = scope["game_name"].value_counts().reset_index().head(head)
    cts.columns = ["game", "reviews"]
    fig = px.bar(
        cts,
        x="reviews",
        y="game",
        orientation="h",
        color="reviews",
        color_continuous_scale=["#1b2838", "#66c0f4"],
        labels={"reviews": "# reviews"},
        title="",
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    fig.update_coloraxes(colorbar=dict(title="", tickfont=dict(size=11)))
    return _finalize_rev_fig(fig, "Most reviewed titles (volume)")


def figure_sentiment_by_genre(scope: pd.DataFrame) -> go.Figure:
    scoped = scope[scope["sentiment_polarity"].isin(("positive", "negative"))].copy()
    scoped["genre_disp"] = scoped["genre"].fillna("Unknown (unmatched)")
    agg = scoped.groupby(["genre_disp", "sentiment_polarity"], observed=False).size().reset_index(name="n")
    if agg.empty:
        return _empty_rev_fig("Sentiment × genre", "Need genres from catalog join + polarity labels.")

    fig = px.bar(
        agg,
        x="genre_disp",
        y="n",
        color="sentiment_polarity",
        barmode="group",
        labels={"genre_disp": "Genre", "n": "Reviews", "sentiment_polarity": "Sentiment"},
        color_discrete_map={"positive": "#3fb950", "negative": "#f85149"},
        title="",
    )
    fig.update_layout(bargap=0.18)
    fig.update_xaxes(tickangle=22)
    return _finalize_rev_fig(fig, "Positive vs Negative volume by genre")


def top_keyword_themes(hit_map: dict[str, int]) -> pd.DataFrame:
    """Descending frequency table suitable for Theme tables."""
    cols = ["theme", "mentions"]
    if not hit_map:
        return pd.DataFrame(columns=cols)
    rows = [{"theme": k, "mentions": int(v)} for k, v in hit_map.items()]
    out = pd.DataFrame(rows).sort_values("mentions", ascending=False).reset_index(drop=True)
    return out


def synthesize_review_insights(scope: pd.DataFrame) -> list[str]:
    """Deterministic briefing lines — reads like analyst commentary."""
    bullets: list[str] = []
    if scope.empty:
        return ["Upload and scope reviews aligned with filtered catalog titles to unlock narrative cues."]

    pos_hits_scope = keyword_hit_counts(scope["review_text"], POSITIVE_THEME_KEYWORDS)
    narr = scope[scope["review_text"].str.contains("storytell", case=False, na=False)]
    rpg_any = scope[scope["genre"].fillna("").str.upper().eq("RPG")]
    if len(rpg_any) and len(narr):
        bullets.append(
            "Players repeatedly surface **story-driven praise** alongside RPG classifications — "
            "position upcoming narrative beats as differentiated value in storefront copy."
        )
    if pos_hits_scope.get("replayability", 0) >= 2 or (
        scope["review_text"].str.contains(r"(?<![a-z])replay", case=False, regex=True).sum() >= 2
    ):
        bullets.append(
            "**Replayability** language clusters with favorable polarity — roadmap communications should cite "
            "systems depth and run variety where applicable."
        )

    compish = scope[scope["genre"].fillna("").str.upper().isin({"ACTION", "MOBA", "SPORTS"})]
    neg_comp = compish.loc[compish["sentiment_polarity"].eq("negative"), "review_text"]
    toxic_hits_neg = keyword_hit_counts(neg_comp, NEGATIVE_THEME_KEYWORDS)
    toxic_total = toxic_hits_neg.get("toxic", 0)
    if toxic_total >= 2 or toxic_hits_neg.get("bugs", 0) + toxic_total >= 3:
        bullets.append(
            "Competitive-adjacent segments show comparatively **elevated toxicity and stability keywords** "
            "in downside reviews — invest in moderation telemetry and escalation paths pre-launch."
        )

    premium = scope[scope["price_category"].isin(["High", "Medium"])]
    perf_neg = premium[
        premium["sentiment_polarity"].eq("negative")
        & premium["review_text"].str.contains(r"performance|optimization|crash|unstable", case=False, regex=True)
    ]
    if len(perf_neg):
        bullets.append(
            "**Performance optics** collide with premium price bands in surfaced negatives — QA sign-off "
            "on frame pacing and patching cadence materially de-risks premium perception."
        )

    if bullets:
        bullets.append(
            "Signals are lexical heuristics; validate with moderated sampling before executive commitments "
            "(no external NLP API attached in this MVP build)."
        )
    else:
        bullets.append(
            "Lexical themes are subdued in this filtered slice — expand review volume or relax filters "
            "to strengthen qualitative coverage."
        )
    return bullets[:7]


def detect_risk_signals(scope: pd.DataFrame) -> list[str]:
    """Translate keyword density into actionable risk shorthand."""
    if scope.empty:
        return []

    alerts: list[str] = []
    neg_scope = scope[scope["sentiment_polarity"].eq("negative")]
    nh = keyword_hit_counts(neg_scope["review_text"], ["unstable", "crashes"]) if len(neg_scope) else {}

    blob_all = keyword_hit_counts(scope["review_text"], NEGATIVE_THEME_KEYWORDS)

    if blob_all.get("unstable", 0) + nh.get("unstable", 0) >= 2 or blob_all.get("crashes", 0) >= 2:
        alerts.append(
            "**Server / client instability cues** detected (unstable servers, crashing sessions) "
            "— prioritize infra postmortems and outage comms readiness."
        )
    if blob_all.get("toxic", 0) >= 2:
        alerts.append(
            "**Community toxicity narratives** recur — escalate trust & safety playbook and escalation SLAs "
            "for flagged comms queues."
        )
    if blob_all.get("optimization", 0) + blob_all.get("performance", 0) >= 2:
        alerts.append(
            "**Optimization friction** surfaced across multiple titles — benchmarking pass on min-spec "
            "configs and DX12/Vulkan regressions advised."
        )
    if blob_all.get("bugs", 0) >= 2:
        alerts.append(
            "**Bug density** materially visible in verbatim text — triage reproducible regressions vs. QoL noise."
        )
    if blob_all.get("difficult", 0) >= 3:
        alerts.append(
            "**Difficulty spikes** resonate in verbatim feedback — reconsider onboarding curves "
            "or optional assist toggles depending on SKU positioning."
        )
    return alerts[:8]


def _insight_sentiment_pie(scope: pd.DataFrame) -> str:
    if scope.empty:
        return "Upload reviews with polarity labels to see sentiment mix."
    vc = scope["sentiment_polarity"].value_counts()
    neg = int(vc.get("negative", 0))
    pos = int(vc.get("positive", 0))
    n = len(scope)
    if neg / max(n, 1) >= 0.45:
        return f"**Negative tone is elevated** ({100 * neg / n:.0f}% of scoped reviews) — prioritize root-cause themes below."
    if pos / max(n, 1) >= 0.6:
        return f"**Positive sentiment leads** ({100 * pos / n:.0f}% positive) — monitor whether volume is concentrated on a few titles."
    return f"Sentiment is mixed across **{n:,}** reviews — use per-game and genre splits to localize friction."


def _insight_top_game(scope: pd.DataFrame) -> str:
    if scope.empty:
        return "Which titles absorb the most review volume?"
    top = scope["game_name"].value_counts()
    g, c = top.index[0], int(top.iloc[0])
    return f"**{g}** accounts for **{c}** reviews ({c / len(scope):.0%} of scope) — disproportionate discussion may mask portfolio-wide trends."


def _insight_genre_sentiment(scope: pd.DataFrame) -> str:
    scoped = scope[scope["sentiment_polarity"].isin(("positive", "negative"))].copy()
    if scoped.empty or "genre" not in scoped.columns:
        return "Genre sentiment needs catalog-matched rows with polarity labels."
    scoped["genre_disp"] = scoped["genre"].fillna("Unknown")
    agg = scoped.groupby("genre_disp")["sentiment_polarity"].apply(
        lambda s: (s == "negative").mean()
    )
    if agg.empty:
        return "No genre-level polarity signal in this slice."
    worst = agg.idxmax()
    return (
        f"**{worst}** shows the highest negative share ({agg.max():.0%}) among matched genres — "
        "validate whether this is SKU-specific or genre-wide."
    )


def render_review_sentiment_intelligence(
    reviews_enriched_global: pd.DataFrame | None,
    reviews_scoped_to_filter: pd.DataFrame,
    filtered_games: pd.DataFrame,
    load_error: str | None = None,
) -> None:
    """Streamlit-rendered section; callers supply already-clean + merged frames."""
    import streamlit as st

    st.caption(
        f"Textual reviews merged to the catalog where `game_name` aligns · "
        f"**{len(filtered_games):,}** titles in the active portfolio filter."
    )

    if load_error:
        st.error(f"**Reviews file could not be processed:** {load_error}")
        return

    if reviews_enriched_global is None:
        st.warning("A reviews CSV was flagged for upload but no merged dataset was produced — try re-uploading the file.")
        return

    if reviews_enriched_global.empty:
        st.warning("The reviews file was loaded but produced **no usable rows** after cleaning (check game names and text).")
        return

    unmatched = int((~reviews_enriched_global["catalog_match"]).sum()) if "catalog_match" in reviews_enriched_global.columns else 0
    if unmatched:
        st.caption(
            f"**Catalog alignment:** `{unmatched}` review rows have no deterministic match in the uploaded catalog."
        )

    scope_strict = reviews_scoped_to_filter
    # Filters may exclude every titled review; fall back so charts still render from the merged upload slice.
    if scope_strict.empty:
        scope = reviews_enriched_global.copy()
        st.warning(
            "**No reviews fall inside the active game filters** — widening genre/year/price or clearing search will rescope rows. "
            "Below, visuals temporarily use **all textual reviews merged to the catalog** from your upload."
        )
    else:
        scope = scope_strict

    kp = review_kpis(scope)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Total reviews</div>"
            f"<div class='kpi-value'>{kp['total']:,}</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        pct = "—" if pd.isna(kp["pos_pct"]) else f"{kp['pos_pct']:.1f}%"
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Positive reviews</div>"
            f"<div class='kpi-value'>{pct}</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        pctn = "—" if pd.isna(kp["neg_pct"]) else f"{kp['neg_pct']:.1f}%"
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Negative reviews</div>"
            f"<div class='kpi-value'>{pctn}</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        md = kp["most_discussed"]
        if len(md) > 36:
            md = md[:33] + "…"
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Most discussed game</div>"
            f"<div class='kpi-value' style='font-size:1rem'>{md}</div></div>",
            unsafe_allow_html=True,
        )

    neg_pct = kp.get("neg_pct")
    if pd.notna(neg_pct) and float(neg_pct) >= 40:
        render_insight(
            f"Negative share is **{neg_pct:.1f}%** in this scope — expand risk signals and theme tables for drivers.",
            variant="warn",
        )

    with st.expander("Sentiment visuals", expanded=True):
        v1, v2 = st.columns(2, gap="large")
        with v1:
            render_chart_block(
                figure_sentiment_pie(scope),
                question="What is the overall sentiment mix?",
                insight=_insight_sentiment_pie(scope),
            )
            render_chart_block(
                figure_most_reviewed_games(scope),
                question="Which titles drive review volume?",
                insight=_insight_top_game(scope),
            )
        with v2:
            render_chart_block(
                figure_sentiment_by_game(scope),
                question="How does sentiment split across top games?",
                insight="Stacked bars show whether negativity clusters on one SKU or is portfolio-wide.",
            )
            render_chart_block(
                figure_sentiment_by_genre(scope),
                question="Where is sentiment weakest by genre?",
                insight=_insight_genre_sentiment(scope),
            )

    pos_full = keyword_hit_counts(scope["review_text"], POSITIVE_THEME_KEYWORDS)
    neg_full = keyword_hit_counts(scope["review_text"], NEGATIVE_THEME_KEYWORDS)

    with st.expander("Keyword & theme diagnostics", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Top positive themes**")
            st.dataframe(top_keyword_themes(pos_full), use_container_width=True, hide_index=True, height=220)
        with t2:
            st.markdown("**Top negative themes**")
            st.dataframe(top_keyword_themes(neg_full), use_container_width=True, hide_index=True, height=220)

    with st.expander("AI-generated review insights", expanded=False):
        st.caption("Offline rules from sentiment labels, lexical themes, genres, and price tier context.")
        for line in synthesize_review_insights(scope):
            st.markdown(f"- {line}")

    risks = detect_risk_signals(scope)
    with st.expander("Risk signals from reviews", expanded=bool(risks)):
        if risks:
            for line in risks:
                st.markdown(f"- {line}")
        else:
            st.markdown("- **No prioritized lexical risk banners** surfaced for this scoped slice.")

    with st.expander("Scoped review excerpts (inspect join quality)", expanded=False):
        show_cols = ["game_name", "genre", "sentiment_polarity", "catalog_match", "review_text"]
        show_cols = [c for c in show_cols if c in scope.columns]
        st.dataframe(scope[show_cols], use_container_width=True, height=240)
