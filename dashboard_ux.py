"""
Shared dashboard UX helpers — presentation only (no analytics logic).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DASH_CHART_HEIGHT = 360
DASH_CHART_MARGIN = dict(l=48, r=28, t=52, b=56)


def inject_dashboard_ux_css() -> None:
    """Complements app.py dark theme — section hierarchy, expanders, insights."""
    st.markdown(
        """
        <style>
            .dash-section {
                margin: 2rem 0 1.25rem 0;
                padding-top: 0.25rem;
            }
            .dash-section:first-of-type { margin-top: 0.5rem; }
            .dash-eyebrow {
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #7a8a9a;
                margin: 0 0 0.35rem 0;
            }
            .dash-section-title {
                font-size: 1.22rem;
                font-weight: 700;
                color: #e8f0f6;
                letter-spacing: -0.02em;
                margin: 0 0 0.35rem 0;
                line-height: 1.25;
            }
            .dash-section-sub {
                font-size: 0.86rem;
                color: #8f98a0;
                line-height: 1.5;
                margin: 0 0 1rem 0;
                max-width: 52rem;
            }
            .dash-insight {
                font-size: 0.84rem;
                color: #a8b8c6;
                line-height: 1.5;
                margin: 0.35rem 0 1rem 0;
                padding: 0.55rem 0.75rem;
                border-left: 3px solid #66c0f4;
                background: rgba(102, 192, 244, 0.06);
                border-radius: 0 8px 8px 0;
            }
            .dash-insight-warn {
                border-left-color: #d4a574;
                background: rgba(212, 165, 116, 0.08);
                color: #c5b8a8;
            }
            .dash-kpi-strip-caption {
                font-size: 0.82rem;
                color: #9aaab8;
                margin: 0.5rem 0 1.25rem 0;
            }
            div[data-testid="stExpander"] {
                background: rgba(22, 32, 45, 0.45);
                border: 1px solid #2a475e;
                border-radius: 12px;
                margin-bottom: 0.85rem;
            }
            div[data-testid="stExpander"] summary {
                font-weight: 600;
                color: #d0dce6;
                font-size: 0.95rem;
            }
            .chart-question {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #7a8a9a;
                margin: 0 0 0.25rem 0;
            }
            @media (max-width: 768px) {
                .dash-section-title { font-size: 1.08rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(
    eyebrow: str,
    title: str,
    subtitle: str = "",
    *,
    extra_class: str = "",
) -> None:
    cls = f"dash-section {extra_class}".strip()
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    st.markdown(f'<p class="dash-eyebrow">{_esc(eyebrow)}</p>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="dash-section-title">{_esc(title)}</h2>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="dash-section-sub">{_esc(subtitle)}</p>', unsafe_allow_html=True)


def close_section_header() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_insight(text: str, *, variant: str = "default") -> None:
    cls = "dash-insight" if variant == "default" else "dash-insight dash-insight-warn"
    st.markdown(f'<p class="{cls}">{text}</p>', unsafe_allow_html=True)


def finalize_dashboard_chart(fig: go.Figure, title: str, *, height: int | None = None) -> go.Figure:
    """Consistent Plotly styling — improved contrast and label spacing."""
    h = height or DASH_CHART_HEIGHT
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=14, color="#d8e4ec"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18, 28, 40, 0.5)",
        font=dict(color="#a8b8c6", size=11),
        height=h,
        margin=DASH_CHART_MARGIN,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            x=0,
            font=dict(size=10, color="#8f98a0"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(47, 74, 99, 0.35)",
            linecolor="rgba(47, 74, 99, 0.5)",
            tickfont=dict(size=10, color="#9aaab8"),
        ),
        yaxis=dict(
            gridcolor="rgba(47, 74, 99, 0.35)",
            linecolor="rgba(47, 74, 99, 0.5)",
            tickfont=dict(size=10, color="#9aaab8"),
        ),
    )
    return fig


def render_chart_block(
    fig: go.Figure,
    *,
    question: str,
    insight: str = "",
) -> None:
    """One chart + optional executive insight line."""
    if question:
        st.markdown(f'<p class="chart-question">{_esc(question)}</p>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    if insight:
        render_insight(insight)


def market_kpi_summary(kpis: dict[str, Any]) -> str:
    ar = kpis.get("avg_rating")
    rating = f"{ar:.0%} average positive share" if pd.notna(ar) else "rating n/a"
    ap = kpis.get("avg_price")
    price = f"${ap:.2f} avg price" if pd.notna(ap) else "pricing n/a"
    tg = kpis.get("top_genre") or "—"
    return (
        f"**{kpis.get('total_games', 0):,}** titles in view · {rating} · {price} · "
        f"strongest genre concentration: **{tg}**."
    )


def insight_for_genre_chart(df: pd.DataFrame) -> str:
    if df.empty:
        return "Which genres dominate the catalog slice?"
    top = df["genre"].value_counts()
    if top.empty:
        return "Genre mix is flat in this filter — broaden genre selection to compare segments."
    g, n = top.index[0], int(top.iloc[0])
    share = n / max(len(df), 1)
    return f"**{g}** leads with **{n}** titles ({share:.0%} of selection) — portfolio weight is concentrated here."


def insight_for_peak_chart(df: pd.DataFrame) -> str:
    if df.empty:
        return "Who drives peak concurrent player attention?"
    row = df.nlargest(1, "peak_players").iloc[0]
    return (
        f"**{row['game_name']}** tops peak CCU at **{int(row['peak_players']):,}** — "
        "use as the engagement benchmark for this filter."
    )


def insight_for_rating_chart(df: pd.DataFrame) -> str:
    rated = df[df["total_reviews"] > 0] if "total_reviews" in df.columns else df
    if rated.empty:
        return "Sentiment by genre needs titles with review volume."
    g = (
        rated.groupby("genre", as_index=False)["rating_percent"]
        .mean()
        .sort_values("rating_percent", ascending=False)
    )
    if g.empty:
        return "No rated genres in scope."
    best = g.iloc[0]
    worst = g.iloc[-1] if len(g) > 1 else best
    return (
        f"**{best['genre']}** shows the strongest positive ratio ({best['rating_percent']:.0%}); "
        f"**{worst['genre']}** trails at **{worst['rating_percent']:.0%}** — watch underperforming clusters."
    )


def insight_for_price_chart(df: pd.DataFrame) -> str:
    if df.empty:
        return "How is the portfolio distributed across price tiers?"
    dist = df["price_category"].value_counts(normalize=True)
    if dist.empty:
        return "Price tiers are not populated for this selection."
    tier = dist.index[0]
    return f"**{tier}** is the largest price tier ({dist.iloc[0]:.0%} of titles) — monetization strategy skews toward this band."


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
