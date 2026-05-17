"""
Module 3 — Comparative Intelligence.

Side-by-side live review + catalog comparison for two Steam App IDs.
Isolated from Market Analytics and single-game Live Intelligence UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from review_sentiment import keyword_hit_counts
from steam_catalog_enrichment import lookup_catalog_profile
from steam_live_executive import LiveExecutiveSnapshot, compute_live_executive_snapshot
from steam_live_lookup import (
    build_live_enriched_for_insights,
    fetch_app_display_name,
    fetch_live_reviews_pipeline,
    parse_app_id_input,
)

# Lexicon groups aligned with Steam Live executive volatility scans
_CMP_INSTABILITY = ["unstable", "crash", "crashes", "disconnect", "lag", "server", "freeze", "stutter"]
_CMP_TOXICITY = ["toxic", "cheat", "cheaters", "grief", "harassment"]
_CMP_PERFORMANCE = ["performance", "optimization", "fps", "stutter", "frame"]


@dataclass
class ComparativeGameSlice:
    app_id: int
    display_name: str
    df_live: pd.DataFrame
    enriched: pd.DataFrame
    catalog_profile: pd.Series | None
    snap: LiveExecutiveSnapshot | None
    hard_error: str | None
    catalog_missing: bool
    no_reviews: bool


def render_compare_sidebar() -> None:
    import streamlit as st

    st.sidebar.markdown('<p class="sidebar-brand">Comparative Intelligence</p>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p class="sidebar-sub">Two App IDs • Live Store reviews • **steam_catalog_real.csv** match by ID.</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="module-banner">Module isolation</div>', unsafe_allow_html=True)
    st.sidebar.caption(
        "Compare mode never writes into Market CSV pipelines or Live session state. Fetches are scoped to this screen."
    )


def inject_comparative_css() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
            .cmp-shell { margin: 0.5rem 0 1.75rem 0; }
            .cmp-banner {
                font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.11em;
                color: #8f98a0; margin-bottom: 0.35rem;
            }
            .cmp-title {
                font-size: 1.28rem; font-weight: 700; color: #e5eef5;
                letter-spacing: -0.02em; margin-bottom: 1rem;
                padding-bottom: 0.45rem; border-bottom: 1px solid #2a475e;
            }
            .cmp-grid-2 {
                display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
                margin-bottom: 1.25rem;
            }
            @media (max-width: 900px) { .cmp-grid-2 { grid-template-columns: 1fr; } }
            .cmp-card {
                background: linear-gradient(165deg, #1a2636 0%, #141c28 55%, #101820 100%);
                border: 1px solid #2f4a63; border-radius: 14px;
                padding: 1.1rem 1.2rem 1.15rem 1.2rem;
                box-shadow: 0 12px 36px rgba(0,0,0,0.4);
            }
            .cmp-card h3 {
                font-size: 0.95rem; color: #66c0f4; margin: 0 0 0.75rem 0;
                font-weight: 600;
            }
            .cmp-row {
                display: flex; justify-content: space-between; gap: 0.75rem;
                padding: 0.38rem 0; border-bottom: 1px solid rgba(42, 71, 94, 0.55);
                font-size: 0.88rem; color: #c5d4e0;
            }
            .cmp-row:last-child { border-bottom: none; }
            .cmp-row span:first-child { color: #8f98a0; flex: 0 0 42%; }
            .cmp-row span:last-child { text-align: right; color: #dfe6ea; font-weight: 500; }
            .cmp-block {
                background: rgba(20, 30, 44, 0.55); border: 1px solid #2a475e;
                border-radius: 12px; padding: 1rem 1.15rem; margin-bottom: 1rem;
            }
            .cmp-block h4 { margin: 0 0 0.55rem 0; font-size: 0.95rem; color: #c5d4e0; }
            .cmp-block p, .cmp-block li { color: #b4c4d4; font-size: 0.9rem; line-height: 1.55; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _count_kw_group(series: pd.Series, words: list[str]) -> int:
    if series.empty or not len(words):
        return 0
    return sum(keyword_hit_counts(series, [w]).get(w, 0) for w in words)


def _text_series(g: ComparativeGameSlice) -> pd.Series:
    if not g.enriched.empty and "review_text" in g.enriched.columns:
        return g.enriched["review_text"]
    if not g.df_live.empty and "review_text" in g.df_live.columns:
        return g.df_live["review_text"]
    return pd.Series(dtype=str)


def _norm_catalog_str(profile: pd.Series | None, key: str, fallback: str = "—") -> str:
    if profile is None:
        return fallback
    v = profile.get(key)
    if pd.isna(v) or v is None or str(v).strip() == "":
        return fallback
    return str(v).strip()


def _norm_price(profile: pd.Series | None) -> str:
    if profile is None:
        return "—"
    try:
        return f"${float(profile.get('price', 0) or 0):.2f}"
    except (TypeError, ValueError):
        return "—"


def _norm_peak(profile: pd.Series | None) -> str:
    if profile is None:
        return "—"
    try:
        return f"{int(profile.get('peak_players', 0) or 0):,}"
    except (TypeError, ValueError):
        return "—"


def _sentiment_pct(df: pd.DataFrame, which: str) -> str:
    if df.empty or "sentiment" not in df.columns:
        return "—"
    n = len(df)
    if n == 0:
        return "—"
    c = (df["sentiment"].astype(str).str.lower() == which).sum()
    return f"{100.0 * int(c) / n:.1f}%"


def _risk_to_score(level: str | None) -> float:
    if not level:
        return 0.0
    return {"Low": 22.0, "Medium": 55.0, "High": 88.0}.get(level, 35.0)


def _engagement_numeric(snap: LiveExecutiveSnapshot | None, profile: pd.Series | None) -> float:
    if profile is None:
        return 0.0
    try:
        pk = float(profile.get("peak_players", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    if pk <= 0 and snap:
        # approximate from engagement label when CCU unknown
        lab = snap.engagement_strength.lower()
        if "very high" in lab:
            return 150_000.0
        if "high" in lab and "sample" not in lab:
            return 40_000.0
        if "moderate" in lab:
            return 12_000.0
        if "building" in lab:
            return 2_000.0
        if "limited" in lab:
            return 400.0
    return max(pk, 0.0)


def load_comparative_game_slice(
    app_id: int,
    steam_catalog_real: pd.DataFrame | None,
    optional_games_catalog: pd.DataFrame,
) -> ComparativeGameSlice:
    """Fetch live reviews + enrichment + executive snapshot for one App ID."""
    sc = steam_catalog_real if steam_catalog_real is not None else pd.DataFrame()
    profile = lookup_catalog_profile(sc, app_id) if not sc.empty else None
    catalog_missing = sc.empty or profile is None

    df, err = fetch_live_reviews_pipeline(app_id)
    if err:
        nm = fetch_app_display_name(app_id)
        if profile is not None and str(profile.get("game_name") or "").strip():
            nm = str(profile.get("game_name")).strip()
        return ComparativeGameSlice(
            app_id=app_id,
            display_name=nm,
            df_live=pd.DataFrame(),
            enriched=pd.DataFrame(),
            catalog_profile=profile,
            snap=None,
            hard_error=err,
            catalog_missing=catalog_missing,
            no_reviews=False,
        )

    assert df is not None
    if df.empty or "sentiment" not in df.columns:
        nm = fetch_app_display_name(app_id)
        if profile is not None and str(profile.get("game_name") or "").strip():
            nm = str(profile.get("game_name")).strip()
        return ComparativeGameSlice(
            app_id=app_id,
            display_name=nm,
            df_live=df,
            enriched=pd.DataFrame(),
            catalog_profile=profile,
            snap=None,
            hard_error=None,
            catalog_missing=catalog_missing,
            no_reviews=True,
        )

    enriched = build_live_enriched_for_insights(df, steam_catalog_real, optional_games_catalog)
    snap = compute_live_executive_snapshot(df, enriched, profile)
    name = str(df["game_name"].iloc[0]) if "game_name" in df.columns else fetch_app_display_name(app_id)

    return ComparativeGameSlice(
        app_id=app_id,
        display_name=name,
        df_live=df,
        enriched=enriched,
        catalog_profile=profile,
        snap=snap,
        hard_error=None,
        catalog_missing=catalog_missing,
        no_reviews=False,
    )


def _metric_row(label: str, value: str) -> str:
    esc = (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f'<div class="cmp-row"><span>{label}</span><span>{esc}</span></div>'


def _render_game_card(title: str, g: ComparativeGameSlice) -> str:
    if g.hard_error:
        body = f'<div class="cmp-row"><span>Status</span><span>{_html(g.hard_error)}</span></div>'
        return f'<div class="cmp-card"><h3>{_html(title)}</h3>{body}</div>'

    prof = g.catalog_profile
    snap = g.snap
    gname = _norm_catalog_str(prof, "game_name", g.display_name)
    genre = _norm_catalog_str(prof, "genre")
    price = _norm_price(prof)
    peak = _norm_peak(prof)
    pos_p = _sentiment_pct(g.df_live, "positive")
    neg_p = _sentiment_pct(g.df_live, "negative")

    rows = [
        _metric_row("Game name", gname[:72] + ("…" if len(gname) > 72 else "")),
        _metric_row("Genre", genre),
        _metric_row("Price", price),
        _metric_row("Peak players", peak),
        _metric_row("Positive %", pos_p),
        _metric_row("Negative %", neg_p),
    ]
    if g.no_reviews:
        rows.append(_metric_row("Reviews", "No reviews in this pull"))
    if snap:
        rows.extend(
            [
                _metric_row("Health score", str(snap.health_score)),
                _metric_row("Risk level", snap.risk_level),
                _metric_row("Engagement strength", snap.engagement_strength),
                _metric_row("Community status", snap.community_sentiment_status),
                _metric_row("Market position", snap.market_position[:80] + ("…" if len(snap.market_position) > 80 else "")),
            ]
        )
    else:
        rows.append(_metric_row("Executive metrics", "— (no review sample)"))

    if g.catalog_missing:
        rows.append(_metric_row("Catalog", "No steam_catalog_real match"))

    return f'<div class="cmp-card"><h3>{_html(title)}</h3>{"".join(rows)}</div>'


def _html(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _cmp_plot_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=15, color="#dfe6ea")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,30,42,0.55)",
        font=dict(color="#b8c6d1", size=12),
        height=320,
        margin=dict(l=48, r=24, t=52, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0),
    )
    return fig


def _fig_sentiment_pair(a: ComparativeGameSlice, b: ComparativeGameSlice, na: str, nb: str) -> go.Figure:
    def pct(df: pd.DataFrame, w: str) -> float:
        if df.empty or "sentiment" not in df.columns or len(df) == 0:
            return 0.0
        return 100.0 * float((df["sentiment"].astype(str).str.lower() == w).sum()) / float(len(df))

    fig = go.Figure(
        data=[
            go.Bar(name="Positive %", x=[na, nb], y=[pct(a.df_live, "positive"), pct(b.df_live, "positive")], marker_color="#3fb950"),
            go.Bar(name="Negative %", x=[na, nb], y=[pct(a.df_live, "negative"), pct(b.df_live, "negative")], marker_color="#f85149"),
        ]
    )
    fig.update_layout(barmode="group")
    return _cmp_plot_layout(fig, "Positive vs negative (live sample)")


def _fig_health_pair(a: ComparativeGameSlice, b: ComparativeGameSlice, na: str, nb: str) -> go.Figure:
    ha = float(a.snap.health_score) if a.snap else 0.0
    hb = float(b.snap.health_score) if b.snap else 0.0
    fig = go.Figure(data=[go.Bar(x=[na, nb], y=[ha, hb], marker_color="#66c0f4", text=[ha, hb], textposition="outside")])
    return _cmp_plot_layout(fig, "Health score comparison")


def _fig_engagement_pair(a: ComparativeGameSlice, b: ComparativeGameSlice, na: str, nb: str) -> go.Figure:
    ya = _engagement_numeric(a.snap, a.catalog_profile)
    yb = _engagement_numeric(b.snap, b.catalog_profile)
    fig = go.Figure(
        data=[
            go.Bar(
                x=[na, nb],
                y=[ya, yb],
                marker_color="#a371f7",
                text=[f"{ya:,.0f}" if ya >= 1000 else f"{ya:.0f}", f"{yb:,.0f}" if yb >= 1000 else f"{yb:.0f}"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(yaxis_title="Peak players (catalog)")
    return _cmp_plot_layout(fig, "Engagement comparison (peak CCU)")


def _fig_risk_pair(a: ComparativeGameSlice, b: ComparativeGameSlice, na: str, nb: str) -> go.Figure:
    ra = _risk_to_score(a.snap.risk_level if a.snap else None)
    rb = _risk_to_score(b.snap.risk_level if b.snap else None)
    fig = go.Figure(
        data=[
            go.Bar(
                x=[na, nb],
                y=[ra, rb],
                marker_color=["#3fb950" if ra < 40 else "#d29922" if ra < 70 else "#f85149", "#3fb950" if rb < 40 else "#d29922" if rb < 70 else "#f85149"],
                text=[a.snap.risk_level if a.snap else "—", b.snap.risk_level if b.snap else "—"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(yaxis_title="Risk intensity (ordinal scale)")
    return _cmp_plot_layout(fig, "Risk level comparison")


def _volatility_count(g: ComparativeGameSlice) -> int:
    if g.snap:
        return len(g.snap.volatility_flags)
    return 0


def _risk_profile(g: ComparativeGameSlice) -> dict[str, Any]:
    ts = _text_series(g)
    return {
        "toxicity_hits": _count_kw_group(ts, _CMP_TOXICITY),
        "instability_hits": _count_kw_group(ts, _CMP_INSTABILITY),
        "performance_hits": _count_kw_group(ts, _CMP_PERFORMANCE),
        "volatility_flags": _volatility_count(g),
        "pos_pct": float(g.snap.pos_ratio_pct) if g.snap else (_parse_pct(_sentiment_pct(g.df_live, "positive"))),
        "neg_pct": float(g.snap.neg_ratio_pct) if g.snap else (_parse_pct(_sentiment_pct(g.df_live, "negative"))),
    }


def _parse_pct(s: str) -> float:
    if s == "—":
        return float("nan")
    try:
        return float(s.replace("%", "").strip())
    except ValueError:
        return float("nan")


def _comparative_executive_summary(ga: ComparativeGameSlice, gb: ComparativeGameSlice) -> list[str]:
    na, nb = ga.display_name, gb.display_name
    if len(na) > 42:
        na = na[:39] + "…"
    if len(nb) > 42:
        nb = nb[:39] + "…"
    lines: list[str] = []
    pa = _engagement_numeric(ga.snap, ga.catalog_profile)
    pb = _engagement_numeric(gb.snap, gb.catalog_profile)
    ta = _risk_profile(ga)["toxicity_hits"]
    tb = _risk_profile(gb)["toxicity_hits"]
    if pa > pb * 1.4 and ta > tb + 1:
        lines.append(f"{na} shows stronger engagement but higher toxicity risk than {nb} in this live slice.")
    elif pb > pa * 1.4 and tb > ta + 1:
        lines.append(f"{nb} shows stronger engagement but higher toxicity risk than {na} in this live slice.")

    if ga.snap and gb.snap and ga.snap.pos_ratio_pct > gb.snap.pos_ratio_pct + 8 and len(ga.snap.volatility_flags) <= len(
        gb.snap.volatility_flags
    ):
        lines.append(f"{na} demonstrates more stable headline sentiment than {nb}.")
    elif gb.snap and ga.snap and gb.snap.pos_ratio_pct > ga.snap.pos_ratio_pct + 8 and len(gb.snap.volatility_flags) <= len(
        ga.snap.volatility_flags
    ):
        lines.append(f"{nb} demonstrates more stable headline sentiment than {na}.")

    va, vb = _volatility_count(ga), _volatility_count(gb)
    if vb > va:
        lines.append(f"{na} currently shows lower volatility than {nb} in the sampled reviews.")
    elif va > vb:
        lines.append(f"{nb} currently shows lower volatility than {na} in the sampled reviews.")

    if not lines:
        lines.append(
            f"Side-by-side comparison of {na} and {nb} — review pulls are short; use charts and risk rows to prioritize follow-up."
        )
    return lines[:4]


def _competitive_risk_analysis(ga: ComparativeGameSlice, gb: ComparativeGameSlice) -> list[str]:
    ra, rb = _risk_profile(ga), _risk_profile(gb)
    na, nb = ga.display_name[:28], gb.display_name[:28]
    out: list[str] = []
    tag = "stronger toxicity signals in"
    if ra["toxicity_hits"] > rb["toxicity_hits"]:
        out.append(f"Toxicity lexicon: {tag} {na} ({ra['toxicity_hits']} vs {rb['toxicity_hits']} hits).")
    elif rb["toxicity_hits"] > ra["toxicity_hits"]:
        out.append(f"Toxicity lexicon: {tag} {nb} ({rb['toxicity_hits']} vs {ra['toxicity_hits']} hits).")
    else:
        out.append(f"Toxicity lexicon: tied at {ra['toxicity_hits']} hits each.")

    if ra["instability_hits"] > rb["instability_hits"]:
        out.append(f"Instability complaints: higher in {na}.")
    elif rb["instability_hits"] > ra["instability_hits"]:
        out.append(f"Instability complaints: higher in {nb}.")
    else:
        out.append("Instability complaints: similar density.")

    if ra["performance_hits"] > rb["performance_hits"]:
        out.append(f"Performance complaints: higher in {na}.")
    elif rb["performance_hits"] > ra["performance_hits"]:
        out.append(f"Performance complaints: higher in {nb}.")
    else:
        out.append("Performance complaints: similar density.")

    if ra["volatility_flags"] > rb["volatility_flags"]:
        out.append(f"Review volatility: more flags in {na} ({ra['volatility_flags']} vs {rb['volatility_flags']}).")
    elif rb["volatility_flags"] > ra["volatility_flags"]:
        out.append(f"Review volatility: more flags in {nb} ({rb['volatility_flags']} vs {ra['volatility_flags']}).")
    else:
        out.append(f"Review volatility: same flag count ({ra['volatility_flags']}).")

    if not (pd.isna(ra["pos_pct"]) or pd.isna(rb["pos_pct"])):
        if ra["pos_pct"] > rb["pos_pct"] + 5:
            out.append(f"Sentiment balance: {na} leans more positive ({ra['pos_pct']:.0f}% vs {rb['pos_pct']:.0f}% positive).")
        elif rb["pos_pct"] > ra["pos_pct"] + 5:
            out.append(f"Sentiment balance: {nb} leans more positive ({rb['pos_pct']:.0f}% vs {ra['pos_pct']:.0f}% positive).")
        else:
            out.append("Sentiment balance: broadly comparable positive ratios.")
    return out


def _strategic_competitive_insights(ga: ComparativeGameSlice, gb: ComparativeGameSlice) -> list[str]:
    lines: list[str] = []
    ha = ga.snap.health_score if ga.snap else None
    hb = gb.snap.health_score if gb.snap else None
    if ha is not None and hb is not None:
        if ha > hb + 5:
            lines.append(f"{ga.display_name[:36]} maintains a higher composite health score than {gb.display_name[:36]}.")
        elif hb > ha + 5:
            lines.append(f"{gb.display_name[:36]} maintains a higher composite health score than {ga.display_name[:36]}.")

    pa = _engagement_numeric(ga.snap, ga.catalog_profile)
    pb = _engagement_numeric(gb.snap, gb.catalog_profile)
    ra_n = ga.snap.risk_signal_count if ga.snap else 0
    rb_n = gb.snap.risk_signal_count if gb.snap else 0
    if pb > pa * 1.2 and rb_n > ra_n:
        lines.append(
            f"{gb.display_name[:36]} shows stronger catalog engagement but elevated structured risk signals "
            f"relative to {ga.display_name[:36]}."
        )

    ga_rpg = "rpg" in _norm_catalog_str(ga.catalog_profile, "genre", "").lower()
    gb_rpg = "rpg" in _norm_catalog_str(gb.catalog_profile, "genre", "").lower()
    if ga_rpg and not gb_rpg and ha is not None and hb is not None and ha >= hb:
        lines.append("Premium RPG positioning appears stronger in game 1 based on catalog genre and health.")
    elif gb_rpg and not ga_rpg and ha is not None and hb is not None and hb >= ha:
        lines.append("Premium RPG positioning appears stronger in game 2 based on catalog genre and health.")

    if not lines:
        lines.append("Tie or mixed signals — widen pulls or align on the same patch window before strategic calls.")
    return lines[:5]


def render_comparative_intelligence_panel(
    optional_games_catalog: pd.DataFrame,
    steam_catalog_real: pd.DataFrame | None,
) -> None:
    import streamlit as st

    inject_comparative_css()

    st.markdown('<div class="cmp-shell">', unsafe_allow_html=True)
    st.markdown('<div class="cmp-banner">Module 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="cmp-title">Comparative Intelligence</div>', unsafe_allow_html=True)
    st.caption(
        "Fetch up to **20** recent English reviews per title, join **steam_catalog_real.csv** by App ID, and contrast "
        "executive signals side by side."
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        raw_a = st.text_input("Game 1 — Steam App ID", placeholder="e.g. 730", key="cmp_intel_app_id_1")
    with c2:
        raw_b = st.text_input("Game 2 — Steam App ID", placeholder="e.g. 570", key="cmp_intel_app_id_2")

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    compare = st.button("Compare Games", type="primary", key="cmp_intel_compare_btn")

    if compare:
        st.session_state["cmp_intel_error"] = None
        st.session_state["cmp_intel_ga"] = None
        st.session_state["cmp_intel_gb"] = None

        id_a, err_a = parse_app_id_input(raw_a)
        id_b, err_b = parse_app_id_input(raw_b)
        if err_a or err_b:
            st.session_state["cmp_intel_error"] = err_a or err_b
        elif id_a is None or id_b is None:
            st.session_state["cmp_intel_error"] = "Enter two valid App IDs."
        elif id_a == id_b:
            st.session_state["cmp_intel_error"] = "Choose two different App IDs to compare."
        else:
            try:
                ga = load_comparative_game_slice(int(id_a), steam_catalog_real, optional_games_catalog)
                gb = load_comparative_game_slice(int(id_b), steam_catalog_real, optional_games_catalog)
                st.session_state["cmp_intel_ga"] = ga
                st.session_state["cmp_intel_gb"] = gb
            except Exception as exc:  # noqa: BLE001
                st.session_state["cmp_intel_error"] = f"Comparison pipeline error: {exc}"

    err = st.session_state.get("cmp_intel_error")
    if err:
        st.error(err)

    ga = st.session_state.get("cmp_intel_ga")
    gb = st.session_state.get("cmp_intel_gb")
    if not isinstance(ga, ComparativeGameSlice) or not isinstance(gb, ComparativeGameSlice):
        st.info("Enter two App IDs and press **Compare Games** to load live intelligence for both titles.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    na = ga.display_name[:24] + ("…" if len(ga.display_name) > 24 else "")
    nb = gb.display_name[:24] + ("…" if len(gb.display_name) > 24 else "")

    grid = f'<div class="cmp-grid-2">{_render_game_card(f"Game 1 — {na}", ga)}{_render_game_card(f"Game 2 — {nb}", gb)}</div>'
    st.markdown(grid, unsafe_allow_html=True)

    with st.expander("Executive summary", expanded=True):
        st.markdown('<div class="cmp-block"><h4>Comparative executive summary</h4>', unsafe_allow_html=True)
        for line in _comparative_executive_summary(ga, gb):
            st.markdown(f"<p>{line}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Competitive risk & strategic insights", expanded=False):
        st.markdown('<div class="cmp-block"><h4>Competitive risk analysis</h4><ul>', unsafe_allow_html=True)
        for line in _competitive_risk_analysis(ga, gb):
            st.markdown(f"<li>{line}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        st.markdown('<div class="cmp-block"><h4>Strategic competitive insights</h4><ul>', unsafe_allow_html=True)
        for line in _strategic_competitive_insights(ga, gb):
            st.markdown(f"<li>{line}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

    with st.expander("Comparison visualizations", expanded=True):
        p1, p2 = st.columns(2, gap="medium")
        with p1:
            st.plotly_chart(_fig_sentiment_pair(ga, gb, na, nb), use_container_width=True)
            st.plotly_chart(_fig_health_pair(ga, gb, na, nb), use_container_width=True)
        with p2:
            st.plotly_chart(_fig_engagement_pair(ga, gb, na, nb), use_container_width=True)
            st.plotly_chart(_fig_risk_pair(ga, gb, na, nb), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
