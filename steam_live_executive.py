"""
Stage 6A — Executive Decision Layer (Steam Live Review Intelligence only).

Rule-based product intelligence derived from live review samples, catalog context,
and shared lexicon helpers. Not used by Market Analytics or CSV dashboards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from review_sentiment import (
    NEGATIVE_THEME_KEYWORDS,
    detect_risk_signals,
    keyword_hit_counts,
)


# Volatility / instability scans (word-boundary style, aligned with keyword_hit_counts)
_VOLATILITY_INSTABILITY = ["unstable", "crash", "crashes", "disconnect", "lag", "server", "freeze", "stutter"]
_VOLATILITY_TOXICITY = ["toxic", "cheat", "cheaters", "grief", "harassment"]
_VOLATILITY_PERFORMANCE = ["performance", "optimization", "fps", "stutter", "frame"]


def inject_live_executive_css() -> None:
    """Scoped styles for the Live module executive strip (safe alongside global theme)."""
    import streamlit as st

    st.markdown(
        """
        <style>
            .live-exec-shell {
                margin: 1.75rem 0 1.5rem 0;
                padding: 0;
            }
            .live-exec-title {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #8f98a0;
                margin-bottom: 0.35rem;
            }
            .live-exec-headline {
                font-size: 1.35rem;
                font-weight: 700;
                color: #e5eef5;
                letter-spacing: -0.02em;
                margin-bottom: 1.1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #2a475e;
            }
            .live-exec-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 1rem;
                margin-bottom: 1.35rem;
            }
            @media (max-width: 1100px) {
                .live-exec-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (max-width: 520px) {
                .live-exec-grid { grid-template-columns: 1fr; }
            }
            .live-exec-card {
                background: linear-gradient(165deg, #1a2636 0%, #141c28 52%, #101820 100%);
                border: 1px solid #2f4a63;
                border-radius: 14px;
                padding: 1.15rem 1.2rem 1.2rem 1.2rem;
                box-shadow: 0 14px 42px rgba(0,0,0,0.42);
                min-height: 108px;
            }
            .live-exec-card .label {
                font-size: 0.68rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: #7a8a9a;
                margin-bottom: 0.45rem;
            }
            .live-exec-card .value {
                font-size: 1.55rem;
                font-weight: 700;
                color: #66c0f4;
                line-height: 1.15;
            }
            .live-exec-card .value.value-tight {
                font-size: 1.12rem;
                line-height: 1.35;
            }
            .live-exec-card .sub {
                margin-top: 0.55rem;
                font-size: 0.82rem;
                color: #9aaab8;
                line-height: 1.4;
            }
            .live-exec-card.risk-low .value { color: #3fb950; }
            .live-exec-card.risk-mid .value { color: #d29922; }
            .live-exec-card.risk-high .value { color: #f85149; }
            .live-exec-block {
                background: rgba(20, 30, 44, 0.55);
                border: 1px solid #2a475e;
                border-radius: 12px;
                padding: 1rem 1.15rem 1.05rem 1.15rem;
                margin-bottom: 0.95rem;
            }
            .live-exec-block h4 {
                font-size: 0.95rem;
                font-weight: 600;
                color: #c5d4e0;
                margin: 0 0 0.55rem 0;
            }
            .live-exec-block p, .live-exec-block li {
                color: #b4c4d4;
                font-size: 0.9rem;
                line-height: 1.55;
            }
            .live-exec-vol-tag {
                display: inline-block;
                margin: 0.25rem 0.35rem 0 0;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                font-size: 0.72rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                background: rgba(248, 81, 73, 0.12);
                border: 1px solid rgba(248, 81, 73, 0.35);
                color: #f0a4a0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _count_kw_group(series: pd.Series, words: list[str]) -> int:
    if series.empty:
        return 0
    return sum(keyword_hit_counts(series, [w]).get(w, 0) for w in words)


def _norm_peak(profile: pd.Series | None) -> float:
    if profile is None:
        return 0.0
    try:
        return float(profile.get("peak_players") or 0)
    except (TypeError, ValueError):
        return 0.0


def _genre_blob(profile: pd.Series | None) -> str:
    if profile is None:
        return ""
    g = profile.get("genre")
    if pd.isna(g):
        return ""
    return str(g).lower()


def _price_cat(profile: pd.Series | None) -> str:
    if profile is None:
        return ""
    return str(profile.get("price_category") or "").lower()


def _reviews_blob_lower(series: pd.Series) -> str:
    if series.empty:
        return ""
    return " ".join(series.astype(str).str.lower().tolist())


@dataclass(frozen=True)
class LiveExecutiveSnapshot:
    health_score: int
    risk_level: str
    community_sentiment_status: str
    engagement_strength: str
    market_position: str
    volatility_flags: list[str]
    executive_summary: list[str]
    strategic_recommendations: list[str]
    neg_keyword_total: int
    pos_ratio_pct: float
    neg_ratio_pct: float
    risk_signal_count: int


def compute_live_executive_snapshot(
    df_live: pd.DataFrame,
    enriched: pd.DataFrame,
    catalog_profile: pd.Series | None,
) -> LiveExecutiveSnapshot:
    """Derive executive metrics from the live sample + enrichment + optional catalog row."""
    n = max(len(df_live), 1)
    pos_n = int((df_live["sentiment"].astype(str).str.lower() == "positive").sum())
    neg_n = int((df_live["sentiment"].astype(str).str.lower() == "negative").sum())
    pos_ratio = 100.0 * pos_n / n
    neg_ratio = 100.0 * neg_n / n

    texts = enriched["review_text"] if not enriched.empty and "review_text" in enriched.columns else df_live["review_text"]
    neg_map = keyword_hit_counts(texts, NEGATIVE_THEME_KEYWORDS)
    neg_keyword_total = int(sum(neg_map.values()))
    neg_kw_per_review = neg_keyword_total / n

    risks = detect_risk_signals(enriched) if not enriched.empty else []
    risk_count = len(risks)

    peak = _norm_peak(catalog_profile)
    peak_component = 40.0
    if peak > 0:
        peak_component = min(100.0, 100.0 * math.log10(peak + 10.0) / math.log10(500_000.0))

    activity_component = min(100.0, (len(df_live) / 20.0) * 100.0)
    sentiment_component = pos_ratio
    neg_kw_component = max(0.0, 100.0 - min(100.0, neg_kw_per_review * 42.0))
    risk_component = max(0.0, 100.0 - min(100.0, risk_count * 28.0))

    health = int(
        round(
            0.34 * sentiment_component
            + 0.22 * neg_kw_component
            + 0.24 * risk_component
            + 0.12 * peak_component
            + 0.08 * activity_component
        )
    )
    health = max(0, min(100, health))

    if risk_count >= 2 or neg_ratio >= 55 or health < 38:
        risk_level = "High"
    elif risk_count == 1 or neg_ratio >= 38 or health < 62:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if pos_ratio >= 76 and neg_kw_per_review < 0.55:
        comm = "Strong"
    elif pos_ratio >= 62:
        comm = "Stable"
    elif pos_ratio >= 48:
        comm = "Mixed"
    elif pos_ratio >= 32:
        comm = "Fragile"
    else:
        comm = "Under Stress"

    if catalog_profile is None or peak <= 0:
        engagement = "Sample-limited (no CCU in context)"
    elif peak >= 120_000:
        engagement = "Very High"
    elif peak >= 25_000:
        engagement = "High"
    elif peak >= 8_000:
        engagement = "Moderate"
    elif peak >= 1_200:
        engagement = "Building"
    else:
        engagement = "Limited"

    inst_hits = _count_kw_group(texts, _VOLATILITY_INSTABILITY)
    tox_hits = _count_kw_group(texts, _VOLATILITY_TOXICITY)
    perf_hits = _count_kw_group(texts, _VOLATILITY_PERFORMANCE)

    volatility: list[str] = []
    if neg_ratio >= 48:
        volatility.append("Elevated negative ratio in the live sample")
    if neg_keyword_total >= max(4, int(0.35 * n)):
        volatility.append("Sudden complaint concentration (lexicon density)")
    if inst_hits >= 2:
        volatility.append("Instability-related language cluster")
    if tox_hits >= 2:
        volatility.append("Toxicity concentration in verbatim text")
    if perf_hits >= max(2, int(0.2 * n)):
        volatility.append("Performance complaint spike")
    texts_lower = _reviews_blob_lower(texts)
    if "matchmaking" in texts_lower and neg_ratio >= 30:
        volatility.append("Matchmaking frustration in verbatim text")

    market_position = _infer_market_position(
        peak=peak,
        neg_ratio=neg_ratio,
        pos_ratio=pos_ratio,
        volatility_n=len(volatility),
        genre_blob=_genre_blob(catalog_profile),
        price_cat=_price_cat(catalog_profile),
        tox_hits=tox_hits,
        texts_lower=texts_lower,
    )

    summary = _build_executive_summary(
        pos_ratio=pos_ratio,
        neg_ratio=neg_ratio,
        risk_count=risk_count,
        engagement=engagement,
        peak=peak,
        perf_hits=perf_hits,
        tox_hits=tox_hits,
        genre_blob=_genre_blob(catalog_profile),
        volatility=volatility,
        texts_lower=texts_lower,
    )
    difficult_hits = int(neg_map.get("difficult", 0))

    recs = _build_strategic_recommendations(
        risk_count=risk_count,
        volatility=volatility,
        inst_hits=inst_hits,
        tox_hits=tox_hits,
        perf_hits=perf_hits,
        neg_ratio=neg_ratio,
        pos_ratio=pos_ratio,
        difficult_hits=difficult_hits,
        texts_lower=texts_lower,
    )

    return LiveExecutiveSnapshot(
        health_score=health,
        risk_level=risk_level,
        community_sentiment_status=comm,
        engagement_strength=engagement,
        market_position=market_position,
        volatility_flags=volatility,
        executive_summary=summary,
        strategic_recommendations=recs,
        neg_keyword_total=neg_keyword_total,
        pos_ratio_pct=pos_ratio,
        neg_ratio_pct=neg_ratio,
        risk_signal_count=risk_count,
    )


def _infer_market_position(
    *,
    peak: float,
    neg_ratio: float,
    pos_ratio: float,
    volatility_n: int,
    genre_blob: str,
    price_cat: str,
    tox_hits: int,
    texts_lower: str,
) -> str:
    is_rpg = "rpg" in genre_blob
    is_indie = "indie" in genre_blob
    is_action = "action" in genre_blob
    premiumish = any(x in price_cat for x in ("high", "medium"))
    multiplayerish = any(
        k in genre_blob for k in ("multiplayer", "mmo", "co-op", "coop", "pvp", "online")
    ) or ("multiplayer" in texts_lower and peak >= 5_000)

    if peak >= 80_000 and neg_ratio < 38:
        return "High Engagement Competitive Title"
    if is_rpg and premiumish and pos_ratio >= 58 and volatility_n <= 1:
        return "Stable Premium RPG"
    if multiplayerish and (tox_hits >= 2 or neg_ratio >= 42):
        return "Community-Risk Multiplayer Game"
    if peak < 8_000 and pos_ratio >= 62 and (is_indie or peak < 3_000):
        return "Strong Indie Opportunity"
    if volatility_n >= 2 and (is_action or peak >= 40_000):
        return "Volatile Live-Service Title"
    if is_rpg and pos_ratio >= 55:
        return "Stable Premium RPG"
    if peak >= 40_000:
        return "High Engagement Competitive Title"
    if is_indie:
        return "Strong Indie Opportunity"
    return "Portfolio Title (mixed signals)"


def _build_executive_summary(
    *,
    pos_ratio: float,
    neg_ratio: float,
    risk_count: int,
    engagement: str,
    peak: float,
    perf_hits: int,
    tox_hits: int,
    genre_blob: str,
    volatility: list[str],
    texts_lower: str,
) -> list[str]:
    lines: list[str] = []

    if pos_ratio >= 65 and perf_hits >= 1:
        lines.append("Community sentiment remains strong despite moderate performance complaints.")
    if engagement in ("Very High", "High") and tox_hits >= 2:
        lines.append("Player engagement is extremely high but toxicity signals require monitoring.")
    if "rpg" in genre_blob and pos_ratio >= 55 and risk_count <= 1 and neg_ratio < 42:
        lines.append("Premium RPG titles continue to maintain strong sentiment stability.")
    if "matchmaking" in texts_lower and neg_ratio >= 35:
        lines.append("Live review sentiment indicates elevated frustration around matchmaking systems.")
    if risk_count >= 2 and pos_ratio >= 58:
        lines.append(
            "Headline sentiment stays constructive while structured risk cues warrant a coordinated response plan."
        )
    if not lines:
        if pos_ratio >= 60:
            lines.append(
                "The recent English-language sample skews constructive — community tone is supportive in this pull."
            )
        elif pos_ratio >= 45:
            lines.append("Sentiment in this sample is mixed; validate themes across additional pulls before major bets.")
        else:
            lines.append(
                "This live slice tilts critical — treat it as an early-warning lens rather than a full sentiment census."
            )
    if peak > 0 and neg_ratio < 40 and len(lines) < 4:
        lines.append(
            f"Catalog-scale engagement context (peak CCU {int(peak):,}) aligns with a title that still commands attention."
        )
    if volatility and len(lines) < 4:
        lines.append("Volatility indicators suggest monitoring the next few review windows for confirmation.")
    return lines[:4]


def _build_strategic_recommendations(
    *,
    risk_count: int,
    volatility: list[str],
    inst_hits: int,
    tox_hits: int,
    perf_hits: int,
    neg_ratio: float,
    pos_ratio: float,
    difficult_hits: int,
    texts_lower: str,
) -> list[str]:
    recs: list[str] = []
    if inst_hits >= 1 or any("Instability" in v for v in volatility):
        recs.append("Monitor server stability")
    if tox_hits >= 2 or any("Toxicity" in v for v in volatility):
        recs.append("Investigate toxicity spikes")
    if perf_hits >= 2 or any("Performance" in v for v in volatility):
        recs.append("Prioritize performance optimization")
    if len(volatility) >= 2 or neg_ratio >= 45 or risk_count >= 2:
        recs.append("Monitor review volatility")
    if pos_ratio >= 62 and risk_count == 0 and neg_ratio < 40:
        recs.append("Maintain pricing strategy")
    if difficult_hits >= 2 or (difficult_hits >= 1 and pos_ratio < 55):
        recs.append("Improve onboarding")
    if "matchmaking" in texts_lower:
        recs.append("Review matchmaking UX and queue health")
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    if not out:
        out.append("Continue periodic live sampling")
    return out[:8]


def render_executive_decision_layer(
    df_live: pd.DataFrame,
    enriched: pd.DataFrame,
    catalog_profile: pd.Series | None,
) -> None:
    """Streamlit: Executive Decision Layer (Live module only)."""
    import streamlit as st

    inject_live_executive_css()

    snap = compute_live_executive_snapshot(df_live, enriched, catalog_profile)

    risk_cls = "risk-low"
    if snap.risk_level == "Medium":
        risk_cls = "risk-mid"
    elif snap.risk_level == "High":
        risk_cls = "risk-high"

    st.markdown('<div class="live-exec-shell">', unsafe_allow_html=True)
    st.markdown('<div class="live-exec-title">Stage 6A · Product intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="live-exec-headline">Executive Decision Layer</div>', unsafe_allow_html=True)

    esc_eng = _html_escape(snap.engagement_strength)
    esc_comm = _html_escape(snap.community_sentiment_status)
    cards_html = f"""
    <div class="live-exec-grid">
      <div class="live-exec-card">
        <div class="label">Health score</div>
        <div class="value">{snap.health_score}</div>
        <div class="sub">0–100 composite from sentiment, lexicon risk, catalog CCU, and sample depth</div>
      </div>
      <div class="live-exec-card {risk_cls}">
        <div class="label">Risk level</div>
        <div class="value">{_html_escape(snap.risk_level)}</div>
        <div class="sub">{snap.risk_signal_count} structured risk banner(s) in this sample</div>
      </div>
      <div class="live-exec-card">
        <div class="label">Engagement strength</div>
        <div class="value value-tight">{esc_eng}</div>
        <div class="sub">Peak players from catalog when matched · else sample-only</div>
      </div>
      <div class="live-exec-card">
        <div class="label">Community status</div>
        <div class="value">{esc_comm}</div>
        <div class="sub">Positive {snap.pos_ratio_pct:.0f}% · Negative {snap.neg_ratio_pct:.0f}%</div>
      </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown(
        f'<div class="live-exec-block"><h4>Market position</h4><p><b>{_html_escape(snap.market_position)}</b> — '
        "positioning label from genre, price tier, CCU, volatility, and this pull's tone.</p></div>",
        unsafe_allow_html=True,
    )

    summary_body = "".join(f"<p>{_html_escape(para)}</p>" for para in snap.executive_summary)
    st.markdown(
        f'<div class="live-exec-block"><h4>Executive summary</h4>{summary_body}</div>',
        unsafe_allow_html=True,
    )

    recs_body = "".join(f"<li>{_html_escape(item)}</li>" for item in snap.strategic_recommendations)
    st.markdown(
        f'<div class="live-exec-block"><h4>Strategic recommendations</h4><ul>{recs_body}</ul></div>',
        unsafe_allow_html=True,
    )

    if snap.volatility_flags:
        tags = "".join(
            f'<span class="live-exec-vol-tag">{_html_escape(v)}</span>' for v in snap.volatility_flags
        )
        vol_body = (
            f"{tags}<p style='margin-top:0.75rem'>These flags compare this review window to calmer samples — "
            "escalate if multiple persist across pulls.</p>"
        )
    else:
        vol_body = "<p>No elevated volatility pattern detected in this pull — continue periodic sampling.</p>"
    st.markdown(
        f'<div class="live-exec-block"><h4>Review volatility signal</h4>{vol_body}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
