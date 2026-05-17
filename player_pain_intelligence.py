"""
Player Pain Intelligence Layer — rule-based complaint taxonomy from Steam review text.

No external APIs or ML models. Designed for 100–5,000 review samples and CSV-backed caches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Analytics logic is self-contained; chart styling uses _finalize_pain_fig below.

# ---------------------------------------------------------------------------
# Lexicons — word-boundary matching (case-insensitive)
# ---------------------------------------------------------------------------

PAIN_CATEGORIES: dict[str, list[str]] = {
    "Performance": [
        "performance",
        "slow",
        "runs slow",
        "lag",
        "laggy",
        "stutter",
        "stuttering",
        "fps",
        "frame rate",
        "low fps",
        "slideshow",
    ],
    "Bugs": [
        "bug",
        "bugs",
        "buggy",
        "glitch",
        "glitches",
        "broken",
        "game breaking",
        "soft lock",
        "softlock",
    ],
    "Crashes": [
        "crash",
        "crashes",
        "crashing",
        "crash to desktop",
        "ctd",
        "freeze",
        "freezes",
        "frozen",
        "not responding",
    ],
    "Optimization": [
        "optimization",
        "optimize",
        "unoptimized",
        "poorly optimized",
        "bad optimization",
        "not optimized",
    ],
    "Difficulty frustration": [
        "too hard",
        "unfair",
        "frustratingly hard",
        "difficulty spike",
        "punishing",
        "bullshit difficulty",
        "unfair difficulty",
        "impossible",
    ],
    "Repetitive gameplay": [
        "repetitive",
        "repetition",
        "same thing over",
        "boring loop",
        "copy paste",
        "recycled content",
        "doing the same",
    ],
    "Balancing": [
        "unbalanced",
        "overpowered",
        "underpowered",
        "nerf",
        "nerfed",
        "meta broken",
        "pay to win",
        "imbalance",
        "imbalanced",
    ],
    "Matchmaking": [
        "matchmaking",
        "queue time",
        "long queue",
        "waiting queue",
        "sbmm",
        "skill based matchmaking",
        "unfair matchmaking",
    ],
    "Server instability": [
        "server",
        "servers",
        "disconnect",
        "disconnected",
        "downtime",
        "high ping",
        "packet loss",
        "rubber band",
        "rubberbanding",
        "unstable server",
    ],
    "Cheaters": [
        "cheat",
        "cheater",
        "cheaters",
        "hacker",
        "hackers",
        "hacking",
        "aimbot",
        "wallhack",
        "smurf",
    ],
    "Monetization": [
        "microtransaction",
        "microtransactions",
        "paywall",
        "greedy",
        "expensive dlc",
        "battle pass",
        "monetization",
        "cash grab",
        "pay to win",
        "predatory",
    ],
    "UI/UX frustration": [
        "bad ui",
        "clunky ui",
        "confusing menu",
        "terrible menu",
        "hud",
        "user interface",
        "ux",
        "poor interface",
        "clunky menus",
    ],
    "Grinding": [
        "grind",
        "grinding",
        "tedious grind",
        "farming",
        "farm for hours",
        "hours of grind",
        "grind fest",
    ],
    "Story dissatisfaction": [
        "bad story",
        "weak story",
        "plot hole",
        "terrible writing",
        "disappointing story",
        "boring story",
        "narrative",
        "poorly written",
    ],
    "Empty world": [
        "empty world",
        "barren",
        "lifeless world",
        "empty map",
        "no content",
        "hollow world",
        "desolate",
    ],
    "AI behavior": [
        "dumb ai",
        "bad ai",
        "brain dead",
        "stupid ai",
        "npc stupid",
        "enemy ai",
        "broken ai",
    ],
    "Controls/Input issues": [
        "controls",
        "bad controls",
        "input lag",
        "keybind",
        "controller",
        "mouse sensitivity",
        "control scheme",
        "unresponsive controls",
    ],
}

EMOTION_LEXICONS: dict[str, list[str]] = {
    "frustration": ["frustrated", "frustrating", "frustration", "annoying", "annoyed", "irritating"],
    "anger": ["angry", "rage", "furious", "pissed", "outraged", "hate this"],
    "burnout": ["burnout", "burned out", "burnt out", "tired of", "exhausted", "drained"],
    "excitement": ["excited", "exciting", "thrilled", "hyped", "can't stop playing"],
    "addiction": ["addictive", "addicted", "addiction", "one more run", "just one more"],
    "immersion": ["immersive", "immersion", "immersed", "atmosphere", "atmospheric"],
    "disappointment": ["disappointed", "disappointing", "let down", "letdown", "expected better"],
}

VETERAN_PLAYTIME_HOURS = 20.0
RECENT_WINDOW_FRAC = 0.25
ESCALATION_RATIO_THRESHOLD = 1.45
ESCALATION_MIN_RECENT = 3


@dataclass
class PainAnalysisResult:
    pain_index: int
    category_table: pd.DataFrame
    veteran_flags: list[dict[str, Any]]
    escalations: list[str]
    emotion_table: pd.DataFrame
    executive_summary: list[str]
    recommendations: list[str]
    tagged_reviews: pd.DataFrame = field(repr=False)
    overlap_stats: dict[str, int] = field(default_factory=dict)


def _compile_patterns(terms: list[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for term in terms:
        t = term.strip().lower()
        if not t:
            continue
        if " " in t:
            pat = re.escape(t)
        else:
            pat = rf"(?<![a-z]){re.escape(t)}(?![a-z])"
        patterns.append(re.compile(pat, re.IGNORECASE))
    return patterns


_CATEGORY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    cat: _compile_patterns(terms) for cat, terms in PAIN_CATEGORIES.items()
}
_EMOTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    emo: _compile_patterns(terms) for emo, terms in EMOTION_LEXICONS.items()
}


def _text_matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in patterns)


def _categories_for_text(text: str) -> list[str]:
    t = str(text or "").lower()
    hits = [cat for cat, pats in _CATEGORY_PATTERNS.items() if _text_matches_any(t, pats)]
    return hits


def _emotions_for_text(text: str) -> list[str]:
    t = str(text or "").lower()
    return [emo for emo, pats in _EMOTION_PATTERNS.items() if _text_matches_any(t, pats)]


def _ensure_bool_series(value: Any, index: pd.Index) -> pd.Series:
    """Coerce ndarray/list/Series into a boolean Series aligned to `index`."""
    if isinstance(value, pd.Series):
        return value.astype(bool).reindex(index, fill_value=False)
    if isinstance(value, (np.ndarray, list, tuple)):
        arr = np.asarray(value, dtype=bool)
        if len(arr) == len(index):
            return pd.Series(arr, index=index, dtype=bool)
        return pd.Series(False, index=index, dtype=bool)
    return pd.Series(bool(value), index=index, dtype=bool)


def _safe_category_mask(tagged: pd.DataFrame, category: str) -> pd.Series:
    """Boolean mask for reviews mentioning a pain category (tolerant of bad cell types)."""
    if tagged.empty or "pain_categories" not in tagged.columns:
        return pd.Series(dtype=bool)

    def _hits(cats: Any) -> bool:
        if isinstance(cats, list):
            return category in cats
        if pd.isna(cats):
            return False
        return category in str(cats)

    raw = tagged["pain_categories"].map(_hits)
    return _ensure_bool_series(raw, tagged.index)


def _review_sentiment_series(df: pd.DataFrame) -> pd.Series:
    if "sentiment" in df.columns:
        return df["sentiment"].astype(str).str.strip().str.lower()
    if "voted_up" in df.columns:
        vu = df["voted_up"]
        if vu.dtype == object:
            pos = vu.astype(str).str.lower().isin(("true", "1", "yes"))
        else:
            pos = vu.fillna(False).astype(bool)
        return pos.map(lambda x: "positive" if x else "negative")
    return pd.Series(["neutral"] * len(df), index=df.index)


def _playtime_hours_series(df: pd.DataFrame) -> pd.Series:
    if "playtime_forever" not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df["playtime_forever"], errors="coerce") / 60.0


def _severity_score(
    mention_pct: float,
    neg_share: float,
    avg_playtime_h: float,
    mention_count: int,
    total: int,
) -> float:
    """0–100 composite severity for a pain category."""
    prevalence = min(100.0, mention_pct * 2.2)
    negativity = min(100.0, neg_share * 100.0)
    depth = min(30.0, (avg_playtime_h / 80.0) * 30.0) if avg_playtime_h > 0 else 0.0
    density = min(25.0, (mention_count / max(total, 1)) * 250.0)
    return round(min(100.0, 0.38 * prevalence + 0.34 * negativity + 0.16 * depth + 0.12 * density), 1)


def _risk_level(severity: float, mention_pct: float) -> str:
    if severity >= 68 or mention_pct >= 22:
        return "High"
    if severity >= 42 or mention_pct >= 10:
        return "Medium"
    return "Low"


def _trend_signal(recent_rate: float, older_rate: float, recent_n: int) -> str:
    if recent_n < ESCALATION_MIN_RECENT and recent_rate == 0:
        return "Stable"
    if older_rate <= 0 and recent_rate > 0:
        return "Rising"
    if recent_rate <= 0 and older_rate > 0:
        return "Falling"
    ratio = recent_rate / older_rate if older_rate > 0 else (2.0 if recent_rate > 0 else 1.0)
    if ratio >= ESCALATION_RATIO_THRESHOLD:
        return "Rising"
    if ratio <= 0.72:
        return "Falling"
    return "Stable"


def tag_reviews_with_pain(df: pd.DataFrame) -> pd.DataFrame:
    """Add pain_categories, pain_count, emotions, playtime_hours, sentiment_norm."""
    if df.empty:
        return df.copy()
    out = df.copy()
    texts = out.get("review_text", pd.Series(dtype=str)).astype(str)
    out["pain_categories"] = texts.map(_categories_for_text)
    out["pain_count"] = out["pain_categories"].map(len)
    out["emotions"] = texts.map(_emotions_for_text)
    out["sentiment_norm"] = _review_sentiment_series(out)
    out["playtime_hours"] = _playtime_hours_series(out)
    out["has_pain"] = out["pain_count"] > 0
    return out


def _split_recent_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series: True = review in the most recent time window."""
    n = len(df)
    if n == 0:
        return pd.Series(dtype=bool)
    if n < 8 or "timestamp_created" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    ts = pd.to_numeric(df["timestamp_created"], errors="coerce")
    if ts.notna().sum() < 6:
        return pd.Series(True, index=df.index, dtype=bool)
    ordered = df.loc[ts.notna()].sort_values("timestamp_created")
    cut = max(1, int(len(ordered) * RECENT_WINDOW_FRAC))
    recent_idx = set(ordered.tail(cut).index)
    return _ensure_bool_series(df.index.isin(recent_idx), df.index)


def compute_category_metrics(tagged: pd.DataFrame) -> pd.DataFrame:
    empty_cols = [
        "Pain Category",
        "Mentions",
        "Mention %",
        "Severity",
        "Risk Level",
        "Avg Playtime (h)",
        "Trend Signal",
    ]
    if tagged.empty:
        return pd.DataFrame(columns=empty_cols)

    total = max(len(tagged), 1)
    recent_mask = _ensure_bool_series(_split_recent_mask(tagged), tagged.index)
    recent_total = max(int(recent_mask.sum()), 1)
    older_total = max(int((~recent_mask).sum()), 1)
    rows: list[dict[str, Any]] = []

    for category in PAIN_CATEGORIES:
        try:
            mask = _safe_category_mask(tagged, category)
            mentions = int(mask.sum())
            if mentions == 0:
                continue

            subset = tagged.loc[mask]
            mention_pct = 100.0 * mentions / total
            play_col = subset["playtime_hours"] if "playtime_hours" in subset.columns else pd.Series(dtype=float)
            avg_play = float(play_col.mean()) if play_col.notna().any() else 0.0
            sent = subset["sentiment_norm"] if "sentiment_norm" in subset.columns else pd.Series(dtype=str)
            pos_n = int((sent == "positive").sum())
            neg_n = int((sent == "negative").sum())
            neg_share = neg_n / mentions if mentions else 0.0

            severity = _severity_score(mention_pct, neg_share, avg_play, mentions, total)
            risk = _risk_level(severity, mention_pct)

            recent_on_subset = _ensure_bool_series(recent_mask, subset.index)
            recent_sub = subset.loc[recent_on_subset]
            older_sub = subset.loc[~recent_on_subset]
            recent_rate = len(recent_sub) / recent_total
            older_rate = len(older_sub) / older_total
            trend = _trend_signal(recent_rate, older_rate, len(recent_sub))
        except (TypeError, ValueError, KeyError):
            continue

        rows.append(
            {
                "Pain Category": category,
                "Mentions": mentions,
                "Mention %": round(mention_pct, 1),
                "Severity": severity,
                "Risk Level": risk,
                "Avg Playtime (h)": round(avg_play, 1),
                "Trend Signal": trend,
                "Positive mentions": pos_n,
                "Negative mentions": neg_n,
                "Neg % in category": round(100.0 * neg_share, 1),
            }
        )

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    table = pd.DataFrame(rows).sort_values(["Severity", "Mentions"], ascending=[False, False])
    return table


def compute_player_pain_index(category_table: pd.DataFrame, tagged: pd.DataFrame) -> int:
    if category_table.empty:
        pain_share = float(tagged["has_pain"].mean()) if len(tagged) else 0.0
        return int(round(min(100.0, pain_share * 40.0)))

    weighted = 0.0
    weight_sum = 0.0
    for _, row in category_table.iterrows():
        w = float(row["Mentions"])
        weighted += float(row["Severity"]) * w
        weight_sum += w
    base = weighted / weight_sum if weight_sum > 0 else 0.0
    breadth = min(25.0, len(category_table) * 2.5)
    neg_boost = 0.0
    if "Neg % in category" in category_table.columns:
        neg_boost = min(15.0, float(category_table["Neg % in category"].mean()) * 0.12)
    return int(round(min(100.0, base * 0.82 + breadth + neg_boost)))


def detect_veteran_frustration(tagged: pd.DataFrame) -> list[dict[str, Any]]:
    if tagged.empty or "playtime_hours" not in tagged.columns or tagged["playtime_hours"].notna().sum() == 0:
        return []
    play = tagged["playtime_hours"]
    threshold = max(VETERAN_PLAYTIME_HOURS, float(play.quantile(0.75)) if play.notna().sum() >= 4 else VETERAN_PLAYTIME_HOURS)
    has_pain = _ensure_bool_series(tagged.get("has_pain", False), tagged.index)
    is_negative = tagged.get("sentiment_norm", pd.Series(dtype=str)).astype(str) == "negative"
    mask = _ensure_bool_series(has_pain & is_negative & (play >= threshold), tagged.index)
    flags: list[dict[str, Any]] = []
    for _, row in tagged.loc[mask].head(12).iterrows():
        cats = row.get("pain_categories") or []
        flags.append(
            {
                "label": "Veteran Player Frustration",
                "playtime_h": round(float(row["playtime_hours"]), 1),
                "categories": ", ".join(cats[:4]) if cats else "General",
                "snippet": str(row.get("review_text", ""))[:160],
            }
        )
    return flags


def detect_escalations(tagged: pd.DataFrame, category_table: pd.DataFrame) -> list[str]:
    if category_table.empty:
        return []
    alerts: list[str] = []
    for _, row in category_table.iterrows():
        if row.get("Trend Signal") == "Rising" and float(row.get("Severity", 0)) >= 40:
            alerts.append(
                f"**{row['Pain Category']}** — rising mention rate in the most recent review window "
                f"({int(row['Mentions'])} total mentions, severity {row['Severity']:.0f})."
            )
    return alerts[:6]


def compute_emotion_signals(tagged: pd.DataFrame) -> pd.DataFrame:
    if tagged.empty or "emotions" not in tagged.columns:
        return pd.DataFrame(columns=["Emotion", "Mentions", "Share %", "Negative %"])
    total = max(len(tagged), 1)
    rows: list[dict[str, Any]] = []
    for emotion in EMOTION_LEXICONS:

        def _emo_hit(ems: Any, e: str = emotion) -> bool:
            if isinstance(ems, list):
                return e in ems
            if pd.isna(ems):
                return False
            return e in str(ems)

        mask = _ensure_bool_series(tagged["emotions"].map(_emo_hit), tagged.index)
        n = int(mask.sum())
        if n == 0:
            continue
        sub = tagged.loc[mask]
        neg_n = int((sub["sentiment_norm"] == "negative").sum())
        rows.append(
            {
                "Emotion": emotion.title(),
                "Mentions": n,
                "Share %": round(100.0 * n / total, 1),
                "Negative %": round(100.0 * neg_n / n, 1) if n else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Emotion", "Mentions", "Share %", "Negative %"])
    return pd.DataFrame(rows).sort_values("Mentions", ascending=False)


def _pain_overlap_stats(tagged: pd.DataFrame) -> dict[str, int]:
    pain = tagged.loc[tagged["has_pain"]]
    if pain.empty:
        return {"pain_positive": 0, "pain_negative": 0, "pain_total": 0}
    pos = int((pain["sentiment_norm"] == "positive").sum())
    neg = int((pain["sentiment_norm"] == "negative").sum())
    return {"pain_positive": pos, "pain_negative": neg, "pain_total": len(pain)}


def build_executive_summaries(
    result: PainAnalysisResult,
    *,
    game_name: str = "this title",
) -> list[str]:
    lines: list[str] = []
    tbl = result.category_table
    if tbl.empty:
        lines.append(
            f"No dominant pain taxonomy hits in the current sample for **{game_name}** — "
            "lexicon signals are quiet; widen the review limit or refresh from Steam."
        )
        return lines

    top = tbl.iloc[0]
    lines.append(
        f"**{top['Pain Category']}** leads player pain mentions ({int(top['Mentions'])} reviews, "
        f"{top['Mention %']:.1f}% of sample) with severity **{top['Severity']:.0f}/100** and **{top['Risk Level']}** risk."
    )

    high_play = tbl.loc[tbl["Avg Playtime (h)"] >= VETERAN_PLAYTIME_HOURS]
    if not high_play.empty:
        cat = high_play.iloc[0]["Pain Category"]
        lines.append(
            f"**{cat}** complaints are concentrated among high-playtime users "
            f"(~{high_play.iloc[0]['Avg Playtime (h)']:.0f}h average), increasing long-term retention risk."
        )

    rising = tbl[tbl["Trend Signal"] == "Rising"]
    if not rising.empty:
        names = ", ".join(rising["Pain Category"].head(3).tolist())
        lines.append(f"Review escalation detected for: **{names}** — recent windows show accelerating mention rates.")

    if result.veteran_flags:
        lines.append(
            f"**Veteran Player Frustration** flagged in **{len(result.veteran_flags)}** high-playtime negative reviews — "
            "these voices often predict churn among your most invested audience."
        )

    if result.pain_index >= 65:
        lines.append(
            f"Global **Player Pain Index** is **{result.pain_index}/100** — prioritize stabilization before feature expansion."
        )
    elif result.pain_index >= 40:
        lines.append(
            f"Player Pain Index **{result.pain_index}/100** sits in a caution band — monitor the next pull for confirmation."
        )

    return lines[:5]


def build_strategic_recommendations(result: PainAnalysisResult) -> list[str]:
    recs: list[str] = []
    tbl = result.category_table
    if tbl.empty:
        return ["Continue periodic review sampling to establish a pain baseline."]

    top_cats = set(tbl.head(5)["Pain Category"].tolist())
    cat_map = {
        "Performance": "Prioritize optimization patches and frame-time profiling",
        "Bugs": "Triage crash/bug clusters with reproducible steps from reviews",
        "Crashes": "Investigate crash clusters and publish hotfix telemetry",
        "Optimization": "Prioritize optimization patches for target hardware tiers",
        "Difficulty frustration": "Reduce onboarding friction and tune difficulty curves",
        "Repetitive gameplay": "Rebalance progression pacing and add mid-loop variety",
        "Balancing": "Rebalance progression pacing and competitive tuning",
        "Matchmaking": "Review matchmaking UX and queue health dashboards",
        "Server instability": "Monitor server stability and regional latency",
        "Cheaters": "Investigate anti-cheat reporting and moderation throughput",
        "Monetization": "Audit monetization messaging and value perception",
        "UI/UX frustration": "Run targeted UI/UX usability passes on high-friction flows",
        "Grinding": "Rebalance progression pacing and reward cadence",
        "Story dissatisfaction": "Validate narrative beats with player-expectation research",
        "Empty world": "Assess content density and world activity in live builds",
        "AI behavior": "Review AI behavior trees and combat readability",
        "Controls/Input issues": "Audit controls/input defaults and rebinding UX",
    }
    for cat in top_cats:
        if cat in cat_map and cat_map[cat] not in recs:
            recs.append(cat_map[cat])

    if result.escalations:
        recs.append("Monitor review volatility for escalating pain categories")
    if result.veteran_flags:
        recs.append("Address veteran-player pain before acquisition spend increases")

    if not recs:
        recs.append("Maintain current quality bar; no dominant pain theme exceeded thresholds")
    return recs[:8]


def analyze_player_pain(df: pd.DataFrame) -> PainAnalysisResult:
    """Full pain intelligence pass on a review dataframe."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        empty = pd.DataFrame()
        return PainAnalysisResult(
            pain_index=0,
            category_table=pd.DataFrame(
                columns=[
                    "Pain Category",
                    "Mentions",
                    "Mention %",
                    "Severity",
                    "Risk Level",
                    "Avg Playtime (h)",
                    "Trend Signal",
                ]
            ),
            veteran_flags=[],
            escalations=[],
            emotion_table=pd.DataFrame(columns=["Emotion", "Mentions", "Share %", "Negative %"]),
            executive_summary=["No reviews available for pain analysis."],
            recommendations=["Load or fetch reviews before running Player Pain Intelligence."],
            tagged_reviews=empty,
            overlap_stats={"pain_positive": 0, "pain_negative": 0, "pain_total": 0},
        )

    tagged = tag_reviews_with_pain(df)
    category_table = compute_category_metrics(tagged)
    pain_index = compute_player_pain_index(category_table, tagged)
    veteran = detect_veteran_frustration(tagged)
    escalations = detect_escalations(tagged, category_table)
    emotions = compute_emotion_signals(tagged)
    overlap = _pain_overlap_stats(tagged)

    game_name = "this title"
    if "game_name" in df.columns and df["game_name"].notna().any():
        game_name = str(df["game_name"].dropna().iloc[0])

    result = PainAnalysisResult(
        pain_index=pain_index,
        category_table=category_table,
        veteran_flags=veteran,
        escalations=escalations,
        emotion_table=emotions,
        executive_summary=[],
        recommendations=[],
        tagged_reviews=tagged,
        overlap_stats=overlap,
    )
    result.executive_summary = build_executive_summaries(result, game_name=game_name)
    result.recommendations = build_strategic_recommendations(result)
    return result


def get_cached_pain_analysis():
    import streamlit as st

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached(app_id: int, limit: int, csv_mtime: float, _df: pd.DataFrame) -> dict[str, Any]:
        _ = app_id, limit, csv_mtime
        result = analyze_player_pain(_df)
        return {
            "pain_index": result.pain_index,
            "category_table": result.category_table,
            "veteran_flags": result.veteran_flags,
            "escalations": result.escalations,
            "emotion_table": result.emotion_table,
            "executive_summary": result.executive_summary,
            "recommendations": result.recommendations,
            "overlap_stats": result.overlap_stats,
        }

    return _cached


def pain_result_from_cache(blob: dict[str, Any], tagged: pd.DataFrame) -> PainAnalysisResult:
    return PainAnalysisResult(
        pain_index=int(blob["pain_index"]),
        category_table=blob["category_table"],
        veteran_flags=blob["veteran_flags"],
        escalations=blob["escalations"],
        emotion_table=blob["emotion_table"],
        executive_summary=blob["executive_summary"],
        recommendations=blob["recommendations"],
        tagged_reviews=tagged,
        overlap_stats=blob.get("overlap_stats", {}),
    )


# ---------------------------------------------------------------------------
# Premium UI palette (presentation only — does not affect analytics)
# ---------------------------------------------------------------------------

_PAIN_CHART_HEIGHT = 340
_PAIN_DONUT_HEIGHT = 300
_PAIN_LINE_COLORS = ["#6b9bd1", "#8bafc9", "#9eb8c8", "#c9a86c", "#b89fd4", "#7eb8d4", "#a8b8a0"]
_PAIN_HEATMAP_SCALE = [
    [0.0, "#152238"],
    [0.45, "#2a4a6e"],
    [0.72, "#c9a86c"],
    [1.0, "#e8b896"],
]
_PAIN_EMOTION_ACCENT: dict[str, str] = {
    "frustration": "#c9a86c",
    "anger": "#d48a7a",
    "burnout": "#9a8fb8",
    "excitement": "#6fad8a",
    "addiction": "#6b9bd1",
    "immersion": "#8bafc9",
    "disappointment": "#a89a8a",
}


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _category_accent_color(category: str) -> str:
    """Deterministic muted accent per pain category (UI only)."""
    base = sum(ord(c) for c in category) % 97
    hues = [212, 198, 175, 155, 225, 185, 200]
    h = hues[base % len(hues)]
    return f"hsl({h}, 38%, 58%)"


def _severity_bar_color(severity: float) -> str:
    if severity >= 68:
        return "#d48a7a"
    if severity >= 42:
        return "#c9a86c"
    return "#5b8fb9"


def _pain_index_band(score: int) -> tuple[str, str, str]:
    if score < 40:
        return "#6fad8a", "Contained", "Pain signals are limited in this sample — continue periodic monitoring."
    if score < 65:
        return "#c9a86c", "Elevated", "Recurring friction themes warrant targeted review before they scale."
    return "#d48a7a", "Critical", "Pain concentration is high — prioritize stabilization and player-facing fixes."


def _finalize_pain_fig(fig: go.Figure, title: str, *, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=15, color="#d8e4ec"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18, 28, 40, 0.45)",
        font=dict(color="#a8b8c6", size=11),
        height=height or _PAIN_CHART_HEIGHT,
        margin=dict(l=48, r=24, t=48, b=48),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            x=0,
            font=dict(size=10, color="#8f98a0"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(47, 74, 99, 0.35)",
            linecolor="rgba(47, 74, 99, 0.5)",
            zerolinecolor="rgba(47, 74, 99, 0.35)",
            tickfont=dict(size=10, color="#9aaab8"),
        ),
        yaxis=dict(
            gridcolor="rgba(47, 74, 99, 0.35)",
            linecolor="rgba(47, 74, 99, 0.5)",
            zerolinecolor="rgba(47, 74, 99, 0.35)",
            tickfont=dict(size=10, color="#9aaab8"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def figure_pain_frequency(category_table: pd.DataFrame) -> go.Figure:
    if category_table.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No pain categories detected",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#8f98a0", size=13),
        )
        return _finalize_pain_fig(fig, "Pain frequency")
    tbl = category_table.sort_values("Mentions", ascending=True).tail(12)
    colors = [_severity_bar_color(float(s)) for s in tbl["Severity"]]
    fig = go.Figure(
        go.Bar(
            x=tbl["Mentions"],
            y=tbl["Pain Category"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0), opacity=0.92),
            text=[f"{int(m)}" for m in tbl["Mentions"]],
            textposition="outside",
            textfont=dict(size=10, color="#c5d4e0"),
            hovertemplate="<b>%{y}</b><br>Mentions: %{x}<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.28, xaxis_title="Mentions", yaxis_title="")
    return _finalize_pain_fig(fig, "Pain frequency")


def figure_severity_heatmap(category_table: pd.DataFrame) -> go.Figure:
    if category_table.empty:
        fig = go.Figure()
        fig.add_annotation(text="No severity data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_pain_fig(fig, "Severity heatmap")
    tbl = category_table.sort_values("Severity", ascending=False).head(10)
    z = [tbl["Severity"].tolist()]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=tbl["Pain Category"].tolist(),
            y=["Severity"],
            colorscale=_PAIN_HEATMAP_SCALE,
            zmin=0,
            zmax=100,
            showscale=True,
            colorbar=dict(
                title=dict(text="Score", font=dict(size=10, color="#9aaab8")),
                tickfont=dict(size=9, color="#8f98a0"),
                len=0.65,
                thickness=12,
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ),
            hovertemplate="<b>%{x}</b><br>Severity: %{z:.0f}<extra></extra>",
            xgap=4,
            ygap=4,
        )
    )
    fig.update_layout(xaxis_tickangle=-32, height=_PAIN_CHART_HEIGHT - 20)
    return _finalize_pain_fig(fig, "Severity heatmap", height=_PAIN_CHART_HEIGHT - 20)


def figure_pain_over_time(tagged: pd.DataFrame, top_k: int = 5) -> go.Figure:
    if tagged.empty or "timestamp_created" not in tagged.columns:
        fig = go.Figure()
        fig.add_annotation(text="No timeline data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_pain_fig(fig, "Pain categories over time")
    work = tagged.loc[tagged["has_pain"]].copy()
    if work.empty:
        fig = go.Figure()
        fig.add_annotation(text="No pain-tagged reviews", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_pain_fig(fig, "Pain categories over time")
    ts = pd.to_numeric(work["timestamp_created"], errors="coerce")
    work = work.loc[ts.notna()].copy()
    if work.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid timestamps", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_pain_fig(fig, "Pain categories over time")
    work["review_date"] = pd.to_datetime(work["timestamp_created"], unit="s", utc=True).dt.date
    cat_counts = work["pain_categories"].explode().value_counts()
    top_cats = cat_counts.head(top_k).index.tolist()
    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        d = row["review_date"]
        cats = row["pain_categories"]
        if not isinstance(cats, list):
            continue
        for c in cats:
            if c in top_cats:
                rows.append({"review_date": d, "Pain Category": c})
    if not rows:
        fig = go.Figure()
        return _finalize_pain_fig(fig, "Pain categories over time")
    daily = pd.DataFrame(rows).groupby(["review_date", "Pain Category"], as_index=False).size()
    daily = daily.rename(columns={"size": "mentions"})
    fig = go.Figure()
    for i, cat in enumerate(top_cats):
        sub = daily[daily["Pain Category"] == cat]
        color = _PAIN_LINE_COLORS[i % len(_PAIN_LINE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=sub["review_date"],
                y=sub["mentions"],
                mode="lines+markers",
                name=cat,
                line=dict(color=color, width=2.6, shape="spline"),
                marker=dict(size=6, color=color, line=dict(width=1, color="#1a2636")),
                hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Mentions: %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Mentions",
        showlegend=len(top_cats) > 1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=9)),
        hovermode="x unified",
    )
    return _finalize_pain_fig(fig, "Pain categories over time", height=_PAIN_CHART_HEIGHT + 24)


def figure_pain_sentiment_overlap(overlap: dict[str, int]) -> go.Figure:
    pos = int(overlap.get("pain_positive", 0))
    neg = int(overlap.get("pain_negative", 0))
    if pos + neg == 0:
        fig = go.Figure()
        fig.add_annotation(text="No pain/sentiment overlap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _finalize_pain_fig(fig, "Pain & sentiment overlap", height=_PAIN_DONUT_HEIGHT)
    labels = ["Negative tone", "Positive tone"]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=[neg, pos],
            hole=0.52,
            marker=dict(colors=["#c97d7d", "#6fad8a"], line=dict(color="#1a2636", width=2)),
            textinfo="percent",
            textposition="outside",
            textfont=dict(size=11, color="#c5d4e0"),
            pull=[0.02, 0.02],
            hovertemplate="<b>%{label}</b><br>%{value} reviews · %{percent}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.05, x=0.5, xanchor="center"))
    return _finalize_pain_fig(fig, "Pain & sentiment overlap", height=_PAIN_DONUT_HEIGHT)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def inject_player_pain_css() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
            .pain-shell {
                margin: 2rem 0 2.25rem 0;
                padding: 0 0.15rem;
            }
            .pain-eyebrow {
                font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.14em;
                color: #7a8a9a; margin: 0 0 0.4rem 0;
            }
            .pain-headline {
                font-size: 1.38rem; font-weight: 700; color: #e8f0f6;
                letter-spacing: -0.025em; margin: 0 0 0.45rem 0; line-height: 1.25;
            }
            .pain-lead {
                font-size: 0.88rem; color: #9aaab8; line-height: 1.55;
                margin: 0 0 1.75rem 0; max-width: 52rem;
            }
            .pain-divider {
                height: 1px; margin: 2rem 0 1.75rem 0;
                background: linear-gradient(90deg, transparent, rgba(47,74,99,0.65), transparent);
                border: none;
            }
            .pain-section-title {
                font-size: 0.95rem; font-weight: 600; color: #d0dce6;
                margin: 0 0 0.35rem 0; letter-spacing: -0.01em;
            }
            .pain-section-sub {
                font-size: 0.8rem; color: #8f98a0; margin: 0 0 1.1rem 0;
            }
            .pain-index-card {
                background: linear-gradient(145deg, rgba(32,48,68,0.55) 0%, rgba(18,28,40,0.92) 48%, rgba(14,20,30,0.98) 100%);
                border: 1px solid rgba(102, 192, 244, 0.18);
                border-radius: 16px;
                padding: 1.5rem 1.65rem 1.45rem 1.65rem;
                margin-bottom: 2rem;
                box-shadow: 0 0 0 1px rgba(47,74,99,0.25), 0 18px 48px rgba(0,0,0,0.35),
                    inset 0 1px 0 rgba(255,255,255,0.04);
            }
            .pain-index-card .label {
                font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.11em;
                color: #8f98a0; margin-bottom: 0.5rem;
            }
            .pain-index-card .value-row {
                display: flex; align-items: baseline; gap: 0.65rem; flex-wrap: wrap;
            }
            .pain-index-card .value {
                font-size: 2.65rem; font-weight: 700; line-height: 1;
                letter-spacing: -0.03em;
            }
            .pain-index-card .value-suffix {
                font-size: 1.05rem; font-weight: 500; color: #6a7a8a;
            }
            .pain-index-card .band {
                display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
                font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em;
                font-weight: 600; margin-left: 0.15rem;
            }
            .pain-index-card .sub {
                font-size: 0.86rem; color: #a8b8c6; margin-top: 0.85rem;
                line-height: 1.5; max-width: 40rem;
            }
            .pain-table-wrap {
                border: 1px solid rgba(47,74,99,0.45); border-radius: 14px;
                overflow: hidden; margin-bottom: 2rem;
                box-shadow: 0 8px 28px rgba(0,0,0,0.22);
            }
            .pain-table {
                width: 100%; border-collapse: collapse; font-size: 0.86rem;
            }
            .pain-table thead th {
                text-align: left; padding: 0.85rem 1rem;
                background: rgba(26, 38, 54, 0.95);
                color: #8f98a0; font-weight: 600; font-size: 0.72rem;
                text-transform: uppercase; letter-spacing: 0.06em;
                border-bottom: 1px solid rgba(47,74,99,0.5);
            }
            .pain-table tbody tr {
                transition: background 0.15s ease;
            }
            .pain-table tbody tr:nth-child(even) {
                background: rgba(20, 30, 44, 0.35);
            }
            .pain-table tbody tr:nth-child(odd) {
                background: rgba(16, 24, 34, 0.25);
            }
            .pain-table tbody tr:hover {
                background: rgba(42, 71, 94, 0.28);
            }
            .pain-table tbody td {
                padding: 0.78rem 1rem; color: #c5d4e0;
                border-bottom: 1px solid rgba(47,74,99,0.22);
                vertical-align: middle;
            }
            .pain-table tbody tr:last-child td { border-bottom: none; }
            .pain-cat-cell {
                display: flex; align-items: center; gap: 0.55rem;
                font-weight: 500; color: #dfe6ea;
            }
            .pain-cat-dot {
                width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
                box-shadow: 0 0 8px currentColor;
            }
            .risk-pill {
                display: inline-block; padding: 0.18rem 0.55rem; border-radius: 6px;
                font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
            }
            .risk-pill-low {
                background: rgba(91, 143, 185, 0.18); color: #8eb8d4;
                border: 1px solid rgba(91, 143, 185, 0.35);
            }
            .risk-pill-med {
                background: rgba(201, 168, 108, 0.15); color: #d4bc82;
                border: 1px solid rgba(201, 168, 108, 0.32);
            }
            .risk-pill-high {
                background: rgba(212, 138, 122, 0.14); color: #e0a89a;
                border: 1px solid rgba(212, 138, 122, 0.32);
            }
            .trend-stable { color: #8f98a0; font-size: 0.82rem; }
            .trend-rising { color: #d4bc82; font-size: 0.82rem; font-weight: 500; }
            .trend-falling { color: #8eb8d4; font-size: 0.82rem; }
            .pain-chart-block { margin-bottom: 1.5rem; }
            .pain-emotion-wrap {
                border: 1px solid rgba(47,74,99,0.4); border-radius: 14px;
                overflow: hidden; margin-bottom: 2rem;
            }
            .pain-emotion-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
            .pain-emotion-table th {
                padding: 0.75rem 1rem; text-align: left;
                background: rgba(26, 38, 54, 0.9); color: #8f98a0;
                font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
            }
            .pain-emotion-table td {
                padding: 0.7rem 1rem; color: #c5d4e0;
                border-top: 1px solid rgba(47,74,99,0.22);
            }
            .pain-emotion-table tr:hover td { background: rgba(42, 71, 94, 0.2); }
            .emo-accent {
                display: inline-block; width: 4px; height: 1.1rem;
                border-radius: 2px; margin-right: 0.55rem; vertical-align: middle;
            }
            .pain-exec-grid {
                display: grid; grid-template-columns: 1fr 1fr; gap: 1.15rem;
                margin: 1.75rem 0 0.5rem 0;
            }
            @media (max-width: 900px) {
                .pain-exec-grid { grid-template-columns: 1fr; }
            }
            .pain-exec-card {
                background: linear-gradient(160deg, rgba(24,36,52,0.7) 0%, rgba(14,22,32,0.92) 100%);
                border: 1px solid rgba(47,74,99,0.45); border-radius: 14px;
                padding: 1.15rem 1.25rem 1.2rem 1.25rem;
                box-shadow: 0 10px 32px rgba(0,0,0,0.25);
            }
            .pain-exec-card h4 {
                margin: 0 0 0.85rem 0; font-size: 0.88rem; font-weight: 600;
                color: #c5d4e0; letter-spacing: 0.02em;
            }
            .pain-exec-card ul { margin: 0; padding-left: 1.1rem; }
            .pain-exec-card li {
                color: #b4c4d4; font-size: 0.875rem; line-height: 1.55;
                margin-bottom: 0.55rem;
            }
            .pain-exec-card li:last-child { margin-bottom: 0; }
            .pain-veteran-card {
                background: rgba(201, 168, 108, 0.06);
                border: 1px solid rgba(201, 168, 108, 0.22);
                border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 0.65rem;
            }
            .pain-veteran-tag {
                display: inline-block; padding: 0.2rem 0.55rem; border-radius: 6px;
                font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
                background: rgba(201, 168, 108, 0.12); border: 1px solid rgba(201, 168, 108, 0.3);
                color: #d4bc82; font-weight: 600;
            }
            .pain-escalation-item {
                padding: 0.65rem 0.9rem; margin-bottom: 0.5rem;
                background: rgba(212, 138, 122, 0.06);
                border-left: 3px solid rgba(212, 138, 122, 0.45);
                border-radius: 0 8px 8px 0; font-size: 0.86rem; color: #c5d4e0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_pain_index_card(score: int) -> str:
    color, band, subtitle = _pain_index_band(score)
    band_bg = {
        "Contained": "rgba(111,173,138,0.15)",
        "Elevated": "rgba(201,168,108,0.15)",
        "Critical": "rgba(212,138,122,0.15)",
    }.get(band, "rgba(102,192,244,0.12)")
    return f"""
    <div class="pain-index-card">
        <div class="label">Player Pain Index</div>
        <div class="value-row">
            <span class="value" style="color:{color}">{score}</span>
            <span class="value-suffix">/ 100</span>
            <span class="band" style="color:{color};background:{band_bg};border:1px solid {color}33">{band}</span>
        </div>
        <div class="sub">{_html_escape(subtitle)} Composite of category severity, mention breadth, and negative tone in pain-tagged reviews.</div>
    </div>
    """


def _risk_pill_class(level: str) -> str:
    return {"Low": "risk-pill-low", "Medium": "risk-pill-med", "High": "risk-pill-high"}.get(level, "risk-pill-low")


def _trend_class(signal: str) -> str:
    s = str(signal).lower()
    if "ris" in s:
        return "trend-rising"
    if "fall" in s:
        return "trend-falling"
    return "trend-stable"


def _render_pain_points_table(category_table: pd.DataFrame) -> str:
    headers = ["Pain Category", "Mentions", "Severity", "Risk", "Avg Playtime", "Trend"]
    head_html = "".join(f"<th>{_html_escape(h)}</th>" for h in headers)
    rows_html: list[str] = []
    for _, row in category_table.iterrows():
        cat = str(row.get("Pain Category", ""))
        accent = _category_accent_color(cat)
        risk = str(row.get("Risk Level", "Low"))
        trend = str(row.get("Trend Signal", "Stable"))
        rows_html.append(
            f"<tr>"
            f'<td><div class="pain-cat-cell"><span class="pain-cat-dot" style="background:{accent};color:{accent}"></span>'
            f"{_html_escape(cat)}</div></td>"
            f"<td>{int(row.get('Mentions', 0))}</td>"
            f"<td>{float(row.get('Severity', 0)):.0f}</td>"
            f'<td><span class="risk-pill {_risk_pill_class(risk)}">{_html_escape(risk)}</span></td>'
            f"<td>{float(row.get('Avg Playtime (h)', 0)):.1f}h</td>"
            f'<td><span class="{_trend_class(trend)}">{_html_escape(trend)}</span></td>'
            f"</tr>"
        )
    body = "\n".join(rows_html)
    return f'<div class="pain-table-wrap"><table class="pain-table"><thead><tr>{head_html}</tr></thead><tbody>{body}</tbody></table></div>'


def _render_emotion_table_html(emotion_table: pd.DataFrame) -> str:
    head = "<tr><th>Emotion</th><th>Mentions</th><th>Share</th><th>Negative tone</th></tr>"
    rows: list[str] = []
    for _, row in emotion_table.iterrows():
        emo = str(row.get("Emotion", ""))
        key = emo.lower()
        accent = _PAIN_EMOTION_ACCENT.get(key, "#8bafc9")
        rows.append(
            f"<tr>"
            f'<td><span class="emo-accent" style="background:{accent}"></span>{_html_escape(emo)}</td>'
            f"<td>{int(row.get('Mentions', 0))}</td>"
            f"<td>{float(row.get('Share %', 0)):.1f}%</td>"
            f"<td>{float(row.get('Negative %', 0)):.1f}%</td>"
            f"</tr>"
        )
    return f'<div class="pain-emotion-wrap"><table class="pain-emotion-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def _render_executive_cards(summaries: list[str], recommendations: list[str]) -> str:
    sum_items = "".join(f"<li>{_html_escape(s)}</li>" for s in summaries)
    rec_items = "".join(f"<li>{_html_escape(r)}</li>" for r in recommendations)
    return f"""
    <div class="pain-exec-grid">
        <div class="pain-exec-card">
            <h4>Executive summary</h4>
            <ul>{sum_items or "<li>No summary available for this sample.</li>"}</ul>
        </div>
        <div class="pain-exec-card">
            <h4>Strategic recommendations</h4>
            <ul>{rec_items or "<li>Continue periodic review sampling.</li>"}</ul>
        </div>
    </div>
    """


def render_player_pain_intelligence(
    df_live: pd.DataFrame,
    *,
    app_id: int,
    limit: int,
    csv_mtime: float = 0.0,
) -> None:
    """Player Pain Intelligence section for the Live Review module."""
    import streamlit as st

    if df_live.empty:
        return

    inject_player_pain_css()
    st.markdown('<div class="pain-shell">', unsafe_allow_html=True)
    st.markdown('<p class="pain-eyebrow">Player Pain Intelligence Layer</p>', unsafe_allow_html=True)
    st.markdown('<h3 class="pain-headline">Player Pain Intelligence</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pain-lead">Rule-based complaint taxonomy, severity scoring, and escalation signals — '
        "reuses your cached review sample with no additional Steam calls.</p>",
        unsafe_allow_html=True,
    )

    try:
        tagged = tag_reviews_with_pain(df_live)
        cached_fn = get_cached_pain_analysis()
        blob = cached_fn(app_id, limit, csv_mtime, df_live)
        result = pain_result_from_cache(blob, tagged)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Player Pain Intelligence could not run ({exc}).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(_render_pain_index_card(result.pain_index), unsafe_allow_html=True)

    st.markdown('<hr class="pain-divider">', unsafe_allow_html=True)
    st.markdown('<p class="pain-section-title">Top Player Pain Points</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pain-section-sub">Ranked by severity and mention volume · risk and trend from this sample.</p>',
        unsafe_allow_html=True,
    )
    if result.category_table.empty:
        st.info("No pain-category keyword hits in this sample — try a larger limit or another App ID.")
    else:
        st.markdown(_render_pain_points_table(result.category_table), unsafe_allow_html=True)

    st.markdown('<hr class="pain-divider">', unsafe_allow_html=True)
    st.markdown('<p class="pain-section-title">Pain analytics</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pain-section-sub">Frequency, severity, timeline, and sentiment overlap for pain-tagged reviews.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="pain-chart-block">', unsafe_allow_html=True)
        st.plotly_chart(figure_pain_frequency(result.category_table), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="pain-chart-block">', unsafe_allow_html=True)
        st.plotly_chart(figure_pain_over_time(result.tagged_reviews), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="pain-chart-block">', unsafe_allow_html=True)
        st.plotly_chart(figure_severity_heatmap(result.category_table), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="pain-chart-block">', unsafe_allow_html=True)
        st.plotly_chart(figure_pain_sentiment_overlap(result.overlap_stats), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    if result.veteran_flags:
        st.markdown('<hr class="pain-divider">', unsafe_allow_html=True)
        st.markdown('<p class="pain-section-title">High engagement frustration</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="pain-section-sub">Veteran players with high playtime leaving negative, pain-tagged reviews.</p>',
            unsafe_allow_html=True,
        )
        for flag in result.veteran_flags[:8]:
            st.markdown(
                f'<div class="pain-veteran-card">'
                f'<span class="pain-veteran-tag">{_html_escape(flag["label"])}</span> '
                f'<strong style="color:#dfe6ea">{flag["playtime_h"]}h</strong> '
                f'<span style="color:#9aaab8">· {_html_escape(flag["categories"])}</span>'
                f'<p style="margin:0.45rem 0 0 0;font-size:0.82rem;color:#8f98a0;line-height:1.45">'
                f'{_html_escape(flag["snippet"])}</p></div>',
                unsafe_allow_html=True,
            )

    if result.escalations:
        st.markdown('<hr class="pain-divider">', unsafe_allow_html=True)
        st.markdown('<p class="pain-section-title">Review escalation detection</p>', unsafe_allow_html=True)
        for line in result.escalations:
            st.markdown(f'<div class="pain-escalation-item">{line}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="pain-divider">', unsafe_allow_html=True)
    st.markdown('<p class="pain-section-title">Player emotion signals</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pain-section-sub">Lexicon-detected emotional tone across the review sample.</p>',
        unsafe_allow_html=True,
    )
    if result.emotion_table.empty:
        st.caption("No emotion lexicon hits in this sample.")
    else:
        st.markdown(_render_emotion_table_html(result.emotion_table), unsafe_allow_html=True)

    st.markdown(
        _render_executive_cards(result.executive_summary, result.recommendations),
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
