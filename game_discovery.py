"""
Game Discovery — lightweight Steam App ID picker (static lists, no external APIs).
"""

from __future__ import annotations

import html
import time
from typing import Any, TypedDict

RECENT_GAMES_KEY = "recently_analyzed_games"
SELECTION_MSG_KEY = "game_discovery_selection_msg"
APP_ID_FIELD_KEY = "steam_live_app_id_field"
MAX_RECENT_GAMES = 5
CARDS_PER_ROW = 3


class DiscoveryGame(TypedDict):
    game_name: str
    app_id: int
    category: str


POPULAR_GAMES: list[DiscoveryGame] = [
    {"game_name": "Battlefield 6", "app_id": 2807960, "category": "FPS"},
    {"game_name": "Counter-Strike 2", "app_id": 730, "category": "FPS"},
    {"game_name": "PUBG: Battlegrounds", "app_id": 578080, "category": "Battle Royale"},
    {"game_name": "Dota 2", "app_id": 570, "category": "MOBA"},
    {"game_name": "Apex Legends", "app_id": 1172470, "category": "Battle Royale"},
    {"game_name": "Cyberpunk 2077", "app_id": 1091500, "category": "RPG"},
    {"game_name": "Elden Ring", "app_id": 1245620, "category": "Action RPG"},
    {"game_name": "Battlefield 2042", "app_id": 1517290, "category": "FPS"},
    {"game_name": "Dying Light", "app_id": 239140, "category": "Action"},
    {"game_name": "Dying Light 2 Stay Human", "app_id": 534380, "category": "Action"},
    {"game_name": "Dying Light: The Beast", "app_id": 3008130, "category": "Action"},
    {"game_name": "Rust", "app_id": 252490, "category": "Survival"},
    {"game_name": "GTA V Legacy", "app_id": 271590, "category": "Action"},
    {"game_name": "No Man's Sky", "app_id": 275850, "category": "Adventure"},
    {"game_name": "Hades", "app_id": 1145360, "category": "Roguelike"},
    {"game_name": "Helldivers 2", "app_id": 553850, "category": "Co-op Shooter"},
    {"game_name": "Starfield", "app_id": 1716740, "category": "RPG"},
    {"game_name": "The Last of Us Part I", "app_id": 1888930, "category": "Action"},
    {"game_name": "Resident Evil 4", "app_id": 2050650, "category": "Survival Horror"},
    {"game_name": "The Outlast Trials", "app_id": 1304930, "category": "Horror"},
]

_TRENDING_NAMES = (
    "Battlefield 6",
    "Counter-Strike 2",
    "PUBG: Battlegrounds",
    "Helldivers 2",
    "Apex Legends",
    "Dying Light: The Beast",
    "Cyberpunk 2077",
    "Elden Ring",
)

_POPULAR_BY_NAME = {g["game_name"]: g for g in POPULAR_GAMES}

TRENDING_GAMES: list[DiscoveryGame] = [
    _POPULAR_BY_NAME[name] for name in _TRENDING_NAMES if name in _POPULAR_BY_NAME
]


class RecentGame(TypedDict):
    game_name: str
    app_id: int
    timestamp: float


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def init_game_discovery_state() -> None:
    import streamlit as st

    if RECENT_GAMES_KEY not in st.session_state:
        st.session_state[RECENT_GAMES_KEY] = []


def record_recent_game(game_name: str, app_id: int) -> None:
    """Keep the last five unique App IDs (most recent first)."""
    import streamlit as st

    init_game_discovery_state()
    entry: RecentGame = {
        "game_name": str(game_name).strip() or f"Steam App {app_id}",
        "app_id": int(app_id),
        "timestamp": time.time(),
    }
    prior: list[dict[str, Any]] = list(st.session_state.get(RECENT_GAMES_KEY) or [])
    filtered = [r for r in prior if int(r.get("app_id", -1)) != entry["app_id"]]
    filtered.insert(0, entry)
    st.session_state[RECENT_GAMES_KEY] = filtered[:MAX_RECENT_GAMES]


def select_game_for_analysis(game_name: str, app_id: int, *, record_recent: bool = True) -> None:
    """Populate the live lookup App ID field without triggering a fetch."""
    import streamlit as st

    st.session_state[APP_ID_FIELD_KEY] = str(int(app_id))
    st.session_state[SELECTION_MSG_KEY] = (
        f"Selected **{game_name}** — App ID **{int(app_id)}**. "
        "Click **Fetch Steam Data** to continue."
    )
    if record_recent:
        record_recent_game(game_name, app_id)


def inject_game_discovery_css() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
            .gd-section {
                margin: 0 0 1.35rem 0;
                padding: 1rem 1.1rem 1.15rem 1.1rem;
                background: rgba(20, 30, 44, 0.45);
                border: 1px solid #2a475e;
                border-radius: 14px;
            }
            .gd-section-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #e5eef5;
                margin: 0 0 0.25rem 0;
                letter-spacing: -0.02em;
            }
            .gd-section-sub {
                font-size: 0.84rem;
                color: #8f98a0;
                margin: 0 0 1rem 0;
                line-height: 1.45;
            }
            .gd-card-shell {
                background: linear-gradient(165deg, #1a2636 0%, #141c28 55%, #101820 100%);
                border: 1px solid #2f4a63;
                border-radius: 12px;
                padding: 0.85rem 0.9rem 0.35rem 0.9rem;
                margin-bottom: 0.35rem;
                min-height: 7.25rem;
                transition: border-color 0.15s ease, box-shadow 0.15s ease;
            }
            .gd-card-shell:hover {
                border-color: #3d6a8a;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            }
            .gd-card-name {
                font-size: 0.92rem;
                font-weight: 600;
                color: #dfe6ea;
                margin: 0 0 0.35rem 0;
                line-height: 1.3;
            }
            .gd-card-meta {
                font-size: 0.78rem;
                color: #8f98a0;
                margin: 0 0 0.2rem 0;
            }
            .gd-card-category {
                display: inline-block;
                font-size: 0.68rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #66c0f4;
                background: rgba(102, 192, 244, 0.1);
                border: 1px solid rgba(102, 192, 244, 0.25);
                border-radius: 6px;
                padding: 0.15rem 0.45rem;
                margin-top: 0.35rem;
            }
            .gd-recent-time {
                font-size: 0.72rem;
                color: #6d7a88;
                margin-top: 0.25rem;
            }
            div[data-testid="stTabs"] button {
                font-weight: 600;
            }
            @media (max-width: 768px) {
                .gd-card-shell { min-height: auto; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_recent_time(ts: float) -> str:
    try:
        delta = max(0, time.time() - float(ts))
    except (TypeError, ValueError):
        return ""
    if delta < 60:
        return "Just now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} hr ago"
    return f"{int(delta // 86400)} d ago"


def _card_markup(game: DiscoveryGame) -> str:
    return (
        f'<div class="gd-card-shell">'
        f'<p class="gd-card-name">{_esc(game["game_name"])}</p>'
        f'<p class="gd-card-meta">App ID {_esc(str(game["app_id"]))}</p>'
        f'<span class="gd-card-category">{_esc(game["category"])}</span>'
        f"</div>"
    )


def _render_game_grid(
    games: list[DiscoveryGame],
    *,
    key_prefix: str,
    button_label: str = "Analyze",
) -> None:
    import streamlit as st

    for row_start in range(0, len(games), CARDS_PER_ROW):
        row = games[row_start : row_start + CARDS_PER_ROW]
        cols = st.columns(CARDS_PER_ROW)
        for col_idx, col in enumerate(cols):
            with col:
                if col_idx >= len(row):
                    continue
                game = row[col_idx]
                st.markdown(_card_markup(game), unsafe_allow_html=True)
                btn_key = f"{key_prefix}_{game['app_id']}"
                if st.button(button_label, key=btn_key, use_container_width=True):
                    select_game_for_analysis(game["game_name"], game["app_id"])
                    st.rerun()


def _render_recent_tab() -> None:
    import streamlit as st

    init_game_discovery_state()
    recent: list[dict[str, Any]] = list(st.session_state.get(RECENT_GAMES_KEY) or [])
    if not recent:
        st.caption("No games analyzed yet. Pick a title above or fetch reviews to build this list.")
        return

    for idx, entry in enumerate(recent[:MAX_RECENT_GAMES]):
        name = str(entry.get("game_name", "Unknown"))
        try:
            aid = int(entry.get("app_id", 0))
        except (TypeError, ValueError):
            continue
        ts = float(entry.get("timestamp", 0) or 0)
        time_label = _format_recent_time(ts)
        time_html = f'<p class="gd-recent-time">{_esc(time_label)}</p>' if time_label else ""
        st.markdown(
            f'<div class="gd-card-shell">'
            f'<p class="gd-card-name">{_esc(name)}</p>'
            f'<p class="gd-card-meta">App ID {_esc(str(aid))}</p>'
            f"{time_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("Analyze Again", key=f"gd_recent_{aid}_{idx}", use_container_width=True):
            select_game_for_analysis(name, aid)
            st.rerun()


def render_game_discovery_section() -> None:
    """Game Discovery block — render above the Steam App ID input."""
    import streamlit as st

    init_game_discovery_state()
    inject_game_discovery_css()

    st.markdown(
        '<div class="gd-section">'
        '<p class="gd-section-title">Game Discovery</p>'
        '<p class="gd-section-sub">Browse popular and trending titles, or reopen a recent analysis. '
        "Selecting a game fills the App ID field only — use Fetch Steam Data when ready.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    tab_popular, tab_trending, tab_recent = st.tabs(
        ["Popular Games", "Trending Games", "Recently Analyzed"]
    )

    with tab_popular:
        _render_game_grid(POPULAR_GAMES, key_prefix="gd_popular")
    with tab_trending:
        _render_game_grid(TRENDING_GAMES, key_prefix="gd_trending")
    with tab_recent:
        _render_recent_tab()


def render_selection_message() -> None:
    import streamlit as st

    msg = st.session_state.pop(SELECTION_MSG_KEY, None)
    if msg:
        st.info(msg, icon=None)
