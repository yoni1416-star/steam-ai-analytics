"""
Game Discovery — lightweight Steam App ID picker (static lists, no external APIs).

Global selection (session): ``selected_game_name``, ``selected_app_id``, and
``steam_live_app_id_field``. Catalog picks use the searchable dropdown only.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

RECENT_GAMES_KEY = "recently_analyzed_games"
SELECTION_MSG_KEY = "game_discovery_selection_msg"
APP_ID_FIELD_KEY = "steam_live_app_id_field"
SELECTED_GAME_NAME_KEY = "selected_game_name"
SELECTED_APP_ID_KEY = "selected_app_id"
WIDGET_SYNC_FLAG_KEY = "_gd_needs_widget_sync"
SELECTION_WARNING_KEY = "game_discovery_selection_warning"
SEARCH_SELECT_KEY = "gd_game_search_app_id"
SEARCH_SYNCED_KEY = "gd_game_search_app_id_synced"
LEGACY_SEARCH_SELECT_KEY = "gd_popular_search_select"
DROPDOWN_PLACEHOLDER_APP_ID = 0
MAX_RECENT_GAMES = 5
_SEARCH_PLACEHOLDER = "— Type to search popular games —"
MAPPING_ERROR_MSG = "No valid App ID mapping found for this game."


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


def _validate_registry(games: list[DiscoveryGame], *, label: str) -> tuple[dict[str, DiscoveryGame], dict[int, DiscoveryGame]]:
    """Build name/app_id maps; reject duplicate names or App IDs in a list."""
    by_name: dict[str, DiscoveryGame] = {}
    by_id: dict[int, DiscoveryGame] = {}
    for game in games:
        name = str(game["game_name"]).strip()
        if not name:
            raise ValueError(f"{label}: game_name must be non-empty")
        try:
            aid = int(game["app_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: invalid app_id for {name!r}") from exc
        if aid <= 0:
            raise ValueError(f"{label}: app_id must be positive for {name!r} ({aid})")
        if name in by_name:
            raise ValueError(f"{label}: duplicate game_name {name!r}")
        if aid in by_id:
            raise ValueError(f"{label}: duplicate app_id {aid} ({name!r} vs {by_id[aid]['game_name']!r})")
        canonical: DiscoveryGame = {
            "game_name": name,
            "app_id": aid,
            "category": str(game.get("category", "")).strip() or "Game",
        }
        by_name[name] = canonical
        by_id[aid] = canonical
    return by_name, by_id


_POPULAR_BY_NAME, REGISTRY_BY_APP_ID = _validate_registry(POPULAR_GAMES, label="POPULAR_GAMES")

GAME_REGISTRY: list[DiscoveryGame] = sorted(
    POPULAR_GAMES,
    key=lambda g: str(g["game_name"]).lower(),
)
DROPDOWN_APP_IDS: list[int] = [DROPDOWN_PLACEHOLDER_APP_ID] + [int(g["app_id"]) for g in GAME_REGISTRY]


class RecentGame(TypedDict):
    game_name: str
    app_id: int
    timestamp: float


def init_game_discovery_state() -> None:
    import streamlit as st

    if RECENT_GAMES_KEY not in st.session_state:
        st.session_state[RECENT_GAMES_KEY] = []


def init_global_game_selection() -> None:
    """Ensure global selection keys exist (values may be None until first pick)."""
    import streamlit as st

    st.session_state.setdefault(SELECTED_GAME_NAME_KEY, None)
    st.session_state.setdefault(SELECTED_APP_ID_KEY, None)


def get_selected_game_name() -> str | None:
    import streamlit as st

    name = st.session_state.get(SELECTED_GAME_NAME_KEY)
    return str(name).strip() if name else None


def get_selected_app_id() -> int | None:
    import streamlit as st

    raw = st.session_state.get(SELECTED_APP_ID_KEY)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _request_widget_sync() -> None:
    import streamlit as st

    st.session_state[WIDGET_SYNC_FLAG_KEY] = True


def resolve_game_selection(game_name: str, app_id: int) -> tuple[DiscoveryGame | None, str | None]:
    """Resolve to a canonical game; never fall back to a different title."""
    try:
        aid = int(app_id)
    except (TypeError, ValueError):
        return None, MAPPING_ERROR_MSG
    if aid <= 0:
        return None, MAPPING_ERROR_MSG

    if aid in REGISTRY_BY_APP_ID:
        return REGISTRY_BY_APP_ID[aid], None

    name = str(game_name).strip()
    if name:
        return {"game_name": name, "app_id": aid, "category": "Recent"}, None

    return None, MAPPING_ERROR_MSG


def _set_selection_warning(message: str) -> None:
    import streamlit as st

    st.session_state[SELECTION_WARNING_KEY] = message


def _clear_selection_warning() -> None:
    import streamlit as st

    st.session_state.pop(SELECTION_WARNING_KEY, None)


def _format_dropdown_option(app_id: int) -> str:
    if int(app_id) == DROPDOWN_PLACEHOLDER_APP_ID:
        return _SEARCH_PLACEHOLDER
    game = REGISTRY_BY_APP_ID.get(int(app_id))
    if game is None:
        return f"Unknown App ID {int(app_id)}"
    return f"{game['game_name']} · {game['category']}"


def _init_search_dropdown_state() -> None:
    """App-ID keyed dropdown; migrate legacy name-based session values once."""
    import streamlit as st

    legacy = st.session_state.pop(LEGACY_SEARCH_SELECT_KEY, None)
    if legacy is not None and SEARCH_SELECT_KEY not in st.session_state:
        if isinstance(legacy, str) and legacy in _POPULAR_BY_NAME:
            st.session_state[SEARCH_SELECT_KEY] = int(_POPULAR_BY_NAME[legacy]["app_id"])
        else:
            st.session_state[SEARCH_SELECT_KEY] = DROPDOWN_PLACEHOLDER_APP_ID

    if SEARCH_SELECT_KEY not in st.session_state:
        st.session_state[SEARCH_SELECT_KEY] = DROPDOWN_PLACEHOLDER_APP_ID
    if SEARCH_SYNCED_KEY not in st.session_state:
        st.session_state[SEARCH_SYNCED_KEY] = DROPDOWN_PLACEHOLDER_APP_ID


def _restore_dropdown_to_global() -> None:
    """Reset dropdown widget to match global selection without changing App ID field."""
    import streamlit as st

    app_id = get_selected_app_id()
    if app_id is not None and app_id in REGISTRY_BY_APP_ID:
        st.session_state[SEARCH_SELECT_KEY] = app_id
        st.session_state[SEARCH_SYNCED_KEY] = app_id
    else:
        st.session_state[SEARCH_SELECT_KEY] = DROPDOWN_PLACEHOLDER_APP_ID
        st.session_state[SEARCH_SYNCED_KEY] = DROPDOWN_PLACEHOLDER_APP_ID


def _sync_dropdown_from_global() -> None:
    """Push global selection into dropdown and App ID field."""
    import streamlit as st

    app_id = get_selected_app_id()

    if app_id is not None:
        st.session_state[APP_ID_FIELD_KEY] = str(app_id)

    if app_id is not None and app_id in REGISTRY_BY_APP_ID:
        st.session_state[SEARCH_SELECT_KEY] = app_id
        st.session_state[SEARCH_SYNCED_KEY] = app_id
    else:
        st.session_state[SEARCH_SELECT_KEY] = DROPDOWN_PLACEHOLDER_APP_ID
        st.session_state[SEARCH_SYNCED_KEY] = DROPDOWN_PLACEHOLDER_APP_ID
        if app_id is not None and app_id not in REGISTRY_BY_APP_ID:
            _set_selection_warning(
                "Selected game is not in the catalog dropdown; App ID field preserved."
            )


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


def select_game_for_analysis(game_name: str, app_id: int, *, record_recent: bool = True) -> bool:
    """Single source of truth for live-module game selection (no fetch). Returns False on invalid mapping."""
    import streamlit as st

    resolved, err = resolve_game_selection(game_name, app_id)
    if err or resolved is None:
        _set_selection_warning(err or MAPPING_ERROR_MSG)
        return False

    _clear_selection_warning()
    title = resolved["game_name"]
    aid = int(resolved["app_id"])

    st.session_state[SELECTED_GAME_NAME_KEY] = title
    st.session_state[SELECTED_APP_ID_KEY] = aid
    st.session_state[APP_ID_FIELD_KEY] = str(aid)
    st.session_state[SELECTION_MSG_KEY] = (
        f"Selected **{title}**. Now click **Fetch Steam Data** to begin analysis."
    )
    _request_widget_sync()
    if record_recent:
        record_recent_game(title, aid)
    return True


def _selection_confirmation_text() -> str | None:
    name = get_selected_game_name()
    if not name:
        return None
    return f"Selected **{name}**. Now click **Fetch Steam Data** to begin analysis."


def inject_game_discovery_css() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
            .gd-section {
                margin: 0 0 0.65rem 0;
                padding: 0.75rem 0.95rem 0.85rem 0.95rem;
                background: rgba(20, 30, 44, 0.45);
                border: 1px solid #2a475e;
                border-radius: 12px;
            }
            .gd-section-title {
                font-size: 0.98rem;
                font-weight: 700;
                color: #e5eef5;
                margin: 0 0 0.2rem 0;
                letter-spacing: -0.02em;
            }
            .gd-section-sub {
                font-size: 0.78rem;
                color: #8f98a0;
                margin: 0;
                line-height: 1.4;
            }
            .gd-onboard-box {
                margin-bottom: 0.85rem;
                background: linear-gradient(160deg, rgba(26, 40, 58, 0.92) 0%, rgba(16, 24, 36, 0.95) 100%);
                border: 1px solid rgba(102, 192, 244, 0.35);
                border-radius: 12px;
                padding: 0.9rem 1rem 1rem 1rem;
                box-shadow: 0 0 18px rgba(61, 139, 198, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.04);
            }
            .gd-onboard-box--compact {
                padding: 0.8rem 0.9rem 0.9rem 0.9rem;
            }
            .gd-onboard-title {
                font-size: 0.95rem;
                font-weight: 700;
                color: #c7e4f7;
                margin: 0 0 0.55rem 0;
                letter-spacing: -0.01em;
            }
            .gd-onboard-steps {
                margin: 0;
                padding-left: 1.15rem;
                color: #b8c5d0;
                font-size: 0.82rem;
                line-height: 1.55;
            }
            .gd-onboard-steps li {
                margin-bottom: 0.3rem;
            }
            .gd-onboard-steps li:last-child {
                margin-bottom: 0;
            }
            .gd-onboard-starter-list {
                margin: 0 0 0.45rem 0;
                padding-left: 1rem;
                color: #9eb0c0;
                font-size: 0.8rem;
                line-height: 1.5;
            }
            .gd-onboard-starter-list li {
                margin-bottom: 0.15rem;
            }
            .gd-onboard-flow {
                font-size: 0.78rem;
                color: #66c0f4;
                margin: 0.35rem 0 0 0;
                letter-spacing: 0.01em;
            }
            .gd-onboard-flow strong {
                color: #dfe6ea;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_game_search_dropdown() -> None:
    """Single catalog selector — only interactive game picker in Game Discovery."""
    import streamlit as st

    _init_search_dropdown_state()

    choice_id = st.selectbox(
        "Select a popular game",
        options=DROPDOWN_APP_IDS,
        key=SEARCH_SELECT_KEY,
        format_func=_format_dropdown_option,
        help="Type to filter the full catalog, then pick a title to fill the App ID field.",
    )
    try:
        choice_app_id = int(choice_id)
    except (TypeError, ValueError):
        _set_selection_warning(MAPPING_ERROR_MSG)
        _restore_dropdown_to_global()
        return

    synced = st.session_state.get(SEARCH_SYNCED_KEY)
    try:
        synced_id = int(synced) if synced is not None else DROPDOWN_PLACEHOLDER_APP_ID
    except (TypeError, ValueError):
        synced_id = DROPDOWN_PLACEHOLDER_APP_ID

    if choice_app_id == DROPDOWN_PLACEHOLDER_APP_ID:
        if synced_id != DROPDOWN_PLACEHOLDER_APP_ID:
            st.session_state[SEARCH_SYNCED_KEY] = DROPDOWN_PLACEHOLDER_APP_ID
        return

    if choice_app_id == synced_id:
        return

    game = REGISTRY_BY_APP_ID.get(choice_app_id)
    if game is None:
        _set_selection_warning(MAPPING_ERROR_MSG)
        _restore_dropdown_to_global()
        return

    if get_selected_app_id() == choice_app_id:
        st.session_state[SEARCH_SYNCED_KEY] = choice_app_id
        return

    if select_game_for_analysis(game["game_name"], game["app_id"]):
        st.session_state[SEARCH_SYNCED_KEY] = choice_app_id
        st.rerun()
        return

    st.session_state[SEARCH_SYNCED_KEY] = choice_app_id
    _restore_dropdown_to_global()


def render_onboarding_guidance() -> None:
    """Compact first-run guidance above Game Discovery (UI only)."""
    import streamlit as st

    col_how, col_new = st.columns([1.65, 1])
    with col_how:
        st.markdown(
            '<div class="gd-onboard-box">'
            '<p class="gd-onboard-title">How to use</p>'
            "<ol class='gd-onboard-steps'>"
            "<li>Choose a game from the <strong>dropdown</strong> below or enter a Steam App ID manually</li>"
            "<li>Select how many reviews to analyze</li>"
            '<li>Click <strong>"Fetch Steam Data"</strong></li>'
            "<li>Explore player pain analysis, sentiment, trends, and executive insights</li>"
            "</ol>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_new:
        st.markdown(
            '<div class="gd-onboard-box gd-onboard-box--compact">'
            '<p class="gd-onboard-title">New here?</p>'
            "<p style='margin:0 0 0.35rem 0;font-size:0.8rem;color:#9eb0c0;'>Try analyzing:</p>"
            "<ul class='gd-onboard-starter-list'>"
            "<li>Battlefield 6</li>"
            "<li>Counter-Strike 2</li>"
            "<li>PUBG</li>"
            "</ul>"
            '<p class="gd-onboard-flow"><strong>Select game</strong> → <strong>Fetch Steam Data</strong></p>'
            "</div>",
            unsafe_allow_html=True,
        )


def render_game_discovery_section() -> None:
    """Game Discovery block — render above the Steam App ID input."""
    import streamlit as st

    init_game_discovery_state()
    init_global_game_selection()
    _init_search_dropdown_state()
    if st.session_state.get(WIDGET_SYNC_FLAG_KEY):
        _sync_dropdown_from_global()
        st.session_state[WIDGET_SYNC_FLAG_KEY] = False

    inject_game_discovery_css()
    render_onboarding_guidance()

    st.markdown(
        '<div class="gd-section">'
        '<p class="gd-section-title">Game Discovery</p>'
        '<p class="gd-section-sub">Search the catalog below or enter an App ID manually, then fetch when ready.</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    _render_game_search_dropdown()


def render_selection_message() -> None:
    import streamlit as st

    warning = st.session_state.get(SELECTION_WARNING_KEY)
    if warning:
        st.warning(warning, icon=None)

    text = _selection_confirmation_text()
    if text:
        st.info(text, icon=None)
