"""
Sports Leaderboard — single-process Streamlit app.

No Flask backend. State is shared across all browser sessions via
@st.cache_resource (all phones hit the same Python process).
Schedule + scores are persisted to a JSON file so data survives restarts.

Gemini API key is read from config.yaml (gemini.api_key).
"""

import base64
import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import requests

import pandas as pd
import streamlit as st
import yaml

logging.basicConfig(level=logging.INFO)

# ===========================================================================
# Config  (config.yaml)
# ===========================================================================

@st.cache_resource
def _cfg() -> dict:
    """
    Load config in priority order:
      1. st.secrets  — Streamlit Community Cloud (secrets set in the dashboard)
      2. config.yaml — local development
    """
    # ── Streamlit Community Cloud ──────────────────────────────────────────
    try:
        if "gemini" in st.secrets:
            return {
                "gemini": dict(st.secrets["gemini"]),
                "app":    dict(st.secrets.get("app", {})),
            }
    except Exception:
        pass

    # ── Local dev: config.yaml ─────────────────────────────────────────────
    p = Path("config.yaml")
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning("Could not read config.yaml: %s", e)
    return {}


def _gemini_key() -> str:
    key = _cfg().get("gemini", {}).get("api_key", "")
    if key and not key.startswith("YOUR_"):
        return key
    return ""


def _data_dir() -> Path:
    # Explicit env var wins (useful for Azure App Service)
    env = os.getenv("DATA_DIR")
    if env:
        return Path(env)
    # config / secrets value
    cfg_dir = _cfg().get("app", {}).get("data_dir", "")
    if cfg_dir:
        return Path(cfg_dir)
    # Default: ./data locally, /tmp on Linux-based clouds
    import platform
    return Path("./data") if platform.system() == "Windows" else Path("/tmp/leaderboard")


def _default_rounds() -> int:
    return int(_cfg().get("app", {}).get("default_rounds", 12))


# ===========================================================================
# Shared persistent state
# ===========================================================================

_file_lock = threading.Lock()

# ---------------------------------------------------------------------------
# GitHub persistence helpers
# ---------------------------------------------------------------------------

_GH_PATH = "data/session.json"   # path inside the repo


def _gh_cfg() -> dict:
    """Return the [github] config block from secrets or config.yaml."""
    try:
        if "github" in st.secrets:
            return dict(st.secrets["github"])
    except Exception:
        pass
    return _cfg().get("github", {})


def _gh_headers() -> dict:
    token = _gh_cfg().get("token", "")
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _gh_url() -> str:
    repo = _gh_cfg().get("repo", "")
    return f"https://api.github.com/repos/{repo}/contents/{_GH_PATH}"


def _github_load() -> Optional[dict]:
    """Fetch session.json from GitHub. Returns None if unavailable."""
    if not _gh_cfg().get("token") or not _gh_cfg().get("repo"):
        return None
    try:
        r = requests.get(_gh_url(), headers=_gh_headers(), timeout=10)
        if r.status_code == 200:
            payload = r.json()
            content = base64.b64decode(payload["content"]).decode("utf-8")
            return json.loads(content)
        if r.status_code == 404:
            return _empty_state()   # file doesn't exist yet — start fresh
        logging.warning("GitHub load HTTP %s", r.status_code)
    except Exception as exc:
        logging.warning("GitHub load error: %s", exc)
    return None


def _github_save(state: dict) -> None:
    """Push session.json to GitHub (create or update)."""
    if not _gh_cfg().get("token") or not _gh_cfg().get("repo"):
        return
    try:
        hdrs = _gh_headers()
        url  = _gh_url()

        # Always fetch the current SHA before writing — avoids 422 errors
        # caused by Streamlit resetting module-level variables on every rerun.
        sha = ""
        r_get = requests.get(url, headers=hdrs, timeout=10)
        if r_get.status_code == 200:
            sha = r_get.json().get("sha", "")

        content_b64 = base64.b64encode(
            json.dumps(state, indent=2).encode("utf-8")
        ).decode("utf-8")
        payload: dict = {
            "message": "leaderboard: update session",
            "content": content_b64,
        }
        if sha:
            payload["sha"] = sha

        r = requests.put(url, headers=hdrs, json=payload, timeout=15)
        if r.status_code in (200, 201):
            logging.info("GitHub save OK (status %s)", r.status_code)
        else:
            logging.warning("GitHub save HTTP %s: %s", r.status_code, r.text[:200])
    except Exception as exc:
        logging.warning("GitHub save error: %s", exc)


# ---------------------------------------------------------------------------
# Shared persistent state
# ---------------------------------------------------------------------------

def _data_file() -> Path:
    return _data_dir() / "session.json"


def _empty_state() -> dict:
    return {
        "players": [],
        "skill_levels": {},
        "schedule": [],
        "scores": {},
        "session_active": False,
    }


def _load_state() -> dict:
    """Load state: GitHub first (persistent), local file as fallback."""
    gh = _github_load()
    if gh is not None:
        logging.info("State loaded from GitHub")
        return gh
    # fallback: local JSON file
    try:
        f = _data_file()
        if f.exists():
            with open(f, encoding="utf-8") as fp:
                logging.info("State loaded from local file")
                return json.load(fp)
    except Exception as e:
        logging.warning("Could not load local session file: %s", e)
    return _empty_state()


@st.cache_resource
def _box() -> dict:
    """Singleton dict shared across ALL browser sessions."""
    return {"s": _load_state()}


def _get() -> dict:
    return _box()["s"]


def _put(state: dict) -> None:
    """Update shared state, persist to GitHub and local file."""
    with _file_lock:
        _box()["s"] = state
        # GitHub (primary — survives restarts / redeploys)
        _github_save(state)
        # Local file (fast fallback)
        try:
            d = _data_dir()
            d.mkdir(parents=True, exist_ok=True)
            tmp = _data_file().with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            tmp.replace(_data_file())
        except Exception as e:
            logging.warning("Could not persist state locally: %s", e)


# ===========================================================================
# Page config + mobile CSS
# ===========================================================================

st.set_page_config(
    page_title="Sports Leaderboard",
    page_icon="🏸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* ── Layout ─────────────────────────────────────────── */
    .main .block-container {
        padding: 0 0.75rem 5rem 0.75rem;
        max-width: 520px;
    }

    /* ── Page header banner ──────────────────────────────── */
    .page-header {
        background: linear-gradient(135deg, #1565C0 0%, #0288D1 100%);
        padding: 1rem 1.15rem 0.9rem;
        border-radius: 14px;
        margin-bottom: 1.1rem;
        color: white;
    }
    .page-header h1 {
        margin: 0;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.2;
        color: white;
    }
    .page-header p {
        margin: 0.2rem 0 0;
        font-size: 0.82rem;
        opacity: 0.88;
        color: white;
    }

    /* ── Section label ───────────────────────────────────── */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin: 1.1rem 0 0.35rem;
    }

    /* ── Team cards (court page) ─────────────────────────── */
    .team-card-a {
        background: #0D2137;
        border-left: 4px solid #29B6F6;
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #E8EAF0;
    }
    .team-card-b {
        background: #2A0D12;
        border-left: 4px solid #EF5350;
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #E8EAF0;
    }

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {
        min-height: 48px;
        font-size: 1rem;
        border-radius: 10px;
        width: 100%;
        font-weight: 500;
        transition: opacity 0.15s;
    }
    .stButton > button:active { opacity: 0.82; }

    /* ── Number / text inputs ────────────────────────────── */
    .stNumberInput input {
        font-size: 1.35rem !important;
        height: 52px !important;
        text-align: center !important;
    }
    .stNumberInput [data-testid="stNumberInputStepDown"],
    .stNumberInput [data-testid="stNumberInputStepUp"] {
        width: 40px; height: 52px;
    }
    .stTextInput input  { font-size: 1rem !important; height: 44px !important; }
    .stSelectbox > div > div { font-size: 1rem !important; min-height: 44px; }

    /* ── Win buttons (court page) ───────────────────────── */
    [data-testid="stHorizontalBlock"]:has(.team-card-a) + div .stButton > button,
    [data-testid="stHorizontalBlock"]:has(.team-card-b) + div .stButton > button {
        min-height: 44px !important;
        font-size: 0.88rem !important;
        padding: 0.25rem 0.3rem !important;
    }


    /* ── Roster radio buttons ────────────────────────────── */
    div[data-testid="stRadio"] > div {
        gap: 0.4rem;
        padding-top: 0.45rem;
    }
    div[data-testid="stRadio"] label {
        font-size: 0.82rem !important;
        padding: 0.2rem 0.1rem;
    }

    /* ── Expanders ───────────────────────────────────────── */
    details { border-radius: 10px !important; }
    details summary {
        font-size: 1rem;
        padding: 0.65rem 0;
        line-height: 1.4;
        font-weight: 500;
    }

    /* ── Metrics ─────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #1A1F2E;
        border-radius: 10px;
        padding: 0.5rem 0.4rem;
        text-align: center;
    }
    [data-testid="stMetricDelta"] { font-size: 0.75rem; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }

    /* ── Progress bar ────────────────────────────────────── */
    [data-testid="stProgressBar"] { height: 8px; border-radius: 5px; }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #1565C0, #0288D1) !important;
    }

    /* ── DataFrame ───────────────────────────────────────── */
    [data-testid="stDataFrame"] { font-size: 0.85rem; border-radius: 8px; overflow: hidden; }

    /* ── Tabs ────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.4rem 0.9rem;
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* ── Alert / info boxes ──────────────────────────────── */
    [data-testid="stAlert"] { border-radius: 10px; }

    /* ── Hide Streamlit chrome ───────────────────────────── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { height: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Session-state init  (per-browser UI state, not shared)
# ===========================================================================

def _init_ui():
    defaults = {
        "page": "setup",
        "num_players": 10,
        "num_courts": 2,
        "games_per_hour": 5,
        "player_names": [f"Player {i + 1}" for i in range(10)],
        "skill_visible": False,
        "show_skill_pw": False,
        "show_gen_pw": False,
        "show_reset_pw": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_ui()


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.markdown("## 🏸 Sports Leaderboard")
    st.divider()

    _s = _get()
    _num_courts_state = _s.get("num_courts", st.session_state.get("num_courts", 2))

    _sb_nav = [("setup", "⚙️  Setup & Schedule")]
    for _c in range(1, _num_courts_state + 1):
        _sb_nav.append((f"court{_c}", f"🏟️  Court {_c}"))
    _sb_nav.append(("leaderboard", "🏆  Leaderboard"))

    for pid, label in _sb_nav:
        if st.button(
            label,
            key=f"sb_{pid}",
            type="primary" if st.session_state.page == pid else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = pid
            st.rerun()

    st.divider()
    if _s.get("session_active"):
        st.success(f"✅ {len(_s['players'])} players · {_num_courts_state} courts")
        total = len(_s["scores"])
        done  = sum(1 for v in _s["scores"].values() if v.get("submitted"))
        if total:
            st.progress(done / total, text=f"{done}/{total} games done")
    else:
        st.info("No active session")

    if not _gemini_key():
        st.warning("⚠️ Gemini key missing.\nAdd it to config.yaml.")


# ===========================================================================
# Bottom nav bar  (same 4 buttons on every page)
# ===========================================================================

def _nav(active: str) -> None:
    st.markdown("---")
    num_courts = _get().get("num_courts", st.session_state.get("num_courts", 2))
    nav = [("setup", "⚙️", "Setup")]
    for c in range(1, num_courts + 1):
        nav.append((f"court{c}", "🏟", f"Crt {c}"))
    nav.append(("leaderboard", "🏆", "Board"))

    cols = st.columns(len(nav))
    for col, (pid, icon, label) in zip(cols, nav):
        with col:
            if st.button(
                f"{icon}\n{label}",
                key=f"bn_{pid}",
                type="primary" if active == pid else "secondary",
                use_container_width=True,
            ):
                st.session_state.page = pid
                st.rerun()


# ===========================================================================
# Page: Setup & Schedule
# ===========================================================================

def show_setup() -> None:
    st.markdown(
        '<div class="page-header">'
        '<h1>⚙️ Setup</h1>'
        '<p>Configure players, courts &amp; generate your schedule</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_p, tab_s = st.tabs(["👥 Players & Courts", "⚙️ Settings & Generate"])

    # ── Players tab ──────────────────────────────────────────────────────────
    with tab_p:
        # Number of players
        num_players = st.number_input(
            "Number of players", min_value=4, max_value=20,
            value=st.session_state.num_players, step=1,
        )
        if num_players != st.session_state.num_players:
            st.session_state.num_players = num_players
            cur = st.session_state.player_names
            if num_players > len(cur):
                cur += [f"Player {i + 1}" for i in range(len(cur), num_players)]
            else:
                st.session_state.player_names = cur[:num_players]
            st.rerun()

        # Number of courts
        num_courts = st.number_input(
            "Number of courts", min_value=1, max_value=6,
            step=1, key="num_courts",
        )

        # Single rate slider
        games_per_hour = st.slider(
            "Games per hour",
            min_value=1, max_value=12, step=1,
            key="games_per_hour",
        )
        mins_per_game = round(60 / games_per_hour, 1)
        st.caption(f"~{mins_per_game} min per game")

        # Per-court hours-booked sliders
        st.markdown('<div class="section-label">Hours booked per court</div>', unsafe_allow_html=True)
        num_games_per_court: Dict[int, int] = {}
        for c in range(1, num_courts + 1):
            hrs_key = f"court_hours_{c}"
            if hrs_key not in st.session_state:
                st.session_state[hrs_key] = 2.0
            court_hrs = st.slider(
                f"Court {c}",
                min_value=0.5, max_value=6.0, step=0.5,
                key=hrs_key,
            )
            num_games_per_court[c] = max(1, round(games_per_hour * court_hrs))
            st.caption(f"→ {num_games_per_court[c]} games")

        # Session summary info box
        n           = st.session_state.num_players
        total_games = sum(num_games_per_court.values())
        detail      = "  \n".join(
            f"Court {c}: {g} games ({st.session_state.get(f'court_hours_{c}', 2.0):.1f}h)"
            for c, g in num_games_per_court.items()
        )
        avg_games   = round(total_games * 4 / max(n, 1), 1)  # 4 players active per game
        st.info(
            f"**{n} players · {num_courts} courts · doubles**  \n"
            f"{detail}  \n"
            f"Total: **{total_games} games** · each player plays ~**{avg_games}**"
        )

        st.markdown('<div class="section-label">Player roster</div>', unsafe_allow_html=True)

        # ── Player rows ───────────────────────────────────────────────────────
        for i in range(st.session_state.num_players):
            # Ensure skill default is set even when radio is hidden
            if f"skill_{i}" not in st.session_state:
                st.session_state[f"skill_{i}"] = "intermediate"

            default = (
                st.session_state.player_names[i]
                if i < len(st.session_state.player_names)
                else f"Player {i + 1}"
            )
            if st.session_state.skill_visible:
                c_name, c_skill = st.columns([5, 4])
                with c_name:
                    st.text_input(
                        f"P{i + 1}", value=default,
                        key=f"pname_{i}", placeholder=f"Player {i + 1}",
                        label_visibility="collapsed",
                    )
                with c_skill:
                    st.radio(
                        "Level",
                        options=["intermediate", "beginner"],
                        key=f"skill_{i}",
                        horizontal=True,
                        label_visibility="collapsed",
                    )
            else:
                st.text_input(
                    f"P{i + 1}", value=default,
                    key=f"pname_{i}", placeholder=f"Player {i + 1}",
                    label_visibility="collapsed",
                )

        # ── Discreet lock toggle (below player list) ──────────────────────────
        st.markdown(
            """
            <style>
            .skill-toggle-anchor + div,
            .skill-toggle-anchor + div > div,
            .skill-toggle-anchor + div > div > div {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
            }
            .skill-toggle-anchor + div button,
            .skill-toggle-anchor + div button:hover,
            .skill-toggle-anchor + div button:focus,
            .skill-toggle-anchor + div button:active {
                min-height: 22px !important;
                height: 22px !important;
                width: auto !important;
                padding: 0 0.4rem !important;
                font-size: 0.6rem !important;
                background: transparent !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
                color: #2e2e3e !important;
                letter-spacing: 0.2em;
            }
            .skill-toggle-anchor + div button:hover {
                color: #555 !important;
            }
            </style>
            <div class="skill-toggle-anchor"></div>
            """,
            unsafe_allow_html=True,
        )
        _vis = st.session_state.skill_visible
        if st.button("· · ·", key="btn_skill_vis", use_container_width=False):
            if _vis:
                st.session_state.skill_visible = False
                st.session_state.show_skill_pw = False
            else:
                st.session_state.show_skill_pw = not st.session_state.show_skill_pw
            st.rerun()

        # ── Password prompt ───────────────────────────────────────────────────
        if st.session_state.show_skill_pw and not st.session_state.skill_visible:
            pw_col, go_col = st.columns([5, 2])
            with pw_col:
                pw_val = st.text_input(
                    "pw", type="password",
                    placeholder="Enter password…",
                    label_visibility="collapsed",
                    key="skill_pw_field",
                )
            with go_col:
                if st.button("Unlock", key="btn_skill_unlock",
                             type="primary", use_container_width=True):
                    if pw_val == "kaushik28":
                        st.session_state.skill_visible = True
                        st.session_state.show_skill_pw = False
                        if "skill_pw_field" in st.session_state:
                            del st.session_state["skill_pw_field"]
                    else:
                        st.error("Incorrect password.")
                    st.rerun()

    # ── Settings & Generate tab ───────────────────────────────────────────────
    with tab_s:
        # Read values set in Players tab
        num_courts     = st.session_state.get("num_courts", 2)
        games_per_hour = st.session_state.get("games_per_hour", 5)
        # Rebuild per-court game counts from stored hour sliders
        _ngpc: Dict[int, int] = {}
        for _c in range(1, num_courts + 1):
            _hrs = float(st.session_state.get(f"court_hours_{_c}", 2.0))
            _ngpc[_c] = max(1, round(games_per_hour * _hrs))
        num_games = max(_ngpc.values())  # generate enough rounds for the busiest court

        has_key = bool(_gemini_key())
        use_agent = st.checkbox(
            "Use AI agent (Gemini Flash)",
            value=has_key,
            disabled=not has_key,
            help="Set gemini.api_key in config.yaml to enable." if not has_key
                 else "Uses Gemini Flash ReAct agent for scheduling.",
        )

        st.markdown(" ")

        # ── Generate Schedule ─────────────────────────────────────────────────
        if st.button("🎲 Generate Schedule", type="primary", use_container_width=True):
            st.session_state.show_gen_pw   = True
            st.session_state.show_reset_pw = False
            st.rerun()

        if st.session_state.show_gen_pw:
            st.caption("Enter password to generate schedule")
            pw_col, go_col = st.columns([5, 2])
            with pw_col:
                gen_pw = st.text_input(
                    "gen_pw", type="password", placeholder="Password…",
                    label_visibility="collapsed", key="gen_pw_field",
                )
            with go_col:
                if st.button("Confirm", key="btn_gen_confirm",
                             type="primary", use_container_width=True):
                    if gen_pw == "kaushik28":
                        st.session_state.show_gen_pw = False
                        # ── Collect names & skills ──────────────────────────
                        raw_names = [
                            (st.session_state.get(f"pname_{i}") or f"Player {i + 1}").strip()
                            for i in range(st.session_state.num_players)
                        ]
                        raw_skills = [
                            st.session_state.get(f"skill_{i}", "intermediate")
                            for i in range(st.session_state.num_players)
                        ]
                        seen: Dict[str, int] = {}
                        players: List[str] = []
                        skill_levels: Dict[str, str] = {}
                        for nm, sk in zip(raw_names, raw_skills):
                            nm = nm or "Player"
                            if nm in seen:
                                seen[nm] += 1
                                nm = f"{nm} ({seen[nm]})"
                            else:
                                seen[nm] = 1
                            players.append(nm)
                            skill_levels[nm] = sk
                        st.session_state.player_names = players

                        with st.spinner("Generating schedule… 🤖"):
                            try:
                                if use_agent and has_key:
                                    from agent.react_agent import GamePlannerAgent
                                    raw_schedule = GamePlannerAgent().generate_schedule(
                                        players, skill_levels,
                                        num_rounds=num_games, num_courts=num_courts,
                                    )
                                    method = "AI agent (Gemini Flash)"
                                else:
                                    from services.schedule_service import ScheduleService
                                    raw_schedule = ScheduleService().generate_schedule(
                                        players, skill_levels,
                                        num_rounds=num_games, num_courts=num_courts,
                                    )
                                    method = "algorithm"
                            except Exception as exc:
                                logging.error("Schedule generation failed: %s", exc)
                                from services.schedule_service import ScheduleService
                                raw_schedule = ScheduleService().generate_schedule(
                                    players, skill_levels,
                                    num_rounds=num_games, num_courts=num_courts,
                                )
                                method = "algorithm (fallback)"

                            court_seen: Dict[int, int] = {c: 0 for c in range(1, num_courts + 1)}
                            schedule = []
                            for g in raw_schedule:
                                c = g["court"]
                                limit = _ngpc.get(c, num_games)
                                if court_seen[c] < limit:
                                    schedule.append(g)
                                    court_seen[c] += 1

                        new_state = {
                            "players":      players,
                            "skill_levels": skill_levels,
                            "num_courts":   num_courts,
                            "schedule":     schedule,
                            "scores": {
                                g["game_id"]: {"score_a": None, "score_b": None, "submitted": False}
                                for g in schedule
                            },
                            "session_active": True,
                        }
                        _put(new_state)
                        st.success(f"✅ {len(schedule)} games generated via {method}")
                        st.rerun()
                    else:
                        st.error("Incorrect password.")

        # ── Reset Session ─────────────────────────────────────────────────────
        st.markdown(" ")
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.show_reset_pw = True
            st.session_state.show_gen_pw   = False
            st.rerun()

        if st.session_state.show_reset_pw:
            st.caption("Enter password to reset session")
            rp_col, rgo_col = st.columns([5, 2])
            with rp_col:
                reset_pw = st.text_input(
                    "reset_pw", type="password", placeholder="Password…",
                    label_visibility="collapsed", key="reset_pw_field",
                )
            with rgo_col:
                if st.button("Confirm", key="btn_reset_confirm",
                             type="primary", use_container_width=True):
                    if reset_pw == "kaushik28":
                        st.session_state.show_reset_pw = False
                        # Clear all per-game widget state (scores, winner selections)
                        for k in list(st.session_state.keys()):
                            if k.startswith(("sa_", "sb_", "winner_", "win_a_", "win_b_")):
                                del st.session_state[k]
                        with st.spinner("Resetting…"):
                            _put(_empty_state())
                        st.rerun()
                    else:
                        st.error("Incorrect password.")

        # ── Schedule preview ──────────────────────────────────────────────────
        state = _get()
        if not state.get("schedule"):
            st.info("No schedule yet — click **Generate Schedule** above.")
        else:
            schedule = state["schedule"]
            scores   = state["scores"]

            st.divider()
            st.subheader(f"📋 Schedule  ·  {len(schedule)} games")

            st.download_button(
                "⬇️ Download Schedule (Word)",
                data=_build_docx(schedule),
                file_name="game_schedule.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

            num_courts_now = state.get("num_courts", 2)
            sched_tab_labels = [f"🏟 Court {c}" for c in range(1, num_courts_now + 1)] + ["📄 All"]
            sched_tabs = st.tabs(sched_tab_labels)
            for c, tab in enumerate(sched_tabs[:-1], start=1):
                with tab:
                    _render_table([g for g in schedule if g["court"] == c], scores)
            with sched_tabs[-1]:
                _render_table(schedule, scores)

    _nav("setup")


def _render_table(games: List[dict], scores: dict) -> None:
    if not games:
        st.info("No games.")
        return
    rows = []
    for game_num, g in enumerate(games, start=1):
        sd   = scores.get(g["game_id"], {})
        done = sd.get("submitted", False)
        rows.append({
            "": "✅" if done else "⏳",
            "Game": game_num,
            "Team A": " & ".join(g["team_a"]),
            "Team B": " & ".join(g["team_b"]),
            "Score": f"{sd['score_a']}–{sd['score_b']}" if done else "—",
            "Rest": ", ".join(g.get("sitting_out", [])),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_docx(schedule: List[dict]) -> bytes:
    from docx import Document
    from docx.shared import Cm, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import io

    doc = Document()

    # Narrow margins
    sec = doc.sections[0]
    sec.top_margin    = Cm(1.5)
    sec.bottom_margin = Cm(1.5)
    sec.left_margin   = Cm(2.0)
    sec.right_margin  = Cm(2.0)

    title = doc.add_heading("Game Schedule", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 4-column table: Game | Court | Team A | Team B
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"

    # Header row
    for cell, text in zip(table.rows[0].cells, ["Game", "Court", "Team A", "Team B"]):
        cell.text = text
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # One game = 2 rows (one per player slot), Game & Court cells merged vertically
    for game_num, game in enumerate(schedule, start=1):
        ta = game.get("team_a", [])
        tb = game.get("team_b", [])

        r1 = table.add_row().cells
        r1[0].text = f"Game {game_num}"
        r1[1].text = f"Court {game['court']}"
        r1[2].text = ta[0] if ta else ""
        r1[3].text = tb[0] if tb else ""

        r2 = table.add_row().cells
        r2[2].text = ta[1] if len(ta) > 1 else ""
        r2[3].text = tb[1] if len(tb) > 1 else ""

        # Merge Game and Court cells across the 2 rows
        r1[0].merge(r2[0])
        r1[1].merge(r2[1])

        # Centre-align merged cells
        for cell in (r1[0], r1[1]):
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Font size for data rows
        for row in (r1, r2):
            for cell in row:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(10)

    # Column widths
    col_widths = [Cm(2.5), Cm(2.5), Cm(5.5), Cm(5.5)]
    for row in table.rows:
        for cell, w in zip(row.cells, col_widths):
            cell.width = w

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# Page: Court  (mobile-first score entry)
# ===========================================================================

def show_court(court: int) -> None:
    st.markdown(
        f'<div class="page-header">'
        f'<h1>🏟 Court {court}</h1>'
        f'<p>Enter scores as each game finishes</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    state = _get()
    if not state.get("schedule"):
        st.warning("No schedule yet — go to **Setup** first.")
        _nav(f"court{court}")
        return

    court_games = [g for g in state["schedule"] if g["court"] == court]
    scores      = state["scores"]

    if not court_games:
        st.info(f"No games for Court {court}.")
        _nav(f"court{court}")
        return

    done  = sum(1 for g in court_games if scores.get(g["game_id"], {}).get("submitted"))
    total = len(court_games)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total",  total)
    c2.metric("Done",   done)
    c3.metric("Left",   total - done)
    st.progress(done / total if total else 0)
    st.divider()

    for game_num, game in enumerate(court_games, start=1):
        gid = game["game_id"]
        sd  = scores.get(gid, {})
        submitted = sd.get("submitted", False)

        # ── Pre-initialise session state for scores & winner ──────────────
        if f"sa_{gid}" not in st.session_state:
            st.session_state[f"sa_{gid}"] = int(sd["score_a"]) if submitted and sd.get("score_a") is not None else 0
        if f"sb_{gid}" not in st.session_state:
            st.session_state[f"sb_{gid}"] = int(sd["score_b"]) if submitted and sd.get("score_b") is not None else 0
        if f"winner_{gid}" not in st.session_state:
            if submitted:
                _sa, _sb = sd.get("score_a") or 0, sd.get("score_b") or 0
                st.session_state[f"winner_{gid}"] = (
                    "Team A" if _sa > _sb else "Team B" if _sb > _sa else "—"
                )
            else:
                st.session_state[f"winner_{gid}"] = "—"

        winner = st.session_state.get(f"winner_{gid}", "—")

        icon = "✅" if submitted else "⏳"
        with st.expander(f"{icon}  Game {game_num}", expanded=not submitted):
            # Team A row: card + Win button
            col_card_a, col_btn_a = st.columns([5, 2])
            with col_card_a:
                st.markdown(
                    f'<div class="team-card-a">'
                    f'<strong>Team A</strong> &nbsp; {" &amp; ".join(game["team_a"])}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_btn_a:
                if st.button(
                    "🏆 Win" if winner != "Team A" else "✅ Won",
                    key=f"win_a_{gid}",
                    type="primary" if winner == "Team A" else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[f"winner_{gid}"] = "Team A"
                    st.session_state[f"sa_{gid}"] = max(11, st.session_state.get(f"sa_{gid}", 0))
                    st.rerun()

            # Team B row: card + Win button
            col_card_b, col_btn_b = st.columns([5, 2])
            with col_card_b:
                st.markdown(
                    f'<div class="team-card-b">'
                    f'<strong>Team B</strong> &nbsp; {" &amp; ".join(game["team_b"])}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_btn_b:
                if st.button(
                    "🏆 Win" if winner != "Team B" else "✅ Won",
                    key=f"win_b_{gid}",
                    type="primary" if winner == "Team B" else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[f"winner_{gid}"] = "Team B"
                    st.session_state[f"sb_{gid}"] = max(11, st.session_state.get(f"sb_{gid}", 0))
                    st.rerun()

            # Score inputs — winner defaults to 11, max 30 for tie-breaks
            col_a, col_b = st.columns(2)
            with col_a:
                score_a = st.number_input(
                    "Team A score", min_value=0, max_value=30,
                    key=f"sa_{gid}",
                )
            with col_b:
                score_b = st.number_input(
                    "Team B score", min_value=0, max_value=30,
                    key=f"sb_{gid}",
                )

            btn = "✏️ Update Score" if submitted else "✅ Submit Score"
            if st.button(btn, key=f"btn_{gid}", type="primary", use_container_width=True):
                new_state = copy.deepcopy(_get())
                new_state["scores"][gid] = {
                    "score_a": score_a, "score_b": score_b, "submitted": True
                }
                with st.spinner("Saving…"):
                    _put(new_state)
                st.toast("Score saved!")
                st.rerun()

            # Result banner
            if submitted:
                sa, sb = sd["score_a"], sd["score_b"]
                if sa > sb:
                    st.success(f"🏆 Team A wins!  {sa} – {sb}")
                elif sb > sa:
                    st.success(f"🏆 Team B wins!  {sb} – {sa}")
                else:
                    st.success(f"🤝 Draw!  {sa} – {sb}")

            resting = game.get("sitting_out", [])
            if resting:
                st.caption(f"Resting: {', '.join(resting)}")

    _nav(f"court{court}")


# ===========================================================================
# Page: Leaderboard
# ===========================================================================

def show_leaderboard() -> None:
    st.markdown(
        '<div class="page-header">'
        '<h1>🏆 Leaderboard</h1>'
        '<p>Live standings · doubles tournament</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    state = _get()

    if not state.get("schedule"):
        st.info("No schedule yet — go to **Setup** first.")
        _nav("leaderboard")
        return

    from services.leaderboard_service import LeaderboardService
    lb    = LeaderboardService().calculate_leaderboard(state["schedule"], state["scores"])
    done  = sum(1 for v in state["scores"].values() if v.get("submitted"))
    total = len(state["scores"])

    if total:
        st.progress(done / total, text=f"{done} / {total} games  ({int(done/total*100)}%)")

    if not lb:
        st.info("No scores yet — enter them on the Court pages.")
        _nav("leaderboard")
        return

    # ── Podium ───────────────────────────────────────────────────────────────
    podium = min(len(lb), 3)
    _medal_bg     = ["#1E1A0A", "#161616", "#1A0E0A"]
    _medal_border = ["#F9A825", "#9E9E9E", "#EF5350"]
    _medals       = ["🥇", "🥈", "🥉"]
    cols = st.columns(podium)
    for col, medal, p, bg, border in zip(cols, _medals, lb[:podium], _medal_bg, _medal_border):
        with col:
            st.markdown(
                f'<div style="background:{bg};border-top:4px solid {border};'
                f'border-radius:12px;padding:0.75rem 0.4rem;text-align:center;">'
                f'<div style="font-size:1.8rem;line-height:1">{medal}</div>'
                f'<div style="font-weight:700;font-size:0.88rem;margin-top:0.35rem;color:#E8EAF0;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{p["name"]}</div>'
                f'<div style="font-size:1.25rem;font-weight:800;color:{border}">'
                f'{p["points_gained"]}</div>'
                f'<div style="font-size:0.72rem;color:#666">{p["games_won"]}W '
                f'/ {p["games_lost"]}L</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Table ────────────────────────────────────────────────────────────────
    rows = [
        {
            "#":    p["rank"],
            "Player": p["name"],
            "W":    p["games_won"],
            "L":    p["games_lost"],
            "For":  p["points_gained"],
            "Agst": p["points_conceded"],
            "Net":  p["net_points"],
        }
        for p in lb
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={"Net": st.column_config.NumberColumn("Net", format="%+d")},
    )

    # ── Bar chart ────────────────────────────────────────────────────────────
    st.divider()
    st.caption("Points scored per player")
    st.bar_chart(
        pd.DataFrame({"Player": [p["name"] for p in lb], "Pts": [p["points_gained"] for p in lb]})
        .set_index("Player"),
        height=250,
    )

    _nav("leaderboard")


# ===========================================================================
# Router
# ===========================================================================

_page = st.session_state.page

if _page == "setup":
    show_setup()
elif _page == "leaderboard":
    show_leaderboard()
elif _page.startswith("court"):
    try:
        show_court(int(_page[5:]))
    except ValueError:
        show_setup()
else:
    show_setup()
