"""
Sports Leaderboard — single-process Streamlit app.

No Flask backend. State is shared across all browser sessions via
@st.cache_resource (all phones hit the same Python process).
Schedule + scores are persisted to a JSON file so data survives restarts.

Gemini API key is read from config.yaml (gemini.api_key).
"""

import copy
import csv
import json
import logging
import os
import threading
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

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


def _load_from_disk() -> dict:
    try:
        f = _data_file()
        if f.exists():
            with open(f, encoding="utf-8") as fp:
                return json.load(fp)
    except Exception as e:
        logging.warning("Could not load session file: %s", e)
    return _empty_state()


@st.cache_resource
def _box() -> dict:
    """
    Singleton dict shared across ALL browser sessions.
    key 's' holds the current application state.
    """
    return {"s": _load_from_disk()}


def _get() -> dict:
    """Return the current shared state."""
    return _box()["s"]


def _put(state: dict) -> None:
    """Update shared state and persist to disk."""
    with _file_lock:
        _box()["s"] = state
        try:
            d = _data_dir()
            d.mkdir(parents=True, exist_ok=True)
            tmp = _data_file().with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            tmp.replace(_data_file())
        except Exception as e:
            logging.warning("Could not persist state: %s", e)


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
        background: #E3F2FD;
        border-left: 4px solid #1565C0;
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .team-card-b {
        background: #FCE4EC;
        border-left: 4px solid #C62828;
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        line-height: 1.5;
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
        background: #F0F7FF;
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
        "games_per_hour": 6,
        "player_names": [f"Player {i + 1}" for i in range(10)],
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
            "Games per hour (all courts)",
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
        for i in range(st.session_state.num_players):
            default = (
                st.session_state.player_names[i]
                if i < len(st.session_state.player_names)
                else f"Player {i + 1}"
            )
            c_name, c_skill = st.columns([3, 2])
            with c_name:
                st.text_input(
                    f"P{i + 1}", value=default,
                    key=f"pname_{i}", placeholder=f"Player {i + 1}",
                    label_visibility="collapsed",
                )
            with c_skill:
                st.selectbox(
                    "Level", options=["beginner", "intermediate"],
                    key=f"skill_{i}", label_visibility="collapsed",
                )

    # ── Settings & Generate tab ───────────────────────────────────────────────
    with tab_s:
        # Read values set in Players tab
        num_courts     = st.session_state.get("num_courts", 2)
        games_per_hour = st.session_state.get("games_per_hour", 6)
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

        if st.button("🎲 Generate Schedule", type="primary", use_container_width=True):
            # Collect names & skills from widget keys
            raw_names = [
                (st.session_state.get(f"pname_{i}") or f"Player {i + 1}").strip()
                for i in range(st.session_state.num_players)
            ]
            raw_skills = [
                st.session_state.get(f"skill_{i}", "beginner")
                for i in range(st.session_state.num_players)
            ]

            # De-duplicate names
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

                # Trim each court to its individual game limit
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

        st.markdown(" ")
        if st.button("🔄 Reset Session", use_container_width=True):
            _put(_empty_state())
            st.rerun()

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
                "⬇️ Download Schedule (CSV)",
                data=_build_csv(schedule),
                file_name="game_schedule.csv",
                mime="text/csv",
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


def _build_csv(schedule: List[dict]) -> str:
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["Game", "Court", "Time", "A-P1", "A-P2", "B-P1", "B-P2", "Resting"])
    for game_num, g in enumerate(schedule, start=1):
        ta, tb = g["team_a"], g["team_b"]
        w.writerow([
            game_num, g["court"], g["time_slot"],
            ta[0] if ta else "",       ta[1] if len(ta) > 1 else "",
            tb[0] if tb else "",       tb[1] if len(tb) > 1 else "",
            ", ".join(g.get("sitting_out", [])),
        ])
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

        icon = "✅" if submitted else "⏳"
        with st.expander(
            f"{icon}  Game {game_num}  ·  {game['time_slot']}",
            expanded=not submitted,
        ):
            # Teams — coloured cards
            st.markdown(
                f'<div class="team-card-a"><strong>Team A</strong> &nbsp; '
                f'{" &amp; ".join(game["team_a"])}</div>'
                f'<div class="team-card-b"><strong>Team B</strong> &nbsp; '
                f'{" &amp; ".join(game["team_b"])}</div>',
                unsafe_allow_html=True,
            )

            # Score inputs — two equal half-screen columns
            sa_val = int(sd["score_a"]) if submitted and sd.get("score_a") is not None else 0
            sb_val = int(sd["score_b"]) if submitted and sd.get("score_b") is not None else 0

            col_a, col_b = st.columns(2)
            with col_a:
                score_a = st.number_input(
                    "Team A", min_value=0, max_value=99,
                    value=sa_val, key=f"sa_{gid}",
                )
            with col_b:
                score_b = st.number_input(
                    "Team B", min_value=0, max_value=99,
                    value=sb_val, key=f"sb_{gid}",
                )

            btn = "✏️ Update Score" if submitted else "✅ Submit Score"
            if st.button(btn, key=f"btn_{gid}", type="primary", use_container_width=True):
                new_state = copy.deepcopy(_get())
                new_state["scores"][gid] = {
                    "score_a": score_a, "score_b": score_b, "submitted": True
                }
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
    _medal_bg     = ["#FFF8E1", "#F5F5F5", "#FBE9E7"]
    _medal_border = ["#F9A825", "#757575", "#BF360C"]
    _medals       = ["🥇", "🥈", "🥉"]
    cols = st.columns(podium)
    for col, medal, p, bg, border in zip(cols, _medals, lb[:podium], _medal_bg, _medal_border):
        with col:
            st.markdown(
                f'<div style="background:{bg};border-top:4px solid {border};'
                f'border-radius:12px;padding:0.75rem 0.4rem;text-align:center;">'
                f'<div style="font-size:1.8rem;line-height:1">{medal}</div>'
                f'<div style="font-weight:700;font-size:0.88rem;margin-top:0.35rem;'
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
