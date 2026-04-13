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
    .main .block-container {
        padding: 0.75rem 0.75rem 4.5rem 0.75rem;
        max-width: 520px;
    }
    .stButton > button {
        min-height: 48px;
        font-size: 1.05rem;
        border-radius: 10px;
        width: 100%;
    }
    .stNumberInput input {
        font-size: 1.4rem !important;
        height: 52px !important;
        text-align: center !important;
    }
    .stNumberInput [data-testid="stNumberInputStepDown"],
    .stNumberInput [data-testid="stNumberInputStepUp"] {
        width: 40px; height: 52px;
    }
    .stTextInput input  { font-size: 1rem !important; height: 44px !important; }
    .stSelectbox > div > div { font-size: 1rem !important; min-height: 44px; }
    details summary { font-size: 1rem; padding: 0.6rem 0; line-height: 1.4; }
    [data-testid="stMetricDelta"] { font-size: 0.78rem; }
    [data-testid="stProgressBar"]  { height: 10px; border-radius: 5px; }
    [data-testid="stDataFrame"]    { font-size: 0.85rem; }
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

    for pid, label in [
        ("setup",       "⚙️  Setup & Schedule"),
        ("court1",      "🏟️  Court 1"),
        ("court2",      "🏟️  Court 2"),
        ("leaderboard", "🏆  Leaderboard"),
    ]:
        if st.button(
            label,
            key=f"sb_{pid}",
            type="primary" if st.session_state.page == pid else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = pid
            st.rerun()

    st.divider()
    s = _get()
    if s.get("session_active"):
        st.success(f"✅ {len(s['players'])} players")
        total = len(s["scores"])
        done  = sum(1 for v in s["scores"].values() if v.get("submitted"))
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
    cols = st.columns(4)
    nav = [
        ("setup",       "⚙️", "Setup"),
        ("court1",      "🏟", "Court 1"),
        ("court2",      "🏟", "Court 2"),
        ("leaderboard", "🏆", "Board"),
    ]
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
    st.title("⚙️ Setup")

    tab_p, tab_s = st.tabs(["👥 Players", "⚙️ Settings & Generate"])

    # ── Players tab ──────────────────────────────────────────────────────────
    with tab_p:
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

        st.caption("Enter each player's name and skill level.")
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
        num_rounds = st.slider(
            "Rounds per court", min_value=6, max_value=24,
            value=_default_rounds(), step=1,
        )
        n = st.session_state.num_players
        total_min = num_rounds * 10
        avg = round(num_rounds * 8 / max(n, 1), 1)

        st.info(
            f"**{n} players · 2 courts · doubles**  \n"
            f"{num_rounds} rounds → ~{total_min} min  \n"
            f"Each player plays ~**{avg} games**"
        )

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
                        schedule = GamePlannerAgent().generate_schedule(
                            players, skill_levels, num_rounds=num_rounds
                        )
                        method = "AI agent (Gemini Flash)"
                    else:
                        from services.schedule_service import ScheduleService
                        schedule = ScheduleService().generate_schedule(
                            players, skill_levels, num_rounds=num_rounds
                        )
                        method = "algorithm"
                except Exception as exc:
                    logging.error("Schedule generation failed: %s", exc)
                    from services.schedule_service import ScheduleService
                    schedule = ScheduleService().generate_schedule(
                        players, skill_levels, num_rounds=num_rounds
                    )
                    method = "algorithm (fallback)"

            new_state = {
                "players": players,
                "skill_levels": skill_levels,
                "schedule": schedule,
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

    # ── Schedule preview ──────────────────────────────────────────────────────
    state = _get()
    if not state.get("schedule"):
        st.info("No schedule yet. Go to **Settings & Generate** tab.")
        _nav("setup")
        return

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

    t1, t2, tall = st.tabs(["🏟 Court 1", "🏟 Court 2", "📄 All"])
    with t1:   _render_table([g for g in schedule if g["court"] == 1], scores)
    with t2:   _render_table([g for g in schedule if g["court"] == 2], scores)
    with tall: _render_table(schedule, scores)

    _nav("setup")


def _render_table(games: List[dict], scores: dict) -> None:
    if not games:
        st.info("No games.")
        return
    rows = []
    for g in games:
        sd   = scores.get(g["game_id"], {})
        done = sd.get("submitted", False)
        rows.append({
            "": "✅" if done else "⏳",
            "Rd": g["round"],
            "Time": g["time_slot"],
            "Team A": " & ".join(g["team_a"]),
            "Team B": " & ".join(g["team_b"]),
            "Score": f"{sd['score_a']}–{sd['score_b']}" if done else "—",
            "Rest": ", ".join(g.get("sitting_out", [])),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_csv(schedule: List[dict]) -> str:
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["Round", "Court", "Time", "A-P1", "A-P2", "B-P1", "B-P2", "Resting"])
    for g in schedule:
        ta, tb = g["team_a"], g["team_b"]
        w.writerow([
            g["round"], g["court"], g["time_slot"],
            ta[0] if ta else "",       ta[1] if len(ta) > 1 else "",
            tb[0] if tb else "",       tb[1] if len(tb) > 1 else "",
            ", ".join(g.get("sitting_out", [])),
        ])
    return buf.getvalue()


# ===========================================================================
# Page: Court  (mobile-first score entry)
# ===========================================================================

def show_court(court: int) -> None:
    st.title(f"🏟 Court {court}")

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
            # Teams — stacked, easy to read on phone
            st.markdown(
                f"**Team A** &nbsp; {' & '.join(game['team_a'])}  \n"
                f"**Team B** &nbsp; {' & '.join(game['team_b'])}"
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
    st.title("🏆 Leaderboard")

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
    for col, medal, p in zip(st.columns(podium), ["🥇", "🥈", "🥉"], lb[:podium]):
        col.metric(
            f"{medal} {p['name']}",
            f"{p['points_gained']} pts",
            f"{p['games_won']}W / {p['games_lost']}L",
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

if   _page == "setup":       show_setup()
elif _page == "court1":      show_court(1)
elif _page == "court2":      show_court(2)
elif _page == "leaderboard": show_leaderboard()
