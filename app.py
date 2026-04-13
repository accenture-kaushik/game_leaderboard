"""
Flask backend — file-backed JSON storage (no database).

State is persisted to a JSON file so scores survive app restarts.
  - Local dev:  ./data/session.json   (set DATA_DIR=./data or leave default)
  - Azure:      /home/data/session.json  (set DATA_DIR=/home/data in App Settings)
"""

import json
import logging
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from services.schedule_service import ScheduleService
from services.leaderboard_service import LeaderboardService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Persistent file-based state
# ---------------------------------------------------------------------------
_DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
_DATA_FILE = _DATA_DIR / "session.json"
_lock = threading.Lock()          # prevents concurrent file writes


def _default_state() -> dict:
    return {
        "players": [],
        "skill_levels": {},
        "schedule": [],
        "scores": {},
        "session_active": False,
    }


def _load() -> dict:
    """Read state from disk.  Returns default if file missing or corrupt."""
    try:
        if _DATA_FILE.exists():
            with _lock:
                with open(_DATA_FILE, encoding="utf-8") as f:
                    return json.load(f)
    except Exception as exc:
        logger.warning("Could not read state file: %s", exc)
    return _default_state()


def _save(state: dict) -> None:
    """Write state to disk atomically."""
    with _lock:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _DATA_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp.replace(_DATA_FILE)          # atomic rename


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------
_schedule_svc = ScheduleService()
_leaderboard_svc = LeaderboardService()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/state", methods=["GET"])
def get_state():
    s = _load()
    completed = sum(1 for v in s["scores"].values() if v.get("submitted"))
    return jsonify(
        {
            "session_active": s["session_active"],
            "players": s["players"],
            "has_schedule": bool(s["schedule"]),
            "games_completed": completed,
            "total_games": len(s["scores"]),
        }
    )


@app.route("/api/setup", methods=["POST"])
def setup():
    data = request.get_json(force=True) or {}
    players = [p.strip() for p in data.get("players", []) if str(p).strip()]
    skill_levels = data.get("skill_levels", {})

    if len(players) < 4:
        return jsonify({"error": "Need at least 4 players."}), 400

    state = _default_state()
    state["players"] = players
    state["skill_levels"] = skill_levels
    state["session_active"] = True
    _save(state)

    logger.info("Session setup: %d players", len(players))
    return jsonify({"message": "Session initialised", "players": players})


@app.route("/api/generate-schedule", methods=["POST"])
def generate_schedule():
    state = _load()
    if not state["session_active"]:
        return jsonify({"error": "No active session. Call /api/setup first."}), 400

    data = request.get_json(force=True) or {}
    num_rounds = int(data.get("num_rounds", 12))
    use_agent = data.get("use_agent", True)

    try:
        if use_agent and os.getenv("GOOGLE_API_KEY"):
            from agent.react_agent import GamePlannerAgent
            schedule = GamePlannerAgent().generate_schedule(
                state["players"], state["skill_levels"], num_rounds=num_rounds
            )
            method = "agent"
        else:
            schedule = _schedule_svc.generate_schedule(
                state["players"], state["skill_levels"], num_rounds=num_rounds
            )
            method = "algorithm"
    except Exception as exc:
        logger.warning("Schedule generation error (%s) — falling back", exc)
        schedule = _schedule_svc.generate_schedule(
            state["players"], state["skill_levels"], num_rounds=num_rounds
        )
        method = "algorithm_fallback"

    state["schedule"] = schedule
    state["scores"] = {
        g["game_id"]: {"score_a": None, "score_b": None, "submitted": False}
        for g in schedule
    }
    _save(state)

    logger.info("Schedule saved via %s: %d games", method, len(schedule))
    return jsonify({"schedule": schedule, "total_games": len(schedule), "method": method})


@app.route("/api/schedule", methods=["GET"])
def get_schedule():
    s = _load()
    return jsonify({"schedule": s["schedule"], "scores": s["scores"]})


@app.route("/api/score/<game_id>", methods=["POST"])
def submit_score(game_id: str):
    state = _load()
    if game_id not in state["scores"]:
        return jsonify({"error": f"Game {game_id!r} not found."}), 404

    data = request.get_json(force=True) or {}
    try:
        score_a = int(data["score_a"])
        score_b = int(data["score_b"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "score_a and score_b must be integers."}), 400

    state["scores"][game_id] = {"score_a": score_a, "score_b": score_b, "submitted": True}
    _save(state)
    return jsonify({"message": "Score saved", "game_id": game_id})


@app.route("/api/leaderboard", methods=["GET"])
def get_leaderboard():
    state = _load()
    if not state["schedule"]:
        return jsonify({"leaderboard": [], "games_completed": 0, "total_games": 0})

    leaderboard = _leaderboard_svc.calculate_leaderboard(state["schedule"], state["scores"])
    completed = sum(1 for v in state["scores"].values() if v.get("submitted"))
    return jsonify(
        {"leaderboard": leaderboard, "games_completed": completed, "total_games": len(state["scores"])}
    )


@app.route("/api/reset", methods=["POST"])
def reset():
    _save(_default_state())
    return jsonify({"message": "Session reset"})


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
