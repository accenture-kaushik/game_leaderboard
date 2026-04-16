"""
ReAct (Reasoning + Acting) Game Planner Agent — Gemini Flash.
All Gemini settings (api_key, model_name, temperature, top_p,
max_output_tokens) are read from config.yaml under the [gemini] section.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert sports game scheduler using the ReAct pattern.

Available tools:
  1. analyze_skill_distribution
     Input: {"players": [...], "skill_levels": {...}}
     Returns: skill statistics and recommended strategy.

  2. select_sitout_players
     Input: {"players": [...], "play_count": {...}, "sitout_count": {...}, "n": <int>}
     Returns: list of n player names to sit out this round.

  3. create_balanced_teams
     Input: {"available": [...], "skill_levels": {...}, "partner_history": {...}}
     Returns: list of [player, player] pairs.

  4. validate_schedule
     Input: {"schedule": [...], "players": [...], "skill_levels": {...}}
     Returns: fairness metrics.

Always use this format:
Thought: <reasoning>
Action: <tool_name>
Action Input: <JSON>
Observation: <filled by system>
...
Final Answer: <conclusion>
"""


# ---------------------------------------------------------------------------
# Config helper — reads everything from config.yaml [gemini] section
# ---------------------------------------------------------------------------

def _load_gemini_config() -> dict:
    """
    Return the [gemini] config block, checking in priority order:
      1. st.secrets  — Streamlit Community Cloud
      2. config.yaml — local development
    """
    # ── Streamlit Community Cloud ──────────────────────────────────────────
    try:
        import streamlit as st
        if "gemini" in st.secrets:
            return dict(st.secrets["gemini"])
    except Exception:
        pass

    # ── Local: config.yaml ─────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("gemini", {})
        except Exception as exc:
            logger.warning("Could not read config.yaml: %s", exc)
    return {}


def _load_api_key() -> str:
    key = _load_gemini_config().get("api_key", "")
    if key and not key.startswith("YOUR_"):
        return key
    return os.getenv("GOOGLE_API_KEY", "")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GamePlannerAgent:
    """ReAct agent using Gemini Flash for intelligent game scheduling."""

    def __init__(self) -> None:
        api_key = _load_api_key()
        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set gemini.api_key in config.yaml."
            )

        gcfg = _load_gemini_config()
        genai.configure(api_key=api_key)

        generation_config = genai.GenerationConfig(
            temperature=float(gcfg.get("temperature", 0.7)),
            top_p=float(gcfg.get("top_p", 0.95)),
            max_output_tokens=int(gcfg.get("max_output_tokens", 4096)),
        )

        self.model = genai.GenerativeModel(
            model_name=gcfg.get("model_name", "gemini-2.0-flash"),
            system_instruction=SYSTEM_PROMPT,
            generation_config=generation_config,
        )
        logger.info(
            "Gemini model ready: %s | temp=%.1f | top_p=%.2f | max_tokens=%d",
            gcfg.get("model_name", "gemini-2.0-flash"),
            generation_config.temperature,
            generation_config.top_p,
            generation_config.max_output_tokens,
        )

    # =========================================================================
    # Public
    # =========================================================================

    def generate_schedule(
        self,
        players: List[str],
        skill_levels: Dict[str, str],
        num_rounds: int = 12,
        num_courts: int = 2,
        special_instructions: str = "",
    ) -> List[Dict]:
        logger.info("GamePlannerAgent: analysis phase")
        self._run_analysis_loop(players, skill_levels, special_instructions)

        logger.info("GamePlannerAgent: building schedule")
        return self._build_schedule(players, skill_levels, num_rounds, num_courts)

    # =========================================================================
    # ReAct analysis loop
    # =========================================================================

    def _run_analysis_loop(
        self,
        players: List[str],
        skill_levels: Dict[str, str],
        special_instructions: str = "",
    ) -> None:
        instructions_block = (
            f"\nSpecial instructions from the organiser (must be respected):\n{special_instructions.strip()}\n"
            if special_instructions and special_instructions.strip()
            else ""
        )
        prompt = (
            f"Plan a fair doubles game schedule.\n"
            f"Players: {players}\n"
            f"Skill levels: {json.dumps(skill_levels)}\n"
            f"{instructions_block}\n"
            "Analyse skill distribution, then describe the team strategy."
        )
        history = [prompt]

        for _ in range(6):
            try:
                response = self.model.generate_content("\n".join(history))
                text = response.text.strip()
                parsed = self._parse_react(text)

                if parsed.get("final_answer"):
                    logger.info("Agent analysis: %s", parsed["final_answer"][:200])
                    return

                action = parsed.get("action")
                if action:
                    ctx = {
                        "players": players, "skill_levels": skill_levels,
                        "play_count": {}, "sitout_count": {}, "partner_history": {},
                    }
                    obs = self._dispatch(action, parsed.get("action_input") or {}, ctx)
                    history.append(text)
                    history.append(f"Observation: {obs}")
                else:
                    break
            except Exception as exc:
                logger.warning("Agent iteration error: %s", exc)
                break

    # =========================================================================
    # Schedule builder (uses agent tools directly)
    # =========================================================================

    def _build_schedule(
        self,
        players: List[str],
        skill_levels: Dict[str, str],
        num_rounds: int,
        num_courts: int,
    ) -> List[Dict]:
        sitouts_per_round = max(0, len(players) - num_courts * 4)
        play_count: Dict[str, int] = {p: 0 for p in players}
        sitout_count: Dict[str, int] = {p: 0 for p in players}
        partner_history: Dict[str, Dict[str, int]] = {p: {} for p in players}
        schedule: List[Dict] = []

        for round_num in range(1, num_rounds + 1):
            sitting_out = self._tool_select_sitouts(
                players, play_count, sitout_count, sitouts_per_round
            )
            available = [p for p in players if p not in sitting_out]
            pairs = self._tool_create_teams(available, skill_levels, partner_history)

            for court_idx in range(num_courts):
                a_idx, b_idx = court_idx * 2, court_idx * 2 + 1
                if b_idx >= len(pairs):
                    break
                team_a, team_b = pairs[a_idx], pairs[b_idx]

                schedule.append({
                    "round": round_num,
                    "court": court_idx + 1,
                    "team_a": team_a,
                    "team_b": team_b,
                    "sitting_out": list(sitting_out),
                    "time_slot": f"{(round_num - 1) * 10}–{round_num * 10} min",
                    "game_id": f"r{round_num}_c{court_idx + 1}",
                })

                for team in [team_a, team_b]:
                    if len(team) >= 2:
                        p1, p2 = team[0], team[1]
                        partner_history.setdefault(p1, {})[p2] = partner_history[p1].get(p2, 0) + 1
                        partner_history.setdefault(p2, {})[p1] = partner_history[p2].get(p1, 0) + 1

            for p in available:
                play_count[p] += 1
            for p in sitting_out:
                sitout_count[p] += 1

        logger.info("Agent built %d games", len(schedule))
        return schedule

    # =========================================================================
    # Tools
    # =========================================================================

    def _tool_select_sitouts(self, players, play_count, sitout_count, n):
        if n <= 0:
            return []
        return sorted(players, key=lambda p: (-play_count.get(p, 0), sitout_count.get(p, 0)))[:n]

    def _tool_create_teams(self, available, skill_levels, partner_history):
        beginners = [p for p in available if skill_levels.get(p) == "beginner"]
        intermediates = [p for p in available if skill_levels.get(p) == "intermediate"]
        random.shuffle(beginners)
        random.shuffle(intermediates)

        pairs: List[List[str]] = []
        b_pool, i_pool = list(beginners), list(intermediates)

        while b_pool and i_pool:
            b = b_pool.pop(0)
            best_i = min(i_pool, key=lambda i: partner_history.get(b, {}).get(i, 0))
            i_pool.remove(best_i)
            pairs.append([b, best_i])

        remaining = b_pool + i_pool
        while len(remaining) >= 2:
            p1 = remaining.pop(0)
            best = min(remaining, key=lambda p: partner_history.get(p1, {}).get(p, 0))
            remaining.remove(best)
            pairs.append([p1, best])

        return pairs

    def _tool_analyze_skills(self, players, skill_levels):
        b = [p for p in players if skill_levels.get(p) == "beginner"]
        i = [p for p in players if skill_levels.get(p) == "intermediate"]
        return {"beginners": b, "intermediates": i, "b_count": len(b), "i_count": len(i)}

    def _tool_validate(self, schedule, players, skill_levels):
        play_count = {p: 0 for p in players}
        for g in schedule:
            for p in g.get("team_a", []) + g.get("team_b", []):
                play_count[p] = play_count.get(p, 0) + 1
        counts = list(play_count.values())
        spread = max(counts) - min(counts) if counts else 0
        return {"play_counts": play_count, "spread": spread, "is_fair": spread <= 1}

    # =========================================================================
    # Dispatch + parser
    # =========================================================================

    def _dispatch(self, tool_name: str, tool_input: Dict, ctx: Dict) -> str:
        try:
            if tool_name == "analyze_skill_distribution":
                r = self._tool_analyze_skills(
                    tool_input.get("players", ctx["players"]),
                    tool_input.get("skill_levels", ctx["skill_levels"]),
                )
            elif tool_name == "select_sitout_players":
                r = self._tool_select_sitouts(
                    tool_input.get("players", ctx["players"]),
                    tool_input.get("play_count", {}),
                    tool_input.get("sitout_count", {}),
                    tool_input.get("n", 2),
                )
            elif tool_name == "create_balanced_teams":
                r = self._tool_create_teams(
                    tool_input.get("available", []),
                    tool_input.get("skill_levels", ctx["skill_levels"]),
                    tool_input.get("partner_history", {}),
                )
            elif tool_name == "validate_schedule":
                r = self._tool_validate(
                    tool_input.get("schedule", []),
                    tool_input.get("players", ctx["players"]),
                    tool_input.get("skill_levels", ctx["skill_levels"]),
                )
            else:
                r = {"error": f"Unknown tool: {tool_name}"}
            return json.dumps(r)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _parse_react(self, text: str) -> Dict:
        result: Dict = {"thought": "", "action": None, "action_input": None, "final_answer": None}
        action_lines: List[str] = []
        in_action_input = False

        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("Thought:"):
                result["thought"] = s[8:].strip(); in_action_input = False
            elif s.startswith("Action:") and not s.startswith("Action Input:"):
                result["action"] = s[7:].strip(); in_action_input = False
            elif s.startswith("Action Input:"):
                action_lines = [s[13:].strip()]; in_action_input = True
            elif s.startswith("Final Answer:"):
                result["final_answer"] = s[13:].strip(); in_action_input = False
            elif in_action_input:
                action_lines.append(s)

        if action_lines:
            try:
                result["action_input"] = json.loads("\n".join(action_lines))
            except json.JSONDecodeError:
                result["action_input"] = {}

        return result
