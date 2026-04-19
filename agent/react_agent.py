"""
Agentic Game Planner — Gemini Flash.
The LLM generates the full schedule as JSON. Python validates priority rules
and feeds specific violations back to the LLM, which iterates until clean.

Priority 1: Every team must have one beginner + one intermediate.
Priority 2: Special instructions (e.g. no all-girl teams) are strictly enforced.
Fairness (sit-out rotation) is a soft goal — consecutive games are acceptable.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 6


def _cot(msg: str) -> None:
    """Print a chain-of-thought trace line to the console."""
    print(msg, flush=True)


SYSTEM_PROMPT = """You are an expert sports tournament scheduler.
Your job is to generate a complete doubles game schedule as valid JSON.
When given feedback about violations in a previous attempt, fix every one of them
and return the complete corrected schedule.
Always output ONLY the raw JSON array — no explanations, no markdown code fences.
"""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_gemini_config() -> dict:
    try:
        import streamlit as st
        if "gemini" in st.secrets:
            return dict(st.secrets["gemini"])
    except Exception:
        pass

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
    """Iterative agentic scheduler: LLM generates, Python validates, repeat."""

    def __init__(self) -> None:
        api_key = _load_api_key()
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set gemini.api_key in config.yaml."
            )

        gcfg = _load_gemini_config()
        genai.configure(api_key=api_key)

        generation_config = genai.GenerationConfig(
            temperature=float(gcfg.get("temperature", 0.7)),
            top_p=float(gcfg.get("top_p", 0.95)),
            max_output_tokens=int(gcfg.get("max_output_tokens", 8192)),
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
        previous_schedule: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        is_refine = bool(previous_schedule)
        mode = "REFINE" if is_refine else "FRESH"
        _cot(f"=== AGENT START ({mode}) ===")
        _cot(f"Players     : {players}")
        _cot(f"Skill levels: {skill_levels}")
        _cot(f"Rounds      : {num_rounds}  Courts: {num_courts}")
        if special_instructions.strip():
            _cot(f"Special instructions: {special_instructions.strip()}")
        if is_refine:
            _cot(f"Previous schedule   : {len(previous_schedule)} games (will be used as context)")

        _cot("\n[Step 1] Extracting structured constraints from special instructions...")
        constraints = self._extract_constraints(special_instructions)
        no_same_group = set(constraints.get("no_same_group", []))
        rule_summary  = constraints.get("rule_summary", "")
        _cot(f"  → no_same_group = {sorted(no_same_group) if no_same_group else '(none)'}")
        _cot(f"  → rule_summary  = {rule_summary if rule_summary else '(none)'}")

        chat = self.model.start_chat(history=[])
        schedule: Optional[List[Dict]] = None

        if is_refine:
            prompt = self._build_refine_prompt(
                players, skill_levels, num_rounds, num_courts,
                special_instructions, no_same_group, rule_summary,
                previous_schedule,
            )
        else:
            prompt = self._build_initial_prompt(
                players, skill_levels, num_rounds, num_courts,
                special_instructions, no_same_group, rule_summary,
            )

        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            _cot(f"\n[Iteration {iteration}/{MAX_AGENT_ITERATIONS}] Sending prompt to LLM...")
            _cot("--- PROMPT ---")
            _cot(prompt)
            _cot("--- END PROMPT ---")

            try:
                response = chat.send_message(prompt)
                text = response.text.strip()

                _cot("--- LLM RESPONSE (raw) ---")
                _cot(text[:2000] + (" ...[truncated]" if len(text) > 2000 else ""))
                _cot("--- END RESPONSE ---")

                parsed = self._parse_schedule_json(text)

                if parsed is None:
                    _cot("  ✗ Could not parse JSON from response. Asking LLM to retry.")
                    prompt = (
                        "Your response did not contain a valid JSON array. "
                        "Output ONLY the JSON array — start with [ and end with ]. "
                        "No markdown, no explanations."
                    )
                    continue

                _cot(f"  ✓ Parsed {len(parsed)} game entries from response.")
                schedule = self._add_metadata(parsed, players)
                violations = self._validate(schedule, players, skill_levels, no_same_group)

                if not violations:
                    _cot(f"\n  ✓ VALID SCHEDULE — no violations. Done in {iteration} iteration(s).")
                    _cot("=== AGENT END ===\n")
                    return schedule

                _cot(f"\n  ✗ Found {len(violations)} violation(s):")
                for v in violations:
                    tag = f"[P{v['priority']}]"
                    _cot(f"    {tag} {v['location']}, {v['team']}: {v['detail']}")

                prompt = self._build_feedback_prompt(violations)

            except Exception as exc:
                _cot(f"  ✗ Agent error: {exc}")
                logger.warning("Agent error at iteration %d: %s", iteration, exc)
                break

        _cot(f"\n  ⚠ Max iterations reached. Returning best schedule (may have violations).")
        _cot("=== AGENT END ===\n")
        return schedule or []

    # =========================================================================
    # Prompt builders
    # =========================================================================

    def _build_initial_prompt(
        self,
        players: List[str],
        skill_levels: Dict[str, str],
        num_rounds: int,
        num_courts: int,
        special_instructions: str,
        no_same_group: set,
        rule_summary: str = "",
    ) -> str:
        sitouts = max(0, len(players) - num_courts * 4)

        player_lines = "\n".join(
            f"  - {p} ({skill_levels.get(p, 'unknown')})" for p in players
        )

        special_block = (
            f"\nSPECIAL INSTRUCTIONS FROM ORGANISER:\n{special_instructions.strip()}\n"
            if special_instructions.strip() else ""
        )

        constraint_block = (
            f"\nEXTRACTED HARD CONSTRAINT: These players must NEVER be on the same team: "
            f"{', '.join(sorted(no_same_group))}\n"
            if no_same_group else ""
        )

        return f"""Generate a complete {num_rounds}-round doubles tournament schedule.

PLAYERS ({len(players)} total):
{player_lines}
{special_block}{constraint_block}
STRICT PRIORITIES (must all be satisfied):
  1. BALANCED MATCHUP — a team of 2 beginners must NEVER play against a team of 2 intermediates.
     A game is only forbidden when ALL beginners face ALL intermediates (pure mismatch).
     Allowed: [beginner+intermediate] vs [beginner+intermediate]
     Allowed: [beginner+intermediate] vs [intermediate+intermediate]
     Allowed: [beginner+beginner] vs [beginner+intermediate]
     FORBIDDEN: [beginner+beginner] vs [intermediate+intermediate]
  2. SPECIAL INSTRUCTION CONSTRAINTS — {
      f"players {sorted(no_same_group)} must NEVER be teammates."
      if no_same_group else "none."
  }{
      f"\n     {chr(10).join('     ' + line for line in rule_summary.splitlines())}"
      if rule_summary else ""
  }

FAIRNESS (soft goal — relax if needed to satisfy priorities above):
  - Try to distribute games evenly across players.
  - A player playing 2 or 3 consecutive rounds is acceptable.

TOURNAMENT SETTINGS:
  - Rounds: {num_rounds}
  - Courts per round: {num_courts}
  - Active players per round: {num_courts * 4}
  - Sitting out per round: {sitouts}

OUTPUT — return ONLY this JSON array, nothing else:
[
  {{
    "round": 1,
    "court": 1,
    "team_a": ["PlayerName", "PlayerName"],
    "team_b": ["PlayerName", "PlayerName"],
    "sitting_out": ["PlayerName"]
  }},
  ...
]

Rules:
  - Exactly {num_rounds * num_courts} entries total ({num_rounds} rounds × {num_courts} courts).
  - Both court entries for the same round must have identical sitting_out lists.
  - sitting_out has exactly {sitouts} name(s) per round.
  - team_a and team_b each have exactly 2 players.
  - Use exact player name spelling from the list above.
  - No player appears more than once in the same round.
"""

    def _build_refine_prompt(
        self,
        players: List[str],
        skill_levels: Dict[str, str],
        num_rounds: int,
        num_courts: int,
        special_instructions: str,
        no_same_group: set,
        rule_summary: str,
        previous_schedule: List[Dict],
    ) -> str:
        sitouts = max(0, len(players) - num_courts * 4)

        player_lines = "\n".join(
            f"  - {p} ({skill_levels.get(p, 'unknown')})" for p in players
        )

        # Strip to essentials — LLM only needs team assignments, not metadata
        stripped = [
            {
                "round": g.get("round"),
                "court": g.get("court"),
                "team_a": g.get("team_a", []),
                "team_b": g.get("team_b", []),
                "sitting_out": g.get("sitting_out", []),
            }
            for g in previous_schedule
        ]

        constraint_block = (
            f"\nHARD CONSTRAINT: Players {sorted(no_same_group)} must NEVER be teammates."
            if no_same_group else ""
        )
        rule_block = (
            f"\n{rule_summary}"
            if rule_summary else ""
        )

        return f"""You previously generated this doubles schedule. Now refine it.

PREVIOUS SCHEDULE (your last output):
{json.dumps(stripped, indent=2)}

PLAYERS ({len(players)} total):
{player_lines}

UPDATED SPECIAL INSTRUCTIONS (full, cumulative — ALL must be respected):
{special_instructions.strip()}
{constraint_block}{rule_block}

STRICT PRIORITIES (unchanged):
  1. BALANCED MATCHUP — FORBIDDEN: [beginner+beginner] vs [intermediate+intermediate]
  2. SPECIAL INSTRUCTION CONSTRAINTS — satisfy everything in the instructions above.

WHAT TO DO:
  - Keep rounds that already satisfy ALL constraints as-is.
  - Fix only the rounds/teams that violate any constraint.
  - Do NOT change sitting_out rotation unless forced by a constraint.

TOURNAMENT SETTINGS:
  - Rounds: {num_rounds}  Courts: {num_courts}
  - Active per round: {num_courts * 4}  Sitting out: {sitouts}

OUTPUT — return ONLY the complete refined JSON array, nothing else.
Same format as the previous schedule: round, court, team_a, team_b, sitting_out.
Exactly {num_rounds * num_courts} entries total.
"""

    def _build_feedback_prompt(self, violations: List[Dict]) -> str:
        p2 = [v for v in violations if v["priority"] == 2]
        p1 = [v for v in violations if v["priority"] == 1]

        lines = [
            f"Your schedule has {len(violations)} violation(s) that MUST be fixed.",
            "Regenerate the COMPLETE schedule correcting every violation listed below.\n",
        ]

        if p2:
            lines.append("PRIORITY 2 — SPECIAL INSTRUCTION VIOLATIONS (most critical):")
            for v in p2:
                lines.append(f"  • {v['location']}, {v['team']}: {v['detail']}")

        if p1:
            lines.append("\nPRIORITY 1 — BALANCED MATCHUP VIOLATIONS:")
            for v in p1:
                lines.append(f"  • {v['location']}, {v['team']}: {v['detail']}")

        lines.append("\nOutput ONLY the corrected JSON array.")
        return "\n".join(lines)

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate(
        self,
        schedule: List[Dict],
        players: List[str],
        skill_levels: Dict[str, str],
        no_same_group: set,
    ) -> List[Dict]:
        player_set = set(players)
        violations: List[Dict] = []

        for entry in schedule:
            rnd = entry.get("round", "?")
            court = entry.get("court", "?")
            loc = f"Round {rnd}, Court {court}"

            team_a = entry.get("team_a", [])
            team_b = entry.get("team_b", [])

            # Validate team sizes and player names
            for team_key, team in (("team_a", team_a), ("team_b", team_b)):
                if len(team) != 2:
                    violations.append({
                        "priority": 1, "location": loc, "team": team_key,
                        "detail": f"team has {len(team)} player(s), expected 2",
                    })
                    continue
                for p in team:
                    if p not in player_set:
                        violations.append({
                            "priority": 1, "location": loc, "team": team_key,
                            "detail": f"unknown player name '{p}'",
                        })

            # Priority 1: forbidden matchup — 2 beginners vs 2 intermediates
            if len(team_a) == 2 and len(team_b) == 2:
                skills_a = {skill_levels.get(p, "") for p in team_a}
                skills_b = {skill_levels.get(p, "") for p in team_b}
                if (skills_a == {"beginner"} and skills_b == {"intermediate"}) or \
                   (skills_a == {"intermediate"} and skills_b == {"beginner"}):
                    violations.append({
                        "priority": 1, "location": loc, "team": "both",
                        "detail": (
                            f"pure mismatch — "
                            f"team_a {team_a} ({'/'.join(skills_a)}) vs "
                            f"team_b {team_b} ({'/'.join(skills_b)}): "
                            "2 beginners must not face 2 intermediates"
                        ),
                    })

            # Priority 2: special instruction constraint (per-team)
            for team_key, team in (("team_a", team_a), ("team_b", team_b)):
                if len(team) == 2 and no_same_group:
                    p1, p2 = team
                    if p1 in no_same_group and p2 in no_same_group:
                        violations.append({
                            "priority": 2, "location": loc, "team": team_key,
                            "detail": f"{p1} and {p2} must NOT be teammates (special instruction)",
                        })

        return violations

    # =========================================================================
    # Helpers
    # =========================================================================

    def _extract_constraints(self, special_instructions: str) -> Dict:
        """Use LLM to extract structured constraints AND a formatted Allowed/Forbidden summary."""
        if not special_instructions or not special_instructions.strip():
            return {}
        prompt = (
            "Analyse the organiser's tournament instructions below and extract two things.\n\n"
            "1. no_same_group: list of player names who must NEVER be on the same doubles team.\n"
            "   These names come only from the instructions — do NOT invent names.\n\n"
            "2. rule_summary: a concise Allowed/Forbidden block (2-4 lines) that re-expresses\n"
            "   the constraint in the same style as this example:\n"
            "     FORBIDDEN: a team where both players are girls "
            "(both from [Girl1, Girl2, Girl3, Girl4])\n"
            "     ALLOWED:   a team with one girl and one non-girl\n"
            "     ALLOWED:   a team with two non-girls\n"
            "   Base the rule_summary entirely on the instructions — do NOT add rules that\n"
            "   are not stated. Gender or group identity comes ONLY from the instructions text.\n\n"
            "Return ONLY valid JSON matching this schema exactly:\n"
            '  {"no_same_group": ["Name1", ...], "rule_summary": "FORBIDDEN: ...\\nALLOWED: ..."}\n'
            "If no pairing constraint exists, return {}.\n\n"
            f"Instructions: {special_instructions.strip()}"
        )
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as exc:
            logger.warning("Could not extract constraints: %s", exc)
        return {}

    def _parse_schedule_json(self, text: str) -> Optional[List[Dict]]:
        """Extract a JSON array from the LLM response, stripping any markdown fences."""
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return None

    def _add_metadata(self, schedule: List[Dict], players: List[str]) -> List[Dict]:
        """Add time_slot and game_id fields if missing."""
        for entry in schedule:
            rnd = entry.get("round", 1)
            court = entry.get("court", 1)
            entry.setdefault("time_slot", f"{(rnd - 1) * 10}–{rnd * 10} min")
            entry.setdefault("game_id", f"r{rnd}_c{court}")
            if "sitting_out" not in entry:
                active = set(entry.get("team_a", []) + entry.get("team_b", []))
                entry["sitting_out"] = [p for p in players if p not in active]
        return schedule
