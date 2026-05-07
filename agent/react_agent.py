"""
Agentic Game Planner — Gemini Flash.

Five validation layers (applied in priority order):
  1  HARD             : slot conflict, skill mismatch, girls rule, no_same_group  → 10 000 each
  2  PARTNER_REPEAT   : same pair partnered > max_partner_repeat times            →    500 × excess
  3  SITOUT_BALANCE   : sit-out spread across players > 2                         →     50 × spread
  4  OPPONENT_REPEAT  : same pair faced each other > max_opponent_repeat times    →     10 × excess
  5  REQUIRED_PAIRING : required partner pair missing or in wrong round           →  5 000 each

Constraints are built dynamically at runtime from user inputs — nothing is hardcoded.
"""

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import google.generativeai as genai

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS   = 6

PENALTY_HARD             = 10_000
PENALTY_REQUIRED_PAIRING =  5_000
PENALTY_PARTNER_REPEAT   =    500
PENALTY_SITOUT_IMBALANCE =     50
PENALTY_OPPONENT_REPEAT  =     10


# ---------------------------------------------------------------------------
# Tournament math: compute minimum inevitable repeat thresholds
# ---------------------------------------------------------------------------

def _compute_repeat_thresholds(
    n_players:  int,
    num_rounds: int,
    num_courts: int,
) -> Tuple[int, int, float, float]:
    """
    Given the tournament size, calculate the minimum partner/opponent repeats
    that are mathematically unavoidable so the validator does not flag them.

    Returns (max_partner_repeat, max_opponent_repeat, partner_avg, opponent_avg).

    Derivation:
      total_games        = num_rounds × num_courts
      unique_pairs       = n × (n-1) / 2
      partner_avg        = (total_games × 2) / unique_pairs   [2 pairs per game]
      opponent_avg       = (total_games × 4) / unique_pairs   [4 cross-pairs per game]
      thresholds         = ceil(average)  — the minimum max that keeps penalties fair
    """
    total_games  = num_rounds * num_courts
    unique_pairs = n_players * (n_players - 1) / 2
    if unique_pairs == 0:
        return 1, 1, 0.0, 0.0

    partner_avg        = (total_games * 2) / unique_pairs
    opponent_avg       = (total_games * 4) / unique_pairs
    max_partner_repeat = max(1, math.ceil(partner_avg))
    max_opponent_repeat= max(1, math.ceil(opponent_avg))

    return max_partner_repeat, max_opponent_repeat, partner_avg, opponent_avg


# ---------------------------------------------------------------------------
# Runtime constraint bag  (assembled fresh each tournament from user inputs)
# ---------------------------------------------------------------------------

@dataclass
class ConstraintSet:
    skill_levels:        Dict[str, str]  = field(default_factory=dict)
    girl_names:          Set[str]        = field(default_factory=set)
    no_same_group:       Set[str]        = field(default_factory=set)
    max_partner_repeat:  int             = 2
    max_opponent_repeat: int             = 2
    # Each entry: {"players": ["A","B"], "round": int|None, "min_count": int}
    # round=None means any round; min_count defaults to 1
    required_partners:   List[Dict]      = field(default_factory=list)
    # Each entry: {"players": ["A","B"], "min_count": int}
    required_opponents:  List[Dict]      = field(default_factory=list)
    # {court_number: first_round_it_is_active}  e.g. {1: 1, 2: 4}
    court_start_rounds:  Dict[int, int]  = field(default_factory=dict)
    # {"Player A": "09:30"} — player must not play in any game starting at/after this time
    player_latest:       Dict[str, str]  = field(default_factory=dict)
    # {"Player A": "08:00"} — player must not play in any game starting before this time
    player_earliest:     Dict[str, str]  = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        skill_levels:        Dict[str, str],
        girl_names:          Optional[List[str]] = None,
        no_same_group:       Optional[List[str]] = None,
        max_partner_repeat:  int = 2,
        max_opponent_repeat: int = 2,
        required_partners:   Optional[List[Dict]] = None,
        required_opponents:  Optional[List[Dict]] = None,
        court_start_rounds:  Optional[Dict[int, int]] = None,
        player_latest:       Optional[Dict[str, str]] = None,
        player_earliest:     Optional[Dict[str, str]] = None,
    ) -> "ConstraintSet":
        return cls(
            skill_levels        = skill_levels,
            girl_names          = set(girl_names    or []),
            no_same_group       = set(no_same_group or []),
            max_partner_repeat  = max_partner_repeat,
            max_opponent_repeat = max_opponent_repeat,
            required_partners   = required_partners or [],
            required_opponents  = required_opponents or [],
            court_start_rounds  = court_start_rounds or {},
            player_latest       = player_latest  or {},
            player_earliest     = player_earliest or {},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cot(msg: str) -> None:
    print(msg, flush=True)


SYSTEM_PROMPT = """You are an expert sports tournament scheduler.
Your job is to generate a complete doubles game schedule as valid JSON.
When given feedback about violations in a previous attempt, fix every one of them
and return the complete corrected schedule.
Output ONLY a JSON array — start with [ and end with ]. No markdown, no explanations, no code fences.
"""


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
# Scoring & validation
# ---------------------------------------------------------------------------

def _validate_and_score(
    schedule:    List[Dict],
    players:     List[str],
    constraints: ConstraintSet,
) -> Tuple[List[Dict], int]:
    """
    Validate schedule against all four constraint layers.
    Returns (violations, total_penalty).
    Each violation: {priority, layer, location, team, detail, penalty}
    """
    violations: List[Dict] = []
    total_penalty = 0
    player_set = set(players)

    # ── Pre-compute pair counts across the whole schedule ────────────────────
    partner_counts:  Counter = Counter()
    opponent_counts: Counter = Counter()
    sitout_counts:   Counter = Counter()

    for entry in schedule:
        ta = entry.get("team_a", [])
        tb = entry.get("team_b", [])
        if len(ta) == 2:
            partner_counts[frozenset(ta)] += 1
        if len(tb) == 2:
            partner_counts[frozenset(tb)] += 1
        if len(ta) == 2 and len(tb) == 2:
            for pa in ta:
                for pb in tb:
                    opponent_counts[frozenset({pa, pb})] += 1
        for p in entry.get("sitting_out", []):
            sitout_counts[p] += 1

    # ── Group by round for slot-conflict detection ───────────────────────────
    rounds: Dict[int, List[Dict]] = defaultdict(list)
    for entry in schedule:
        rounds[entry.get("round", 0)].append(entry)

    # ── Layer 1: Hard constraints ────────────────────────────────────────────
    for rnd, games in rounds.items():

        # Court availability: game on a court before it opens
        if constraints.court_start_rounds:
            for g in games:
                court     = g.get("court", 1)
                avail_rnd = constraints.court_start_rounds.get(court, 1)
                if rnd < avail_rnd:
                    total_penalty += PENALTY_HARD
                    violations.append({
                        "priority": 1, "layer": "HARD",
                        "location": f"Round {rnd}, Court {court}",
                        "team": "court_availability",
                        "detail": (
                            f"Court {court} not available until round {avail_rnd} "
                            f"but game scheduled in round {rnd}"
                        ),
                        "penalty": PENALTY_HARD,
                    })

        # Slot conflict: same player active on two courts in the same round
        active: List[str] = []
        for g in games:
            active.extend(g.get("team_a", []) + g.get("team_b", []))
        for p, cnt in Counter(active).items():
            if cnt > 1:
                total_penalty += PENALTY_HARD
                violations.append({
                    "priority": 1, "layer": "HARD",
                    "location": f"Round {rnd}", "team": "slot_conflict",
                    "detail": f"{p} plays on {cnt} courts simultaneously",
                    "penalty": PENALTY_HARD,
                })

        for entry in games:
            loc    = f"Round {entry.get('round','?')}, Court {entry.get('court','?')}"
            team_a = entry.get("team_a", [])
            team_b = entry.get("team_b", [])

            # Team size & unknown player names
            for team_key, team in (("team_a", team_a), ("team_b", team_b)):
                if len(team) != 2:
                    total_penalty += PENALTY_HARD
                    violations.append({
                        "priority": 1, "layer": "HARD",
                        "location": loc, "team": team_key,
                        "detail": f"team has {len(team)} player(s), expected 2",
                        "penalty": PENALTY_HARD,
                    })
                    continue
                for p in team:
                    if p not in player_set:
                        total_penalty += PENALTY_HARD
                        violations.append({
                            "priority": 1, "layer": "HARD",
                            "location": loc, "team": team_key,
                            "detail": f"unknown player name '{p}'",
                            "penalty": PENALTY_HARD,
                        })

            if len(team_a) != 2 or len(team_b) != 2:
                continue

            # Skill mismatch: pure beginner team vs pure intermediate team
            skills_a = {constraints.skill_levels.get(p, "") for p in team_a}
            skills_b = {constraints.skill_levels.get(p, "") for p in team_b}
            if (skills_a == {"beginner"} and skills_b == {"intermediate"}) or \
               (skills_a == {"intermediate"} and skills_b == {"beginner"}):
                total_penalty += PENALTY_HARD
                violations.append({
                    "priority": 1, "layer": "HARD",
                    "location": loc, "team": "both",
                    "detail": (
                        f"pure skill mismatch — "
                        f"{team_a} ({'/'.join(skills_a)}) vs "
                        f"{team_b} ({'/'.join(skills_b)})"
                    ),
                    "penalty": PENALTY_HARD,
                })

            # Girls rule: 2 girls must not face an all-boys team
            if constraints.girl_names:
                for team_key, team, other in (
                    ("team_a", team_a, team_b),
                    ("team_b", team_b, team_a),
                ):
                    girls_here  = [p for p in team  if p in constraints.girl_names]
                    girls_there = [p for p in other if p in constraints.girl_names]
                    if len(girls_here) == 2 and len(girls_there) == 0:
                        total_penalty += PENALTY_HARD
                        violations.append({
                            "priority": 1, "layer": "HARD",
                            "location": loc, "team": team_key,
                            "detail": (
                                f"2 girls ({', '.join(girls_here)}) facing "
                                f"all-boys team ({', '.join(other)})"
                            ),
                            "penalty": PENALTY_HARD,
                        })

            # no_same_group: named players must never be teammates
            if constraints.no_same_group:
                for team_key, team in (("team_a", team_a), ("team_b", team_b)):
                    p1, p2 = team
                    if p1 in constraints.no_same_group and p2 in constraints.no_same_group:
                        total_penalty += PENALTY_HARD
                        violations.append({
                            "priority": 1, "layer": "HARD",
                            "location": loc, "team": team_key,
                            "detail": f"{p1} and {p2} must NOT be teammates",
                            "penalty": PENALTY_HARD,
                        })

            # Player time availability
            if constraints.player_latest or constraints.player_earliest:
                ts = entry.get("time_slot", "")
                if ts and "–" in ts:
                    start_str = ts.split("–")[0].strip()
                    try:
                        sh, sm   = map(int, start_str.split(":"))
                        start_mins = sh * 60 + sm
                        for p in team_a + team_b:
                            if p in constraints.player_latest:
                                lh, lm = map(int, constraints.player_latest[p].split(":"))
                                if start_mins >= lh * 60 + lm:
                                    total_penalty += PENALTY_HARD
                                    violations.append({
                                        "priority": 1, "layer": "HARD",
                                        "location": loc, "team": "time_availability",
                                        "detail": (
                                            f"{p} scheduled at {start_str} but must not"
                                            f" play at/after {constraints.player_latest[p]}"
                                        ),
                                        "penalty": PENALTY_HARD,
                                    })
                            if p in constraints.player_earliest:
                                eh, em = map(int, constraints.player_earliest[p].split(":"))
                                if start_mins < eh * 60 + em:
                                    total_penalty += PENALTY_HARD
                                    violations.append({
                                        "priority": 1, "layer": "HARD",
                                        "location": loc, "team": "time_availability",
                                        "detail": (
                                            f"{p} scheduled at {start_str} but must not"
                                            f" play before {constraints.player_earliest[p]}"
                                        ),
                                        "penalty": PENALTY_HARD,
                                    })
                    except ValueError:
                        pass

    # ── Layer 2: Partner repeats ─────────────────────────────────────────────
    # Build a lookup: pair → required min_count (so required pairs aren't
    # penalised for doing exactly what they were told to do)
    _required_partner_min: Dict[frozenset, int] = {}
    for req in constraints.required_partners:
        fp = frozenset(req.get("players", []))
        if fp:
            _required_partner_min[fp] = max(
                _required_partner_min.get(fp, 0),
                req.get("min_count", 1),
            )

    for pair, count in partner_counts.items():
        # Raise the threshold for explicitly required pairs so their mandated
        # repetitions are never flagged as violations
        effective_max = max(
            constraints.max_partner_repeat,
            _required_partner_min.get(pair, 0),
        )
        if count > effective_max:
            excess = count - effective_max
            names  = sorted(pair)
            pen    = PENALTY_PARTNER_REPEAT * excess
            total_penalty += pen
            violations.append({
                "priority": 2, "layer": "PARTNER_REPEAT",
                "location": "whole schedule",
                "team": f"{names[0]} & {names[1]}",
                "detail": f"partnered {count}× (max {effective_max})",
                "penalty": pen,
            })

    # ── Layer 3: Sit-out imbalance ───────────────────────────────────────────
    if sitout_counts:
        max_so = max(sitout_counts.values())
        min_so = min(sitout_counts.get(p, 0) for p in players)
        spread = max_so - min_so
        if spread > 2:
            pen = PENALTY_SITOUT_IMBALANCE * spread
            total_penalty += pen
            violations.append({
                "priority": 3, "layer": "SITOUT_BALANCE",
                "location": "whole schedule", "team": "sit-outs",
                "detail": (
                    f"spread={spread} (max {max_so}, min {min_so}) — "
                    f"counts: {dict(sorted(sitout_counts.items()))}"
                ),
                "penalty": pen,
            })

    # ── Layer 4: Opponent repeats ────────────────────────────────────────────
    for pair, count in opponent_counts.items():
        if count > constraints.max_opponent_repeat:
            excess = count - constraints.max_opponent_repeat
            names  = sorted(pair)
            pen    = PENALTY_OPPONENT_REPEAT * excess
            total_penalty += pen
            violations.append({
                "priority": 4, "layer": "OPPONENT_REPEAT",
                "location": "whole schedule",
                "team": f"{names[0]} vs {names[1]}",
                "detail": f"faced each other {count}× (max {constraints.max_opponent_repeat})",
                "penalty": pen,
            })

    # ── Layer 5: Required partner pairings ───────────────────────────────────
    for req in constraints.required_partners:
        req_pair  = frozenset(req.get("players", []))
        req_round = req.get("round")        # exact round pin (None = no pin)
        by_round  = req.get("by_round")     # must occur at or before this round
        min_count = req.get("min_count", 1)
        names     = sorted(req_pair)
        label     = f"{names[0]} & {names[1]}"

        if req_round is not None:
            # Exact round — must partner in that specific round
            found = any(
                frozenset(entry.get("team_a", [])) == req_pair or
                frozenset(entry.get("team_b", [])) == req_pair
                for entry in schedule
                if entry.get("round") == req_round
            )
            if not found:
                total_penalty += PENALTY_REQUIRED_PAIRING
                violations.append({
                    "priority": 5, "layer": "REQUIRED_PAIRING",
                    "location": f"Round {req_round}",
                    "team": label,
                    "detail": f"{label} must be partners in round {req_round} but are not",
                    "penalty": PENALTY_REQUIRED_PAIRING,
                })
        elif by_round is not None:
            # Must partner at least min_count times within rounds 1..by_round
            early_count = sum(
                1 for entry in schedule
                if entry.get("round", 0) <= by_round and (
                    frozenset(entry.get("team_a", [])) == req_pair or
                    frozenset(entry.get("team_b", [])) == req_pair
                )
            )
            if early_count < min_count:
                shortfall = min_count - early_count
                pen       = PENALTY_REQUIRED_PAIRING * shortfall
                total_penalty += pen
                violations.append({
                    "priority": 5, "layer": "REQUIRED_PAIRING",
                    "location": f"Rounds 1–{by_round}",
                    "team": label,
                    "detail": (
                        f"{label} must partner ≥{min_count}× by round {by_round} "
                        f"but only do so {early_count}× (short by {shortfall})"
                    ),
                    "penalty": pen,
                })
        else:
            # Must partner at least min_count times anywhere in the schedule
            actual = partner_counts.get(req_pair, 0)
            if actual < min_count:
                shortfall = min_count - actual
                pen       = PENALTY_REQUIRED_PAIRING * shortfall
                total_penalty += pen
                violations.append({
                    "priority": 5, "layer": "REQUIRED_PAIRING",
                    "location": "whole schedule",
                    "team": label,
                    "detail": (
                        f"{label} must partner ≥{min_count}× "
                        f"but only do so {actual}× (short by {shortfall})"
                    ),
                    "penalty": pen,
                })

    # ── Layer 5b: Required opponent pairings ─────────────────────────────────
    for req in constraints.required_opponents:
        req_pair  = frozenset(req.get("players", []))
        by_round  = req.get("by_round")     # must occur at or before this round
        min_count = req.get("min_count", 1)
        names     = sorted(req_pair)
        label     = f"{names[0]} vs {names[1]}"

        if by_round is not None:
            # Must face each other at least min_count times within rounds 1..by_round
            early_count = sum(
                1 for entry in schedule
                if entry.get("round", 0) <= by_round
                for pa in entry.get("team_a", [])
                for pb in entry.get("team_b", [])
                if frozenset({pa, pb}) == req_pair
            )
            if early_count < min_count:
                shortfall = min_count - early_count
                pen       = PENALTY_REQUIRED_PAIRING * shortfall
                total_penalty += pen
                violations.append({
                    "priority": 5, "layer": "REQUIRED_PAIRING",
                    "location": f"Rounds 1–{by_round}",
                    "team": label,
                    "detail": (
                        f"{label} must face each other ≥{min_count}× by round {by_round} "
                        f"but only do so {early_count}× (short by {shortfall})"
                    ),
                    "penalty": pen,
                })
        else:
            actual = opponent_counts.get(req_pair, 0)
            if actual < min_count:
                shortfall = min_count - actual
                pen       = PENALTY_REQUIRED_PAIRING * shortfall
                total_penalty += pen
                violations.append({
                    "priority": 5, "layer": "REQUIRED_PAIRING",
                    "location": "whole schedule",
                    "team": label,
                    "detail": (
                        f"{label} must face each other ≥{min_count}× "
                        f"but only do so {actual}× (short by {shortfall})"
                    ),
                    "penalty": pen,
                })

    return violations, total_penalty


def _identify_clean_rounds(
    schedule:    List[Dict],
    players:     List[str],
    constraints: ConstraintSet,
) -> Set[int]:
    """Return round numbers that have zero hard-layer violations."""
    violations, _ = _validate_and_score(schedule, players, constraints)
    dirty: Set[int] = set()
    for v in violations:
        if v["layer"] == "HARD":
            m = re.match(r"Round\s+(\d+)", v["location"])
            if m:
                dirty.add(int(m.group(1)))
    all_rounds = {entry.get("round") for entry in schedule}
    return all_rounds - dirty


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GamePlannerAgent:
    """Iterative agentic scheduler: LLM generates, Python validates+scores, repeat."""

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
            response_mime_type="application/json",
        )
        self.model = genai.GenerativeModel(
            model_name=gcfg.get("model_name", "gemini-2.0-flash"),
            system_instruction=SYSTEM_PROMPT,
            generation_config=generation_config,
        )
        logger.info(
            "Gemini model ready: %s | temp=%.1f | top_p=%.2f",
            gcfg.get("model_name", "gemini-2.0-flash"),
            generation_config.temperature,
            generation_config.top_p,
        )

    # =========================================================================
    # Public
    # =========================================================================

    def generate_schedule(
        self,
        players:              List[str],
        skill_levels:         Dict[str, str],
        num_rounds:           int = 12,
        num_courts:           int = 2,
        special_instructions: str = "",
        previous_schedule:    Optional[List[Dict]] = None,
        girl_names:           Optional[List[str]] = None,
        court_start_rounds:   Optional[Dict[int, int]] = None,
        court_start_times:    Optional[Dict[int, str]] = None,
        mins_per_game:        int = 10,
    ) -> List[Dict]:
        is_refine = bool(previous_schedule)
        mode      = "REFINE" if is_refine else "FRESH"
        _cot(f"=== AGENT START ({mode}) ===")
        _cot(f"Players     : {players}")
        _cot(f"Skill levels: {skill_levels}")
        _cot(f"Girl names  : {girl_names or '(none)'}")
        _cot(f"Rounds      : {num_rounds}  Courts: {num_courts}")
        _cot(f"Mins/game   : {mins_per_game}")
        if court_start_rounds:
            for _c, _r in sorted(court_start_rounds.items()):
                _t = (court_start_times or {}).get(_c, "?")
                _cot(f"  Court {_c}: available from round {_r} ({_t})")
        if special_instructions.strip():
            _cot(f"Special     : {special_instructions.strip()}")
        if is_refine:
            _cot(f"Previous    : {len(previous_schedule)} games")

        # Store timing metadata on self so _add_metadata can use it
        self._court_start_times  = court_start_times  or {}
        self._court_start_rounds = court_start_rounds or {}
        self._mins_per_game      = mins_per_game

        # Step 1: Extract free-text constraints via LLM
        _cot("\n[Step 1] Extracting constraints from special instructions...")
        extracted          = self._extract_constraints(special_instructions)
        no_same_group      = set(extracted.get("no_same_group", []))
        required_partners  = extracted.get("required_partners", [])
        required_opponents = extracted.get("required_opponents", [])
        player_latest      = extracted.get("player_latest", {})
        player_earliest    = extracted.get("player_earliest", {})
        rule_summary       = extracted.get("rule_summary", "")
        _cot(f"  → no_same_group      : {sorted(no_same_group) or '(none)'}")
        _cot(f"  → required_partners  : {required_partners or '(none)'}")
        _cot(f"  → required_opponents : {required_opponents or '(none)'}")
        _cot(f"  → player_latest      : {player_latest or '(none)'}")
        _cot(f"  → player_earliest    : {player_earliest or '(none)'}")
        _cot(f"  → rule_summary       : {rule_summary or '(none)'}")

        # Step 2: Compute tournament-specific repeat thresholds, then build constraint set
        _max_partner, _max_opponent, _partner_avg, _opponent_avg = _compute_repeat_thresholds(
            len(players), num_rounds, num_courts
        )
        _cot(f"\n[Step 2] Repeat thresholds (computed for {len(players)}p × {num_rounds}r × {num_courts}c):")
        _cot(f"  → partner avg  {_partner_avg:.2f}  → max allowed {_max_partner}")
        _cot(f"  → opponent avg {_opponent_avg:.2f}  → max allowed {_max_opponent}")

        constraints = ConstraintSet.build(
            skill_levels        = skill_levels,
            girl_names          = girl_names,
            no_same_group       = list(no_same_group),
            max_partner_repeat  = _max_partner,
            max_opponent_repeat = _max_opponent,
            required_partners   = required_partners,
            required_opponents  = required_opponents,
            court_start_rounds  = court_start_rounds,
            player_latest       = player_latest,
            player_earliest     = player_earliest,
        )

        # Step 3 (refine only): score previous schedule, identify clean rounds
        clean_rounds:    Set[int]   = set()
        prev_violations: List[Dict] = []
        if is_refine:
            _cot("\n[Step 3] Scoring previous schedule and locking clean rounds...")
            prev_violations, prev_penalty = _validate_and_score(
                previous_schedule, players, constraints
            )
            clean_rounds = _identify_clean_rounds(previous_schedule, players, constraints)
            _cot(f"  → Previous penalty : {prev_penalty:,}")
            _cot(f"  → Clean rounds     : {sorted(clean_rounds)}")
            _cot(f"  → Violations       : {len(prev_violations)}")

        # Step 4: Build opening prompt
        if is_refine:
            prompt = self._build_refine_prompt(
                players, skill_levels, num_rounds, num_courts,
                special_instructions, constraints, rule_summary,
                previous_schedule, prev_violations, clean_rounds,
                court_start_rounds=court_start_rounds,
                court_start_times=court_start_times,
                mins_per_game=mins_per_game,
            )
        else:
            prompt = self._build_initial_prompt(
                players, skill_levels, num_rounds, num_courts,
                special_instructions, constraints, rule_summary,
                court_start_rounds=court_start_rounds,
                court_start_times=court_start_times,
                mins_per_game=mins_per_game,
            )

        # Step 5: Feedback loop
        chat: genai.ChatSession = self.model.start_chat(history=[])
        schedule:              Optional[List[Dict]] = None
        last_violations:       List[Dict]           = []
        best_schedule:         Optional[List[Dict]] = None   # lowest total penalty (fallback)
        best_violations:       List[Dict]           = []
        best_penalty           = float("inf")
        best_clean_schedule:   Optional[List[Dict]] = None   # zero hard, lowest soft penalty
        best_clean_violations: List[Dict]           = []
        best_clean_penalty     = float("inf")
        best_clean_iter        = 0

        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            _cot(f"\n[Iteration {iteration}/{MAX_AGENT_ITERATIONS}]")
            _cot("--- PROMPT ---")
            _cot(prompt)
            _cot("--- END PROMPT ---")

            try:
                response = chat.send_message(prompt)
                text     = response.text.strip()
                _cot("--- LLM RESPONSE ---")
                _cot(text[:2000] + (" ...[truncated]" if len(text) > 2000 else ""))
                _cot("--- END RESPONSE ---")

                parsed = self._parse_schedule_json(text)
                if parsed is None:
                    _cot("  ✗ No valid JSON array found. Asking LLM to retry.")
                    prompt = (
                        "Your response did not contain a valid JSON array. "
                        "Output ONLY the JSON array — start with [ and end with ]. "
                        "No markdown, no explanations."
                    )
                    continue

                _cot(f"  ✓ Parsed {len(parsed)} entries.")
                schedule                  = self._add_metadata(parsed, players)
                violations, penalty       = _validate_and_score(schedule, players, constraints)
                last_violations           = violations

                hard_count   = sum(1 for v in violations if v["layer"] == "HARD")
                soft_penalty = sum(v["penalty"] for v in violations if v["layer"] != "HARD")

                _cot(f"  → Penalty : {penalty:,}  (hard={hard_count}, soft={soft_penalty:,})")
                for v in violations:
                    _cot(f"    [{v['layer']}] {v['location']} | {v['team']}: {v['detail']} (−{v['penalty']})")

                # Track absolute best (any penalty) as fallback
                if penalty < best_penalty:
                    best_penalty     = penalty
                    best_schedule    = schedule
                    best_violations  = violations

                # Track best clean (zero hard violations, lowest soft penalty)
                if hard_count == 0 and soft_penalty < best_clean_penalty:
                    best_clean_penalty     = soft_penalty
                    best_clean_schedule    = schedule
                    best_clean_violations  = violations
                    best_clean_iter        = iteration
                    _cot(f"  ★ New best clean schedule (iter {iteration}, soft penalty={soft_penalty:,})")

                if penalty == 0:
                    _cot(f"\n  ✓ PERFECT — zero penalty. Done in {iteration} iteration(s).")
                    _cot("=== AGENT END ===\n")
                    return schedule, violations

                prompt = self._build_feedback_prompt(violations, penalty)

            except Exception as exc:
                _cot(f"  ✗ Agent error: {exc}")
                logger.warning("Agent error at iteration %d: %s", iteration, exc)
                break

        if best_clean_schedule is not None:
            _cot(f"\n  ✓ All iterations done. Returning best clean schedule"
                 f" from iteration {best_clean_iter}"
                 f" (0 hard, soft penalty={best_clean_penalty:,}).")
            _cot("=== AGENT END ===\n")
            return best_clean_schedule, best_clean_violations
        else:
            _cot(f"\n  ⚠ No clean schedule found. Returning best overall (penalty={best_penalty:,}).")
            _cot("=== AGENT END ===\n")
            return (best_schedule or schedule or []), (best_violations or last_violations)

    # =========================================================================
    # Prompt builders
    # =========================================================================

    def _build_initial_prompt(
        self,
        players:              List[str],
        skill_levels:         Dict[str, str],
        num_rounds:           int,
        num_courts:           int,
        special_instructions: str,
        constraints:          ConstraintSet,
        rule_summary:         str = "",
        court_start_rounds:   Optional[Dict[int, int]] = None,
        court_start_times:    Optional[Dict[int, str]] = None,
        mins_per_game:        int = 10,
    ) -> str:
        sitouts      = max(0, len(players) - num_courts * 4)
        total_games  = num_rounds * num_courts
        unique_pairs = len(players) * (len(players) - 1) // 2
        _, _, partner_avg, opponent_avg = _compute_repeat_thresholds(
            len(players), num_rounds, num_courts
        )
        player_lines = "\n".join(
            f"  - {p} ({skill_levels.get(p, 'unknown')})" for p in players
        )
        special_block = (
            f"\nSPECIAL INSTRUCTIONS FROM ORGANISER:\n{special_instructions.strip()}\n"
            if special_instructions.strip() else ""
        )
        constraint_block = ""
        if constraints.no_same_group:
            constraint_block += (
                f"\nHARD CONSTRAINT — these players must NEVER be teammates: "
                f"{', '.join(sorted(constraints.no_same_group))}\n"
            )
        if constraints.girl_names:
            constraint_block += (
                f"\nGIRLS CONSTRAINT — girls are: {', '.join(sorted(constraints.girl_names))}\n"
                f"  FORBIDDEN: a team of 2 girls facing an all-boys team.\n"
                f"  ALLOWED:   one girl + one non-girl on any team.\n"
                f"  ALLOWED:   two non-girls on any team.\n"
            )
        if constraints.required_partners:
            lines = []
            for req in constraints.required_partners:
                p  = req.get("players", [])
                r  = req.get("round")
                br = req.get("by_round")
                n  = req.get("min_count", 1)
                if len(p) == 2:
                    if r is not None:
                        lines.append(f"  • {p[0]} and {p[1]} MUST be partners in exactly round {r}.")
                    elif br is not None and n > 1:
                        lines.append(f"  • {p[0]} and {p[1]} MUST be partners ≥{n} times, first time by round {br}.")
                    elif br is not None:
                        lines.append(f"  • {p[0]} and {p[1]} MUST be partners at least once within rounds 1–{br}.")
                    elif n > 1:
                        lines.append(f"  • {p[0]} and {p[1]} MUST be partners at least {n} times.")
                    else:
                        lines.append(f"  • {p[0]} and {p[1]} MUST be partners at least once.")
            constraint_block += "\nREQUIRED PAIRINGS — these partner pairs are mandatory:\n" + "\n".join(lines) + "\n"
        if constraints.required_opponents:
            lines = []
            for req in constraints.required_opponents:
                p  = req.get("players", [])
                br = req.get("by_round")
                n  = req.get("min_count", 1)
                if len(p) == 2:
                    times = f"at least {n} time{'s' if n > 1 else ''}"
                    if br is not None:
                        lines.append(f"  • {p[0]} and {p[1]} MUST face each other {times} within rounds 1–{br}.")
                    else:
                        lines.append(f"  • {p[0]} and {p[1]} MUST face each other as opponents {times}.")
            constraint_block += "\nREQUIRED OPPONENTS — these matchups are mandatory:\n" + "\n".join(lines) + "\n"
        if constraints.player_latest:
            lines = [
                f"  • {p} must NOT be in any game starting at or after {t}."
                for p, t in sorted(constraints.player_latest.items())
            ]
            constraint_block += "\nPLAYER AVAILABILITY (latest) — HARD CONSTRAINT:\n" + "\n".join(lines) + "\n"
        if constraints.player_earliest:
            lines = [
                f"  • {p} must NOT be in any game starting before {t}."
                for p, t in sorted(constraints.player_earliest.items())
            ]
            constraint_block += "\nPLAYER AVAILABILITY (earliest) — HARD CONSTRAINT:\n" + "\n".join(lines) + "\n"
        rule_block = (
            f"\nRULE SUMMARY (from special instructions):\n"
            + "\n".join(f"  {l}" for l in rule_summary.splitlines()) + "\n"
            if rule_summary else ""
        )

        math_block = (
            f"\nMATHEMATICAL LIMITS (pre-calculated for this tournament):\n"
            f"  {len(players)} players · {total_games} games · {unique_pairs} unique pairs\n"
            f"  Each pair will partner on average {partner_avg:.1f}× "
            f"→ repeats above {constraints.max_partner_repeat} are avoidable and should be minimised.\n"
            f"  Each pair will face each other on average {opponent_avg:.1f}× "
            f"→ repeats above {constraints.max_opponent_repeat} are avoidable and should be minimised.\n"
            f"  These averages are unavoidable — distribute pairings as evenly as possible within them.\n"
        )

        # Court availability block — only shown when courts have staggered starts
        court_avail_block = ""
        staggered = court_start_rounds and any(r > 1 for r in court_start_rounds.values())
        if staggered:
            lines = []
            for c in sorted(court_start_rounds.keys()):
                r   = court_start_rounds[c]
                t   = (court_start_times or {}).get(c, "?")
                if r == 1:
                    lines.append(f"  Court {c}: active from round 1 onwards ({t})")
                else:
                    lines.append(
                        f"  Court {c}: NOT active in rounds 1–{r - 1}; "
                        f"active from round {r} onwards ({t})"
                    )
            # Describe sit-out counts per phase
            early_sitout = len(players) - 4          # only 1 court active
            late_sitout  = max(0, len(players) - num_courts * 4)
            first_stagger = min(r for r in court_start_rounds.values() if r > 1)
            lines.append(
                f"\n  Rounds 1–{first_stagger - 1}: only 1 court active → "
                f"{early_sitout} players sit out each round."
            )
            lines.append(
                f"  Round {first_stagger} onwards: all {num_courts} courts active → "
                f"{late_sitout} player(s) sit out each round."
            )
            court_avail_block = "\nCOURT AVAILABILITY (staggered start):\n" + "\n".join(lines) + "\n"

        # Dynamic format rule — entry count varies when courts are staggered
        if staggered:
            total_entries = sum(
                num_rounds - (court_start_rounds.get(c, 1) - 1)
                for c in range(1, num_courts + 1)
            )
            format_entry_rule = (
                f"  - Total entries = {total_entries} "
                f"(courts start at different rounds — see COURT AVAILABILITY)."
            )
            format_sitout_rule = (
                f"  - sitting_out varies by round: "
                f"{early_sitout} names before round {first_stagger}, "
                f"{late_sitout} name(s) from round {first_stagger} onwards."
            )
        else:
            total_entries      = num_rounds * num_courts
            format_entry_rule  = f"  - Exactly {total_entries} entries ({num_rounds} rounds × {num_courts} courts)."
            format_sitout_rule = f"  - sitting_out has exactly {sitouts} name(s) per round."

        return f"""Generate a complete {num_rounds}-round doubles tournament schedule.

PLAYERS ({len(players)} total):
{player_lines}
{special_block}{constraint_block}{court_avail_block}{rule_block}{math_block}
STRICT PRIORITIES (satisfy in order):
  1. BALANCED MATCHUP   — FORBIDDEN: [beginner+beginner] vs [intermediate+intermediate].
  2. HARD CONSTRAINTS   — every constraint listed above must be satisfied.
  3. REQUIRED PAIRINGS  — all mandatory partner pairs must appear in the schedule.
  4. PARTNER VARIETY    — same two players should partner at most {constraints.max_partner_repeat} times.
  5. SIT-OUT BALANCE    — distribute sit-outs as evenly as possible across all players.
  6. OPPONENT VARIETY   — same two players should face each other at most {constraints.max_opponent_repeat} times.

TOURNAMENT SETTINGS:
  - Rounds              : {num_rounds}
  - Minutes per game    : {mins_per_game}
  - Courts per round    : up to {num_courts} (see COURT AVAILABILITY)
  - Sitting out/round   : varies (see above)

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

FORMAT RULES:
{format_entry_rule}
  - Only include a court entry for rounds where that court is active.
{format_sitout_rule}
  - team_a and team_b each have exactly 2 players.
  - Use exact player name spelling from the list above.
  - No player appears more than once in the same round.
"""

    def _build_refine_prompt(
        self,
        players:              List[str],
        skill_levels:         Dict[str, str],
        num_rounds:           int,
        num_courts:           int,
        special_instructions: str,
        constraints:          ConstraintSet,
        rule_summary:         str,
        previous_schedule:    List[Dict],
        violations:           List[Dict],
        clean_rounds:         Set[int],
        court_start_rounds:   Optional[Dict[int, int]] = None,
        court_start_times:    Optional[Dict[int, str]] = None,
        mins_per_game:        int = 10,
    ) -> str:
        player_lines = "\n".join(
            f"  - {p} ({skill_levels.get(p, 'unknown')})" for p in players
        )
        stripped = [
            {
                "round":       g.get("round"),
                "court":       g.get("court"),
                "team_a":      g.get("team_a", []),
                "team_b":      g.get("team_b", []),
                "sitting_out": g.get("sitting_out", []),
            }
            for g in previous_schedule
        ]
        constraint_block = ""
        if constraints.no_same_group:
            constraint_block += (
                f"\nHARD: Players {sorted(constraints.no_same_group)} must NEVER be teammates."
            )
        if constraints.girl_names:
            constraint_block += (
                f"\nHARD: Girls are {sorted(constraints.girl_names)}. "
                f"A team of 2 girls must NEVER face an all-boys team."
            )
        if constraints.required_partners:
            for req in constraints.required_partners:
                p  = req.get("players", [])
                r  = req.get("round")
                br = req.get("by_round")
                n  = req.get("min_count", 1)
                if len(p) == 2:
                    if r is not None:
                        constraint_block += f"\nREQUIRED: {p[0]} and {p[1]} MUST be partners in exactly round {r}."
                    elif br is not None and n > 1:
                        constraint_block += f"\nREQUIRED: {p[0]} and {p[1]} MUST be partners ≥{n} times, first by round {br}."
                    elif br is not None:
                        constraint_block += f"\nREQUIRED: {p[0]} and {p[1]} MUST be partners at least once within rounds 1–{br}."
                    elif n > 1:
                        constraint_block += f"\nREQUIRED: {p[0]} and {p[1]} MUST be partners at least {n} times."
                    else:
                        constraint_block += f"\nREQUIRED: {p[0]} and {p[1]} MUST be partners at least once."
        if constraints.required_opponents:
            for req in constraints.required_opponents:
                p  = req.get("players", [])
                br = req.get("by_round")
                n  = req.get("min_count", 1)
                if len(p) == 2:
                    times = f"at least {n} time{'s' if n > 1 else ''}"
                    if br is not None:
                        constraint_block += f"\nREQUIRED: {p[0]} and {p[1]} MUST face each other {times} within rounds 1–{br}."
                    else:
                        constraint_block += f"\nREQUIRED: {p[0]} and {p[1]} MUST face each other as opponents {times}."
        if rule_summary:
            constraint_block += f"\n{rule_summary}"

        # Court availability block for refine mode
        court_avail_block = ""
        staggered = court_start_rounds and any(r > 1 for r in court_start_rounds.values())
        if staggered:
            lines = []
            for c in sorted(court_start_rounds.keys()):
                r = court_start_rounds[c]
                t = (court_start_times or {}).get(c, "?")
                if r == 1:
                    lines.append(f"  Court {c}: active from round 1 ({t})")
                else:
                    lines.append(
                        f"  Court {c}: NOT active in rounds 1–{r - 1}; "
                        f"active from round {r} ({t})"
                    )
            court_avail_block = "\nCOURT AVAILABILITY:\n" + "\n".join(lines) + "\n"

        locked_block = (
            f"\nLOCKED ROUNDS — these are clean, do NOT change them: "
            f"{', '.join(str(r) for r in sorted(clean_rounds))}\n"
            if clean_rounds else ""
        )

        violation_block = ""
        if violations:
            hard = [v for v in violations if v["layer"] == "HARD"]
            soft = [v for v in violations if v["layer"] != "HARD"]
            lines = []
            if hard:
                lines.append(
                    f"HARD VIOLATIONS — must fix "
                    f"({sum(v['penalty'] for v in hard):,} penalty pts):"
                )
                for v in hard:
                    lines.append(f"  • [{v['location']}] {v['team']}: {v['detail']}")
            if soft:
                lines.append(
                    f"\nSOFT VIOLATIONS — improve if possible "
                    f"({sum(v['penalty'] for v in soft):,} penalty pts):"
                )
                for v in soft:
                    lines.append(f"  • {v['team']}: {v['detail']} (−{v['penalty']} pts)")
            violation_block = "\nCURRENT ISSUES:\n" + "\n".join(lines)

        total_entries = sum(
            num_rounds - (court_start_rounds.get(c, 1) - 1)
            for c in range(1, num_courts + 1)
        ) if staggered else num_rounds * num_courts

        return f"""You previously generated this doubles schedule. Refine it.

PREVIOUS SCHEDULE:
{json.dumps(stripped, indent=2)}

PLAYERS ({len(players)} total):
{player_lines}

INSTRUCTIONS (ALL must be respected):
{special_instructions.strip()}
{constraint_block}
{court_avail_block}{locked_block}{violation_block}

PRIORITIES:
  1. Fix all HARD violations — these make the schedule invalid.
  2. Improve soft violations — partner/opponent variety, sit-out balance.
  3. Do NOT change locked rounds — they are already clean.

TOURNAMENT SETTINGS:
  Rounds: {num_rounds}  Courts: up to {num_courts}  Minutes/game: {mins_per_game}
  Sitting out: varies by round (see COURT AVAILABILITY above)

OUTPUT — return ONLY the complete refined JSON array.
{total_entries} entries total. Same format: round, court, team_a, team_b, sitting_out.
Only include a court in a round when that court is active.
"""

    def _build_feedback_prompt(
        self,
        violations:    List[Dict],
        total_penalty: int,
    ) -> str:
        hard = [v for v in violations if v["layer"] == "HARD"]
        soft = [v for v in violations if v["layer"] != "HARD"]
        lines = [
            f"Your schedule has a total penalty of {total_penalty:,} points. "
            "Regenerate the COMPLETE schedule correcting every issue below.\n",
        ]
        if hard:
            lines.append(
                f"HARD VIOLATIONS — must fix "
                f"({sum(v['penalty'] for v in hard):,} pts):"
            )
            for v in hard:
                lines.append(f"  • [{v['location']}] {v['team']}: {v['detail']}")
        if soft:
            lines.append(
                f"\nSOFT VIOLATIONS — improve "
                f"({sum(v['penalty'] for v in soft):,} pts):"
            )
            for v in soft:
                lines.append(f"  • {v['team']}: {v['detail']} (−{v['penalty']} pts)")
        lines.append("\nOutput ONLY the corrected JSON array.")
        return "\n".join(lines)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _extract_constraints(self, special_instructions: str) -> Dict:
        """Use LLM to extract structured constraints from free-text instructions."""
        if not special_instructions or not special_instructions.strip():
            return {}
        prompt = (
            "Analyse the organiser's tournament instructions below and extract six things.\n\n"
            "1. no_same_group: list of player names who must NEVER be on the same doubles team.\n"
            "   These names come only from the instructions — do NOT invent names.\n\n"
            "2. required_partners: players who MUST be on the same team (partners) in some games.\n"
            "   This includes phrases like:\n"
            "     'Keep N games for/with/between A and B'  — they must partner at least N times\n"
            "     'A and B should play together N times'   — partner min_count=N\n"
            "     'at least N games for A and B'           — partner min_count=N\n"
            "     'A and B must partner in round 3'        — exact round=3\n"
            "     'within the first 4 rounds'              — by_round=4\n"
            "     'early / up in the order'                — by_round approx first 3-4 rounds\n"
            "   Each entry: "
            '{"players":["Name1","Name2"],"round":<int|null>,"by_round":<int|null>,"min_count":<int>}\n'
            "   Examples from realistic instructions:\n"
            "     'Keep 2 games at least for Boy 6 and Girl 4'\n"
            "       -> {\"players\":[\"Boy 6\",\"Girl 4\"],\"round\":null,\"by_round\":null,\"min_count\":2}\n"
            "     'Keep 2 games at least for Boy 6 and Girl 4. This game should be within rounds 1 to 4'\n"
            "       -> {\"players\":[\"Boy 6\",\"Girl 4\"],\"round\":null,\"by_round\":4,\"min_count\":2}\n"
            "     'Boy 6 and Girl 4 must partner in round 3'\n"
            "       -> {\"players\":[\"Boy 6\",\"Girl 4\"],\"round\":3,\"by_round\":null,\"min_count\":1}\n"
            "   Only include entries explicitly stated.\n\n"
            "3. required_opponents: players who MUST face each other on opposing teams.\n"
            "   This includes phrases like:\n"
            "     'Keep 1 game where A plays against B'    — min_count=1\n"
            "     'A must face B at least once'            — min_count=1\n"
            "     'A vs B within the first 2 rounds'       — min_count=1, by_round=2\n"
            "     'schedule A against B early'             — min_count=1, by_round approx 3\n"
            "   Each entry: "
            '{"players":["Name1","Name2"],"by_round":<int|null>,"min_count":<int>}\n'
            "   Examples:\n"
            "     'Keep 1 game where Boy 6 plays against Girl 4'\n"
            "       -> {\"players\":[\"Boy 6\",\"Girl 4\"],\"by_round\":null,\"min_count\":1}\n"
            "     'Schedule Boy 6 vs Girl 4 within the first 3 rounds'\n"
            "       -> {\"players\":[\"Boy 6\",\"Girl 4\"],\"by_round\":3,\"min_count\":1}\n"
            "   Only include entries explicitly stated.\n\n"
            "4. rule_summary: a concise Allowed/Forbidden block (2-4 lines) re-expressing any\n"
            "   remaining constraints NOT already captured in required_partners, required_opponents,\n"
            "   player_latest, or player_earliest.\n"
            "   Do NOT repeat constraints already captured in the other fields.\n\n"
            "5. player_latest: players who must NOT be scheduled in any game that starts at or after\n"
            "   a given time (i.e. they need to leave by that time).\n"
            "   Trigger phrases: 'not after', 'must leave by', 'not available after',\n"
            "                    'should not be scheduled after', 'matches should not start at or after'\n"
            "   Format: object mapping player name to time string 'HH:MM' (24-hour).\n"
            "   Examples:\n"
            "     'Boy 3 must not be scheduled after 14:00'\n"
            "       -> {\"Boy 3\": \"14:00\"}\n"
            "     'Girl 2 is not available after 15:30'\n"
            "       -> {\"Girl 2\": \"15:30\"}\n"
            "   Only include entries explicitly stated.\n\n"
            "6. player_earliest: players who must NOT be scheduled in any game that starts before\n"
            "   a given time (i.e. they arrive late or are not available early).\n"
            "   Trigger phrases: 'not before', 'arrives at', 'not available before',\n"
            "                    'should not be scheduled before', 'matches should not start before'\n"
            "   Format: object mapping player name to time string 'HH:MM' (24-hour).\n"
            "   Examples:\n"
            "     'Boy 5 should not be scheduled before 10:30'\n"
            "       -> {\"Boy 5\": \"10:30\"}\n"
            "     'Girl 1 is not available before 11:00'\n"
            "       -> {\"Girl 1\": \"11:00\"}\n"
            "   Only include entries explicitly stated.\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            '  {"no_same_group": ["Name1", ...],\n'
            '   "required_partners":  [{"players":["A","B"], "round":null, "by_round":4, "min_count":2}],\n'
            '   "required_opponents": [{"players":["A","B"], "by_round":null, "min_count":1}],\n'
            '   "rule_summary": "FORBIDDEN: ...",\n'
            '   "player_latest":   {"PlayerName": "HH:MM"},\n'
            '   "player_earliest": {"PlayerName": "HH:MM"}}\n'
            "Omit any key that is empty or not applicable. If nothing to extract, return {}.\n\n"
            f"Instructions: {special_instructions.strip()}"
        )
        try:
            response = self.model.generate_content(prompt)
            text     = response.text.strip()
            _cot(f"  [extraction raw] {text[:500]}")
            text  = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                _cot(f"  [extraction parsed] {result}")
                return result
        except Exception as exc:
            logger.warning("Could not extract constraints: %s", exc)
            _cot(f"  [extraction error] {exc}")
        return {}

    def _parse_schedule_json(self, text: str) -> Optional[List[Dict]]:
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        idx  = text.find("[")
        if idx == -1:
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, idx)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
        return None

    def _add_metadata(self, schedule: List[Dict], players: List[str]) -> List[Dict]:
        for entry in schedule:
            rnd   = entry.get("round", 1)
            court = entry.get("court", 1)

            if self._court_start_times and court in self._court_start_times:
                start_str   = self._court_start_times[court]
                sh, sm      = map(int, start_str.split(":"))
                court_abs   = sh * 60 + sm
                round_off   = self._court_start_rounds.get(court, 1) - 1
                g_start     = court_abs + (rnd - 1 - round_off) * self._mins_per_game
                g_end       = g_start + self._mins_per_game
                entry.setdefault(
                    "time_slot",
                    f"{g_start // 60:02d}:{g_start % 60:02d}–"
                    f"{g_end   // 60:02d}:{g_end   % 60:02d}",
                )
            else:
                entry.setdefault("time_slot", f"{(rnd - 1) * 10}–{rnd * 10} min")

            entry.setdefault("game_id", f"r{rnd}_c{court}")
            if "sitting_out" not in entry:
                active = set(entry.get("team_a", []) + entry.get("team_b", []))
                entry["sitting_out"] = [p for p in players if p not in active]
        return schedule
