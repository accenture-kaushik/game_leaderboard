import random
from typing import List, Dict, Tuple


class ScheduleService:
    """
    Generates a fair, balanced doubles schedule.

    Logic:
    - Each round: num_courts * 4 players active, remainder sit out.
    - With 10 players + 2 courts: 8 play, 2 rest per round.
    - Sit-out rotation is fairness-driven (least rest gets priority to rest next).
    - Teams balanced by skill level: pair beginner+intermediate where possible.
    - Partner variety: track history to avoid same partners too often.
    """

    def generate_schedule(
        self,
        players: List[str],
        skill_levels: Dict[str, str],
        num_rounds: int = 12,
        num_courts: int = 2,
    ) -> List[Dict]:
        n = len(players)
        players_active = num_courts * 4
        sitouts_per_round = max(0, n - players_active)

        play_count = {p: 0 for p in players}
        sitout_count = {p: 0 for p in players}
        partner_history = {p: {q: 0 for q in players} for p in players}

        schedule = []

        for round_num in range(1, num_rounds + 1):
            # --- select sit-outs --------------------------------------------------
            sitting_out = self._select_sitouts(
                players, play_count, sitout_count, sitouts_per_round
            )
            available = [p for p in players if p not in sitting_out]

            # --- create teams -----------------------------------------------------
            pairs = self._create_pairs(available, skill_levels, partner_history)

            for court_idx in range(num_courts):
                pa_idx = court_idx * 2
                pb_idx = court_idx * 2 + 1
                if pb_idx >= len(pairs):
                    break
                team_a = pairs[pa_idx]
                team_b = pairs[pb_idx]

                schedule.append(
                    {
                        "round": round_num,
                        "court": court_idx + 1,
                        "team_a": team_a,
                        "team_b": team_b,
                        "sitting_out": list(sitting_out),
                        "time_slot": f"{(round_num - 1) * 10}–{round_num * 10} min",
                        "game_id": f"r{round_num}_c{court_idx + 1}",
                    }
                )

                # update partner history
                self._update_history(partner_history, team_a, team_b)

            # update play / sitout counts
            for p in available:
                play_count[p] += 1
            for p in sitting_out:
                sitout_count[p] += 1

        return schedule

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _select_sitouts(
        self,
        players: List[str],
        play_count: Dict[str, int],
        sitout_count: Dict[str, int],
        n: int,
    ) -> List[str]:
        """Return the n players who should rest this round."""
        if n <= 0:
            return []
        # Primary sort: most games played first (they deserve rest).
        # Tie-break: fewest sit-outs first (they haven't rested yet).
        sorted_p = sorted(players, key=lambda p: (-play_count[p], sitout_count[p]))
        return sorted_p[:n]

    def _create_pairs(
        self,
        available: List[str],
        skill_levels: Dict[str, str],
        partner_history: Dict[str, Dict[str, int]],
    ) -> List[List[str]]:
        """Return list of [player, player] pairs, skill-balanced."""
        beginners = [p for p in available if skill_levels.get(p) == "beginner"]
        intermediates = [p for p in available if skill_levels.get(p) == "intermediate"]

        random.shuffle(beginners)
        random.shuffle(intermediates)

        pairs: List[List[str]] = []
        b_pool = list(beginners)
        i_pool = list(intermediates)

        # Mix skill levels: pair each beginner with an intermediate (least history first)
        while b_pool and i_pool:
            b = b_pool.pop(0)
            best_i = min(i_pool, key=lambda i: partner_history.get(b, {}).get(i, 0))
            i_pool.remove(best_i)
            pairs.append([b, best_i])

        # Pair remaining same-skill players (least history first)
        remaining = b_pool + i_pool
        while len(remaining) >= 2:
            p1 = remaining.pop(0)
            best_p2 = min(remaining, key=lambda p: partner_history.get(p1, {}).get(p, 0))
            remaining.remove(best_p2)
            pairs.append([p1, best_p2])

        return pairs

    def _update_history(
        self,
        partner_history: Dict[str, Dict[str, int]],
        team_a: List[str],
        team_b: List[str],
    ) -> None:
        for team in [team_a, team_b]:
            if len(team) >= 2:
                p1, p2 = team[0], team[1]
                partner_history.setdefault(p1, {})[p2] = (
                    partner_history.get(p1, {}).get(p2, 0) + 1
                )
                partner_history.setdefault(p2, {})[p1] = (
                    partner_history.get(p2, {}).get(p1, 0) + 1
                )
