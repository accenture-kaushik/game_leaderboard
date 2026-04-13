from typing import List, Dict


class LeaderboardService:
    """Calculates per-player statistics from submitted game scores."""

    def calculate_leaderboard(
        self, schedule: List[Dict], scores: Dict[str, Dict]
    ) -> List[Dict]:
        player_stats: Dict[str, Dict] = {}

        for game in schedule:
            game_id = game["game_id"]
            score_data = scores.get(game_id, {})

            if not score_data.get("submitted"):
                continue

            score_a = int(score_data["score_a"])
            score_b = int(score_data["score_b"])
            team_a: List[str] = game["team_a"]
            team_b: List[str] = game["team_b"]

            self._update_players(player_stats, team_a, score_a, score_b)
            self._update_players(player_stats, team_b, score_b, score_a)

        # Compute derived fields and sort
        for stat in player_stats.values():
            stat["net_points"] = stat["points_gained"] - stat["points_conceded"]
            stat["win_rate"] = (
                round(stat["games_won"] / stat["games_played"] * 100, 1)
                if stat["games_played"] > 0
                else 0.0
            )

        leaderboard = sorted(
            player_stats.values(),
            key=lambda x: (-x["games_won"], -x["net_points"], -x["points_gained"]),
        )

        for rank, player in enumerate(leaderboard, start=1):
            player["rank"] = rank

        return leaderboard

    # -------------------------------------------------------------------------

    def _update_players(
        self,
        stats: Dict[str, Dict],
        team: List[str],
        for_score: int,
        against_score: int,
    ) -> None:
        for player in team:
            if player not in stats:
                stats[player] = {
                    "name": player,
                    "games_played": 0,
                    "games_won": 0,
                    "games_lost": 0,
                    "points_gained": 0,
                    "points_conceded": 0,
                }
            s = stats[player]
            s["games_played"] += 1
            s["points_gained"] += for_score
            s["points_conceded"] += against_score
            if for_score > against_score:
                s["games_won"] += 1
            else:
                s["games_lost"] += 1
