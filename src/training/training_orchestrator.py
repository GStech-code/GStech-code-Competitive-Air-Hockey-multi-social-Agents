from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Type
import itertools


class TrainingOrchestrator:
    """
    Thin mediator between RosMock and the trainer/env.
    - Chooses which *game* (scenario) to run next
    - Provides team sizes, team policy classes, and rules per game
    - Provides reward shaping weights (kept simple)

    Config schema (minimal):
      episodes: int
      steps_per_episode: int
      goal_limit: int
      output_dir: str
      defaults:
        num_agents_team_a: int
        num_agents_team_b: int
        team_a: "neural" | "simple"
        team_b: "neural" | "simple"
        rules: { width, height, puck_radius, paddle_radius, friction_per_tick, puck_max_speed }
        rewards: { score_delta: float, dist_to_puck_penalty: float }
      games:   # optional list; each overrides any defaults fields it mentions
        - name: "g1"
          num_agents_team_a: 2
          num_agents_team_b: 2
          team_a: "neural"
          team_b: "simple"
          rules: {...}
          rewards: {...}
    """

    def __init__(self, config: Dict[str, Any], policy_classes: Dict[str, Type], ros_mock):
        self.cfg = config or {}
        self.policy_classes = policy_classes
        self.ros_mock = ros_mock

        self.defaults = self.cfg.get("defaults", {})
        self.games: List[Dict[str, Any]] = list(self.cfg.get("games", []))

        # If no explicit games, synthesize one from defaults (least confusing)
        if not self.games:
            self.games = [
                {
                    "name": "default",
                    **{k: v for k, v in self.defaults.items() if k in {
                        "num_agents_team_a", "num_agents_team_b", "team_a", "team_b", "rules", "rewards"
                    }},
                }
            ]

        # Round-robin iterator over games
        self._cycler = itertools.cycle(self.games)

        # Global session settings
        self.steps_per_episode = int(self.cfg.get("steps_per_episode", 2000))
        self.goal_limit = int(self.cfg.get("goal_limit", 10))

    def get_team_policy_class(self, name: str):
        return self.policy_classes[name]
    # -----------------
    # Game selection
    # -----------------
    def next_game(self) -> Dict[str, Any]:
        """Return a dict describing the next game's full config, merged with defaults."""
        game = next(self._cycler).copy()

        # Merge with defaults shallowly
        merged: Dict[str, Any] = {
            "num_agents_team_a": game.get("num_agents_team_a", self.defaults.get("num_agents_team_a", 2)),
            "num_agents_team_b": game.get("num_agents_team_b", self.defaults.get("num_agents_team_b", 2)),
            "team_a": game.get("team_a", self.defaults.get("team_a", "neural")),
            "team_b": game.get("team_b", self.defaults.get("team_b", "simple")),
            "rules": {**self.defaults.get("rules", {}), **game.get("rules", {})},
            "rewards": {**self.defaults.get("rewards", {}), **game.get("rewards", {})},
        }

        # Map aliases to actual classes (no global registration)
        try:
            merged["team_a_policy_class"] = self.policy_classes[merged["team_a"]]
            merged["team_b_policy_class"] = self.policy_classes[merged["team_b"]]
        except KeyError as e:
            raise ValueError(f"Unknown team policy alias: {e}")

        # Attach episode limits (can be overridden per game later if desired)
        merged["steps_per_episode"] = int(self.cfg.get("steps_per_episode", 2000))
        merged["goal_limit"] = int(self.cfg.get("goal_limit", 10))
        return merged

    # -----------------
    # Reward weights
    # -----------------
    def reward_weights_for(self, game: Dict[str, Any]) -> Dict[str, float]:
        r = game.get("rewards", {})
        return {
            "score_delta": float(r.get("score_delta", 1.0)),
            "dist_to_puck_penalty": float(r.get("dist_to_puck_penalty", 0.0)),
        }
