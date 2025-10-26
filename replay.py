# replay.py
from __future__ import annotations

import os
import re
import ast
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional

# --- Optional ROS share lookup (won't hard fail if ament isn't available) ---
def _get_share_dir(pkg: str) -> Optional[Path]:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory(pkg))
    except Exception:
        return None

# --- Flexible import for PygameView (project structures vary slightly) ---
def _import_pygame_view():
    candidates = [
        "air_hockey_ros.pygame_view:PygameView",
        "air_hockey_ros.PygameView:PygameView",
        "src.air_hockey_ros.pygame_view:PygameView",
        "src.air_hockey_ros:PygameView",
        ".src.air_hockey_ros:PygameView",
    ]
    last_err = None
    for cand in candidates:
        mod, _, attr = cand.partition(":")
        try:
            module = __import__(mod, fromlist=[attr])
            return getattr(module, attr)
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import PygameView. Tried: {candidates}\nLast error: {last_err}")

PygameView = _import_pygame_view()

# --- Scenario helpers ---
def _flatten_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten_dict(v))
        else:
            out[k] = v
    return out

def _resolve_scenario_path(user_path: Optional[str]) -> Optional[Path]:
    """
    Resolution order:
      1) explicit --scenario if provided
      2) CWD relative fallback: ./src/air_hockey_ros/game_scenarios/default_scenario.yaml
      3) ROS share path if available: <share>/src/air_hockey_ros/game_scenarios/default_scenario.yaml
    """
    if user_path:
        p = Path(user_path)
        if p.is_file():
            return p

    cwd_default = Path.cwd() / "src" / "air_hockey_ros" / "game_scenarios" / "default_scenario.yaml"
    if cwd_default.is_file():
        return cwd_default

    share_dir = _get_share_dir("air_hockey_ros")
    if share_dir:
        share_default = share_dir / "src" / "air_hockey_ros" / "game_scenarios" / "default_scenario.yaml"
        if share_default.is_file():
            return share_default

    return None  # scenario is optional for replay; we can still run with CLI defaults

# --- Log parsing ---
_START_RE = re.compile(
    r"Starting game:\s*.+?-\s*.+?:\s*(\d+)\s+vs\s+.+?:\s*(\d+)"
)
_DICT_RE = re.compile(r"\{.*\}")  # greedy; we take first {...} per line

def parse_log(log_path: Path) -> Tuple[Optional[int], Optional[int], List[Dict]]:
    """
    Returns (num_agents_team_a, num_agents_team_b, frames)
    Frames is a list of dicts in the exact shape expected by the viewer.
    """
    a_count = b_count = None
    frames: List[Dict] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # team sizes
            if a_count is None or b_count is None:
                m = _START_RE.search(line)
                if m:
                    a_count = int(m.group(1))
                    b_count = int(m.group(2))
                    continue

            # frame dicts
            m2 = _DICT_RE.search(line)
            if m2:
                try:
                    payload = ast.literal_eval(m2.group(0))  # safe parse of Python literal
                    if isinstance(payload, dict):
                        frames.append(payload)
                except Exception:
                    # ignore malformed lines
                    pass

    return a_count, b_count, frames

# --- Replay ---
def play(frames: Iterable[Dict], *, width: int, height: int, title: str, hz: int, **params):
    view = PygameView()
    view.reset(width=width, height=height, title=title, hz=hz, **params)
    try:
        for ws in frames:
            # PygameView.draw should return False to stop; if it doesn't, we just keep going
            if not view.draw(ws):
                break
            view.tick()
    finally:
        view.close()

# --- CLI / main ---
def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Replay Air Hockey game from log.")
    parser.add_argument("--log", default=str(Path("game_logs") / "game.log"), help="Path to game.log")
    parser.add_argument("--scenario", default=None, help="Path to scenario YAML (optional)")
    parser.add_argument("--width", type=int, default=None, help="Override width")
    parser.add_argument("--height", type=int, default=None, help="Override height")
    parser.add_argument("--hz", type=int, default=None, help="Override tick rate (Hz)")
    parser.add_argument("--title", default=None, help="Window title")
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"[replay] Log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    # Parse log
    num_a, num_b, frames = parse_log(log_path)
    if not frames:
        print("[replay] No frames parsed from log.", file=sys.stderr)
        sys.exit(2)

    # Scenario (optional)
    scenario_path = _resolve_scenario_path(args.scenario)
    params = {}
    if scenario_path and scenario_path.is_file():
        with scenario_path.open("r", encoding="utf-8") as f:
            scn = yaml.safe_load(f) or {}
        scn.pop("name", None)
        params.update(_flatten_dict(scn))

    # Inject team sizes (from log wins over scenario)
    if num_a is not None:
        params["num_agents_team_a"] = num_a
    if num_b is not None:
        params["num_agents_team_b"] = num_b

    # Visual defaults (scenario values if present, else CLI, else hardcoded)
    if "width" not in params and "width" in args:
        params["width"] = args.width
    if "height" not in params and "height" in args:
        params["height"] = args.height
    if "hz" in args:
        params["hz"] = args.hz
    if "title" in args:
        params["title"] = args.title

    # Run
    play(frames, **params)

if __name__ == "__main__":
    main()
