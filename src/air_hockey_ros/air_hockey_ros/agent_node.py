#!/usr/bin/env python3
from typing import Tuple, Dict, Optional
import sys
import traceback
import logging
import argparse
import pickle
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import SingleThreadedExecutor

from air_hockey_ros.msg import AgentCommand, WorldState

QUEUE_SIZE = 100

def extract_stamp_tuple(msg) -> Tuple[int, int]:
    return (msg.stamp.sec, msg.stamp.nanosec)

def world_state_to_dict(msg) -> Dict:
    sec = int(msg.stamp.sec)
    nsec = int(msg.stamp.nanosec)
    d = {
        "stamp_sec": sec,
        "stamp_nanosec": nsec,

        "team_a_score": msg.team_a_score,
        "team_b_score": msg.team_b_score,

        "puck_x": msg.puck_x,
        "puck_y": msg.puck_y,
        "puck_vx": msg.puck_vx,
        "puck_vy": msg.puck_vy,

        # ensure plain Python lists (not numpy/array.array) for the sim
        "agent_x": msg.agent_x,
        "agent_y": msg.agent_y,
        "agent_vx": msg.agent_vx,
        "agent_vy": msg.agent_vy,
    }
    return d

def get_logger(enable: bool, team: str, agent_id: int):
    name = f"agent.{team}.{agent_id}"
    logger = logging.getLogger(name)
    logger.propagate = False
    if not enable:
        logger.addHandler(logging.NullHandler())
        return logger
    logger.setLevel(logging.INFO)
    if not logger.handlers:  # avoid duplicate handlers on reload
        fh = logging.FileHandler(f"agent_{team}_{agent_id}.log")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

class AgentNode(Node):
    def __init__(self, agent_id: int, team: str, policy_path: str, log: bool):
        self.agent_id = agent_id
        super().__init__(f'agent_{agent_id}')
        try:
            with open(policy_path, 'rb') as f:
                self.policy = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load policy at '{policy_path}': {e}") from e

        self.logger = get_logger(log, team, agent_id)

        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.sub = self.create_subscription(
            WorldState,
            f'/world_update_{team}',
            self._on_world_update,
            qos
        )

        self.cmd_pub = self.create_publisher(
            AgentCommand,
            '/agent_command',
            QUEUE_SIZE
        )

        self._lock = threading.Lock()
        self._have_new = threading.Event()
        self._latest: Optional['WorldState'] = None
        self._last_processed: Optional[Tuple[int, int]] = None
        self._stop = False

        # Worker thread: latest-only, worker-when-free
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.policy.on_agent_init()

        self.logger.info("Agent node initiated.")

    def _on_world_update(self, msg: WorldState):
        # Only _latest is shared → protect it
        with self._lock:
            self._latest = msg
        self._have_new.set()

    def _worker_loop(self):
        while not self._stop:
            self._have_new.wait(timeout=0.1)
            if self._stop:
                break

            # Snapshot freshest safely
            with self._lock:
                state = self._latest
            if state is None:
                self._have_new.clear()
                continue

            stamp = extract_stamp_tuple(state)
            if self._last_processed is not None and stamp <= self._last_processed:
                self._have_new.clear()
                continue

            # Level-triggered: clear once before compute
            self._have_new.clear()

            # Re-snapshot after clear to grab the freshest if cache changed
            with self._lock:
                latest = self._latest
            if latest is None:
                continue
            if extract_stamp_tuple(latest) != stamp:
                state = latest
                stamp = extract_stamp_tuple(state)

            # Freshness re-check (out-of-order / duplicates)
            if self._last_processed is not None and stamp <= self._last_processed:
                continue

            try:
                vx, vy = self.policy.update(world_state_to_dict(state))
                cmd = AgentCommand(agent_id=self.agent_id, vx=vx, vy=vy)
                self.cmd_pub.publish(cmd)
                self._last_processed = stamp
                self.logger.info(f"vx={vx}, vy={vy}")
            except Exception as e:
                self.logger.error(f"Policy error: {e}")

    def destroy_node(self):
        self._stop = True
        self._have_new.set()  # unblock wait()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
        # Cleanly stop policy threads (idempotent)
        try:
            self.policy.on_agent_close()
        except Exception:
            pass

        super().destroy_node()

def parse_args():
    parser = argparse.ArgumentParser(description='Air Hockey Agent Node')
    parser.add_argument('--agent_id', type=int, required=True, help="Agent id, string of an integer")
    parser.add_argument('--team', required=True, choices=['a', 'b'], help="Team this agent belongs to (A/B),"
                                                                          " use lower case (a/b)")
    parser.add_argument('--policy_path', required=True, help="Path to pickled policy object")
    parser.add_argument('--log', required=False, default=False, help='log to file')
    # ros2 passes remapping args too; ignore unknowns, so we don't crash
    args, _ = parser.parse_known_args()
    if args.log.lower() == 'true':
        args.log = True
    else:
        args.log = False
    return args

def main():
    rclpy.init()
    args = parse_args()

    node = None
    try:
        node = AgentNode(team=args.team,
                         policy_path=args.policy_path,
                         agent_id=args.agent_id,
                         log=args.log)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()

    except KeyboardInterrupt:
        # normal Ctrl+C / external shutdown
        pass

    except Exception:
        # real bug: print full stacktrace to stderr and propagate non-zero exit
        traceback.print_exc()
        return 1

    finally:
        # destroy node if it existed
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        # only shutdown if the context is still active
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
    return 0

if __name__ == "__main__":
    sys.exit(main())
