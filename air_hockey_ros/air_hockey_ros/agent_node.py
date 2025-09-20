#!/usr/bin/env python3
from typing import Tuple, Dict
import sys
import traceback
import logging
import argparse
import pickle

import rclpy
from rclpy.node import Node
from air_hockey_ros.msg import AgentCommand, WorldState

QUEUE_SIZE = 100

def world_state_to_dict(msg) -> Dict:
    sec = int(msg.stamp.sec)
    nsec = int(msg.stamp.nanosec)
    d = {
        "stamp_sec": sec,
        "stamp_nanosec": nsec,
        "stamp_ms": sec * 1000.0 + nsec / 1e6,

        "puck_x": float(msg.puck_x),
        "puck_y": float(msg.puck_y),
        "puck_vx": float(msg.puck_vx),
        "puck_vy": float(msg.puck_vy),

        # ensure plain Python lists (not numpy/array.array) for the sim
        "agent_x": msg.agent_x,
        "agent_y": msg.agent_y,
        "agent_vx": msg.agent_vx,
        "agent_vy": msg.agent_vy,
    }
    return d


def get_logger(enable: bool, team: str, agent_id: int):
    if not enable:
        # logging disabled → use a dummy logger that ignores everything
        logger = logging.getLogger(f"agent.{team}.{agent_id}")
        logger.addHandler(logging.NullHandler())
        return logger

    logfile = f"agent_{team}_{agent_id}.log"
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(f"agent.{team}.{agent_id}")


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

        self.sub = self.create_subscription(
            WorldState,
            f'/world_update_{team}',
            self._on_world_update,
            QUEUE_SIZE
        )

        self.cmd_pub = self.create_publisher(
            AgentCommand,
            '/agent_command',
            QUEUE_SIZE
        )

    def _on_world_update(self, msg: WorldState):
        try:
            world = world_state_to_dict(msg)
        except Exception as e:
            self.logger.error(f'Failed converting WorldState to dict: {e}')
            return

        try:
            vx, vy = self.policy.update(world)
            cmd = AgentCommand(agent_id=self.agent_id, vx=vx, vy=vy)
            self.cmd_pub.publish(cmd)
            self.logger.info(f"vx={vx}, vy={vy}")
        except Exception as e:
            self.logger.error(f'Error: {e}')

def parse_args():
    parser = argparse.ArgumentParser(description='Air Hockey Agent Node')
    parser.add_argument('--agent_id', type=int, required=True, help="Agent id, string of an integer")
    parser.add_argument('--team', required=True, choices=['a', 'b'], help="Team this agent belongs to (A/B),"
                                                                          " use lower case (a/b)")
    parser.add_argument('--policy_path', required=True, help="Path to pickled policy object")
    parser.add_argument('--log', required=False, default=False, help='stores if to log to file',)
    # ros2 passes remapping args too; ignore unknowns so we don't crash
    args, _ = parser.parse_known_args()
    return args

def main():
    rclpy.init()
    args = parse_args()

    node = None
    try:
        node = AgentNode(team=args.team,
                         policy_path=args.policy_path,
                         agent_id=int(args.agent_id),
                         log=bool(args.log))
        rclpy.spin(node)

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
