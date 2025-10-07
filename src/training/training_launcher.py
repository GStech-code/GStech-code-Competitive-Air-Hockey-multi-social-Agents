import json
import argparse
from pathlib import Path
import yaml
import os
import sys
from ros_mock import RosMock
from trainer_plain_rosmock import train
from training_orchestrator import TrainingOrchestrator

# add air_hockey_ros sibling to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from air_hockey_ros import BaseSimulation, NeuralTeamPolicy, SimpleTeamPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sim = BaseSimulation()
    ros_mock = RosMock(sim)

    # map names to policy classes
    policy_classes = {
        "neural": NeuralTeamPolicy,
        "simple": SimpleTeamPolicy,
    }

    # orchestrator handles scenario rotation, reward shaping, etc.
    orchestrator = TrainingOrchestrator(config=config, policy_classes=policy_classes, ros_mock=ros_mock)

    train(ros_mock, orchestrator, config)
    ros_mock.close()


if __name__ == "__main__":
    main()
