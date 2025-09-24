from setuptools import setup, find_packages

setup(
    name="air_hockey_ros",
    version="0.0.1",
    packages=find_packages(include=["air_hockey_ros", "air_hockey_ros.*"]),
    entry_points={
        "console_scripts": [
            "game_manager_node = air_hockey_ros.game_manager_node:main",
            "agent_node = air_hockey_ros.agent_node:main",
        ],
    },
    install_requires=[],
)
