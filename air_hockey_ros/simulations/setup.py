from setuptools import setup, find_packages
setup(
    name="simulations",
    version="0.0.1",
    packages=find_packages(include=["simulations", "simulations.*"]),
)
