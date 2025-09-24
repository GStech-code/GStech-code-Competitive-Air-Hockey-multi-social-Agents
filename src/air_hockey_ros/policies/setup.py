from setuptools import setup, find_packages
setup(
    name="policies",
    version="0.0.1",
    packages=find_packages(include=["policies", "policies.*"]),
)
