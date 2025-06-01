# setup.py
import os
from setuptools import setup, find_packages


def load_requirements(fname="requirements.txt"):
    here = os.path.abspath(os.path.dirname(__file__))
    req_path = os.path.join(here, fname)

    with open(req_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    reqs = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    return reqs


setup(
    name="quantum_watermarking",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=load_requirements("requirements.txt"),
)
