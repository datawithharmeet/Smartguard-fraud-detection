from setuptools import find_packages,setup
from typing import List

def get_requirements() -> List[str]:
    """
    Returns a list of required packages from requirements.txt
    """
    try:
        with open('requirements.txt', 'r') as file:
            return [
                line.strip()
                for line in file.readlines()
                if line.strip() and not line.strip().startswith("-e")
            ]
    except FileNotFoundError:
        print("requirements.txt not found.")
        return []

setup(
    name="SMARTGAURDFRAUDDETECTION",
    version="0.0.1",
    author="Harmeet Singh",
    author_email="harmeetsingh5906@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)