"""
Hugo Setup
----------
Installation configuration for Hugo AI Assistant.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="hugo-ai",
    version="0.1.0",
    description="Hugo - Your Local-First Autonomous AI Assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mike McGhen",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/hugo",
    packages=find_packages(exclude=["tests", "docs", "services"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hugo=runtime.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    include_package_data=True,
    zip_safe=False,
)
