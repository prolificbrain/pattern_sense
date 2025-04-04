"""Setup script for PatternSense."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="patternsense",
    version="0.2.0",
    author="Prolific Brain",
    author_email="research@ntwrkd.xyz",
    description="Advanced pattern recognition and cognitive processing framework built on trinary logic",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prolificbrain/pattern_sense",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "seaborn>=0.11.0",
    ],
)
