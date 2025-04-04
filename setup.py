"""Setup script for PatternSense."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="patternsense",
    version="0.2.0",
    author="Prolific Brain",
    author_email="research@ntwrkd.xyz",
    description="Advanced pattern recognition and cognitive processing framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prolificbrain/PatternSense",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "seaborn",
    ],
)
