# Contributing to PatternSense

Thank you for your interest in contributing to PatternSense! This document provides guidelines and instructions for contributing to the project.

## Development Environment Setup

### Prerequisites

- Python 3.12 or higher
- uv (Python package manager)

### Setting Up Your Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/prolificbrain/PatternSense.git
   cd PatternSense
   ```

2. Create and activate a virtual environment using uv:
   ```bash
   uv venv .venv --python 3.12
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

## Development Workflow

### Branching Strategy

- `main` - Stable release branch
- `develop` - Development branch for integration
- `feature/*` - Feature branches
- `fix/*` - Bug fix branches

### Creating a New Feature or Fix

1. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes, following the coding standards below.

3. Write tests for your changes.

4. Run tests to ensure everything works:
   ```bash
   python -m unittest discover tests
   ```

5. Submit a pull request to the `develop` branch.

## Coding Standards

### Python Style Guide

- Follow PEP 8 style guide for Python code.
- Use 4 spaces for indentation (no tabs).
- Maximum line length of 88 characters.
- Use docstrings for all public modules, functions, classes, and methods.

### Documentation

- Use Google-style docstrings for code documentation.
- Update README.md and other documentation when adding new features.
- Include examples in docstrings when appropriate.

### Testing

- Write unit tests for all new functionality.
- Ensure all tests pass before submitting a pull request.
- Aim for high test coverage for critical components.

## Package Management

- Use uv for Python package management.
- Update uv.lock when adding new dependencies.
- Specify minimum version requirements in setup.py.

## Pull Request Process

1. Ensure your code follows the coding standards.
2. Update documentation as necessary.
3. Include tests for new functionality.
4. Update the CHANGELOG.md with details of changes.
5. The PR will be reviewed by maintainers and merged if approved.

## License

By contributing to PatternSense, you agree that your contributions will be licensed under the project's MIT License.
