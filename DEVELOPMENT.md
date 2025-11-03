# Development Workflow Guide

This document outlines the professional development workflow for this project to ensure code quality and prevent CI/CD failures.

## Quick Start

1. **Install pre-commit hooks** (one-time setup):
   ```bash
   pip install pre-commit
   # Install both commit and push hooks so Ruff runs locally before push
   pre-commit install --hook-type pre-commit --hook-type pre-push
   ```

2. **Before making changes**, ensure your environment is ready:
   ```bash
   # Install development dependencies
   pip install -r requirements-dev.txt

   # Or install the package in development mode with dev dependencies
   pip install -e ".[dev]"
   ```

3. **Make your changes** and commit normally. Pre-commit hooks will automatically run.

## Code Quality Standards

This project uses **Ruff** for both linting and formatting, configured in `pyproject.toml`. The same rules that run in CI/CD also run locally via pre-commit hooks.

### Ruff Configuration

- **Line length**: 88 characters (Black-compatible)
- **Target Python**: 3.8+ (matches project requirements)
- **Enabled rules**: pycodestyle, Pyflakes, isort, flake8-bugbear, comprehensions, pyupgrade, simplify, and Ruff-specific rules
- **Import sorting**: Configured to be compatible with Black formatting

## Development Workflow

### 1. Pre-Commit Hooks (Automatic)

When you run `git commit`, the following checks run automatically:

- **Ruff linting**: Catches code quality issues and auto-fixes many of them
- **Ruff formatting**: Ensures consistent code formatting
- **File checks**: Large files, merge conflicts, YAML/TOML syntax
- **Cleanup**: Trailing whitespace, end-of-file newlines
- **Security**: Detects accidentally committed private keys

### 2. Manual Code Quality Checks

You can run these commands manually at any time:

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run the same checks that CI runs on push (repo-wide)
pre-commit run --all-files --hook-stage push

# Run only Ruff linting
ruff check .

# Run Ruff linting with auto-fix
ruff check . --fix

# Check code formatting
ruff format --check .

# Auto-format code
ruff format .
```

### 3. Testing

Before pushing changes, run the test suite:

```bash
# Run unit tests only
pytest -v -m 'not integration' tests/

# Run all tests (requires test data)
pytest -v tests/
```

## Troubleshooting Common Issues

### Import Sorting Errors (I001)

If you get import sorting errors:

```bash
# Auto-fix import sorting
ruff check . --fix

# Or format the specific file
ruff check path/to/file.py --fix
```

### Pre-commit Hook Failures

If pre-commit hooks fail:

1. **Review the output** - hooks often auto-fix issues
2. **Stage the auto-fixed files**: `git add .`
3. **Commit again**: `git commit -m "Your message"`

### Bypassing Hooks (Not Recommended)

Only in emergency situations:

```bash
# Skip pre-commit hooks (NOT RECOMMENDED)
git commit --no-verify -m "Emergency commit"
```

## CI/CD Pipeline

The GitHub Actions pipeline runs the same checks as your local pre-commit hooks:

1. **Code Quality** (pre-commit: Ruff linting and formatting)
2. **pipx Installation Testing** (Linux/macOS, Python 3.9-3.13)
3. **Unit Tests** (Linux/macOS, Python 3.9-3.13)
4. **Integration Tests** (Ubuntu only, Python 3.10-3.11)

### Pipeline Requirements

- All code quality checks must pass before other jobs run
- Tests run on Linux and macOS only (no Windows support)
- Package must be installable via pipx
- All CLI entry points must be functional

## Best Practices

### Before Committing

1. **Run tests locally**: `pytest -v -m 'not integration' tests/`
2. **Check your changes**: `git diff --staged`
3. **Commit with descriptive messages**: Follow conventional commit format if possible

### Code Style

- **Follow PEP 8** (enforced by Ruff)
- **Use type hints** where appropriate
- **Write docstrings** for public functions and classes
- **Keep functions focused** and reasonably sized

### Dependencies

- **Use package managers** for dependency management (pip, not manual editing)
- **Pin versions** in requirements.txt for reproducibility
- **Test pipx compatibility** for CLI tools

## Getting Help

If you encounter issues with the development workflow:

1. **Check this document** for common solutions
2. **Review CI/CD logs** on GitHub for specific error messages
3. **Run pre-commit hooks manually** to debug issues locally
4. **Check Ruff documentation** for rule-specific guidance

## Configuration Files

- **`.pre-commit-config.yaml`**: Pre-commit hook configuration
- **`pyproject.toml`**: Ruff configuration and project metadata
- **`.github/workflows/ci.yml`**: CI/CD pipeline configuration
- **`requirements-dev.txt`**: Development dependencies
