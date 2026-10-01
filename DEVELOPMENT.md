# Development Workflow Guide

This project uses [uv](https://docs.astral.sh/uv/) for environments, dependencies, builds
and installation. The setup follows the
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv) template, adapted to this
repository: the flat `ffsubsync/` package layout is kept, and the project is not managed by
Copier.

## Quick Start

1. **Install uv** (0.12.x, see `required-version` in `uv.toml`):
   <https://docs.astral.sh/uv/getting-started/installation/>. uv downloads a matching
   Python (3.11 or later) when necessary.

2. **Install the project and all development dependencies** into `.venv`:
   ```bash
   make install
   # same as: uv sync --all-groups --extra tenvad-onnx
   ```

3. **Run the checks**:
   ```bash
   make lint    # codespell, ruff check --fix, ruff format
   make test    # unit tests
   ```

4. **Optional: install the git hooks**:
   ```bash
   uvx pre-commit install --hook-type pre-commit --hook-type pre-push
   ```

`ffmpeg` and `ffprobe` must be on `PATH` for real synchronization runs and for the
integration tests.

## Make Targets

The Makefile is a thin wrapper around uv. It sets `UV_CONFIG_FILE` to the checked-in
`uv.toml`, so user-level uv settings do not change `uv.lock`.

| Target | What it does |
|--------|--------------|
| `make install` | `uv sync` with all dependency groups and the `tenvad-onnx` extra |
| `make lint` | codespell, `ruff check --fix`, `ruff format` (changes files) |
| `make lint-check` | The same checks without changes. CI runs this. |
| `make typecheck` | `basedpyright ffsubsync`. Not a CI gate; the code is only partially typed. |
| `make test` | Unit tests (`pytest -m 'not integration'`) |
| `make test-integration` | Integration tests (`INTEGRATION=1`, requires test data) |
| `make upgrade` | Upgrade all locked dependencies and sync |
| `make build` | Build the sdist and wheel into `dist/` |
| `make clean` | Remove build output, caches and `.venv` |

You can also call the tools directly:

```bash
uv run pytest -q tests/test_ssync.py
uv run ruff check .
uv run ruff format --check .
uv run ssync --help
```

## Dependencies

- Runtime dependencies and extras are in `[project]` in `pyproject.toml`.
- Development tools are in `[dependency-groups] dev`. Add one with `uv add --dev <name>`.
- `uv.lock` is committed. CI installs with `uv sync --locked` and fails when the lock
  file does not match `pyproject.toml`. After you change dependencies, run `make install`
  and commit `uv.lock`.
- `uv.toml` sets `exclude-newer = "14 days"`: uv ignores releases younger than 14 days.
  This is a supply-chain cool-off period.
- The `tenvad` extra is a git dependency without Linux ARM64 support, so `make install`
  does not install it. Use `uv sync --all-groups --extra tenvad` on Linux x64 or macOS.

## Code Quality Standards

**Ruff** does linting and formatting. The configuration is in `pyproject.toml`.

- **Line length**: 88 characters
- **Target Python**: 3.11+
- **Enabled rules**: pycodestyle, Pyflakes, isort, flake8-bugbear, comprehensions, pyupgrade, simplify, and Ruff-specific rules

**codespell** checks spelling in the source, the tests and `README.md`.

### Import Sorting Errors (I001)

```bash
uv run ruff check . --fix
```

### Pre-commit Hooks

The hooks in `.pre-commit-config.yaml` are optional. They run Ruff plus some file
checks (large files, merge conflicts, YAML/TOML syntax, trailing whitespace, private
keys). If a hook fixes a file, stage the file and commit again.

## Versioning and Builds

The build backend is hatchling. The version comes from git tags through
[uv-dynamic-versioning](https://github.com/ninoseki/uv-dynamic-versioning/). There is no
version file to edit.

- A tagged commit builds as that version, for example tag `0.4.30` gives `0.4.30`.
- Other commits build as a development version of the next patch release, for example
  `0.4.30.dev63+aad5757`.
- To release, create a tag (`git tag 0.4.30`) and push it.
- Git must be able to see the tags. Use a full clone (`fetch-depth: 0` in CI).

This fork is installed from git and is not published to PyPI. The
`Private :: Do Not Upload` classifier makes PyPI reject an accidental upload.

## CI/CD Pipeline

`.github/workflows/ci.yml` calls uv directly:

1. **Code Quality**: `uv run python devtools/lint.py --check`
2. **uv tool Installation Test** (Linux/macOS, Python 3.11-3.14): builds the wheel,
   installs it with `uv tool install`, and runs `--help` for all five commands
3. **Unit Tests** (Linux x64, Linux ARM64, macOS, Python 3.11-3.14)
4. **Integration Tests** (Ubuntu only, Python 3.11-3.12)

Python 3.14 jobs report their result without failing the workflow. The actions are
pinned to full commit SHAs. Change the SHA and the version comment together.

## Configuration Files

- **`pyproject.toml`**: project metadata, dependencies, build and tool configuration
- **`uv.toml`**: uv version range and resolution policy
- **`uv.lock`**: locked dependency versions
- **`Makefile`**: development shortcuts
- **`devtools/lint.py`**: lint runner used by `make lint` and CI
- **`.pre-commit-config.yaml`**: optional git hooks
- **`.github/workflows/ci.yml`**: CI pipeline
