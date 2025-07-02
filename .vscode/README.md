# VS Code Ruff Configuration

This directory contains VS Code workspace settings to ensure the Ruff extension properly detects and uses the project's configuration from `pyproject.toml`.

## Changes Made

### Migration to Native Server
- **Enabled Ruff Native Server**: Set `"ruff.nativeServer": "on"` to use the modern Rust-based language server instead of the deprecated Python-based `ruff-lsp`
- **Configuration Priority**: Set `"ruff.configurationPreference": "filesystemFirst"` to prioritize `pyproject.toml` over VS Code defaults
- **Environment Integration**: Set `"ruff.importStrategy": "fromEnvironment"` to use the project's Ruff installation

### Disabled Conflicting Linters
- Disabled Python's built-in linting to prevent conflicts with Ruff
- Disabled pylint, flake8, and mypy linting extensions to avoid duplication

## Expected Behavior

After applying these settings, VS Code's Ruff extension should:

1. ✅ **Detect project configuration**: No more "No workspace options found" messages
2. ✅ **Use consistent settings**: Same linting/formatting rules as CLI (`ruff check`, `ruff format`)
3. ✅ **Apply project rules**: Line length 88, target Python 3.8+, comprehensive rule set
4. ✅ **Format on save**: Automatic code formatting with project settings
5. ✅ **Organize imports**: Automatic import sorting using isort configuration

## Testing the Configuration

1. **Restart VS Code** after applying these settings
2. **Open a Python file** in the project (e.g., `ffsubsync/__init__.py`)
3. **Check the status bar** - should show Ruff as the active linter/formatter
4. **Test formatting**: Make a formatting change and save - should apply project rules
5. **Check VS Code logs**: Go to Output > Ruff - should show successful configuration detection

## Troubleshooting

If issues persist:

1. **Check Ruff version**: Ensure Ruff v0.5.3+ is installed (`ruff --version`)
2. **Verify extension**: Ensure VS Code Ruff extension is v2024.32.0+
3. **Check logs**: View Output > Ruff for detailed server information
4. **Restart extension**: Use Command Palette > "Developer: Reload Window"

## Configuration Files

- `.vscode/settings.json`: VS Code workspace settings for Ruff
- `pyproject.toml`: Project-wide Ruff configuration (main source of truth)
- `.pre-commit-config.yaml`: Pre-commit hooks using same Ruff configuration
