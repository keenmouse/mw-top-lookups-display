# TODO

## Cross-Platform Compatibility

- Add POSIX non-blocking keyboard input support for interactive controls (`termios`/`tty`/`select`).
- Make default log directory platform-native:
  - Linux: XDG (`$XDG_STATE_HOME` or `$XDG_DATA_HOME` fallback)
  - macOS: `~/Library/Application Support/MWTopLookups/logs`
  - Windows: keep `%LOCALAPPDATA%\\MWTopLookups\\logs`
- Add optional `--portable` mode that stores logs under the working directory (for example `./logs`) without changing the default behavior.
- Make screenshot font selection platform-aware (avoid Windows-only font assumptions in `tools/generate_screenshots.py`).

## Next Platform

- Add a Grafana dashboard web interface while retaining the TUI.
- Add optional database logging backend while retaining file-based CSV logging.
