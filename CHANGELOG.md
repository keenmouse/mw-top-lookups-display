# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]
### Removed
- Removed screenshot generation tooling (`tools/generate_screenshots.py`) from the project.

### Changed
- `requirements.txt` now reflects stdlib-only project dependencies.

## [0.4.0] - 2026-02-13

### Changed
- Header clarity and density improvements:
  - `Logger` is now presented as source type (`file`) in preparation for future DB/web modes.
  - Removed redundant source/status lines and consolidated poll/retention context.
  - Moved Aggregate Freshness Ratio, Entropy, and Active Unique Terms below the horizontal rule on the Main page.
- Added `P` keybind to toggle `CURRENT SNAPSHOT` section visibility in Main view (default OFF).
- Reworked window coverage reporting:
  - clearer coverage labeling
  - coverage percentage now interval-adjusted for polling cadence
  - explicit raw sample-span display
  - low-coverage warning rendered as red text when below configured threshold.
- Auto-fit (`A`) now aligns with state-evaluation logic by selecting the effective evaluable window rather than a simple span ceiling.
- Screenshot tooling updated to match new render signatures/options.

## [0.3.0] - 2026-02-13

### Changed
- State evaluation no longer resets incorrectly on window change; it now re-evaluates from current window data each cycle.
- Added `--state-min-fill` (default `0.70`) to control minimum estimated window fill for state-eval window selection.
- State evaluation now chooses the largest sufficiently filled/evaluable window and displays a `STATE EVAL:` header note when it downshifts from the requested window.
- `--clear-logs` behavior is clear-and-exit for short-session workflows.
- Expanded docs and roadmap:
  - README intro clarifies meme/news-cycle effects and state-eval behavior.
  - TODO now includes Grafana/web interface, optional DB logging, `--mode tui|webui|both`, and term-level `(research)` assist ideas.

## [0.2.0] - 2026-02-13

### Changed
- Adaptive state-threshold history is now capped to recent data by configurable CLI option:
  - `--adaptive-cap-hours` (default `24`, i.e., 1 day)
- State classification paths (runtime, logger updates, meta summary, startup mode selection) now consistently use the adaptive history cap.
- Improved screenshot generation quality and consistency:
  - ANSI color-aware rendering
  - glyph fallback for block/arrow/sparkline symbols
  - fixed line alignment to prevent overlap
  - screenshot state bootstrapping from logs to avoid `unknown (0s)` headers
- Main and Meta state presentation refinements:
  - explicit active-state duration wording in header
  - transition rows show duration on `from` state only
  - transition rows use icon-only confidence drift indicators (`▲`, `●`, `▼`)
- Transition logging behavior:
  - suppress duplicate consecutive `old_state -> new_state` transition rows
  - log `state -> state` rows on confidence label/trend changes
  - preserve state-duration continuity by ignoring confidence-only self-transitions as state-start anchors
- Replaced startup reset behavior with `--clear-logs` (clear logs and exit).
- Main view readability improvements:
  - adaptive blank-line spacing for TREND and HISTORY when enough rows fit
  - HISTORY/TREND layout threshold tuned for smaller terminals

## [0.1.0] - 2026-02-12

### Added
- Initial public release of MW Top Lookups Display.
- Live terminal dashboard with Main and Meta views.
- Built-in continuous logger with rotated CSV logs for terms, meta metrics, and state transitions.
- Adaptive time windows and keyboard-driven interaction (`M`, `S`, `V`, `+/-`, `A`, etc.).
- Ignore filtering with wildcard support via `--ignore`.
- Startup reset option via `--start-fresh`.
- System-state classification, confidence trend indicators, and transition history.
- Documentation upgrades including usage, controls, metrics, and screenshots.
- Screenshot generation tooling in `tools/generate_screenshots.py`.
- MIT license.
