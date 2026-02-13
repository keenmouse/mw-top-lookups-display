# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

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
- Added startup reset flag `--start-fresh` to clear logs at launch.
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
