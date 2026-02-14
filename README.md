# MW Top Lookups Display

MW Top Lookups Display is a terminal dashboard plus built-in logger for tracking Merriam-Webster Top Lookups over time.

It continuously polls the lookup endpoint, writes rotated CSV logs, computes trend and meta metrics, classifies system state, and renders a live TUI for short- and long-horizon analysis.

It is designed to make meme and news-cycle effects visible in near real time, including surge/decay behavior, narrative convergence, novelty bursts, and attention turnover across terms.

## What It Does

- Polls Merriam-Webster Top Lookups on an interval (default 31s).
- Stores terms, meta metrics, and state transitions as daily rotated CSV files.
- Renders two pages in the terminal.
- Main page focuses on term-level trends and history.
- Meta page focuses on system diagnostics, anomaly flags, and state changes.
- Supports ignore filters with wildcard patterns (for recurring sticky/control terms).
- Supports retention cleanup and window presets up to retention horizon.

## Runtime Behavior

- The app logs continuously while it is running.
- Running with no arguments launches immediately using defaults.

## Requirements

- Windows terminal with at least 80 columns x 24 rows.
- Python 3.10+ recommended.
- No third-party Python packages required (stdlib only).
- Network access to the endpoint.

## Quick Start

1. Open PowerShell.
2. Run:

```powershell
python MWTopLookupsDisplay.py
```

3. Use `Q` to quit.

## Screenshots

Main view:

![Main view (compact, rank-weighted)](docs/images/main-view.png?v=20260213-2)

Main view (expanded, momentum):

![Main view (expanded, momentum)](docs/images/main-view-expanded-momentum.png?v=20260213-2)

Meta view:

![Meta view](docs/images/meta-view.png?v=20260213-2)

## Command Line Usage

```powershell
python MWTopLookupsDisplay.py [options]
```

Show help:

```powershell
python MWTopLookupsDisplay.py -h
python MWTopLookupsDisplay.py --help
```

### Key options

- `--log-dir <path>`
Default base directory for logs. Default is `%LOCALAPPDATA%/MWTopLookups/logs`.

- `--retain-days <int>`
Daily log retention horizon. Also controls max selectable time window.

- `--refresh <seconds>`
UI refresh interval.

- `--log-interval <seconds>`
Polling interval for the built-in logger.

- `--adaptive-cap-hours <hours>`
Cap for adaptive state-threshold history. Default is `24` (1 day).

- `--state-min-fill <ratio>`
Minimum estimated fill ratio for state-eval window selection. Default is `0.70`.

- `--ignore <pattern>`
Ignore term or wildcard pattern. Repeatable.
Example: `--ignore "chargo*" --ignore "wordle"`

- `--spark-scale per-term|global|global-sqrt`
Initial sparkline scaling mode. Default is `global-sqrt`.

- `--color auto|on|off`
ANSI color mode.

- `--once`
Render once and exit.

- `--clear-logs`
Clear terms/meta/state logs and exit.

## Keyboard Controls

- `+` / `-`: Increase or decrease selected time window.
- `A`: Auto-select the smallest window that includes all available data.
- `V`: Toggle Main/Meta page.
- `F`: Toggle `FilterIgnored`.
- `N`: Toggle normalization (only when `FilterIgnored` is ON).
- `M`: Next metric.
Main page: cycles chart metric.
Meta page: cycles focused metric-help target and pointer.
- `S`: Cycle sparkline scaling (Main page history).
- `C`: Toggle compact header.
- `Q`: Quit.

## Main Page

- `TREND (<window>)` shows the selected metric bars.
- `Metric (M: next): ...` shows metric meaning and interaction cue.
- `HISTORY (<window>), Scaling (S: next): ...` shows per-term sparklines.
- Sparklines include gap markers for missing log intervals (shown in the legend as `░ gap (missing logs)`).
- If terminal height allows at least 7 paired term rows, both TREND and HISTORY switch to spaced layout (blank line above the first term and between term rows) for readability.
- In compact mode, non-essential header lines are hidden.

### Main metric modes (`M`)

- `Rank-Weighted Activity`: higher weight for higher-ranked terms over the selected window.
- `Presence`: how often a term appears in snapshots in the selected window.
- `Momentum`: net change in rank-weighted activity between adjacent windows (up/down movement).
- `Freshness Ratio`: term presence ratio `(10m/60m)` at sample time.

## Meta Page

- `META SNAPSHOT` lists key diagnostics (entropy, concentration, novelty, turnover, freshness, active terms).
- Pointer highlights the currently selected meta-help metric.
- `Metric (M: next): ...` explains the focused metric.
- Includes anomaly flags, summary interpretation, and recent state transitions.

### Meta metric focus (`M`)

- `M` does not change the page layout.
- It moves the focus pointer across the listed meta metrics.
- The `Metric (M: next): ...` line always describes the currently focused metric.

### Meta metrics (displayed list)

- `Entropy`: diversity of attention across terms in the selected window.
- `Top-1 Share`: concentration measure; fraction of total activity owned by the top term.
- `New-Term Rate`: share of currently active terms not seen in the earlier comparison window.
- `Turnover`: top-set change rate versus the previous comparison window.
- `Mean Freshness`: average per-term freshness ratio `(10m/60m)`.
- `Max Freshness`: highest per-term freshness ratio `(10m/60m)`.
- `Active Unique Terms`: count of distinct active terms in the selected window.

## Sparkline scaling modes (`S`)

- `per-term`: each term sparkline is scaled independently (best intra-term shape visibility).
- `global`: all term sparklines share one scale (best absolute comparability).
- `global-sqrt`: global scale with square-root compression (reduces domination by extreme values while preserving comparability).

## System State

- `System State` is inferred from meta metrics (entropy, concentration, novelty, turnover, freshness) using adaptive thresholds when enough recent history is available (capped by `--adaptive-cap-hours`, default 24h), otherwise fallback thresholds.
- State evaluation window selection prefers the largest window with sufficient fill (controlled by `--state-min-fill`, default `0.70`).
- Header duration (`active for ...`) is how long the current state has been active.
- On startup, the app restores state continuity from historical transition logs for the selected window when available.
- If the requested window is underfilled, the header shows a `STATE EVAL:` note indicating the effective evaluation window.
- In `RECENT STATE CHANGES`, the duration shown after the `from` state is how long that state lasted before transitioning.
- Trend symbols in transition rows represent confidence-direction drift only:
  - `▲` rising
  - `●` flat
  - `▼` falling

## Metrics Notes

- Freshness Ratio is shown as:
`Freshness Ratio = term presence (10m/60m) at sample time`.
- Main term metrics and meta metrics are related but not identical views.
- Freshness and momentum are directional and window-sensitive.

## Ignore Filtering and Normalization

- `FilterIgnored` removes terms matching `--ignore` patterns.
- Matching is normalized and wildcard-aware (`*`, `?`, and bracket classes).
- `Normalize` applies only if `FilterIgnored` is ON.
- If a filtered term occupied rank 1, normalization shifts subsequent ranks up by 1.
- Header shows `Ignored terms: <count>` when `FilterIgnored` is ON.
Count is the number of configured ignore patterns, not live matched terms.

## Logs and Rotation

Files are written as daily CSV rotation in the target log directory:

- `terms_YYYY-MM-DD.csv`
- `meta_YYYY-MM-DD.csv`
- `state_YYYY-MM-DD.csv`

Retention cleanup runs daily and removes files older than `--retain-days`.

If legacy undated files are found, startup migrates them to dated names when safe.

## Default Paths

- Default log directory:
`%LOCALAPPDATA%\MWTopLookups\logs`

- If you pass explicit paths (`--csv`, `--meta-csv`, `--state-log-csv`),
rotation still writes dated files in the same directory as those paths.

## Example Commands

Run with defaults:

```powershell
python MWTopLookupsDisplay.py
```

Clear logs and exit:

```powershell
python MWTopLookupsDisplay.py --clear-logs
```

Use custom log directory and 14-day retention:

```powershell
python MWTopLookupsDisplay.py --log-dir "<log-dir>" --retain-days 14
```

Ignore recurring terms and force color:

```powershell
python MWTopLookupsDisplay.py --ignore "chargo*" --ignore "wordle" --color on
```

Faster polling for short-term monitoring:

```powershell
python MWTopLookupsDisplay.py --log-interval 15 --refresh 10
```

## Repository Layout

- `MWTopLookupsDisplay.py`
Main app: logger, analytics, rendering, controls.

- `README.md`
Project documentation.

- `CHANGELOG.md`
Release-to-release change history.

- `VERSION`
Current project version.

## Troubleshooting

- Terminal too small:
Increase terminal to at least `80x24`.

- No data appears yet:
Wait for one or more poll cycles. Logger starts immediately.

- Endpoint errors:
Check network access and endpoint availability. Use retries/timeouts options as needed.

- Colors not showing:
Use `--color on`.

## License

This project is licensed under the [MIT License](LICENSE).

## Versioning and Releases

- This project uses Semantic Versioning (`MAJOR.MINOR.PATCH`).
- Current version is defined in `VERSION`.
- Release history is tracked in `CHANGELOG.md`.
- GitHub releases are tagged as `v<version>` (for example, `v0.1.0`) with release notes.
