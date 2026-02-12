# MW Top Lookups Display

MW Top Lookups Display is a terminal dashboard plus built-in logger for tracking Merriam-Webster Top Lookups over time.

It continuously polls the lookup endpoint, writes rotated CSV logs, computes trend and meta metrics, classifies system state, and renders a live TUI for short- and long-horizon analysis.

## What It Does

- Polls Merriam-Webster Top Lookups on an interval (default 31s).
- Stores terms, meta metrics, and state transitions as daily rotated CSV files.
- Renders two pages in the terminal.
- Main page focuses on term-level trends and history.
- Meta page focuses on system diagnostics, anomaly flags, and state changes.
- Supports ignore filters with wildcard patterns (for recurring sticky/control terms).
- Supports retention cleanup and window presets up to retention horizon.

## Runtime Behavior

- Logger is always ON during runtime.
- There is no runtime key to disable logging.
- Running with no arguments launches immediately using defaults.

## Requirements

- Windows terminal with at least 80 columns x 24 rows.
- Python 3.10+ recommended.
- Network access to the endpoint.

## Quick Start

1. Open PowerShell.
2. Run:

```powershell
python MWTopLookupsDisplay.py
```

3. Use `Q` to quit.

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

- `--ignore <pattern>`
Ignore term or wildcard pattern. Repeatable.
Example: `--ignore "chargo*" --ignore "wordle"`

- `--spark-scale per-term|global|global-sqrt`
Initial sparkline scaling mode. Default is `global-sqrt`.

- `--color auto|on|off`
ANSI color mode.

- `--once`
Render once and exit.

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
- In compact mode, non-essential header lines are hidden.

## Meta Page

- `META SNAPSHOT` lists key diagnostics (entropy, concentration, novelty, turnover, freshness, active terms).
- Pointer highlights the currently selected meta-help metric.
- `Metric (M: next): ...` explains the focused metric.
- Includes anomaly flags, summary interpretation, and recent state transitions.

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

No license file is currently included. Add one if you plan to distribute publicly.
