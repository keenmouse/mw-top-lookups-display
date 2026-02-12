#!/usr/bin/env python3
"""
Live terminal dashboard for Merriam-Webster Top Lookups CSV logs.

Controls:
  + / - : increase / decrease rolling window preset
  A     : auto-fit window to all available data
  V     : toggle Main/Meta view
  F     : toggle ignored-term filter
  M     : next metric (Main: chart metric, Meta: highlighted metric-help target)
  S     : cycle sparkline scaling (per-term / global / global-sqrt)
  C     : toggle compact/full header
  N     : toggle normalization (only when FilterIgnored is ON)
  Q     : quit
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import statistics
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
import re
import glob
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple


WINDOW_PRESETS_MINUTES = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
METRICS = ["rank_weighted", "presence", "momentum", "spike_ratio"]
SPARK_SCALE_MODES = ["per-term", "global", "global-sqrt"]
META_HELP_METRICS = [
    "entropy_selected",
    "top1_share_60m",
    "new_term_rate_60m",
    "turnover_10m",
    "mean_spike_ratio_10m_60m",
    "max_spike_ratio_10m_60m",
    "active_unique_selected",
]
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
SPARK_WIDTH = 24
MIN_TERM_COLS = 80
MIN_TERM_ROWS = 24
SYSTEM_STATES = [
    "Meme turbulence",
    "Distributed curiosity",
    "Reinforcement plateau",
    "Narrative convergence",
]
STATE_COLOR_CODES = {
    "Meme turbulence": 31,         # red
    "Distributed curiosity": 34,   # blue
    "Reinforcement plateau": 33,   # yellow
    "Narrative convergence": 35,   # magenta
}
SHORT_LABELS = {
    "Rank-Weighted Activity": "Rank-Weighted",
    "Active Unique Terms": "Active Terms",
    "Mean Rank-Weighted Score": "Mean RW Score",
    "Freshness Ratio": "Freshness",
    "Top-1 Share": "Top1 Share",
    "New-Term Rate": "New-Term",
    "Churn (Count)": "Churn",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LOG_DATE_RE = re.compile(r"^(terms|meta|state)_(\d{4}-\d{2}-\d{2})\.csv$")

STICKY_WORDS_DEFAULT: set[str] = set()
METRIC_UI: Dict[str, Dict[str, str]] = {
    "rank_weighted": {
        "label": "Rank-Weighted Activity",
        "help": "Metric help: Rank-Weighted Activity = time-weighted score based on how high a term ranks in each snapshot.",
    },
    "presence": {
        "label": "Presence",
        "help": "Metric help: Presence = count of snapshots in the window where the term appears (frequency of appearance).",
    },
    "momentum": {
        "label": "Momentum",
        "help": "Metric help: Momentum = net change in rank-weighted score over the window (rising vs falling attention).",
    },
    "spike_ratio": {
        "label": "Freshness Ratio",
        "help": "Metric help: Freshness Ratio = term presence (10m/60m) at sample time; higher = more of the term's 60m activity is recent.",
    },
    "entropy_selected": {
        "label": "Entropy",
        "help": "Metric help: Entropy = diversity of terms in the selected window; higher = more distributed attention.",
    },
    "entropy": {
        "label": "Entropy",
        "help": "Metric help: Entropy = diversity of terms in the selected window; higher = more distributed attention.",
    },
    "active_unique_selected": {
        "label": "Active Unique Terms",
        "help": "Metric help: Active Unique Terms = number of distinct terms seen in the selected window.",
    },
    "active_unique": {
        "label": "Active Unique Terms",
        "help": "Metric help: Active Unique Terms = number of distinct terms seen in the selected window.",
    },
    "top1_share_60m": {
        "label": "Top-1 Share",
        "help": "Metric help: Top-1 Share = fraction of total score owned by the top term; higher = more concentration.",
    },
    "top1_share": {
        "label": "Top-1 Share",
        "help": "Metric help: Top-1 Share = fraction of total score owned by the top term; higher = more concentration.",
    },
    "new_term_rate_60m": {
        "label": "New-Term Rate",
        "help": "Metric help: New-Term Rate = fraction of currently seen terms that were not present earlier in the window.",
    },
    "new_term_rate": {
        "label": "New-Term Rate",
        "help": "Metric help: New-Term Rate = fraction of currently seen terms that were not present earlier in the window.",
    },
    "turnover_10m": {
        "label": "Turnover",
        "help": "Metric help: Turnover = how much the top-term set changes vs the previous window (set change rate).",
    },
    "turnover": {
        "label": "Turnover",
        "help": "Metric help: Turnover = how much the top-term set changes vs the previous window (set change rate).",
    },
    "churn_10m": {
        "label": "Churn (Count)",
        "help": "Metric help: Churn (Count) = number of terms entering/leaving the tracked set vs the previous window.",
    },
    "churn": {
        "label": "Churn (Count)",
        "help": "Metric help: Churn (Count) = number of terms entering/leaving the tracked set vs the previous window.",
    },
    "mean_rank_weighted_score_selected": {
        "label": "Mean Rank-Weighted Score",
        "help": "Metric help: Mean Rank-Weighted Score = average rank-weighted score across tracked terms in the window.",
    },
    "mean_rank_weighted": {
        "label": "Mean Rank-Weighted Score",
        "help": "Metric help: Mean Rank-Weighted Score = average rank-weighted score across tracked terms in the window.",
    },
    "mean_spike_ratio_10m_60m": {
        "label": "Mean Freshness",
        "help": "Metric help: Mean Freshness = average Freshness Ratio across tracked terms.",
    },
    "mean_spike_ratio": {
        "label": "Mean Freshness",
        "help": "Metric help: Mean Freshness = average Freshness Ratio across tracked terms.",
    },
    "max_spike_ratio_10m_60m": {
        "label": "Max Freshness",
        "help": "Metric help: Max Freshness = highest Freshness Ratio among tracked terms.",
    },
    "max_spike_ratio": {
        "label": "Max Freshness",
        "help": "Metric help: Max Freshness = highest Freshness Ratio among tracked terms.",
    },
}


@dataclass
class Snapshot:
    poll_id: str
    ts_utc: datetime
    ranks: Dict[str, int]
    source_timestamp: str = ""


@dataclass
class MetaPoint:
    poll_id: str
    poll_index: int
    ts_utc: datetime
    source_timestamp: str
    metric_window_minutes: int
    spike_ratio_10m_60m: float
    entropy_selected: float
    active_unique_terms_selected: int
    total_score_10m: float
    total_score_60m: float
    total_score_selected: float
    top1_share_60m: float
    new_term_rate_60m: float
    turnover_10m: float
    churn_10m: int
    mean_rank_weighted_score_selected: float
    mean_spike_ratio_10m_60m: float
    max_spike_ratio_10m_60m: float
    top_term_60m: str
    top_term_score_60m: float
    top_term_selected: str
    top_term_score_selected: float
    system_state: str = ""
    state_confidence: str = ""


@dataclass
class StateTransitionPoint:
    timestamp_utc: datetime
    old_state: str
    new_state: str
    old_duration: str
    conf_now: str
    conf_delta: float
    conf_trend: str
    window_minutes: int


@dataclass
class State:
    window_idx: int = 2  # default 60 minutes
    metric_idx: int = 0  # default rank_weighted
    filter_sticky: bool = False
    normalize: bool = False
    log_enabled: bool = True
    screen_mode: str = "main"
    meta_mode: str = "snapshot"
    current_state: str = ""
    current_conf_label: str = "low"
    current_conf_value: float = 0.0
    current_threshold_mode: str = "fallback"
    state_started_at: datetime | None = None
    prev_state: str = ""
    prev_state_duration_sec: float = 0.0
    pending_state: str = ""
    pending_count: int = 0
    last_state_window_minutes: int = 60
    last_state_point_key: str = ""
    transition_banner_once: str = ""
    conf_history: List[float] = field(default_factory=list)
    conf_delta: float = 0.0
    conf_drift_suffix: str = ""
    conf_up_streak: int = 0
    conf_down_streak: int = 0
    retention_last_run_local: str = ""
    spark_scale: str = "global-sqrt"
    meta_help_idx: int = 0
    compact_header: bool = False

    @property
    def window_minutes(self) -> int:
        return WINDOW_PRESETS_MINUTES[self.window_idx]

    @property
    def metric(self) -> str:
        return METRICS[self.metric_idx]


def parse_args() -> argparse.Namespace:
    local_appdata = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    default_log_dir = os.path.join(local_appdata, "MWTopLookups", "logs")
    parser = argparse.ArgumentParser(description="MW Top Lookups Display")
    parser.add_argument(
        "--log-dir",
        default=default_log_dir,
        help="Directory for all log files (default: %%LOCALAPPDATA%%/MWTopLookups/logs)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Base path for terms logs (directory inferred from this path)",
    )
    parser.add_argument(
        "--meta-csv",
        default=None,
        help="Base path for meta logs (directory inferred from this path)",
    )
    parser.add_argument(
        "--state-log-csv",
        default=None,
        help="Base path for state transition logs (directory inferred from this path)",
    )
    parser.add_argument(
        "--retain-days",
        type=int,
        default=7,
        help="Retention window for rotated daily logs",
    )
    parser.add_argument("--refresh", type=float, default=30.0, help="Refresh interval seconds")
    parser.add_argument("--top-k", type=int, default=15, help="Top terms to show in trend bars")
    parser.add_argument(
        "--current-n",
        type=int,
        default=10,
        help="Current snapshot items to show (capped by source list size; currently 10)",
    )
    parser.add_argument("--history-points", type=int, default=20, help="Sparkline points per term")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    parser.add_argument("--log-interval", type=float, default=31.0, help="Embedded logger poll interval seconds")
    parser.add_argument(
        "--log-endpoint",
        default="https://factotum196p963.m-w.com:6058/lapi/v1/mwol-mp/get-lookups-data-homepage",
        help="Merriam-Webster lookups JSON endpoint",
    )
    parser.add_argument("--log-timeout", type=int, default=20, help="Embedded logger HTTP timeout seconds")
    parser.add_argument("--log-retries", type=int, default=3, help="Embedded logger retry attempts")
    parser.add_argument("--log-retry-delay", type=float, default=2.0, help="Embedded logger retry delay seconds")
    parser.add_argument(
        "--gap-pct-threshold",
        type=float,
        default=0.50,
        help="Gap threshold as fraction above expected interval (default: 0.50)",
    )
    parser.add_argument(
        "--spark-scale",
        choices=["per-term", "global", "global-sqrt"],
        default="global-sqrt",
        help="Sparkline scaling mode",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "on", "off"],
        default="auto",
        help="Color mode for optional accenting (default: auto)",
    )
    parser.add_argument(
        "--ignore",
        dest="ignore_words",
        action="append",
        default=[],
        help='Ignored term/pattern (repeatable). Supports wildcards, e.g. --ignore "chargo*"',
    )
    return parser.parse_args()


def parse_iso_utc(value: str) -> datetime:
    value = (value or "").strip()
    if not value:
        return datetime.now(timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_term(term: str) -> str:
    return " ".join((term or "").strip().lower().split())


def is_ignored_term(term: str, ignore_patterns: set[str]) -> bool:
    term_n = normalize_term(term)
    for pattern in ignore_patterns:
        p = normalize_term(pattern)
        if not p:
            continue
        if any(ch in p for ch in ["*", "?", "["]):
            if fnmatch.fnmatchcase(term_n, p):
                return True
        elif term_n == p:
            return True
    return False


def hash_list(words: List[str]) -> str:
    joined = "|".join(words)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def parse_source_timestamp_utc(value: str) -> datetime:
    value = (value or "").strip()
    if not value:
        return datetime.now(timezone.utc)
    try:
        # Source format observed: "YYYY-MM-DD HH:MM:SS" (UTC).
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def format_source_timestamp_local(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    try:
        return format_local_dt(parse_source_timestamp_utc(raw), include_date=False)
    except Exception:
        return raw


def load_snapshots(csv_path: str) -> List[Snapshot]:
    files = list_rotated_log_files(csv_path, "terms")
    if not files:
        return []

    grouped: Dict[str, Dict[str, object]] = {}
    for csv_file in files:
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                poll_id = (row.get("poll_id") or "").strip()
                if not poll_id:
                    continue

                rank_raw = row.get("rank") or ""
                word_raw = row.get("word") or ""
                ts_raw = row.get("polled_at_utc") or row.get("polled_at_local") or ""
                source_ts_raw = (row.get("source_timestamp") or "").strip()

                try:
                    rank = int(rank_raw)
                except ValueError:
                    continue

                word = normalize_term(word_raw)
                if not word:
                    continue

                ts = parse_iso_utc(ts_raw)

                if poll_id not in grouped:
                    grouped[poll_id] = {"ts": ts, "ranks": {}, "source_timestamp": source_ts_raw}

                # Keep best (lowest) rank if duplicate rows exist.
                ranks: Dict[str, int] = grouped[poll_id]["ranks"]  # type: ignore[assignment]
                current_rank = ranks.get(word)
                if current_rank is None or rank < current_rank:
                    ranks[word] = rank

                prev_ts: datetime = grouped[poll_id]["ts"]  # type: ignore[assignment]
                if ts > prev_ts:
                    grouped[poll_id]["ts"] = ts
                if source_ts_raw:
                    grouped[poll_id]["source_timestamp"] = source_ts_raw

    snapshots: List[Snapshot] = []
    for poll_id, payload in grouped.items():
        snapshots.append(
            Snapshot(
                poll_id=poll_id,
                ts_utc=payload["ts"],  # type: ignore[arg-type]
                ranks=payload["ranks"],  # type: ignore[arg-type]
                source_timestamp=payload.get("source_timestamp", ""),  # type: ignore[arg-type]
            )
        )

    snapshots.sort(key=lambda s: s.ts_utc)
    return snapshots


def apply_filter_and_normalization(
    snapshot: Snapshot,
    filter_sticky: bool,
    normalize: bool,
    sticky_words: set[str],
) -> Snapshot:
    ranks = dict(snapshot.ranks)

    if filter_sticky:
        ranks = {term: rank for term, rank in ranks.items() if not is_ignored_term(term, sticky_words)}

    if filter_sticky and normalize:
        # If sticky is rank 1, drop it and shift all other ranks up by 1.
        sticky_at_rank1 = any(r == 1 and is_ignored_term(term, sticky_words) for term, r in snapshot.ranks.items())
        if sticky_at_rank1:
            adjusted: Dict[str, int] = {}
            for term, rank in ranks.items():
                if rank > 1:
                    adjusted[term] = rank - 1
                elif not is_ignored_term(term, sticky_words):
                    adjusted[term] = rank
            ranks = adjusted

    return Snapshot(poll_id=snapshot.poll_id, ts_utc=snapshot.ts_utc, ranks=ranks)


def windowed_snapshots(snapshots: List[Snapshot], minutes: int) -> Tuple[List[Snapshot], List[Snapshot], datetime]:
    if not snapshots:
        now = datetime.now(timezone.utc)
        return [], [], now

    now = snapshots[-1].ts_utc
    window = timedelta(minutes=minutes)
    cutoff = now - window
    prev_cutoff = cutoff - window

    current = [s for s in snapshots if cutoff <= s.ts_utc <= now]
    previous = [s for s in snapshots if prev_cutoff <= s.ts_utc < cutoff]
    return current, previous, now


def compute_meta_metrics(snapshots: List[Snapshot], selected_window_minutes: int) -> Dict[str, object]:
    snaps_10m, _, _ = windowed_snapshots(snapshots, 10)
    snaps_60m, _, _ = windowed_snapshots(snapshots, 60)
    snaps_selected, _, _ = windowed_snapshots(snapshots, selected_window_minutes)
    _, prev_10m, _ = windowed_snapshots(snapshots, 10)
    _, prev_60m, _ = windowed_snapshots(snapshots, 60)

    rw_10m = compute_rank_weighted_scores(snaps_10m)
    rw_60m = compute_rank_weighted_scores(snaps_60m)
    rw_selected = compute_rank_weighted_scores(snaps_selected)

    total_10m = float(sum(rw_10m.values()))
    total_60m = float(sum(rw_60m.values()))
    total_selected = float(sum(rw_selected.values()))

    # Fixed-window "spike/change" signals.
    spike_ratio_10m_60m = (total_10m / total_60m) if total_60m > 0 else 0.0

    active_terms_10m = {k for k, v in rw_10m.items() if v > 0}
    prev_terms_10m = {k for k, v in compute_rank_weighted_scores(prev_10m).items() if v > 0}
    overlap_10m = active_terms_10m & prev_terms_10m
    union_10m = active_terms_10m | prev_terms_10m
    turnover_10m = (1.0 - (len(overlap_10m) / len(union_10m))) if union_10m else 0.0
    churn_10m = len(active_terms_10m - prev_terms_10m)

    active_terms_60m = {k for k, v in rw_60m.items() if v > 0}
    prev_terms_60m = {k for k, v in compute_rank_weighted_scores(prev_60m).items() if v > 0}
    new_term_rate_60m = (len(active_terms_60m - prev_terms_60m) / len(active_terms_60m)) if active_terms_60m else 0.0

    top_term_60m = ""
    top_term_score_60m = 0.0
    if rw_60m:
        top_term_60m, top_term_score_60m = max(rw_60m.items(), key=lambda kv: kv[1])
    top1_share_60m = (top_term_score_60m / total_60m) if total_60m > 0 else 0.0

    spike_per_term_10m_60m = compute_spike_ratio_scores(rw_10m, rw_60m)
    spike_values = [v for k, v in spike_per_term_10m_60m.items() if rw_60m.get(k, 0.0) > 0]
    mean_spike_ratio_10m_60m = (sum(spike_values) / len(spike_values)) if spike_values else 0.0
    max_spike_ratio_10m_60m = max(spike_values) if spike_values else 0.0

    # Window-relative "state/distribution" signals.
    active_unique_selected = float(sum(1 for v in rw_selected.values() if v > 0))
    if total_selected > 0:
        entropy_selected = -sum(
            (v / total_selected) * math.log(v / total_selected) for v in rw_selected.values() if v > 0
        )
    else:
        entropy_selected = 0.0

    mean_rank_weighted_score_selected = (total_selected / active_unique_selected) if active_unique_selected > 0 else 0.0
    top_term_selected = ""
    top_term_score_selected = 0.0
    if rw_selected:
        top_term_selected, top_term_score_selected = max(rw_selected.items(), key=lambda kv: kv[1])

    return {
        "selected_window_minutes": float(selected_window_minutes),
        "spike_ratio_10m_60m": spike_ratio_10m_60m,
        "entropy_selected": entropy_selected,
        "active_unique_selected": active_unique_selected,
        "total_score_10m": total_10m,
        "total_score_60m": total_60m,
        "total_score_selected": total_selected,
        "top1_share_60m": top1_share_60m,
        "new_term_rate_60m": new_term_rate_60m,
        "turnover_10m": turnover_10m,
        "churn_10m": float(churn_10m),
        "mean_rank_weighted_score_selected": mean_rank_weighted_score_selected,
        "mean_spike_ratio_10m_60m": mean_spike_ratio_10m_60m,
        "max_spike_ratio_10m_60m": max_spike_ratio_10m_60m,
        "top_term_60m": top_term_60m,
        "top_term_score_60m": top_term_score_60m,
        "top_term_selected": top_term_selected,
        "top_term_score_selected": top_term_score_selected,
    }


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def metric_value_from_point(point: MetaPoint, key: str) -> float:
    if key == "E":
        return float(point.entropy_selected)
    if key == "C":
        return float(point.top1_share_60m)
    if key == "N":
        return float(point.new_term_rate_60m)
    if key == "T":
        return float(point.turnover_10m)
    if key == "S":
        return float(point.mean_spike_ratio_10m_60m)
    return 0.0


def fixed_thresholds(metric_key: str) -> Tuple[float, float]:
    # Conservative fallbacks while history is sparse.
    # Returns (low_threshold, high_threshold).
    defaults = {
        "E": (2.4, 3.2),    # entropy
        "C": (0.12, 0.25),  # concentration/top1 share
        "N": (0.25, 0.60),  # novelty
        "T": (0.30, 0.65),  # turnover
        "S": (0.20, 0.45),  # mean spike ratio
    }
    return defaults.get(metric_key, (0.0, 1.0))


def bucket_label(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "normal"


def derive_metric_labels(
    current_metrics: Dict[str, object],
    scoped_points: List[MetaPoint],
) -> Tuple[Dict[str, str], bool]:
    # Adaptive once enough points exist for selected window.
    use_adaptive = len(scoped_points) >= 60
    labels: Dict[str, str] = {}
    for key in ["E", "C", "N", "T", "S"]:
        if key == "E":
            value = float(current_metrics.get("entropy_selected", 0.0))
        elif key == "C":
            value = float(current_metrics.get("top1_share_60m", 0.0))
        elif key == "N":
            value = float(current_metrics.get("new_term_rate_60m", 0.0))
        elif key == "T":
            value = float(current_metrics.get("turnover_10m", 0.0))
        else:
            value = float(current_metrics.get("mean_spike_ratio_10m_60m", 0.0))

        if use_adaptive:
            series = [metric_value_from_point(p, key) for p in scoped_points]
            low = percentile(series, 0.10)
            high = percentile(series, 0.90)
        else:
            low, high = fixed_thresholds(key)
        labels[key] = bucket_label(value, low, high)
    return labels, use_adaptive


def classify_system_state(
    current_metrics: Dict[str, object],
    scoped_points: List[MetaPoint],
) -> Tuple[str, str, float, bool]:
    labels, adaptive = derive_metric_labels(current_metrics, scoped_points)
    E = labels["E"]
    C = labels["C"]
    N = labels["N"]
    T = labels["T"]
    S = labels["S"]

    # Tie-breaks first
    if C == "high" and E == "low":
        return "Narrative convergence", "high", 1.0, adaptive
    if T == "high" and C != "high":
        return "Meme turbulence", "high", 1.0, adaptive

    candidates: List[Tuple[str, int, int, float]] = []

    # (state, strong_match_count, total_condition_count)
    turb_strong = int(T == "high") + int(S == "high") + int(N == "high") + int(C in {"low", "normal"})
    candidates.append(("Meme turbulence", turb_strong, 4, turb_strong / 4.0))

    dist_strong = int(E == "high") + int(C == "low") + int(N in {"high", "normal"}) + int(T == "normal")
    candidates.append(("Distributed curiosity", dist_strong, 4, dist_strong / 4.0))

    plat_strong = int(C in {"normal", "high"}) + int(E == "normal") + int(N == "low") + int(T in {"low", "normal"}) + int(S == "low")
    candidates.append(("Reinforcement plateau", plat_strong, 5, plat_strong / 5.0))

    conv_strong = int(C == "high") + int(E == "low") + int(N == "low") + int(T == "low")
    candidates.append(("Narrative convergence", conv_strong, 4, conv_strong / 4.0))

    best_state, best_score, total, best_ratio = max(candidates, key=lambda x: (x[1], -x[2], x[3]))

    ratio = (best_score / total) if total > 0 else 0.0
    if ratio >= 0.85:
        conf = "high"
    elif ratio >= 0.55:
        conf = "med"
    else:
        conf = "low"

    return best_state, conf, best_ratio, adaptive


def color_enabled(color_mode: str) -> bool:
    if color_mode == "on":
        return True
    if color_mode == "off":
        return False
    return sys.stdout.isatty()


def metric_label(metric_key: str) -> str:
    ui = METRIC_UI.get(metric_key)
    return ui["label"] if ui else metric_key.replace("_", " ").title()


def metric_help_text(metric_key: str) -> str:
    ui = METRIC_UI.get(metric_key)
    text = ui["help"] if ui and ui.get("help") else f"{metric_label(metric_key)} = short description unavailable."
    prefix = "Metric help: "
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def metric_help_line(metric_key: str, width: int = 140) -> str:
    return fit_line(f"Metric help: {metric_help_text(metric_key)}", width)


def colored_metric_help_line(metric_key: str, width: int, use_color: bool, color_code: int = 96) -> str:
    base = metric_help_line(metric_key, width)
    if not use_color:
        return base
    prefix = "Metric help: "
    if not base.startswith(prefix):
        return base
    tail = base[len(prefix):]
    return fit_line(f"{prefix}\x1b[{color_code}m{tail}\x1b[0m", width)


def colored_metric_help_line_with_prompt(
    metric_key: str,
    width: int,
    use_color: bool,
    prefix: str = "Metric (M: next): ",
    color_code: int = 96,
) -> str:
    help_text = metric_help_text(metric_key)
    if not use_color:
        return fit_line(f"{prefix}{help_text}", width)
    return fit_line(f"{prefix}\x1b[{color_code}m{help_text}\x1b[0m", width)


def current_meta_help_metric(state: State) -> str:
    if not META_HELP_METRICS:
        return "turnover_10m"
    idx = max(0, state.meta_help_idx) % len(META_HELP_METRICS)
    return META_HELP_METRICS[idx]


def controls_line(state: State, use_color: bool = False) -> str:
    base = "+/- window  A all-data  V main/meta  F ignored-filter"
    normalize_seg = "  N normalize" if state.filter_sticky else ""
    tail_main = f"S spark  C compact{normalize_seg}  Q quit"
    tail_meta = f"C compact{normalize_seg}  Q quit"
    m_label = "M metric"
    if state.screen_mode == "meta":
        return f"Controls: {base}  {m_label}  {tail_meta}"
    return f"Controls: {base}  {m_label}  {tail_main}"


def short_metric_label(metric_key: str, width: int) -> str:
    label = metric_label(metric_key)
    if width < 110:
        label = SHORT_LABELS.get(label, label)
    if width < 92:
        label = SHORT_LABELS.get(label, label)
    return label


def visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def fit_line(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if visible_len(text) <= width:
        return text
    plain = ANSI_RE.sub("", text)
    if len(plain) <= width:
        return plain
    if width <= 1:
        return plain[:width]
    return plain[: width - 1] + "…"


def format_local_dt(dt: datetime, include_date: bool = True) -> str:
    local_dt = dt.astimezone()
    if include_date:
        return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    return local_dt.strftime("%H:%M:%S %Z")


def format_window_label(minutes: int) -> str:
    m = max(1, int(minutes))
    if m % 1440 == 0:
        d = m // 1440
        return f"{d}d"
    if m % 60 == 0:
        h = m // 60
        return f"{h}h"
    return f"{m}m"


def build_window_presets(retain_days: int) -> List[int]:
    max_minutes = max(10, int(retain_days) * 1440)
    base = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
    out = [v for v in base if v <= max_minutes]
    # Add daily checkpoints up to retention horizon.
    day = 2
    while day * 1440 <= max_minutes:
        out.append(day * 1440)
        day += 1
    out = sorted(set(out))
    if 10 not in out:
        out.insert(0, 10)
    if max_minutes not in out:
        out.append(max_minutes)
    return out


def local_day_str(now: datetime | None = None) -> str:
    dt = now.astimezone() if now is not None else datetime.now().astimezone()
    return dt.strftime("%Y-%m-%d")


def rotated_log_path(base_path: str, log_type: str, day_str: str | None = None) -> str:
    day = day_str or local_day_str()
    folder = os.path.dirname(base_path) or "."
    return os.path.join(folder, f"{log_type}_{day}.csv")


def list_rotated_log_files(base_path: str, log_type: str) -> List[str]:
    folder = os.path.dirname(base_path) or "."
    pattern = os.path.join(folder, f"{log_type}_*.csv")
    files = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    files.sort()
    return files


def enforce_retention_for_type(base_path: str, log_type: str, retain_days: int, today_local: datetime) -> int:
    keep_days = max(1, int(retain_days))
    cutoff = (today_local.date() - timedelta(days=keep_days - 1))
    deleted = 0
    for path in list_rotated_log_files(base_path, log_type):
        name = os.path.basename(path)
        m = LOG_DATE_RE.match(name)
        if not m:
            continue
        if m.group(1) != log_type:
            continue
        try:
            d = datetime.strptime(m.group(2), "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < cutoff:
            try:
                os.remove(path)
                deleted += 1
            except OSError:
                pass
    return deleted


def rotate_legacy_base_file(base_path: str, log_type: str, day_str: str | None = None) -> None:
    # Migrate legacy undated file names (e.g., MWTopLookups.csv) to dated rotation file.
    if not base_path:
        return
    base_name = os.path.basename(base_path)
    if LOG_DATE_RE.match(base_name):
        return
    if not os.path.exists(base_path) or not os.path.isfile(base_path):
        return
    target = rotated_log_path(base_path, log_type, day_str)
    if os.path.abspath(base_path) == os.path.abspath(target):
        return
    if os.path.exists(target):
        # Keep operation safe and non-destructive; if target already exists, leave legacy file untouched.
        return
    try:
        os.replace(base_path, target)
    except OSError:
        pass


def format_duration(seconds: float) -> str:
    secs = max(0, int(seconds))
    if secs >= 3600:
        h = secs // 3600
        m = (secs % 3600) // 60
        return f"{h}h{m:02d}m"
    if secs >= 60:
        m = secs // 60
        return f"{m}m"
    return f"{secs}s"


def state_status_line(state: State, now_utc: datetime, use_color: bool, width: int) -> str:
    active = state.current_state or "unknown"
    started = state.state_started_at or now_utc
    duration = format_duration((now_utc - started).total_seconds())
    conf_txt = state.current_conf_label
    if state.conf_drift_suffix:
        conf_txt = f"{conf_txt} ({state.conf_drift_suffix})"
    dot = "●"
    if use_color:
        code = STATE_COLOR_CODES.get(active, 37)
        dot = f"\x1b[{code}m●\x1b[0m"
    if width < 110:
        th = "adp" if state.current_threshold_mode == "adaptive" else "fb"
        return fit_line(f"STATE: {dot} {active} {duration} | CONF: {conf_txt} | TH: {th}", width)
    return fit_line(
        f"SYSTEM STATE: {dot} {active} ({duration}) | CONF: {conf_txt} | thresholds={state.current_threshold_mode}",
        width,
    )


def append_state_transition_csv(
    path: str,
    timestamp_utc: datetime,
    old_state: str,
    new_state: str,
    old_duration_sec: float,
    conf_now: str,
    conf_delta: float,
    conf_trend: str,
    window_minutes: int,
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    file_exists = os.path.exists(path)
    fields = [
        "timestamp_utc",
        "old_state",
        "new_state",
        "old_duration",
        "conf_now",
        "conf_delta",
        "conf_trend",
        "window_minutes",
    ]
    row = {
        "timestamp_utc": timestamp_utc.isoformat(),
        "old_state": old_state,
        "new_state": new_state,
        "old_duration": format_duration(old_duration_sec),
        "conf_now": conf_now,
        "conf_delta": f"{conf_delta:+.3f}",
        "conf_trend": conf_trend or "",
        "window_minutes": int(window_minutes),
    }
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def point_key_for_window(snapshots: List[Snapshot], meta_points: List[MetaPoint], window_minutes: int) -> str:
    scoped = [p for p in meta_points if p.metric_window_minutes == window_minutes]
    if scoped:
        p = scoped[-1]
        return f"meta:{window_minutes}:{p.poll_id}:{p.poll_index}"
    if snapshots:
        s = snapshots[-1]
        return f"snap:{window_minutes}:{s.poll_id}:{s.ts_utc.isoformat()}"
    return f"none:{window_minutes}"


def update_confidence_drift(state: State, conf_value: float) -> None:
    state.conf_history.append(conf_value)
    if len(state.conf_history) > 12:
        state.conf_history = state.conf_history[-12:]

    if len(state.conf_history) < 4:
        state.conf_delta = 0.0
        state.conf_drift_suffix = ""
        state.conf_up_streak = 0
        state.conf_down_streak = 0
        return

    k = 4
    prev_vals = state.conf_history[-(k + 1):-1]
    baseline = (sum(prev_vals) / len(prev_vals)) if prev_vals else conf_value
    delta = conf_value - baseline
    state.conf_delta = delta

    enter = 0.04
    exit_band = 0.02
    if delta >= enter:
        state.conf_up_streak += 1
        state.conf_down_streak = 0
    elif delta <= -enter:
        state.conf_down_streak += 1
        state.conf_up_streak = 0
    elif abs(delta) <= exit_band:
        state.conf_up_streak = 0
        state.conf_down_streak = 0
        state.conf_drift_suffix = ""
        return
    else:
        state.conf_up_streak = 0
        state.conf_down_streak = 0

    if state.conf_up_streak >= 2:
        state.conf_drift_suffix = "rising"
    elif state.conf_down_streak >= 2:
        state.conf_drift_suffix = "falling"


def update_runtime_state(
    state: State,
    candidate_state: str,
    conf_label: str,
    conf_value: float,
    threshold_mode: str,
    now_utc: datetime,
    point_key: str,
    logger_enabled: bool,
    transition_csv_path: str,
) -> None:
    if state.last_state_window_minutes != state.window_minutes:
        state.last_state_window_minutes = state.window_minutes
        state.current_state = candidate_state
        state.current_conf_label = conf_label
        state.current_conf_value = conf_value
        state.current_threshold_mode = threshold_mode
        state.state_started_at = now_utc
        state.prev_state = ""
        state.prev_state_duration_sec = 0.0
        state.pending_state = ""
        state.pending_count = 0
        state.last_state_point_key = point_key
        state.transition_banner_once = ""
        state.conf_history = []
        state.conf_delta = 0.0
        state.conf_drift_suffix = ""
        state.conf_up_streak = 0
        state.conf_down_streak = 0
        update_confidence_drift(state, conf_value)
        return

    if not state.current_state:
        state.current_state = candidate_state
        state.current_conf_label = conf_label
        state.current_conf_value = conf_value
        state.current_threshold_mode = threshold_mode
        state.state_started_at = now_utc

    new_point = (point_key != state.last_state_point_key)
    if new_point:
        state.last_state_point_key = point_key
        update_confidence_drift(state, conf_value)

    if candidate_state == state.current_state:
        state.pending_state = ""
        state.pending_count = 0
    elif new_point:
        needed = 2 + (1 if threshold_mode == "fallback" else 0)
        if state.pending_state == candidate_state:
            state.pending_count += 1
        else:
            state.pending_state = candidate_state
            state.pending_count = 1

        if state.pending_count >= needed:
            old_state = state.current_state
            started = state.state_started_at or now_utc
            old_duration = max(0.0, (now_utc - started).total_seconds())
            state.prev_state = old_state
            state.prev_state_duration_sec = old_duration
            state.current_state = candidate_state
            state.state_started_at = now_utc
            state.pending_state = ""
            state.pending_count = 0
            state.transition_banner_once = (
                f">>> STATE CHANGED: {old_state} -> {candidate_state} "
                f"(after {format_duration(old_duration)}) at {format_local_dt(now_utc, include_date=False)}"
            )
            if logger_enabled:
                append_state_transition_csv(
                    path=transition_csv_path,
                    timestamp_utc=now_utc,
                    old_state=old_state,
                    new_state=candidate_state,
                    old_duration_sec=old_duration,
                    conf_now=conf_label,
                    conf_delta=state.conf_delta,
                    conf_trend=state.conf_drift_suffix or "flat",
                    window_minutes=state.window_minutes,
                )

    state.current_conf_label = conf_label
    state.current_conf_value = conf_value
    state.current_threshold_mode = threshold_mode


def compute_rank_weighted_scores(snaps: List[Snapshot]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for s in snaps:
        if not s.ranks:
            continue
        n = max(s.ranks.values())
        for term, rank in s.ranks.items():
            weight = max(1, n - rank + 1)
            scores[term] = scores.get(term, 0.0) + float(weight)
    return scores


def compute_presence_scores(snaps: List[Snapshot]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for s in snaps:
        for term in s.ranks.keys():
            scores[term] = scores.get(term, 0.0) + 1.0
    return scores


def compute_scores(metric: str, current_snaps: List[Snapshot], previous_snaps: List[Snapshot]) -> Dict[str, float]:
    if metric == "presence":
        return compute_presence_scores(current_snaps)

    if metric == "rank_weighted":
        return compute_rank_weighted_scores(current_snaps)

    # momentum = rank-weighted(current window) - rank-weighted(previous window)
    cur = compute_rank_weighted_scores(current_snaps)
    prev = compute_rank_weighted_scores(previous_snaps)
    terms = set(cur.keys()) | set(prev.keys())
    return {t: cur.get(t, 0.0) - prev.get(t, 0.0) for t in terms}


def compute_spike_ratio_scores(scores_10m: Dict[str, float], scores_60m: Dict[str, float]) -> Dict[str, float]:
    terms = set(scores_10m.keys()) | set(scores_60m.keys())
    ratios: Dict[str, float] = {}
    for term in terms:
        s60 = scores_60m.get(term, 0.0)
        if s60 > 0:
            ratios[term] = scores_10m.get(term, 0.0) / s60
        else:
            ratios[term] = 0.0
    return ratios


def term_series(snaps: List[Snapshot], term: str, points: int, metric: str) -> List[float]:
    if points <= 0:
        return []
    tail = snaps[-points:]
    series: List[float] = []
    for s in tail:
        rank = s.ranks.get(term)
        if rank is None:
            series.append(0.0)
            continue
        if metric == "presence":
            series.append(1.0)
        else:
            n = max(s.ranks.values()) if s.ranks else rank
            series.append(float(max(1, n - rank + 1)))
    while len(series) < points:
        series.insert(0, 0.0)
    return series


def sparkline(values: List[float]) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        idx = 0 if hi <= 0 else len(SPARK_BLOCKS) - 1
        return SPARK_BLOCKS[idx] * len(values)
    out = []
    for v in values:
        pos = int((v - lo) / (hi - lo) * (len(SPARK_BLOCKS) - 1))
        out.append(SPARK_BLOCKS[pos])
    return "".join(out)


def resample_values(values: List[float], target: int) -> List[float]:
    if target <= 0:
        return []
    if not values:
        return [0.0] * target
    if len(values) == target:
        return values
    if len(values) < target:
        return [0.0] * (target - len(values)) + values
    # Evenly sample when shrinking.
    if target == 1:
        return [values[-1]]
    idxs = [round(i * (len(values) - 1) / (target - 1)) for i in range(target)]
    return [values[i] for i in idxs]


def sparkline_width(values: List[float], width: int) -> str:
    return sparkline(resample_values(values, width))


def choose_gap_glyph() -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "░".encode(enc)
        return "░"
    except Exception:
        return "."


def estimate_expected_interval_seconds(times: List[datetime], default_seconds: float = 30.0) -> float:
    if len(times) < 2:
        return default_seconds
    xs = sorted(times)
    intervals = [(xs[i] - xs[i - 1]).total_seconds() for i in range(1, len(xs))]
    intervals = [d for d in intervals if d > 0]
    if not intervals:
        return default_seconds
    med = statistics.median(intervals)
    if med <= 0:
        return default_seconds
    filtered = [d for d in intervals if (0.25 * med) <= d <= (3.0 * med)]
    if len(filtered) >= max(2, len(intervals) // 2):
        med = statistics.median(filtered)
    return max(1.0, float(med))


def sparkline_from_timed_samples(
    samples: List[Tuple[datetime, float]],
    width: int,
    gap_pct_threshold: float,
    gap_glyph: str,
    scale_mode: str = "per-term",
    global_max: float = 0.0,
) -> str:
    if width <= 0:
        return ""
    if not samples:
        return sparkline_width([], width)
    samples = sorted(samples, key=lambda x: x[0])
    times = [t for t, _ in samples]
    expected = estimate_expected_interval_seconds(times, default_seconds=30.0)
    gap_threshold = expected * (1.0 + max(0.0, gap_pct_threshold))

    t0 = times[0]
    t1 = times[-1]
    span = (t1 - t0).total_seconds()
    if span <= 0:
        vals = [0.0] * width
        vals[-1] = samples[-1][1]
        return sparkline(vals)

    vals = [0.0] * width
    counts = [0] * width
    gaps = [False] * width

    def bucket_idx(ts: datetime) -> int:
        pos = ((ts - t0).total_seconds() / span) * (width - 1)
        return max(0, min(width - 1, int(pos)))

    for ts, v in samples:
        i = bucket_idx(ts)
        vals[i] += float(v)
        counts[i] += 1

    for i in range(width):
        if counts[i] > 0:
            vals[i] /= counts[i]

    for i in range(1, len(times)):
        dt = (times[i] - times[i - 1]).total_seconds()
        if dt > gap_threshold:
            a = bucket_idx(times[i - 1])
            b = bucket_idx(times[i])
            lo, hi = (a, b) if a <= b else (b, a)
            for k in range(lo, hi + 1):
                gaps[k] = True

    if scale_mode == "per-term":
        chars = list(sparkline(vals))
    else:
        gm = max(0.0, float(global_max))
        if gm <= 0.0:
            chars = list(sparkline([0.0] * width))
        else:
            scaled: List[float] = []
            for v in vals:
                if v <= 0:
                    scaled.append(0.0)
                elif scale_mode == "global-sqrt":
                    scaled.append(math.sqrt(v / gm))
                else:
                    scaled.append(v / gm)
            scaled = [max(0.0, min(1.0, x)) for x in scaled]
            chars = []
            top = len(SPARK_BLOCKS) - 1
            for x in scaled:
                idx = int(round(x * top))
                chars.append(SPARK_BLOCKS[max(0, min(top, idx))])
    if len(chars) < width:
        chars = chars + [" "] * (width - len(chars))
    for i, is_gap in enumerate(gaps[: len(chars)]):
        if is_gap:
            chars[i] = gap_glyph
    return "".join(chars)


def clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def clear_screen() -> None:
    sys.stdout.write("\x1b[2J\x1b[H")


def fetch_top_lookups(endpoint: str, timeout: int, retries: int, retry_delay: float) -> Dict[str, object]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.merriam-webster.com/",
        "Origin": "https://www.merriam-webster.com",
        "Accept": "application/json, text/plain, */*",
    }
    req = urllib.request.Request(endpoint, headers=headers, method="GET")

    attempts = max(1, retries)
    last_error: Exception | None = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                data = payload.get("data", {})
                words = data.get("words", [])
                source_timestamp = data.get("timestamp", "")
                if not isinstance(words, list):
                    raise ValueError("JSON payload missing data.words list")
                return {"words": [str(w) for w in words], "source_timestamp": str(source_timestamp)}
        except Exception as exc:
            last_error = exc
            if i + 1 < attempts:
                time.sleep(max(0.1, retry_delay))
    raise RuntimeError(f"Logger request failed: {last_error}")


def latest_signature(snapshot: Snapshot | None) -> Tuple[str, Tuple[Tuple[str, int], ...]]:
    if snapshot is None:
        return ("", tuple())
    ranked = tuple(sorted(snapshot.ranks.items(), key=lambda kv: (kv[1], kv[0])))
    return (snapshot.source_timestamp, ranked)


def payload_signature(source_timestamp: str, words: List[str]) -> Tuple[str, Tuple[Tuple[str, int], ...]]:
    norm_words = [normalize_term(w) for w in words if normalize_term(w)]
    ranked = tuple((w, i + 1) for i, w in enumerate(norm_words))
    return (source_timestamp, ranked)


def dataset_span_minutes(snapshots: List[Snapshot]) -> int:
    if len(snapshots) < 2:
        return 10
    span_seconds = (snapshots[-1].ts_utc - snapshots[0].ts_utc).total_seconds()
    if span_seconds <= 0:
        return 10
    return max(10, int(math.ceil(span_seconds / 60.0)))


def choose_window_idx_for_span(span_minutes: int) -> int:
    target = max(10, int(span_minutes))
    for i, minutes in enumerate(WINDOW_PRESETS_MINUTES):
        if minutes >= target:
            return i
    return len(WINDOW_PRESETS_MINUTES) - 1


def get_max_poll_index(csv_path: str) -> int:
    max_idx = 0
    for csv_file in list_rotated_log_files(csv_path, "terms"):
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = (row.get("poll_index") or "").strip()
                try:
                    idx = int(raw)
                except ValueError:
                    continue
                if idx > max_idx:
                    max_idx = idx
    return max_idx


def append_poll_to_csv(
    csv_path: str, poll_index: int, endpoint: str, words: List[str], source_timestamp: str
) -> Dict[str, object]:
    norm_words = [normalize_term(w) for w in words if normalize_term(w)]
    if not norm_words:
        return {"row_count": 0}

    now_local = datetime.now().astimezone()
    now_utc = now_local.astimezone(timezone.utc)
    poll_id = f"{now_utc.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    list_hash = hash_list(norm_words)

    rows = []
    for i, raw_word in enumerate(words, start=1):
        word = (raw_word or "").strip()
        if not word:
            continue
        rows.append(
            {
                "poll_id": poll_id,
                "poll_index": poll_index,
                "polled_at_utc": now_utc.isoformat(),
                "polled_at_local": now_local.isoformat(),
                "source_timestamp": source_timestamp,
                "rank": i,
                "word": word,
                "word_normalized": normalize_term(word),
                "list_hash": list_hash,
                "endpoint": endpoint,
            }
        )

    if not rows:
        return {"row_count": 0}

    out_dir = os.path.dirname(csv_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    fieldnames = [
        "poll_id",
        "poll_index",
        "polled_at_utc",
        "polled_at_local",
        "source_timestamp",
        "rank",
        "word",
        "word_normalized",
        "list_hash",
        "endpoint",
    ]
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    return {
        "row_count": len(rows),
        "poll_id": poll_id,
        "poll_index": poll_index,
        "polled_at_utc": now_utc.isoformat(),
        "polled_at_local": now_local.isoformat(),
        "source_timestamp": source_timestamp,
        "endpoint": endpoint,
    }


def append_meta_to_csv(meta_csv_path: str, poll_info: Dict[str, object], metrics: Dict[str, object]) -> None:
    if not poll_info.get("poll_id"):
        return

    out_dir = os.path.dirname(meta_csv_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(meta_csv_path)
    fieldnames = [
        "poll_id",
        "poll_index",
        "polled_at_utc",
        "polled_at_local",
        "source_timestamp",
        "metric_window_minutes",
        "spike_ratio_10m_60m",
        "entropy_selected",
        "active_unique_terms_selected",
        "total_score_10m",
        "total_score_60m",
        "total_score_selected",
        "top1_share_60m",
        "new_term_rate_60m",
        "turnover_10m",
        "churn_10m",
        "mean_rank_weighted_score_selected",
        "mean_spike_ratio_10m_60m",
        "max_spike_ratio_10m_60m",
        "top_term_60m",
        "top_term_score_60m",
        "top_term_selected",
        "top_term_score_selected",
        "system_state",
        "state_confidence",
    ]

    row = {
        "poll_id": poll_info.get("poll_id", ""),
        "poll_index": poll_info.get("poll_index", 0),
        "polled_at_utc": poll_info.get("polled_at_utc", ""),
        "polled_at_local": poll_info.get("polled_at_local", ""),
        "source_timestamp": poll_info.get("source_timestamp", ""),
        "metric_window_minutes": int(metrics.get("selected_window_minutes", 0.0)),
        "spike_ratio_10m_60m": f"{metrics.get('spike_ratio_10m_60m', 0.0):.6f}",
        "entropy_selected": f"{metrics.get('entropy_selected', 0.0):.6f}",
        "active_unique_terms_selected": int(metrics.get("active_unique_selected", 0.0)),
        "total_score_10m": f"{metrics.get('total_score_10m', 0.0):.3f}",
        "total_score_60m": f"{metrics.get('total_score_60m', 0.0):.3f}",
        "total_score_selected": f"{metrics.get('total_score_selected', 0.0):.3f}",
        "top1_share_60m": f"{metrics.get('top1_share_60m', 0.0):.6f}",
        "new_term_rate_60m": f"{metrics.get('new_term_rate_60m', 0.0):.6f}",
        "turnover_10m": f"{metrics.get('turnover_10m', 0.0):.6f}",
        "churn_10m": int(metrics.get("churn_10m", 0.0)),
        "mean_rank_weighted_score_selected": f"{metrics.get('mean_rank_weighted_score_selected', 0.0):.6f}",
        "mean_spike_ratio_10m_60m": f"{metrics.get('mean_spike_ratio_10m_60m', 0.0):.6f}",
        "max_spike_ratio_10m_60m": f"{metrics.get('max_spike_ratio_10m_60m', 0.0):.6f}",
        "top_term_60m": str(metrics.get("top_term_60m", "")),
        "top_term_score_60m": f"{metrics.get('top_term_score_60m', 0.0):.6f}",
        "top_term_selected": str(metrics.get("top_term_selected", "")),
        "top_term_score_selected": f"{metrics.get('top_term_score_selected', 0.0):.6f}",
        "system_state": str(metrics.get("system_state", "")),
        "state_confidence": str(metrics.get("state_confidence", "")),
    }

    with open(meta_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_meta_points(meta_csv_path: str) -> List[MetaPoint]:
    files = list_rotated_log_files(meta_csv_path, "meta")
    if not files:
        return []

    points: List[MetaPoint] = []
    for csv_file in files:
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                poll_id = (row.get("poll_id") or "").strip()
                if not poll_id:
                    continue
                try:
                    poll_index = int((row.get("poll_index") or "0").strip() or "0")
                except ValueError:
                    poll_index = 0
                ts_utc = parse_iso_utc((row.get("polled_at_utc") or "").strip())
                source_ts = (row.get("source_timestamp") or "").strip()
                try:
                    metric_window_minutes = int((row.get("metric_window_minutes") or "60").strip() or "60")
                except ValueError:
                    metric_window_minutes = 60
                try:
                    spike = float(
                        (row.get("spike_ratio_10m_60m") or row.get("spike_ratio_10m_window") or "0").strip() or "0"
                    )
                except ValueError:
                    spike = 0.0
                try:
                    entropy = float((row.get("entropy_selected") or row.get("entropy_window") or row.get("entropy_60m") or "0").strip() or "0")
                except ValueError:
                    entropy = 0.0
                try:
                    active = int(
                        (row.get("active_unique_terms_selected") or row.get("active_unique_terms_window") or row.get("active_unique_terms_60m") or "0").strip() or "0"
                    )
                except ValueError:
                    active = 0
                try:
                    score10 = float((row.get("total_score_10m") or "0").strip() or "0")
                except ValueError:
                    score10 = 0.0
                try:
                    score60 = float((row.get("total_score_60m") or row.get("total_score_window") or "0").strip() or "0")
                except ValueError:
                    score60 = 0.0
                try:
                    score_selected = float((row.get("total_score_selected") or row.get("total_score_window") or "0").strip() or "0")
                except ValueError:
                    score_selected = score60
                try:
                    top1_share = float((row.get("top1_share_60m") or row.get("top1_share_window") or "0").strip() or "0")
                except ValueError:
                    top1_share = 0.0
                try:
                    new_term_rate = float(
                        (row.get("new_term_rate_60m") or row.get("new_term_rate_window") or "0").strip() or "0"
                    )
                except ValueError:
                    new_term_rate = 0.0
                try:
                    turnover = float((row.get("turnover_10m") or row.get("turnover_window") or "0").strip() or "0")
                except ValueError:
                    turnover = 0.0
                try:
                    churn = int((row.get("churn_10m") or row.get("churn_window") or "0").strip() or "0")
                except ValueError:
                    churn = 0
                try:
                    mean_rw = float(
                        (row.get("mean_rank_weighted_score_selected") or row.get("mean_rank_weighted_score_window") or row.get("mean_rank_weighted_score_60m") or "0").strip() or "0"
                    )
                except ValueError:
                    mean_rw = 0.0
                try:
                    mean_spike = float(
                        (row.get("mean_spike_ratio_10m_60m") or row.get("mean_spike_ratio_10m_window") or "0").strip() or "0"
                    )
                except ValueError:
                    mean_spike = 0.0
                try:
                    max_spike = float(
                        (row.get("max_spike_ratio_10m_60m") or row.get("max_spike_ratio_10m_window") or "0").strip() or "0"
                    )
                except ValueError:
                    max_spike = 0.0
                top_term_60m = (row.get("top_term_60m") or row.get("top_term_window") or "").strip()
                top_term_selected = (row.get("top_term_selected") or row.get("top_term_window") or top_term_60m).strip()
                try:
                    top_term_score_60m = float(
                        (row.get("top_term_score_60m") or row.get("top_term_score_window") or "0").strip() or "0"
                    )
                except ValueError:
                    top_term_score_60m = 0.0
                try:
                    top_term_score_selected = float(
                        (row.get("top_term_score_selected") or row.get("top_term_score_window") or "0").strip() or "0"
                    )
                except ValueError:
                    top_term_score_selected = top_term_score_60m
                system_state = (row.get("system_state") or "").strip()
                state_confidence = (row.get("state_confidence") or "").strip()

                points.append(
                    MetaPoint(
                        poll_id=poll_id,
                        poll_index=poll_index,
                        ts_utc=ts_utc,
                        source_timestamp=source_ts,
                        metric_window_minutes=metric_window_minutes,
                        spike_ratio_10m_60m=spike,
                        entropy_selected=entropy,
                        active_unique_terms_selected=active,
                        total_score_10m=score10,
                        total_score_60m=score60,
                        total_score_selected=score_selected,
                        top1_share_60m=top1_share,
                        new_term_rate_60m=new_term_rate,
                        turnover_10m=turnover,
                        churn_10m=churn,
                        mean_rank_weighted_score_selected=mean_rw,
                        mean_spike_ratio_10m_60m=mean_spike,
                        max_spike_ratio_10m_60m=max_spike,
                        top_term_selected=top_term_selected,
                        top_term_score_selected=top_term_score_selected,
                        top_term_60m=top_term_60m,
                        top_term_score_60m=top_term_score_60m,
                        system_state=system_state,
                        state_confidence=state_confidence,
                    )
                )

    points.sort(key=lambda p: p.ts_utc)
    return points


def load_state_transitions(path: str) -> List[StateTransitionPoint]:
    files = list_rotated_log_files(path, "state")
    if not files:
        return []
    points: List[StateTransitionPoint] = []
    for csv_file in files:
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_raw = (row.get("timestamp_utc") or "").strip()
                if not ts_raw:
                    continue
                try:
                    ts = parse_iso_utc(ts_raw)
                except Exception:
                    continue
                old_state = (row.get("old_state") or "").strip()
                new_state = (row.get("new_state") or "").strip()
                old_duration = (row.get("old_duration") or "").strip()
                conf_now = (row.get("conf_now") or "").strip()
                try:
                    conf_delta = float((row.get("conf_delta") or "0").strip() or "0")
                except ValueError:
                    conf_delta = 0.0
                conf_trend = (row.get("conf_trend") or "").strip()
                if not conf_trend:
                    # Backward compatibility for older logs.
                    if conf_delta > 0:
                        conf_trend = "rising"
                    elif conf_delta < 0:
                        conf_trend = "falling"
                    else:
                        conf_trend = "flat"
                try:
                    window_minutes = int((row.get("window_minutes") or "0").strip() or "0")
                except ValueError:
                    window_minutes = 0
                points.append(
                    StateTransitionPoint(
                        timestamp_utc=ts,
                        old_state=old_state,
                        new_state=new_state,
                        old_duration=old_duration,
                        conf_now=conf_now,
                        conf_delta=conf_delta,
                        conf_trend=conf_trend,
                        window_minutes=window_minutes,
                    )
                )
    points.sort(key=lambda p: p.timestamp_utc)
    return points


def build_render(
    snapshots: List[Snapshot],
    state: State,
    top_k: int,
    current_n: int,
    history_points: int,
    sticky_words: set[str],
    logger_status: str,
    color_mode: str,
    retain_days: int,
    gap_pct_threshold: float,
) -> str:
    size = shutil.get_terminal_size((140, 40))
    width = size.columns
    height = size.lines
    if width < MIN_TERM_COLS or height < MIN_TERM_ROWS:
        return f"Terminal too small ({width}x{height}). Enlarge to at least {MIN_TERM_COLS}x{MIN_TERM_ROWS}."

    term_width = max(18, min(36, width // 4))
    value_w = 9
    if state.metric == "spike_ratio":
        value_w = 8
    elif state.metric == "momentum":
        value_w = 9
    bar_width = max(16, min(60, width - term_width - value_w - 4))
    spark_width = max(16, min(60, width - term_width - 2))

    adjusted = [
        apply_filter_and_normalization(s, state.filter_sticky, state.normalize, sticky_words)
        for s in snapshots
    ]

    current_win, previous_win, now = windowed_snapshots(adjusted, state.window_minutes)
    rw_window = compute_rank_weighted_scores(windowed_snapshots(adjusted, state.window_minutes)[0])
    rw_60m = compute_rank_weighted_scores(windowed_snapshots(adjusted, 60)[0])
    meta = compute_meta_metrics(adjusted, state.window_minutes)
    rw_10m = compute_rank_weighted_scores(windowed_snapshots(adjusted, 10)[0])
    now_utc = adjusted[-1].ts_utc if adjusted else datetime.now(timezone.utc)

    use_color = color_enabled(color_mode)
    header_lines: List[str] = []
    selected_metric_label = short_metric_label(state.metric, width)
    normalize_part = f" | Normalize: {'ON' if state.normalize else 'OFF'}" if state.filter_sticky else ""
    header_lines.append(
        f"MW Top Lookups Display ({state.screen_mode}) | Metric: {selected_metric_label} | Window: {format_window_label(state.window_minutes)} | "
        f"FilterIgnored: {'ON' if state.filter_sticky else 'OFF'}"
        f"{normalize_part} | "
        f"Logger: {'ON' if state.log_enabled else 'OFF'}"
    )
    header_lines.append(state_status_line(state, now_utc, use_color, width))
    if not state.compact_header:
        header_lines.append(controls_line(state, use_color))
        header_lines.append(
            f"Logger: {'ON' if state.log_enabled else 'OFF'} | Rotate: daily (local) | Retain: {max(1, int(retain_days))}d"
        )
        header_lines.append(f"Logger Status: {logger_status}")
        total_selected = float(meta.get("total_score_selected", 0.0))
        agg_freshness_window = (float(meta.get("total_score_10m", 0.0)) / total_selected) if total_selected > 0 else 0.0
        header_lines.append(
            f"{short_metric_label('spike_ratio', width)} (10m/{format_window_label(state.window_minutes)}): {agg_freshness_window * 100.0:.1f}%"
        )
        header_lines.append(
            f"{short_metric_label('entropy_selected', width)} ({format_window_label(state.window_minutes)}): {meta['entropy_selected']:.2f}"
        )
        header_lines.append(
            f"{short_metric_label('active_unique_selected', width)} ({format_window_label(state.window_minutes)}): {int(meta['active_unique_selected'])}"
        )
        if state.filter_sticky:
            header_lines.append(f"Ignored terms: {len(sticky_words)}")
    header_lines.append("-" * width)
    if state.transition_banner_once:
        header_lines.append(state.transition_banner_once)

    header_lines = [fit_line(line, width) for line in header_lines]
    # Reserve one row because main loop writes rendered + "\n".
    content_rows = max(0, (height - 1) - len(header_lines))

    if not adjusted:
        out = header_lines + [fit_line("No data found yet. Start the logger and wait for CSV rows.", width)]
        return "\n".join(out[:height])

    latest = adjusted[-1]
    current_terms = sorted(latest.ranks.items(), key=lambda x: x[1])[:current_n]

    snapshot_lines: List[str] = []
    snapshot_lines.append("CURRENT SNAPSHOT")
    snapshot_lines.append(f"Latest poll: {format_local_dt(latest.ts_utc)}")
    if current_terms:
        for term, rank in current_terms:
            snapshot_lines.append(f" {rank:>2} {clip(term, width - 6)}")
    else:
        snapshot_lines.append(" (empty after active filters)")

    spike_raw_scores: Dict[str, float] = {}
    if state.metric == "spike_ratio":
        spike_raw_scores = compute_spike_ratio_scores(rw_10m, rw_60m)
        scores = dict(spike_raw_scores)
    else:
        scores = compute_scores(state.metric, current_win, previous_win)

    ranked_source = scores
    # Momentum/Freshness sort descending by value.
    if state.metric == "momentum":
        ranked = sorted(ranked_source.items(), key=lambda x: x[1], reverse=True)
    elif state.metric == "spike_ratio":
        # Freshness view: keep currently active terms and sort by freshest first.
        active_terms = {t: v for t, v in scores.items() if rw_10m.get(t, 0.0) > 0.0}
        ranked = sorted(active_terms.items(), key=lambda x: x[1], reverse=True)
    else:
        ranked = sorted(ranked_source.items(), key=lambda x: x[1], reverse=True)

    current_terms_set = {t for t, _ in current_terms}
    selected: List[Tuple[str, float]] = []
    seen = set()

    if state.metric == "momentum":
        pos = [(t, v) for t, v in ranked if v > 0]
        # Select strongest negatives first, then reorder selected negatives for display.
        neg = sorted([(t, v) for t, v in ranked if v < 0], key=lambda x: x[1])

        pos_quota = max(1, top_k // 2)
        neg_quota = top_k - pos_quota

        selected_pos = pos[:pos_quota]
        selected_neg = neg[:neg_quota]
        selected_neg = sorted(selected_neg, key=lambda x: x[1], reverse=True)
        selected.extend(selected_pos)
        selected.extend(selected_neg)

        # Backfill if one side has insufficient terms.
        if len(selected) < top_k:
            extra_pos = pos[pos_quota:]
            extra_neg = neg[neg_quota:]
            extras = extra_pos + extra_neg
            for term, score in extras:
                if len(selected) >= top_k:
                    break
                if term in {t for t, _ in selected}:
                    continue
                selected.append((term, score))

        seen = {t for t, _ in selected}
    else:
        for term, score in ranked:
            if term in seen:
                continue
            selected.append((term, score))
            seen.add(term)
            if len(selected) >= top_k:
                break

    for term in current_terms_set:
        if term not in seen:
            selected.append((term, scores.get(term, 0.0)))
            seen.add(term)

    trend_title = f"TREND ({format_window_label(state.window_minutes)})"
    trend_prefix: List[str] = ["", trend_title]
    metric_desc = metric_help_text(state.metric)
    if use_color:
        trend_prefix.append(f"Metric (M: next): \x1b[96m{metric_desc}\x1b[0m")
    else:
        trend_prefix.append(f"Metric (M: next): {metric_desc}")
    trend_item_lines: List[str] = []
    gap_glyph = choose_gap_glyph()
    history_prefix: List[str] = [
        "",
        f"HISTORY ({format_window_label(state.window_minutes)}), Scaling (S: next): {state.spark_scale}",
        f"Sparkline: █ activity | {gap_glyph} gap (missing logs)",
    ]
    history_item_lines: List[str] = []
    render_items = selected[: max(top_k, len(current_terms_set))]

    if not selected:
        trend_prefix.append(" (no terms in active window)")
    else:
        max_abs = max(abs(v) for _, v in render_items) if render_items else 1.0
        if max_abs <= 0:
            max_abs = 1.0

        # Momentum gets a diverging chart: negatives left of zero, positives right.
        if state.metric == "momentum":
            half_width = max(5, (bar_width - 1) // 2)
            for term, value in render_items:
                magnitude = int((abs(value) / max_abs) * half_width)
                term_txt = clip(term, term_width)
                if value < 0:
                    left = "░" * max(1 if value != 0 else 0, magnitude)
                    left = left.rjust(half_width)
                    right = " " * half_width
                elif value > 0:
                    left = " " * half_width
                    right = ("█" * max(1 if value != 0 else 0, magnitude)).ljust(half_width)
                else:
                    left = " " * half_width
                    right = " " * half_width
                bar = f"{left}|{right}"
                val_txt = f"{value:+8.1f}"
                trend_item_lines.append(f"{term_txt:<{term_width}} {bar} {val_txt}")
        else:
            for term, value in render_items:
                fill = int((abs(value) / max_abs) * bar_width)
                bar_char = "█" if value >= 0 else "░"
                bar = bar_char * max(1 if value != 0 else 0, fill)
                term_txt = clip(term, term_width)
                if state.metric == "spike_ratio":
                    val_txt = f"{value * 100.0:7.1f}%"
                else:
                    val_txt = f"{value:8.0f}"
                trend_item_lines.append(f"{term_txt:<{term_width}} {bar:<{bar_width}} {val_txt}")

        spark_metric = "presence" if state.metric == "presence" else "rank_weighted"
        timed_by_term: Dict[str, List[Tuple[datetime, float]]] = {}
        for term, _ in render_items:
            timed_samples: List[Tuple[datetime, float]] = []
            for s in current_win:
                rank = s.ranks.get(term)
                if rank is None:
                    val = 0.0
                elif spark_metric == "presence":
                    val = 1.0
                else:
                    n = max(s.ranks.values()) if s.ranks else rank
                    val = float(max(1, n - rank + 1))
                timed_samples.append((s.ts_utc, val))
            timed_by_term[term] = timed_samples

        global_spark_max = 0.0
        if state.spark_scale in {"global", "global-sqrt"}:
            for samples in timed_by_term.values():
                for _, v in samples:
                    if v > global_spark_max:
                        global_spark_max = v

        for term, _ in render_items:
            timed_samples = timed_by_term.get(term, [])
            history_item_lines.append(
                f"{clip(term, term_width):<{term_width}} "
                f"{sparkline_from_timed_samples(timed_samples, spark_width, gap_pct_threshold, gap_glyph, state.spark_scale, global_spark_max)}"
            )

    def trim_section(section: List[str], budget: int) -> List[str]:
        if budget <= 0:
            return []
        if len(section) <= budget:
            return section
        if budget == 1:
            return [section[0]]
        return section[:budget]

    out_lines: List[str] = list(header_lines)
    remaining = content_rows

    snap_chunk = trim_section(snapshot_lines, remaining)
    out_lines.extend(snap_chunk)
    remaining -= len(snap_chunk)

    if remaining > 0:
        if not selected:
            t_chunk = trim_section(trend_prefix, remaining)
            out_lines.extend(t_chunk)
            remaining -= len(t_chunk)
        else:
            # Couple TREND/HISTORY item counts. Show HISTORY only if >=3 paired rows fit.
            min_with_history = len(trend_prefix) + len(history_prefix) + 2 * 3
            can_show_history = remaining >= min_with_history and len(history_item_lines) >= 3 and len(trend_item_lines) >= 3

            if can_show_history:
                pair_budget = (remaining - len(trend_prefix) - len(history_prefix)) // 2
                pair_count = max(0, min(pair_budget, len(trend_item_lines), len(history_item_lines)))
                if pair_count >= 3:
                    out_lines.extend(trend_prefix)
                    out_lines.extend(trend_item_lines[:pair_count])
                    out_lines.extend(history_prefix)
                    out_lines.extend(history_item_lines[:pair_count])
                else:
                    can_show_history = False

            if not can_show_history:
                t_budget = max(0, remaining - len(trend_prefix))
                t_count = min(len(trend_item_lines), t_budget)
                out_lines.extend(trend_prefix)
                out_lines.extend(trend_item_lines[:t_count])

    return "\n".join(fit_line(line, width) for line in out_lines[: max(0, height - 1)])


def build_render_meta(
    meta_points: List[MetaPoint],
    transitions: List[StateTransitionPoint],
    snapshots: List[Snapshot],
    state: State,
    history_points: int,
    sticky_words: set[str],
    logger_status: str,
    color_mode: str,
    retain_days: int,
    gap_pct_threshold: float,
    spark_scale: str,
) -> str:
    size = shutil.get_terminal_size((140, 40))
    width = size.columns
    height = size.lines
    if width < MIN_TERM_COLS or height < MIN_TERM_ROWS:
        return f"Terminal too small ({width}x{height}). Enlarge to at least {MIN_TERM_COLS}x{MIN_TERM_ROWS}."

    adjusted = [
        apply_filter_and_normalization(s, state.filter_sticky, state.normalize, sticky_words)
        for s in snapshots
    ]
    now_utc = adjusted[-1].ts_utc if adjusted else datetime.now(timezone.utc)
    current_meta = compute_meta_metrics(adjusted, state.window_minutes)
    scoped_points = [p for p in meta_points if p.metric_window_minutes == state.window_minutes]
    focus_meta_metric = current_meta_help_metric(state)
    use_color = color_enabled(color_mode)

    # Prefer logged values for trend continuity; fallback to computed when logs are empty.
    latest = scoped_points[-1] if scoped_points else None
    prev = scoped_points[-2] if len(scoped_points) > 1 else None

    def current_value(metric_id: str) -> float:
        if latest is None:
            fallback = {
                "entropy": float(current_meta["entropy_selected"]),
                "top1_share": float(current_meta["top1_share_60m"]),
                "new_term_rate": float(current_meta["new_term_rate_60m"]),
                "turnover": float(current_meta["turnover_10m"]),
                "mean_spike": float(current_meta["mean_spike_ratio_10m_60m"]),
                "max_spike": float(current_meta["max_spike_ratio_10m_60m"]),
                "active_terms": float(current_meta["active_unique_selected"]),
            }
            return fallback[metric_id]

        lookup = {
            "entropy": float(latest.entropy_selected),
            "top1_share": float(latest.top1_share_60m),
            "new_term_rate": float(latest.new_term_rate_60m),
            "turnover": float(latest.turnover_10m),
            "mean_spike": float(latest.mean_spike_ratio_10m_60m),
            "max_spike": float(latest.max_spike_ratio_10m_60m),
            "active_terms": float(latest.active_unique_terms_selected),
        }
        return lookup[metric_id]

    selected_win_label = format_window_label(state.window_minutes)
    metric_rows = [
        ("entropy", f"{metric_label('entropy')} ({selected_win_label}):", "{:7.3f}", "entropy_selected"),
        ("top1_share", f"{metric_label('top1_share')} (60m):", "{:7.3f}", "top1_share_60m"),
        ("new_term_rate", f"{metric_label('new_term_rate')} (60m):", "{:7.3f}", "new_term_rate_60m"),
        ("turnover", f"{metric_label('turnover')} (10m):", "{:7.3f}", "turnover_10m"),
        ("mean_spike", f"{metric_label('mean_spike_ratio')} (10m/60m):", "{:7.3f}", "mean_spike_ratio_10m_60m"),
        ("max_spike", f"{metric_label('max_spike_ratio')} (10m/60m):", "{:7.3f}", "max_spike_ratio_10m_60m"),
        ("active_terms", f"{metric_label('active_unique_selected')} ({selected_win_label}):", "{:7.0f}", "active_unique_terms_selected"),
    ]
    max_label = max(len(label) for _, label, _, _ in metric_rows)
    label_w = max(14, min(max_label + 1, 28))
    value_w = 9
    delta_w = 10
    spark_w = max(16, min(60, width - (2 + label_w + 1 + value_w + 3 + delta_w + 3)))

    # Keep metric columns stable across modes so sparks begin at the same column.
    row_prefix = "  "
    fixed_cols = len(row_prefix) + 2 + label_w + 1 + value_w + 3 + delta_w + 3
    if fixed_cols + spark_w > width:
        label_w = max(12, label_w - ((fixed_cols + spark_w) - width))

    gap_glyph = choose_gap_glyph()

    def series_samples_from_points(attr: str) -> List[Tuple[datetime, float]]:
        return [(p.ts_utc, float(getattr(p, attr))) for p in scoped_points]

    header_lines: List[str] = []
    normalize_part = f" | Normalize: {'ON' if state.normalize else 'OFF'}" if state.filter_sticky else ""
    header_lines.append(
        f"MW Top Lookups Display ({state.screen_mode}) | Window: {format_window_label(state.window_minutes)} | "
        f"FilterIgnored: {'ON' if state.filter_sticky else 'OFF'}"
        f"{normalize_part} | "
        f"Logger: {'ON' if state.log_enabled else 'OFF'}"
    )
    header_lines.append(state_status_line(state, now_utc, use_color, width))
    if not state.compact_header:
        header_lines.append(controls_line(state, use_color))
        header_lines.append(
            f"Logger: {'ON' if state.log_enabled else 'OFF'} | Rotate: daily (local) | Retain: {max(1, int(retain_days))}d"
        )
        header_lines.append(f"Logger Status: {logger_status}")
    header_lines.append("-" * width)
    header_lines = [fit_line(line, width) for line in header_lines]
    # Reserve one row because main loop writes rendered + "\n".
    content_rows = max(0, (height - 1) - len(header_lines))

    body_lines: List[str] = []
    body_lines.append("META SNAPSHOT")
    body_lines.append("")
    body_lines.append(f"Sparkline: █ activity | {gap_glyph} gap (missing logs)")
    body_lines.append("")
    for idx, (metric_id, label, fmt, attr) in enumerate(metric_rows):
        val = current_value(metric_id)
        if prev is None:
            delta = 0.0
        else:
            prev_map = {
                "entropy": float(prev.entropy_selected),
                "top1_share": float(prev.top1_share_60m),
                "new_term_rate": float(prev.new_term_rate_60m),
                "turnover": float(prev.turnover_10m),
                "mean_spike": float(prev.mean_spike_ratio_10m_60m),
                "max_spike": float(prev.max_spike_ratio_10m_60m),
                "active_terms": float(prev.active_unique_terms_selected),
            }
            delta = val - prev_map[metric_id]
        spark = sparkline_from_timed_samples(
            series_samples_from_points(attr),
            spark_w,
            gap_pct_threshold,
            gap_glyph,
            "per-term",
            0.0,
        )
        if metric_id == "active_terms":
            delta_txt = f"Δ {delta:+.0f}"
        else:
            delta_txt = f"Δ {delta:+.3f}"
        alias_map = {
            "entropy": "entropy_selected",
            "top1_share": "top1_share_60m",
            "new_term_rate": "new_term_rate_60m",
            "turnover": "turnover_10m",
            "mean_spike": "mean_spike_ratio_10m_60m",
            "max_spike": "max_spike_ratio_10m_60m",
            "active_terms": "active_unique_selected",
        }
        key_for_help = alias_map.get(metric_id, metric_id)
        is_focus = (key_for_help == focus_meta_metric) and (not state.compact_header)
        mark = "  "
        if use_color and is_focus:
            mark = "\x1b[96;1m▶▶\x1b[0m"
        elif is_focus:
            mark = ">>"
        body_lines.append(
            f"{row_prefix}{mark} {label:<{label_w}} {fmt.format(val):>{value_w}}   {delta_txt:>{delta_w}}   {spark}"
        )
        if idx < len(metric_rows) - 1:
            body_lines.append("")

    # Keep metric help adjacent to the metric block (not in header/footer).
    body_lines.append("")
    body_lines.append(
        colored_metric_help_line_with_prompt(
            focus_meta_metric,
            width,
            use_color,
            "Metric (M: next): ",
            96,
        )
    )
    body_lines.append("")

    labels, adaptive = derive_metric_labels(current_meta, scoped_points)
    flags: List[str] = []
    if labels["N"] == "high":
        flags.append("HIGH NOVELTY")
    if labels["E"] == "low":
        flags.append("LOW ENTROPY")
    if labels["C"] == "high":
        flags.append("HIGH CONCENTRATION")
    if labels["T"] == "high":
        flags.append("HIGH TURNOVER")
    if labels["S"] == "high":
        flags.append("HIGH SPIKE")

    if flags:
        body_lines.append("ANOMALY FLAGS:")
        for flag in flags:
            body_lines.append(f" • {flag}")

    system_state, _, _, _ = classify_system_state(current_meta, scoped_points)
    summary_map = {
        "Distributed curiosity": "Entropy high + novelty high/normal + concentration low + turnover normal.",
        "Meme turbulence": "Turnover high + spike/novelty high with concentration low/normal.",
        "Narrative convergence": "Entropy low + concentration high + novelty low + turnover low.",
        "Reinforcement plateau": "Concentration normal/high + entropy normal + novelty low + turnover low/normal.",
    }
    method = "adaptive" if adaptive else "fallback"
    if not body_lines or body_lines[-1] != "":
        body_lines.append("")
    body_lines.append(f"SUMMARY: {summary_map.get(system_state, system_state)} ({system_state}, thresholds={method})")

    # Append as many recent state changes as fit.
    remaining_rows = max(0, content_rows - len(body_lines))
    if remaining_rows >= 2:
        changes_header = ["", "RECENT STATE CHANGES"]
        remaining_rows -= len(changes_header)
        change_lines: List[str] = []
        use_color = color_enabled(color_mode)
        scoped = [t for t in transitions if t.window_minutes == state.window_minutes]
        source = scoped if scoped else transitions
        for t in reversed(source):
            from_dot = "●"
            to_dot = "●"
            if use_color:
                from_code = STATE_COLOR_CODES.get(t.old_state, 37)
                to_code = STATE_COLOR_CODES.get(t.new_state, 37)
                from_dot = f"\x1b[{from_code}m●\x1b[0m"
                to_dot = f"\x1b[{to_code}m●\x1b[0m"
            ts_short = t.timestamp_utc.astimezone().strftime("%H:%M:%S")
            conf_part = t.conf_now or "unknown"
            conf_part = re.sub(r"\s*\((?:rising|falling|flat)\)\s*$", "", conf_part, flags=re.IGNORECASE)
            trend_raw = (t.conf_trend or "flat").lower()
            if trend_raw == "rising":
                trend_part = "▲"
            elif trend_raw == "falling":
                trend_part = "▼"
            else:
                trend_part = "●"
            w_part = format_window_label(t.window_minutes) if t.window_minutes > 0 else format_window_label(state.window_minutes)
            line = (
                f"{ts_short} {from_dot} {t.old_state} -> {to_dot} {t.new_state} ({t.old_duration}) | "
                f"conf={conf_part} | {trend_part} | w={w_part}"
            )
            change_lines.append(line)
            if len(change_lines) >= remaining_rows:
                break
        if change_lines:
            body_lines.extend(changes_header)
            body_lines.extend(change_lines)
        else:
            body_lines.extend(changes_header)
            body_lines.append("No state transitions logged yet.")

    if content_rows > 0:
        out = header_lines + body_lines[:content_rows]
    else:
        out = header_lines
    return "\n".join(fit_line(line, width) for line in out[: max(0, height - 1)])


def read_key() -> str | None:
    if os.name == "nt":
        try:
            import msvcrt  # type: ignore
        except ImportError:
            return None

        if not msvcrt.kbhit():
            return None

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            # Special keys are two-char sequences on Windows consoles.
            if not msvcrt.kbhit():
                return None
            special = msvcrt.getwch()
            if special == "H":
                return "up"
            if special == "P":
                return "down"
            return None
        if ch == "\x1b":
            # ANSI escape sequence, common in Windows Terminal / ConPTY:
            # ESC [ A (up), ESC [ B (down)
            if msvcrt.kbhit():
                c2 = msvcrt.getwch()
                if c2 == "[" and msvcrt.kbhit():
                    c3 = msvcrt.getwch()
                    if c3 == "A":
                        return "up"
                    if c3 == "B":
                        return "down"
            return None
        return ch.lower()

    # POSIX fallback: non-blocking not implemented for simplicity.
    return None


def apply_key(state: State, key: str) -> bool:
    if key == "q":
        return False
    if key in {"up", "+", "="}:
        state.window_idx = min(len(WINDOW_PRESETS_MINUTES) - 1, state.window_idx + 1)
    elif key in {"down", "-", "_"}:
        state.window_idx = max(0, state.window_idx - 1)
    elif key == "f":
        state.filter_sticky = not state.filter_sticky
        if not state.filter_sticky:
            state.normalize = False
    elif key == "m":
        if state.screen_mode == "meta":
            if META_HELP_METRICS:
                state.meta_help_idx = (state.meta_help_idx + 1) % len(META_HELP_METRICS)
        else:
            state.metric_idx = (state.metric_idx + 1) % len(METRICS)
    elif key == "s":
        try:
            idx = SPARK_SCALE_MODES.index(state.spark_scale)
        except ValueError:
            idx = 0
        state.spark_scale = SPARK_SCALE_MODES[(idx + 1) % len(SPARK_SCALE_MODES)]
    elif key == "c":
        state.compact_header = not state.compact_header
    elif key == "n":
        if state.filter_sticky:
            state.normalize = not state.normalize
    elif key == "v":
        state.screen_mode = "meta" if state.screen_mode == "main" else "main"
    return True


def main() -> int:
    global WINDOW_PRESETS_MINUTES
    args = parse_args()
    retain_days = max(1, int(args.retain_days))
    WINDOW_PRESETS_MINUTES = build_window_presets(retain_days)

    log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    args.csv = args.csv or os.path.join(log_dir, "terms.csv")
    args.meta_csv = args.meta_csv or os.path.join(log_dir, "meta.csv")
    args.state_log_csv = args.state_log_csv or os.path.join(log_dir, "state.csv")

    try:
        # Improve Unicode rendering for blocks/sparklines on Windows terminals.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    sticky_words = set(STICKY_WORDS_DEFAULT)
    sticky_words.update({normalize_term(w) for w in args.ignore_words if w.strip()})

    startup_day = local_day_str()
    rotate_legacy_base_file(args.csv, "terms", startup_day)
    rotate_legacy_base_file(args.meta_csv, "meta", startup_day)
    rotate_legacy_base_file(args.state_log_csv, "state", startup_day)

    state = State(log_enabled=True)
    state.spark_scale = args.spark_scale
    if 60 in WINDOW_PRESETS_MINUTES:
        state.window_idx = WINDOW_PRESETS_MINUTES.index(60)
    state.last_state_window_minutes = state.window_minutes
    poll_index = get_max_poll_index(args.csv)
    today_local = datetime.now().astimezone()
    keep_days = retain_days
    enforce_retention_for_type(args.csv, "terms", keep_days, today_local)
    enforce_retention_for_type(args.meta_csv, "meta", keep_days, today_local)
    enforce_retention_for_type(args.state_log_csv, "state", keep_days, today_local)
    state.retention_last_run_local = local_day_str(today_local)
    next_log_at = time.time()
    logger_status = "Idle"

    running = True
    while running:
        now_local = datetime.now().astimezone()
        today_str = local_day_str(now_local)
        if state.retention_last_run_local != today_str:
            deleted = 0
            deleted += enforce_retention_for_type(args.csv, "terms", keep_days, now_local)
            deleted += enforce_retention_for_type(args.meta_csv, "meta", keep_days, now_local)
            deleted += enforce_retention_for_type(args.state_log_csv, "state", keep_days, now_local)
            state.retention_last_run_local = today_str
            if deleted > 0:
                logger_status = f"Retention cleanup removed {deleted} old file(s)"

        snapshots = load_snapshots(args.csv)
        meta_points = load_meta_points(args.meta_csv)
        transitions = load_state_transitions(args.state_log_csv)

        now_ts = time.time()
        if now_ts >= next_log_at:
            try:
                payload = fetch_top_lookups(
                    endpoint=args.log_endpoint,
                    timeout=max(1, int(args.log_timeout)),
                    retries=max(1, int(args.log_retries)),
                    retry_delay=max(0.1, float(args.log_retry_delay)),
                )
                words = payload["words"]  # type: ignore[assignment]
                source_timestamp = str(payload.get("source_timestamp", ""))
                source_timestamp_local = format_source_timestamp_local(source_timestamp)
                latest = snapshots[-1] if snapshots else None
                if latest_signature(latest) == payload_signature(source_timestamp, list(words)):  # type: ignore[arg-type]
                    src = source_timestamp_local or source_timestamp
                    logger_status = f"Skipped duplicate poll ({src})"
                else:
                    poll_index += 1
                    terms_path = rotated_log_path(args.csv, "terms", local_day_str())
                    poll_result = append_poll_to_csv(
                        csv_path=terms_path,
                        poll_index=poll_index,
                        endpoint=args.log_endpoint,
                        words=list(words),  # type: ignore[arg-type]
                        source_timestamp=source_timestamp,
                    )
                    row_count = int(poll_result.get("row_count", 0))
                    snapshots = load_snapshots(args.csv)
                    meta_metrics_window = compute_meta_metrics(snapshots, state.window_minutes)
                    scoped = [p for p in meta_points if p.metric_window_minutes == state.window_minutes]
                    state_name, state_conf, _, _ = classify_system_state(meta_metrics_window, scoped)
                    meta_metrics_window["system_state"] = state_name
                    meta_metrics_window["state_confidence"] = state_conf
                    meta_path = rotated_log_path(args.meta_csv, "meta", local_day_str())
                    append_meta_to_csv(meta_path, poll_result, meta_metrics_window)
                    meta_points = load_meta_points(args.meta_csv)
                    logger_status = (
                        f"Logged {row_count} term rows + 1 meta row at {datetime.now().strftime('%H:%M:%S')} "
                        f"({(source_timestamp_local or source_timestamp)}, window={format_window_label(state.window_minutes)})"
                    )
            except Exception as exc:
                logger_status = f"Log error: {exc}"
            finally:
                next_log_at = time.time() + max(1.0, float(args.log_interval))

        # Runtime state machine (duration, drift, anti-chatter, transition banner/logging).
        if snapshots:
            adjusted_for_state = [
                apply_filter_and_normalization(s, state.filter_sticky, state.normalize, sticky_words)
                for s in snapshots
            ]
            metrics_for_state = compute_meta_metrics(adjusted_for_state, state.window_minutes)
            scoped_for_state = [p for p in meta_points if p.metric_window_minutes == state.window_minutes]
            cand_state, cand_conf, cand_conf_value, adaptive = classify_system_state(metrics_for_state, scoped_for_state)
            threshold_mode = "adaptive" if adaptive else "fallback"
            point_key = point_key_for_window(snapshots, meta_points, state.window_minutes)
            now_state = adjusted_for_state[-1].ts_utc if adjusted_for_state else datetime.now(timezone.utc)
            update_runtime_state(
                state=state,
                candidate_state=cand_state,
                conf_label=cand_conf,
                conf_value=cand_conf_value,
                threshold_mode=threshold_mode,
                now_utc=now_state,
                point_key=point_key,
                logger_enabled=True,
                transition_csv_path=rotated_log_path(args.state_log_csv, "state", local_day_str()),
            )

        if state.screen_mode == "meta":
            rendered = build_render_meta(
                meta_points=meta_points,
                transitions=transitions,
                snapshots=snapshots,
                state=state,
                history_points=max(1, args.history_points),
                sticky_words=sticky_words,
                logger_status=logger_status,
                color_mode=args.color,
                retain_days=max(1, int(args.retain_days)),
                gap_pct_threshold=max(0.0, float(args.gap_pct_threshold)),
                spark_scale=state.spark_scale,
            )
        else:
            rendered = build_render(
                snapshots=snapshots,
                state=state,
                top_k=max(1, args.top_k),
                current_n=max(1, args.current_n),
                history_points=max(1, args.history_points),
                sticky_words=sticky_words,
                logger_status=logger_status,
                color_mode=args.color,
                retain_days=max(1, int(args.retain_days)),
                gap_pct_threshold=max(0.0, float(args.gap_pct_threshold)),
            )

        clear_screen()
        sys.stdout.write(rendered + "\n")
        sys.stdout.flush()
        if state.screen_mode == "main" and state.transition_banner_once:
            state.transition_banner_once = ""

        if args.once:
            break

        deadline = time.time() + max(1.0, args.refresh)
        while time.time() < deadline:
            key = read_key()
            if key:
                if key == "a":
                    state.window_idx = choose_window_idx_for_span(dataset_span_minutes(snapshots))
                    key = None
                if key is None:
                    # Redraw on next outer-loop iteration.
                    break
                running = apply_key(state, key)
                if not running:
                    break
                # Redraw immediately after any handled key (also refreshes terminal-size layout).
                break
            time.sleep(0.05)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
