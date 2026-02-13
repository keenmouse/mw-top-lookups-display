#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")


def load_app_module(repo_root: Path):
    app_path = repo_root / "MWTopLookupsDisplay.py"
    spec = importlib.util.spec_from_file_location("mw_display", str(app_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {app_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def ansi_color_map() -> dict[int, Tuple[int, int, int]]:
    # Basic 16-color ANSI map tuned for dark backgrounds.
    return {
        30: (0, 0, 0),
        31: (220, 95, 95),
        32: (120, 200, 120),
        33: (225, 190, 95),
        34: (110, 155, 235),
        35: (200, 135, 235),
        36: (95, 205, 205),
        37: (220, 220, 220),
        90: (120, 120, 120),
        91: (255, 120, 120),
        92: (140, 230, 140),
        93: (245, 220, 130),
        94: (150, 190, 255),
        95: (225, 170, 255),
        96: (140, 235, 235),
        97: (255, 255, 255),
    }


def parse_ansi_segments(line: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    color_lookup = ansi_color_map()
    default = color_lookup[37]
    segments: List[Tuple[str, Tuple[int, int, int]]] = []
    current = default
    idx = 0
    for m in ANSI_RE.finditer(line):
        if m.start() > idx:
            segments.append((line[idx:m.start()], current))
        codes = [c for c in m.group(1).split(";") if c]
        if not codes:
            codes = ["0"]
        for code_str in codes:
            try:
                code = int(code_str)
            except ValueError:
                continue
            if code == 0:
                current = default
            elif code in color_lookup:
                current = color_lookup[code]
            elif code == 1:
                # Bold: brighten current color slightly.
                current = tuple(min(255, int(v * 1.1 + 8)) for v in current)
        idx = m.end()
    if idx < len(line):
        segments.append((line[idx:], current))
    return segments


def visible_line(line: str) -> str:
    return ANSI_RE.sub("", line).replace("\t", "    ")


def load_font_candidates(size: int = 20) -> List[ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    paths = [
        r"C:\Windows\Fonts\CascadiaMono.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
        r"C:\Windows\Fonts\seguisym.ttf",  # Segoe UI Symbol
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    fonts: List[ImageFont.FreeTypeFont | ImageFont.ImageFont] = []
    for path in paths:
        if os.path.exists(path):
            try:
                fonts.append(ImageFont.truetype(path, size))
            except Exception:
                pass
    if not fonts:
        fonts.append(ImageFont.load_default())
    return fonts


def has_glyph(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, ch: str) -> bool:
    try:
        mask = font.getmask(ch)
        return mask.getbbox() is not None
    except Exception:
        return False


def choose_font_for_char(
    ch: str, fonts: List[ImageFont.FreeTypeFont | ImageFont.ImageFont]
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Prefer monospace primary first; fallback to symbol fonts only if needed.
    for font in fonts:
        if has_glyph(font, ch):
            return font
    return fonts[0]


def render_terminal_frame(text: str, out_path: Path, title: str) -> None:
    fonts = load_font_candidates(20)
    primary_font = fonts[0]
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    ascent, descent = primary_font.getmetrics() if hasattr(primary_font, "getmetrics") else (16, 5)
    line_h = ascent + descent + 8
    char_w = int(draw_probe.textlength("M", font=primary_font))
    if char_w <= 0:
        char_w = 10

    lines = text.splitlines() or ["(no output)"]
    vis_lines = [visible_line(line) for line in lines]
    max_cols = max((len(v) for v in vis_lines), default=1)

    pad_x = 24
    pad_y = 20
    title_h = line_h + 12
    width = max(900, pad_x * 2 + max_cols * char_w)
    height = pad_y * 2 + title_h + len(lines) * line_h

    bg = (17, 23, 30)
    header_bg = (27, 34, 45)
    text_default = (225, 230, 235)
    title_color = (178, 216, 255)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, pad_y + title_h], fill=header_bg)
    draw.text((pad_x, pad_y // 2), title, fill=title_color, font=primary_font)

    y = pad_y + title_h
    for raw in lines:
        x = pad_x
        segments = parse_ansi_segments(raw)
        if not segments:
            segments = [(visible_line(raw), text_default)]
        col_idx = 0
        for seg_text, color in segments:
            if not seg_text:
                continue
            seg_text = seg_text.replace("\t", "    ")
            for ch in seg_text:
                font = choose_font_for_char(ch, fonts)
                draw.text((x + col_idx * char_w, y), ch, fill=color, font=font)
                col_idx += 1
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    mod = load_app_module(repo_root)

    local_appdata = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    log_dir = Path(local_appdata) / "MWTopLookups" / "logs"
    terms = log_dir / "terms.csv"
    meta = log_dir / "meta.csv"
    state_path = log_dir / "state.csv"

    snapshots = mod.load_snapshots(str(terms))
    meta_points = mod.load_meta_points(str(meta))
    transitions = mod.load_state_transitions(str(state_path))

    def initialize_runtime_state(s: object) -> None:
        # Match app startup behavior for historical continuity in headers.
        scoped = [p for p in meta_points if p.metric_window_minutes == s.window_minutes]
        threshold_mode = "adaptive" if len(scoped) >= 60 else "fallback"
        restored = mod.bootstrap_state_from_transitions(
            state=s,
            transitions=transitions,
            window_minutes=s.window_minutes,
            threshold_mode=threshold_mode,
        )
        if restored:
            return
        # Fallback: compute current state from available data if no transition history exists.
        adjusted = [mod.apply_filter_and_normalization(x, s.filter_sticky, s.normalize, set()) for x in snapshots]
        if adjusted:
            current_meta = mod.compute_meta_metrics(adjusted, s.window_minutes)
            state_name, conf_label, conf_value, adaptive = mod.classify_system_state(current_meta, scoped)
            s.current_state = state_name
            s.current_conf_label = conf_label
            s.current_conf_value = conf_value
            s.current_threshold_mode = "adaptive" if adaptive else "fallback"
            s.state_started_at = adjusted[-1].ts_utc

    state_main = mod.State()
    if 60 in mod.WINDOW_PRESETS_MINUTES:
        state_main.window_idx = mod.WINDOW_PRESETS_MINUTES.index(60)
    state_main.screen_mode = "main"
    state_main.compact_header = False
    initialize_runtime_state(state_main)
    main_text = mod.build_render(
        snapshots=snapshots,
        state=state_main,
        top_k=15,
        current_n=10,
        history_points=20,
        sticky_words=set(),
        logger_status="Running",
        color_mode="on",
        retain_days=7,
        gap_pct_threshold=0.5,
    )

    state_meta = mod.State()
    if 60 in mod.WINDOW_PRESETS_MINUTES:
        state_meta.window_idx = mod.WINDOW_PRESETS_MINUTES.index(60)
    state_meta.screen_mode = "meta"
    state_meta.compact_header = False
    initialize_runtime_state(state_meta)
    meta_text = mod.build_render_meta(
        meta_points=meta_points,
        transitions=transitions,
        snapshots=snapshots,
        state=state_meta,
        history_points=20,
        sticky_words=set(),
        logger_status="Running",
        color_mode="on",
        retain_days=7,
        gap_pct_threshold=0.5,
        spark_scale="global-sqrt",
        adaptive_cap_hours=24.0,
    )

    out_dir = repo_root / "docs" / "images"
    render_terminal_frame(main_text, out_dir / "main-view.png", "MW Top Lookups Display - Main View")
    render_terminal_frame(meta_text, out_dir / "meta-view.png", "MW Top Lookups Display - Meta View")
    print(f"Wrote {out_dir / 'main-view.png'}")
    print(f"Wrote {out_dir / 'meta-view.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
