from pathlib import Path
import random
import struct
import zlib

import numpy as np
import pandas as pd


SEED = 53197
WIDTH = 720
HEIGHT = 480
N_SCENES = 44

COLORS = {
    "white": (242, 245, 248),
    "panel": (34, 42, 50),
    "panel_dark": (19, 24, 31),
    "grid": (69, 83, 96),
    "slot": (99, 108, 118),
    "blue": (64, 144, 255),
    "amber": (255, 178, 58),
    "green": (40, 208, 120),
    "red": (238, 76, 76),
    "cyan": (67, 220, 232),
    "yellow": (250, 226, 72),
    "magenta": (244, 70, 215),
    "orange": (255, 130, 42),
    "purple": (168, 108, 250),
    "gray": (154, 164, 173),
    "black": (8, 10, 14),
}

WAFER_COLORS = ["blue", "amber", "green", "red", "cyan"]
LED_COLORS = ["red", "amber", "green", "blue", "cyan"]
CABLE_COLORS = ["yellow", "cyan", "purple", "orange", "green"]
MODULES = ["A", "B", "C", "D", "E"]
PORTS = ["P1", "P2", "P3", "P4", "P5"]
ZONES = ["upper-left", "upper-right", "center", "lower-left", "lower-right"]


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["10010", "10010", "10010", "11111", "00010", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01111", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "11110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}

SEGMENTS = {
    0: "abcedf",
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgecd",
    7: "abc",
    8: "abcdefg",
    9: "abfgcd",
}


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].astype(np.uint8).tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")
    path.write_bytes(png)


def color(name: str) -> np.ndarray:
    return np.array(COLORS[name], dtype=np.float32)


def rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, c: str | tuple[int, int, int]) -> None:
    col = np.array(COLORS[c] if isinstance(c, str) else c, dtype=np.uint8)
    x0, x1 = max(0, x0), min(WIDTH, x1)
    y0, y1 = max(0, y0), min(HEIGHT, y1)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = col


def blend_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, c: str, alpha: float) -> None:
    x0, x1 = max(0, x0), min(WIDTH, x1)
    y0, y1 = max(0, y0), min(HEIGHT, y1)
    if x0 >= x1 or y0 >= y1:
        return
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1].astype(np.float32) * (1 - alpha) + color(c) * alpha).clip(0, 255).astype(np.uint8)


def circle(img: np.ndarray, cx: int, cy: int, r: int, c: str) -> None:
    x0, x1 = max(0, cx - r), min(WIDTH, cx + r + 1)
    y0, y1 = max(0, cy - r), min(HEIGHT, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= r * r
    patch = img[y0:y1, x0:x1]
    patch[mask] = np.array(COLORS[c], dtype=np.uint8)


def line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, c: str, thickness: int = 3) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for t in np.linspace(0, 1, steps + 1):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        circle(img, x, y, thickness, c)


def text(img: np.ndarray, x: int, y: int, s: int, msg: str, c: str = "white") -> None:
    cursor = x
    for ch in msg.upper():
        glyph = FONT.get(ch, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    rect(img, cursor + gx * s, y + gy * s, cursor + (gx + 1) * s, y + (gy + 1) * s, c)
        cursor += 6 * s


def seg_boxes(x: int, y: int, s: int) -> dict[str, tuple[int, int, int, int]]:
    t = max(2, s // 2)
    w = 5 * s
    h = 9 * s
    return {
        "a": (x + t, y, x + w - t, y + t),
        "b": (x + w - t, y + t, x + w, y + h // 2 - t),
        "c": (x + w - t, y + h // 2 + t, x + w, y + h - t),
        "d": (x + t, y + h - t, x + w - t, y + h),
        "e": (x, y + h // 2 + t, x + t, y + h - t),
        "f": (x, y + t, x + t, y + h // 2 - t),
        "g": (x + t, y + h // 2 - t // 2, x + w - t, y + h // 2 + t // 2),
    }


def seven_digit(img: np.ndarray, x: int, y: int, digit: int, s: int, c: str) -> None:
    boxes = seg_boxes(x, y, s)
    for seg in SEGMENTS[int(digit)]:
        rect(img, *boxes[seg], c)


def seven_number(img: np.ndarray, x: int, y: int, value: int, s: int, c: str, digits: int = 2) -> None:
    label = str(value).zfill(digits)
    for i, ch in enumerate(label):
        seven_digit(img, x + i * (6 * s), y, int(ch), s, c)


def make_options(answer: str, candidates: list[str], rng: random.Random) -> tuple[dict[str, str], str]:
    values = [answer] + [c for c in candidates if c != answer]
    values = values[:5]
    rng.shuffle(values)
    letters = ["A", "B", "C", "D", "E"]
    opts = dict(zip(letters, values))
    answer_letter = next(k for k, v in opts.items() if v == answer)
    return opts, answer_letter


def overlaps_any(rect_xyxy: tuple[int, int, int, int], regions: list[tuple[int, int, int, int]]) -> bool:
    x0, y0, x1, y1 = rect_xyxy
    for rx0, ry0, rx1, ry1 in regions:
        if x0 < rx1 and x1 > rx0 and y0 < ry1 and y1 > ry0:
            return True
    return False


def point_in_regions(x: int, y: int, regions: list[tuple[int, int, int, int]]) -> bool:
    return any(rx0 <= x <= rx1 and ry0 <= y <= ry1 for rx0, ry0, rx1, ry1 in regions)


def safe_point(np_rng: np.random.Generator, protected: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    for _ in range(80):
        cx = int(np_rng.integers(25, WIDTH - 25))
        cy = int(np_rng.integers(25, HEIGHT - 25))
        if not point_in_regions(cx, cy, protected):
            return cx, cy
    return int(np_rng.integers(25, WIDTH - 25)), int(np_rng.integers(25, HEIGHT - 25))


def safe_rect(
    np_rng: np.random.Generator,
    protected: list[tuple[int, int, int, int]],
    width_range: tuple[int, int],
    height_range: tuple[int, int],
    x_range: tuple[int, int] = (30, WIDTH - 30),
    y_range: tuple[int, int] = (35, HEIGHT - 45),
) -> tuple[int, int, int, int]:
    for _ in range(120):
        w = int(np_rng.integers(width_range[0], width_range[1] + 1))
        h = int(np_rng.integers(height_range[0], height_range[1] + 1))
        x0 = int(np_rng.integers(x_range[0], max(x_range[0] + 1, x_range[1] - w)))
        y0 = int(np_rng.integers(y_range[0], max(y_range[0] + 1, y_range[1] - h)))
        candidate = (x0, y0, x0 + w, y0 + h)
        if not overlaps_any(candidate, protected):
            return candidate
    return (34, HEIGHT - 38, WIDTH - 34, HEIGHT - 14)


def split_group(scene_idx: int, rng: random.Random) -> tuple[str, str, str, str, str]:
    if scene_idx < 16:
        return "standard_holdout", "standard", "normal", "low", "easy"
    if scene_idx < 28:
        lighting = rng.choice(["normal", "blue_cast", "low_light"])
        axis = "low_light" if lighting == "low_light" else "standard"
        return "standard_holdout", axis, lighting, "medium", "medium"
    if scene_idx < 32:
        return "ood_low_light", "low_light", "low_light", "medium", "hard"
    if scene_idx < 36:
        return "ood_glare", "glare", "glare", "medium", "hard"
    if scene_idx < 40:
        return "ood_occlusion", "occlusion", "normal", "high", "hard"
    return "ood_dense_clutter", "dense_clutter", rng.choice(["normal", "blue_cast"]), "high", "hard"


def scene_trace(scene_idx: int, rng: random.Random) -> dict:
    group, axis, lighting, clutter, difficulty = split_group(scene_idx, rng)
    slot_colors = rng.sample(WAFER_COLORS, 5)
    slot_map = {i + 1: slot_colors[i] for i in range(5)}
    slot_map[6] = "gray"
    led_map = dict(zip(MODULES, rng.sample(LED_COLORS, 5)))
    cable_map = dict(zip(PORTS, rng.sample(CABLE_COLORS, 5)))
    route = rng.sample(["1", "2", "3", "4", "5"], 5)
    route_query = rng.choice(route[:-1])
    layout = rng.choice(["left_cassette", "right_cassette", "wide_ports", "compact_panel"])
    return {
        "scene_id": f"scene_{scene_idx:03d}",
        "image": f"images/scene_{scene_idx:03d}.png",
        "split_group": group,
        "ood_axis": axis,
        "lighting_condition": lighting,
        "clutter_level": clutter,
        "difficulty_tier": difficulty,
        "layout_family": layout,
        "slot_map": slot_map,
        "slot_query_color": rng.choice(slot_colors),
        "gauge": rng.choice([18, 22, 26, 31, 35, 39, 44, 48, 53, 57, 62, 67, 71, 76, 82, 88]),
        "led_map": led_map,
        "led_query_color": rng.choice(list(led_map.values())),
        "cable_map": cable_map,
        "cable_query_color": rng.choice(list(cable_map.values())),
        "robot_target": rng.choice(MODULES),
        "defect_zone": rng.choice(ZONES),
        "lot_code": rng.randint(10, 98),
        "route": route,
        "route_query": route_query,
    }


def draw_scene(trace: dict, out_path: Path, np_rng: np.random.Generator) -> None:
    base = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    base[:] = np.array((30, 35, 42), dtype=np.uint8)
    rect(base, 20, 20, 700, 460, "panel")
    for x in range(40, 700, 40):
        line(base, x, 28, x, 452, "grid", 1)
    for y in range(40, 460, 40):
        line(base, 28, y, 692, y, "grid", 1)

    cassette_x = 55 if trace["layout_family"] != "right_cassette" else 470
    cassette_y = 70
    rect(base, cassette_x - 10, cassette_y - 20, cassette_x + 172, cassette_y + 318, "panel_dark")
    text(base, cassette_x + 18, cassette_y - 14, 2, "SLOTS", "gray")
    for slot in range(1, 7):
        y = cassette_y + (slot - 1) * 48
        rect(base, cassette_x, y, cassette_x + 150, y + 33, "slot")
        rect(base, cassette_x + 4, y + 4, cassette_x + 146, y + 29, "panel_dark")
        wafer_color = trace["slot_map"][slot]
        if wafer_color != "gray":
            circle(base, cassette_x + 75, y + 16, 15, wafer_color)
        text(base, cassette_x - 26, y + 7, 2, str(slot), "white")

    gauge_x, gauge_y = (300, 78) if trace["layout_family"] != "right_cassette" else (240, 78)
    rect(base, gauge_x - 18, gauge_y - 18, gauge_x + 140, gauge_y + 92, "black")
    text(base, gauge_x - 10, gauge_y - 12, 2, "PSI", "gray")
    seven_number(base, gauge_x + 18, gauge_y + 20, trace["gauge"], 6, "green")

    lot_x, lot_y = (320, 205) if trace["layout_family"] != "right_cassette" else (250, 205)
    rect(base, lot_x - 18, lot_y - 28, lot_x + 136, lot_y + 88, "panel_dark")
    text(base, lot_x - 10, lot_y - 20, 2, "LOT", "gray")
    seven_number(base, lot_x + 18, lot_y + 8, trace["lot_code"], 6, "cyan")

    module_x = 530 if trace["layout_family"] != "right_cassette" else 60
    module_y = 72
    module_centers = {}

    def draw_modules() -> None:
        for i, mod in enumerate(MODULES):
            y = module_y + i * 60
            rect(base, module_x, y, module_x + 130, y + 42, "panel_dark")
            text(base, module_x + 10, y + 11, 3, mod, "white")
            circle(base, module_x + 95, y + 21, 13, trace["led_map"][mod])
            module_centers[mod] = (module_x + 65, y + 21)

    draw_modules()

    ports_y = 392
    port_start = 255 if trace["layout_family"] != "compact_panel" else 300
    port_gap = 78 if trace["layout_family"] == "wide_ports" else 62
    port_positions = {}
    for i, port in enumerate(PORTS):
        x = port_start + i * port_gap
        port_positions[port] = (x, ports_y + 5)

    def draw_ports() -> None:
        for port, (x, _) in port_positions.items():
            rect(base, x - 15, ports_y - 10, x + 15, ports_y + 20, "panel_dark")
            circle(base, x, ports_y + 5, 8, "gray")
            text(base, x - 10, ports_y + 28, 2, port, "white")

    draw_ports()
    cable_origin_y = 456
    for i, (port, c) in enumerate(trace["cable_map"].items()):
        x0 = 260 + i * 76
        x1, y1 = port_positions[port]
        line(base, x0, cable_origin_y, x1, y1, c, 3 if trace["difficulty_tier"] == "easy" else 4)
    draw_ports()

    robot_base = (360, 300)
    target = module_centers[trace["robot_target"]]
    circle(base, robot_base[0], robot_base[1], 18, "gray")
    line(base, robot_base[0], robot_base[1], target[0], target[1], "orange", 4)
    circle(base, target[0], target[1], 9, "orange")
    draw_modules()
    circle(base, target[0], target[1], 7, "orange")

    zone_coords = {
        "upper-left": (cassette_x + 30, cassette_y + 36),
        "upper-right": (cassette_x + 120, cassette_y + 36),
        "center": (cassette_x + 78, cassette_y + 145),
        "lower-left": (cassette_x + 36, cassette_y + 256),
        "lower-right": (cassette_x + 122, cassette_y + 256),
    }
    dx, dy = zone_coords[trace["defect_zone"]]
    for j in range(7):
        circle(base, dx + int(np_rng.normal(0, 5)), dy + int(np_rng.normal(0, 5)), 3 + (j % 2), "magenta")

    route_x, route_y = 280 if trace["layout_family"] != "right_cassette" else 235, 320
    rect(base, route_x - 18, route_y - 22, route_x + 195, route_y + 44, "black")
    text(base, route_x - 10, route_y - 16, 2, "ROUTE", "gray")
    for i, wp in enumerate(trace["route"]):
        seven_number(base, route_x + 20 + i * 31, route_y + 6, int(wp), 3, "yellow", digits=1)
        if i < 4:
            line(base, route_x + 37 + i * 31, route_y + 20, route_x + 47 + i * 31, route_y + 20, "gray", 1)

    protected = [
        (cassette_x - 35, cassette_y - 28, cassette_x + 182, cassette_y + 326),
        (gauge_x - 26, gauge_y - 26, gauge_x + 152, gauge_y + 102),
        (lot_x - 26, lot_y - 36, lot_x + 148, lot_y + 100),
        (module_x - 16, module_y - 14, module_x + 146, module_y + 5 * 60 + 48),
        (route_x - 28, route_y - 30, route_x + 210, route_y + 54),
        (port_start - 32, ports_y - 28, port_start + port_gap * 4 + 34, ports_y + 60),
    ]
    clutter = trace["clutter_level"]
    n_clutter = {"low": 4, "medium": 16, "high": 32}[clutter]
    for _ in range(n_clutter):
        cx, cy = safe_point(np_rng, protected)
        c = str(np_rng.choice(["gray", "white", "cyan", "purple"]))
        circle(base, cx, cy, int(np_rng.integers(1, 3)), c)

    if trace["lighting_condition"] == "low_light":
        base = (base.astype(np.float32) * 0.48).clip(0, 255).astype(np.uint8)
    elif trace["lighting_condition"] == "blue_cast":
        base[:, :, 2] = np.clip(base[:, :, 2].astype(np.int16) + 35, 0, 255)
        base[:, :, 0] = (base[:, :, 0].astype(np.float32) * 0.82).astype(np.uint8)
    elif trace["lighting_condition"] == "glare":
        glare_count = 1 if trace["difficulty_tier"] == "medium" else 2
        glare_alpha = 0.28 if trace["difficulty_tier"] == "medium" else 0.36
        for _ in range(glare_count):
            gx0, gy0, gx1, gy1 = safe_rect(np_rng, protected, (85, 135), (24, 42), (160, 650), (45, 300))
            blend_rect(base, gx0, gy0, gx1, gy1, "white", glare_alpha)

    if trace["ood_axis"] == "occlusion":
        ox0, oy0, ox1, oy1 = safe_rect(np_rng, protected, (150, 240), (14, 22))
        blend_rect(base, ox0, oy0, ox1, oy1, "gray", 0.48)
        ox0, oy0, ox1, oy1 = safe_rect(np_rng, protected, (180, 290), (12, 20), (45, WIDTH - 45), (290, HEIGHT - 40))
        blend_rect(base, ox0, oy0, ox1, oy1, "black", 0.34)

    noise_sigma = {"easy": 3.0, "medium": 4.6, "hard": 5.8}[trace["difficulty_tier"]]
    noise = np_rng.normal(0, noise_sigma, base.shape)
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Final readability pass: hard cases keep clutter/lighting, but answer-bearing
    # displays and labels are re-stamped so questions remain objectively solvable.
    text(base, cassette_x + 18, cassette_y - 14, 2, "SLOTS", "gray")
    for slot in range(1, 7):
        y = cassette_y + (slot - 1) * 48
        text(base, cassette_x - 26, y + 7, 2, str(slot), "white")
        wafer_color = trace["slot_map"][slot]
        if wafer_color != "gray":
            circle(base, cassette_x + 75, y + 16, 13, wafer_color)

    rect(base, gauge_x - 18, gauge_y - 18, gauge_x + 140, gauge_y + 92, "black")
    text(base, gauge_x - 10, gauge_y - 12, 2, "PSI", "gray")
    seven_number(base, gauge_x + 18, gauge_y + 20, trace["gauge"], 6, "green")

    rect(base, lot_x - 18, lot_y - 28, lot_x + 136, lot_y + 88, "panel_dark")
    text(base, lot_x - 10, lot_y - 20, 2, "LOT", "gray")
    seven_number(base, lot_x + 18, lot_y + 8, trace["lot_code"], 6, "cyan")

    draw_modules()
    circle(base, target[0], target[1], 7, "orange")
    draw_ports()

    rect(base, route_x - 18, route_y - 22, route_x + 195, route_y + 44, "black")
    text(base, route_x - 10, route_y - 16, 2, "ROUTE", "gray")
    for i, wp in enumerate(trace["route"]):
        seven_number(base, route_x + 20 + i * 31, route_y + 6, int(wp), 3, "yellow", digits=1)
        if i < 4:
            line(base, route_x + 37 + i * 31, route_y + 20, route_x + 47 + i * 31, route_y + 20, "gray", 1)

    circle(base, dx, dy, 5, "magenta")
    write_png(out_path, base)


def add_row(rows: list[dict], trace: dict, qnum: int, query_type: str, question: str, answer_text: str, candidates: list[str], task_family: str, rng: random.Random) -> None:
    opts, answer_letter = make_options(answer_text, candidates, rng)
    rows.append(
        {
            "id": f"{trace['scene_id'].replace('scene_', 'cw_scene')}_q{qnum:02d}",
            "scene_id": trace["scene_id"],
            "image": trace["image"],
            "query_type": query_type,
            "question": question,
            "option_A": opts["A"],
            "option_B": opts["B"],
            "option_C": opts["C"],
            "option_D": opts["D"],
            "option_E": opts["E"],
            "answer": answer_letter,
            "answer_text": answer_text,
            "split_group": trace["split_group"],
            "ood_axis": trace["ood_axis"],
            "layout_family": trace["layout_family"],
            "lighting_condition": trace["lighting_condition"],
            "clutter_level": trace["clutter_level"],
            "difficulty_tier": trace["difficulty_tier"],
            "task_family": task_family,
        }
    )


def question_rows(trace: dict, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    q = 0

    slot_answer = next(f"Slot {slot}" for slot, c in trace["slot_map"].items() if c == trace["slot_query_color"])
    add_row(rows, trace, q, "slot_color", f"Which cassette slot contains the {trace['slot_query_color']} wafer?", slot_answer, [f"Slot {i}" for i in range(1, 6)], "wafer_tracking", rng)
    q += 1

    gauge_candidates = [str(v) for v in [18, 22, 26, 31, 35, 39, 44, 48, 53, 57, 62, 67, 71, 76, 82, 88]]
    add_row(rows, trace, q, "gauge_reading", "What two-digit pressure value is displayed on the PSI gauge?", str(trace["gauge"]), gauge_candidates, "instrument_reading", rng)
    q += 1

    led_answer = next(f"Module {m}" for m, c in trace["led_map"].items() if c == trace["led_query_color"])
    add_row(rows, trace, q, "led_state", f"Which module shows the {trace['led_query_color']} status LED?", led_answer, [f"Module {m}" for m in MODULES], "instrument_reading", rng)
    q += 1

    cable_answer = next(port for port, c in trace["cable_map"].items() if c == trace["cable_query_color"])
    add_row(rows, trace, q, "cable_connection", f"Which port is connected to the {trace['cable_query_color']} cable?", cable_answer, PORTS, "connection_audit", rng)
    q += 1

    add_row(rows, trace, q, "robot_alignment", "Which station is the orange robot arm pointing toward?", f"Station {trace['robot_target']}", [f"Station {m}" for m in MODULES], "wafer_tracking", rng)
    q += 1

    add_row(rows, trace, q, "defect_location", "Where is the magenta particle marker located on the cassette area?", trace["defect_zone"], ZONES, "defect_audit", rng)
    q += 1

    lot_candidates = [str(v) for v in rng.sample([n for n in range(10, 99) if n != trace["lot_code"]], 4)] + [str(trace["lot_code"])]
    add_row(rows, trace, q, "lot_code", "What two-digit lot code is shown on the pod label?", str(trace["lot_code"]), lot_candidates, "instrument_reading", rng)
    q += 1

    route = trace["route"]
    route_answer = route[route.index(trace["route_query"]) + 1]
    add_row(rows, trace, q, "route_order", f"In the route strip, which waypoint comes immediately after waypoint {trace['route_query']}?", route_answer, ["1", "2", "3", "4", "5"], "route_memory", rng)
    return rows


def main() -> None:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    root = Path(__file__).resolve().parent
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for scene_idx in range(N_SCENES):
        trace = scene_trace(scene_idx, rng)
        draw_scene(trace, root / trace["image"], np_rng)
        rows.extend(question_rows(trace, rng))

    df = pd.DataFrame(rows)
    df.to_csv(root / "questions.csv", index=False)
    print(f"Wrote {len(df)} rows and {N_SCENES} images to {root}")


if __name__ == "__main__":
    main()
