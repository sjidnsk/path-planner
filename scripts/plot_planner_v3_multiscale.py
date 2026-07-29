"""Render offline figures for the planner-v3 multiscale latency experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


SCALES = ("TEN_METER", "HUNDRED_METER", "THOUSAND_METER")
SCENES = ("G1_LOW_KNOWN", "G1_MEDIUM_KNOWN", "G1_HIGH_FRONTIER")
PLATFORM_STYLES = {
    "WHEELED": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "LEGGED": {"color": "#D55E00", "linestyle": "--", "marker": "^"},
    "HOPPER": {"color": "#009E73", "linestyle": ":", "marker": "s"},
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"responses.jsonl line {line_number} must be an object")
                records.append(value)
    return records


def _polygon_points(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for point in value:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return points


def _region_polygons(regions: Any, key: str) -> list[list[tuple[float, float]]]:
    if not isinstance(regions, list):
        return []
    aliases = {
        "known": {"known", "known_terrain"},
        "unknown": {"unknown", "unknown_region"},
        "obstacles": {"obstacle", "obstacles", "hard_obstacle"},
        "proxy_obstacles": {"synthetic_terrain_obstacle_proxy"},
    }[key]
    return [
        points
        for region in regions
        if isinstance(region, dict)
        and str(region.get("kind", "")).lower() in aliases
        and (key != "proxy_obstacles" or _is_synthetic_proxy(region))
        and (points := _polygon_points(region.get("polygon_xy_m", [])))
    ]


def _is_synthetic_proxy(region: dict[str, Any]) -> bool:
    provenance = region.get("provenance", region.get("source_kind"))
    if isinstance(provenance, dict):
        provenance = provenance.get("source_kind", provenance.get("micro_source_kind"))
    return provenance == "synthetic_terrain_obstacle_proxy/v1"


def _merged_region_polygons(records: list[dict[str, Any]], key: str) -> list[list[tuple[float, float]]]:
    unique: dict[tuple[tuple[float, float], ...], list[tuple[float, float]]] = {}
    for record in records:
        for points in _region_polygons(record.get("regions", []), key):
            unique.setdefault(tuple(points), points)
    return list(unique.values())


def _map_bounds(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, dict):
        return (0.0, 1.0, 0.0, 1.0)
    minimum = _polygon_points([value.get("minimum_xy_m", [])])
    maximum = _polygon_points([value.get("maximum_xy_m", [])])
    if not minimum or not maximum:
        return (0.0, 1.0, 0.0, 1.0)
    return (minimum[0][0], maximum[0][0], minimum[0][1], maximum[0][1])


def _first_record(records: list[dict[str, Any]], scale: str, scene: str) -> dict[str, Any] | None:
    return next((record for record in records if str(record.get("scale")) == scale and str(record.get("scene")) == scene), None)


def _draw_scenario_panel(ax: Any, records: list[dict[str, Any]], scale: str, scene: str) -> None:
    panel_records = [record for record in records if str(record.get("scale")) == scale and str(record.get("scene")) == scene]
    source = panel_records[0] if panel_records else _first_record(records, scale, scene)
    if source is None:
        ax.text(0.5, 0.5, "No representative response", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_axis_off()
        return

    xmin, xmax, ymin, ymax = _map_bounds(source.get("map_bounds"))
    ax.set_facecolor("#f2ead6")
    region_records = panel_records or [source]
    for points in _merged_region_polygons(region_records, "known"):
        ax.add_patch(Polygon(points, closed=True, facecolor="#d9c7a2", edgecolor="#8c795a", alpha=0.65, label="Known terrain"))
    for points in _merged_region_polygons(region_records, "unknown"):
        ax.add_patch(Polygon(points, closed=True, facecolor="#e6e6e6", edgecolor="#888888", hatch="//", alpha=0.9, label="Unknown region"))
    for points in _merged_region_polygons(region_records, "obstacles"):
        ax.add_patch(Polygon(points, closed=True, facecolor="#6b6259", edgecolor="#222222", label="Hard obstacle"))
    for points in _merged_region_polygons(region_records, "proxy_obstacles"):
        ax.add_patch(Polygon(points, closed=True, facecolor="#4d4d4d", edgecolor="#222222", label="Proxy obstacle"))

    reference_records = panel_records or [source]
    seen_platforms: set[str] = set()
    for record in reference_records:
        platform = str(record.get("platform", "Reference"))
        style = PLATFORM_STYLES.get(platform, {"color": "#555555", "linestyle": "-", "marker": None})
        points = _polygon_points(record.get("reference_polyline_xy_m", []))
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, linewidth=1.8, markersize=3, label=platform, **style)
            seen_platforms.add(platform)
    for platform, style in PLATFORM_STYLES.items():
        if platform not in seen_platforms:
            ax.plot([], [], linewidth=1.8, markersize=3, label=platform, **style)

    start = _polygon_points([source.get("start_xy_m", [])])
    goal = _polygon_points([source.get("goal_xy_m", [])])
    if start:
        ax.scatter(*start[0], color="#111111", marker="o", s=28, zorder=5, label="Start")
    if goal:
        label = "Nominal goal" if scene == "G1_HIGH_FRONTIER" else "Goal"
        ax.scatter(*goal[0], color="#cc3311", marker="*", s=56, zorder=5, label=label)
    if scene == "G1_HIGH_FRONTIER":
        for record in reference_records:
            if str(record.get("platform")) not in {"WHEELED", "LEGGED"}:
                continue
            points = _polygon_points(record.get("reference_polyline_xy_m", []))
            if points:
                ax.scatter(*points[-1], color="#6a3d9a", marker="D", s=28, zorder=6, label="Safe frontier")

    hopper = next((record for record in reference_records if str(record.get("platform")) == "HOPPER"), None)
    if hopper is None and str(source.get("platform")) == "HOPPER":
        hopper = source
    landing = _polygon_points(hopper.get("landing_polygon_xy_m", [])) if hopper else []
    if landing:
        ax.add_patch(Polygon(landing, closed=True, fill=False, edgecolor="#009E73", linewidth=1.2, linestyle="--", label="Hopper landing region"))

    arc = _polygon_points(hopper.get("ballistic_arc_xz_m", [])) if hopper else []
    if arc:
        inset = ax.inset_axes([0.62, 0.62, 0.32, 0.27])
        arc_x, arc_z = zip(*arc)
        inset.plot(arc_x, arc_z, color="#009E73", linewidth=1.4)
        inset.fill_between(arc_x, arc_z, 0, color="#009E73", alpha=0.15)
        inset.set_title("Hopper arc", fontsize=6)
        inset.tick_params(labelsize=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    scene_label = scene.replace("G1_", "G1 ").replace("_", " ").title()
    ax.set_title(f"{scene_label} · {scale.replace('_METER', '').replace('_', ' ').title()}", fontsize=10)
    ax.tick_params(labelsize=7)


def plot_scenario_overview(records: list[dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for row, scene in enumerate(SCENES):
        for column, scale in enumerate(SCALES):
            _draw_scenario_panel(axes[row, column], records, scale, scene)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=5, fontsize=8, frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = summary.get("scale_platform_results", [])
    if isinstance(raw_rows, dict):
        raw_rows = list(raw_rows.values())
    if not isinstance(raw_rows, list):
        raise ValueError("summary.json scale_platform_results must be a list or object")
    return [row for row in raw_rows if isinstance(row, dict)]


def plot_latency_results(summary: dict[str, Any], output_path: Path) -> None:
    rows = _metric_rows(summary)
    figure, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scale", "unknown"))].append(row)
    scales = [scale for scale in SCALES if scale in grouped] + sorted(set(grouped) - set(SCALES))
    positions: list[float] = []
    labels: list[str] = []
    for scale_index, scale in enumerate(scales):
        for platform_index, row in enumerate(sorted(grouped[scale], key=lambda item: str(item.get("platform", "")))):
            position = scale_index * 4 + platform_index
            platform = str(row.get("platform", "Unknown"))
            style = PLATFORM_STYLES.get(platform, {"color": "#555555"})
            p50 = float(row["p50_ns"]) / 1_000_000
            p95 = float(row["p95_ns"]) / 1_000_000
            maximum = float(row["maximum_ns"]) / 1_000_000
            ax.vlines(position, p50, maximum, color=style["color"], linewidth=1.5)
            ax.scatter(position, p50, color=style["color"], marker="o", s=38, zorder=3)
            ax.scatter(position, p95, color=style["color"], marker="s", s=38, zorder=3)
            success = 100 * float(row.get("success_rate", 0.0))
            ax.annotate(f"{success:.0f}%", (position, maximum), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
            positions.append(position)
            labels.append(f"{scale}\n{platform}")
    ax.axhline(1000, color="#aa3377", linestyle="--", linewidth=1.2, label="1000 ms target")
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_title("Multiscale planner latency: P50 ○, P95 □, maximum whisker")
    ax.grid(axis="y", which="both", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def render(input_root: Path, output_dir: Path) -> None:
    summary = _read_json(input_root / "summary.json")
    if "measured_call_count" not in summary:
        raise ValueError("summary.json is missing required measured_call_count")
    for field in ("scenario_results", "scale_platform_results", "platform_results"):
        if field not in summary:
            raise ValueError(f"summary.json is missing required {field}")
    records = _read_jsonl(input_root / "responses.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scenario_overview(records, output_dir / "scenario-overview.png")
    plot_latency_results(summary, output_dir / "latency-results.png")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        render(args.input_root, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        print(f"plot planner v3 multiscale: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
