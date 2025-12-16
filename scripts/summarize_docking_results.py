#!/usr/bin/env python3
"""Summarise docking CSV outputs in result/docking_results and produce plots."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_DIR = Path("result/docking_results")


def load_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def parse_float(value: str) -> float:
    if value is None:
        return math.nan
    value = value.strip()
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_bool(value: str) -> Optional[bool]:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def ligand_group(ligand_name: str) -> str:
    if "_baseline" in ligand_name:
        return "baseline"
    if "_focus" in ligand_name:
        return "focus"
    return "unknown"


def ligand_core(ligand_name: str) -> str:
    for suffix in ("_baseline_out", "_focus_out", "_baseline", "_focus"):
        if suffix in ligand_name:
            return ligand_name.replace(suffix, "")
    return ligand_name


@dataclass
class AggregateMetrics:
    ligand: str
    subset: str
    entries: int = 0
    geom_pass: int = 0
    best_affinity: float = math.inf
    mean_affinity: float = math.nan
    min_distance: float = math.nan
    mean_distance: float = math.nan
    notes: List[str] = None


def aggregate_normalized(rows: Iterable[dict]) -> Dict[str, AggregateMetrics]:
    by_ligand: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_ligand[row["ligand"]].append(row)

    aggregates: Dict[str, AggregateMetrics] = {}
    for ligand, items in by_ligand.items():
        subset = ligand_group(ligand)
        affinities = [parse_float(item["affinity"]) for item in items]
        distances = [parse_float(item["d_sg_cyl"]) for item in items]
        passes = [int(parse_float(item.get("meets_geometry", "0")) == 1) for item in items]
        finite_affinities = [val for val in affinities if not math.isnan(val)]
        finite_distances = [val for val in distances if not math.isnan(val)]

        agg = AggregateMetrics(
            ligand=ligand,
            subset=subset,
            entries=len(items),
            geom_pass=sum(passes),
            best_affinity=min(finite_affinities) if finite_affinities else math.nan,
            mean_affinity=mean(finite_affinities) if finite_affinities else math.nan,
            min_distance=min(finite_distances) if finite_distances else math.nan,
            mean_distance=mean(finite_distances) if finite_distances else math.nan,
            notes=[],
        )
        aggregates[ligand] = agg
    return aggregates


def summarise_selected(rows: Iterable[dict]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for row in rows:
        core = ligand_core(row["ligand"])
        summary[core] = {
            "ligand": row["ligand"],
            "file": row["file"],
            "affinity": parse_float(row["affinity"]),
            "d_sg_cyl": parse_float(row["d_sg_cyl"]),
            "angle_bd": parse_float(row["angle_bd"]),
            "reason": row.get("selected_reason", ""),
        }
    return summary


def write_report(
    aggregates: Dict[str, AggregateMetrics],
    selected: Dict[str, dict],
    normalized_rows: List[dict],
    candidates_rows: List[dict],
    output_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("Docking results summary")
    lines.append("=" * 30)
    lines.append("")
    lines.append(f"Total normalized poses: {len(normalized_rows)}")
    lines.append(f"Total filtered candidates: {len(candidates_rows)}")
    lines.append(f"Total selected poses: {len(selected)}")
    lines.append("")

    cores = sorted({ligand_core(name) for name in aggregates})
    for core in cores:
        baseline_key = f"{core}_baseline_out"
        focus_key = f"{core}_focus_out"
        lines.append(f"{core}:")
        baseline = aggregates.get(baseline_key)
        focus = aggregates.get(focus_key)
        for label, record in (("baseline", baseline), ("focus", focus)):
            if record is None:
                continue
            lines.append(
                f"  {label:8s} entries={record.entries:3d} geom_pass={record.geom_pass:2d} "
                f"best_aff={record.best_affinity:5.2f} min_d={record.min_distance:4.2f}"
            )
        selection = selected.get(core)
        if selection:
            lines.append(
                f"  selected -> {selection['ligand']} (aff={selection['affinity']:.2f}, "
                f"d_sg_cyl={selection['d_sg_cyl'] or float('nan'):.2f}, "
                f"angle={selection['angle_bd'] or float('nan'):.1f}); reason={selection['reason']}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_affinity_vs_distance(rows: Iterable[dict], output_path: Path) -> None:
    xs: List[float] = []
    ys: List[float] = []
    colors: List[str] = []
    for row in rows:
        dist_val = parse_float(row["d_sg_cyl"])
        affinity = parse_float(row["affinity"])
        if math.isnan(dist_val) or math.isnan(affinity):
            continue
        xs.append(dist_val)
        ys.append(affinity)
        colors.append("#1f77b4" if "_baseline" in row["ligand"] else "#d62728")
    if not xs:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, c=colors, alpha=0.7, edgecolors="k", linewidths=0.2)
    plt.xlabel("Cys-SG to electrophilic carbon distance (Å)")
    plt.ylabel("Affinity (kcal/mol)")
    plt.title("Affinity vs. SG distance")
    plt.axvline(3.6, color="gray", linestyle="--", linewidth=1, label="3.6 Å threshold")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_selected_affinities(selected: Dict[str, dict], output_path: Path) -> None:
    if not selected:
        return
    items = sorted(selected.values(), key=lambda row: row["affinity"])
    labels = [item["ligand"] for item in items]
    affinities = [item["affinity"] for item in items]
    plt.figure(figsize=(8, 4 + len(labels) * 0.2))
    bars = plt.barh(labels, affinities, color="#4a90e2")
    for bar, affinity in zip(bars, affinities, strict=False):
        plt.text(
            affinity - 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{affinity:.2f}",
            va="center",
            ha="right",
            color="white",
            fontsize=9,
        )
    plt.xlabel("Affinity (kcal/mol)")
    plt.title("Selected pose affinities")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    normalized_path = RESULT_DIR / "normalized_all.csv"
    candidates_path = RESULT_DIR / "candidates_after_filters.csv"
    selected_path = RESULT_DIR / "selected_poses.csv"

    if not normalized_path.exists() or not candidates_path.exists() or not selected_path.exists():
        missing = [
            path.name
            for path in (normalized_path, candidates_path, selected_path)
            if not path.exists()
        ]
        raise FileNotFoundError(f"Missing expected CSV files: {', '.join(missing)}")

    normalized_rows = load_csv(normalized_path)
    candidates_rows = load_csv(candidates_path)
    selected_rows = load_csv(selected_path)

    aggregates = aggregate_normalized(normalized_rows)
    selected = summarise_selected(selected_rows)

    report_path = RESULT_DIR / "docking_analysis_summary.txt"
    write_report(aggregates, selected, normalized_rows, candidates_rows, report_path)

    plot_affinity_vs_distance(normalized_rows, RESULT_DIR / "plot_affinity_vs_sg_distance.png")
    plot_selected_affinities(selected, RESULT_DIR / "plot_selected_affinities.png")

    print(f"Wrote summary to {report_path}")
    print("Generated plots: plot_affinity_vs_sg_distance.png, plot_selected_affinities.png")


if __name__ == "__main__":
    main()
