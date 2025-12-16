#!/usr/bin/env python3
"""Recompute Cys209–SG distances for poses listed in selected_poses.csv and plot."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AtomRecord = Tuple[int, str, str, str, str, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute minimum SG–C distances for selected docking poses."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("result/docking_results"),
        help="Directory containing selected_poses.csv and ligand PDBQT files.",
    )
    parser.add_argument(
        "--receptor",
        type=Path,
        default=Path("result/dock_out_p2r/receptor.pdbqt"),
        help="Receptor PDBQT file containing the catalytic cysteine.",
    )
    parser.add_argument(
        "--cys-resi",
        default="209",
        help="Residue index for the catalytic cysteine (default: 209).",
    )
    parser.add_argument(
        "--cys-resn",
        default="CYS",
        help="Residue name for the catalytic cysteine (default: CYS).",
    )
    parser.add_argument(
        "--cys-atom",
        default="SG",
        help="Atom name for the catalytic sulfur (default: SG).",
    )
    parser.add_argument(
        "--cys-chain",
        default="",
        help="Optional chain identifier for the catalytic cysteine.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional output CSV path; defaults to <results-dir>/selected_pose_distances.csv",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        help="Optional output plot path; defaults to <results-dir>/selected_pose_distances.png",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "focus", "baseline"],
        default="all",
        help="Limit outputs to poses containing '_focus' or '_baseline' (default: all).",
    )
    return parser.parse_args()


def parse_float(value: str) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_pdbqt_models(path: Path) -> List[List[AtomRecord]]:
    models: List[List[AtomRecord]] = []
    current: List[AtomRecord] = []
    in_model = False
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            rec = line[:6].strip()
            if rec == "MODEL":
                if current:
                    models.append(current)
                current = []
                in_model = True
                continue
            if line.startswith("ENDMDL"):
                models.append(current)
                current = []
                in_model = False
                continue
            if rec in {"ATOM", "HETATM"}:
                try:
                    serial = int(line[6:11])
                except ValueError:
                    serial = -1
                name = line[12:16].strip()
                resn = line[17:20].strip()
                chain = line[21:22].strip()
                resi = line[22:26].strip()
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    parts = line.split()
                    try:
                        x, y, z = map(float, parts[-3:])
                    except Exception:
                        x = y = z = float("nan")
                current.append((serial, name, resn, resi, chain, x, y, z))
    if current:
        models.append(current)
    if not models:
        raise ValueError(f"No atoms parsed from {path}")
    return models


def find_receptor_atom(
    atoms: Sequence[AtomRecord],
    resn: str,
    resi: str,
    atom_name: str,
    chain: str = "",
) -> Tuple[float, float, float]:
    resn = resn.strip().upper()
    resi = resi.strip()
    atom_name = atom_name.strip().upper()
    chain = chain.strip()
    for record in atoms:
        _serial, name, rresn, rresi, rchain, x, y, z = record
        if name.strip().upper() != atom_name:
            continue
        if rresn.strip().upper() != resn:
            continue
        if rresi.strip() != resi:
            continue
        if chain and rchain.strip() != chain:
            continue
        return (x, y, z)
    raise FileNotFoundError(
        f"Receptor atom {resn} {resi} {atom_name} not found (chain='{chain}')"
    )


def is_hydrogen(atom_name: str, resn: str) -> bool:
    stripped = atom_name.strip()
    return stripped.upper().startswith("H")


def min_distance_to_point(atoms: Sequence[AtomRecord], point: Sequence[float]) -> float:
    min_dist = math.inf
    for atom in atoms:
        if is_hydrogen(atom[1], atom[2]):
            continue
        _, _, _, _, _, x, y, z = atom
        dist_val = math.dist((x, y, z), point)
        if dist_val < min_dist:
            min_dist = dist_val
    return min_dist


def read_selected_rows(selected_csv: Path) -> List[dict]:
    with selected_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def plot_distances(rows: List[dict], output_path: Path) -> None:
    valid_rows = [row for row in rows if not math.isnan(row["computed_distance"])]
    if not valid_rows:
        return
    valid_rows.sort(key=lambda row: row["computed_distance"])
    labels = [row["ligand"] for row in valid_rows]
    distances = [row["computed_distance"] for row in valid_rows]
    colors = [
        "#1f77b4" if "_baseline" in label else "#d62728" for label in labels
    ]
    plt.figure(figsize=(8, 4 + len(labels) * 0.25))
    bars = plt.barh(labels, distances, color=colors)
    for bar, distance in zip(bars, distances, strict=False):
        plt.text(
            distance + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{distance:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    plt.axvline(3.6, color="gray", linestyle="--", linewidth=1, label="3.6 Å threshold")
    plt.xlabel("Min heavy-atom distance to Cys SG (Å)")
    plt.title("Cys209–SG proximity for selected poses")
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_affinity_vs_distance(rows: List[dict], output_path: Path) -> None:
    valid_rows = [
        row
        for row in rows
        if not math.isnan(row["computed_distance"]) and not math.isnan(row["affinity"])
    ]
    if not valid_rows:
        return
    xs = [row["computed_distance"] for row in valid_rows]
    ys = [row["affinity"] for row in valid_rows]
    colors = [
        "#1f77b4" if "_baseline" in row["ligand"] else "#d62728"
        for row in valid_rows
    ]
    labels = [row["ligand"] for row in valid_rows]
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, c=colors, edgecolors="k", linewidths=0.2, s=60)
    for row, x, y in zip(valid_rows, xs, ys, strict=False):
        label = row["ligand"]
        affinity = row["affinity"]
        plt.annotate(
            f"{label}\n{affinity:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=8,
        )
    plt.axvline(3.6, color="gray", linestyle="--", linewidth=1, label="3.6 Å threshold")
    plt.xlabel("Cys-SG to electrophilic carbon distance (Å)")
    plt.ylabel("Affinity (kcal/mol)")
    plt.title("Affinity vs. SG distance for selected poses")
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_affinities(rows: List[dict], output_path: Path) -> None:
    valid_rows = [row for row in rows if not math.isnan(row["affinity"])]
    if not valid_rows:
        return
    valid_rows.sort(key=lambda row: row["affinity"])
    labels = [row["ligand"] for row in valid_rows]
    affinities = [row["affinity"] for row in valid_rows]
    colors = [
        "#1f77b4" if "_baseline" in label else "#d62728"
        for label in labels
    ]
    plt.figure(figsize=(8, 4 + len(labels) * 0.25))
    bars = plt.barh(labels, affinities, color=colors)
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
    plt.title("Affinity of selected poses")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    receptor_path = args.receptor.expanduser().resolve()
    selected_csv = results_dir / "selected_poses.csv"

    if not selected_csv.exists():
        raise FileNotFoundError(f"selected_poses.csv not found in {results_dir}")
    if not receptor_path.exists():
        raise FileNotFoundError(f"Receptor file not found: {receptor_path}")

    receptor_models = parse_pdbqt_models(receptor_path)
    receptor_atoms = receptor_models[0]
    sg_coord = find_receptor_atom(
        receptor_atoms,
        args.cys_resn,
        args.cys_resi,
        args.cys_atom,
        args.cys_chain,
    )

    rows = read_selected_rows(selected_csv)
    all_results: List[dict] = []
    for row in rows:
        ligand_file = results_dir / row["file"]
        ligand_name = row["ligand"]
        if not ligand_file.exists():
            computed_distance = math.nan
            reason = f"missing file: {ligand_file.name}"
        else:
            try:
                models = parse_pdbqt_models(ligand_file)
                # default to first model (best affinity)
                first_model_atoms = models[0]
                computed_distance = min_distance_to_point(first_model_atoms, sg_coord)
                reason = ""
            except Exception as exc:
                computed_distance = math.nan
                reason = f"error: {exc}"
        all_results.append(
            {
                "ligand": ligand_name,
                "file": row["file"],
                "model": row.get("model"),
                "affinity": parse_float(row.get("affinity")),
                "existing_d_sg_cyl": parse_float(row.get("d_sg_cyl")),
                "computed_distance": computed_distance,
                "notes": reason,
            }
        )

    subset = args.subset
    if subset == "focus":
        results = [row for row in all_results if "_focus" in row["ligand"]]
    elif subset == "baseline":
        results = [row for row in all_results if "_baseline" in row["ligand"]]
    else:
        results = all_results

    if not results:
        raise SystemExit(f"No poses match subset '{subset}'.")

    suffix = "" if subset == "all" else f"_{subset}"

    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else results_dir / f"selected_pose_distances{suffix}.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "ligand",
            "file",
            "model",
            "affinity",
            "existing_d_sg_cyl",
            "computed_distance",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    output_plot = (
        args.output_plot.expanduser().resolve()
        if args.output_plot
        else results_dir / f"selected_pose_distances{suffix}.png"
    )
    plot_distances(results, output_plot)

    scatter_path = output_plot.with_name(output_plot.stem + "_affinity.png")
    plot_affinity_vs_distance(results, scatter_path)

    affinity_plot = output_plot.with_name(output_plot.stem + "_affinity_only.png")
    plot_affinities(results, affinity_plot)

    print(f"Wrote distances to {output_csv}")
    print(f"Generated plot at {output_plot}")
    print(f"Generated affinity-distance plot at {scatter_path}")
    print(f"Generated affinity-only plot at {affinity_plot}")


if __name__ == "__main__":
    main()
