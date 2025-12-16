#!/usr/bin/env python3
"""Parse AutoDock Vina logs and summarize outcomes with optional plots."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError as exc:  # pragma: no cover - defensive guard for missing deps
    raise SystemExit(
        "matplotlib is required to run this script. Install it and retry."
    ) from exc


DEFAULT_RESULT_DIR = Path("result/dock_out_p2r")
DEFAULT_SKIP_LIGAND_NAMES = {"IDTT", "LDTT"}
DEFAULT_EXTRA_PDBQT_LIGANDS = {"LDTT_focus": "LDTT_focus_out.pdbqt"}

LOG_PATTERN = re.compile(
    r"^\s*(?P<mode>\d+)\s+(?P<affinity>-?\d+(?:\.\d+)?)\s+"
    r"(?P<rmsd_lb>\d+(?:\.\d+)?)\s+(?P<rmsd_ub>\d+(?:\.\d+)?)"
)

FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class DockingPose:
    ligand: str  # display name
    mode: int
    affinity: float
    rmsd_lb: float
    rmsd_ub: float
    file_prefix: str  # original prefix used for locating pose files


AtomRecord = Tuple[int, str, str, str, str, float, float, float]


@dataclass(frozen=True)
class PoseModel:
    model_id: int
    affinity: Optional[float]
    atoms: List[AtomRecord]


def parse_log_file(log_path: Path) -> List[DockingPose]:
    """Extract docking poses from a Vina log file."""
    file_prefix = log_path.stem.replace("_log", "")
    ligand = file_prefix.removesuffix("_pH45_min_p2r")
    poses: List[DockingPose] = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            match = LOG_PATTERN.match(line)
            if not match:
                continue
            poses.append(
                DockingPose(
                    ligand=ligand,
                    mode=int(match.group("mode")),
                    affinity=float(match.group("affinity")),
                    rmsd_lb=float(match.group("rmsd_lb")),
                    rmsd_ub=float(match.group("rmsd_ub")),
                    file_prefix=file_prefix,
                )
            )
    if not poses:
        raise ValueError(f"No docking entries captured from {log_path}")
    poses.sort(key=lambda pose: pose.affinity)
    return poses


def summarise_ligand(poses: Iterable[DockingPose]) -> dict[str, float | int | str]:
    """Derive summary statistics from the docking poses."""
    pose_list = list(poses)
    affinities = [pose.affinity for pose in pose_list]
    top_best = pose_list[0]
    top_affinities = affinities[: min(3, len(affinities))]
    top5_affinities = affinities[: min(5, len(affinities))]
    span = (
        top5_affinities[-1] - top5_affinities[0] if len(top5_affinities) > 1 else 0.0
    )
    return {
        "ligand": top_best.ligand,
        "file_prefix": top_best.file_prefix,
        "poses": len(pose_list),
        "best_mode": top_best.mode,
        "best_affinity": top_best.affinity,
        "best_rmsd_lb": top_best.rmsd_lb,
        "best_rmsd_ub": top_best.rmsd_ub,
        "mean_top3": mean(top_affinities),
        "mean_top5": mean(top5_affinities),
        "top5_span": span,
    }


def summarise_pdbqt(
    ligand_name: str, pdbqt_path: Path
) -> dict[str, float | int | str]:
    """Summarize docking statistics from a PDBQT file when no log is available."""
    affinities: list[float] = []
    with pdbqt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if "minimizedAffinity" in line:
                try:
                    affinities.append(float(line.rsplit(maxsplit=1)[-1]))
                except ValueError:
                    continue
    if not affinities:
        raise ValueError(f"No affinity data found in {pdbqt_path}")
    ordered_affinities = sorted(affinities)
    top_best_affinity = ordered_affinities[0]
    best_mode_index = affinities.index(top_best_affinity) + 1
    top3_affinities = ordered_affinities[: min(3, len(ordered_affinities))]
    top5_affinities = ordered_affinities[: min(5, len(ordered_affinities))]
    span = (
        top5_affinities[-1] - top5_affinities[0] if len(top5_affinities) > 1 else 0.0
    )
    file_prefix = pdbqt_path.stem
    if file_prefix.endswith("_out"):
        file_prefix = file_prefix[: -len("_out")]
    return {
        "ligand": ligand_name,
        "file_prefix": file_prefix,
        "poses": len(affinities),
        "best_mode": best_mode_index,
        "best_affinity": top_best_affinity,
        "best_rmsd_lb": math.nan,
        "best_rmsd_ub": math.nan,
        "mean_top3": mean(top3_affinities),
        "mean_top5": mean(top5_affinities),
        "top5_span": span,
    }


def _parse_residue_index(resi: str) -> int:
    """Convert a residue identifier into a sortable integer."""
    digits = re.findall(r"-?\d+", resi)
    if not digits:
        return 10**9
    try:
        return int(digits[0])
    except ValueError:
        return 10**9


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Angle ABC in degrees."""
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    baz = a[2] - b[2]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    bcz = c[2] - b[2]
    ba2 = bax * bax + bay * bay + baz * baz
    bc2 = bcx * bcx + bcy * bcy + bcz * bcz
    if ba2 == 0 or bc2 == 0:
        return float("nan")
    dot = bax * bcx + bay * bcy + baz * bcz
    cosv = max(-1.0, min(1.0, dot / math.sqrt(ba2 * bc2)))
    return math.degrees(math.acos(cosv))


def _parse_affinity_from_remark(line: str) -> Optional[float]:
    """Extract an affinity value from a REMARK line."""
    upper = line.upper()
    if "VINA RESULT" not in upper and "MINIMIZEDAFFINITY" not in upper:
        return None
    matches = FLOAT_RE.findall(line)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def parse_pdbqt_models(pdbqt_path: Path) -> List[PoseModel]:
    """Parse all models from a PDBQT pose file."""
    models: List[PoseModel] = []
    current_atoms: list[AtomRecord] = []
    current_affinity: Optional[float] = None
    current_model_id: Optional[int] = None
    explicit_models = False
    next_model_id = 1

    def flush() -> None:
        nonlocal current_atoms, current_affinity, current_model_id, next_model_id, explicit_models
        if not current_atoms:
            return
        model_id = current_model_id if current_model_id is not None else next_model_id
        models.append(
            PoseModel(
                model_id=model_id,
                affinity=current_affinity,
                atoms=current_atoms,
            )
        )
        current_atoms = []
        current_affinity = None
        if not explicit_models:
            next_model_id += 1
        current_model_id = None

    with pdbqt_path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            record = line[:6].strip().upper()
            if record == "MODEL":
                flush()
                explicit_models = True
                parts = line.split()
                if len(parts) > 1:
                    try:
                        current_model_id = int(parts[1])
                    except ValueError:
                        current_model_id = None
                else:
                    current_model_id = None
                continue
            if line.startswith("ENDMDL"):
                flush()
                explicit_models = False
                continue
            if record in {"ATOM", "HETATM"}:
                try:
                    serial = int(line[6:11])
                except ValueError:
                    try:
                        serial = int(line.split()[1])
                    except Exception:
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
                current_atoms.append((serial, name, resn, resi, chain, x, y, z))
                continue
            if line.startswith("REMARK"):
                affinity = _parse_affinity_from_remark(line)
                if affinity is not None:
                    current_affinity = affinity
                continue
        flush()
    return models


def parse_pdb_models(pdb_path: Path) -> List[PoseModel]:
    """Parse all models from a PDB pose file."""
    models: List[PoseModel] = []
    current_atoms: list[AtomRecord] = []
    current_model_id: Optional[int] = None
    explicit_models = False
    next_model_id = 1

    def flush() -> None:
        nonlocal current_atoms, current_model_id, next_model_id, explicit_models
        if not current_atoms:
            return
        model_id = current_model_id if current_model_id is not None else next_model_id
        models.append(
            PoseModel(
                model_id=model_id,
                affinity=None,
                atoms=current_atoms,
            )
        )
        current_atoms = []
        if not explicit_models:
            next_model_id += 1
        current_model_id = None

    with pdb_path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            record = line[:6].strip().upper()
            if record == "MODEL":
                flush()
                explicit_models = True
                parts = line.split()
                if len(parts) > 1:
                    try:
                        current_model_id = int(parts[1])
                    except ValueError:
                        current_model_id = None
                else:
                    current_model_id = None
                continue
            if line.startswith("ENDMDL"):
                flush()
                explicit_models = False
                continue
            if record in {"ATOM", "HETATM"}:
                try:
                    serial = int(line[6:11])
                except ValueError:
                    try:
                        serial = int(line.split()[1])
                    except Exception:
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
                current_atoms.append((serial, name, resn, resi, chain, x, y, z))
        flush()
    return models


def load_pose_models(pose_path: Path) -> List[PoseModel]:
    """Dispatch parsing depending on pose file extension."""
    suffix = pose_path.suffix.lower()
    if suffix == ".pdbqt":
        return parse_pdbqt_models(pose_path)
    if suffix == ".pdb":
        return parse_pdb_models(pose_path)
    raise ValueError(f"Unsupported pose format: {pose_path}")


def find_ligand_atom(
    atoms: Sequence[AtomRecord], candidate_names: Sequence[str]
) -> Optional[AtomRecord]:
    """Return the first ligand atom matching any of the candidate names."""
    if not candidate_names:
        return None
    candidates = {name.strip().upper() for name in candidate_names if name.strip()}
    if not candidates:
        return None
    best: Optional[AtomRecord] = None
    best_resi = 10**9
    for atom in atoms:
        name = atom[1].strip().upper()
        if name not in candidates:
            continue
        resi_value = _parse_residue_index(atom[3])
        if best is None or resi_value < best_resi:
            best = atom
            best_resi = resi_value
    return best


def evaluate_productive_fraction(
    pose_path: Path,
    sg_coord: Optional[Tuple[float, float, float]],
    n_candidates: Sequence[str],
    electrophile_candidates: Sequence[str],
    ns_max: Optional[float],
    angle_min: Optional[float],
    angle_max: Optional[float],
    anchor_coord: Optional[Tuple[float, float, float]],
    ligand_anchor_candidates: Optional[Sequence[str]],
    anchor_max: Optional[float],
) -> dict[str, float | int]:
    """Compute the fraction of models that meet productive geometry criteria."""
    if sg_coord is None:
        return {
            "geometry_total_poses": 0,
            "geometry_productive_poses": 0,
            "frac_prod": math.nan,
        }
    try:
        models = load_pose_models(pose_path)
    except (FileNotFoundError, ValueError):
        return {
            "geometry_total_poses": 0,
            "geometry_productive_poses": 0,
            "frac_prod": math.nan,
        }
    if not models:
        return {
            "geometry_total_poses": 0,
            "geometry_productive_poses": 0,
            "frac_prod": math.nan,
        }

    productive = 0
    evaluated = 0
    anchor_candidates = (
        ligand_anchor_candidates if ligand_anchor_candidates else n_candidates
    )

    for model in models:
        atoms = model.atoms
        n_atom = find_ligand_atom(atoms, n_candidates)
        electrophile_atom = find_ligand_atom(atoms, electrophile_candidates)

        if n_atom is None or electrophile_atom is None:
            evaluated += 1
            continue

        n_coord = (n_atom[5], n_atom[6], n_atom[7])
        electrophile_coord = (
            electrophile_atom[5],
            electrophile_atom[6],
            electrophile_atom[7],
        )

        ns_distance = _distance(sg_coord, n_coord)
        angle_value = _angle(sg_coord, n_coord, electrophile_coord)

        if ns_max is not None and ns_max > 0 and ns_distance > ns_max:
            evaluated += 1
            continue
        if (
            angle_min is not None
            and angle_max is not None
            and not (angle_min <= angle_value <= angle_max)
        ):
            evaluated += 1
            continue

        anchor_ok = True
        if anchor_coord is not None and anchor_max is not None and anchor_max > 0:
            anchor_atom = find_ligand_atom(atoms, anchor_candidates)
            if anchor_atom is None:
                anchor_ok = False
            else:
                anchor_coord_ligand = (
                    anchor_atom[5],
                    anchor_atom[6],
                    anchor_atom[7],
                )
                anchor_distance = _distance(anchor_coord, anchor_coord_ligand)
                anchor_ok = anchor_distance <= anchor_max
        if anchor_ok:
            productive += 1
        evaluated += 1

    fraction = productive / evaluated if evaluated else math.nan
    return {
        "geometry_total_poses": evaluated,
        "geometry_productive_poses": productive,
        "frac_prod": fraction,
    }

def parse_receptor_atom(
    receptor_path: Path,
    residue_number: int,
    atom_name: str,
    residue_name: str = "CYS",
    chain_id: str = "",
) -> Tuple[float, float, float]:
    """Locate a specific atom within the receptor file."""
    target_resn = residue_name.strip().upper()
    target_atom = atom_name.strip().upper()
    target_chain = chain_id.strip()
    with receptor_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            residue_name = line[17:20].strip()
            if residue_name.upper() != target_resn:
                continue
            residue_index = line[22:26].strip()
            if not residue_index:
                continue
            if int(residue_index) != residue_number:
                continue
            atom_name = line[12:16].strip()
            if atom_name.upper() != target_atom:
                continue
            if target_chain and line[21:22].strip() != target_chain:
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            return (x, y, z)
    raise ValueError(
        f"Could not locate {target_atom} atom for {target_resn} {residue_number} in {receptor_path}"
    )


def parse_receptor_cys209_sg(
    receptor_path: Path, residue_number: int = 209
) -> Tuple[float, float, float]:
    """Locate the SG atom for the catalytic cysteine within the receptor file."""
    return parse_receptor_atom(
        receptor_path=receptor_path,
        residue_number=residue_number,
        atom_name="SG",
        residue_name="CYS",
    )


def _iter_first_model_atom_lines(pdb_path: Path) -> Iterable[str]:
    """Yield ATOM/HETATM lines for the first model in the ligand PDB file."""
    with pdb_path.open(encoding="utf-8") as handle:
        in_model = False
        for line in handle:
            if line.startswith("MODEL"):
                if not in_model:
                    in_model = True
                continue
            if line.startswith("ENDMDL") and in_model:
                break
            if line.startswith(("ATOM", "HETATM")):
                if not in_model:
                    in_model = True
                yield line


def _atom_is_hydrogen(atom_name: str, element_hint: str) -> bool:
    """Identify hydrogen atoms using element symbol or atom name."""
    if element_hint:
        return element_hint.upper().startswith("H")
    stripped = atom_name.strip()
    return bool(stripped) and stripped[0].upper() == "H"


def min_distance_to_cys_sg(pdb_path: Path, sg_coord: Sequence[float]) -> float:
    """Minimum heavy-atom distance from the ligand pose to the cysteine SG atom."""
    min_distance: Optional[float] = None
    for line in _iter_first_model_atom_lines(pdb_path):
        atom_name = line[12:16]
        element = line[76:78].strip()
        if _atom_is_hydrogen(atom_name, element):
            continue
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        distance = math.dist((x, y, z), sg_coord)
        if min_distance is None or distance < min_distance:
            min_distance = distance
    if min_distance is None:
        raise ValueError(f"No heavy atoms detected in {pdb_path}")
    return min_distance


def resolve_pose_path(result_dir: Path, file_prefix: str) -> Optional[Path]:
    """Return the first pose file that exists for the given prefix."""
    for suffix in (".pdb", ".pdbqt"):
        candidate = result_dir / f"{file_prefix}_out{suffix}"
        if candidate.exists():
            return candidate
    return None


def resolve_geometry_pose_path(result_dir: Path, file_prefix: str) -> Optional[Path]:
    """Return the preferred pose file for geometry analysis."""
    for suffix in (".pdbqt", ".pdb"):
        candidate = result_dir / f"{file_prefix}_out{suffix}"
        if candidate.exists():
            return candidate
    return None


def attach_cys_distance(
    summary: dict[str, float | int | str],
    result_dir: Path,
    cys_coord: Optional[Tuple[float, float, float]],
) -> None:
    """Populate the Cys209 distance metric for a summary row."""
    if cys_coord is None:
        summary["cys209_distance"] = math.nan
        return
    pose_path = resolve_pose_path(result_dir, str(summary["file_prefix"]))
    if pose_path is None:
        print(
            f"Warning: pose file for {summary['ligand']} "
            f"({summary['file_prefix']}_out.*) is missing"
        )
        summary["cys209_distance"] = math.nan
        return
    try:
        summary["cys209_distance"] = min_distance_to_cys_sg(pose_path, cys_coord)
    except ValueError as exc:
        print(f"Warning: {exc}")
        summary["cys209_distance"] = math.nan


def write_csv(
    summary_rows: Iterable[dict[str, float | int | str]], output_path: Path
) -> None:
    """Write summary statistics to CSV."""
    rows = list(summary_rows)
    if not rows:
        raise ValueError("No rows provided to write_csv")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text_report(
    summary_rows: Iterable[dict[str, float | int | str]],
    output_path: Path,
    result_dir: Path,
) -> None:
    """Render a human-readable summary report."""
    title = f"Docking summary for {result_dir}"
    lines = [
        title,
        "=" * len(title),
        "",
        "More negative affinities indicate stronger predicted binding.",
        "",
    ]
    for row in summary_rows:
        distance = row.get("cys209_distance")
        rmsd_lb = row.get("best_rmsd_lb")
        rmsd_ub = row.get("best_rmsd_ub")
        if isinstance(rmsd_lb, (int, float)) and isinstance(rmsd_ub, (int, float)):
            if math.isnan(rmsd_lb) or math.isnan(rmsd_ub):
                rmsd_msg = "RMSD N/A"
            else:
                rmsd_msg = f"RMSD {rmsd_lb:.2f}-{rmsd_ub:.2f} Å"
        else:
            rmsd_msg = "RMSD N/A"
        distance_msg = (
            f"; min heavy-atom distance to Cys209 SG {distance:.2f} Å"
            if isinstance(distance, (int, float)) and not math.isnan(distance)
            else ""
        )
        frac = row.get("frac_prod")
        productive = row.get("geometry_productive_poses")
        total = row.get("geometry_total_poses")
        if (
            isinstance(frac, (int, float))
            and not math.isnan(frac)
            and isinstance(productive, (int, float))
            and isinstance(total, (int, float))
            and total > 0
        ):
            frac_msg = f"; Frac_prod {int(productive)}/{int(total)} ({frac*100:.1f}%)"
        else:
            frac_msg = ""
        lines.append(
            f"{row['ligand']}: best affinity {row['best_affinity']:.2f} kcal/mol "
            f"(mode {row['best_mode']}, {rmsd_msg}); "
            f"avg top3 {row['mean_top3']:.2f} kcal/mol; "
            f"top5 spread {row['top5_span']:.2f} kcal/mol{distance_msg}{frac_msg}."
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_best_affinities(
    summary_rows: Iterable[dict[str, float | int | str]], output_path: Path
) -> None:
    """Create a horizontal bar chart for best affinities per ligand."""
    rows = sorted(summary_rows, key=lambda row: row["best_affinity"])
    ligands = [row["ligand"] for row in rows]
    affinities = [row["best_affinity"] for row in rows]

    if not ligands:
        return

    width = max(3.5, 0.65 * len(ligands))
    height = max(5.0, 1.4 * len(ligands))
    fig, ax = plt.subplots(figsize=(width, height))
    positions = list(range(len(ligands)))
    plot_values = [-aff for aff in affinities]
    bar_container = ax.bar(
        positions,
        plot_values,
        color="#2a9d8f",
        edgecolor="#1f6f63",
        linewidth=1.1,
        width=0.42,
    )

    for bar, affinity, plot_value in zip(bar_container, affinities, plot_values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            plot_value + 0.05,
            f"{affinity:.2f}",
            va="bottom",
            ha="center",
            color="#0f1c1a",
            fontsize=7.5,
            fontweight="bold",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(ligands, rotation=18, ha="right", fontsize=7, fontweight="bold")
    ax.set_xlabel(
        "Ligand",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Best affinity (kcal/mol)",
        fontsize=8,
        fontweight="bold",
    )
    ymax = max(plot_values) * 1.05
    ax.set_ylim(0, ymax if ymax > 0 else 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{-val:.1f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=1.0)
    ax.tick_params(axis="y", labelsize=8, width=1.0, length=3)
    ax.tick_params(axis="x", labelsize=8, width=0.8, length=2, pad=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(
        "AutoDock Vina predicted affinities (lower is better)",
        fontsize=7.5,
        fontweight="bold",
        x=0.5,
        y=0.99,
        ha="center",
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_affinity_vs_distance(
    summary_rows: Iterable[dict[str, float | int | str]], output_path: Path
) -> None:
    """Scatter plot showing affinity against Cys209 proximity."""
    rows = [
        row
        for row in summary_rows
        if isinstance(row.get("cys209_distance"), (int, float))
        and not math.isnan(row["cys209_distance"])
    ]
    if not rows:
        return

    distances = [row["cys209_distance"] for row in rows]
    affinities = [row["best_affinity"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        distances,
        affinities,
        color="#e76f51",
        edgecolor="#9c3a2d",
        s=80,
        linewidths=1.1,
    )

    for row in rows:
        ax.annotate(
            row["ligand"],
            (row["cys209_distance"], row["best_affinity"]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel(
        "Minimum heavy-atom distance to Cys209 SG (Å)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Best affinity (kcal/mol)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_title(
        "Affinity vs. distance to catalytic Cys209",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=1.0)
    ax.tick_params(axis="both", labelsize=13, width=1.2, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _parse_extra_pdbqt(values: Sequence[str]) -> dict[str, str]:
    """Parse NAME=PATH overrides for extra PDBQT ligands."""
    result: dict[str, str] = {}
    for entry in values:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                f"Invalid --extra-pdbqt value '{entry}'. Expected NAME=PATH."
            )
        name, rel_path = entry.split("=", maxsplit=1)
        name = name.strip()
        rel_path = rel_path.strip()
        if not name or not rel_path:
            raise argparse.ArgumentTypeError(
                f"Invalid --extra-pdbqt value '{entry}'. Expected NAME=PATH."
            )
        result[name] = rel_path
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build and parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize AutoDock Vina results and generate plots."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=f"Directory containing docking logs and poses (default: {DEFAULT_RESULT_DIR})",
    )
    parser.add_argument(
        "--skip-ligand",
        dest="skip_ligands",
        action="append",
        default=None,
        metavar="NAME",
        help="Ligand name to exclude (can be repeated).",
    )
    parser.add_argument(
        "--clear-default-skips",
        action="store_true",
        help="Do not apply the built-in skip ligand list.",
    )
    parser.add_argument(
        "--extra-pdbqt",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help=(
            "Additional ligand inference from a PDBQT file when no log exists "
            "(can be repeated)."
        ),
    )
    parser.add_argument(
        "--clear-default-extra",
        action="store_true",
        help="Do not include the built-in extra PDBQT ligands.",
    )
    parser.add_argument(
        "--geometry",
        action="store_true",
        help=(
            "Compute productive conformation fraction using catalytic geometry "
            "criteria."
        ),
    )
    parser.add_argument(
        "--ns-max",
        type=float,
        default=3.5,
        help="Maximum distance (Å) between ligand N and Cys SG for productivity.",
    )
    parser.add_argument(
        "--angle-min",
        type=float,
        default=95.0,
        help="Minimum attack (Bürgi–Dunitz) angle in degrees.",
    )
    parser.add_argument(
        "--angle-max",
        type=float,
        default=125.0,
        help="Maximum attack (Bürgi–Dunitz) angle in degrees.",
    )
    parser.add_argument(
        "--n-atoms",
        default="N,N1,N2",
        help="Comma-separated ligand atom names representing the nucleophilic N.",
    )
    parser.add_argument(
        "--electrophile-atoms",
        default="C,C7,C8",
        help=(
            "Comma-separated ligand atom names for the electrophilic carbon used "
            "to evaluate the attack angle."
        ),
    )
    parser.add_argument(
        "--ligand-anchor-atoms",
        default=None,
        help=(
            "Comma-separated ligand atom names for the anchor hydrogen-bond "
            "distance (defaults to the N atom set when omitted)."
        ),
    )
    parser.add_argument(
        "--anchor-resn",
        help="Residue name of the receptor anchor atom.",
    )
    parser.add_argument(
        "--anchor-resi",
        help="Residue index of the receptor anchor atom.",
    )
    parser.add_argument(
        "--anchor-atom",
        help="Atom name of the receptor anchor atom.",
    )
    parser.add_argument(
        "--anchor-chain",
        default="",
        help="Optional chain identifier for the receptor anchor atom.",
    )
    parser.add_argument(
        "--anchor-distance-max",
        type=float,
        default=None,
        help=(
            "Maximum distance (Å) between the receptor anchor atom and the ligand "
            "anchor atom. Set to a positive value to enable the anchor constraint."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result_dir = args.result_dir
    if not result_dir.exists():
        raise SystemExit(f"Result directory {result_dir} does not exist.")

    log_files = sorted(
        set(result_dir.glob("*_log.txt")) | set(result_dir.glob("*.log"))
    )
    if not log_files:
        raise SystemExit(f"No log files found in {result_dir}")

    receptor_path = result_dir / "receptor.pdbqt"
    cys_coord: Optional[Tuple[float, float, float]] = None
    if receptor_path.exists():
        try:
            cys_coord = parse_receptor_cys209_sg(receptor_path)
        except ValueError as exc:
            print(f"Warning: {exc}")
    else:
        print(f"Warning: receptor file {receptor_path} not found; geometry metrics disabled.")

    geometry_enabled = bool(args.geometry)
    anchor_coord: Optional[Tuple[float, float, float]] = None
    geometry_config: Optional[dict[str, object]] = None
    if geometry_enabled:
        if cys_coord is None:
            print(
                "Warning: catalytic cysteine coordinates unavailable; "
                "skipping geometry evaluation."
            )
        else:
            n_candidates = [
                item.strip()
                for item in args.n_atoms.split(",")
                if item.strip()
            ]
            electrophile_candidates = [
                item.strip()
                for item in args.electrophile_atoms.split(",")
                if item.strip()
            ]
            ligand_anchor_candidates: Optional[list[str]] = None
            if args.ligand_anchor_atoms:
                ligand_anchor_candidates = [
                    item.strip()
                    for item in args.ligand_anchor_atoms.split(",")
                    if item.strip()
                ]

            if args.anchor_resn and args.anchor_resi and args.anchor_atom:
                try:
                    anchor_resi_value = int(args.anchor_resi)
                except ValueError:
                    anchor_resi_value = _parse_residue_index(args.anchor_resi)
                try:
                    anchor_coord = parse_receptor_atom(
                        receptor_path=receptor_path,
                        residue_number=anchor_resi_value,
                        atom_name=args.anchor_atom,
                        residue_name=args.anchor_resn,
                        chain_id=args.anchor_chain,
                    )
                except ValueError as exc:
                    print(f"Warning: {exc}")
                    anchor_coord = None
                except FileNotFoundError as exc:
                    print(f"Warning: {exc}")
                    anchor_coord = None

            geometry_config = {
                "sg_coord": cys_coord,
                "n_candidates": n_candidates,
                "electrophile_candidates": electrophile_candidates,
                "ns_max": args.ns_max if args.ns_max and args.ns_max > 0 else None,
                "angle_min": args.angle_min,
                "angle_max": args.angle_max,
                "anchor_coord": anchor_coord,
                "ligand_anchor_candidates": ligand_anchor_candidates,
                "anchor_max": (
                    args.anchor_distance_max
                    if args.anchor_distance_max and args.anchor_distance_max > 0
                    else None
                ),
            }

    skip_ligands: set[str] = set()
    if not args.clear_default_skips:
        skip_ligands.update(DEFAULT_SKIP_LIGAND_NAMES)
    if args.skip_ligands:
        skip_ligands.update(args.skip_ligands)

    extra_pdbqt: dict[str, str] = {}
    if not args.clear_default_extra:
        extra_pdbqt.update(DEFAULT_EXTRA_PDBQT_LIGANDS)
    if args.extra_pdbqt:
        extra_pdbqt.update(_parse_extra_pdbqt(args.extra_pdbqt))

    default_geometry_result: dict[str, float | int | str] = {
        "geometry_total_poses": 0,
        "geometry_productive_poses": 0,
        "frac_prod": math.nan,
    }

    all_summaries: list[dict[str, float | int | str]] = []
    for log_path in log_files:
        poses = parse_log_file(log_path)
        summary = summarise_ligand(poses)
        if summary["ligand"] in skip_ligands:
            continue
        attach_cys_distance(summary, result_dir, cys_coord)
        if geometry_enabled:
            if geometry_config is None:
                summary.update(default_geometry_result)
            else:
                pose_path_geom = resolve_geometry_pose_path(
                    result_dir, str(summary["file_prefix"])
                )
                if pose_path_geom is None:
                    summary.update(default_geometry_result)
                else:
                    geometry_result = evaluate_productive_fraction(
                        pose_path_geom,
                        geometry_config["sg_coord"],
                        geometry_config["n_candidates"],
                        geometry_config["electrophile_candidates"],
                        geometry_config["ns_max"],
                        geometry_config["angle_min"],
                        geometry_config["angle_max"],
                        geometry_config["anchor_coord"],
                        geometry_config["ligand_anchor_candidates"],
                        geometry_config["anchor_max"],
                    )
                    summary.update(geometry_result)
        all_summaries.append(summary)

    for ligand_name, relative_path in extra_pdbqt.items():
        pdbqt_path = result_dir / relative_path
        if not pdbqt_path.exists():
            print(f"Warning: extra pose file {pdbqt_path} is missing")
            continue
        try:
            summary = summarise_pdbqt(ligand_name, pdbqt_path)
        except ValueError as exc:
            print(f"Warning: {exc}")
            continue
        attach_cys_distance(summary, result_dir, cys_coord)
        if geometry_enabled:
            if geometry_config is None:
                summary.update(default_geometry_result)
            else:
                geometry_result = evaluate_productive_fraction(
                    pdbqt_path,
                    geometry_config["sg_coord"],
                    geometry_config["n_candidates"],
                    geometry_config["electrophile_candidates"],
                    geometry_config["ns_max"],
                    geometry_config["angle_min"],
                    geometry_config["angle_max"],
                    geometry_config["anchor_coord"],
                    geometry_config["ligand_anchor_candidates"],
                    geometry_config["anchor_max"],
                )
                summary.update(geometry_result)
        all_summaries.append(summary)

    public_rows: list[dict[str, float | int | str]] = []
    for row in all_summaries:
        row_copy = {k: v for k, v in row.items() if k != "file_prefix"}
        public_rows.append(row_copy)

    csv_path = result_dir / "docking_summary.csv"
    write_csv(public_rows, csv_path)

    text_report_path = result_dir / "docking_summary.txt"
    write_text_report(public_rows, text_report_path, result_dir)

    plot_path = result_dir / "docking_best_affinities.png"
    plot_best_affinities(public_rows, plot_path)

    scatter_path = result_dir / "docking_affinity_vs_cys_distance.png"
    plot_affinity_vs_distance(public_rows, scatter_path)
    print(f"Wrote summary CSV to {csv_path}")
    print(f"Wrote text report to {text_report_path}")
    print(f"Wrote bar chart to {plot_path}")
    if cys_coord is not None:
        print(f"Wrote affinity-distance scatter plot to {scatter_path}")


if __name__ == "__main__":
    main()
