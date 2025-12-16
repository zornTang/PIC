#!/usr/bin/env python3
"""
Score docking poses for catalytic geometry and affinity directly from PDBQT files.

Outputs a CSV compatible with pareto_select.py containing:
ligand,file,model,affinity,d_sg_cyl,d_n_to_anch,angle_bd,meets_geometry,reason
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# -------------------------- geometry helpers -------------------------- #

AtomRecord = Tuple[int, str, str, str, str, float, float, float]
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Return angle ABC in degrees."""
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


def parse_resi(resi: str) -> int:
    """Extract an integer residue index for sorting (fallback to large int)."""
    try:
        digits = re.findall(r"-?\d+", resi)
        if not digits:
            return 10**9
        return int(digits[0])
    except Exception:
        return 10**9


# -------------------------- PDBQT parsing ------------------------------ #


def parse_affinity(line: str) -> Optional[float]:
    upper = line.upper()
    if "VINA RESULT" in upper or "MINIMIZEDAFFINITY" in upper:
        matches = FLOAT_RE.findall(line)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                return None
    return None


def parse_pdbqt_models(pdbqt_path: Path) -> List[dict]:
    """Parse a PDBQT into a list of models including atoms and affinity."""
    models: List[dict] = []
    current_atoms: List[AtomRecord] = []
    current_affinity: Optional[float] = None
    model_index: Optional[int] = None
    explicit_model = False
    next_model_id = 1

    def flush() -> None:
        nonlocal current_atoms, current_affinity, model_index, next_model_id, explicit_model
        if not current_atoms:
            return
        assigned_model = model_index if model_index is not None else next_model_id
        models.append(
            {
                "model": assigned_model,
                "atoms": current_atoms,
                "affinity": current_affinity,
            }
        )
        current_atoms = []
        current_affinity = None
        if not explicit_model:
            next_model_id += 1
        model_index = None

    with pdbqt_path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            rec = line[:6].strip()
            if rec == "MODEL":
                flush()
                explicit_model = True
                parts = line.split()
                if len(parts) > 1:
                    try:
                        model_index = int(parts[1])
                    except ValueError:
                        model_index = None
                else:
                    model_index = None
                continue
            if line.startswith("ENDMDL"):
                flush()
                explicit_model = False
                continue
            if rec in {"ATOM", "HETATM"}:
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
                affinity_value = parse_affinity(line)
                if affinity_value is not None:
                    current_affinity = affinity_value
                continue
        flush()
    return models


# ----------------------- receptor utilities ---------------------------- #


def find_receptor_atom(
    atoms: Sequence[AtomRecord],
    resn: str,
    resi: str,
    atom_name: str,
    chain: str = "",
) -> Tuple[float, float, float]:
    target_resn = resn.upper()
    target_atom = atom_name.upper()
    target_resi = resi.strip()
    target_chain = chain.strip()
    for record in atoms:
        _serial, name, rresn, rresi, rchain, x, y, z = record
        if name.strip().upper() != target_atom:
            continue
        if rresn.strip().upper() != target_resn:
            continue
        if rresi.strip() != target_resi:
            continue
        if target_chain and rchain.strip() != target_chain:
            continue
        return (x, y, z)
    raise FileNotFoundError(
        f"Unable to locate receptor atom {target_resn} {target_resi} {target_atom}"
        + (f" chain {target_chain}" if target_chain else "")
    )


# ----------------------- ligand utilities ------------------------------ #


def find_candidate_atom(atoms: Sequence[AtomRecord], candidates: Sequence[str]) -> Optional[AtomRecord]:
    candidate_set = {name.strip().upper() for name in candidates if name.strip()}
    best: Optional[AtomRecord] = None
    best_resi = 10**9
    for record in atoms:
        if record[1].strip().upper() not in candidate_set:
            continue
        resi_value = parse_resi(record[3])
        if best is None or resi_value < best_resi:
            best = record
            best_resi = resi_value
    return best


# ----------------------------- scoring -------------------------------- #


def score_pose(
    ligand_name: str,
    ligand_file: Path,
    model: dict,
    cat_xyz: Tuple[float, float, float],
    anchor_xyz: Optional[Tuple[float, float, float]],
    n_names: Sequence[str],
    carbon_names: Sequence[str],
    sg_cyl_max: Optional[float],
    n_anchor_max: Optional[float],
    bd_min: Optional[float],
    bd_max: Optional[float],
) -> dict:
    atoms = model["atoms"]
    n_atom = find_candidate_atom(atoms, n_names)
    c_atom = find_candidate_atom(atoms, carbon_names)
    reason_parts: List[str] = []
    ok = True

    if n_atom is None:
        ok = False
        reason_parts.append("N_missing")
    if c_atom is None:
        ok = False
        reason_parts.append("Cyl_missing")

    d_sg_cyl: Optional[float] = None
    d_n_anch: Optional[float] = None
    angle_bd: Optional[float] = None

    if c_atom is not None:
        d_sg_cyl = dist(cat_xyz, (c_atom[5], c_atom[6], c_atom[7]))
        if sg_cyl_max is not None and d_sg_cyl > sg_cyl_max:
            ok = False
            reason_parts.append(f"d_sg_cyl>{sg_cyl_max:.2f}")
    if anchor_xyz is not None and n_atom is not None:
        d_n_anch = dist(anchor_xyz, (n_atom[5], n_atom[6], n_atom[7]))
        if n_anchor_max is not None and d_n_anch > n_anchor_max:
            ok = False
            reason_parts.append(f"d_n_to_anch>{n_anchor_max:.2f}")
    elif anchor_xyz is not None:
        reason_parts.append("N_missing_for_anchor")

    if n_atom is not None and c_atom is not None:
        angle_bd = angle(cat_xyz, (n_atom[5], n_atom[6], n_atom[7]), (c_atom[5], c_atom[6], c_atom[7]))
        if (
            angle_bd is not None
            and not math.isnan(angle_bd)
            and bd_min is not None
            and bd_max is not None
        ):
            if not (bd_min <= angle_bd <= bd_max):
                ok = False
                reason_parts.append(f"angle_bd_outside[{bd_min:.0f},{bd_max:.0f}]")
    else:
        reason_parts.append("angle_na")

    if ok and not reason_parts:
        reason_parts.append("ok")
    if ok and reason_parts and reason_parts[-1] != "ok":
        reason_parts.append("ok")

    def classify(distance: Optional[float], angle_value: Optional[float], anchor_dist: Optional[float]) -> str:
        if distance is None or math.isnan(distance):
            return "inactive"
        if distance > 4.0:
            return "inactive"
        angle_ok = angle_value is not None and not math.isnan(angle_value) and 100.0 <= angle_value <= 112.0
        anchor_ok = anchor_dist is None or math.isnan(anchor_dist) or (n_anchor_max is None or anchor_dist <= n_anchor_max)
        if distance <= 3.8 and angle_ok and anchor_ok:
            return "reactive"
        return "pseudo_binding"

    reaction_state = classify(d_sg_cyl, angle_bd, d_n_anch)

    return {
        "ligand": ligand_name,
        "file": ligand_file.name,
        "model": model["model"],
        "affinity": model.get("affinity"),
        "d_sg_cyl": None if d_sg_cyl is None else round(d_sg_cyl, 3),
        "d_n_to_anch": None if d_n_anch is None else round(d_n_anch, 3),
        "angle_bd": None if angle_bd is None or math.isnan(angle_bd) else round(angle_bd, 2),
        "meets_geometry": 1 if ok else 0,
        "reason": ";".join(reason_parts),
        "reaction_feasibility": reaction_state,
    }


# ----------------------------- CLI ------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score catalytic geometry for PDBQT docking poses."
    )
    parser.add_argument(
        "-r",
        "--receptor",
        required=True,
        type=Path,
        help="Receptor PDBQT containing the catalytic cysteine.",
    )
    parser.add_argument(
        "--cys-resi",
        required=True,
        help="Residue index of the catalytic cysteine (CAT).",
    )
    parser.add_argument(
        "--cys-resn",
        default="CYS",
        help="Residue name for the catalytic cysteine (default: CYS).",
    )
    parser.add_argument(
        "--cys-chain",
        default="",
        help="Optional chain ID for the catalytic cysteine.",
    )
    parser.add_argument(
        "--cys-atom",
        default="SG",
        help="Atom name for the catalytic sulfur (default: SG).",
    )
    parser.add_argument(
        "--anchor-resn",
        help="Residue name for optional anchor atom.",
    )
    parser.add_argument(
        "--anchor-resi",
        help="Residue index for optional anchor atom.",
    )
    parser.add_argument(
        "--anchor-atom",
        help="Atom name for optional anchor atom.",
    )
    parser.add_argument(
        "--anchor-chain",
        default="",
        help="Chain identifier for optional anchor atom.",
    )
    parser.add_argument(
        "--n-names",
        default="N,N1,N2",
        help="Comma-separated candidate atom names for ligand N-terminus.",
    )
    parser.add_argument(
        "--carbon-names",
        default="C,C7,C8",
        help="Comma-separated candidate atom names for ligand electrophilic carbon.",
    )
    parser.add_argument(
        "--sg-cyl-max",
        type=float,
        default=3.6,
        help="Maximum allowed distance (Å) between Cys SG and ligand electrophilic carbon (<= disables when set to <=0).",
    )
    parser.add_argument(
        "--n-anchor-max",
        type=float,
        default=4.0,
        help="Maximum allowed distance (Å) between ligand N and anchor atom (<=0 disables).",
    )
    parser.add_argument(
        "--bd-min",
        type=float,
        default=95.0,
        help="Minimum Bürgi–Dunitz angle (degrees).",
    )
    parser.add_argument(
        "--bd-max",
        type=float,
        default=125.0,
        help="Maximum Bürgi–Dunitz angle (degrees).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pose_geometry_scores.csv"),
        help="Output CSV path (default: pose_geometry_scores.csv).",
    )
    parser.add_argument(
        "--ligand-label",
        help="Optional ligand label to use instead of file stem (only valid for a single input file).",
    )
    parser.add_argument(
        "ligands",
        nargs="+",
        type=Path,
        help="One or more ligand PDBQT files to score.",
    )
    args = parser.parse_args()
    if args.ligand_label and len(args.ligands) > 1:
        parser.error("--ligand-label can only be used with a single ligand file.")
    return args


# ----------------------------- main ----------------------------------- #


def load_receptor_atoms(receptor_path: Path) -> Sequence[AtomRecord]:
    models = parse_pdbqt_models(receptor_path)
    if not models:
        raise ValueError(f"No atoms parsed from receptor PDBQT: {receptor_path}")
    return models[0]["atoms"]


def main() -> None:
    args = parse_args()
    receptor_path = args.receptor.expanduser().resolve()
    if not receptor_path.exists():
        raise FileNotFoundError(f"Receptor file not found: {receptor_path}")

    receptor_atoms = load_receptor_atoms(receptor_path)
    cat_xyz = find_receptor_atom(
        receptor_atoms,
        args.cys_resn,
        args.cys_resi,
        args.cys_atom,
        args.cys_chain,
    )

    anchor_xyz: Optional[Tuple[float, float, float]] = None
    if args.anchor_resn or args.anchor_resi or args.anchor_atom:
        if not (args.anchor_resn and args.anchor_resi and args.anchor_atom):
            raise ValueError("Anchor specification requires --anchor-resn, --anchor-resi, and --anchor-atom.")
        anchor_xyz = find_receptor_atom(
            receptor_atoms,
            args.anchor_resn,
            args.anchor_resi,
            args.anchor_atom,
            args.anchor_chain,
        )

    n_names = [name.strip() for name in args.n_names.split(",") if name.strip()]
    carbon_names = [name.strip() for name in args.carbon_names.split(",") if name.strip()]
    sg_cyl_max = args.sg_cyl_max if args.sg_cyl_max and args.sg_cyl_max > 0 else None
    n_anchor_max = args.n_anchor_max if args.n_anchor_max and args.n_anchor_max > 0 else None
    bd_min = args.bd_min if args.bd_min is not None else None
    bd_max = args.bd_max if args.bd_max is not None else None

    rows: List[dict] = []
    for ligand_path in args.ligands:
        ligand_path = ligand_path.expanduser().resolve()
        if not ligand_path.exists():
            raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
        models = parse_pdbqt_models(ligand_path)
        if not models:
            continue
        ligand_name = args.ligand_label or ligand_path.stem
        for model in models:
            row = score_pose(
                ligand_name,
                ligand_path,
                model,
                cat_xyz,
                anchor_xyz,
                n_names,
                carbon_names,
                sg_cyl_max,
                n_anchor_max,
                bd_min,
                bd_max,
            )
            rows.append(row)

    if not rows:
        print("No poses scored.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ligand",
        "file",
        "model",
        "affinity",
        "d_sg_cyl",
        "d_n_to_anch",
        "angle_bd",
        "meets_geometry",
        "reason",
        "reaction_feasibility",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(row["meets_geometry"] for row in rows)
    print(f"Wrote {len(rows)} pose scores to {args.output} (meets_geometry={passed}).")


if __name__ == "__main__":
    main()
