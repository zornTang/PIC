#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _fg_paths import CCK8_293T_XLSX, CCK8_HL60_XLSX, FIGURES_DIR
ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_cck8_file1_figure import (  # type: ignore
    DISPLAY_GROUPS,
    extract_difference_matrix,
    group_values,
    mean,
    p_to_stars,
    welch_p_value,
    wells_for_group,
)
from npj_redraw_figures import COLORS, apply_npj_style, polish, emphasize_panel


OUT_DIR = FIGURES_DIR
COMMON_ORDER = ["b2_pma", "b2", "sc_pma", "sc", "wt_pma"]
COMMON_LABELS = {
    "b2_pma": "ATP6V1B2\n+PMA",
    "b2": "ATP6V1B2",
    "sc_pma": "Scramble\n+PMA",
    "sc": "Scramble",
    "wt_pma": "PMA",
    "pma_only": "PMA",
}
COMMON_SOURCE_KEYS_293T = {
    "b2_pma": "b2_pma",
    "b2": "b2_dmso",
    "sc_pma": "sc_pma",
    "sc": "sc_dmso",
    "wt_pma": "pma_only",
}
COMMON_COLORS = {
    "b2_pma": COLORS["orange"],
    "b2": "#E7AD75",
    "sc_pma": COLORS["teal"],
    "sc": "#8BC6BC",
    "wt_pma": COLORS["gray"],
    "pma_only": COLORS["gray"],
}
DISPLAY_GROUPS_HL60 = [
    {"key": "b2_pma", "plot_label": "ATP6V1B2\n+PMA", "source_groups": [1], "blank_group": 8, "control_group": 6, "color": COMMON_COLORS["b2_pma"]},
    {"key": "b2", "plot_label": "ATP6V1B2", "source_groups": [2], "blank_group": 8, "control_group": 6, "color": COMMON_COLORS["b2"]},
    {"key": "sc_pma", "plot_label": "Scramble\n+PMA", "source_groups": [3], "blank_group": 8, "control_group": 6, "color": COMMON_COLORS["sc_pma"]},
    {"key": "sc", "plot_label": "Scramble", "source_groups": [4], "blank_group": 8, "control_group": 6, "color": COMMON_COLORS["sc"]},
    {"key": "wt_pma", "plot_label": "PMA", "source_groups": [5], "blank_group": 8, "control_group": 6, "color": COMMON_COLORS["wt_pma"]},
]


def normalize_groups_293t() -> list[dict[str, object]]:
    matrix = extract_difference_matrix(CCK8_293T_XLSX)
    groups: list[dict[str, object]] = []
    display_by_key = {str(group["key"]): group for group in DISPLAY_GROUPS}
    for key in COMMON_ORDER:
        source_key = COMMON_SOURCE_KEYS_293T[key]
        group = display_by_key[source_key]
        blank = mean(wells_for_group(matrix, int(group["blank_group"])))
        control = mean(wells_for_group(matrix, int(group["control_group"])))
        values = group_values(matrix, group)
        normalized = [((value - blank) / (control - blank)) * 100 for value in values]
        groups.append({"key": key, "plot_label": COMMON_LABELS[key], "values": normalized, "color": COMMON_COLORS[key]})
    return groups


def normalize_groups_hl60() -> list[dict[str, object]]:
    matrix = extract_difference_matrix(CCK8_HL60_XLSX)
    groups: list[dict[str, object]] = []
    for group in DISPLAY_GROUPS_HL60:
        blank = mean(wells_for_group(matrix, int(group["blank_group"])))
        control = mean(wells_for_group(matrix, int(group["control_group"])))
        values = group_values(matrix, group)
        normalized = [((value - blank) / (control - blank)) * 100 for value in values]
        groups.append({"key": group["key"], "plot_label": group["plot_label"], "values": normalized, "color": group["color"]})
    return groups


def add_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, label: str) -> None:
    h = 2.4
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=COLORS["dark"], lw=1.25, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 1.3, label, ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLORS["dark"])


def plot_panel(ax: plt.Axes, groups: list[dict[str, object]], reference_key: str, comparison_keys: list[str]) -> None:
    x = np.arange(len(groups))
    means = [float(np.mean(g["values"])) for g in groups]
    sds = [float(np.std(g["values"], ddof=1)) for g in groups]
    colors = [str(g["color"]) for g in groups]

    ax.bar(
        x,
        means,
        yerr=sds,
        width=0.66,
        color=colors,
        edgecolor=COLORS["dark"],
        linewidth=0.9,
        capsize=2.8,
        error_kw={"elinewidth": 1.0, "ecolor": COLORS["dark"], "capthick": 1.0},
        zorder=2,
    )
    for i, group in enumerate(groups):
        values = np.asarray(group["values"], dtype=float)
        jitter = np.linspace(-0.15, 0.15, len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=22,
            facecolors=colors[i],
            edgecolors="white",
            linewidths=0.85,
            alpha=0.95,
            zorder=3,
        )

    ax.axhline(100, color=COLORS["gray"], ls="--", lw=0.8, dashes=(3, 2), zorder=1)
    ax.set_ylim(0, 148)
    ax.set_yticks(np.arange(0, 121, 20))
    ax.set_xticks(x)
    ax.set_xticklabels([str(g["plot_label"]) for g in groups], rotation=35, ha="right")
    ax.set_ylabel("Viability (%)")
    ax.grid(False)
    polish(ax)
    emphasize_panel(ax)
    ax.tick_params(axis="x", pad=3.0)
    ax.set_ylabel("Viability (%)", fontweight="bold")

    index_by_key = {str(g["key"]): idx for idx, g in enumerate(groups)}
    ref_idx = index_by_key[reference_key]
    ref_vals = list(groups[ref_idx]["values"])
    base_y = max(m + s for m, s in zip(means, sds)) + 4.0
    for level, key in enumerate(comparison_keys):
        idx = index_by_key[key]
        label = p_to_stars(welch_p_value(list(groups[idx]["values"]), ref_vals))
        add_sig_bracket(ax, ref_idx, idx, base_y + level * 10.5, label)


def save_panel(groups: list[dict[str, object]], reference_key: str, comparison_keys: list[str], stem: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 2.7), constrained_layout=True)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    plot_panel(ax, groups, reference_key, comparison_keys)
    for ext in ("png",):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", facecolor="none", transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_npj_style()
    comparison_keys = ["b2", "sc_pma", "wt_pma"]
    save_panel(normalize_groups_293t(), "b2_pma", comparison_keys, "fig_10e_cck8_293t")
    save_panel(normalize_groups_hl60(), "b2_pma", comparison_keys, "fig_10f_cck8_hl60")
    print(f"Wrote {OUT_DIR / 'fig_10e_cck8_293t.png'}")
    print(f"Wrote {OUT_DIR / 'fig_10f_cck8_hl60.png'}")


if __name__ == "__main__":
    main()
