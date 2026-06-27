#!/usr/bin/env python3
from __future__ import annotations

import math
import statistics
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import mpmath as mp


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


DISPLAY_GROUPS = [
    {
        "key": "b2_pma",
        "label": "B2 +\nPMA",
        "plot_label": "B2+PMA",
        "source_groups": [3, 4],
        "pairwise_mean": True,
        "blank_group": 1,
        "control_group": 10,
        "color": "#D4622A",
    },
    {
        "key": "sc_pma",
        "label": "SC +\nPMA",
        "plot_label": "SC+PMA",
        "source_groups": [5, 6],
        "pairwise_mean": True,
        "blank_group": 1,
        "control_group": 10,
        "color": "#2C5F8A",
    },
    {
        "key": "b2_dmso",
        "label": "B2\nno PMA",
        "plot_label": "B2",
        "source_groups": [7],
        "blank_group": 2,
        "control_group": 10,
        "color": "#E8A982",
    },
    {
        "key": "sc_dmso",
        "label": "SC\nno PMA",
        "plot_label": "SC",
        "source_groups": [8],
        "blank_group": 2,
        "control_group": 10,
        "color": "#9FBFE0",
    },
    {
        "key": "pma_only",
        "label": "PMA\nonly",
        "plot_label": "PMA",
        "source_groups": [9],
        "blank_group": 1,
        "control_group": 10,
        "color": "#B7BEC8",
    },
]


def col_letters_to_index(letters: str) -> int:
    value = 0
    for char in letters:
        value = value * 26 + (ord(char.upper()) - 64)
    return value


def split_ref(ref: str) -> tuple[int, int]:
    letters = "".join(ch for ch in ref if ch.isalpha())
    digits = "".join(ch for ch in ref if ch.isdigit())
    return int(digits), col_letters_to_index(letters)


def load_shared_strings(xlsx_path: Path) -> list[str]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_path = "xl/sharedStrings.xml"
        if shared_path not in zf.namelist():
            return []
        root = ET.fromstring(zf.read(shared_path))
        strings = []
        for si in root.findall("a:si", NS):
            text = "".join(node.text or "" for node in si.findall(".//a:t", NS))
            strings.append(text)
        return strings


def load_cells(xlsx_path: Path) -> dict[tuple[int, int], object]:
    shared_strings = load_shared_strings(xlsx_path)
    with zipfile.ZipFile(xlsx_path) as zf:
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    cells: dict[tuple[int, int], object] = {}
    for cell in root.findall(".//a:c", NS):
        ref = cell.get("r")
        if not ref:
            continue
        key = split_ref(ref)
        cell_type = cell.get("t")
        value_node = cell.find("a:v", NS)
        if value_node is None or value_node.text is None:
            continue
        raw = value_node.text
        if cell_type == "s":
            value: object = shared_strings[int(raw)]
        else:
            try:
                numeric = float(raw)
                value = int(numeric) if numeric.is_integer() else numeric
            except ValueError:
                value = raw
        cells[key] = value
    return cells


def extract_difference_matrix(xlsx_path: Path) -> dict[str, dict[int, float]]:
    cells = load_cells(xlsx_path)
    difference_anchor = None
    for key, value in cells.items():
        if value == "Difference":
            difference_anchor = key
            break
    if difference_anchor is None:
        raise RuntimeError("Could not find the 'Difference' table in the workbook.")

    start_row, start_col = difference_anchor
    matrix: dict[str, dict[int, float]] = {}
    for offset, row_name in enumerate("ABCDEFGH", start=2):
        row_idx = start_row + offset
        label = cells.get((row_idx, start_col))
        if label != row_name:
            raise RuntimeError(f"Unexpected row label at row {row_idx}: {label!r}")
        matrix[row_name] = {}
        for plate_col in range(1, 13):
            value = cells.get((row_idx, start_col + plate_col))
            if value is None:
                raise RuntimeError(f"Missing value for well {row_name}{plate_col}")
            matrix[row_name][plate_col] = float(value)
    return matrix


def wells_for_group(matrix: dict[str, dict[int, float]], group_number: int) -> list[float]:
    plate_col = group_number + 1
    return [matrix[row][plate_col] for row in "BCDEFG"]


def combine_groups(matrix: dict[str, dict[int, float]], group_numbers: list[int]) -> list[float]:
    values = []
    for group_number in group_numbers:
        values.extend(wells_for_group(matrix, group_number))
    return values


def group_values(matrix: dict[str, dict[int, float]], group: dict[str, object]) -> list[float]:
    source_groups = group["source_groups"]
    if group.get("pairwise_mean") and len(source_groups) == 2:
        first = wells_for_group(matrix, source_groups[0])
        second = wells_for_group(matrix, source_groups[1])
        return [(a + b) / 2 for a, b in zip(first, second)]
    return combine_groups(matrix, source_groups)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def t_cdf(t_value: float, degrees_freedom: float) -> float:
    if t_value == 0:
        return 0.5
    x_value = degrees_freedom / (degrees_freedom + t_value * t_value)
    ibeta = mp.betainc(degrees_freedom / 2, 0.5, 0, x_value, regularized=True)
    if t_value > 0:
        return float(1 - 0.5 * ibeta)
    return float(0.5 * ibeta)


def welch_p_value(values_a: list[float], values_b: list[float]) -> float:
    n_a = len(values_a)
    n_b = len(values_b)
    mean_a = mean(values_a)
    mean_b = mean(values_b)
    var_a = statistics.variance(values_a)
    var_b = statistics.variance(values_b)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 1.0
    t_value = (mean_a - mean_b) / se
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    degrees_freedom = numerator / denominator if denominator else 1.0
    return max(0.0, min(1.0, 2 * (1 - t_cdf(abs(t_value), degrees_freedom))))


def p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"
