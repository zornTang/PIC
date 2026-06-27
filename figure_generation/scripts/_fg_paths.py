from __future__ import annotations

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FG_ROOT = SCRIPT_DIR.parent
CODE_ROOT = FG_ROOT.parent

DATA_ROOT = FG_ROOT / "data"
OUTPUT_ROOT = FG_ROOT / "figures_final"
EXPERIMENTS_ROOT = DATA_ROOT / "experiments"

FIGURES_DIR = Path(os.environ["FG_FIGURES_DIR"]) if os.environ.get("FG_FIGURES_DIR") else OUTPUT_ROOT
FIGURES_NPJ_DIR = Path(os.environ["FG_FIGURES_NPJ_DIR"]) if os.environ.get("FG_FIGURES_NPJ_DIR") else OUTPUT_ROOT

PIC2_ANALYSIS_DATA = DATA_ROOT / "analysis_data"
PIC2_ANALYSIS_RESULTS = DATA_ROOT / "analysis_results"
PIC2_MODEL_RESULTS_ROOT = DATA_ROOT / "model_results"
PIC2_MODEL_DIR = PIC2_MODEL_RESULTS_ROOT / "PIC_human"
PIC2_ABLATIONS_DIR = DATA_ROOT / "ablations"
PIC2_PREDICTIONS_DIR = DATA_ROOT / "predictions"
PIC2_DOCKING_DIR = DATA_ROOT / "docking"
PIC2_SEQ_EMBED_DIR = DATA_ROOT / "seq_embedding"
PIC2_INTEGRATED_JSON = DATA_ROOT / "integrated_gencode_uniprot_detailed.json"

DEP_MAP_TABLES = DATA_ROOT / "depmap_tables"

CCK8_293T_XLSX = EXPERIMENTS_ROOT / "cck8_293t_20260327.xlsx"
CCK8_HL60_XLSX = EXPERIMENTS_ROOT / "cck8_hl60_20260328.xlsx"
