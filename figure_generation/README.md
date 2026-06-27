# Figure Generation Bundle

This directory now keeps only the scripts and data that map to figures actually used in the current thesis workflow, based on the cited figure-editing threads plus `/Users/zorn/Library/Mobile Documents/com~apple~CloudDocs/硕士课题/fig/唐健 20260506 论文图.pptx`.

All figure scripts now default to a single formal output directory:

- `figures_final/`

## Layout

- `scripts/`
  - Centralized plotting/redrawing scripts.
  - `_fg_paths.py` is the shared path config for all consolidated scripts.
- `data/`
  - flattened by usage instead of mirroring the source project tree.
  - `analysis_data/`, `analysis_results/`, `model_results/`, `ablations/`, `predictions/`, `seq_embedding/`, `docking/`
  - `integrated_gencode_uniprot_detailed.json`
  - `depmap_tables/`
  - `experiments/`: copied CCK-8 source workbooks used by the plotting scripts.

## Curated Script Set

- `make_attention_panel.py`
  - `06_atp6v1b2_attention_functional_sites_npj.png`
- `visualize_attention_dense.py`
  - thesis Chapter 3 dense attention figures:
    - `ATP6V1A_dense.png`
    - `ATP6V1B2_dense.png`
    - `H2BC11_dense.png`
    - `PLBD1_dense.png`
  - minimal runtime dependencies kept in-bundle:
    - `scripts/module/`
    - `data/model_results/PIC_human/`
    - `data/seq_embedding/{4914,4920,23413,99999}.pt`
- `recreate_summary_figures.py`
  - `08a_training_loss_npj.png`
  - `08b_validation_metrics_npj.png`
  - `08c_test_roc_npj.png`
  - `08d_test_pr_npj.png`
  - `11a_attention_heads_val_auprc_npj.png`
  - `11b_attention_heads_val_auroc_npj.png`
  - `11c_attention_heads_test_pr_npj.png`
  - `11d_attention_heads_test_roc_npj.png`
- `redraw_cck8_fig10_panels.py`
  - `figures/10e_cck8_293t_npj.png`
  - `figures/10f_cck8_hl60_npj.png`
  - helper modules kept for this script only:
    - `plot_cck8_file1_figure.py`
    - `npj_redraw_figures.py`
- `make_model_comparison.py`
  - `12a` to `12e` single-panel PNGs
  - 3 legend-only PNGs for figure 12
- `visualize_deep_immune_analysis.py`
  - `13a` to `13d`
  - `14a` to `14c`
  - `15a` to `15c`
  - `16a` to `16c`
- `redraw_deep_aa_composition_review.py`
  - `figures/visualizations_deep__E_aa_composition_preview.png`
- `make_docking_panel.py`
  - `17a_docking_affinity_bars_npj.png`
  - `17b_docking_affinity_vs_distance_npj.png`
- `make_hpa_panel.py`
  - `18_hpa_expression_heatmap_npj.png`
- `make_depmap_panel.py`
  - `20a` to `20d`

## Outputs

- `figures_final/`
  - all final renamed figure assets from the curated script set
  - naming convention uses `fig_...`

## Notes

- Data under `data/` is intentionally trimmed to the files required by the curated scripts above.
- Redundant script files and old generation paths were removed or trimmed so the kept scripts only emit the figure assets that are currently in use.
- Temporary test output directories were removed after consolidating the renamed final image set into `figures_final/`.
- `make_attention_panel.py` can run directly from the cached attention array already copied into `data/analysis_results/`.
  - Recomputing attention from the raw model would still require the full pretrained/model artifacts, which were not copied here.
