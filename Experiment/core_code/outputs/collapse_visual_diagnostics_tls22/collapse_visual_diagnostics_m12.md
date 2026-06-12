# Collapse Confusion and t-SNE Diagnostics

- Target period: `M-2022-12`
- Selected collapse pairs: `56->96, 48->14, 66->71, 69->105, 104->2, 163->46`

## Outputs

- Confusion CSV: `outputs/collapse_visual_diagnostics_tls22/selected_collapse_confusion_m12.csv`
- Confusion heatmap: `outputs/collapse_visual_diagnostics_tls22/selected_collapse_confusion_heatmap_m12.png`
- Pair summary CSV: `outputs/collapse_visual_diagnostics_tls22/selected_collapse_pair_summary_m12.csv`
- Global t-SNE: `outputs/collapse_visual_diagnostics_tls22/tsne_selected_collapse_absorbers_m12.png`
- Pair t-SNE: `outputs/collapse_visual_diagnostics_tls22/tsne_pair_56_to_96_m12.png`
- Pair t-SNE: `outputs/collapse_visual_diagnostics_tls22/tsne_pair_48_to_14_m12.png`
- Pair t-SNE: `outputs/collapse_visual_diagnostics_tls22/tsne_pair_66_to_71_m12.png`
- Pair t-SNE: `outputs/collapse_visual_diagnostics_tls22/tsne_pair_69_to_105_m12.png`
- Pair t-SNE: `outputs/collapse_visual_diagnostics_tls22/tsne_pair_104_to_2_m12.png`
- Pair t-SNE: `outputs/collapse_visual_diagnostics_tls22/tsne_pair_163_to_46_m12.png`

## How To Read

- The confusion heatmap shows where selected collapsed classes are predicted.
- The global t-SNE shows selected collapsed classes, their absorber classes, and optional stable anchors.
- Each pair t-SNE has two panels: true labels on the left and predicted labels on the right. If collapsed-class points overlap with absorber points and are mostly colored as absorber on the prediction panel, the failure is representation/decision collapse rather than a simple global shift.
