# IoTJ review draft with exploratory IoT stress test (2026-11-07)

This draft starts from the latest JNCA submission source at
`../../../traffic-drift2/Publication/jnca_submission_20261019/source/`, not from
the older `Publication/paper/main_v4.tex`.

The first revision added an IoT traffic-inference motivation in the abstract
and introduction, a short related-work bridge with three IoT references,
explicit dataset and external-validity boundaries, IEEE citation formatting,
and wide-float layout. The CARE method and original CESNET experimental
protocols, tables, figures, and reported values remain intact.

The current revision adds a separate 11-home IoT device-type stress test.
Development results were used to specify a frozen final-window evaluation;
the final predictions passed a complete preflight before device labels were
decoded. The final-window report is at
`/data/xjw/tma-care-data/unsw_iot_final_holdout_v1/evaluation/REPORT.txt` on
the server, with per-home and per-class CSVs beside it. The frozen protocol and
prediction/evaluation code are in `Experiment/unsw_home_temporal_v1/`.
CARE improves final low-recall-class mean F1, but its equal-home macro-F1 is
0.767 percentage points below static, while a fixed source-target interpolation
control is 0.852 points above static. This is a temporally held-out test on the
same examined archive, not an independent cross-site validation or a full CARE
workflow test. The IoT derived CSVs and prediction records are not in the older
Figshare S1 archive. Use `export_iot_supplement.py` on the server to verify and
package them as S3, then make that package reviewer-accessible before submission.

Author names, affiliations, contact details, the contribution statement, and
the PDF author metadata are intentionally omitted from this review draft.

Build from this directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The current IEEE double-column PDF is 19 pages. The private Figshare reviewer
link is inherited from the JNCA submission. See `SUBMISSION_STATUS.md` for the
remaining author, journal-status, and supplement steps. This is a review draft,
not a submission package.
