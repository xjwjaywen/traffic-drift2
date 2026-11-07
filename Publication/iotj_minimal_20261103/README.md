# IoTJ review draft with exploratory IoT stress test (2026-11-07)

This draft starts from the latest JNCA submission source at
`../../../traffic-drift2/Publication/jnca_submission_20261019/source/`, not from
the older `Publication/paper/main_v4.tex`.

The first revision added an IoT traffic-inference motivation in the abstract
and introduction, a short related-work bridge with three IoT references,
explicit dataset and external-validity boundaries, IEEE citation formatting,
and wide-float layout. The CARE method and original CESNET experimental
protocols, tables, figures, and reported values remain intact.

The current revision adds a separate, exploratory 11-home IoT device-type
stress test based on the already completed server-side development evaluation.
It reports per-home macro-F1 and negative flips, 17 low-baseline-recall
home/device-type pairs, and class-level gains and harms. The source report is
`/data/xjw/tma-care-data/unsw_iotj_scope_review_v1/REPORT.txt` on the server;
its inputs are `unsw_home_temporal_development_v1/{summary,per_class}.csv` and
`unsw_home_temporal_interference_v1/by_home_class.csv`. The results are not an
independent final-window validation or a full CARE workflow test. The new IoT
derived CSVs and prediction records are not yet in the older Figshare S1
archive; add them to reviewer-accessible supplementary materials before
submission if this section is retained.

Author names, affiliations, contact details, the contribution statement, and
the PDF author metadata are intentionally omitted from this review draft.

Build from this directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The current IEEE double-column PDF is 19 pages. The private Figshare reviewer
link is inherited from the JNCA submission and should be checked before a new
submission. This is a review draft, not a submission package.
