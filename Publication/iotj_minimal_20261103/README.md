# Minimal IoTJ revision draft (2026-11-03)

This draft starts from the latest JNCA submission source at
`../../../traffic-drift2/Publication/jnca_submission_20261019/source/`, not from
the older `Publication/paper/main_v4.tex`.

Changes are limited to an IoT traffic-inference motivation in the abstract and
introduction, a short related-work bridge with three IoT references, explicit
dataset and external-validity boundaries, IEEE citation formatting, and
wide-float layout. The CARE method, dataset protocols, tables, figures, and
reported experimental values were retained. No IoT-device experimental result
is claimed.

Author names, affiliations, contact details, the contribution statement, and
the PDF author metadata are intentionally omitted from this review draft.

Build from this directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The current IEEE double-column PDF is 18 pages. The private Figshare reviewer
link is inherited from the JNCA submission and should be checked before a new
submission. This is a review draft, not a submission package.
