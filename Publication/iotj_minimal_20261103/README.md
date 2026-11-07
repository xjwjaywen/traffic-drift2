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
Figshare S1 archive. They are packaged as S3 at
`/data/xjw/tma-care-data/unsw_iot_final_holdout_v1/submission/S3_IoT_final_holdout.zip`
on the server (16,728,571 bytes; SHA256
`f2e7bf501e24366c48bf3a72bc71f3a75bfbadf41f1fafe1572a0b5d6f558e0c`).
Attach this exact archive to the IoT-J submission.

The author line follows the teacher's requested order, with Zhiquan Liu
penultimate and Dongjie Liu corresponding. The first, second, third, and last
authors' contact details come from the previous submission. Zhiquan Liu's
affiliation and email were checked against Jinan University's faculty page
(`https://faculty.jnu.edu.cn/wlkjaqxy/lzq3/list.htm`); IEEE VTS uses the
English spelling `Zhiquan Liu` (`https://vtsociety.org/contact/zhiquan-liu`).
All authors should review this information and supply ORCIDs in the portal.

Build from this directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The current IEEE double-column PDF is 19 pages. IEEE Author Center advises
uploading cited supplementary files separately, so the relabeled S1 ZIP and
updated S2 PDF in `submission_files/` are prepared for the portal. The S3 archive
remains on the server and must be provided for review because the paper cites
it. The manuscript-source ZIP is available if the portal requests it. See
`SUBMISSION_STATUS.md` for the upload map and author checks.
