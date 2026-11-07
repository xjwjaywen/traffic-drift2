# IoT-J submission status — 2026-11-07

## Ready for author review

- `main.tex` and `main.pdf`: 19-page IEEE double-column review draft. The original CESNET results, the 11-home final-window table, and the mixed IoT outcomes are in the paper.
- `FINAL_HOLDOUT_PROTOCOL.md` and the associated experiment code are committed under `Experiment/unsw_home_temporal_v1/`.
- `COVER_LETTER_DRAFT.txt` is a submission letter draft. Its signature remains to be filled after author confirmation.
- `export_iot_supplement.py` verifies the frozen server records and packages the IoT evaluation CSVs, per-row predictions, code, and metadata as S3. It does not change or rerun the experiment. The raw public archive and large intermediate feature arrays stay outside S3.

## Required before submission

1. Confirm that JNCA-D-26-04646 is no longer under consideration. The current manuscript builds on that submission; do not submit it to IoT-J concurrently.
2. Confirm the complete author order, 刘志全老师's official English name, affiliations, emails, corresponding author, and ORCID for every author. Then add the author block and PDF metadata to `main.tex`, update the cover letter and supplements, and rebuild the PDF. The present PDF deliberately has an empty author field.
3. Run the S3 export on the server and make the resulting ZIP accessible to reviewers. The current Figshare S1 is the original JNCA archive; it does **not** contain the new IoT results. Update S1's journal/author metadata or provide a clearly labeled IoT-J replacement before submission. Confirm the reviewer link works for an unauthenticated reviewer.
4. Add the S3 access and contents to the manuscript's Data Availability section, then rebuild and inspect the final PDF. Check that every supplementary citation resolves to the supplied file.
5. Have all authors review the final manuscript and approve the author list, conflict-of-interest statement, AI-use statement, supplementary data, and OA choice before the corresponding author submits through IEEE Author Portal.

## Server export

After pulling the branch, run:

```sh
python -u Publication/iotj_minimal_20261103/export_iot_supplement.py
```

The program prints the S3 ZIP path, size, and SHA256. The ZIP is created under `/data/xjw/tma-care-data/unsw_iot_final_holdout_v1/submission/` and will not overwrite an existing package. Upload it as a supplement or to a stable private reviewer link. The archive contains per-row predictions but no raw traffic; the public archive cited in the paper is required for independent rescoring.
