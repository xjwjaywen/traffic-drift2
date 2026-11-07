# IoT-J submission status — 2026-11-07

## Ready for author review

- `main.tex` and `main.pdf`: 19-page IEEE double-column review draft. The original CESNET results, the 11-home final-window table, and the mixed IoT outcomes are in the paper.
- `FINAL_HOLDOUT_PROTOCOL.md` and the associated experiment code are committed under `Experiment/unsw_home_temporal_v1/`.
- `COVER_LETTER_DRAFT.txt` is a submission letter draft signed by the previously designated corresponding author, Dongjie Liu.
- `export_iot_supplement.py` verified the frozen server records and packaged the IoT evaluation CSVs, per-row predictions, code, and metadata as S3. The server reports 16,728,571 bytes and SHA256 `f2e7bf501e24366c48bf3a72bc71f3a75bfbadf41f1fafe1572a0b5d6f558e0c`. It did not change or rerun the experiment. The raw public archive and large intermediate feature arrays stay outside S3.
- `repack_s1.py` produced a new S1 ZIP for IoT-J. The 1,612 scientific members were checked byte-for-byte against the original JNCA S1; only README, result index, and manifest changed. Its result index records the SHA256 of the current manuscript PDF. The two-page S2 PDF now has the same five-author byline as the manuscript.

## Files for the submission portal

| Role | File | SHA256 |
| --- | --- | --- |
| Manuscript PDF | `main.pdf` | `2c9f1f85dc954eafc0e3b79e07633c9dc7e62282f5e937b9ed4c6d9e2aede987` |
| Manuscript source | `submission_files/LaTeX_source_IoTJ.zip` | `37bb2f599dd58fe302145cca8d74a7fe599c9847a7551a2c70261f039ba3ce31` |
| Supplement S1 | `submission_files/S1_CARE_IoTJ.zip` | `a2d27c01c90f93785449cc7a8664b07a79bc312fb0bed1e2aa44b14155f4ea25` |
| Supplement S2 | `submission_files/S2_CARE_component_controls.pdf` | `fed563c6b1189f292d3e0a174d444944eee883bd077c8618ffaa1ad224edea68` |
| Supplement S3 (server) | `/data/xjw/tma-care-data/unsw_iot_final_holdout_v1/submission/S3_IoT_final_holdout.zip` | `f2e7bf501e24366c48bf3a72bc71f3a75bfbadf41f1fafe1572a0b5d6f558e0c` |

Paste `COVER_LETTER_DRAFT.txt` into the cover-letter field after all authors approve it.

## Required before submission

1. The user reports that JNCA-D-26-04646 has been withdrawn or rejected. Keep the decision or withdrawal confirmation in the submission records.
2. Have every author verify the author order and contact details now printed in `main.tex`. Zhiquan Liu's English spelling, Jinan University affiliation, and public email were checked online; the other four authors' details were carried from the previous submission. Supply each author's ORCID in IEEE Author Portal.
3. Transfer the exported S3 ZIP from the server and attach it alongside the updated S1 and S2 files in the IoT-J submission portal. The paper no longer points to the old JNCA Figshare review link. Verify the uploaded S3 checksum.
4. The S3 contents are now stated in the manuscript's Data and Code Availability section. Check that every supplementary citation resolves to the supplied file and that the uploaded S3 checksum matches the value in this status file.
5. Have all authors review the final manuscript and approve the author list, conflict-of-interest statement, AI-use statement, supplementary data, and OA choice before the corresponding author submits through IEEE Author Portal.

## Server export

After pulling the branch, run:

```sh
python -u Publication/iotj_minimal_20261103/export_iot_supplement.py
```

The program has completed on the server. The archive contains per-row predictions but no raw traffic; the public archive cited in the paper is required for independent rescoring. If the final manuscript changes again, rerun `repack_s1.py` against the rebuilt PDF to refresh the S1 result-index fingerprint.
