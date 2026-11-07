# IoT-J submission status — 2026-11-07

## Ready for author review

- `main.tex` and `main.pdf`: 19-page IEEE double-column review draft. The original CESNET results, the 11-home final-window table, and the mixed IoT outcomes are in the paper.
- `FINAL_HOLDOUT_PROTOCOL.md` and the associated experiment code are committed under `Experiment/unsw_home_temporal_v1/`.
- `COVER_LETTER_DRAFT.txt` is a submission letter draft signed by the previously designated corresponding author, Dongjie Liu.
- `export_iot_supplement.py` verified the frozen server records and packaged the IoT evaluation CSVs, per-row predictions, code, and metadata as S3. The server reports 16,728,571 bytes and SHA256 `f2e7bf501e24366c48bf3a72bc71f3a75bfbadf41f1fafe1572a0b5d6f558e0c`. It did not change or rerun the experiment. The raw public archive and large intermediate feature arrays stay outside S3.
- `repack_s1.py` produced a new S1 ZIP for IoT-J. The 1,612 scientific members were checked byte-for-byte against the original JNCA S1; only README, result index, and manifest changed. Its result index records the SHA256 of the current manuscript PDF. The two-page S2 PDF now has the same five-author byline as the manuscript.

## Submission route checked against IEEE guidance

The preceding JNCA submission used a private Figshare link for S1 and uploaded
`07_Supplementary_Material_S1_access.txt`; its 47 MB S1 ZIP was not directly
uploaded to that portal. IEEE Author Center instead advises that supplementary
files be labeled and uploaded separately during submission. This manuscript
therefore describes the current S1 ZIP as a separate archive. The old private
Figshare link is a record of the previous submission, is not cited in the
current manuscript, and does not contain the new IoT results. If the IoT-J portal rejects the
archive due to file size or type, ask the journal office how to submit S1;
do not silently omit a supplement described as submitted in the manuscript.

Upload the manuscript PDF and provide S1, S2, and S3 as separately labeled
supplements because the manuscript cites them. The cover letter is prepared
for the portal, though the public IoT-J guidelines do not label it mandatory.
The source ZIP is
available if the portal requests source. S3 can be uploaded from the server
desktop or transferred by the user's usual remote-file workflow; this Mac
cannot resolve `ub01-desktop` and does not need SSH access to finish the paper.

## Prepared files

| Role | File | SHA256 |
| --- | --- | --- |
| Manuscript PDF | `main.pdf` | `4b0e41f522a34e0c5d29a0f529a15e81cb220b1ad7d64d5f5b8ab686d06e01f3` |
| Manuscript source | `submission_files/LaTeX_source_IoTJ.zip` | `e6e0010eaadc1a653bab0a4a0280c2d8a58037d8e5aeb95348382a9406fbd205` |
| Supplement S1 | `submission_files/S1_CARE_IoTJ.zip` | `e43fdb707190d6bd2556b9029a7cc5862e6821a6f3383160194d6c6e10aad50b` |
| Supplement S2 | `submission_files/S2_CARE_component_controls.pdf` | `fed563c6b1189f292d3e0a174d444944eee883bd077c8618ffaa1ad224edea68` |
| Supplement S3 (server) | `/data/xjw/tma-care-data/unsw_iot_final_holdout_v1/submission/S3_IoT_final_holdout.zip` | `f2e7bf501e24366c48bf3a72bc71f3a75bfbadf41f1fafe1572a0b5d6f558e0c` |

Paste `COVER_LETTER_DRAFT.txt` into the cover-letter field after all authors approve it.

## Official-rule checks (7 November 2026)

- The PDF uses IEEE double-column format, has 19 pages, and has a one-paragraph 184-word abstract, within the journal's 150–250-word rule. It identifies all five authors because IoT-J uses single-anonymous peer review.
- The journal requires each author's ORCID and subject classification keywords at submission. Suggested reviewers are encouraged but optional; avoid conflicts of interest.
- IEEE's AI-generated-content policy calls for disclosure in the Acknowledgment with the system, affected sections, and level of use. The manuscript's disclosure has been moved and made section-specific.
- IEEE Author Center advises that supplementary materials be labeled and uploaded separately. S1, S2, and S3 are cited in this manuscript. Graphical abstracts are accepted but not required.
- The journal has mandatory $175 charges per published page beyond eight. With 19 final pages, the illustrative charge would be $1,925. IEEE's 2026 hybrid-journal OA APC is $2,800, separate from overlength and any applicable tax or institutional arrangement. The journal's author-guidelines page still displays an older $2,695 OA figure; the 2026 IEEE pricing page controls the current estimate. Submission acknowledges the mandatory page-charge policy.
- Copyright form and final production files are post-acceptance tasks, not separate files required for the initial peer-review submission.

## Required before submission

1. The user reports that JNCA-D-26-04646 has been withdrawn or rejected. Keep the decision or withdrawal confirmation in the submission records.
2. Have every author verify the author order and contact details now printed in `main.tex`. Zhiquan Liu's English spelling, Jinan University affiliation, and public email were checked online; the other four authors' details were carried from the previous submission. Supply each author's ORCID in IEEE Author Portal.
3. Upload the local S1 ZIP and S2 PDF and provide the S3 ZIP from the server for peer review. Verify the S3 checksum after transfer.
4. The S3 contents are now stated in the manuscript's Data and Code Availability section. Check that every supplementary citation resolves to the supplied file and that the uploaded S3 checksum matches the value in this status file.
5. Obtain and enter each author's ORCID. Have all authors review the final manuscript and approve the author list, funding and conflict-of-interest declarations, AI-use statement, supplementary data, and OA choice before the corresponding author submits through IEEE Author Portal. Confirm who pays OA and mandatory overlength charges.

## Server export

After pulling the branch, run:

```sh
python -u Publication/iotj_minimal_20261103/export_iot_supplement.py
```

The program has completed on the server. The archive contains per-row predictions but no raw traffic; the public archive cited in the paper is required for independent rescoring. If the final manuscript changes again, rerun `repack_s1.py` against the rebuilt PDF to refresh the S1 result-index fingerprint.
