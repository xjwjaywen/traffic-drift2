# Ten-page IoT-J manuscript

This version starts from the complete 19-page IoT-J manuscript through the separately preserved nine-page revision. It retains the same seven main sections, CARE method and algorithm, principal TLS and QUIC comparisons, per-class recovery, budget and replay analyses, detector evaluation, and the 11-home IoT final holdout. A new table brings the prespecified IoT class-level outcomes into the main paper. Secondary Fisher, TTA-detail, and detector-signal tables and three redundant figures were omitted; their archived records remain in the complete version and S1.

Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex`. The visually checked `main.pdf` has 10 IEEE two-column pages, including references. No unresolved references or overfull boxes were reported.

Checked artifacts:

- `main.pdf`: 402,235 bytes; SHA-256 `e79094fa0f6be3f7b5c220eb5c2f8b9ba492070f0169f6eefff81008fef38a7b`.
- `submission_files/S1_CARE_IoTJ_ten_page.zip`: 49,278,327 bytes; SHA-256 `2e067488c22cdee15a979b0ebcef4b28dffe68148cc82efa1f09625b13ecc64d`.
- `submission_files/LaTeX_source_IoTJ_ten_page.zip`: 92,274 bytes; SHA-256 `b84422302d2cfbdfe5e87504d9e31473a8d985513c2468ddfe2b4ee0f2fc28b9`.

S1 scientific payloads match the verified complete-version archive byte-for-byte. A local copy of S2 is in `submission_files/`; the server-generated S3 accompanies the IoT final-window results. Submission ZIP files are local artifacts and are ignored by Git. See `SUBMISSION_CHECK.md` for the current submission audit and author/portal items.
