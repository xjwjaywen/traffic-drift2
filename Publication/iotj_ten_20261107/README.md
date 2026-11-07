# Ten-page IoT-J manuscript

This version starts from the complete 19-page IoT-J manuscript through the separately preserved nine-page revision. It retains the same seven main sections, CARE method and algorithm, principal TLS and QUIC comparisons, per-class recovery, budget and replay analyses, detector evaluation, and the 11-home IoT final holdout. A new table brings the prespecified IoT class-level outcomes into the main paper. Secondary Fisher, TTA-detail, and detector-signal tables and three redundant figures were omitted; their archived records remain in the complete version and S1.

Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex`. The visually checked `main.pdf` has 10 IEEE two-column pages, including references. No unresolved references or overfull boxes were reported.

Checked artifacts:

- `main.pdf`: 401,898 bytes; SHA-256 `517a46019004f40b9b87c3c4b1c1d84aefb6945a69d6db0c578a187bc68f29cd`.
- `submission_files/S1_CARE_IoTJ_ten_page.zip`: 49,278,330 bytes; SHA-256 `34af8da2761bf77ac095353a3d025cc360bf07de2162bbebe5f9930feb3882e3`.
- `submission_files/LaTeX_source_IoTJ_ten_page.zip`: 92,192 bytes; SHA-256 `d4f80a2ac8bcc24ead69231a397e8cb8ee7bcdede7bdeb2c0fd067dc0a2006e1`.

S1 scientific payloads match the verified complete-version archive byte-for-byte. S2 remains in `../iotj_minimal_20261103/submission_files/`; the server-generated S3 accompanies the IoT final-window results. Submission ZIP files are local artifacts and are ignored by Git.
