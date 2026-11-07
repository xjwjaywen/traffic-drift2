# Balanced IoT-J manuscript

This nine-page version is **edited from the complete 19-page IoT-J manuscript** in `../iotj_minimal_20261103/`. The eight-page compact draft was used only as a comparison. The complete IoT-J and pre-IoT manuscripts remain intact; see `../REVISION_VERSIONS.md`.

The revision preserves the CARE monitoring and repair definitions, algorithm, principal TLS/QUIC results, ablations, per-class recovery, and the 11-home IoT final holdout. It removes repeated explanations and secondary figures/tables. The S1 archive retains full archived scientific records. The separate S2 and server-generated S3 supplements still apply.

Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex`. The checked PDF is `main.pdf`. `repack_s1.py` creates an S1 archive tied to that PDF's SHA-256 from the verified full-version S1 without changing any scientific payload files.

Checked artifacts:

- `main.pdf`: 9 pages; SHA-256 `5c89688a6a64c66aa3355b5ac4fc7867861d59dbcbe97a6f08699fc1c1e43524`.
- `submission_files/S1_CARE_IoTJ_balanced.zip`: 49,278,303 bytes; SHA-256 `bdf38fcc012f23775a2110d608f7528d7fe090abc860c5d687ed328dba407701`.
- `submission_files/LaTeX_source_IoTJ_balanced.zip`: 76,782 bytes; SHA-256 `2a93241dc438ebe2890411a9f428807722713efff56bd1683a176091874dc7ee`.

The ZIP files are local submission artifacts and are ignored by Git. S2 remains in `../iotj_minimal_20261103/submission_files/`; S3 remains on the server.
