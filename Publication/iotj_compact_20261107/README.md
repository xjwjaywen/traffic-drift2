# Compact IoT-J manuscript

`main.tex` and `main.pdf` are an 8-page teacher-review draft condensed from `../iotj_minimal_20261103/`. The 19-page IoT version and the pre-IoT manuscript are preserved in their own directories; see `../REVISION_VERSIONS.md`.

Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex`, then copy `build/main.pdf` to `main.pdf` after visual review. The draft keeps the original CARE method and quantitative claims, including the mixed 11-home final-window result. Expanded methods and records remain in the full version and S1--S3.

The compact S1 archive was prepared at `submission_files/S1_CARE_IoTJ_compact.zip` (49,278,378 bytes; SHA-256 `257caa4e82683de950adb10144841c0064b61c8a3d4702cef3112047bd32a6c8`). It is ignored by Git because of its size. `repack_s1.py` verifies its manifest and checks that every scientific payload member matches the full-version S1. Its index records the compact PDF hash (`a111b1c2d8830e93bbcf0cde367645c402cc96387d98a5af0df4f0d645fa43fa`) and maps compact table/figure labels to the original result entries. The separate S2 PDF remains in `../iotj_minimal_20261103/submission_files/`; S3 remains on the server and should be supplied alongside whichever IoT version is ultimately submitted.
