# Per-Class Collapse Diagnosis

- Reference period: `W-2022-45`
- Final period: `W-2022-47`
- Collapse threshold: recall < `0.1`
- Min support: `50`
- Ever-collapsed classes: `1`
- Final collapsed classes: `1`

## Period Summary

| period | supported | collapsed | severe | collapsed fraction | median recall |
|---|---:|---:|---:|---:|---:|
| W-2022-45 | 102 | 0 | 0 | 0.000 | 0.8164 |
| W-2022-46 | 102 | 0 | 0 | 0.000 | 0.7952 |
| W-2022-47 | 102 | 1 | 0 | 0.010 | 0.8053 |

## Final Collapsed Classes

| class | first collapse | final recall | final F1 | absorber | absorber rate | pattern |
|---:|---|---:|---:|---:|---:|---|
| 71 | W-2022-47 | 0.0669 | 0.1182 | 98 | 0.3206 | gradual |

## Top Collapse-Pair Rows

| period | true | pred | rank | rate | count |
|---|---:|---:|---:|---:|---:|
| W-2022-45 | 71 | 98 | 1 | 0.3578 | 122 |
| W-2022-45 | 71 | 18 | 2 | 0.0968 | 33 |
| W-2022-45 | 71 | 91 | 3 | 0.0880 | 30 |
| W-2022-45 | 71 | 97 | 4 | 0.0674 | 23 |
| W-2022-45 | 71 | 77 | 5 | 0.0616 | 21 |
| W-2022-46 | 71 | 98 | 1 | 0.3613 | 43 |
| W-2022-46 | 71 | 18 | 2 | 0.0924 | 11 |
| W-2022-46 | 71 | 97 | 3 | 0.0672 | 8 |
| W-2022-46 | 71 | 67 | 4 | 0.0504 | 6 |
| W-2022-46 | 71 | 77 | 5 | 0.0504 | 6 |
| W-2022-47 | 71 | 98 | 1 | 0.3206 | 460 |
| W-2022-47 | 71 | 91 | 2 | 0.1568 | 225 |
| W-2022-47 | 71 | 18 | 3 | 0.1477 | 212 |
| W-2022-47 | 71 | 77 | 4 | 0.0669 | 96 |
| W-2022-47 | 71 | 67 | 5 | 0.0571 | 82 |
