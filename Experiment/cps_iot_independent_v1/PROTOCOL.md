# Independent IoT development and held-out evaluation protocol

This protocol was fixed after inspecting only training labels and the published
file structure. The test device labels have not been read. The data provider's
train/test division is kept intact.

## Chronological development split

For each of the 12 homes, sort `train.csv` by `flowEndMilliseconds`, breaking
ties by original row number. Use the first 60% of rows as labeled source
training data, the next 20% as an adaptation query pool, and the last 20% as
development evaluation. Apply `floor(0.6*n)` and `floor(0.8*n)` boundaries.
The published `test_full.csv` is the final held-out evaluation set. No labels
from development evaluation or final test may enter query selection or model
training.

For each home and seed, select at most 1000 unique query-pool rows. Only those
labels may be used for repair. Use paired query identities when comparing
CARE, target-only and frozen random forest heads; report selection-policy
comparisons separately. Historical reference rows must come from the same
source portion, with the same count per class for all applicable methods.
Use source classes as the closed-set prediction vocabulary, set before repair.
Report query classes absent from the source and final test rows outside the
source vocabulary separately.

For the initial baseline port, fit the TMA pilot's signed-log standardized
preprocessor on source rows only. Train its 256/128 tabular MLP with source
seed 0 for 30 fixed epochs; do not select a checkpoint using development
labels. Use 500 uniform query anchors plus 500 BADGE selections from the
remaining pool. Select five source references per class. Fit target-only and
CARE heads for 500 steps with the pilot's optimizer and CARE loss. Fit the
frozen-feature random forest with 200 trees, minimum leaf size 2, balanced
subsample class weights and the same paid labels plus references. These are
ported controls rather than a reproduction of the manuscript's source CNN.

## Evaluation and decision rule

Develop the method using the 12 development sets only. Compare at least the
static source MLP, target-only linear head, original CARE head, and nonlinear
frozen-feature random forest under the same 1000-label budget. Use paired
repair seeds 0-4 and keep the source split fixed. Report per-home results and
means; homes, not repair seeds, are the independent units.

Primary metric: macro F1 of source-vocabulary classes present in the evaluated
period. Also report each class's support/F1, the number of source-vocabulary
classes with zero recall, and negative-flip rate among examples the static
source model classified correctly. Show the macro-F1/negative-flip tradeoff;
do not select a method from macro F1 alone.

Freeze the new method, hyperparameters, selection rule, seeds and analysis
script after development evaluation and before opening any `test_full.csv`
device labels or generating final test predictions. Run the final test once.
The data set was designed to study label shift; an improvement here shows
out-of-dataset robustness, not necessarily TMA-style conditional drift repair.
