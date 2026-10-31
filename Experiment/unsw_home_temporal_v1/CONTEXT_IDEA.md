# Flow-context information test and conditional repair hypothesis

Status: exploratory hypothesis. Development labels for the 11 homes were
already inspected; no final-window data have been used for this direction.
The home07 class pair below was selected after inspecting development results.

## Observation

On home07, a frozen RF recovers some class-9 flows but greatly worsens the
whole model. Hard class-prior correction shifts too many originally correct
class-6 and class-10 flows to class 9. Source-logit ranking within
source-predicted class 6 is poor on the existing paid labels. Thus a larger
global correction is not justified. These controls all make a decision from
one flow's 28 features; they do not establish whether the **unlabeled flows
around that flow** provide additional evidence.

## Necessary-condition probe

`context_information_probe.py` computes, for each selected flow, the count
and mean traffic statistics of *other* flows in the same five-minute home
window. It never uses MAC or a device ID as an input. It trains three fixed
binary controls on the same labeled source rows: flow only, context only,
and flow plus context. It evaluates the already paid query rows whose true
labels are class 6 or 9. This true-class restriction is an oracle subset,
used only to test separability, so its AP/AUC is not a deployable multiclass
score. Five paid seeds overlap and are not independent experiments.

The predeclared criterion to continue this direction is that flow plus
context improves paid-query AP and AUC over flow only in at least four of
five seeds, with a material effect rather than a numerical tie. This is a
screen, not a statistical significance claim. If it fails, stop: a more
complex temporal model has no demonstrated extra information to exploit.

## Method to investigate only if the screen passes

The unit would become a *bag of flows in a short time window*, with the
device assignment latent. A source model supplies per-flow class costs.
An auxiliary model estimates the plausible class composition of the bag
from its unlabeled context and the limited paid labels. A constrained
assignment then couples the two: it can move a class-6 prediction to class
9 only when both the flow evidence and bag-level evidence support it. The
composition is an interval, not a hard quota, so it cannot force every
class-6 flow to class 9 as the prior correction did. One mathematical
candidate is unbalanced entropic optimal transport with per-class interval
constraints and an explicit cost for changing a high-margin source
prediction. An alternative is a latent point-process model with the same
source-preservation constraint. These are research candidates, not methods
validated by the present probe.

This would differ from the existing class-prior correction by using local,
unlabeled co-traffic and by modeling uncertainty about class counts. It
would differ from the existing RF/nearest-anchor controls by coupling
decisions across contemporaneous flows. Temporal IoT identification,
label-shift adjustment, and optimal-transport adaptation already exist;
the combination needs a much fuller prior-art review and strong ablations
before any novelty claim.

Important controls if the information screen passes: flow-only model of
equal capacity; time-of-day-only context; context shuffled among windows
within a day; standard prior correction; unconstrained local quota; source
model; and the original CARE model. Measure whole-model macro F1, class-9
F1, class-6/8/10 F1, and negative flips. Fix the method on the inspected
development period before opening any final window.

Relevant prior work:

- 2025 IoT label-shift study: https://www2.ee.unsw.edu.au/~hhabibi/pubs/conf/25cpsiotsec-labelshift.pdf
- 2018/2020 optimal-transport shift adaptation: https://arxiv.org/abs/1803.04899 ; https://arxiv.org/abs/2006.08161
- 2023 longitudinal IoT study: https://www2.ee.unsw.edu.au/~hhabibi/pubs/jrnl/23conceptDriftJIoT.pdf
