# Service-vocabulary drift: candidate research direction

Status: research hypothesis, **not** an implemented or validated method.
This is a separate device-associated identification problem, not a claimed
improvement to the current 28-feature, post-NAT, per-flow UNSW experiment.

## Why the current per-flow repair line stops here

The already examined UNSW development window does not support a stable
home07 class-6-to-class-9 correction from the source logit difference. On
five overlapping 1,000-label query draws, its AP is below the within-group
positive prevalence and its AUC ranges from 0.21 to 0.44. Development AUC is
0.559. Hard-count prior correction rescues class 9 but removes all class-6
and class-10 predictions. Soft-count correction preserves those two classes yet barely
recovers class 9; its whole-model development macro F1 remains below the
static source model. Random forest and nearest-anchor controls improve class
9 F1 modestly while substantially reducing whole-model F1. These are
exploratory observations on a window seen during design. They do not prove
that the raw traffic features are intrinsically non-identifying.

The published [2023 UNSW flow study][flow] describes 28 activity and broad
protocol indicators as model features. The current experiment deliberately
excludes identifiers and ports. Its labels identify devices using MACs in the
released archive, so grouping records by the `device` field would give the
model an identity key and would change the inference problem.

## Concrete gap in adjacent work

The [IFIP Networking 2026 backend-service fingerprint paper][service]
represents each device by a persistent service vocabulary and several
service-usage fingerprints. It evaluates approximately 50 million flow
records from 60 device types across two multi-year segments. Its method
assumes that flows can already be associated with an individual device using
a pre-NAT network identifier and aggregates them over time. These conditions
must be explicit in any comparison with our per-flow experiment.

The paper already filters transient ports and services absent from every
reference vocabulary. Nevertheless, it reports test-period service changes
(including TCP/853 use by Apple devices and TCP/443 use by a camera) that
cause its vocabulary confinement to drop the true class. It then excludes
those affected devices from subsequent evaluation. This is a potential
failure mode to study, rather than an untouched research area. For 8-day
windows in its 2022 segment, the reported miss-out rates are 100%, 57.63%,
55.93%, and 48.59% for the four named affected devices.
The informative event is a service new to the *true type* but already in
another type's reference vocabulary: a globally new service would be
ignored by the published confinement rule.

Plain service fingerprints, multiple activity modes, or semantic matching
of unseen services are not novelty claims. In particular, [2026 semantic
MUD matching][mud] already considers unseen behavior primitives and partial
runtime observation; it is a relevant baseline and related work.

## Falsifiable problem and possible method

**Question:** Given a device-associated stream whose service set changes
after training, can a classifier retain the correct type among candidates
without admitting so many types that precision collapses, using no future
labels and a fixed, small label budget before the evaluation window?

One possible formulation is *uncertainty-aware confinement*: estimate, from
training blocks, how stable each candidate type's observed-service pattern
is, then score a test window allowing a bounded amount of novel-service
contamination. A small paid sample could distinguish persistent service
additions from a different device type; a candidate would be retained only
when its adjusted evidence passes a threshold fixed before test. The final
matching stage must still discriminate among surviving types. This is only
a design sketch: robust matching, soft set similarity, and [online prediction
sets][online] already exist. In particular, online conformal methods can adapt to
drift **when feedback labels arrive**; their coverage claims cannot simply
be transferred to unlabeled device windows. A literature audit and ordinary
baselines are required before claiming novelty or a guarantee.

The immediate necessary-condition audit is more important than writing that
method: enumerate services absent from the true type's Jan--Jun vocabulary
but present in another type's vocabulary during July--September in the public
[IoT-ServTrace dataset][data]. Separate persistent additions from one-off
ports, count affected device-time windows, quantify how often the published
confinement rule eliminates the correct class, and verify enough
events remain for a chronological development/held-out evaluation. For the
2022 segment, use the paper's Jan--Jun training interval, reserve Jul--Sep
for method development, and leave Oct--Dec unopened until the rule is fixed.
This is stricter than the paper's Jul--Dec test interval and needs enough
July--September events to support the extra split. The published paper's
aggregate July--December findings motivated this question, so this is a
method-development holdout, not an untouched research question. A negative
audit ends this direction. The dataset is distributed as Git LFS
JSON files per device type; its record fields are timestamp, transport
protocol, and port number. Its per-device file organization must not be
mistaken for a method that discovers device associations at inference.

If the audit passes, compare against the paper's hard confinement and
fingerprint matching, no confinement, simple soft set similarity, and
semantic matching where applicable. Report candidate-set recall/size and
type-level macro F1, with devices or time blocks as independent units; do
not count overlapping windows or seeds as independent replicates. Fix all
rules before opening the held-out period.

[flow]: https://www2.ee.unsw.edu.au/~hhabibi/pubs/jrnl/23conceptDriftJIoT.pdf
[service]: https://www2.ee.unsw.edu.au/~hhabibi/pubs/conf/26ifipNetworking.pdf
[mud]: https://arxiv.org/abs/2606.12793
[data]: https://github.com/shay-fi/IoT-ServTrace
[online]: https://jmlr.org/beta/papers/v25/22-1218.html
