# Related-Work Table for Advisor Discussion

| Method / paper | Type | Target labels? | Continual / online? | Main assumption | Relevance to TLS22 collapse |
|---|---|---:|---:|---|---|
| Tent | Test-time adaptation | No | Online batch stream | Entropy minimization improves target predictions | Weak for collapsed classes because wrong absorber predictions can be confident |
| EATA | Test-time adaptation | No | Online stream | Filter unreliable/redundant samples and regularize against forgetting | More stable than Tent but still depends on usable pseudo-label/entropy signal |
| CoTTA | Continual test-time adaptation | No | Yes | Teacher-student pseudo-labels, augmentation averaging, stochastic restoration | Relevant to long streams; cannot easily recover classes that are already absorbed |
| SAR | Stable test-time adaptation | No | Online stream | Reliable entropy + sharpness-aware updates; GN/LN can be more stable than BN | Useful baseline; handles adaptation instability more than class-conditional concept drift |
| NOTE | Continual test-time adaptation | No | Yes | Temporally correlated streams need balanced memory and instance-aware BN | Relevant to traffic streams; still weak if collapsed classes are never predicted |
| AdaBN | Domain/statistics adaptation | No | Usually offline or per-domain | Target shift can be handled by replacing BN running statistics | Good sanity check for covariate shift; our results show limited collapsed-class gains |
| FDAN | Encrypted-traffic domain adaptation | No target relabeling | Offline source-target DA | Learn domain-invariant features for drifted encrypted traffic | Closest encrypted-traffic DA baseline; less strict than pure TTA because target-domain data is available for adaptation |
| FG-Net | Drift-robust traffic representation | Not a pure TTA setting | Offline training/evaluation | Flow-level relations are more stable than packet-level fingerprints | Supports the idea that robust features may require representation redesign |
| Encrypted traffic drift empirical study | Empirical drift analysis | N/A | Time-aware evaluation | Feature distributions and classifier performance drift over time | Supports motivation for temporal evaluation and collapse diagnosis |
| This project diagnosis | Drift/collapse analysis | N/A | Time-aware TLS22 stream | Long-term drift creates abrupt/gradual class-conditional collapse | Shows why generic TTA and normalization adaptation are insufficient |

Short takeaway:

Generic TTA methods mostly target global target-distribution shift or adaptation instability. AdaBN targets normalization-statistics shift. Encrypted traffic temporal drift can include class-conditional and event-driven concept drift, where some classes are confidently absorbed by other classes. This explains why the observed improvements from generic TTA and AdaBN are small on TLS22 collapsed classes.

