# methodology

how we run, measure, and what differs from the paper.

## deviations

- **v_a init 0.05 instead of zero (appendix b.1).** with v_a exactly zero
  the alignment gradient vanishes and the model trains as a language model.
  at our small scale the paper recipe never wakes the attention. a small
  constant start keeps the alignment formula and training identical.
  everything else follows appendix b.1 exactly.
