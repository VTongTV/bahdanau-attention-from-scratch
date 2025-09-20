# paper notes — equations to modules

maps the equations in arXiv:1409.0473v7 to the code modules that
implement them. the paper source is the only ground truth.

## encoder (supp.tex, model architecture)

| equation | meaning | module |
|----------|---------|--------|
| h_i = (1 - z_i) h_{i-1} + z_i h~_i | forward gru state | `nmt/model/gru.py` |
| h~_i = tanh(W E x_i + U [r_i h_{i-1}]) | proposed state | `nmt/model/gru.py` |
| z_i = sigmoid(W_z E x_i + U_z h_{i-1}) | update gate | `nmt/model/gru.py` |
| r_i = sigmoid(W_r E x_i + U_r h_{i-1}) | reset gate | `nmt/model/gru.py` |
| h_i = [fwd_i ; bwd_i] | annotation concat | `nmt/model/encoder.py` |

the forward and backward rnns share the source embedding matrix E.
they never share their weight matrices.

## alignment and attention (main.tex eq 6-8)

| equation | meaning | module |
|----------|---------|--------|
| e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j) | alignment score | `nmt/model/alignment.py` |
| alpha_ij = softmax_j(e_ij) | attention weight | `nmt/model/attention.py` |
| c_i = sum_j alpha_ij h_j | context vector | `nmt/model/attention.py` |

u_a h_j does not depend on the decoder step. precompute it once per
encoder call and reuse it across steps.

## decoder (supp.tex)

| equation | meaning | module |
|----------|---------|--------|
| s_i = (1 - z_i) s_{i-1} + z_i s~_i | decoder state | `nmt/model/decoder.py` |
| s~_i = tanh(W E y + U [r_i s_{i-1}] + C c_i) | proposed state | `nmt/model/decoder.py` |
| z_i = sigmoid(W_z E y + U_z s_{i-1} + C_z c_i) | update gate | `nmt/model/decoder.py` |
| r_i = sigmoid(W_r E y + U_r s_{i-1} + C_r c_i) | reset gate | `nmt/model/decoder.py` |
| s_0 = tanh(W_s bwd_h_1) | initial state | `nmt/model/decoder.py` |

the decoder uses a distinct context vector c_i at every step. the
rnnencdec baseline fixes c_i to the last forward state.

## deep output (supp.tex)

| equation | meaning | module |
|----------|---------|--------|
| t~_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i | deep output pre-activation | `nmt/model/head.py` |
| t_i = max-pool pairs of t~_i | maxout hidden layer | `nmt/model/head.py` |
| p(y_i) ~ exp(y_i^T W_o t_i) | vocab projection | `nmt/model/head.py` |

the maxout layer pools adjacent pairs of the 2l pre-activations into
l units. the projection maps to the full shortlist.

## training (supp.tex, training procedure)

| recipe | value | module |
|--------|-------|--------|
| sgd + adadelta | rho 0.95, eps 1e-6 | `nmt/train/optimizer.py` |
| gradient clip | l2 norm cap 1.0 | `nmt/train/clip.py` |
| minibatch | 80 sentences | `nmt/data/bucket.py` |
| rebucket | 1600 pairs every 20 updates | `nmt/data/bucket.py` |
| init | orthogonal u, gaussian w, zero v_a/bias | `nmt/model/params.py` |

## decoding (main.tex, experiment settings)

| recipe | value | module |
|--------|-------|--------|
| beam search | width 10 | `nmt/decode/beam.py` |
| eos stop | stop at </s> | `nmt/decode/beam.py` |
| unk guard | never emit <unk> | `nmt/decode/unk.py` |