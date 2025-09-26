# architecture

how the paper equations map to the code modules. the model is rnnsearch of
bahdanau, cho & bengio 2015 (arXiv 1409.0473).

## data flow

```
src_ids ──► Encoder ──► annotations (batch, src_len, 2n)
                │
                └──► Attention.cache (u_a h_j precomputed)
                     │
decoder state s_i ──► Attention ──► context c_i, weights alpha
                     │
tgt_ids ──► decoder.embedding ──► DecoderCell ──► state s_{i+1}
                     │
state + emb + ctx ──► DeepHead ──► logits (batch, vocab)
```

## module map

| paper piece | module | file |
| --- | --- | --- |
| gated hidden unit (eq 7-10, context-free) | `ContextFreeCell` | `nmt/model/gru.py` |
| gated hidden unit with context | `DecoderCell`, `GRUCell` | `nmt/model/decoder.py` |
| bidirectional encoder, shared source embedding | `Encoder` | `nmt/model/encoder.py` |
| annotations h_j = [fwd_j; bwd_j] | `Encoder.forward` | `nmt/model/encoder.py` |
| alignment score e_ij (eq 6) | `Alignment.score` | `nmt/model/alignment.py` |
| cached source encodings u_a h_j | `Alignment.cache` | `nmt/model/alignment.py` |
| softmax weights + context c_i (eq 5) | `Attention.forward` | `nmt/model/attention.py` |
| initial state s_0 = tanh(W_s back_h_1) | `Decoder.initial_state` | `nmt/model/decoder.py` |
| deep output maxout (eq 4) | `MaxoutHead`, `DeepHead` | `nmt/model/head.py` |
| rnnsearch assembly | `RNNsearch` | `nmt/model/rnnsearch.py` |
| no-attention baseline | `RNNencdec` | `nmt/model/rnnencdec.py` |

## sizes

paper values live in `nmt/config.py`. each experiment can override them
(the cpu runs use smaller dims; the constants stay at paper values).

| piece | paper size |
| --- | --- |
| hidden n | 1000 |
| embedding m | 620 |
| alignment hidden n' | 1000 |
| maxout l | 500 |
| vocab shortlist | 30000 |

## init scheme

`nmt/model/params.py` holds the primitives. `init_parameters` on each
module applies the appendix b.1 scheme: orthogonal recurrent matrices,
gaussian 0.01 weights, gaussian 0.001 alignment matrices, zero biases
and v_a.

## masking

padded target positions have a mask from the collate step. the masked
nll ignores them. padded source positions get -inf alignment scores, so
their attention weight is zero.