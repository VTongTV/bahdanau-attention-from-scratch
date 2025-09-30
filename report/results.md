# results

tables and numbers from our runs. this file mirrors the paper tables.

## bleu by model and max length (paper table 1)

| method | all sentences | no unk subset |
| --- | --- | --- |
| rnnencdec-30 | | |
| rnnsearch-30 | | |
| rnnencdec-50 | | |
| rnnsearch-50 | | |

## train and dev nll (paper table 2)

| model | updates | epochs | train nll | dev nll |
| --- | --- | --- | --- | --- |
| rnnencdec-30 | | | | |
| rnnsearch-30 | 3920 | 9 | 0.0016 | 0.0013 |
| rnnencdec-50 | | | | |
| rnnsearch-50 | | | | |

## notes

- bleu is corpus bleu with n-grams 1-4 and the brevity penalty.
- no-unk subsets drop triples where source or reference has the unk token.
- all runs use small-scale cpu dims. see `nmt/exp/configs.py` for the matrix.
- the 30-word mode trains on 64000 news-commentary pairs capped at 30 words.