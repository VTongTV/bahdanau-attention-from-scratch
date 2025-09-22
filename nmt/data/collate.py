"""padded collator producing id and mask tensors."""

import torch

_PAD = 0


def pad_rows(rows):
    """pad a list of id rows into a dense tensor."""
    return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=_PAD)


def collate(src_rows, tgt_rows):
    """return padded ids and masks for a batch."""
    src_len = [len(r) for r in src_rows]
    tgt_len = [len(r) for r in tgt_rows]
    src = pad_rows(src_rows)
    tgt = pad_rows(tgt_rows)
    src_mask = torch.arange(src.size(1)).unsqueeze(0) < torch.tensor(src_len).unsqueeze(1)
    tgt_mask = torch.arange(tgt.size(1)).unsqueeze(0) < torch.tensor(tgt_len).unsqueeze(1)
    return src, src_mask, tgt, tgt_mask