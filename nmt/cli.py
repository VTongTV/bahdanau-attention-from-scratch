"""cli entry points: translate a source file with a trained model."""

import argparse
from pathlib import Path

import torch

from nmt.config import ExperimentConfig
from nmt.decode.beam import beam_search
from nmt.decode.unk import drop_unk
from nmt.model.rnnencdec import RNNencdec
from nmt.model.rnnsearch import RNNsearch
from nmt.train.checkpoint import load_checkpoint
from nmt.utils.device import pick_device
from nmt.vocab.detokenizer import detokenize
from nmt.vocab.special import special_ids
from nmt.vocab.tokenizer import tokenize
from nmt.vocab.vocabulary import Vocab


def train(argv=None):
    """train entry point, same flags as nmt/train/train.py."""
    from nmt.args import config_from_args
    from nmt.train.train import run

    run(config_from_args(argv))


def evaluate(argv=None):
    """corpus bleu of a hypotheses file against a references file."""
    from nmt.eval.corpus_bleu import corpus_bleu
    from nmt.eval.scorer import parse_ids

    p = argparse.ArgumentParser(description="score hypotheses with corpus bleu")
    p.add_argument("--hypotheses", required=True)
    p.add_argument("--references", required=True)
    args = p.parse_args(argv)
    hypotheses = [parse_ids(line) for line in open(args.hypotheses, encoding="utf-8")]
    references = [parse_ids(line) for line in open(args.references, encoding="utf-8")]
    return print_bleu(corpus_bleu(hypotheses, references))


def print_bleu(score):
    """print a bleu score and return it for tests."""
    print(f"bleu {score:.2f}")
    return score


def build_parser() -> argparse.ArgumentParser:
    """return the translate cli parser."""
    p = argparse.ArgumentParser(description="translate a source text file")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--beam-size", type=int, default=10)
    p.add_argument("--no-unk-suppress", action="store_true")
    p.add_argument("--drop-unk", action="store_true")
    return p


def translate_file(input_path, output_path, ckpt_path, data_dir, beam_size,
                   unk_suppress=True, drop_unk_flag=False):
    """translate every source line and write the target text."""
    raw = torch.load(ckpt_path, weights_only=False)
    config = ExperimentConfig.from_dict(raw["config"])
    config.beam_size = beam_size
    model = RNNsearch(config) if config.model == "rnnsearch" else RNNencdec(config)
    load_checkpoint(ckpt_path, model, None, config)
    model.to(pick_device(config.device))
    model.eval()
    src_vocab = Vocab.load(Path(data_dir) / "vocab.src")
    tgt_vocab = Vocab.load(Path(data_dir) / "vocab.tgt")
    special = special_ids(tgt_vocab)
    device = next(model.parameters()).device
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fi, \
            open(out, "w", encoding="utf-8") as fo:
        for line in fi:
            ids = ([special["bos"]]
                   + [src_vocab.id(t) for t in tokenize(line)]
                   + [special["eos"]])
            src = torch.tensor([ids], dtype=torch.long, device=device)
            tokens = beam_search(
                model, src, bos_id=special["bos"], eos_id=special["eos"],
                unk_id=special["unk"], beam_size=beam_size,
                unk_suppress=unk_suppress, max_len=config.max_len,
            )[0]
            if drop_unk_flag:
                tokens = drop_unk(tokens, special["unk"])
            words = [tgt_vocab.token_of(t) for t in tokens
                     if t not in (special["bos"], special["eos"])]
            fo.write(detokenize(words) + "\n")


def main(argv=None) -> None:
    """run the translate pipeline from cli flags."""
    args = build_parser().parse_args(argv)
    translate_file(
        args.input, args.output, args.ckpt, args.data_dir, args.beam_size,
        unk_suppress=not args.no_unk_suppress, drop_unk_flag=args.drop_unk,
    )


if __name__ == "__main__":
    main()


translate = main