#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path

import torch
from typing import Optional, Union

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams, BOS, EOS

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="number of samples to generate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="maximum length of samples to generate",
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed for the random number generator",
        default=42,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="temperature for sampling (controls open-endedness)",
        default=1.0,
    )
    parser.add_argument(
        "--topk",
        type=int,
        help="",
        default=10,
    )
    
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob, num_tokens(file)


def _get_log_probs_vector(lm: LanguageModel, x: Wordtype, y: Wordtype, vlist: list[Wordtype]) -> torch.Tensor:
    vlist = list(lm.vocab)
    V = lm.vocab_size
    out = torch.empty(V, dtype=torch.float32)
    for i, z in enumerate(vlist):
        out[i] = lm.log_prob(x, y, z)
    
    return torch.clamp(out, min=-1e10)

def _apply_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    V = logits.numel()
    k = max(1, min(k, V))
    if k == V: return logits
    _, idx = torch.topk(logits, k=k)
    keep = set(idx.tolist())
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[list(keep)] = False
    return logits.masked_fill(mask, float("-inf"))

def _probs_vector(
    lm: LanguageModel,
    x: Wordtype,
    y: Wordtype,
    temperature: float = 1.0,
    topk: Optional[int] = None,
    vlist: list[Wordtype] = None,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logp = _get_log_probs_vector(lm, x, y, vlist) / temperature

    if topk is not None:
        try:
            logp = _apply_topk(logp, topk)
        except ValueError as e:
            log.warning(f"Error applying topk: {e}")
            V = logp.shape[0]
            k = max(1, min(topk, V))
            if k < V:
                vals, _ = torch.topk(logp, k=k)
                cutoff = vals[-1]
                logp = logp.masked_fill(logp < cutoff, float("-inf"))
    
    return torch.softmax(logp, dim=0)

@torch.no_grad()
def sample(
    lm: LanguageModel,
    max_length: int = 20,
    temperature: float = 1.0,
    topk: Optional[int] = None,
    seed: Optional[int] = None,
    vlist: list[Wordtype] = None,
) -> list[Wordtype]:
    
    if seed is not None:
        torch.manual_seed(seed)

    if vlist is None:
        vlist = list(lm.vocab)

    x, y = BOS, BOS
    out: list[Wordtype] = []

    for _ in range(max_length):
        probs = _probs_vector(lm, x, y, temperature=temperature, 
                              topk=topk, vlist=vlist)
        idx = torch.multinomial(probs, num_samples=1).item()
        z = vlist[idx]

        if z == EOS:
            break
        if z != BOS:
            out.append(z)

        x, y = y, z

    return out

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    log.info("Testing...")
    lm = LanguageModel.load(args.model, device=args.device)
    vlist = list(lm.vocab)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    for i in range(int(args.num_samples)):
        seed_i = None if args.seed is None else (args.seed + i)
        tokens = sample(
            lm,
            max_length=args.max_length,
            temperature=args.temperature,
            topk=args.topk,
            seed=seed_i,
            vlist=vlist,
        )
        truncated = (len(tokens) >= args.max_length)
        line = " ".join(tokens) if tokens else "(empty)"
        print(line + (" ..." if truncated else ""))

    if args.test_files:
        log.info("Per-file log-probabilities:")
        total_log_prob = 0.0
        for file in args.test_files:
            log_prob, tokens = file_log_prob(file, lm)
            # per-token cross-entropy in bits
            print(f"{(-log_prob / math.log(2)) / tokens:g}\t{file.name}")
            total_log_prob += log_prob

        bits = -total_log_prob / math.log(2)
        tokens = sum(num_tokens(f) for f in args.test_files)
        print(f"Overall cross-entropy:\t{bits / tokens:.5f} bits per token")


if __name__ == "__main__":
    main()
