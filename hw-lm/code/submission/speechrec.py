#!/usr/bin/env python3
import argparse, logging, math
from pathlib import Path
from typing import Tuple, List
import torch

from probs import Wordtype, LanguageModel, BOS, EOS

log = logging.getLogger(Path(__file__).stem)


def read_candidates(file: Path, vocab: list) -> Tuple[int, List[Tuple[float, float, List[Wordtype]]]]:
    def to_tokens(words: List[str]) -> List[Wordtype]:
        toks: List[Wordtype] = [BOS, BOS]
        for w in words:
            if w == "~~":
                continue
            if w in vocab:
                toks.append(w)
            elif "OOV" in vocab:
                toks.append("OOV")
            else:
                toks.append(w)
        toks.append(EOS)
        return toks

    candidates = []
    
    with open(file, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            raise ValueError(f"{file}: empty file")
        first_parts = first.lstrip().split(maxsplit=1)
        try:
            true_len = int(first_parts[0])
        except ValueError as e:
            raise ValueError(f"{file}: first token must be an integer reference length") from e

        for line_num, line in enumerate(f, start=2):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                raise ValueError(f"{file}:{line_num}: need at least 3 fields (WER, log2, hyp_len)")
            try:
                wer = float(parts[0])
                acoustic_log2 = float(parts[1])
                hyp_len = int(parts[2])
            except ValueError as e:
                raise ValueError(f"{file}:{line_num}: malformed numeric header") from e

            if len(parts) < 3 + hyp_len:
                raise ValueError(f"{file}:{line_num}: expected {hyp_len} words, found {len(parts)-3}")
            words = parts[3:3+hyp_len]

            tokens = to_tokens(words)
            candidates.append((wer, acoustic_log2, tokens))

    return true_len, candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Choose best ASR hypothesis by Bayes: argmax log2 p(u|w) + log2 p_lm(w)")
    p.add_argument("model", type=Path, help="path to the trained LM model")
    p.add_argument("utterance_files", type=Path, nargs="+", help="utterance IDs like easy025 (no extension)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda","mps"],
                   help="PyTorch device")
    p.set_defaults(logging_level=logging.INFO)
    g = p.add_mutually_exclusive_group()
    g.add_argument("-v","--verbose", dest="logging_level", action="store_const", const=logging.DEBUG)
    g.add_argument("-q","--quiet",   dest="logging_level", action="store_const", const=logging.WARNING)
    return p.parse_args()


def trigrams_from_tokens(tokens: List[Wordtype]) -> List[Tuple[Wordtype, Wordtype, Wordtype]]:
    """Generate all trigrams from a token sequence."""
    trigrams = []
    for i in range(len(tokens) - 2):
        trigrams.append((tokens[i], tokens[i+1], tokens[i+2]))
    return trigrams


def candidate_log_prob(tokens: List[Wordtype], lm: LanguageModel) -> float:
    """Calculate log probability of a token sequence under the language model."""
    lp = 0.0
    for x, y, z in trigrams_from_tokens(tokens):
        lp += lm.log_prob(x, y, z)
        if lp == -math.inf:
            break
    return lp


def choose_best_utterance(file: Path, lm: LanguageModel) -> Tuple[float, int]:
    """
    Returns (selected_wer, true_len)
    Selection = argmax over candidates of: acoustic_log2 + prior_log2 (LM)
    """
    true_len, cands = read_candidates(file, lm.vocab)
    best_score = -math.inf
    best_wer = 1.0

    for wer, acoustic_log2, tokens in cands:
        prior_log2 = candidate_log_prob(tokens, lm)
        posterior = acoustic_log2 + prior_log2
        log.debug(f"{file.name}: cand WER={wer:.3f}  acoust={acoustic_log2:.2f}  prior={prior_log2:.2f}  post={posterior:.2f}")
        if posterior > best_score:
            best_score = posterior
            best_wer = wer

    return best_wer, true_len


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    # Device checks for mps
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                log.critical("PyTorch not built with MPS.")
            else:
                log.critical("MPS unavailable on this macOS/device.")
            raise SystemExit(1)
    torch.set_default_device(args.device)

    log.info("Loading model...")
    lm = LanguageModel.load(args.model, device=args.device)

    log.info("Processing utterances...")
    total_errors = 0.0
    total_words = 0

    for utt in args.utterance_files:
        file = Path(utt)
        sel_wer, true_len = choose_best_utterance(file, lm)
        print(f"{sel_wer:.3f}\t{file.name}")
        total_errors += sel_wer * true_len
        total_words  += true_len

    overall = (total_errors / total_words) if total_words > 0 else 0.0
    print(f"{overall:.3f}\tOVERALL")


if __name__ == "__main__":
    main()
