#!/usr/bin/env python3
import argparse, logging, math
from pathlib import Path
from typing import Iterator, Tuple, List
import torch

from probs import Wordtype, LanguageModel, BOS, EOS

log = logging.getLogger(Path(__file__).stem)

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

def read_candidates(file: Path, vocab: dict) -> Tuple[int, Iterator[Tuple[float, float, List[Wordtype]]]]:
    """
    Returns (true_len, iterator over candidates),
    where each candidate is (wer, acoustic_log2, tokens_as_wordtypes_with_BOS_BOS_and_EOS).
    """
    def tokenize(words: List[str]) -> List[Wordtype]:
        toks: List[Wordtype] = [BOS, BOS]
        for w in words:
            if w == "~~":  # dataset sometimes marks deletions; skip
                continue
            toks.append(vocab.get(w, vocab.get("OOV", w)))
        toks.append(EOS)
        return toks

    with open(file, "r", encoding="utf-8") as f:
        # First line: gold transcription length, followed by the reference text (ignore text)
        first = f.readline()
        first_parts = first.strip().split()
        if not first_parts:
            raise ValueError(f"Empty first line in {file}")
        true_len = int(first_parts[0])  # number of words in the gold transcription

        def gen():
            for line in f:
                parts = line.rstrip("\n").split()
                if not parts:
                    continue
                wer = float(parts[0])                 # first column: word error rate for this hypothesis
                acoustic_log2 = float(parts[1])       # second: log2 p(u|w)
                hyp_len = int(parts[2])               # third: hypothesis length
                words = parts[3:3+hyp_len]            # exactly hyp_len words
                tokens = tokenize(words)
                yield (wer, acoustic_log2, tokens)
        return true_len, gen()

def trigrams_from_tokens(tokens: List[Wordtype]) -> Iterator[Tuple[Wordtype, Wordtype, Wordtype]]:
    for i in range(len(tokens) - 2):
        yield (tokens[i], tokens[i+1], tokens[i+2])

def candidate_log_prob(tokens: List[Wordtype], lm: LanguageModel) -> float:
    lp = 0.0
    for x, y, z in trigrams_from_tokens(tokens):
        lp += lm.log_prob(x, y, z)    # this should already be log2 in the starter code
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
        posterior = acoustic_log2 + prior_log2  # Bayes in log-space
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

    total_errors = 0.0
    total_words = 0

    for utt in args.utterance_files:
        file = Path(utt)  # homework expects bare IDs like easy025; pass full path if needed
        sel_wer, true_len = choose_best_utterance(file, lm)
        print(f"{sel_wer:.3f}\t{file.name}")
        total_errors += sel_wer * true_len
        total_words  += true_len

    overall = (total_errors / total_words) if total_words > 0 else 0.0
    print(f"{overall:.3f}\tOVERALL")

if __name__ == "__main__":
    main()
