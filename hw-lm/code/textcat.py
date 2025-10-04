#!/usr/bin/env python3
import argparse
import logging
import math
from pathlib import Path
import sys
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams, BackoffAddLambdaLanguageModel

log = logging.getLogger(Path(__file__).stem)

def model_path(p: str) -> Path:
    path = Path(p)
    if path.suffix != ".model":
        raise argparse.ArgumentTypeError(f"{p} is not a .model file")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{p} does not exist")
    return path

def txt_path(p: str) -> Path:
    path = Path(p)
    if path.suffix != ".txt":
        raise argparse.ArgumentTypeError(f"{p} is not a .txt file")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{p} does not exist")
    return path

def prob_float(x: str) -> float:
    try:
        v = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Threshold must be a float, got {x!r}")
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError("Threshold must be in [0, 1]")
    return v


def _split_models_threshold_texts(tokens: list[str]):
    """
    Split tokens into ([models...], threshold, [texts...]) by finding the first
    token that parses as a float. Everything before it must be .model files,
    everything after it must be .txt files.
    """
    if len(tokens) < 3:
        raise argparse.ArgumentTypeError(
            "Usage: <models...> <threshold> <texts...>  (need at least 1 model, 1 threshold, 1 text)"
        )

    thresh_idx = None
    for i, t in enumerate(tokens):
        try:
            float(t)
            thresh_idx = i
            break
        except ValueError:
            continue

    if thresh_idx is None:
        raise argparse.ArgumentTypeError("Missing threshold (a float) in the arguments")

    models_tokens = tokens[:thresh_idx]
    if not models_tokens:
        raise argparse.ArgumentTypeError("At least one .model file must precede the threshold")

    texts_tokens = tokens[thresh_idx + 1:]
    if not texts_tokens:
        raise argparse.ArgumentTypeError("At least one .txt file must follow the threshold")

    threshold = prob_float(tokens[thresh_idx])

    if len(models_tokens) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected exactly 2 .model files (gen.model spam.model), got {len(models_tokens)}"
        )

    models = [model_path(m) for m in models_tokens]
    texts = [txt_path(t) for t in texts_tokens]
    return models, threshold, texts

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text categorizer: <gen.model> <spam.model> <prior_in_[0,1]> <texts...>"
    )

    parser.add_argument(
        "items",
        nargs="+",
        help="Positional args in order: two .model files, a float prior (P(gen)), then one or more .txt files.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="device to use for PyTorch",
    )

    vb = parser.add_mutually_exclusive_group()
    vb.add_argument("-v", "--verbose", dest="logging_level", action="store_const", const=10)  # DEBUG
    vb.add_argument("-q", "--quiet",   dest="logging_level", action="store_const", const=30)  # WARNING

    args = parser.parse_args()

    models, threshold, texts = _split_models_threshold_texts(args.items)
    args.models = models
    args.prior = threshold
    args.texts = texts
    del args.items
    if not hasattr(args, "logging_level"):
        args.logging_level = 20  # INFO

    return args

def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """Total natural log-probability of all sentences in the file under lm."""
    log_prob = 0.0
    x: Wordtype; y: Wordtype; z: Wordtype
    for (x, y, z) in read_trigrams(file, lm.vocab):
        lp = lm.log_prob(x, y, z)
        log_prob += lp
        if log_prob == -math.inf:
            break
    return log_prob

def classify_file(file_path: Path, gen_model: LanguageModel, spam_model: LanguageModel, p_gen: float):
    
    assert gen_model.vocab == spam_model.vocab
    gen_lp  = file_log_prob(file_path, gen_model)  + math.log(p_gen)
    spam_lp = file_log_prob(file_path, spam_model) + math.log(1.0 - p_gen)
    
    ground_truth = file_path.name.split(".")[0] 
    if gen_lp >= spam_lp:
        return "gen", gen_lp, spam_lp, ground_truth
    else:
        return "spam", gen_lp, spam_lp, ground_truth

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                log.critical("MPS not available because PyTorch wasn't built with MPS.")
            else:
                log.critical("MPS not available: macOS < 12.3 or no MPS device.")
            sys.exit(1)
    torch.set_default_device(args.device)

    gen_path, spam_path = args.models
    gen_model  = LanguageModel.load(gen_path,  device=args.device)
    spam_model = LanguageModel.load(spam_path, device=args.device)

    if gen_model.vocab != spam_model.vocab:
        log.critical("Model vocabulary mismatch between %s and %s", gen_path, spam_path)
        sys.exit(1)

    prior = args.prior
    total = 0
    gen_wins = 0
    spam_wins = 0
    error_count = 0
    
    predictions = []
    for txt in args.texts:
        winner, gen_lp, spam_lp, ground_truth = classify_file(txt, gen_model, spam_model, prior)
        if winner == "gen":
            print(f"{gen_path.name}\t{txt.name}")
            gen_wins += 1
        else:
            print(f"{spam_path.name}\t{txt.name}")
            spam_wins += 1
        
        if winner != ground_truth:
            error_count += 1
        
        total += 1
        
    
    
    if total > 0:
        gen_pct  = 100.0 * gen_wins  / total
        spam_pct = 100.0 * spam_wins / total
        if gen_wins == 1:
            print(f"1 files were more probably from {gen_path.name} ({gen_pct:.2f}%)")
        else:
            print(f"{gen_wins} files were more probably from {gen_path.name} ({gen_pct:.2f}%)")
        if spam_wins == 1:
            print(f"1 files were more probably from {spam_path.name} ({spam_pct:.2f}%)")
        else:
            print(f"{spam_wins} files were more probably from {spam_path.name} ({spam_pct:.2f}%)")
        log.info(f"Error Rate: {error_count / total * 100:.2f}%")


if __name__ == "__main__":
    main()
