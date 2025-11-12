#!/usr/bin/env python3
import argparse
import logging
import math
from pathlib import Path
import sys
import torch
from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Text categorizer using two language models",
        usage="%(prog)s model1 model2 prior texts [texts ...]"
    )
    
    parser.add_argument("items", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="device to use for PyTorch",
    )
    vb = parser.add_mutually_exclusive_group()
    vb.add_argument("-v", "--verbose", dest="logging_level", action="store_const", const=10)
    vb.add_argument("-q", "--quiet",   dest="logging_level", action="store_const", const=30)

    args = parser.parse_args()
    
    if len(args.items) < 4:
        parser.error("Need at least 4 arguments: model1 model2 prior text1 [text2 ...]")
    
    model1_str = args.items[0]
    model2_str = args.items[1]
    prior_str = args.items[2]
    text_files = args.items[3:]
    
    try:
        prior = float(prior_str)
    except ValueError:
        parser.error(f"Third argument must be a number (prior probability), got '{prior_str}'")
    
    if not (0.0 <= prior <= 1.0):
        parser.error(f"Prior must be between 0 and 1, got {prior}")
    
    model1_path = Path(model1_str)
    model2_path = Path(model2_str)
    
    args.model1 = model1_path
    args.model2 = model2_path
    args.prior = prior
    args.texts = text_files
    
    if not hasattr(args, "logging_level"):
        args.logging_level = 20
    
    del args.items
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

def classify_file(file_path, model1: LanguageModel, model2: LanguageModel, 
                  p_model1: float):
    """
    Classify a file using Bayes' theorem.
    Returns which model (1 or 2) is more likely.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    log_posterior1 = file_log_prob(file_path, model1) + math.log(p_model1)
    log_posterior2 = file_log_prob(file_path, model2) + math.log(1.0 - p_model1)
    
    return 1 if log_posterior1 >= log_posterior2 else 2

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

    try:
        model1 = LanguageModel.load(args.model1, device=args.device)
    except (FileNotFoundError, OSError) as e:
        log.critical(f"Cannot load model from {args.model1}: {e}")
        sys.exit(1)
    
    try:
        model2 = LanguageModel.load(args.model2, device=args.device)
    except (FileNotFoundError, OSError) as e:
        log.critical(f"Cannot load model from {args.model2}: {e}")
        sys.exit(1)

    if model1.vocab != model2.vocab:
        log.critical("Model vocabulary mismatch between %s and %s", args.model1, args.model2)
        sys.exit(1)

    model1_name = args.model1.name
    model2_name = args.model2.name
    
    model1_count = 0
    model2_count = 0
    
    for txt in args.texts:
        txt_path = Path(txt)
        winner = classify_file(txt_path, model1, model2, args.prior)
        
        if winner == 1:
            print(f"{model1_name}\t{txt_path}")
            model1_count += 1
        else:
            print(f"{model2_name}\t{txt_path}")
            model2_count += 1
    
    total = len(args.texts)
    if total > 0:
        model1_pct = 100.0 * model1_count / total
        model2_pct = 100.0 * model2_count / total
        
        if model1_count == 1:
            print(f"1 files were more probably from {model1_name} ({model1_pct:.2f}%)")
        else:
            print(f"{model1_count} files were more probably from {model1_name} ({model1_pct:.2f}%)")
        
        if model2_count == 1:
            print(f"1 files were more probably from {model2_name} ({model2_pct:.2f}%)")
        else:
            print(f"{model2_count} files were more probably from {model2_name} ({model2_pct:.2f}%)")


if __name__ == "__main__":
    main()
