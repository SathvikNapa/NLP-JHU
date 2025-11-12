#!/usr/bin/env python3
"""
Compare two HMM models on a dev corpus and list tokens that were
wrong in model1 but correct in model2. For each fixed token, show:
- delta_emission: log p(w|t_gold) change
- delta_prev:     log p(t_gold | t_{j-1}) change   (uses gold prev tag)
- delta_next:     log p(t_{j+1} | t_gold) change   (uses gold next tag)

Also prints overall accuracy for both models and counts of Fixed / Regressed / Unchanged.

Paths for --dev/--sup/--raw can be a file, a directory, or a glob.

Usage example:
  python analyze_fixes.py \
      --model1 my_hmm_before.pkl \
      --model2 my_hmm_after.pkl \
      --dev   data/endev \
      --sup   data/ensup \
      --raw   data/enraw \
      --max 40
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Union

import torch

from corpus import TaggedCorpus, Sentence

# Adjust these imports if your package layout differs
from hmm import HiddenMarkovModel

# ---------------------------
# Path normalization utilities
# ---------------------------

Pathish = Union[str, os.PathLike, Path]


def _as_iter(paths: Union[Pathish, Iterable[Pathish]]) -> Iterable[Pathish]:
    if isinstance(paths, (str, os.PathLike, Path)):
        return [paths]
    return paths


def _flatten_any(seq) -> Iterable:
    for x in seq:
        if isinstance(x, (list, tuple)):
            yield from _flatten_any(x)
        else:
            yield x


def _normalize_to_paths(paths: Union[Pathish, Iterable[Pathish]]) -> List[Path]:
    """
    Accept file/dir/glob or an iterable of those and return a **flat** List[Path].
    - Directories -> their immediate files (non-recursive)
    - Globs like 'data/ensup*' -> expanded from CWD
    - Single files -> kept
    """
    out: List[Path] = []

    for p in _as_iter(paths):
        p = Path(p)
        s = str(p)

        # Glob pattern
        if any(ch in s for ch in "*?[]"):
            out.extend(sorted(Path().glob(s)))
            continue

        # Directory -> immediate files
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if q.is_file()))
            continue

        # Single file (existing or not yet existing)
        out.append(p)

    # Extra safety: flatten again in case an upstream caller slipped in nesting
    out = [Path(x) for x in _flatten_any(out)]
    return out


def load_corpus(
    paths: Union[str, os.PathLike, Path, Iterable[Union[str, os.PathLike, Path]]],
    tagset=None,
    vocab=None,
) -> TaggedCorpus:
    files = _normalize_to_paths(paths)
    if not files:
        raise FileNotFoundError(f"No files matched: {paths!r}")

    return TaggedCorpus(*files, tagset=tagset, vocab=vocab)


# ---------------------------
# Small helpers
# ---------------------------


def strip_tags(sent: Sentence) -> Sentence:
    """Return a copy of `sent` with interior tags removed (BOS/EOS kept)."""
    out = []
    n = len(sent)
    for i, (w, t) in enumerate(sent):
        if i == 0 or i == n - 1:
            out.append((w, t))  # keep BOS/EOS tags
        else:
            out.append((w, None))  # strip gold tag
    return type(sent)(out)


def viterbi_tags(model: HiddenMarkovModel, sent: Sentence, corpus: TaggedCorpus):
    """Run Viterbi and return the tag *indices* (not tag objects)."""
    tagged = model.viterbi_tagging(sent, corpus)
    return [t for (_, t) in tagged]


def gold_tags(sent: Sentence):
    return [t for (_, t) in sent]


def logp_emission(model: HiddenMarkovModel, tag_idx: int, word_idx: int) -> float:
    return float(torch.log(model.B[tag_idx, word_idx].clamp_min(1e-20)))


def logp_trans(model: HiddenMarkovModel, prev_tag_idx: int, tag_idx: int) -> float:
    return float(torch.log(model.A[prev_tag_idx, tag_idx].clamp_min(1e-20)))


def word_status(word: str, sup_vocab: set, raw_vocab: set) -> str:
    if word in sup_vocab:
        return "Known"
    if word in raw_vocab:
        return "Seen"
    return "Novel"


# ---------------------------
# Main analysis
# ---------------------------


def main(args):
    # Load models
    m1: HiddenMarkovModel = HiddenMarkovModel.load(args.model1, device="cpu")
    m2: HiddenMarkovModel = HiddenMarkovModel.load(args.model2, device="cpu")

    # Load dev using model1's tagset/vocab to ensure consistent integerization
    dev = load_corpus(args.dev, tagset=m1.tagset, vocab=m1.vocab)

    # Optional: load sup/raw to categorize Known/Seen/Novel (by word strings)
    sup_vocab: set = set()
    raw_vocab: set = set()
    if args.sup:
        sup_c = load_corpus(
            args.sup
        )  # its own tagset/vocab is fine; we only need surface types
        sup_vocab = {w for w in sup_c.vocab}
    if args.raw:
        raw_c = load_corpus(args.raw)
        raw_vocab = {w for w in raw_c.vocab}

    # Quick param deltas (are the models actually different?)
    with torch.no_grad():
        dA = torch.max(torch.abs(m2.A - m1.A)).item()
        dB = torch.max(torch.abs(m2.B - m1.B)).item()
    print(f"Param deltas: max|ΔA|={dA:.3e}  max|ΔB|={dB:.3e}")

    fixes = []
    tot_tokens = 0
    fixed = 0
    regressed = 0
    unchanged = 0

    for sent in dev:
        # skip sentences that are only BOS/EOS
        if len(sent) <= 2:
            continue

        # Gold and unlabeled copies
        tags_gold = gold_tags(sent)
        sent_unl = strip_tags(sent)

        # Predict with both models (unconstrained)
        tags_m1 = viterbi_tags(m1, sent_unl, dev)
        tags_m2 = viterbi_tags(m2, sent_unl, dev)

        # integerize once for emission indexing
        isent = dev.integerize_sentence(sent)

        for j in range(1, len(sent) - 1):
            word, gold_tag = sent[j]
            if gold_tag is None:
                continue  # dev should be supervised, but be safe

            pred1 = tags_m1[j]
            pred2 = tags_m2[j]
            gold = gold_tag

            correct1 = pred1 == gold
            correct2 = pred2 == gold

            if correct1 and correct2:
                unchanged += 1
            elif (not correct1) and correct2:
                fixed += 1
            elif correct1 and (not correct2):
                regressed += 1
            else:
                unchanged += 1
            tot_tokens += 1

            # Only compute attributions for FIXED tokens
            if (not correct1) and correct2:
                # indices
                wj_idx, _ = isent[j]
                prev_gold = tags_gold[j - 1]
                next_gold = tags_gold[j + 1]

                t_prev = m1.tagset.index(prev_gold)
                t_gold = m1.tagset.index(gold)
                t_next = m1.tagset.index(next_gold)

                if None in (t_prev, t_gold, t_next, wj_idx):
                    continue

                t_prev = int(t_prev)
                t_gold = int(t_gold)
                t_next = int(t_next)
                w_idx = int(wj_idx)

                # Component deltas (model2 - model1)
                de = logp_emission(m2, t_gold, w_idx) - logp_emission(m1, t_gold, w_idx)
                dp = logp_trans(m2, t_prev, t_gold) - logp_trans(m1, t_prev, t_gold)
                dn = logp_trans(m2, t_gold, t_next) - logp_trans(m1, t_gold, t_next)

                parts = {"emission": de, "prev_trans": dp, "next_trans": dn}
                top_part = max(parts, key=parts.get)

                status = (
                    word_status(word, sup_vocab, raw_vocab)
                    if (args.sup or args.raw)
                    else "-"
                )

                fixes.append(
                    {
                        "sent_str": " ".join([w for (w, _) in sent[1:-1]]),
                        "position": j,
                        "word": word,
                        "gold": str(gold),
                        "m1_pred": str(pred1),
                        "m2_pred": str(pred2),
                        "delta_emission": round(de, 4),
                        "delta_prev": round(dp, 4),
                        "delta_next": round(dn, 4),
                        "top_cause": top_part,
                        "status": status,
                    }
                )

    # Summary
    if tot_tokens == 0:
        print("\nNo tokens to evaluate (did dev load correctly?).")
        return

    acc1 = (unchanged + regressed) / tot_tokens
    acc2 = (unchanged + fixed) / tot_tokens
    print(f"\nTokens (excl. BOS/EOS): {tot_tokens}")
    print(f"Model1 acc: {acc1:.3%}")
    print(f"Model2 acc: {acc2:.3%}")
    print(f"Fixed: {fixed}  Regressed: {regressed}  Unchanged: {unchanged}")

    # Sort and print fixes
    fixes.sort(
        key=lambda r: (
            max(r["delta_emission"], r["delta_prev"], r["delta_next"]),
            r["delta_emission"] + r["delta_prev"] + r["delta_next"],
        ),
        reverse=True,
    )

    print(f"\nFound {len(fixes)} fixed tokens (wrong in model1, correct in model2).")
    print("Showing up to", args.max, "examples:\n")
    for r in fixes[: args.max]:
        print(
            f"- '{r['word']}' @pos {r['position']} ({r['status']}) "
            f"gold={r['gold']} m1→{r['m1_pred']}  m2→{r['m2_pred']}"
        )
        print(
            f"  Δemiss={r['delta_emission']:+.3f}  Δprev={r['delta_prev']:+.3f}  "
            f"Δnext={r['delta_next']:+.3f}  ⇒ top: {r['top_cause']}"
        )
        print(f"  sent: {r['sent_str']}")
        print()

    if not fixes:
        print(
            "No fixes found. Ensure models differ and dev has gold tags; "
            "also verify semi-supervised training actually updated A/B."
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model1", required=True, help="Path to HMM before EM / baseline")
    p.add_argument("--model2", required=True, help="Path to HMM after EM / improved")
    p.add_argument(
        "--dev",
        required=True,
        help="Path (file/dir/glob) to supervised dev corpus (endev)",
    )
    p.add_argument(
        "--sup",
        default=None,
        help="Path (file/dir/glob) to ensup (for Known/Seen/Novel)",
    )
    p.add_argument(
        "--raw",
        default=None,
        help="Path (file/dir/glob) to enraw (for Known/Seen/Novel)",
    )
    p.add_argument("--max", type=int, default=30)
    args = p.parse_args()
    main(args)
