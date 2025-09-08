#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import os
import sys
import random
import argparse
import re
from collections import ChainMap

terminal_pattern = r"(^[a-z\s]+)$"
pre_terminal_pattern = r"^[A-Za-z]{2,}$"

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="Generate random sentences from a PCFG"
    )
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str,
        required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file

        Returns:
            self
        """
        # Parse the input grammar file
        self.rules = None
        self._load_rules_from_file(grammar_file)

    @staticmethod
    def _select_probable_choice(items, weights):
        return random.choices(items, weights, k=1)[0]

    @staticmethod
    def _convert_tokens_to_sentence(tokens):
        sentence = " ".join(tokens)
        # sentence = re.sub(r"(?<!\.)\s+([.!?])", r"\1", sentence)
        sentence = re.sub(r"\s{2,}", "'", sentence)
        return sentence

    @staticmethod
    def _is_terminal(text):
        return bool(re.search(terminal_pattern, text))

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file
        """

        def _is_comment(line):
            if re.search(r"^[#\s()]", line):
                return 1

        grammar_text = open(grammar_file, "rb").read().decode("utf-8")
        
        valid_symbols = list(
            filter(
                lambda text: len(text) > 1 and not _is_comment(text),
                grammar_text.split("\n"),
            )
        )
        cleaned_valid_symbols = list(
            map(lambda line: line.split("#")[0].strip().split("\t"), valid_symbols)
        )

        grammar_hash = {}
        for symbol in cleaned_valid_symbols:
            if self._is_terminal(symbol[2]):
                if symbol[1] not in grammar_hash:
                    grammar_hash[symbol[1]] = {str(symbol[2]): float(symbol[0])}
                    continue
                grammar_hash[symbol[1]].update({str(symbol[2]): float(symbol[0])})
                continue
            else:
                if symbol[1] not in grammar_hash:
                    grammar_hash[symbol[1]] = {
                        tuple(map(str, symbol[2].split())): float(symbol[0])
                    }
                    continue
                grammar_hash[symbol[1]].update(
                    {tuple(map(str, symbol[2].split())): float(symbol[0])}
                )
                continue
        self.rules = grammar_hash

    def sample(self, derivation_tree, max_expansions, start_symbol):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent
                the tree (using bracket notation) that records how the sentence
                was derived
            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """
        if start_symbol not in self.rules:
            return start_symbol.strip()

        def _is_nonterminal(sym):
            return sym.isupper() or bool(re.search(pre_terminal_pattern, sym))

        n_expansions = 0
        def _tree_expand(symbol):
            nonlocal n_expansions

            if symbol not in self.rules:
                tokens = symbol.split()
                return tokens, symbol

            if _is_nonterminal(symbol):
                n_expansions += 1
                if n_expansions > max_expansions:
                    return [], f"({symbol} ...)"

            items = list(self.rules[symbol].keys())
            weights = list(self.rules[symbol].values())
            right_split = self._select_probable_choice(items=items, weights=weights)

            if isinstance(right_split, (tuple, list)):
                tokens_list, subtrees = [], []                
                for daughter in right_split:
                    if (daughter in self.rules) and \
                        (_is_nonterminal(daughter)) and \
                            (n_expansions >= max_expansions):
                        tokens_list.append("")
                        subtrees.append("...")
                        continue
                    tokens, subtree = _tree_expand(daughter)
                    tokens_list.extend(tokens)
                    subtrees.append(subtree)
                return tokens_list, f"({symbol} {' '.join(subtrees)})"
            else:
                return right_split.split(), f"({symbol} {right_split})"
        
        tokens, tree_str = _tree_expand(start_symbol)
        return tree_str if derivation_tree else self._convert_tokens_to_sentence(tokens)


####################
### Main Program
####################
def main():
    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
            derivation_tree=args.tree,
            max_expansions=args.max_expansions,
            start_symbol=args.start_symbol,
        )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), "prettyprint")
            t = os.system(f"echo '{sentence}' | perl {prettyprint_path}")
        else:
            print(sentence)


if __name__ == "__main__":
    main()
