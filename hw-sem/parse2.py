#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple

log = logging.getLogger(
    Path(__file__).stem
)  # For usage, see findsim.py in earlier assignment.


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences",
        type=Path,
        help="Path to .sen file containing tokenized input sentences",
    )
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        dest="logging_level",
        action="store_const",
        const=logging.DEBUG,
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        dest="logging_level",
        action="store_const",
        const=logging.WARNING,
    )

    return parser.parse_args()


class EarleyChart:
    """A chart for Earley's algorithm."""

    def __init__(
        self, tokens: List[str], grammar: Grammar, progress: bool = False
    ) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self._run_earley()  # run Earley's algorithm to construct self.cols

    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        for item in self.cols[-1].all():  # the last column
            if (
                item.rule.lhs == self.grammar.start_symbol  # a ROOT item in this column
                and item.next_symbol() is None  # that is complete
                and item.start_position == 0
            ):  # and started back at position 0
                return True
        return False  # we didn't find any appropriate item

    def best_parse(self) -> Optional[Item]:
        """Return the best parse of the sentence."""
        best_item = None
        best_weight = float("inf")
        for item in self.cols[-1].all():
            if (
                item.rule.lhs == self.grammar.start_symbol
                and item.next_symbol() is None
                and item.start_position == 0
                and item.weight < best_weight
            ):
                best_item = item
                best_weight = item.weight
        return best_item

    def build_tree(self, item: Item) -> str:
        """Reconstruct parse tree from a complete item."""
        if item.next_symbol() is not None:
            raise ValueError("Can only reconstruct complete items")
        
        # Handle empty productions
        if len(item.rule.rhs) == 0:
            return f"({item.rule.lhs})"
        
        # Reconstruct the tree by following backpointers
        children = self._reconstruct_children(item)
        
        return f"({item.rule.lhs} {' '.join(children)})"
    
    def _reconstruct_children(self, item: Item) -> List[str]:
        """Reconstruct children by carefully following backpointers."""
        if item.dot_position == 0:
            return []
        
        # For a complete item, we need to reconstruct all its children
        # We'll walk backwards through the derivation
        result = []
        current = item
        
        # Process from right to left, then reverse
        while current.dot_position > 0:
            if len(current.backpointers) == 1:
                # Scan operation - previous item plus a terminal
                prev_item = current.backpointers[0]
                # The scanned terminal is at position dot_position-1 in the RHS
                terminal = current.rule.rhs[current.dot_position - 1]
                result.append(terminal)
                current = prev_item
                
            elif len(current.backpointers) == 2:
                # Attach operation
                customer, completed = current.backpointers
                # Build tree for the completed constituent
                completed_tree = self.build_tree(completed)
                result.append(completed_tree)
                current = customer
                
            else:
                # No backpointers - shouldn't happen for dot_position > 0
                break
        
        # Reverse to get left-to-right order
        result.reverse()
        return result


    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [Agenda(self.grammar) for _ in range(len(self.tokens) + 1)]

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        #
        # The iterator over numbered columns is `enumerate(self.cols)`.
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(
            enumerate(self.cols), total=len(self.cols), disable=not self.progress
        ):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            
            # Keep processing until the column is truly empty
            # This handles reprocessing when better paths are found
            while column:  # while agenda isn't empty
                item = column.pop()  # dequeue the next unprocessed item
                next = item.next_symbol()
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position, with left-corner filtering."""
        column = self.cols[position]

        if nonterminal in column.predicted_categories:
            log.debug(f"\tAlready predicted {nonterminal} in column {position}, skipping batch")
            return
        column.predicted_categories.add(nonterminal)

        grammar = self.grammar
        if position < len(self.tokens):
            word = self.tokens[position]
            Sj = self._compute_Sj(word)
            allowed_Bs = Sj.get(nonterminal, None)
        else:
            allowed_Bs = None  # no lookahead at sentence end

        # Predict only rules whose first symbol is plausible given left-corner info
        for rule in grammar.expansions(nonterminal):
            if not rule.rhs:
                column.push(Item(rule, 0, position, rule.weight, ()))
                self.profile["PREDICT"] += 1
                continue

            B = rule.rhs[0]
            if allowed_Bs is None or B in allowed_Bs:
                new_item = Item(rule, 0, position, rule.weight, ())
                column.push(new_item)
                log.debug(f"\tPredicted (LC-filtered): {new_item} in column {position}")
                self.profile["PREDICT"] += 1


    from functools import lru_cache

    def _compute_Sj(self, word: str) -> Dict[str, set[str]]:
        """
        Correct and efficient S_j computation.

        Walk UP the left-corner graph starting from the terminal `word` using Grammar.P,
        where P[Y] = { X | X -> Y ... } (Y is the FIRST symbol on X's RHS).

        For every edge X -> Y encountered on any upward path from `word`,
        record that Y is an allowed immediate first symbol for X:
            Sj[X].add(Y)

        Cached per word.
        """
        if not hasattr(self, "_Sj_cache"):
            self._Sj_cache = {}

        if word in self._Sj_cache:
            return self._Sj_cache[word]

        Sj: Dict[str, set[str]] = {}
        visited = set()
        stack = [word]  # start from the terminal itself

        while stack:
            Y = stack.pop()
            if Y in visited:
                continue
            visited.add(Y)

            # For every parent X with a rule X -> Y ...
            for X in self.grammar.P.get(Y, set()):
                if X not in Sj:
                    Sj[X] = set()
                Sj[X].add(Y)
                # Continue upward: ancestors of X may also begin with a chain leading to `word`
                stack.append(X)

        self._Sj_cache[word] = Sj
        return Sj



    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position,
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced(
                backpointers=(item,),
                new_weight=item.weight,
            )
            self.cols[position + 1].push(new_item)
            log.debug(f"\tScanned to get: {new_item} in column {position+1}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns."""
        start_col = self.cols[item.start_position]

        # Only iterate customers that were actually waiting for this LHS
        for customer in start_col.waiting.get(item.rule.lhs, []):
            if customer.next_symbol() != item.rule.lhs:
                # (defensive; the waiting list can contain older dominated variants)
                continue
            new_item = customer.with_dot_advanced(
                backpointers=(customer, item),
                new_weight=customer.weight + item.weight,
            )
            self.cols[position].push(new_item)
            log.debug(f"\tAttached to get: {new_item} in column {position}")
            self.profile["ATTACH"] += 1


from collections import defaultdict

class Agenda:
    def __init__(self, grammar: Grammar = None) -> None:
        self._items: List[Item] = []
        # cheaper dedup key: (rule, dot, start)
        self._index: Dict[Tuple[Rule, int, int], int] = {}
        self._next = 0
        self.grammar = grammar
        self.predicted_categories: set[str] = set()
        # index items that are waiting for a specific next symbol
        self.waiting: Dict[str, List[Item]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self._items) - self._next

    def push(self, item: Item) -> None:
        key = (item.rule, item.dot_position, item.start_position)

        if key not in self._index:
            self._items.append(item)
            self._index[key] = len(self._items) - 1

            nxt = item.next_symbol()
            if nxt:
                self.waiting[nxt].append(item)
        else:
            idx = self._index[key]
            old = self._items[idx]
            if item.weight < old.weight:
                self._items[idx] = item
                # if it was already popped, queue the improved one for processing later
                if idx < self._next:
                    self._items.append(item)
                nxt = item.next_symbol()
                if nxt:
                    self.waiting[nxt].append(item)

    def pop(self) -> Item:
        if len(self) == 0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        return self._items

    def __repr__(self):
        next_ = self._next
        return f"{self.__class__.__name__}({self._items[:next_]}; {self._items[next_:]})"


class Grammar:
    """Represents a weighted context-free grammar."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol and read grammar files."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}
        self.R: Dict[Tuple[str, str], List[Rule]] = {}  # R(A,B): rules A → B ...
        self.P: Dict[str, set[str]] = {}  # P(B): all A s.t. A → B ...

        for file in files:
            self.add_rules_from_file(file)

        # --- New: precompute left-corner closures for each symbol ---
        self.leftcorner_closure: Dict[str, set[str]] = {
            sym: self._compute_leftcorner_closure(sym) for sym in self._expansions
        }

    def _compute_leftcorner_closure(self, symbol: str) -> set[str]:
        """Compute all symbols that can appear as a left corner of `symbol`."""
        visited = set()
        stack = [symbol]
        while stack:
            Y = stack.pop()
            if Y in visited:
                continue
            visited.add(Y)
            for X in self.P.get(Y, set()):
                stack.append(X)
        return visited

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line)."""
        with open(file, "r") as f:
            for line in f:
                line = line.split("#")[0].rstrip()
                if not line:
                    continue

                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))

                # Normal expansions
                self._expansions.setdefault(lhs, []).append(rule)

                # Build left-corner maps for non-empty RHS
                if rhs:
                    B = rhs[0]
                    self.R.setdefault((lhs, B), []).append(rule)
                    self.P.setdefault(B, set()).add(lhs)


    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions.get(lhs, [])

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions



@dataclass(frozen=True)
class Rule:
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.

    >>> r = Rule('S',('NP','VP'),3.14)
    >>> r
    S → NP VP
    >>> r.weight
    3.14
    >>> r.weight = 2.718
    Traceback (most recent call last):
    dataclasses.FrozenInstanceError: cannot assign to field 'weight'
    """

    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def update_weight(self, new_weight: float) -> Rule:
        """Return a new Rule just like this one but with the given weight."""
        return Rule(lhs=self.lhs, rhs=self.rhs, weight=new_weight)

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        return f"{self.lhs} → {' '.join(self.rhs)}"

@dataclass(frozen=True)
class Item:
    """An item in the Earley parse chart, representing one or more subtrees
    that could yield a particular substring."""

    rule: Rule
    dot_position: int
    start_position: int
    weight: float = field(default=float("inf"), compare=False, hash=False)
    backpointers: Optional[Tuple] = field(default=(), compare=False, hash=False)

    def update_rule_weight(self, new_weight: float) -> Item:
        """Return a new Item just like this one but with the given rule weight."""
        new_rule = self.rule.update_weight(new_weight)
        return Item(
            rule=new_rule,
            dot_position=self.dot_position,
            start_position=self.start_position,
        )

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(
        self, backpointers: Optional[Tuple], new_weight: float
    ) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(
            rule=self.rule,
            dot_position=self.dot_position + 1,
            start_position=self.start_position,
            weight=new_weight,
            backpointers=backpointers,
        )

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # parse the sentence
                log.debug("=" * 70)
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                best_item = chart.best_parse()
                if best_item:
                    tree_str = chart.build_tree(best_item)
                    print(tree_str)
                    print(best_item.weight)
                else:
                    print("NONE")


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)  # run tests
    main()
