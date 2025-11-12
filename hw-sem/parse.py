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
        """Start looking for this nonterminal at the given position."""
        if nonterminal in self.grammar._expansions:
            for rule in self.grammar.expansions(nonterminal):
                new_item = Item(
                    rule=rule,
                    dot_position=0,
                    start_position=position,
                    weight=rule.weight,
                    backpointers=(),
                )
                self.cols[position].push(new_item)
                log.debug(f"\tPredicted: {new_item} in column {position}")
                self.profile["PREDICT"] += 1

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
        # Look through all items in the column where this item started
        start_col = self.cols[item.start_position]
        
        # Check all items that were waiting for this nonterminal
        for customer in start_col.all():
            if customer.next_symbol() == item.rule.lhs:
                # Create new item with dot advanced
                new_item = customer.with_dot_advanced(
                    backpointers=(customer, item),
                    new_weight=customer.weight + item.weight,
                )
                self.cols[position].push(new_item)
                log.debug(f"\tAttached to get: {new_item} in column {position}")
                self.profile["ATTACH"] += 1



class Agenda:
    """An agenda of items that need to be processed.  Newly built items
    may be enqueued for processing by `push()`, and should eventually be
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.

    >>> a = Agenda()
    >>> a.push(3)
    >>> a.push(5)
    >>> a.push(3)   # duplicate ignored
    >>> a
    Agenda([]; [3, 5])
    >>> a.pop()
    3
    >>> a
    Agenda([3]; [5])
    >>> a.push(3)   # duplicate ignored
    >>> a.push(7)
    >>> a
    Agenda([3]; [5, 7])
    >>> while a:    # that is, while len(a) != 0
    ...    print(a.pop())
    5
    7

    """

    def __init__(self, grammar: Grammar = None) -> None:
        self._items: List[Item] = []
        self._index: Dict[Item, int] = {}
        self._next = 0
        self.grammar = grammar

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    def push(self, item: Item) -> None:
        """Add (enqueue) the item, unless it was previously added with equal or better weight."""
        if item not in self._index:
            # First time seeing this item
            self._items.append(item)
            self._index[item] = len(self._items) - 1
        else:
            # We've seen this item before
            index = self._index[item]
            old_item = self._items[index]
            
            # Only update if new item has better (lower) weight
            # Check if items have weight attribute (for Item objects, not doctests)
            if hasattr(item, 'weight') and hasattr(old_item, 'weight') and item.weight < old_item.weight:
                # Update the item in place
                self._items[index] = item
                
                # If already processed, add to end for reprocessing
                if index < self._next:
                    self._items.append(item)

    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self) == 0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a human-readable string REPResentation of this Agenda."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"


class Grammar:
    """Represents a weighted context-free grammar."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol,
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited line of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

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