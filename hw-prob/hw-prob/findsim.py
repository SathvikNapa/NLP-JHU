#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from integerize import Integerizer   # look at integerize.py for more info

# Needed for Python's optional type annotations.
# We've included type annotations and recommend that you do the same, 
# so that mypy (or a similar package) can catch type errors in your code.
from typing import List, Optional

try:
    # PyTorch is your friend. Not using it will make your program so slow.
    # And it's also required for this assignment. ;-)
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    import torch    
    import torch.nn as nn
    import torch.nn.functional as F

except ImportError:
    print("\nERROR! You need to install Miniconda, then create and activate the nlp-class environment.  See the INSTRUCTIONS file.\n")
    raise


log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.

# Logging is in general a good practice to monitor the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# 
# - It prints to standard error (stderr), not standard output (stdout) by
#   default.  So these messages will normally go to your screen, even if
#   you have redirected stdout to a file.  And they will not be seen by
#   the autograder, so the autograder won't be confused by them.
# 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
# 
# In `parse_args`, we provided two command-line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'.
#
# More info: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
# 
# In all the starter code for the NLP course, we've elected to create a separate
# logger for each source code file, stored in a variable named log that
# is globally visible throughout the file.  That way, calls like log.info(...)
# will use the logger for the current source code file and thus their output will 
# helpfully show the filename.  You could configure the current file's logger using
# log.basicConfig(...), whereas logging.basicConfig(...) affects all of the loggers.
# The command-line options affect all of the loggers.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", type=Path, help="path to word embeddings file")
    parser.add_argument("word", type=str, help="word to look up")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error(f"Embeddings file {args.embeddings} not found")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both `--plus` and `--minus` or neither")

    return args

import numpy as np
from typing import Dict

class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words("bagpipe")
    """

    def __init__(self, *, vocab: Integerizer[str], token_embeddings_tensor: torch.Tensor) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""
        # FINISH THIS FUNCTION
        # Store your stuff! Both the word-index mapping and the embedding matrix.
        #
        # Do something with this size info?
        # PyTorch's torch.Tensor objects rely on fixed-size arrays in memory.
        # One of the worst things you can do for efficiency is
        # append row-by-row, like you would with a Python list.
        #
        # Probably make the entire list all at once, then convert to a torch.Tensor.
        # Otherwise, make the torch.Tensor and overwrite its contents row-by-row.
        self.vocab = vocab 
        self.token_embeddings_tensor = token_embeddings_tensor

    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        return Lexicon.WordEmbeddingBuilder(file).build()

    class WordEmbeddingBuilder:
        def __init__(self, file_path: Path, normalize: bool = True):
            self.file_path = file_path
            self._normalized = normalize
        
        def build(self):
            token_list = []
            embedding_list = []

            with open(self.file_path) as f:
                first_line = next(f)  # Peel off the special first line.
                for line in f:  # All of the other lines are regular.
                    splits = line.strip().split()
                    if len(splits) == 0:
                        continue
                    token = splits[0]
                    embedding = [float(split) for split in splits[1:]]
                    
                    token_list.append(token)
                    embedding_list.append(embedding)
                
                vocab = Integerizer(token_list)
                
                token_embeddings_tensor = torch.tensor(embedding_list, dtype=torch.float32)

                if self._normalized:
                    # for numerical stability
                    normalized = token_embeddings_tensor.norm(dim=1, keepdim=True)
                    token_embeddings_tensor = token_embeddings_tensor / (normalized + 1e-12)

                return Lexicon(vocab=vocab,token_embeddings_tensor=token_embeddings_tensor)  
        
    def find_similar_words(
        self, word: str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ) -> List[str]:
        """Find most similar words, in terms of embeddings, to a query."""
        # FINISH THIS FUNCTION

        # The star above forces you to use `plus` and `minus` only
        # as named arguments. This helps avoid mixups or readability
        # problems where you forget which comes first.
        #
        # We've also given `plus` and `minus` the type annotation
        # Optional[str]. This means that the argument may be None, or
        # it may be a string. If you don't provide these, it'll automatically
        # use the default value we provided: None.
        if word not in self.vocab:
            log.error(f"Word {word} not found in lexicon")
            raise ValueError(f"Word {word} not found in lexicon")

        if (minus is None) != (plus is None):  # != is the XOR operation!
            raise TypeError("Must include both of `plus` and `minus` or neither.")
        if plus is not None and minus is not None:
            if plus not in self.vocab or minus not in self.vocab:
                log.error(f"Word {plus} or {minus} not found in lexicon")
                raise ValueError(f"Word {plus} or {minus} not found in lexicon")
            
            plus_ix = self.vocab.index(plus)
            minus_ix = self.vocab.index(minus)
            word_ix = self.vocab.index(word)
            
            word_embedding = self.token_embeddings_tensor[word_ix]
            plus_embedding = self.token_embeddings_tensor[plus_ix]
            minus_embedding = self.token_embeddings_tensor[minus_ix]
            
            query_embedding = (word_embedding + plus_embedding - minus_embedding)
        else:
            query_embedding = self.token_embeddings_tensor[self.vocab.index(word)]
        
        # Keep going!
        # Be sure that you use fast, batched computations
        # instead of looping over the rows. If you use a loop or a comprehension
        # in this function, you've probably made a mistake.
        
        
        # Compute similarities using cosine similarity
        similarities = F.cosine_similarity(
                            self.token_embeddings_tensor,
                            query_embedding.unsqueeze(0).expand_as(self.token_embeddings_tensor),
                            dim=1
                        )

        # Exclude the query word itself from results
        exclude = {self.vocab.index(word)}
        if plus is not None and minus is not None:
            exclude |= {self.vocab.index(plus), self.vocab.index(minus)}
        for ix in exclude:
            similarities[ix] = -float("inf")
        available = similarities.numel() - len(exclude)
        min_k = min(10, available)
        topk_ix = torch.topk(similarities, k=min_k).indices.tolist()
        return [self.vocab[ix] for ix in topk_ix]



def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus
    )
    print(" ".join(similar_words))


if __name__ == "__main__":
    main()
