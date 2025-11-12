#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu> 
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from contextlib import nullcontext
import logging
import math
import pickle
import sys
import random
from SGD_convergent import ConvergentSGD
from pathlib import Path
from functools import lru_cache
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from jaxtyping import Float
from typeguard import typechecked
from typing import Counter, Collection
from collections import Counter

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Collection[Wordtype]   # and change this to Integerizer[str]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]
TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process(token)
    # Whenever the `for` loop needs another token, read_tokens magically picks up 
    # where it left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    # Convert from an unordered Set to an ordered List.  This ensures that iterating
    # over the vocab will always hit the words in the same order, so that you can 
    # safely store a list or tensor of embeddings in that order, for example.
    return sorted(vocab)   
    # Alternatively, you could choose to represent a Vocab as an Integerizer (see above).
    # Then you won't need to sort, since Integerizers already have a stable iteration order.

##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the 
    # denominator and c(yz) for the backed-off numerator.  Both of these 
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    # 
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    # 
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.  
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly, 
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    def save(self, model_path: Path) -> None:
        log.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            # torch.save is similar to pickle.dump but handles tensors too
        log.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> "LanguageModel":
        log.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
            # torch.load is similar to pickle.load but handles tensors too
            # map_location allows loading tensors on different device than saved
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from file {model_path}")
        log.info(f"Loaded model from {model_path}")
        return model

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )

class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # TODO: Reimplement me so that I do backoff
        lamda_ = self.lambda_
        V_size = self.vocab_size
        
        @lru_cache(maxsize=None) # zerogram
        def _p0() -> float:
            return 1.0/V_size

        @lru_cache(maxsize=None) # unigram
        def _p1(z_: Wordtype) -> float:
            N = self.context_count[()]
            if N == 0:
                return _p0()
            num = self.event_count[(z_,)] + lamda_ * V_size * _p0()
            den = N + lamda_ * V_size
            return num / den

        @lru_cache(maxsize=None) # bigram
        def _p2(y_: Wordtype, z_: Wordtype) -> float:
            c_y = self.context_count[(y_,)]
            if c_y == 0:
                return _p1(z_)
            num = self.event_count[(y_, z_)] + lamda_ * V_size * _p1(z_)
            den = c_y + lamda_ * V_size
            return num / den

        c_xy = self.context_count[(x, y)]
        if c_xy == 0:
            return _p2(y, z)

        num = self.event_count[(x, y, z)] + lamda_ * V_size * _p2(y, z)
        den = c_xy + lamda_ * V_size
        return num / den # trigram

class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int, lr: float=0.01) -> None:
        super().__init__(vocab)
        nn.Module.__init__(self)
        LanguageModel.__init__(self, vocab)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = (  # normalize
        #     torch.device("cuda") if self.device == "cuda" and torch.cuda.is_available() else
        #     torch.device("mps")  if self.device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
        #     torch.device("cpu")
        # )
        if l2 < 0:
            raise ValueError("Negative regularization strength {l2}")
        self.l2: float = l2
        self.lr: float = lr

        # Read the lexicon first to get the embedding dimension
        self._read_lexicon(lexicon_file)
        self.vocab_list = list(self.vocab)
        self.v2i = {w: i for i, w in enumerate(self.vocab_list)}

        self._prepare_vocab_embeddings()
        # We wrap the following matrices in nn.Parameter objects.
        # This lets PyTorch know that these are parameters of the model
        # that should be listed in self.parameters() and will be
        # updated during training.
        #
        # We can also store other tensors in the model class,
        # like constant coefficients that shouldn't be altered by
        # training, but those wouldn't use nn.Parameter.
        
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim), device=self.device), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim), device=self.device), requires_grad=True)
        self.epochs: int = epochs
    def _prepare_vocab_embeddings(self):
        E = []
        for w in self.vocab_list:
            E.append(self._get_embedding(w))
        self.E_vocab = torch.stack(E).to(self.device)          # [V, dim]
        self.E_vocab_T = self.E_vocab.t().contiguous()          # [dim, V]

    def _indexify_trigrams(self, file: Path):
        xs, ys, zs = [], [], []
        v2i = self.v2i
        for (x,y,z) in read_trigrams(file, self.vocab):
            xs.append(v2i.get(x, v2i[OOV]))
            ys.append(v2i.get(y, v2i[OOV]))
            zs.append(v2i.get(z, v2i[OOV]))
        X_idx = torch.tensor(xs, dtype=torch.long)
        Y_idx = torch.tensor(ys, dtype=torch.long)
        Z_idx = torch.tensor(zs, dtype=torch.long)
        return X_idx, Y_idx, Z_idx

    def _batch_logits(self, x_idx: torch.LongTensor, y_idx: torch.LongTensor):
        # E_vocab: [V, d], X,Y: [d,d], E_vocab_T: [d, V]
        # Pre-multiply once per step
        WX = self.X @ self.E_vocab_T    # [d, V]
        WY = self.Y @ self.E_vocab_T    # [d, V]

        Ex = self.E_vocab.index_select(0, x_idx)   # [B, d]
        Ey = self.E_vocab.index_select(0, y_idx)   # [B, d]

        # [B,d]@[d,V] → [B,V]
        return Ex @ WX + Ey @ WY

    def _read_lexicon(self, lexicon_file: Path) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        tokens = []
        vectors = []
        dim = None
        seen = set()

        # 'utf-8-sig' automatically strips BOM if present
        with open(lexicon_file, "r", encoding="utf-8-sig") as f:
            for line_num, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                # Header like "75 10" or "74981 10" → skip
                cols = line.split()
                if len(cols) == 2 and cols[0].isdigit() and cols[1].isdigit():
                    continue

                # First column is token; rest are floats (tab or space separated)
                token, *vals = cols
                try:
                    vec_vals = [float(x) for x in vals]
                except ValueError:
                    log.warning(f"Skipping line {line_num} with non-floats: {raw.rstrip()!r}")
                    continue

                if dim is None:
                    dim = len(vec_vals)
                    if dim == 0:
                        log.warning(f"Skipping empty vector at line {line_num}: {raw.rstrip()!r}")
                        continue
                elif len(vec_vals) != dim:
                    log.warning(
                        f"Skipping token {token!r} at line {line_num}: dim {len(vec_vals)} != expected {dim}"
                    )
                    continue

                if token in seen:
                    log.warning(f"Duplicate token {token!r} in {lexicon_file}")
                    continue
                seen.add(token)

                tokens.append(token)
                vectors.append(torch.tensor(vec_vals, dtype=torch.float32))

        if dim is None or not tokens:
            raise ValueError(f"No embeddings loaded from {lexicon_file}")

        self.embeddings = torch.stack(vectors).to(self.device)
        self.word2idx = {w: i for i, w in enumerate(tokens)}
        self.dim = dim

        # Heuristic: if most entries are single characters → char lexicon
        n_len1 = sum(1 for w in tokens if len(w) == 1 or w == "SPACE")
        self.is_char_lexicon = (n_len1 / max(1, len(tokens))) > 0.8

        # Fallback vector for OOL/OOV
        if OOL in self.word2idx:
            self._ool_vec = self.embeddings[self.word2idx[OOL]]
        else:
            self._ool_vec = torch.zeros(self.dim, device=self.device)

        # Small cache for composed embeddings (useful if you compose words from chars)
        self._compose_cache = {}


    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return self.log_prob_tensor(x, y, z).item()

    @typechecked
    # def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
    #     """Return the same value as log_prob, but stored as a tensor."""
        
    #     # As noted below, it's important to use a tensor for training.
    #     # Most of your intermediate quantities, like logits below, will
    #     # also be stored as tensors.  (That is normal in PyTorch, so it
    #     # would be weird to append `_tensor` to their names.  We only
    #     # appended `_tensor` to the name of this method to distinguish
    #     # it from the class's general `log_prob` method.)
    #     # TODO: IMPLEMENT ME!
    #     # This method should call the logits helper method.
    #     # You are free to define other helper methods too.
    #     #
    #     # Be sure to use vectorization over the vocabulary to
    #     # compute the normalization constant Z, or this method
    #     # will be very slow. Some useful functions of pytorch that could
    #     # be useful are torch.logsumexp and torch.log_softmax.
    #     #
    #     # The return type, TorchScalar, represents a torch.Tensor scalar.
    #     # See Question 7 in INSTRUCTIONS.md for more info about fine-grained 
    #     # type annotations for Tensors.

    #     # Computes logits for all vocabulary words from f1 and f2
    #     all_logits = self.logits(x, y)
        
    #     # Finds the index of z in vocabulary to compute log probability
    #     vocab_list = list(self.vocab)
    #     z_idx = vocab_list.index(z)
        
    #     # Computes log probability using log-softmax
    #     log_prob = all_logits[z_idx] - torch.logsumexp(all_logits, dim=0)
        
    #     return log_prob
    
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype):
        x_i = torch.tensor([self.v2i.get(x, self.v2i[OOV])], device=self.device)
        y_i = torch.tensor([self.v2i.get(y, self.v2i[OOV])], device=self.device)
        z_i = torch.tensor([self.v2i.get(z, self.v2i[OOV])], device=self.device)
        logits = self._batch_logits(x_i, y_i)  # [1, V]
        lp = torch.log_softmax(logits, dim=-1)[0, z_i]
        return lp.squeeze()

    def _get_embedding(self, word: Wordtype) -> Float[torch.Tensor, "embedding"]:
        if word == " " and "SPACE" in self.word2idx:
            word = "SPACE"
        idx = self.word2idx.get(word)
        if idx is not None:
            return self.embeddings[idx]
        if word in (OOV, OOL):
            return self._ool_vec
        if self.is_char_lexicon and len(word) > 1:
            if word in self._compose_cache:
                return self._compose_cache[word]
            chars = [self.embeddings[self.word2idx.get(ch, self.word2idx.get("SPACE", -1))]
                    if (ch in self.word2idx or "SPACE" in self.word2idx and ch == " ")
                    else self._ool_vec
                    for ch in word]
            vec = torch.stack(chars, 0).mean(0) if chars else self._ool_vec
            self._compose_cache[word] = vec
            return vec
        return self._ool_vec

    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor,"vocab"]:
        """Return a vector of the logs of the unnormalized probabilities f(xyz) * θ 
        for the various types z in the vocabulary.
        These are commonly known as "logits" or "log-odds": the values that you 
        exponentiate and renormalize in order to get a probability distribution."""
        
        embedding_x: Float[torch.Tensor, "embedding"] = self.E_vocab[self.v2i.get(x, self.v2i.get(OOV))]   # [dim]
        embedding_y: Float[torch.Tensor, "embedding"] = self.E_vocab[self.v2i.get(y, self.v2i.get(OOV))]   # [dim]

        # (ex @ X) @ E^T  +  (ey @ Y) @ E^T
        f1 = (embedding_x @ self.X) @ self.E_vocab_T                     # [V]
        f2 = (embedding_y @ self.Y) @ self.E_vocab_T                     # [V]
        return f1 + f2

    # def train(self, file: Path):    # type: ignore
        
        ### Technically this method shouldn't be called `train`,
        ### because this means it overrides not only `LanguageModel.train` (as desired)
        ### but also `nn.Module.train` (which has a different type). 
        ### However, we won't be trying to use the latter method.
        ### The `type: ignore` comment above tells the type checker to ignore this inconsistency.
        
        # Optimization hyperparameters.
        eta0 = self.lr
        C = self.l2

        # This is why we needed the nn.Parameter above.
        # The optimizer needs to know the list of parameters
        # it should be trying to update.
        # optimizer = ConvergentSGD(self.parameters(), eta0=eta0, lambda_=2*C)
        N = num_tokens(file)
        optimizer = optim.SGD(self.parameters(), lr=eta0)

        # Initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)   # type: ignore
        nn.init.zeros_(self.Y)   # type: ignore

        
        log.info(f"Start optimizing on {N} training tokens...")


        for epoch in range(self.epochs):
            sum_logprob = 0.0
            num_examples = 0
            for trigram in read_trigrams(file, self.vocab):
                
                log_prob = self.log_prob_tensor(trigram[0], trigram[1], trigram[2])
                loss = -log_prob

                loss.backward()

                with torch.no_grad():
                        for param in self.parameters():
                            if param.grad is not None:
                                param.grad.add_(param, alpha=(2 * C / N))
                
                optimizer.step()
                optimizer.zero_grad()
                
                sum_logprob += log_prob.item()
                num_examples += 1
                self.show_progress()

            F = sum_logprob / num_examples
            print(f"epoch {epoch+1}: F = {F}")
        sys.stderr.write("\n")
        print(f"Finished training on {N} tokens")
        log.info("done optimizing.")

        # So how does the `backward` method work?
        #
        # As Python sees it, your parameters and the values that you compute
        # from them are not actually numbers.  They are `torch.Tensor` objects.
        # A Tensor may represent a numeric scalar, vector, matrix, etc.
        #
        # Every Tensor knows how it was computed.  For example, if you write `a
        # = b + exp(c)`, PyTorch not only computes `a` but also stores
        # backpointers in `a` that remember how the numeric value of `a` depends
        # on the numeric values of `b` and `c`.  In turn, `b` and `c` have their
        # own backpointers that remember what they depend on, and so on, all the
        # way back to the parameters.  This is just like the backpointers in
        # parsing!
        #
        # Every Tensor has a `backward` method that computes the gradient of its
        # numeric value with respect to the parameters, using "back-propagation"
        # through this computation graph.  In particular, once you've computed
        # the forward quantity F_i(θ) as a tensor, you can trace backwards to
        # get its gradient -- i.e., to find out how rapidly it would change if
        # each parameter were changed slightly.

    def train(self, file: Path):  # type: ignore
        eta0 = self.lr
        C = self.l2

        # ===== 1) Dataset to tensors =====
        X_idx_cpu, Y_idx_cpu, Z_idx_cpu = self._indexify_trigrams(file)
        N = X_idx_cpu.numel()

        # Move once to GPU (non_blocking requires pinned mem; skip for brevity)
        X_idx = X_idx_cpu.to(self.device, non_blocking=False)
        Y_idx = Y_idx_cpu.to(self.device, non_blocking=False)
        Z_idx = Z_idx_cpu.to(self.device, non_blocking=False)

        ds = TensorDataset(X_idx, Y_idx, Z_idx)
        # crank batch_size up until you see GPU mem climb
        loader = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)

        # ===== 2) Init, optimizer =====
        nn.init.zeros_(self.X)
        nn.init.zeros_(self.Y)

        # L2 via weight_decay = 2*C/N (matches your added-grad when using plain SGD)
        optimizer = optim.SGD(self.parameters(), lr=eta0, weight_decay=(2.0*C)/float(N))

        # AMP context (fp16/bf16 if available)
        use_amp = torch.cuda.is_available()
        amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        torch.set_float32_matmul_precision("high")

        log.info(f"Start optimizing on {N} training tokens...")

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_items = 0

            for x_b, y_b, z_b in loader:
                optimizer.zero_grad(set_to_none=True)

                with amp_ctx():
                    logits = self._batch_logits(x_b, y_b)   # [B, V]
                    # CE = mean over batch; CE does log-softmax internally
                    loss = F.cross_entropy(logits, z_b, reduction="mean")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * x_b.size(0)
                total_items += x_b.size(0)

            F_epoch = - total_loss / total_items   # your F is avg log-prob; CE is -avg log-prob
            print(f"epoch {epoch+1}: F = {F_epoch}")
        print(f"Finished training on {N} tokens")
class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    # TODO: IMPLEMENT ME!
    
    # This is where you get to come up with some features of your own, as
    # described in the reading handout.  This class inherits from
    # EmbeddingLogLinearLanguageModel and you can override anything, such as
    # `log_prob`.

    # OTHER OPTIONAL IMPROVEMENTS: You could override the `train` method.
    # Instead of using 10 epochs, try "improving the SGD training loop" as
    # described in the reading handout.  Some possibilities:
    #
    # * You can use the `draw_trigrams_forever` function that we
    #   provided to shuffle the trigrams on each epoch.
    #
    # * You can choose to compute F_i using a mini-batch of trigrams
    #   instead of a single trigram, and try to vectorize the computation
    #   over the mini-batch.
    #
    # * Instead of running for exactly 10*N trigrams, you can implement
    #   early stopping by giving the `train` method access to dev data.
    #   This will run for as long as continued training is helpful,
    #   so it might run for more or fewer than 10*N trigrams.
    #
    # * You could use a different optimization algorithm instead of SGD, such
    #   as `torch.optim.Adam` (https://pytorch.org/docs/stable/optim.html).
    #
    pass
