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

import logging
import math
import pickle
import sys
from SGD_convergent import ConvergentSGD
from pathlib import Path
from functools import lru_cache

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
        try:
            obj = torch.load(model_path, map_location=device)  # weights_only=True (default in 2.6+)
        except Exception:
            # fallback for old, fully-pickled checkpoints (trusted only)
            obj = torch.load(model_path, map_location=device, weights_only=False)
            # torch.load is similar to pickle.load but handles tensors too
            # map_location allows loading tensors on different device than saved
        # if not isinstance(model, cls):
        #     raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from file {model_path}")
        # log.info(f"Loaded model from {model_path}")
        return obj

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
        self.E_vocab = torch.stack(E).to(self.device)
        self.E_vocab_T = self.E_vocab.t().contiguous()

    def _read_lexicon(self, lexicon_file: Path) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        tokens = []
        vectors = []
        dim = None
        seen = set()

        with open(lexicon_file, "r", encoding="utf-8-sig") as f:
            for line_num, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                cols = line.split()
                if len(cols) == 2 and cols[0].isdigit() and cols[1].isdigit():
                    continue

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

        n_len1 = sum(1 for w in tokens if len(w) == 1 or w == "SPACE")
        self.is_char_lexicon = (n_len1 / max(1, len(tokens))) > 0.8

        if OOL in self.word2idx:
            self._ool_vec = self.embeddings[self.word2idx[OOL]]
        else:
            self._ool_vec = torch.zeros(self.dim, device=self.device)

        self._compose_cache = {}


    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        return self.log_prob_tensor(x, y, z).item()

    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        """Return the same value as log_prob, but stored as a tensor."""
        
        # As noted below, it's important to use a tensor for training.
        # Most of your intermediate quantities, like logits below, will
        # also be stored as tensors.  (That is normal in PyTorch, so it
        # would be weird to append `_tensor` to their names.  We only
        # appended `_tensor` to the name of this method to distinguish
        # it from the class's general `log_prob` method.)
        # TODO: IMPLEMENT ME!
        # This method should call the logits helper method.
        # You are free to define other helper methods too.
        #
        # Be sure to use vectorization over the vocabulary to
        # compute the normalization constant Z, or this method
        # will be very slow. Some useful functions of pytorch that could
        # be useful are torch.logsumexp and torch.log_softmax.
        #
        # The return type, TorchScalar, represents a torch.Tensor scalar.
        # See Question 7 in INSTRUCTIONS.md for more info about fine-grained 
        # type annotations for Tensors.

        # Computes logits for all vocabulary words from f1 and f2
        all_logits = self.logits(x, y)
        
        # Finds the index of z in vocabulary to compute log probability
        vocab_list = list(self.vocab)
        z_idx = vocab_list.index(z)
        
        # Computes log probability using log-softmax
        log_prob = all_logits[z_idx] - torch.logsumexp(all_logits, dim=0)
        
        return log_prob


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

        f1 = (embedding_x @ self.X) @ self.E_vocab_T
        f2 = (embedding_y @ self.Y) @ self.E_vocab_T
        return f1 + f2

    def train(self, file: Path):    # type: ignore
        
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
        optimizer = optim.SGD(self.parameters(), lr=eta0, weight_decay=2*C / N)

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

class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    """New Features:
    - Additional features
    - Adam optimizer
    - Mini-batching
    - Shuffling using draw_trigrams_forever
    - Unigram feature
    - Trigram interaction feature
    - OOV feature
    """
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int, 
                 lr: float=0.001, batch_size: int=32, use_adam: bool=True) -> None:
        super().__init__(vocab, lexicon_file, l2, epochs, lr)
        
        self.batch_size = batch_size
        self.use_adam = use_adam
        
        # parameters for tuning new features
        self.beta = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.theta_oov = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        self.Z = nn.Parameter(torch.zeros((self.dim,), device=self.device), requires_grad=True)
        self.W = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        
    def _compute_unigram_counts(self, file: Path):
        """Compute unigram counts from training file."""
        self.unigram_counts = Counter()
        for token in read_tokens(file, self.vocab):
            self.unigram_counts[token] += 1
    
    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor, "vocab"]:
        """Enhanced logits with additional features."""
        
        embedding_x = self.E_vocab[self.v2i.get(x, self.v2i.get(OOV))]
        embedding_y = self.E_vocab[self.v2i.get(y, self.v2i.get(OOV))]
        
        # Original features
        f1 = (embedding_x @ self.X) @ self.E_vocab_T
        f2 = (embedding_y @ self.Y) @ self.E_vocab_T
        
        # Unigram feature
        f_unigram = self.Z @ self.E_vocab_T
        
        # Trigram interaction feature
        xy_interaction = (embedding_x * embedding_y).sum()
        f_trigram = self.W * xy_interaction * torch.ones(len(self.vocab), device=self.device)
        
        logits = f1 + f2 + f_unigram + f_trigram
        
        # Unigram log-probability feature
        if hasattr(self, 'unigram_counts'):
            log_unigram_probs = torch.tensor([
                math.log(self.unigram_counts.get(w, 0) + 1) 
                for w in self.vocab_list
            ], device=self.device)
            logits = logits + self.beta * log_unigram_probs
        
        # OOV feature
        oov_idx = self.v2i.get(OOV)
        if oov_idx is not None:
            logits[oov_idx] = logits[oov_idx] + self.theta_oov
        
        return logits
    
    def logits_batch(self, x_batch: List[Wordtype], y_batch: List[Wordtype]) -> Float[torch.Tensor, "batch vocab"]:
        """Compute logits for a batch of contexts."""
        batch_size = len(x_batch)
        
        # Get batch embeddings
        x_indices = [self.v2i.get(x, self.v2i.get(OOV)) for x in x_batch]
        y_indices = [self.v2i.get(y, self.v2i.get(OOV)) for y in y_batch]
        
        embedding_x_batch = self.E_vocab[x_indices]
        embedding_y_batch = self.E_vocab[y_indices]
        
        f1 = (embedding_x_batch @ self.X) @ self.E_vocab_T
        f2 = (embedding_y_batch @ self.Y) @ self.E_vocab_T
        
        # Unigram feature
        f_unigram = self.Z @ self.E_vocab_T
        f_unigram = f_unigram.unsqueeze(0).expand(batch_size, -1)
        
        # Trigram interaction
        xy_interaction = (embedding_x_batch * embedding_y_batch).sum(dim=1)  # [batch]
        f_trigram = self.W * xy_interaction.unsqueeze(1)  # [batch, 1]
        
        logits = f1 + f2 + f_unigram + f_trigram
        
        # Add unigram log-probability feature
        if hasattr(self, 'unigram_counts'):
            log_unigram_probs = torch.tensor([
                math.log(self.unigram_counts.get(w, 0) + 1) 
                for w in self.vocab_list
            ], device=self.device)
            logits = logits + self.beta * log_unigram_probs.unsqueeze(0)
        
        # Add OOV feature
        oov_idx = self.v2i.get(OOV)
        if oov_idx is not None:
            logits[:, oov_idx] = logits[:, oov_idx] + self.theta_oov
        
        return logits
    
    def log_prob_tensor_batch(self, x_batch: List[Wordtype], y_batch: List[Wordtype], 
                             z_batch: List[Wordtype]) -> Float[torch.Tensor, "batch"]:
        """Compute log probabilities for a batch of trigrams."""
        
        # Get logits for all contexts in batch: [batch, vocab]
        all_logits = self.logits_batch(x_batch, y_batch)
        
        z_indices = torch.tensor([self.v2i[z] for z in z_batch], device=self.device)
        
        # Get logits for actual z words
        selected_logits = all_logits[torch.arange(len(z_batch)), z_indices]
        
        # Compute log Z for each context
        log_Z = torch.logsumexp(all_logits, dim=1)
        
        # Compute log probabilities
        log_probs = selected_logits - log_Z
        
        return log_probs
    
    def train(self, file: Path):
        """Enhanced training with mini-batching, shuffling via draw_trigrams_forever, and Adam optimizer."""
        
        # Compute unigram counts
        self._compute_unigram_counts(file)
        
        eta0 = self.lr
        C = self.l2
        N = num_tokens(file)
        
        if self.use_adam: # default is Adam optimizer
            optimizer = optim.Adam(self.parameters(), lr=eta0)
            log.info("Using Adam optimizer")
        else:
            optimizer = ConvergentSGD(self.parameters(), eta0=eta0, lambda_=2*C/N)
            log.info("Using SGD optimizer")
        
        nn.init.zeros_(self.X)
        nn.init.zeros_(self.Y)
        nn.init.zeros_(self.Z)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.theta_oov)
        nn.init.zeros_(self.W)
        
        log.info(f"Start optimizing on {N} training tokens...")
        log.info(f"Model has {sum(p.numel() for p in self.parameters())} parameters")
        log.info(f"Batch size: {self.batch_size}")
        
        for epoch in range(self.epochs):
            sum_logprob = 0.0
            num_examples = 0
            
            trigram_iter = draw_trigrams_forever(file, self.vocab, randomize=True)
            
            batch_x, batch_y, batch_z = [], [], []
            
            # Process N trigrams
            for _ in range(N):
                x, y, z = next(trigram_iter)
                
                batch_x.append(x)
                batch_y.append(y)
                batch_z.append(z)
                
                # Process batch when full
                if len(batch_x) == self.batch_size:
                    log_probs = self.log_prob_tensor_batch(batch_x, batch_y, batch_z)
                    loss = -log_probs.mean()
                    loss.backward()
                    
                    with torch.no_grad():
                        for param in self.parameters():
                            if param.grad is not None:
                                param.grad.add_(param, alpha=(2 * C / N))
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    sum_logprob += log_probs.sum().item()
                    num_examples += len(batch_x)
                    
                    for _ in range(len(batch_x)):
                        self.show_progress()
                    
                    batch_x, batch_y, batch_z = [], [], []
            
            # Process remaining partial batch
            if batch_x:
                log_probs = self.log_prob_tensor_batch(batch_x, batch_y, batch_z)
                loss = -log_probs.mean()
                loss.backward()
                
                with torch.no_grad():
                    for param in self.parameters():
                        if param.grad is not None:
                            param.grad.add_(param, alpha=(2 * C / N))
                
                optimizer.step()
                optimizer.zero_grad()
                
                sum_logprob += log_probs.sum().item()
                num_examples += len(batch_x)
                
                for _ in range(len(batch_x)):
                    self.show_progress()
            
            F = sum_logprob / num_examples
            print(f"epoch {epoch+1}: F = {F}")
        
        sys.stderr.write("\n")
        print(f"Finished training on {N} tokens")
        log.info("done optimizing.")
        
        log.info(f"Learned weights: beta={self.beta.item():.4f}, "
                f"theta_oov={self.theta_oov.item():.4f}, "
                f"W={self.W.item():.4f}")
