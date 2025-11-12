#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations

import logging
import os
import pickle
import time
from math import inf
from pathlib import Path
from typing import Callable, Optional, List

import torch
from jaxtyping import Float
from torch import Tensor, cuda
from tqdm import tqdm  # type: ignore
from typeguard import typechecked

from corpus import (
    BOS_TAG,
    BOS_WORD,
    EOS_TAG,
    EOS_WORD,
    Sentence,
    Tag,
    TaggedCorpus,
    IntegerizedSentence,
    Word,
    OOV_WORD,
)
from integerize import Integerizer

TorchScalar = Float[Tensor, ""]  # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(
    Path(__file__).stem
)  # For usage, see findsim.py in earlier assignment.
# Note: We use the name "logger" this time rather than "log" since we
# are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available


###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.

    We'll refer to the HMM states as "tags" and the HMM observations
    as "words."
    """

    # As usual in Python, attributes and methods starting with _ are intended as private;
    # in this case, they might go away if you changed the parametrization of the model.

    def __init__(
        self, tagset: Integerizer[Tag], vocab: Integerizer[Word], unigram: bool = False
    ):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.

        Normally this is an ordinary first-order (bigram) HMM.  The `unigram` flag
        says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended to
        support higher-order HMMs: trigram HMMs used to be popular.)"""

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)  # number of tag types
        self.V = (
            len(vocab) - 2
        )  # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram  # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None  # we need this to exist
        self.eye: Tensor = torch.eye(
            self.k
        )  # identity matrix, used as a collection of one-hot tag vectors
        self.mu_A = 0.0
        self.mu_B = 0.0
        self.A_prior = None
        self.B_prior = None
        self.w_sup = 10.0
        self.w_raw = 1.0
        self.allowed_tags_for_word = None

        self.init_params()  # create and initialize model parameters

    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).
        We respect structural zeroes ("Don't guess when you know").

        If you prefer, you may change the class to represent the parameters in logspace,
        as discussed in the reading handout as one option for avoiding underflow; then name
        the matrices lA, lB instead of A, B, and construct them by logsoftmax instead of softmax.
        """

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively),
        # which don't have columns in this matrix).
        ###
        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.
        ###
        n_rows = 1 if self.unigram else self.k
        WB = 0.01 * torch.rand(self.k, self.V)  # choose random logits
        self.B = WB.softmax(dim=1)  # construct emission distributions p(w | t)

        WA = 0.01 * torch.rand(n_rows, self.k)
        WA[:, self.bos_t] = -inf  # correct the BOS_TAG column

        self.B[self.eos_t, :] = 0  # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0  # BOS_TAG can't emit any column's word
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)

        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where
            # p(t | s) doesn't depend on s.
            #
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
            self.A = self.A.repeat(self.k, 1)  # copy the single row k times
        self.A[:, self.bos_t] = 0.0
        self.A[self.eos_t, :] = 0.0

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [
            "(" + str(self.tagset[t]) + "|...)" for t in range(self.A.size(1))
        ]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):  # rows
            row = [str(self.tagset[s])] + [
                f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))
            ]
            print("\t".join(row))
        print("\nEmission matrix B:")
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):  # rows
            row = [str(self.tagset[t])] + [
                f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))
            ]
            print("\t".join(row))
        print("\n")

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using the expected
        counts (A_counts, B_counts) that were accumulated by the E step.
        The `λ` parameter will be used for add-λ smoothing.
        We respect structural zeroes ("don't guess when you know")."""

        # we should have seen no emissions from BOS or EOS tags
        assert self.B_counts[self.eos_t : self.bos_t, :].any() == 0, (
            "Your expected emission counts "
            "from EOS and BOS are not all zero, meaning you've accumulated them incorrectly!"
        )

        # Update emission probabilities (self.B).
        # self.B_counts += λ  # smooth the counts (EOS_WORD and BOS_WORD remain at 0 since they're not in the matrix)
        legal_rows = torch.ones(self.k, dtype=torch.bool, device=self.B_counts.device)
        legal_rows[self.bos_t] = False
        legal_rows[self.eos_t] = False
        self.B_counts[legal_rows, :] += λ
        row_sums = self.B_counts.sum(dim=1, keepdim=True).clamp_min(1e-24)

        self.B = torch.zeros_like(self.B_counts)
        # self.B = self.B_counts / self.B_counts.sum(
        #     dim=1, keepdim=True
        # )  # normalize into prob distributions

        self.B[legal_rows, :] = self.B_counts[legal_rows, :] / row_sums[legal_rows]
        self.B[self.eos_t, :] = (
            0  # replace these nan values with structural zeroes, just as in init_params
        )
        self.B[self.bos_t, :] = 0

        # we should have seen no "tag -> BOS" or "BOS -> tag" transitions
        assert self.A_counts[:, self.bos_t].any() == 0, (
            "Your expected transition counts "
            "to BOS are not all zero, meaning you've accumulated them incorrectly!"
        )
        assert self.A_counts[self.eos_t, :].any() == 0, (
            "Your expected transition counts "
            "from EOS are not all zero, meaning you've accumulated them incorrectly!"
        )

        # Update transition probabilities (self.A)
        legal = torch.ones(
            self.k, self.k, dtype=self.A_counts.dtype, device=self.A_counts.device
        )
        legal[:, self.bos_t] = 0
        legal[self.eos_t, :] = 0

        counts = self.A_counts * legal
        counts = counts + legal * λ
        row_sums = counts.sum(dim=1, keepdim=True)
        A_copy = torch.where(row_sums > 0, counts / row_sums, torch.zeros_like(counts))
        A_copy[:, self.bos_t] = 0.0
        A_copy[self.eos_t, :] = 0.0

        # Don't forget to respect the settings self.unigram and λ.
        # unigram case.
        if self.unigram:
            self.Aunigram = (self.A_counts * legal).sum(dim=0)
            self.Aunigram[self.eos_t] = 0.0
            self.Aunigram[self.bos_t] = 0.0
            self.Aunigram += λ
            Z_scaler = self.Aunigram.sum().clamp_min(1e-24)
            if Z_scaler > 0:
                self.Aunigram /= Z_scaler
            else:
                self.Aunigram = torch.zeros_like(self.Aunigram)
            A_copy = self.Aunigram.unsqueeze(0).repeat(self.k, 1)
            A_copy[self.bos_t, :] = 0.0
            A_copy[self.eos_t, :] = 0.0

        self.A = A_copy
        # raise NotImplementedError  # you fill this in!

    def _zero_counts(self):
        """Set the expected counts to 0.
        (This creates the count attributes if they didn't exist yet.)"""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(
        self,
        corpus: TaggedCorpus,
        loss: Callable[[HiddenMarkovModel], float],
        λ: float = 0,
        tolerance: float = 0.001,
        max_steps: int = 50000,
        save_path: Optional[Path | str] = "my_hmm.pkl",
    ) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        We will stop when the relative improvement of the development loss,
        since the last epoch, is less than the tolerance.  In particular,
        we will stop when the improvement is negative, i.e., the development loss
        is getting worse (overfitting).  To prevent running forever, we also
        stop if we exceed the max number of steps."""

        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
            # Smooth the counts by a tiny amount to avoid a problem where the M
            # step gets transition probabilities p(t | s) = 0/0 = nan for
            # context tags s that never occur at all, in particular s = EOS.
            #
            # These 0/0 probabilities are never needed since those contexts
            # never occur.  So their value doesn't really matter ... except that
            # we do have to keep their value from being nan.  They show up in
            # the matrix version of the forward algorithm, where they are
            # multiplied by 0 and added into a sum.  A summand of 0 * nan would
            # regrettably turn the entire sum into nan.

        self._save_time = time.time()  # mark start of training
        dev_loss = loss(self)  # evaluate the model at the start of training
        old_dev_loss: float = dev_loss  # loss from the last epoch
        steps: int = 0  # total number of sentences the model has been trained on so far
        while steps < max_steps:

            # E step: Run forward-backward on each sentence, and accumulate the
            # expected counts into self.A_counts, self.B_counts.
            #
            # Note: If you were using a GPU, you could get a speedup by running
            # forward-backward on several sentences in parallel.  This would
            # require writing the algorithm using higher-dimensional tensor
            # operations, allowing PyTorch to take advantage of hardware
            # parallelism.  For example, you'd update alpha[j-1] to alpha[j] for
            # all the sentences in the minibatch at once (with appropriate
            # handling for short sentences of length < j-1).

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)
                steps += 1

            # M step: Update the parameters based on the accumulated counts.
            self.M_step(λ)
            if save_path:
                self.save(
                    save_path, checkpoint=steps
                )  # save incompletely trained model in case we crash

            # Evaluate with the new parameters
            dev_loss = loss(self)  # this will print its own log messages
            if dev_loss >= old_dev_loss * (1 - tolerance):
                # we haven't gotten much better, so perform early stopping
                break
            old_dev_loss = dev_loss  # remember for next eval batch

        # Save the trained model.
        if save_path:
            self.save(save_path)

    def _integerize_sentence(
        self, sentence: Sentence, corpus: TaggedCorpus
    ) -> IntegerizedSentence:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            # Sentence comes from some other corpus that this HMM was not set up to handle.
            raise TypeError(
                "The corpus that this sentence came from uses a different tagset or vocab"
            )

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet.

        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent)

    # def E_step(self, isent: IntegerizedSentence, mult: float = 1):
    #   # Relative frequency
    #     for j in range(1, len(isent)):
    #         wj_minus_1, tj_minus_1 = isent[j - 1]
    #         wj, tj = isent[j]
    #
    #         # relative frequency estimation of Transition Matrix
    #         if (tj_minus_1 is not None and tj is not None) and (
    #             tj_minus_1 != self.eos_t and tj != self.bos_t
    #         ):
    #             self.A_counts[tj_minus_1, tj] += mult
    #
    #         # relative frequency estimation of Emission Matrix
    #         if j < len(isent) - 1:
    #             if tj is not None and wj is not None:
    #                 self.B_counts[tj, wj] += mult

    def E_step_supervised(self, isent, mult: float = 1.0) -> None:
        for j in range(1, len(isent)):
            w_prev, t_prev = isent[j - 1]
            w, t = isent[j]
            if j < len(isent) - 1:
                self.B_counts[t, w] += mult
            if t_prev != self.eos_t and t != self.bos_t:
                self.A_counts[t_prev, t] += mult

    def E_step(
        self,
        isent: IntegerizedSentence,
        mult: float = 1,
        temperature_scaling: float | None = None,
    ) -> None:
        """Runs the forward backward algorithm on the given sentence. The forward step computes
        the alpha probabilities.  The backward step computes the beta probabilities and
        adds expected counts to self.A_counts and self.B_counts.

        The multiplier `mult` says how many times to count this sentence.

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""

        # Forward-backward algorithm.
        log_Z_forward = self.forward_pass(isent)
        log_Z_backward = self.backward_pass(
            isent, mult=mult, temperature_scaling=temperature_scaling
        )

        # Check that forward and backward passes found the same total
        # probability of all paths (up to floating-point error).
        assert torch.isclose(
            log_Z_forward, log_Z_backward, atol=1e-4
        ), f"backward log-probability {log_Z_backward} doesn't match forward log-probability {log_Z_forward}!"

    def build_constraints_from_supervision(
        self, sup_corpus, min_count: int = 2
    ) -> None:
        counts = {}
        for sent in sup_corpus:
            isent = self._integerize_sentence(sent, sup_corpus)
            for j in range(1, len(isent) - 1):
                w, t = isent[j]
                d = counts.setdefault(w, {})
                d[t] = d.get(t, 0) + 1
        self.allowed_tags_for_word = {
            w: {t for t, c in d.items() if c >= min_count}
            for w, d in counts.items()
            if any(c >= min_count for c in d.values())
        }

    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged,
        or partially tagged sentence.  Return log Z (the log of the forward
        probability) as a TorchScalar.  If the sentence is not fully tagged, the
        forward probability will marginalize over all possible tags.

        As a side effect, remember the alpha probabilities and log_Z
        (store some representation of them into attributes of self)
        so that they can subsequently be used by the backward pass."""

        # The "nice" way to construct the sequence of vectors alpha[0],
        # alpha[1], ...  is by appending to a List[Tensor] at each step.
        # But to better match the notation in the handout, we'll instead
        # preallocate a list alpha of length n+2 so that we can assign
        # directly to each alpha[j] in turn.
        EPSILON = 1e-20
        NEGINF = -float("inf")
        lA, lB = torch.log(self.A.clamp_min(EPSILON)), torch.log(
            self.B.clamp_min(EPSILON)
        )
        # alpha = [torch.empty(self.k) for _ in isent]
        # alpha[0] = self.eye[self.bos_t]  # vector that is one-hot at BOS_TAG
        # log_alpha = torch.log(torch.stack(alpha).clamp_min(eps))
        log_alpha = torch.full((len(isent), self.k), NEGINF)
        log_alpha[0, self.bos_t] = 0.0

        # Note: once you have this working on the ice cream data, you may
        # have to modify this design slightly to avoid underflow on the
        # English tagging data. See section C in the reading handout.

        for j in range(1, len(isent)):
            if j == len(isent) - 1:
                lB_wj = torch.full((self.k,), NEGINF)
                lB_wj[self.eos_t] = 0.0
            else:
                wj, required_tag = isent[j]
                if required_tag is None:
                    lB_wj = lB[:, wj].clone()
                    lB_wj[self.eos_t] = NEGINF
                    lB_wj[self.bos_t] = NEGINF
                else:
                    lB_wj = torch.full((self.k,), NEGINF)
                    lB_wj[required_tag] = lB[required_tag, wj]
            if j < len(isent) - 1:
                log_alpha[j - 1][self.eos_t] = NEGINF

            scores = log_alpha[j - 1].unsqueeze(1) + lA
            log_alpha[j] = torch.logsumexp(scores, dim=0) + lB_wj

        self.log_Z = log_alpha[len(isent) - 1, self.eos_t]
        self.log_alpha = log_alpha
        return self.log_Z

    @typechecked
    def backward_pass(
        self,
        isent: IntegerizedSentence,
        mult: float = 1,
        temperature_scaling: float | None = None,
    ) -> TorchScalar:
        """Run the backwards algorithm from the handout on a tagged, untagged,
        or partially tagged sentence.  Return log Z (the log of the backward
        probability).

        As a side effect, add the expected transition and emission counts (times
        mult) into self.A_counts and self.B_counts.  These depend on the alpha
        values and log Z, which were stored for us (in self) by the forward
        pass."""

        # Pre-allocate beta just as we pre-allocated alpha.
        EPSILON = 1e-20
        NEGINF = -float("inf")
        N = len(isent)

        lA, lB = torch.log(self.A.clamp_min(EPSILON)), torch.log(
            self.B.clamp_min(EPSILON)
        )
        log_alpha = self.log_alpha

        # beta = [torch.empty(self.k) for _ in isent]
        # beta[-1] = self.eye[self.eos_t]  # vector that is one-hot at EOS_TAG
        log_beta = torch.full((N, self.k), NEGINF)
        # log_beta[0, self.bos_t] = 0.0
        log_beta[N - 1][self.eos_t] = 0.0

        for j in range(N - 2, -1, -1):
            if j == N - 2:
                lB_wj = torch.full((self.k,), NEGINF)
                lB_wj[self.eos_t] = 0.0
            else:
                wj, tag_in_pos = isent[j + 1]
                if tag_in_pos is None:
                    lB_wj = lB[:, wj].clone()
                    lB_wj[self.bos_t] = NEGINF
                    lB_wj[self.eos_t] = NEGINF
                else:
                    lB_wj = torch.full((self.k,), NEGINF)
                    lB_wj[tag_in_pos] = lB[tag_in_pos, wj]

            scores = lA + lB_wj.unsqueeze(0) + log_beta[j + 1, :].unsqueeze(0)
            log_beta[j, :] = torch.logsumexp(scores, dim=1)

            # wj, gold_tag = isent[1]
            # if gold_tag is None:
            #     lB_w1 = lB[:, wj].clone()
            #     lB_w1[self.bos_t] = NEGINF
            #     lB_w1[self.eos_t] = NEGINF
            # else:
            #     lB_w1 = torch.full((self.k,), NEGINF)
            #     lB_w1[gold_tag] = lB[gold_tag, wj]
            # Handle setup for log_Z_backward assertion
            if N > 2:
                wj, gold_tag = isent[1]
                if gold_tag is None:
                    lB_w1 = lB[:, wj].clone()
                    lB_w1[self.bos_t] = NEGINF
                    lB_w1[self.eos_t] = NEGINF
                else:
                    lB_w1 = torch.full((self.k,), NEGINF, device=lB.device)
                    lB_w1[gold_tag] = lB[gold_tag, wj]
            elif N == 2:
                lB_w1 = torch.full((self.k,), NEGINF, device=lB.device)
                lB_w1[self.eos_t] = 0.0
            else:
                lB_w1 = torch.full((self.k,), NEGINF, device=lB.device)

        for j in range(1, N - 1):
            Wj, required_tag = isent[j]
            log_gamma = log_alpha[j] + log_beta[j] - self.log_Z
            log_gamma[self.bos_t], log_gamma[self.eos_t] = NEGINF, NEGINF

            if (
                temperature_scaling is not None
                and temperature_scaling > 0
                and temperature_scaling != 1.0
            ):
                log_gamma = log_gamma / temperature_scaling
                logZ_local = torch.logsumexp(log_gamma, dim=0)
                log_gamma = log_gamma - logZ_local

            gamma = torch.exp(log_gamma)
            gamma[self.bos_t], gamma[self.eos_t] = 0.0, 0.0

            if required_tag is not None:
                temp = torch.zeros_like(gamma)
                temp[required_tag] = gamma[required_tag]
                gamma = temp
            self.B_counts[:, Wj] += gamma * float(mult)

        for j in range(1, N):
            if j == N - 1:
                logB_wj = torch.full((self.k,), NEGINF)
                logB_wj[self.eos_t] = 0.0
                allowed_transitions = torch.zeros(self.k, dtype=torch.bool)
                allowed_transitions[self.eos_t] = True

            else:
                Wj, required_tag = isent[j]
                if required_tag is None:
                    logB_wj = lB[:, Wj].clone()
                    logB_wj[self.bos_t] = NEGINF
                    logB_wj[self.eos_t] = NEGINF
                    allowed_transitions = torch.ones(self.k, dtype=torch.bool)
                    allowed_transitions[self.bos_t] = False
                    allowed_transitions[self.eos_t] = False
                else:
                    logB_wj = torch.full((self.k,), NEGINF)
                    logB_wj[required_tag] = lB[required_tag, Wj]
                    allowed_transitions = torch.zeros(self.k, dtype=torch.bool)
                    allowed_transitions[required_tag] = True

            log_transition_count_at_j = (
                log_alpha[j - 1].unsqueeze(1)
                + lA
                + logB_wj.unsqueeze(0)
                + log_beta[j].unsqueeze(0)
                - self.log_Z
            )
            if (
                temperature_scaling is not None
                and temperature_scaling > 0
                and temperature_scaling != 1.0
            ):
                mask = allowed_transitions.unsqueeze(0)  # shape (k, )
                ltc = log_transition_count_at_j.clone()
                # mask columns not allowed
                disallowed = (~mask).expand_as(ltc)
                ltc[disallowed] = NEGINF
                ltc = ltc / temperature_scaling
                logZ_xi = torch.logsumexp(ltc.view(-1), dim=0)
                ltc = ltc - logZ_xi
                transition_counts_at_j = torch.exp(ltc)
            else:
                transition_counts_at_j = torch.exp(log_transition_count_at_j)

            transition_counts_at_j[:, self.bos_t] = 0.0
            transition_counts_at_j[self.eos_t, :] = 0.0
            transition_counts_at_j *= allowed_transitions.unsqueeze(0)
            self.A_counts += transition_counts_at_j * float(mult)

        self.log_Z_backward = torch.logsumexp(
            lA[self.bos_t, :] + lB_w1 + log_beta[1, :], dim=0
        )
        self.log_beta = log_beta
        return self.log_Z_backward

    def _tags_at_j(self, j: int, isent: List) -> List[int]:
        if j == 0:
            return [self.bos_t]
        if j == len(isent) - 1:
            return [self.eos_t]

        _, tag_in_pos = isent[j]
        if tag_in_pos is not None:
            return [tag_in_pos]

        return [tag for tag in range(self.k) if tag not in (self.bos_t, self.eos_t)]

    def _tags_at_j_matrix(self, j: int, isent: List):
        if j == 0:
            return [self.bos_t]
        if j == len(isent) - 1:
            return [self.eos_t]
        _, gold = isent[j]
        if gold is not None:
            return [gold]
        return [t for t in range(self.k) if t not in (self.bos_t, self.eos_t)]

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # The code continues to use the name alpha, rather than \hat{alpha}
        # as in the handout.

        # We'll start by integerizing the input Sentence. You'll have to
        # deintegerize the words and tags again when constructing the return
        # value, since the type annotation on this method says that it returns a
        # Sentence object, and that's what downstream methods like eval_tagging
        # will expect.  (Running mypy on your code will check that your code
        # conforms to the type annotations ...)

        isent = self._integerize_sentence(sentence, corpus)

        # See comments in log_forward on preallocation of alpha.
        # alpha = [torch.empty(self.k) for _ in isent]
        # alpha = [torch.zeros(self.k) for _ in range(len(isent))]
        # alpha[0][self.bos_t] = 1.0
        # tags = [
        #     torch.tensor(self._tags_at_j_matrix(j, isent=isent))
        #     for j in range(len(isent))
        # ]

        # for j in range(1, len(isent)):
        #     wj, _ = isent[j]
        #     scale_k = 0
        #     for tj in tags[j].tolist():
        #         if j not in (0, len(isent) - 1):
        #             beta_score = float(self.B[tj][wj].item())
        #         else:
        #             beta_score = 1.0
        #
        #         for tj_minus_1 in tags[j - 1].tolist():
        #             cur_p = (
        #                 float(alpha[j - 1][tj_minus_1]) * float(self.A[tj_minus_1, tj])
        #             ) * beta_score
        #             if alpha[j][tj] < cur_p:
        #                 alpha[j][tj] = cur_p
        #                 backpointers[j][tj] = tj_minus_1
        #     alpha[j] = alpha[j] / sum(alpha[j])

        # for j in range(1, len(isent)):
        #     if j == len(isent) - 1:
        #         B_wj = torch.zeros(self.k)
        #         B_wj[self.eos_t] = 1.0
        #     else:
        #         wj, _ = isent[j]
        #         B_wj = torch.zeros(self.k)
        #         allowed = tags[j]
        #         B_wj[allowed] = self.B[allowed, wj]
        #
        #     scores = alpha[j - 1].unsqueeze(1) * self.A
        #     best_scores, backpointers[j] = scores.max(dim=0)
        #     alpha[j] = best_scores * B_wj
        eps = 1e-40
        NEGINF = -1e30
        lA, lB = torch.log(self.A.clamp_min(eps)), torch.log(self.B.clamp_min(eps))
        log_alpha = torch.full((len(isent), self.k), NEGINF)
        backpointers = torch.full((len(isent), self.k), -1, dtype=torch.long)
        log_alpha[0, self.bos_t] = 0.0

        for j in range(1, len(isent)):
            if j == len(isent) - 1:
                lB_wj = torch.full((self.k,), NEGINF)
                lB_wj[self.eos_t] = 0.0
            else:
                wj, gold = isent[j]
                if gold is None:
                    lB_wj = lB[:, wj].clone()
                    lB_wj[self.eos_t] = NEGINF
                    lB_wj[self.bos_t] = NEGINF
                else:
                    lB_wj = torch.full((self.k,), NEGINF)
                    lB_wj[gold] = lB[gold, wj]
            if j < len(isent) - 1:
                log_alpha[j - 1][self.eos_t] = NEGINF

            scores = log_alpha[j - 1].unsqueeze(1) + lA
            best_sequence_so_far, bp_j = scores.max(dim=0)
            log_alpha[j] = best_sequence_so_far + lB_wj
            bp_j[torch.isinf(lB_wj)] = -1
            backpointers[j] = bp_j

        best_sequence = [0] * len(isent)
        best_sequence[-1] = self.eos_t
        for j in range(len(isent) - 1, 0, -1):
            best_sequence[j - 1] = int(backpointers[j, best_sequence[j]].item())

        # Make a new tagged sentence with the old words and the chosen tags
        # (using self.tagset to deintegerize the chosen tags).
        sentence = Sentence(
            [
                (
                    OOV_WORD if word not in self.vocab else word,
                    self.tagset[best_sequence[j]],
                )
                for j, (word, tag) in enumerate(sentence)
            ]
        )
        return sentence

    def posterior_decoding(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        isent = self._integerize_sentence(sentence, corpus)
        NEGINF = -1e30
        tag_sequence = [0] * len(isent)
        tag_sequence[0] = self.bos_t
        tag_sequence[-1] = self.eos_t

        _ = self.forward_pass(isent)
        _ = self.backward_pass(isent, mult=0)

        tag_sequence = []
        for j in range(1, len(isent) - 1):
            _, gold = isent[j]
            log_gamma = self.log_alpha[j] + self.log_beta[j] - self.log_Z
            log_gamma[self.eos_t] = NEGINF
            log_gamma[self.bos_t] = NEGINF

            if gold is not None:
                masked = torch.full_like(log_gamma, NEGINF)
                masked[gold] = log_gamma[gold]
                log_gamma = masked
            tag_sequence.append(torch.argmax(log_gamma).item())
        best_sequence = [self.bos_t] + tag_sequence + [self.eos_t]
        return Sentence(
            (
                OOV_WORD if word not in self.vocab else word,
                self.tagset[best_sequence[j]],
            )
            for j, (word, _) in enumerate(sentence)
        )

    def save(
        self, path: Path | str, checkpoint=None, checkpoint_interval: int = 300
    ) -> None:
        """Save this model to the file named by path.  Or if checkpoint is not None, insert its
        string representation into the filename and save to a temporary checkpoint file (but only
        do this save if it's been at least checkpoint_interval seconds since the last save).  If
        the save is successful, then remove the previous checkpoint file, if any."""

        if isinstance(path, str):
            path = Path(path)  # convert str argument to Path if needed

        now = time.time()
        old_save_time = getattr(self, "_save_time", None)
        old_checkpoint_path = getattr(self, "_checkpoint_path", None)
        old_total_training_time = getattr(self, "total_training_time", 0)

        if checkpoint is None:
            self._checkpoint_path = None  # this is a real save, not a checkpoint
        else:
            if old_save_time is not None and now < old_save_time + checkpoint_interval:
                return  # we already saved too recently to save another temp version
            path = path.with_name(
                f"{path.stem}-{checkpoint}{path.suffix}"
            )  # use temp filename
            self._checkpoint_path = path

        # update the elapsed training time since we started training or last saved (if that happened)
        if old_save_time is not None:
            self.total_training_time = old_total_training_time + (now - old_save_time)
        del self._save_time

        # Save the model with the fields set as above, so that we'll
        # continue from it correctly when we reload it.
        try:
            torch.save(self, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved model to {path}")
        except Exception as e:
            # something went wrong with the save; so restore our old fields,
            # so that caller can potentially catch this exception and try again
            self._save_time = old_save_time
            self._checkpoint_path = old_checkpoint_path
            self.total_training_time = old_total_training_time
            raise e

        # Since save was successful, remember it and remove old temp version (if any)
        self._save_time = now
        if old_checkpoint_path:
            try:
                os.remove(old_checkpoint_path)
            except FileNotFoundError:
                pass  # don't complain if the user already removed it manually

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> HiddenMarkovModel:
        if isinstance(path, str):
            path = Path(path)  # convert str argument to Path if needed

        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        model = torch.load(path, map_location=device)

        if not isinstance(model, cls):
            raise ValueError(
                f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} "
                f"from saved file {path}."
            )

        logger.info(f"Loaded model from {path}")
        return model
