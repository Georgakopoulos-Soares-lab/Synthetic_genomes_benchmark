#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
constrained_decoding.py

Inference-time constraints for autoregressive DNA generation.

The benchmarks in this repository diagnose *where* generated sequence departs
from natural sequence. This module is one of the two interventions we tried in
response to those diagnoses: leave the model weights alone and bias the logits
at each sampling step, penalising the specific failure modes the benchmarks
detect.

Two constraints are provided, both of which target failures that
``composition.py`` measures directly:

``HomopolymerPenalty``
    Subtracts a penalty from the base that would extend the current
    homopolymer run, scaled by how long that run already is. Targets the
    inflated ``homopolymer_frac`` seen in unconstrained output.

``KmerOverrepresentationPenalty``
    Tracks the k-mer composition of what has been generated so far and
    penalises any next base that would extend an already over-represented
    k-mer, relative to a reference k-mer distribution taken from natural
    sequence. Targets ``kmer_jsd`` and ``low_complexity``.

Both are stateful per batch element and cost O(1) per step, so they can wrap a
model's existing cached-generation loop without changing its throughput
characteristics.

Design
------
The constraints know nothing about any particular model. They speak only in
terms of a ``NucleotideVocab`` mapping token ids to A/C/G/T, so the same code
drives a toy character-level GRU and Evo 2. ``logits`` may be a torch tensor or
a numpy array; only basic indexing and in-place subtraction are used.

Honest expectations
-------------------
In our hands these constraints reliably improve the statistic they target and
leave the *global* divergence largely intact: penalising homopolymers lowers
homopolymer content without making the sequence meaningfully harder to tell
from natural. Run ``evaluate_interventions.py`` on your own output rather than
assuming a benefit -- constraints that push a metric without improving
detectability are moving the symptom, not the cause.

Usage
-----
    from constrained_decoding import (
        NucleotideVocab, ConstraintStack, HomopolymerPenalty,
        KmerOverrepresentationPenalty,
    )

    vocab = NucleotideVocab.from_bytes()          # A=65, C=67, G=71, T=84
    stack = ConstraintStack([
        HomopolymerPenalty(vocab, strength=2.5),
        KmerOverrepresentationPenalty(vocab, reference_freqs, k=4),
    ])
    stack.reset(batch_size)

    for step in range(n):
        logits = model(...)                        # (B, vocab)
        logits = stack.process(logits)
        tokens = sample(logits)                    # your sampler, unchanged
        stack.record(tokens)

Self-test
---------
    python generation/improve/constrained_decoding.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

BASES = "ACGT"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@dataclass
class NucleotideVocab:
    """Maps model token ids onto nucleotide indices 0..3 (A, C, G, T)."""

    token_ids: list[int]                 # [id_A, id_C, id_G, id_T]

    def __post_init__(self) -> None:
        if len(self.token_ids) != 4:
            raise ValueError("token_ids must have exactly 4 entries (A, C, G, T)")
        if len(set(self.token_ids)) != 4:
            raise ValueError(f"token_ids must be distinct, got {self.token_ids}")
        self.token_to_base = {int(t): i for i, t in enumerate(self.token_ids)}

    @classmethod
    def from_bytes(cls) -> "NucleotideVocab":
        """Byte-level tokenizer: the ASCII code of each uppercase base."""
        return cls([ord(b) for b in BASES])

    @classmethod
    def identity(cls) -> "NucleotideVocab":
        """Four-symbol vocabulary where the token id *is* the base index."""
        return cls([0, 1, 2, 3])

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "NucleotideVocab":
        """Best-effort resolution of A/C/G/T ids from a model's tokenizer.

        Tries ``tokenize`` then ``encode``; falls back to the ASCII byte value,
        which is what byte-level genomic tokenizers (including Evo 2's) use.
        """
        ids = []
        for base in BASES:
            tid = None
            for method in ("tokenize", "encode"):
                fn = getattr(tokenizer, method, None)
                if fn is None:
                    continue
                try:
                    out = fn(base)
                except Exception:
                    continue
                if isinstance(out, int):
                    tid = out
                elif hasattr(out, "__len__") and len(out) >= 1:
                    first = out[0]
                    tid = int(first) if not isinstance(first, str) else ord(first)
                if tid is not None:
                    break
            ids.append(int(tid) if tid is not None else ord(base))
        return cls(ids)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class DecodingConstraint:
    """Base class: a stateful, per-batch-element logit modifier."""

    def reset(self, batch_size: int) -> None:
        raise NotImplementedError

    def process(self, logits):
        """Return ``logits`` with this constraint's penalties applied."""
        raise NotImplementedError

    def record(self, base_indices: np.ndarray) -> None:
        """Update state with the base index (0..3, or -1) chosen per batch row."""
        raise NotImplementedError


class HomopolymerPenalty(DecodingConstraint):
    """Discourage extending a homopolymer run.

    The penalty grows linearly with the current run length and saturates at
    ``cap`` so a long run cannot drive the logit to negative infinity:

        penalty = strength * min(run_length, cap) / cap

    ``strength`` is in logit units. At strength 2.5 a base that would extend a
    run of ``cap`` or more is down-weighted by e^-2.5 ~ 8x relative to its
    unconstrained probability.
    """

    def __init__(self, vocab: NucleotideVocab, strength: float = 2.5,
                 cap: int = 6) -> None:
        self.vocab = vocab
        self.strength = float(strength)
        self.cap = int(cap)
        self.reset(1)

    def reset(self, batch_size: int) -> None:
        self.last = np.full(batch_size, -1, dtype=np.int64)
        self.run = np.zeros(batch_size, dtype=np.int64)

    def process(self, logits):
        for b in range(len(self.last)):
            base = int(self.last[b])
            if base < 0:
                continue
            penalty = self.strength * min(int(self.run[b]), self.cap) / self.cap
            logits[b, self.vocab.token_ids[base]] -= penalty
        return logits

    def record(self, base_indices: np.ndarray) -> None:
        for b, base in enumerate(base_indices):
            base = int(base)
            if base < 0:
                continue
            if base == int(self.last[b]):
                self.run[b] += 1
            else:
                self.last[b] = base
                self.run[b] = 1


class KmerOverrepresentationPenalty(DecodingConstraint):
    """Penalise next bases that extend an over-represented k-mer.

    At each step the running frequency of every k-mer produced so far is
    compared with ``reference_freqs``. If a candidate base would complete a
    k-mer whose observed frequency already exceeds ``tolerance`` times its
    reference frequency, that base is penalised by

        strength * tanh(observed / reference - tolerance)

    The ``tanh`` bounds the penalty so a single wildly over-represented k-mer
    cannot dominate the distribution, and ``tolerance > 1`` leaves a band in
    which natural over-representation is not punished.

    ``reference_freqs`` should be a length-``4**k`` distribution measured on
    natural sequence of the same type (see ``reference_kmer_freqs``). Positions
    are indexed with A=0, C=1, G=2, T=3, most significant base first.
    """

    def __init__(self, vocab: NucleotideVocab, reference_freqs: np.ndarray,
                 k: int = 4, strength: float = 1.5,
                 tolerance: float = 1.5) -> None:
        ref = np.asarray(reference_freqs, dtype=np.float64)
        if ref.size != 4 ** k:
            raise ValueError(
                f"reference_freqs has {ref.size} entries, expected {4 ** k} for k={k}"
            )
        total = ref.sum()
        if total <= 0:
            raise ValueError("reference_freqs must contain positive mass")
        self.reference = ref / total + 1e-9
        self.vocab = vocab
        self.k = int(k)
        self.strength = float(strength)
        self.tolerance = float(tolerance)
        self.powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
        self.reset(1)

    def reset(self, batch_size: int) -> None:
        self.history: list[list[int]] = [[] for _ in range(batch_size)]
        self.counts = np.zeros((batch_size, 4 ** self.k), dtype=np.float64)
        self.total = np.zeros(batch_size, dtype=np.float64)

    def process(self, logits):
        for b, hist in enumerate(self.history):
            if len(hist) < self.k - 1:
                continue
            context = hist[-(self.k - 1):] if self.k > 1 else []
            prefix = int(np.dot(context, self.powers[1:])) if context else 0
            denom = self.total[b] + 1e-9
            for candidate in range(4):
                code = prefix + candidate * int(self.powers[0])
                observed = (self.counts[b, code] + 1e-9) / denom
                excess = observed / self.reference[code] - self.tolerance
                if excess > 0:
                    logits[b, self.vocab.token_ids[candidate]] -= (
                        self.strength * float(np.tanh(excess))
                    )
        return logits

    def record(self, base_indices: np.ndarray) -> None:
        for b, base in enumerate(base_indices):
            base = int(base)
            if base < 0:
                continue
            hist = self.history[b]
            hist.append(base)
            if len(hist) >= self.k:
                code = int(np.dot(hist[-self.k:], self.powers))
                self.counts[b, code] += 1
                self.total[b] += 1
            # Only the last k-1 bases are ever needed again.
            if len(hist) > self.k:
                del hist[:-self.k]


class ConstraintStack(DecodingConstraint):
    """Apply several constraints in sequence, sharing one token decode."""

    def __init__(self, constraints: list[DecodingConstraint],
                 vocab: NucleotideVocab | None = None) -> None:
        if not constraints:
            raise ValueError("ConstraintStack needs at least one constraint")
        self.constraints = constraints
        self.vocab = vocab or getattr(constraints[0], "vocab", None)
        if self.vocab is None:
            raise ValueError("no vocabulary available; pass vocab=")

    def reset(self, batch_size: int) -> None:
        for c in self.constraints:
            c.reset(batch_size)

    def process(self, logits):
        for c in self.constraints:
            logits = c.process(logits)
        return logits

    def record(self, base_indices: np.ndarray) -> None:
        for c in self.constraints:
            c.record(base_indices)

    def record_tokens(self, tokens) -> None:
        """Convenience: decode model token ids to base indices, then record."""
        arr = np.asarray(_to_numpy(tokens)).ravel()
        bases = np.array(
            [self.vocab.token_to_base.get(int(t), -1) for t in arr],
            dtype=np.int64,
        )
        self.record(bases)


def _to_numpy(x):
    """Detach a torch tensor to numpy; pass numpy through untouched."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Reference distribution
# ---------------------------------------------------------------------------

def reference_kmer_freqs(sequences, k: int = 4) -> np.ndarray:
    """k-mer frequency vector over one or more natural sequences.

    Use the natural windows you are trying to imitate, not the model's own
    training corpus average -- the constraint is only as good as its reference.
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    counts = np.zeros(4 ** k, dtype=np.float64)
    lookup = np.full(256, -1, dtype=np.int64)
    for i, base in enumerate(BASES):
        lookup[ord(base)] = lookup[ord(base.lower())] = i
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)

    for seq in sequences:
        idx = lookup[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
        if idx.size < k:
            continue
        win = np.lib.stride_tricks.sliding_window_view(idx, k)
        valid = (win >= 0).all(axis=1)
        if not valid.any():
            continue
        codes = (win[valid] * powers).sum(axis=1)
        counts += np.bincount(codes, minlength=4 ** k)

    total = counts.sum()
    if total == 0:
        raise ValueError("no valid k-mers in the reference sequences")
    return counts / total


# ---------------------------------------------------------------------------
# Evo 2 adapter
# ---------------------------------------------------------------------------

class evo2_constrained:
    """Context manager patching Evo 2's sampler to apply a constraint stack.

    Evo 2 samples through ``vortex.model.generation.sample``. Wrapping that one
    function keeps the fast cached-generation loop intact, so throughput is
    close to unconstrained generation.

        vocab = NucleotideVocab.from_tokenizer(model.tokenizer)
        stack = ConstraintStack([...])
        stack.reset(batch_size)
        with evo2_constrained(stack):
            output = model.generate(...)

    Note that Evo 2 produces tensors under ``inference_mode``, which cannot be
    modified in place; the patched sampler clones the logits first.
    """

    def __init__(self, stack: ConstraintStack) -> None:
        self.stack = stack
        self._module = None
        self._original = None

    def __enter__(self) -> "evo2_constrained":
        import vortex.model.generation as generation

        self._module = generation
        self._original = generation.sample
        original = self._original
        stack = self.stack

        def patched(logits, top_k=1, top_p=0.0, temperature=1.0):
            logits = logits.clone()
            logits = stack.process(logits)
            tokens = original(logits, top_k=top_k, top_p=top_p,
                              temperature=temperature)
            stack.record_tokens(tokens)
            return tokens

        generation.sample = patched
        return self

    def __exit__(self, *exc) -> None:
        if self._module is not None and self._original is not None:
            self._module.sample = self._original
        return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Check each constraint does what it claims on a controllable model.

    The 'model' is a fixed logit vector, so any change in the output is caused
    by the constraints alone.
    """
    rng = np.random.default_rng(0)
    vocab = NucleotideVocab.identity()
    failures = []

    def generate(stack, n=4000, bias=None):
        """Sample n bases from a fixed distribution, optionally constrained."""
        base_logits = np.zeros(4) if bias is None else np.array(bias, float)
        if stack is not None:
            stack.reset(1)
        out = []
        for _ in range(n):
            logits = base_logits.copy().reshape(1, 4)
            if stack is not None:
                logits = stack.process(logits)
            p = np.exp(logits[0] - logits[0].max())
            p /= p.sum()
            token = int(rng.choice(4, p=p))
            out.append(token)
            if stack is not None:
                stack.record(np.array([token]))
        return np.array(out)

    def homopolymer_frac(idx, min_run=5):
        if idx.size < 2:
            return 0.0
        change = np.flatnonzero(idx[1:] != idx[:-1]) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [idx.size]])
        lengths = ends - starts
        return float(lengths[lengths >= min_run].sum() / idx.size)

    # -- 1. Homopolymer penalty reduces homopolymer content -----------------
    # A model heavily biased toward 'A' produces long poly-A runs.
    bias = [3.0, 0.0, 0.0, 0.0]
    plain = generate(None, bias=bias)
    constrained = generate(
        ConstraintStack([HomopolymerPenalty(vocab, strength=4.0)]), bias=bias
    )
    hp_plain = homopolymer_frac(plain)
    hp_con = homopolymer_frac(constrained)
    print(f"  homopolymer fraction: unconstrained={hp_plain:.4f} "
          f"constrained={hp_con:.4f}")
    if not hp_con < hp_plain * 0.5:
        failures.append("HomopolymerPenalty did not halve homopolymer content")

    # -- 2. k-mer penalty pulls composition toward the reference -----------
    # Reference is uniform; the model is biased, so the penalty should pull
    # the generated 1-mer distribution back toward uniform.
    ref = np.ones(4) / 4
    plain2 = generate(None, bias=bias)
    con2 = generate(
        ConstraintStack([KmerOverrepresentationPenalty(
            vocab, ref, k=1, strength=4.0, tolerance=1.0)]),
        bias=bias,
    )
    dev = lambda x: float(np.abs(np.bincount(x, minlength=4) / x.size - 0.25).sum())
    print(f"  deviation from reference: unconstrained={dev(plain2):.4f} "
          f"constrained={dev(con2):.4f}")
    if not dev(con2) < dev(plain2):
        failures.append("KmerOverrepresentationPenalty did not reduce deviation")

    # -- 3. An unconstrained stack must not change the logits --------------
    logits = rng.normal(size=(3, 4))
    stack = ConstraintStack([HomopolymerPenalty(vocab)])
    stack.reset(3)
    if not np.allclose(stack.process(logits.copy()), logits):
        failures.append("penalty applied before any base was recorded")

    # -- 4. Batch elements must stay independent ---------------------------
    stack = ConstraintStack([HomopolymerPenalty(vocab, strength=5.0, cap=2)])
    stack.reset(2)
    for _ in range(4):
        stack.record(np.array([0, 1]))          # row 0 runs A, row 1 runs C
    out = stack.process(np.zeros((2, 4)))
    if not (out[0, 0] < -1 and out[1, 1] < -1 and out[0, 1] == 0 and out[1, 0] == 0):
        failures.append(f"batch rows are not independent: {out}")

    # -- 5. reference_kmer_freqs is a proper distribution -------------------
    seq = "".join(rng.choice(list(BASES), 5000))
    freqs = reference_kmer_freqs(seq, k=3)
    if not (abs(freqs.sum() - 1.0) < 1e-9 and freqs.size == 64):
        failures.append("reference_kmer_freqs is not a normalised 4**k vector")

    # -- 6. Vocab guards ----------------------------------------------------
    for bad in ([0, 1, 2], [0, 0, 1, 2]):
        try:
            NucleotideVocab(bad)
        except ValueError:
            pass
        else:
            failures.append(f"NucleotideVocab accepted invalid ids {bad}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall self-tests passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="Verify the constraints behave as documented.")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
