#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
structural_losses.py

Auxiliary training losses that penalise the failure modes the benchmarks find.

This is the training-time counterpart to ``constrained_decoding.py``. Rather
than biasing logits at sampling time, these terms are added to the training
objective so the model learns not to produce the offending structure in the
first place.

Both losses are differentiable functions of the model's next-token
distribution, so they drop into any autoregressive DNA training loop:

    logits, _ = model(inputs)                       # (B, T, 4)
    loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
    loss = loss + 0.5 * homopolymer_loss(logits, inputs)
    loss = loss + 1.0 * dinucleotide_kl_loss(logits, inputs, reference_dinuc)

Both expect a 4-symbol vocabulary ordered A, C, G, T. Weights in the 0.1-1.0
range are a reasonable starting point: large enough to move the statistic,
small enough that cross-entropy still dominates. Watch validation perplexity --
if it degrades noticeably the weight is too high, and you are trading away
sequence modelling for a cosmetic improvement in one statistic.

What to expect
--------------
These terms shape *marginal* statistics. They reliably move the property they
target, but a model that produces natural-looking mononucleotide and
dinucleotide statistics can still be trivially separable from natural sequence
by a k-mer classifier. Measure with ``detectability.py`` and
``natural_baseline.py``, not with the training loss.

Self-test
---------
    python generation/improve/structural_losses.py --self-test
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

__all__ = [
    "homopolymer_loss",
    "dinucleotide_kl_loss",
    "reference_dinucleotide",
]


def homopolymer_loss(logits: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Mean probability the model assigns to repeating the current base.

    Parameters
    ----------
    logits : (B, T, 4) next-token logits.
    inputs : (B, T) base indices the logits are conditioned on -- that is, the
        base at each position, whose repetition we want to discourage.

    Returns a scalar in [0, 1]. Minimising it pushes probability mass off the
    "same base again" transition, which is what produces homopolymer runs.
    Note this penalises *all* repetition, including the modest amount natural
    sequence contains, so pair it with a weight small enough to leave the
    natural rate intact.
    """
    _check(logits, inputs)
    probs = F.softmax(logits, dim=-1)
    p_same = probs.gather(-1, inputs.unsqueeze(-1)).squeeze(-1)
    return p_same.mean()


def dinucleotide_kl_loss(
    logits: torch.Tensor,
    inputs: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """KL(reference || model) between dinucleotide distributions.

    The model's expected dinucleotide distribution is formed by pairing each
    conditioning base with the full predicted distribution over the next base,
    so the term is differentiable without sampling:

        expected[a, b] = sum over positions of  1[input = a] * P(next = b)

    Parameters
    ----------
    logits : (B, T, 4) next-token logits.
    inputs : (B, T) conditioning base indices.
    reference : (4, 4) natural dinucleotide distribution, rows = current base,
        columns = next base. Use :func:`reference_dinucleotide`.

    The KL is taken in the ``KL(natural || model)`` direction, which is
    mode-covering: it heavily punishes assigning near-zero probability to a
    dinucleotide that is common in natural sequence. That is the right
    direction for CpG, the usual casualty in vertebrate genome generation.
    """
    _check(logits, inputs)
    if reference.shape != (4, 4):
        raise ValueError(f"reference must be (4, 4), got {tuple(reference.shape)}")

    probs = F.softmax(logits, dim=-1)                      # (B, T, 4)
    onehot = F.one_hot(inputs, 4).to(probs.dtype)          # (B, T, 4)
    expected = torch.einsum("bti,btj->ij", onehot, probs)  # (4, 4)
    expected = expected / expected.sum().clamp_min(eps)

    ref = reference.to(probs.dtype).to(probs.device)
    ref = ref / ref.sum().clamp_min(eps)
    ref = ref.clamp_min(eps)
    model_p = expected.clamp_min(eps)
    return (ref * (ref.log() - model_p.log())).sum()


def reference_dinucleotide(sequences, device=None) -> torch.Tensor:
    """Natural (4, 4) dinucleotide distribution from DNA strings.

    Rows are the current base, columns the next base, and the whole matrix sums
    to 1. Dinucleotides containing a non-ACGT base are skipped.
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    counts = torch.zeros(4, 4, dtype=torch.float64)
    table = {b: i for i, b in enumerate("ACGT")}
    for seq in sequences:
        prev = -1
        for char in seq.upper():
            cur = table.get(char, -1)
            if prev >= 0 and cur >= 0:
                counts[prev, cur] += 1
            prev = cur
    total = counts.sum()
    if total == 0:
        raise ValueError("no valid dinucleotides in the reference sequences")
    out = (counts / total).to(torch.float32)
    return out.to(device) if device is not None else out


def _check(logits: torch.Tensor, inputs: torch.Tensor) -> None:
    if logits.dim() != 3 or logits.shape[-1] != 4:
        raise ValueError(f"logits must be (B, T, 4), got {tuple(logits.shape)}")
    if inputs.shape != logits.shape[:2]:
        raise ValueError(
            f"inputs must be (B, T) matching logits, got {tuple(inputs.shape)} "
            f"vs {tuple(logits.shape[:2])}"
        )
    if inputs.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"inputs must be integer base indices, got {inputs.dtype}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    torch.manual_seed(0)
    failures = []
    b, t = 4, 64
    inputs = torch.randint(0, 4, (b, t))

    # 1. homopolymer_loss is maximal when the model always repeats.
    repeat = F.one_hot(inputs, 4).float() * 20.0
    avoid = (1.0 - F.one_hot(inputs, 4).float()) * 20.0
    uniform = torch.zeros(b, t, 4)
    hl_repeat = homopolymer_loss(repeat, inputs).item()
    hl_avoid = homopolymer_loss(avoid, inputs).item()
    hl_uniform = homopolymer_loss(uniform, inputs).item()
    print(f"  homopolymer_loss: always-repeat={hl_repeat:.4f} "
          f"uniform={hl_uniform:.4f} never-repeat={hl_avoid:.4f}")
    if not (hl_repeat > 0.99 and abs(hl_uniform - 0.25) < 1e-5 and hl_avoid < 0.01):
        failures.append("homopolymer_loss does not span [0, 1] as documented")

    # 2. dinucleotide_kl_loss is ~0 when the model matches the reference.
    ref = reference_dinucleotide("ACGTACGTTTACGGCATTACGATCGATTTACG" * 40)
    # Build logits whose expected dinucleotide distribution equals the
    # reference conditional, so the joint matches whenever inputs are drawn
    # from the reference marginal.
    cond = (ref / ref.sum(dim=1, keepdim=True).clamp_min(1e-9)).log()
    matched = cond[inputs]                                   # (B, T, 4)
    kl_matched = dinucleotide_kl_loss(matched, inputs, ref).item()
    kl_uniform = dinucleotide_kl_loss(uniform, inputs, ref).item()
    print(f"  dinucleotide_kl_loss: matched={kl_matched:.4f} "
          f"uniform-model={kl_uniform:.4f}")
    if not kl_matched < kl_uniform:
        failures.append("dinucleotide_kl_loss does not favour the matched model")
    if kl_matched < 0:
        failures.append("dinucleotide_kl_loss returned a negative KL")

    # 3. Both losses are differentiable.
    logits = torch.zeros(b, t, 4, requires_grad=True)
    (homopolymer_loss(logits, inputs)
     + dinucleotide_kl_loss(logits, inputs, ref)).backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        failures.append("gradients are missing or non-finite")

    # 4. reference_dinucleotide is a normalised 4x4 joint.
    if not (ref.shape == (4, 4) and abs(ref.sum().item() - 1.0) < 1e-6):
        failures.append("reference_dinucleotide is not a normalised (4, 4) joint")

    # 5. Shape guards fire.
    for bad_logits, bad_inputs in (
        (torch.zeros(b, t, 5), inputs),
        (torch.zeros(b, t, 4), torch.randint(0, 4, (b, t + 1))),
        (torch.zeros(b, t, 4), torch.rand(b, t)),
    ):
        try:
            homopolymer_loss(bad_logits, bad_inputs)
        except ValueError:
            pass
        else:
            failures.append("a shape/dtype guard did not fire")

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
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
