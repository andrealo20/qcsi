#!/usr/bin/env python3
"""
Generate the parity vectors that tie the C pipeline to the Python reference.

The two implementations are supposed to compute the same feature vector. That
claim was previously made only in docstrings, and it was wrong in two places at
once: the phase statistics were emitted in a different order, and the Doppler
bins were log-compressed on one side and not on the other. Nothing failed,
because nothing compared them.

This script produces what does compare them. It needs no dataset, so the test
it feeds runs in CI on every commit:

  1. deterministic pseudo-random CSI frames, shaped the way the C consumes
     them, with a static per-subcarrier gain and a small modulation at a
     different Doppler rate per window: the shape of real CSI, not white noise;
  2. the reference feature vector for each window, from tools/qcsi_data.py,
     converted into the C's units;
  3. a small linear model, its scores, and the class it predicts.

Everything is written to tests/vectors/parity.h, which tests/test_parity.c
includes.

## Units

The reference works in the units of whatever it is fed; the C works in Q15.
Feeding the reference the Q15 integers themselves is what makes the two
comparable, and then only two conversions are left:

  phase     radians -> binary angular measure, times 32768/pi
  Doppler   nats -> Q10, times 1024

The Doppler case is the one that needs care, because a logarithm is not
linear: log(a*x) is not log(x) plus anything the classifier can absorb. The C
therefore undoes the 1/n_fft scaling its fixed-point FFT applies before
compressing, which puts its argument on the same scale as the reference's.

## Tolerances

They are what the test is worth, so they are set from the measured
disagreement rather than chosen to be comfortable. The dominant terms are the
CORDIC magnitude error on the amplitudes, about 5 LSB, and the rounding in the
Q15 FFT, which is a fraction of a percent of the power and therefore a
fraction of a percent of a nat. The maxima over these vectors, which
tests/test_parity.c prints on every run, are 4 LSB on the amplitude block, 2
on the phase block and 19 on the Doppler block. The tolerances below sit
between three and four times that: enough room for a compiler or a qdsp
revision to move a rounding, and nowhere near enough for a reordered or
uncompressed feature, since either of the two original bugs is off by
thousands of LSB rather than tens.

Usage:
    python3 tools/gen_parity_vectors.py

Requires: numpy.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcsi_data import extract_features   # noqa: E402

# Small enough for a committed header, large enough that every stage does
# something: four windows, each with its own Doppler rate inside the band the
# feature vector keeps.
N_SUB = 12
N_ANT = 3
N_FRAMES = 16
N_FFT = 16
N_DOPPLER = 4
N_WINDOWS = 4
N_CLASSES = 5
ANT_A = 0
ANT_B = 1
SEED = 20240917

N_FEATURES = N_SUB * 5 + N_DOPPLER

TOL_AMPLITUDE = 16
TOL_PHASE = 8
TOL_DOPPLER = 64

BAM_PER_RADIAN = 32768.0 / np.pi
Q10 = 1024.0

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "tests", "vectors", "parity.h")


def synth_frames(rng):
    """Deterministic CSI frames, as (re, im) integer arrays.

    Shaped (windows, frames, subcarriers, antennas), which is the layout
    tools/qcsi_data.py expects. The magnitude is a large static per-subcarrier
    gain times a small modulation, and the phase carries an offset and a slope
    common to every antenna, which is what the conjugate product exists to
    remove. Magnitudes stay below 0.85 of full scale so nothing saturates and
    the comparison measures rounding rather than clipping.
    """
    shape = (N_WINDOWS, N_FRAMES, N_SUB, N_ANT)
    t = np.arange(N_FRAMES)[None, :, None, None]
    k = np.arange(N_SUB)[None, None, :, None]

    base = rng.uniform(0.30, 0.80, size=(N_WINDOWS, 1, N_SUB, N_ANT))
    rate = np.arange(1, N_WINDOWS + 1, dtype=float)[:, None, None, None]
    mod = 1.0 + 0.12 * np.sin(2.0 * np.pi * rate * t / N_FRAMES)
    mag = (base * mod + rng.normal(0.0, 0.004, size=shape)) * 32768.0

    phase = rng.uniform(-np.pi, np.pi, size=(N_WINDOWS, 1, 1, N_ANT))
    phase = phase + rng.uniform(-0.4, 0.4, size=shape)
    phase = phase + 1.7 * np.sin(0.31 * t) + 0.05 * np.cos(0.17 * t) * k

    z = mag * np.exp(1j * phase)
    re = np.clip(np.round(z.real), -32768, 32767).astype(np.int64)
    im = np.clip(np.round(z.imag), -32768, 32767).astype(np.int64)
    return re, im


def reference_features(re, im):
    """The reference feature vector, in the C's units, rounded to Q15."""
    csi = re.astype(np.float64) + 1j * im.astype(np.float64)
    f = extract_features(csi, ANT_A, ANT_B, n_fft=N_FFT, n_doppler=N_DOPPLER)

    amp_end = N_SUB * 3
    phase_end = amp_end + 2 * N_SUB
    f[:, amp_end:phase_end] *= BAM_PER_RADIAN
    f[:, phase_end:] *= Q10

    r = np.round(f)
    if np.max(np.abs(r)) > 32767:
        raise SystemExit("a reference feature does not fit in Q15; the C would "
                         "saturate it and the comparison would be meaningless")
    return f, r.astype(np.int64)


def feature_tolerances():
    """Per-feature tolerance, in the units of that feature."""
    tol = np.empty(N_FEATURES, dtype=np.int64)
    amp_end = N_SUB * 3
    phase_end = amp_end + 2 * N_SUB
    tol[:amp_end] = TOL_AMPLITUDE
    tol[amp_end:phase_end] = TOL_PHASE
    tol[phase_end:] = TOL_DOPPLER
    return tol


def synth_model(rng, feats):
    """A small linear model whose prediction the tolerances cannot change.

    A parity test that asserted only "the features are close" would pass a
    pipeline whose output was close and useless. Asserting the predicted class
    is the claim worth making, and it is only worth making if the margin is
    provably wider than the feature tolerance can move a score.

    The bound is exact rather than statistical: a score can move by at most
    sum(|w| * tolerance) over the features, so a margin above twice that
    cannot flip, whatever the C rounds differently. Candidate models are drawn
    until one clears the bound by a factor of eight and predicts at least
    three distinct classes: a model that answered the same class for every
    window would be satisfied by a pipeline that ignored its input.
    """
    tol = feature_tolerances()

    for _ in range(4096):
        weights = rng.integers(-4000, 4001, size=(N_CLASSES, N_FEATURES))
        bias = rng.integers(-2 * 10 ** 8, 2 * 10 ** 8 + 1, size=N_CLASSES)

        bound = int(np.max(np.abs(weights) @ tol))
        scores = feats @ weights.T + bias[None, :]
        order = np.sort(scores, axis=1)
        margin = order[:, -1] - order[:, -2]
        pred = np.argmax(scores, axis=1)
        if int(np.min(margin)) > 8 * bound and len(np.unique(pred)) >= 3:
            return weights, bias, scores, pred, margin, bound

    raise SystemExit("no model with a wide enough margin was found; widen the "
                     "weight range or lower the tolerances")


def emit_int_array(out, ctype, name, size, values, per_line, width):
    out.append(f"static const {ctype} {name}[{size}] = {{")
    vals = list(values)
    for i in range(0, len(vals), per_line):
        row = ", ".join(f"{v:{width}d}" for v in vals[i:i + per_line])
        out.append(f"    {row},")
    out.append("};")
    out.append("")


def emit(path, re, im, feats_q15, weights, bias, pred, margin, bound):
    frames = np.empty((N_WINDOWS, N_FRAMES, N_ANT, N_SUB, 2), dtype=np.int64)
    # The C reads frame[a * n_sub + k], so the antenna axis comes first.
    frames[..., 0] = np.transpose(re, (0, 1, 3, 2))
    frames[..., 1] = np.transpose(im, (0, 1, 3, 2))

    out = []
    out.append("/* Generated by tools/gen_parity_vectors.py - "
               "do not edit by hand. */")
    out.append("#ifndef QCSI_TEST_VECTORS_PARITY_H")
    out.append("#define QCSI_TEST_VECTORS_PARITY_H")
    out.append("")
    out.append("#include <stdint.h>")
    out.append("")
    out.append("/*")
    out.append(" * Frames, expected features, model and predictions for the")
    out.append(" * parity test between the C pipeline and the floating-point")
    out.append(" * reference in tools/qcsi_data.py.")
    out.append(" *")
    out.append(" * Features are in the C's units: amplitude statistics in Q15,")
    out.append(" * phase statistics in binary angular measure, Doppler bins as")
    out.append(" * log(1 + power) in Q10. The tolerances are per block because")
    out.append(" * the three blocks fail differently: the amplitude block")
    out.append(" * carries the CORDIC magnitude error, the Doppler block")
    out.append(" * carries the Q15 FFT's rounding through a logarithm.")
    out.append(" */")
    out.append("")
    for name, value in (("PARITY_N_SUB", N_SUB),
                        ("PARITY_N_ANT", N_ANT),
                        ("PARITY_N_FRAMES", N_FRAMES),
                        ("PARITY_N_FFT", N_FFT),
                        ("PARITY_N_DOPPLER", N_DOPPLER),
                        ("PARITY_ANT_A", ANT_A),
                        ("PARITY_ANT_B", ANT_B),
                        ("PARITY_N_WINDOWS", N_WINDOWS),
                        ("PARITY_N_CLASSES", N_CLASSES),
                        ("PARITY_N_FEATURES", N_FEATURES)):
        out.append(f"#define {name:<20} {value}")
    out.append("")
    out.append("/* Per-block tolerance, in the units of that block. */")
    for name, value in (("PARITY_TOL_AMPLITUDE", TOL_AMPLITUDE),
                        ("PARITY_TOL_PHASE", TOL_PHASE),
                        ("PARITY_TOL_DOPPLER", TOL_DOPPLER)):
        out.append(f"#define {name:<20} {value}")
    out.append("")
    out.append("/* Largest amount those tolerances can move a class score:")
    out.append("   sum(|weight| * tolerance) over the feature vector. A margin")
    out.append("   above twice this cannot be flipped by the difference")
    out.append("   between the two implementations, whichever way it rounds. */")
    out.append(f"#define PARITY_SCORE_BOUND   INT64_C({bound})")
    out.append("")

    emit_int_array(out, "int16_t", "parity_frames",
                   "PARITY_N_WINDOWS * PARITY_N_FRAMES * PARITY_N_ANT * "
                   "PARITY_N_SUB * 2",
                   frames.reshape(-1), 8, 7)
    emit_int_array(out, "int16_t", "parity_features",
                   "PARITY_N_WINDOWS * PARITY_N_FEATURES",
                   feats_q15.reshape(-1), 8, 7)
    emit_int_array(out, "int16_t", "parity_weights",
                   "PARITY_N_CLASSES * PARITY_N_FEATURES",
                   weights.reshape(-1), 8, 7)
    out.append("static const int64_t parity_bias[PARITY_N_CLASSES] = {")
    out.append("    " + ", ".join(f"INT64_C({int(v)})" for v in bias) + ",")
    out.append("};")
    out.append("")
    emit_int_array(out, "int16_t", "parity_prediction", "PARITY_N_WINDOWS",
                   pred, 8, 4)
    out.append("static const int64_t parity_margin[PARITY_N_WINDOWS] = {")
    out.append("    " + ", ".join(f"INT64_C({int(v)})" for v in margin) + ",")
    out.append("};")
    out.append("")
    out.append("#endif /* QCSI_TEST_VECTORS_PARITY_H */")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    re, im = synth_frames(rng)
    feats, feats_q15 = reference_features(re, im)
    weights, bias, scores, pred, margin, bound = synth_model(rng, feats_q15)

    emit(args.out, re, im, feats_q15, weights, bias, pred, margin, bound)

    amp_end = N_SUB * 3
    phase_end = amp_end + 2 * N_SUB
    print(f"wrote {os.path.normpath(args.out)}")
    print(f"  {N_WINDOWS} windows of {N_FRAMES} frames, "
          f"{N_SUB} subcarriers x {N_ANT} antennas, "
          f"{N_FEATURES} features")
    print(f"  amplitude block |value| <= {np.max(np.abs(feats[:, :amp_end])):.0f}"
          f"   tolerance {TOL_AMPLITUDE}")
    print(f"  phase block     |value| <= "
          f"{np.max(np.abs(feats[:, amp_end:phase_end])):.0f}"
          f"   tolerance {TOL_PHASE}")
    print(f"  Doppler block   |value| <= {np.max(np.abs(feats[:, phase_end:])):.0f}"
          f"   tolerance {TOL_DOPPLER}")
    print(f"  predictions {list(int(p) for p in pred)}")
    print(f"  smallest margin {int(np.min(margin))}, "
          f"score bound {bound}, ratio {int(np.min(margin)) / bound:.1f}x")
    print("\nThe ratio is what the test rests on: below 2 the tolerances could "
          "flip a prediction and the class assertion would be measuring luck.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
