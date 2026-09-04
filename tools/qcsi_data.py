#!/usr/bin/env python3
"""
Loading and reference processing for SignFi-style CSI datasets.

This module is the floating-point reference the C implementation is measured
against. It deliberately mirrors the structure of the C pipeline rather than
being written in the most natural NumPy style, so that a discrepancy points
at one stage instead of at the whole thing.

Run directly to inspect a .mat file without loading it fully:

    python3 tools/qcsi_data.py --inspect path/to/dataset.mat

Requires: numpy, scipy.
"""

import argparse
import sys

import numpy as np

# Subcarrier counts the common CSI tools actually report. Used to identify
# axes by meaning rather than by position, because datasets order them
# differently and a wrong guess produces confident nonsense.
SUBCARRIER_COUNTS = (30, 52, 56, 64, 114, 128, 234, 256)


def inspect(path):
    """List the variables in a .mat file without interpreting them."""
    from scipy.io import whosmat
    print(f"file: {path}\n")
    entries = whosmat(path)
    if not entries:
        print("  (no variables)")
        return
    width = max(len(name) for name, _, _ in entries)
    for name, shape, dtype in sorted(entries):
        print(f"  {name:<{width}}  {str(shape):<24} {dtype}")

    csi_vars = sorted(n for n, _, _ in entries if n.lower().startswith("csi"))
    lbl_vars = sorted(n for n, _, _ in entries if "label" in n.lower())
    print()
    if len(csi_vars) > 1:
        print(f"  {len(csi_vars)} CSI variables: {csi_vars}")
        print("  If these are one per user, they give a subject-wise split for")
        print("  free, which is the split this dataset actually needs.")
    if lbl_vars:
        print(f"  label variables: {lbl_vars}")


def identify_axes(shape):
    """Return (instance, packet, subcarrier, antenna) axis indices."""
    ax_sub = next((i for i, n in enumerate(shape) if n in SUBCARRIER_COUNTS), None)
    ax_ant = next((i for i, n in enumerate(shape) if n <= 8 and i != ax_sub), None)
    if ax_sub is None or ax_ant is None:
        raise ValueError(f"cannot identify subcarrier/antenna axes in {shape}")
    rest = [i for i in range(len(shape)) if i not in (ax_sub, ax_ant)]
    if len(rest) != 2:
        raise ValueError(f"expected 4 axes, got {shape}")
    ax_inst = max(rest, key=lambda i: shape[i])
    ax_pkt = [i for i in rest if i != ax_inst][0]
    return ax_inst, ax_pkt, ax_sub, ax_ant


def load_subjects(path, max_per_subject=None, verbose=True):
    """Load a .mat file as a list of (csi, labels) per subject.

    csi is (instances, packets, subcarriers, antennas) complex.

    If the file holds several CSI variables they are treated as separate
    subjects, which is what makes an honest subject-wise split possible. If
    it holds one, everything is one subject and the caller is told so, an
    accuracy figure from a single-subject split is not comparable to a
    subject-wise one and should not be presented as if it were.
    """
    from scipy.io import loadmat

    mat = loadmat(path)
    csi_names = sorted(k for k, v in mat.items()
                       if isinstance(v, np.ndarray) and np.iscomplexobj(v)
                       and v.size > 1000)
    if not csi_names:
        raise SystemExit("no complex CSI array found in this file")

    label_names = sorted(k for k, v in mat.items()
                         if isinstance(v, np.ndarray) and "label" in k.lower())

    if verbose:
        print(f"CSI variables   : {csi_names}")
        print(f"label variables : {label_names}")
        if len(csi_names) == 1:
            print("NOTE: a single CSI variable means a single subject. A "
                  "subject-wise split is not possible from this file alone.")

    # Load every CSI array first, because how the labels have to be split
    # depends on the total instance count across all of them.
    arrays = []
    for name in csi_names:
        arr = np.squeeze(mat[name])
        ai, ap, asb, aa = identify_axes(arr.shape)
        arrays.append(np.transpose(arr, (ai, ap, asb, aa)))

    counts = [a.shape[0] for a in arrays]
    total = sum(counts)

    # Labels come in one of two shapes, and confusing them silently
    # mislabels every subject but the first.
    #
    #   - one variable per CSI array, matched by suffix
    #   - a single variable holding every instance of every subject,
    #     concatenated in the order the CSI variables are named
    #
    # SignFi's lab_150 file is the second kind: five 1500-instance CSI
    # arrays and one 7500-entry label vector. Taking the first 1500 entries
    # for each subject, which an earlier version of this function did, gives
    # subject 2 the labels of subject 1.
    shared = None
    if len(label_names) == 1:
        cand = np.asarray(np.squeeze(mat[label_names[0]])).ravel()
        if cand.size == total:
            shared = cand
            if verbose:
                print(f"labels          : one shared vector of {total} entries, "
                      f"split {counts} across subjects")
        elif cand.size == counts[0] and len(csi_names) > 1:
            raise SystemExit(
                f"the label vector has {cand.size} entries but the CSI arrays "
                f"hold {total} instances in total.\n"
                "        Refusing to guess how they line up: mislabelled "
                "subjects would silently corrupt every later result.")

    subjects, offset = [], 0
    for idx, (name, arr) in enumerate(zip(csi_names, arrays)):
        if shared is not None:
            lbl = shared[offset:offset + arr.shape[0]]
        else:
            suffix = name[3:]
            lbl = None
            for cand_name in label_names:
                if suffix and cand_name.endswith(suffix):
                    lbl = np.asarray(np.squeeze(mat[cand_name])).ravel()
                    break
            if lbl is None and idx < len(label_names):
                lbl = np.asarray(np.squeeze(mat[label_names[idx]])).ravel()
            if lbl is None:
                raise SystemExit(f"no labels found for '{name}'")
            if lbl.size != arr.shape[0]:
                raise SystemExit(
                    f"'{name}' has {arr.shape[0]} instances but its label "
                    f"vector has {lbl.size} entries")

        offset += arr.shape[0]

        if max_per_subject is not None:
            arr = arr[:max_per_subject]
            lbl = lbl[:max_per_subject]

        subjects.append((arr, lbl))
        if verbose:
            print(f"  {name}: {arr.shape}, {len(np.unique(lbl))} classes, "
                  f"labels {int(lbl.min())}..{int(lbl.max())}")

    return subjects


# ---------------------------------------------------------------------------
# Reference pipeline: mirrors the C implementation stage by stage
# ---------------------------------------------------------------------------

def phase_difference(csi, ant_a, ant_b):
    """Phase of antenna a times the conjugate of antenna b.

    Mirrors qcsi_phase_difference(). This is the stage that cancels the
    carrier and sampling frequency offsets, which are common to both
    antennas of the same receiver.
    """
    return np.angle(csi[..., ant_a] * np.conj(csi[..., ant_b]))


def unwrap_detrend(phase):
    """Unwrap across subcarriers and remove the best-fit line.

    Mirrors qcsi_unwrap() followed by qcsi_detrend(). Any residual slope
    across subcarrier index is timing error rather than physics, and the
    constant term is arbitrary.
    """
    u = np.unwrap(phase, axis=-1)
    n_sub = u.shape[-1]
    k = np.arange(n_sub)
    flat = u.reshape(-1, n_sub)
    slope, intercept = np.polyfit(k, flat.T, 1)
    fit = slope[:, None] * k[None, :] + intercept[:, None]
    return (flat - fit).reshape(u.shape)


def amplitude_stats(amp):
    """Mean, standard deviation and peak-to-peak per subcarrier.

    Mirrors qcsi_amplitude_stats(). Standard deviation rather than variance,
    for the reason given in docs/design.md: a variance quantises as the
    square of the signal level and collapses on small CSI ripples.
    """
    return np.stack([amp.mean(axis=1), amp.std(axis=1),
                     amp.max(axis=1) - amp.min(axis=1)], axis=-1)


def doppler_power(x, n_fft):
    """Power spectrum along the packet axis, static component removed first.

    Mirrors qcsi_remove_static_component() followed by qcsi_doppler_power().
    Each subcarrier carries a large fixed gain that would otherwise dominate
    bin zero.
    """
    centred = x - x.mean(axis=1, keepdims=True)
    spec = np.fft.fft(centred, n=n_fft, axis=1)
    half = n_fft // 2
    return np.abs(spec[:, :half]) ** 2


def extract_features(csi, ant_a=0, ant_b=1, n_fft=64, n_doppler=16):
    """Feature vector per instance: amplitude statistics plus Doppler shape.

    The feature set is deliberately small and interpretable. Its purpose is
    to make the cost of every stage visible and to give the quantised C
    implementation something concrete to be compared against, not to
    maximise accuracy.
    """
    amp = np.abs(csi[..., ant_a])                    # (inst, pkt, sub)
    stats = amplitude_stats(amp)                     # (inst, sub, 3)

    # Blocked, not interleaved: every mean first, then every standard
    # deviation. That is the order build_features() in src/pipeline.c writes
    # them in, and the two have to agree or the C consumes a feature vector
    # whose second half is permuted with respect to the one the model was
    # trained on. The amplitude block above is interleaved per subcarrier in
    # both implementations, so this is the only place the two orders differ.
    #
    # Changing the order here does not change any accuracy reported in the
    # README: the classifier is linear, so permuting the features permutes the
    # learned weights the same way and every score is identical. It matters
    # only when a model trained by these tools is handed to the C.
    phase = unwrap_detrend(phase_difference(csi, ant_a, ant_b))
    phase_stats = np.concatenate([phase.mean(axis=1), phase.std(axis=1)],
                                 axis=-1)

    dop = doppler_power(amp, n_fft)                  # (inst, half, sub)
    dop = dop[:, 1:n_doppler + 1].mean(axis=2)       # average over subcarriers
    # Log compression: Doppler power spans several decades, and a linear
    # classifier on raw power would be driven entirely by the loudest bin.
    #
    # Mirrors qcsi_log1p_q10() as applied in build_features(). A logarithm is
    # not linear, so the two agree only on the same units: the C compresses
    # the power of the unnormalised transform of the Q15 amplitudes, having
    # first undone the 1/n_fft scaling its fixed-point FFT applies. Feeding
    # this function amplitudes on that same integer scale is what makes the
    # two comparable, and is what tools/gen_parity_vectors.py does.
    dop = np.log1p(dop)

    n = csi.shape[0]
    return np.concatenate([stats.reshape(n, -1),
                           phase_stats.reshape(n, -1),
                           dop.reshape(n, -1)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--inspect", action="store_true",
                    help="list variables without loading the data")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.path)
        return 0

    subjects = load_subjects(args.path, max_per_subject=50)
    print(f"\nloaded {len(subjects)} subject(s)")
    for i, (csi, lbl) in enumerate(subjects):
        feats = extract_features(csi)
        print(f"  subject {i}: features {feats.shape}, labels {lbl.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
