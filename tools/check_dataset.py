#!/usr/bin/env python3
"""
Check whether a CSI capture is usable by the qcsi phase front end.

Run this before writing any code against a new dataset. It answers four
questions, in increasing order of importance:

  1. Is the CSI complex, i.e. is there any phase at all?
  2. Are the antennas balanced enough for a phase difference to mean
     anything? (See the note below - this one was learned the hard way.)
  3. Is the phase RAW, or has it been sanitised upstream? Several widely
     used datasets ship phase that has already been unwrapped and
     detrended, which leaves this front end with nothing to remove.
  4. Does the conjugate product between two antennas actually stabilise the
     phase on THIS data? That is the premise qcsi rests on. It holds by
     construction on synthetic signals; whether it holds on a given real
     receiver is an empirical question, and this is the answer.

All statistics are computed WITHIN a single continuous capture and then
summarised across captures. Pooling separate captures gives a false
negative - see the note on saturation in the code below.

## Why the antenna balance check exists

The first version of this script did not have it, and on a real Intel 5300
capture it reported that the conjugate product barely helped and actually
made the residual slope five times worse. The premise looked wrong.

The premise was fine; the capture was not. Mean amplitudes across the three
receive antennas were 14.3, 1.3 and 1.4 - the second and third antennas were
around 20 dB down, so their phase was mostly noise. Differencing against a
noisy reference adds noise instead of removing an offset. Picking the two
antennas with comparable strength (rx0 and rx2) instead gave a 3.1x offset
reduction and a 1.8x slope reduction, in the right direction.

The lesson generalises: the conjugate product cancels what is COMMON, and
two antennas only share a common term if both are actually receiving. Check
the balance before trusting the result.

Usage:
    python3 tools/check_dataset.py capture.dat          # Intel 5300 raw
    python3 tools/check_dataset.py capture.mat          # SignFi-style
    python3 tools/check_dataset.py input_walk_1.csv     # UT-HAR style

Requires: numpy. Also csiread for .dat, scipy for .mat.
"""

import os
import sys

import numpy as np


def load_dat(path):
    """Intel 5300 raw capture -> (packets, subcarriers, antennas) complex."""
    try:
        import csiread
    except ImportError:
        raise SystemExit("reading .dat needs csiread:  pip install csiread")
    d = csiread.Intel(path, nrxnum=3, ntxnum=2, pl_size=10)
    d.read()
    csi = d.get_scaled_csi()          # (packets, sub, rx, tx)
    return csi[:, :, :, 0]            # first transmit stream


def load_mat(path):
    """SignFi-style .mat -> (packets, subcarriers, antennas) complex."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise SystemExit("reading .mat needs scipy:  pip install scipy")
    mat = loadmat(path)
    arrays = [(k, v) for k, v in mat.items()
              if isinstance(v, np.ndarray) and np.iscomplexobj(v)]
    if not arrays:
        real = [k for k, v in mat.items()
                if isinstance(v, np.ndarray) and v.ndim >= 2
                and not k.startswith("__")]
        raise SystemExit(
            "no complex array in this .mat file.\n"
            f"        real arrays present: {real}\n"
            "        If the phase was stripped upstream, this capture cannot\n"
            "        exercise the phase front end.")
    name, arr = max(arrays, key=lambda kv: kv[1].size)
    print(f"array     : '{name}' {arr.shape} {arr.dtype}")
    arr = np.squeeze(arr)

    # Axes are identified by what their lengths can plausibly mean rather
    # than by position, because datasets order them differently and getting
    # this wrong silently produces confident nonsense.
    #
    #   antennas    : a small axis, at most 8
    #   subcarriers : one of the counts the CSI tools actually report
    #   instances   : present in segmented datasets like SignFi, where each
    #                 instance is a separately captured gesture
    #   packets     : whatever is left
    SUBCARRIER_COUNTS = (30, 52, 56, 64, 114, 128, 234, 256)

    shape = arr.shape
    ax_sub = next((i for i, n in enumerate(shape) if n in SUBCARRIER_COUNTS), None)
    ax_ant = next((i for i, n in enumerate(shape)
                   if n <= 8 and i != ax_sub), None)
    if ax_sub is None or ax_ant is None:
        raise SystemExit(
            f"cannot identify the subcarrier and antenna axes in {shape}.\n"
            "        Pass the file through a loader of your own, or extend\n"
            "        SUBCARRIER_COUNTS in this script.")

    rest = [i for i in range(arr.ndim) if i not in (ax_sub, ax_ant)]

    if len(rest) == 2:
        # Segmented dataset: the larger remaining axis is the instance count.
        ax_inst = max(rest, key=lambda i: shape[i])
        ax_pkt = [i for i in rest if i != ax_inst][0]
        n_inst, n_pkt = shape[ax_inst], shape[ax_pkt]
        print(f"layout    : {n_inst} instances x {n_pkt} packets x "
              f"{shape[ax_sub]} subcarriers x {shape[ax_ant]} antennas")

        # Only a subset of instances is loaded. The impairment this script
        # measures varies packet to packet, so a few hundred instances are
        # plenty, and holding the whole array costs hundreds of megabytes.
        keep = min(n_inst, 200)
        arr = np.transpose(arr, (ax_inst, ax_pkt, ax_sub, ax_ant))
        arr = arr[:keep]
        if keep < n_inst:
            print(f"            using the first {keep} instances")
        # The instance axis is kept, not flattened. See the note on
        # within-instance measurement below: merging instances destroys the
        # very quantity this script is trying to measure.
    elif len(rest) == 1:
        arr = np.transpose(arr, (rest[0], ax_sub, ax_ant))
    else:
        raise SystemExit(f"unexpected number of axes in {shape}")

    return arr


def load_csv(path, max_rows=3000):
    """UT-HAR style CSV: timestamp, 90 amplitudes, 90 phases."""
    data = np.loadtxt(path, delimiter=",", max_rows=max_rows)
    if data.ndim == 1:
        data = data[None, :]
    n_sub, n_ant = 30, 3
    need = 1 + 2 * n_sub * n_ant
    if data.shape[1] < need:
        raise SystemExit(f"expected >= {need} columns, found {data.shape[1]}")
    frames = data.shape[0]
    amp = data[:, 1:1 + n_sub * n_ant].reshape(frames, n_ant, n_sub)
    pha = data[:, 1 + n_sub * n_ant:need].reshape(frames, n_ant, n_sub)
    return (amp * np.exp(1j * pha)).transpose(0, 2, 1)


UNIFORM_STD = float(np.pi / np.sqrt(3.0))   # std of a phase spread over the
                                            # whole circle: 1.814 rad


def circular_std(angles):
    """Standard deviation of angles, computed on the circle.

    An ordinary std is wrong for phase: values near +pi and -pi are close
    together but the arithmetic treats them as maximally distant. This uses
    the resultant length instead, which has no seam.
    """
    r = np.abs(np.mean(np.exp(1j * angles)))
    r = min(max(r, 1e-12), 1.0 - 1e-12)
    return float(np.sqrt(-2.0 * np.log(r)))


def stats(phase_2d):
    """Offset spread at a middle subcarrier, and spread of per-packet slope.

    Measured over one group of packets that belong together, a single
    continuous capture. Merging separate captures is what the first version
    of this script did, and it produced a false negative: see below.
    """
    n_sub = phase_2d.shape[1]
    offset = circular_std(phase_2d[:, n_sub // 2])
    slopes = np.polyfit(np.arange(n_sub),
                        np.unwrap(phase_2d, axis=1).T, 1)[0]
    return offset, float(np.std(slopes))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    path = sys.argv[1]
    ext = os.path.splitext(path)[1].lower()
    print(f"file      : {path}")

    if ext == ".dat":
        csi = load_dat(path)
    elif ext == ".mat":
        csi = load_mat(path)
    elif ext == ".csv":
        csi = load_csv(path)
    else:
        raise SystemExit(f"unsupported extension '{ext}' (.dat, .mat or .csv)")

    if csi.ndim == 3:
        csi = csi[None, ...]          # one group: a single continuous capture
    if csi.ndim != 4:
        raise SystemExit(f"unexpected shape {csi.shape}")
    n_grp, n_pkt, n_sub, n_ant = csi.shape
    print(f"shape     : {n_grp} capture(s) x {n_pkt} packets x "
          f"{n_sub} subcarriers x {n_ant} antennas")

    # --- 1. complex? ---------------------------------------------------
    if not np.iscomplexobj(csi):
        print("VERDICT   : amplitude only. The phase front end cannot be "
              "exercised on this capture.")
        return 1
    print("complex   : yes")

    if n_ant < 2:
        print("VERDICT   : only one antenna. The conjugate product needs two.")
        return 1

    # --- 2. antenna balance --------------------------------------------
    levels = np.abs(csi).mean(axis=(0, 1, 2))
    print("antennas  : mean amplitude " +
          ", ".join(f"rx{i}={v:.2f}" for i, v in enumerate(levels)))
    db = 20.0 * np.log10(levels / levels.max() + 1e-12)
    weak = [i for i, v in enumerate(db) if v < -10.0]
    if weak:
        print(f"            rx{weak} are more than 10 dB down; their phase is "
              "largely noise")

    # --- 3. raw phase, measured WITHIN each capture ----------------------
    #
    # Statistics are computed per capture and then summarised, never over a
    # pool of separate captures. The first version of this script pooled
    # them, and on a segmented dataset it reported that the conjugate
    # product did nothing at all: both the raw and the corrected spread came
    # out at 1.83 rad, which is 1.814 - the standard deviation of a phase
    # spread uniformly over the whole circle. The measurement had saturated,
    # so there was no room left for any method to show an improvement.
    #
    # The reason is that separate captures do not share an antenna phase
    # offset: the receiver re-runs gain control and antenna selection
    # between them. Pooling therefore measures capture-to-capture variation,
    # which the conjugate product neither can nor should remove.
    ref = int(np.argmax(levels))
    raw = np.array([stats(np.angle(csi[g, :, :, ref])) for g in range(n_grp)])
    off_raw, slope_raw = float(np.median(raw[:, 0])), float(np.median(raw[:, 1]))
    print(f"raw phase : offset std {off_raw:.3f} rad, "
          f"slope std {slope_raw:.4f} rad/subcarrier  (median over captures)")

    if off_raw > 0.95 * UNIFORM_STD:
        print(f"            at {UNIFORM_STD:.3f} the phase is uniform over the "
              "circle and the measurement is saturated")
    if off_raw < 0.2:
        print("            phase barely varies between packets: it has probably "
              "been sanitised upstream")

    # --- 4. does the conjugate product help? ----------------------------
    # Every pair is tried: which one works depends on the antenna balance,
    # and picking one blindly is how an earlier version of this script
    # reached the wrong conclusion on a different capture.
    print("conj prod :  (median over captures)")
    best = None
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            per = np.array([stats(np.angle(csi[g, :, :, i] *
                                           np.conj(csi[g, :, :, j])))
                            for g in range(n_grp)])
            off, slope = float(np.median(per[:, 0])), float(np.median(per[:, 1]))
            r_off = off_raw / off if off > 1e-9 else float("inf")
            r_slope = slope_raw / slope if slope > 1e-9 else float("inf")
            print(f"            rx{i}-rx{j}: offset {off:.3f} ({r_off:.1f}x), "
                  f"slope {slope:.4f} ({r_slope:.2f}x)")
            score = min(r_off, r_slope)
            if best is None or score > best[0]:
                best = (score, i, j, r_off, r_slope)

    _, bi, bj, r_off, r_slope = best
    print(f"best pair : rx{bi}-rx{bj}  offset {r_off:.1f}x, slope {r_slope:.2f}x")

    # --- verdict --------------------------------------------------------
    print()
    if off_raw < 0.2:
        print("VERDICT   : phase already sanitised. Use an amplitude-only "
              "pipeline, or find a capture with raw phase.")
        return 1
    if r_off >= 2.0 and r_slope >= 1.5:
        print(f"VERDICT   : usable. Use antennas {bi} and {bj}. Record these "
              "reduction factors in the README - they are the empirical "
              "evidence for the design premise.")
        return 0
    if weak:
        print("VERDICT   : inconclusive, and the antenna imbalance above is the "
              "likely reason. Try a capture where all antennas receive "
              "comparable power before concluding anything about the method.")
        return 2
    print("VERDICT   : inconclusive. The phase is raw and the antennas are "
          "balanced, yet the conjugate product does not clearly help on this "
          "capture. Worth understanding before building on it - and worth "
          "reporting either way.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
