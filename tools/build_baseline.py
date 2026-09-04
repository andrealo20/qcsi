#!/usr/bin/env python3
"""
Build the M1 floating-point baseline and freeze the evaluation split.

This produces three things:

  1. A split, written to a JSON file and never regenerated. Every later
     result is quoted against this exact split, so an accuracy change means
     the pipeline changed and not the shuffle.
  2. A baseline accuracy from a linear classifier on floating-point
     features. That number is the ceiling the quantised C implementation is
     measured against; a milestone that cannot beat it has not failed, but a
     milestone that appears to beat it has a bug.
  3. The trained model in the form the C classifier consumes: a plain weight
     matrix and bias vector, with the feature standardisation folded in.

Parity between this reference and the C implementation is tested separately,
by tools/gen_parity_vectors.py and tests/test_parity.c, which need no dataset
and therefore run in CI.

Two operating points are built, and they answer different questions:

  cross-subject  train on some people, test on others. This is what a
                 deployed system faces, and it is the honest headline number.
  within-subject train and test on the same people, different instances.
                 An easier problem, and NOT comparable to the number above.

Both are reported because they answer different questions, and neither
should be quoted in place of the other. The cross-subject figure is the
honest result.

A note on the within-subject point, which was originally added for a reason
that turned out to be wrong. The idea was that a higher-accuracy operating
point would resolve small quantisation losses better. In absolute percentage
points it does not: binomial standard error peaks near 50% accuracy, so the
within-subject point actually has the *larger* error bar. It helps only if
the loss is proportional to accuracy, and even then only marginally. The
resolution problem is solved properly in M3 by comparing float and fixed
point on the same samples rather than by comparing two independent
accuracies - see the note printed at the end of a run.

Usage:
    python3 tools/build_baseline.py dataset_lab_150.mat --ant-a 0 --ant-b 1
    python3 tools/build_baseline.py dataset_lab_150.mat --max-per-subject 300

Requires: numpy, scipy, scikit-learn.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcsi_data import extract_features, load_subjects   # noqa: E402


def build_splits(subjects, seed=0):
    """Build every operating point the data supports, once and for all."""
    splits = {}

    if len(subjects) > 1:
        # The held-out subject is fixed rather than random: a split that
        # moves is not a split.
        splits["cross_subject"] = {"kind": "subject",
                                   "test_subject": len(subjects) - 1}

    # Within-subject: the same 70/30 instance indices in every subject, so
    # the two operating points differ only in which axis they generalise
    # along.
    rng = np.random.default_rng(seed)
    per_subject = {}
    for i, (csi, _) in enumerate(subjects):
        idx = rng.permutation(csi.shape[0])
        cut = int(0.7 * len(idx))
        per_subject[str(i)] = {"train": idx[:cut].tolist(),
                               "test": idx[cut:].tolist()}
    splits["within_subject"] = {"kind": "instance", "seed": int(seed),
                                "per_subject": per_subject}
    return splits


def apply_split(feats, labels, split):
    """Slice precomputed features according to one split."""
    if split["kind"] == "subject":
        t = split["test_subject"]
        xtr = np.concatenate([f for i, f in enumerate(feats) if i != t])
        ytr = np.concatenate([l for i, l in enumerate(labels) if i != t])
        return xtr, ytr, feats[t], labels[t]

    xtr, ytr, xte, yte = [], [], [], []
    for i in range(len(feats)):
        sel = split["per_subject"][str(i)]
        tr, te = np.array(sel["train"]), np.array(sel["test"])
        xtr.append(feats[i][tr]); ytr.append(np.asarray(labels[i])[tr])
        xte.append(feats[i][te]); yte.append(np.asarray(labels[i])[te])
    return (np.concatenate(xtr), np.concatenate(ytr),
            np.concatenate(xte), np.concatenate(yte))


def evaluate(xtr, ytr, xte, yte):
    """Fit the linear baseline and return accuracy with its standard error."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(xtr, np.asarray(ytr).ravel())
    acc = float(clf.score(xte, np.asarray(yte).ravel()))
    n = len(yte)
    # Binomial standard error. Quoted because it sets the resolution of every
    # later comparison: a quantisation loss smaller than about twice this
    # cannot be distinguished from sampling noise.
    se = float(np.sqrt(acc * (1.0 - acc) / n))
    return clf, acc, se, n


def fold_scaler(clf):
    """Fold the StandardScaler into the linear layer.

    The baseline is fitted as StandardScaler followed by LogisticRegression,
    because standardising the features is what makes the optimiser converge on
    a feature set whose columns differ by orders of magnitude. But
    qcsi_linear_scores() is a plain w.x + b with no standardisation step, so a
    model exported straight from the pipeline would be scored on inputs it was
    never trained on and would produce confident nonsense on target.

    The standardisation is affine, so it folds into the layer exactly:

        w.((x - mean) / scale) + b  ==  (w / scale).x + (b - sum(w * mean / scale))

    What comes back is therefore the same function of the raw features, in the
    form the C consumes. It is folded rather than reimplemented on target
    because a division per feature per inference is real cost for something a
    multiplication into the weights removes entirely.
    """
    scaler = clf.named_steps["standardscaler"]
    lr = clf.named_steps["logisticregression"]

    w = lr.coef_ / scaler.scale_[None, :]
    b = lr.intercept_ - (lr.coef_ * (scaler.mean_ / scaler.scale_)[None, :]).sum(axis=1)
    return w, b, lr.classes_


def export_model(clf, x_test, path):
    """Write the folded model, after checking it against the pipeline.

    The check is cheap and worth having: a folding mistake produces a model
    that still scores every sample and still looks like a classifier.
    """
    w, b, classes = fold_scaler(clf)
    folded = x_test @ w.T + b[None, :]
    reference = clf.decision_function(x_test)
    err = float(np.max(np.abs(folded - reference)))

    np.savez(path, weights=w, bias=b, classes=classes)
    return w.shape, err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--ant-a", type=int, default=0,
                    help="first antenna of the conjugate pair")
    ap.add_argument("--ant-b", type=int, default=1,
                    help="second antenna; measure this pair with "
                         "check_dataset.py before choosing it")
    ap.add_argument("--max-per-subject", type=int, default=None)
    ap.add_argument("--split-file", default="split.json")
    ap.add_argument("--model", default="baseline_model.npz",
                    help="where to write the folded weights and bias")
    args = ap.parse_args()

    subjects = load_subjects(args.path, max_per_subject=args.max_per_subject)

    if os.path.exists(args.split_file):
        with open(args.split_file) as f:
            splits = json.load(f)
        print(f"\nreusing the frozen splits in {args.split_file}")
    else:
        splits = build_splits(subjects)
        with open(args.split_file, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"\nwrote new splits to {args.split_file} - keep it under "
              "version control and do not regenerate it")

    # Features are computed once and reused by both operating points, so any
    # difference between them comes from the split and nothing else.
    print("computing features ...")
    feats = [extract_features(csi, args.ant_a, args.ant_b) for csi, _ in subjects]
    labels = [np.asarray(lbl).ravel() for _, lbl in subjects]
    n_classes = len(np.unique(np.concatenate(labels)))

    results, held_out = {}, None
    for name in ("cross_subject", "within_subject"):
        if name not in splits:
            continue
        xtr, ytr, xte, yte = apply_split(feats, labels, splits[name])
        clf, acc, se, n = evaluate(xtr, ytr, xte, yte)
        results[name] = (acc, se, n)
        if name == "within_subject":
            held_out = (clf, xte)
        print(f"  {name:<14}: train {xtr.shape[0]}, test {xte.shape[0]}")

    chance = 1.0 / n_classes
    print()
    print(f"classes   : {n_classes}   chance: {chance * 100:.2f}%")
    for name, (acc, se, n) in results.items():
        print(f"{name:<14}: {acc * 100:5.2f}% +/- {se * 100:.2f}%  "
              f"({acc / chance:.0f}x chance, n={n})")

    if "cross_subject" in results and "within_subject" in results:
        print()
        print("The cross-subject figure is the honest result: it is what the "
              "system does on a person it has never seen. The within-subject "
              "figure is NOT comparable to it, nor to published accuracies "
              "unless those used the same kind of split.")
        print()
        worst = max(se for _, se, _ in results.values())
        print("On measuring quantisation loss in M3: comparing two "
              f"independent accuracies resolves nothing below about "
              f"{2 * worst * 100:.1f} points, at either operating point. "
              "Note that the higher-accuracy point has the LARGER error bar, "
              "because binomial variance peaks near 50%.")
        print("M3 therefore compares float and fixed point on the SAME test "
              "samples and counts disagreements. Samples that agree "
              "contribute no variance, so with a few percent of outcomes "
              "changing, the resolution is around 0.5 points - roughly four "
              "times better than the numbers above allow.")

    if results and max(a for a, _, _ in results.values()) < 2.0 * chance:
        print("\nWARNING: barely above chance. Check the antenna pair with "
              "check_dataset.py before reading anything into these numbers.")

    if held_out is not None:
        os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
        shape, err = export_model(held_out[0], held_out[1], args.model)
        print(f"\nmodel     : wrote {shape[0]} x {shape[1]} folded weights and "
              f"{shape[0]} biases to {args.model}")
        print(f"            standardisation folded in; largest score "
              f"difference against the fitted pipeline {err:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
