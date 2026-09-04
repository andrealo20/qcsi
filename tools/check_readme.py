#!/usr/bin/env python3
"""
Check that the README's tables are still internally consistent.

The quantisation table quotes raw counts and derived columns side by side: the
number of predictions that changed, the accuracy that implies, and how those
disagreements split across the margin distribution. Those columns are not
independent, so most ways of getting one of them wrong show up as a
disagreement with the others. This recomputes every derived column from the
counts and fails if the README does not agree.

It does not, and cannot, verify that the counts themselves are what the sweep
produced: that needs the SignFi dataset. What it catches is the far more
likely failure, a table edited by hand and left half-updated.

Usage:
    python3 tools/check_readme.py [README.md]
"""

import os
import re
import sys

# The sweep splits its 2250 test samples at the median decision margin, so the
# narrow and wide halves hold this many samples each. The percentages in the
# last two columns are fractions of one half, not of the whole test set.
N_TEST = 2250
N_HALF = N_TEST // 2
N_CLASSES = 150

# Tolerances. The README rounds accuracy and the margin percentages to two
# decimals, so a recomputed count can land half a unit either side.
COUNT_TOL = 1.0
ACC_TOL = 0.015
SE_TOL = 0.02


def normalise(cell):
    """Strip markdown emphasis and unify the several dash characters."""
    cell = cell.strip().strip("*` ")
    # Built with chr() so this source file itself stays free of the characters
    # it is normalising away.
    for code in (0x2212, 0x2013, 0x2014):  # minus sign, en dash, em dash
        cell = cell.replace(chr(code), "-")
    return cell


def number(cell):
    """Return the number in a cell, or None when it carries no value."""
    cell = normalise(cell).rstrip("%").replace("+", "")
    if cell in ("", "-", "n/a", "N/A"):
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def table_rows(text, header_contains):
    """Rows of the first markdown table whose header holds every given word."""
    rows, in_table, matched = [], False, False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if matched and in_table:
                break
            in_table = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if set(stripped) <= set("|-: "):
            continue
        if not in_table:
            in_table = True
            matched = all(w.lower() in stripped.lower() for w in header_contains)
            if matched:
                continue
        if matched:
            rows.append(cells)
    return rows


def check_quantisation(text, fail):
    rows = table_rows(text, ["Word length", "Disagreements", "Narrow margin"])
    if not rows:
        fail("quantisation table not found")
        return

    baseline = None
    for cells in rows:
        if len(cells) < 6:
            continue
        label = normalise(cells[0])
        acc, change = number(cells[1]), number(cells[2])
        dis, narrow, wide = number(cells[3]), number(cells[4]), number(cells[5])

        if label.lower() == "float":
            baseline = acc
            if baseline is None:
                fail("float row carries no accuracy")
            continue
        if None in (acc, dis, narrow, wide):
            fail(f"{label}: a column is missing a value")
            continue

        # The two margin percentages are fractions of one half of the test set,
        # so together they have to account for every disagreement.
        implied = (narrow + wide) / 100.0 * N_HALF
        if abs(implied - dis) > COUNT_TOL:
            fail(f"{label}: margin percentages imply {implied:.1f} disagreements, "
                 f"table says {dis:.0f}")

        # Every disagreement is one prediction that changed. It can go either
        # way, so this bounds the accuracy change rather than fixing it.
        if abs(acc - baseline) * N_TEST / 100.0 > dis + COUNT_TOL:
            fail(f"{label}: accuracy moved {abs(acc - baseline):.2f} points, "
                 f"which needs more than the {dis:.0f} disagreements listed")

        if change is not None and abs((acc - baseline) - change) > ACC_TOL:
            fail(f"{label}: change column says {change:+.2f}, "
                 f"accuracy implies {acc - baseline:+.2f}")


def check_accuracy(text, fail):
    rows = table_rows(text, ["Operating point", "Accuracy", "chance"])
    if not rows:
        fail("accuracy table not found")
        return

    chance = 100.0 / N_CLASSES
    m = re.search(r"chance \((\d+\.\d+)%\)", text)
    if m and abs(float(m.group(1)) - chance) > 0.01:
        fail(f"quoted chance level {m.group(1)}% is not 100/{N_CLASSES} "
             f"= {chance:.2f}%")

    for cells in rows:
        if len(cells) < 3:
            continue
        label = normalise(cells[0])
        m = re.search(r"([\d.]+)%\s*(?:±|\+/-)\s*([\d.]+)%", cells[1])
        if not m:
            continue
        acc, se = float(m.group(1)), float(m.group(2))

        # The quoted uncertainty is the binomial standard error, so it is fixed
        # by the accuracy and the sample count and cannot be chosen freely.
        n = N_TEST if "within" in label.lower() else None
        if n:
            p = acc / 100.0
            expected = (p * (1 - p) / n) ** 0.5 * 100.0
            if abs(expected - se) > SE_TOL:
                fail(f"{label}: standard error {se:.2f} does not match the "
                     f"binomial value {expected:.2f} at n={n}")

        mult = re.search(r"(\d+)\s*[x×]", cells[2])
        if mult:
            expected = acc / chance
            if abs(expected - float(mult.group(1))) > 1.0:
                fail(f"{label}: quoted {mult.group(1)}x chance, "
                     f"accuracy implies {expected:.1f}x")


def check_paired_claim(text, fail):
    """The paired-comparison argument quotes a standard error of a difference."""
    m = re.search(r"standard error of a difference[^.]*?is ([\d.]+) points", text)
    if not m:
        return
    quoted = float(m.group(1))
    p = 0.4227
    single = (p * (1 - p) / N_TEST) ** 0.5 * 100.0
    expected = single * (2 ** 0.5)
    if abs(expected - quoted) > SE_TOL:
        fail(f"paired-comparison claim quotes {quoted} points, "
             f"two independent estimates give {expected:.2f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    problems = []
    fail = problems.append

    check_quantisation(text, fail)
    check_accuracy(text, fail)
    check_paired_claim(text, fail)

    if problems:
        print(f"{os.path.basename(path)}: {len(problems)} inconsistencies")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"{os.path.basename(path)}: tables are internally consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
