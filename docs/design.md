# Design notes

## M0: Phase front end

### Binary angular measure, not radians

Angles are held as a signed 16-bit value where the full turn is 2^16, so
32768 maps to pi and the representable range is exactly [-pi, pi).

The reason is not compactness. Because the type wraps modulo a full turn, the
difference of two angles computed in that type *is* the shortest-path
difference, automatically. Phase unwrapping, normally a loop full of
comparisons against pi and corrections at the seam, reduces to summing those
differences, with no branch and no special case. The seam that causes bugs in
floating-point implementations simply does not exist here, because the
hardware's own wrap-around is doing the modular arithmetic.

The cost is resolution: one LSB is 360/65536 degrees, about 0.0055°. That is
far below the noise on any real CSI measurement, so it is not a limitation in
practice.

### CORDIC rather than a polynomial

`atan2` and magnitude are computed by CORDIC in vectoring mode. The reasons,
in order of weight:

- It needs only shifts, adds and a small table. No multiplications, no
  division, which is what makes it right for a core without a hardware
  multiplier, the same class of part this whole project targets.
- One pass yields magnitude *and* angle. A pipeline that needs both, as this
  one does, pays once instead of twice.
- Accuracy is predictable: roughly one bit of angle per iteration, so the
  iteration count is a dial between cost and precision rather than a
  guess.

Measured against libm over the full circle with 15 iterations: 3 LSB maximum
error, which is 0.017°.

Two implementation details worth knowing. The iteration only converges within
about ±99.7°, so inputs in the left half-plane are folded first by a
180° pre-rotation, exact, since it costs only sign changes. And the rotation
stretches the vector by a fixed gain of 1.6467602, so the magnitude is scaled
back by its reciprocal in Q15 at the end.

### Why the conjugate product, and what it actually buys

Raw single-antenna phase is dominated by carrier and sampling frequency
offsets, which is why most published work throws phase away and keeps
amplitude. But those offsets originate in one local oscillator and one
sampling clock shared by all antennas of the receiver, so they are identical
across antennas and cancel in the difference. `a * conj(b)` produces exactly
that difference, in one complex multiply, with no estimation step and nothing
to tune.

This is worth stating precisely because it is easy to overclaim: the
conjugate product removes what is *common*. Anything antenna-specific,
individual RF chain delays, per-antenna hardware phase offsets, survives it
and has to be handled elsewhere. The property is tested directly, by running
the same synthetic scene twice, once clean and once with a large common
offset and slope, and comparing: residual 2 LSB, 0.011°.

### Detrending

Any slope left across subcarrier index after the conjugate product is timing
error rather than physics, and the constant term is arbitrary. Both are
removed by an ordinary least-squares line fit, computed with 64-bit
accumulators and a Q16 slope so that a fractional slope survives integer
division. The index sums have closed forms, so they are computed rather than
accumulated.

### The same undefined-behaviour bug, in a second repository

`(num << 16)` on a signed value that can be negative is undefined behaviour
in C99. It is the identical mistake, in the identical class, that UBSan
caught twice in `qdsp`, and it was caught here by the same CI job, on the
first run, before the code had ever been used for anything.

The honest reading is not that the sanitizer is clever but that this
particular error is easy to make repeatedly: shifting is the natural way to
write a power-of-two scale, and it is correct right up until the value can go
negative. Multiplying by 65536 instead costs nothing and is always defined.
The lesson is about the value of having the check run automatically rather
than about having learned the rule.

### A test that was wrong while the code was right

The first version of the detrending test asserted that a sine survives
detrending untouched. It failed. The code was correct: a sine observed over a
non-integer number of periods genuinely has a non-zero least-squares linear
component, and removing it is exactly what the function is supposed to do.

The fix was to test the property the function actually promises, that
detrending is linear, so `detrend(x + line) == detrend(x)`. That version
passes to within 2 LSB. Worth recording because a failing test is not
automatically a failing implementation, and reaching for the code first is
the more expensive mistake.

## M2: Feature extraction

### Standard deviation, not variance

The obvious way to summarise how much a subcarrier moves over a window is
its variance. In Q15 that is the wrong choice, and the reason is worth
spelling out: a variance is a squared quantity, so it quantises as the
*square* of the signal level. Working through it for realistic CSI ripples:

| Amplitude ripple | Variance stored in Q15 | Quantisation error |
|---|---|---|
| 5.0% | 40.96 | 0.1% |
| 2.0% | 6.55 | 6.8% |
| 1.0% | 1.64 | 22% |
| 0.5% | 0.41 | rounds to zero |

CSI amplitude ripples live in exactly that bottom half of the table, so the
feature would have degraded into noise on precisely the signals it exists to
describe. Taking the square root first puts the quantity back on the same
scale as the signal, where Q15 has resolution to spare.

The square root is an integer digit-by-digit descent rather than `sqrt()`
from libm: pulling in the soft-float library for one square root would defeat
the purpose of a fixed-point pipeline on a target with no FPU.

### Two algebraically identical formulas that are not numerically identical

The first implementation computed the variance as `E[x^2] - mean^2`, with the
mean held as an integer. That form squares an integer-truncated mean, and the
truncation error gets multiplied by the mean itself, so the larger the
static component, the worse the inflation.

Measured on a 1% ripple at 0.9 full-scale mean: the truncated form
overestimated the spread by 10%, which the test caught. Using
`(n*sum_sq - sum^2) / n^2` instead removes the intermediate mean entirely and
lands within 1% across the whole range from 2% down to 0.5% ripple.

This one is worth remembering as a pattern rather than a fact: two formulas
that a textbook prints as equal can behave completely differently once an
integer division sits in the middle of one of them.

### Removing the static component before the Doppler transform

Each subcarrier carries a large fixed gain, set by antenna patterns and by
whatever is standing still in the room. It carries no motion information and
it would dominate bin 0 of any transform along time. Removing the
per-subcarrier mean first is what makes the Doppler spectrum readable: the
test injects an 8% tone on a 70% static component and recovers the correct
bin, which does not happen without the removal step.

### Why no neural network

The classifier planned for M3 is linear, over this feature set. That is a
choice, not a gap. A small feature set feeding a linear model is
interpretable, fits in a few kilobytes, and keeps the cost of every stage
visible, which is the point of the whole project. A quantised network can be
added on top of the same features later if the measured accuracy gap
justifies the cost, and that comparison is more interesting than starting
from the network.

## Validating the premise on real data

### Result on SignFi (Intel 5300, 1500 captures, lab, 5 users)

| Antenna pair | Offset reduction | Slope reduction |
|---|---|---|
| **rx0 - rx1** | **2.5x** | **1.52x** |
| rx0 - rx2 | 1.1x | 0.97x |
| rx1 - rx2 | 1.1x | 0.97x |

Median over 200 captures, measured within each capture. Mean antenna
amplitudes were 74.0, 74.3 and 96.6, so all three were receiving well and
imbalance does not explain the pattern.

The premise holds, but only for one of the three pairs. The two antennas
with nearly identical mean amplitude share their impairment; the third,
which is about 2.3 dB stronger, does not share it with either of them. Why
that should be is **not established here**. Plausible candidates are a
separate RF chain with its own oscillator path, or an antenna-selection
mechanism that treats that port differently, but neither has been tested,
and asserting one would be a guess dressed as a finding.

The practical consequence is clear enough without the explanation: the
antenna pair cannot be chosen a priori. It has to be measured per receiver,
which is why `check_dataset.py` tries every pair and reports all of them
rather than picking one.

### How a saturated measurement produced a false negative

M0's design rests on one claim: that CFO and SFO are common to every antenna
of a receiver, so a conjugate product between two antennas cancels them. That
is true by construction on synthetic signals, which is what the unit tests
show. Whether it is true of a given real receiver is a separate question, and
the answer turned out to be more interesting than expected.

Getting to those numbers took two corrections to the measurement, both of
which initially looked like evidence against the method.

Run on a real Intel 5300 capture, the first version of `check_dataset.py`
reported that the conjugate product barely reduced the phase offset and made
the residual slope five times *worse*. On its face, the premise was wrong.

The premise was fine. The mean amplitudes across the three receive antennas
were 14.3, 1.3 and 1.4: the second and third were roughly 20 dB down, so
their phase was mostly noise. Differencing against a noisy reference adds
noise rather than removing an offset. The correlation between per-packet
phase slopes on rx0 and rx1 was 0.11, which is what "no shared term" looks
like. Choosing the two antennas that were actually comparable gave the
expected result:

| Antenna pair | Offset reduction | Slope reduction |
|---|---|---|
| rx0 - rx1 | 1.7x | 0.21x (worse) |
| rx0 - rx2 | **3.1x** | **1.81x** |
| rx1 - rx2 | 1.7x | 0.22x (worse) |

Two things came out of this. The checker now measures per-antenna amplitude
balance, warns when an antenna is more than 10 dB down, and tries every pair
instead of assuming rx0 and rx1, because picking a pair blindly is exactly
how the first version reached the wrong conclusion. And the qualification
belongs in the design itself: the conjugate product cancels what is *common*,
and two antennas share a common term only if both are actually receiving. A
receiver with one dominant antenna cannot use this method, however sound the
theory.

That is a real limitation of the approach, found by running it on real data
rather than by reasoning about it, and it would have been discovered at M4,
after the classifier was built on top, if the check had come later.

### The second false negative: pooling separate captures

On SignFi the corrected script still reported no improvement at all: 1.83 rad
of phase spread before the conjugate product and 1.88 after. The number that
gave it away is 1.814, the standard deviation of a phase distributed
uniformly over the whole circle. Both measurements had saturated, so no
method could have shown an improvement, whatever its merits.

The cause was in the script, not the data. It pooled 200 separately captured
gestures before computing statistics. Separate captures do not share an
antenna phase offset: the receiver re-runs gain control and antenna selection
between them. Pooling therefore measured capture-to-capture variation, which
the conjugate product neither can nor should remove.

Statistics are now computed within each capture and summarised by median
across captures, and the spread uses a circular standard deviation rather
than an ordinary one, on the circle, +pi and -pi are adjacent, and ordinary
arithmetic treats them as maximally distant. With both fixes, the same Intel
5300 file that had shown 3.1x shows 8.1x, and SignFi moves from "no effect"
to 2.5x.

Worth stating plainly: two separate measurement errors both pointed the same
way, towards rejecting a method that was working. A saturated statistic does
not announce itself: it looks like a clean negative result. Comparing the
observed value against what a maximally-spread distribution would give is
what caught it, and that check now runs automatically.

## M1: Baseline and split

### The split is a file, not a function call

`split.json` is written once and then reused. A split regenerated on every
run is not a split: any accuracy difference between two runs mixes the change
you made with the shuffle you got. Freezing it costs nothing and makes every
subsequent number comparable.

### Subject-wise, when the data allows it

SignFi's lab_150 file holds five users, and if they arrive as separate
variables the split is by subject: train on four people, test on the fifth.
That is the only split that answers the question a deployed system faces,
which is whether the model works on someone it has never seen.

A random split over instances of the same person measures generalisation to
new recordings of a known person, which is a much easier problem. Both
numbers are legitimate; presenting the second as if it were the first is not,
and it is common enough in this literature that two published accuracies
often cannot be compared at all. The tool prints which kind of split it used
and warns when only the easier one is available.

### Two operating points, and why one number was not enough

The first run on real data gave 10.2% cross-subject accuracy over 150
classes, fifteen times chance, so the pipeline is clearly extracting real
information, but low in absolute terms. That figure is the honest result and
stays the headline.

The problem it created is one of measurement resolution rather than of
quality. With 1500 test instances at 10% accuracy, the binomial standard
error is 0.78%, so nothing smaller than about 1.6 points can be
distinguished from sampling noise, and quantisation loss is typically
smaller than that.

A second operating point was added in response: a within-subject split, same
features, same code, only the split differs. It gives 42.27% ± 1.04%.

**The stated reason for adding it was wrong, and the numbers say so.** The
argument had been that a higher-accuracy operating point resolves small
losses better. In absolute percentage points it does the opposite: binomial
variance peaks near 50% accuracy, so the within-subject point has the larger
error bar, 1.04% against 0.78%. Measured:

| Operating point | Accuracy | Std error | Resolves |
|---|---|---|---|
| cross-subject | 10.20% | 0.78% | 1.6 points |
| within-subject | 42.27% | 1.04% | 2.1 points |

The second point does help, but only under an assumption that had not been
made explicit: that quantisation loss is *proportional* to accuracy rather
than a fixed number of points. Under a 3% relative loss it is worth 0.31
points at the first operating point and 1.27 at the second, so the
loss-to-noise ratio goes from 0.39 to 1.22, better, and still marginal.

### The measurement that actually works: paired comparison

Comparing two independently estimated accuracies throws away the fact that
both are computed on the same test samples. Running the float and fixed-point
pipelines over identical inputs and counting the samples whose outcome
*changes* gives a far tighter estimate, because every sample that agrees
contributes no variance at all. With 1-3% of outcomes changing on 2250 test
samples, the standard error is 0.21-0.37%, resolving differences of about
0.5 points, roughly four times better than either accuracy comparison
allows. This is McNemar's setting, and it is the right tool for comparing two
classifiers on one test set.

M3 uses the paired comparison as its primary measurement. The two operating
points are still both reported, because they answer genuinely different
questions about the system, but the resolution argument for the second one
has been corrected rather than quietly left standing.

The lesson worth keeping is narrow and useful: "use a higher operating point
to see a small effect" sounds like sound methodology and is not. Whether a
change is measurable depends on the variance of the specific comparison being
made, and the way to find out is to compute it before building on it.

### Why a linear classifier

The baseline is logistic regression on a small feature set. It is not
intended to be competitive with the deep models published on these datasets,
and the README should not pretend otherwise. Its job is to be a *reference*:
a number the quantised C implementation can be measured against, computed by
a model simple enough that a gap points at the pipeline rather than at
training dynamics.

There is also a diagnostic value. If the baseline barely beats chance, the
problem is upstream, a wrong antenna pair, a mis-identified axis, and no
amount of model capacity will fix it. A deep model would have hidden that by
partially compensating for the broken input.

### A label-matching bug that would have been invisible

SignFi's lab_150 file holds five CSI variables of 1500 instances each and a
single label vector of 7500 entries, every subject's labels concatenated in
order. The first version of the loader matched labels by name suffix, fell
back to "if there is only one label variable, use it", and then truncated to
the length of the CSI array. Subject 1 got the right labels. Subjects 2
through 5 all got subject 1's.

Nothing about that fails loudly. Training would have run, an accuracy would
have been printed, and it would have been meaningless. The loader now
distinguishes the two layouts explicitly, one label variable per subject,
or one shared vector split by instance count, and refuses to proceed when
the totals do not line up, rather than guessing.

The general point: silent data-loading bugs are worse than crashes, because
they produce a number instead of an error, and a number gets believed.

### Features mirror the C stages

`qcsi_data.py` is written to follow the C pipeline stage by stage rather than
in the most natural NumPy style. That is deliberate: when the C output
diverges from the reference, the structure lets the discrepancy be traced to
one stage instead of to the whole chain. The exported header carries the
quantisation scale explicitly so the C side can reproduce the same fixed-point
input rather than approximating it.

## M3: Quantised classifier

### One scale for the whole model

Weights and features are both Q15 with a single scale factor per model, not
per class or per feature. Per-class scales would quantise more accurately but
would leave the class scores on different scales, so the argmax would need
each one rescaled first, and an argmax over rescaled values is exactly where
a subtle bug lives quietly. One scale keeps the comparison exact and the
argmax trivially correct.

### Ties have to be specified

In floating point, two class scores are never exactly equal. Once quantised,
they are, and often enough to matter. An unspecified tie-break would make the
C and Python implementations disagree at random on precisely the samples a
parity test is designed to examine, the marginal ones. Ties therefore break
towards the lower class index, in both implementations, and there is a test
that says so.

### The margin explains which samples are lost, not just how many

An accuracy figure says how many predictions changed. The decision margin,
the gap between the best and second-best score, says which, and it should be
the narrow ones. This is worth reporting because it is also a diagnostic: if
disagreements appear at wide margins, the difference is not quantisation
noise, and the likely cause is a scaling or alignment error between the two
pipelines. Precision loss has a signature, and checking for it is cheap.

Measured on the unit tests, with a perturbation of ±800 LSB per feature:
12.7% of low-margin samples flip, 0% of high-margin ones.

### A test that measured nothing

The first version of the fragility test used a ±10 LSB perturbation and
flipped zero samples in both groups. It then asserted that the low-margin flip
rate exceeded the high-margin one, an assertion on 0 against 0, which is an
assertion on nothing. Measuring the actual margin distribution (5.5e5 to
2.5e8, mean 4.7e7) and calibrating the perturbation to flip a few percent of
samples turned it into a test that can fail for a real reason. The test now
also asserts that at least one sample flipped, so it cannot silently degrade
back into measuring nothing.

### Paired comparison, and what it found

Comparing two independently estimated accuracies discards the fact that both
come from the same test samples. Running both pipelines over identical inputs
and counting changed predictions is far tighter, because agreeing samples
contribute no variance: about 0.5 points of resolution against 2.

How much it mattered can be put precisely. On the real measurement, Q8 loses
0.22 accuracy points. The standard error of a difference between two
independently estimated accuracies at this operating point is 1.47 points, so
that result would have been entirely invisible, indistinguishable from zero,
and equally indistinguishable from a one-point loss. The paired count, 71
disagreements out of 2250, has a standard error of 0.37 points and leaves no
room for argument.

### What the sweep found on real data

| Word length | Accuracy | Change | Disagreements | Low margin | High margin |
|---|---|---|---|---|---|
| float | 42.27% | n/a | n/a | n/a | n/a |
| Q15 | 42.27% | +0.00 | 0 | 0.00% | 0.00% |
| Q12 | 42.27% | +0.00 | 2 | 0.18% | 0.00% |
| Q10 | 42.18% | −0.09 | 19 | 1.69% | 0.00% |
| Q8 | 42.04% | −0.22 | 71 | 6.31% | 0.00% |
| Q7 | 41.78% | −0.49 | 143 | 12.71% | 0.00% |
| Q6 | 41.33% | −0.93 | 332 | 29.07% | 0.44% |
| Q5 | 39.29% | −2.98 | 642 | 49.33% | 7.73% |
| Q4 | 32.31% | −9.96 | 1080 | 66.13% | 29.87% |

**Q15 is bit-exact.** Zero disagreements out of 2250. For a linear model of
this size, the whole question of what 16-bit fixed point costs has the answer
"nothing measurable", and that is a stronger result than a small loss would
have been.

**Q8 costs 0.22 points**, with all 71 disagreements in the narrow-margin
half. For a linear model the weight table dominates the memory footprint, so
halving the word length halves it, and this says that trade is nearly free
on this task. That is the kind of finding the sweep exists to produce and a
single Q15 measurement could not have.

**The breakdown is orderly, and that is itself evidence.** Disagreements stay
entirely in the low-margin half down to Q7, first appear at wide margins at
Q6 (0.44%), and only become substantial at Q4. That progression is what
precision loss looks like. A scaling error would have scattered disagreements
across the margin range from the start, at every word length, and the check
for it costs one extra column in the table.

The earlier worry about measurement resolution turns out to have been mostly
moot at Q15, since there was nothing to resolve. But that could not have been
known in advance, and the same instrument is what makes the Q8 result
trustworthy.

## M4: Pipeline, cost and footprint

### One context, supplied by the caller

Every buffer lives in one struct that the caller owns. No allocation, no
hidden globals, and therefore two pipelines can run side by side on different
antenna pairs, which is tested, because "should be fine" is not a property.

The struct is large enough that it belongs in static storage rather than on a
stack, and `qcsi_pipeline_footprint()` returns its size so a caller can check
rather than guess. A number the code reports stays correct when the
configuration changes; a number in a comment does not.

### Phase statistics are accumulated, not stored

Keeping the phase history for a window would cost `n_frames * n_sub` words on
top of the amplitude window. The feature vector only needs the mean and the
standard deviation, so those are accumulated per frame and the history is
discarded. On the reference configuration that is the difference between 1.9
KiB and 3.8 KiB for the same output, the kind of saving that decides whether
a window fits in the RAM of a small part.

### The window does not slide

A completed window is consumed and the next starts empty. A sliding window
would emit a decision per frame at `n_frames` times the cost. That may well
be what an application wants, but on a microcontroller it is a decision worth
making explicitly rather than inheriting from a library default.

### Rejecting a degenerate antenna pair

`ant_a == ant_b` would make the conjugate product a constant zero phase:
every phase feature would be zero, the pipeline would run happily, and the
output would be quietly meaningless. It is refused at init, along with a
model whose feature count does not match the configuration, which would
otherwise read past the end of the feature vector.

Both are configuration mistakes that produce plausible behaviour rather than
a crash, which is exactly the class worth spending validation code on.

### What is measured, and what is not

| | Per window | Per frame |
|---|---|---|
| multiply-accumulate | 13,440 | 420 |
| accumulator to Q15 | 6,720 | 210 |

Static context: 34.3 KiB, dominated by an amplitude window sized for the
compile-time maximum rather than for the configuration in use (32 KiB against
the 1.9 KiB this configuration needs).

These are operation counts. They are exact, deterministic and independent of
the target, which is what makes them worth watching for regressions. They are
**not** cycle counts, and no cycle count is reported here.

The reason is the same one recorded in qdsp's M4: this environment has
neither an ARM cross-compiler nor Renode, so a cycle figure would have to be
either invented or taken from an emulator that does not model the pipeline.
Either would look like a measurement and be worth less than admitting the
gap. The same applies to flash and RAM: the footprint above is the size of a
struct on the host, not a linker map from a Cortex-M build.

What it would take to close this is specific and small: `arm-none-eabi-gcc`
for a real `size` report, and Renode with a Cortex-M4 platform file for
instruction counts under a modelled core. It is first on the open list for
that reason.

## Still open

- **Cycle counts and a real memory map.** Needs `arm-none-eabi-gcc` and
  Renode with a Cortex-M4 platform. Everything else is done; this is the one
  claim the project deliberately does not make.
- **Why only one antenna pair works.** On SignFi, rx0-rx1 cancels the
  impairment and neither pair involving rx2 does, despite all three antennas
  receiving well. Unexplained, and recorded as such above.
- **Per-antenna hardware phase offsets** are not removed by the conjugate
  product and would need calibration or a reference antenna.
- **A tighter default configuration.** The context is 34.3 KiB because the
  buffers are sized for the compile-time maximum; a deployment with known
  dimensions would use a fraction of that.
- **Cross-environment generalisation** is untested: everything here is the
  lab recording. SignFi's home dataset would answer it.
