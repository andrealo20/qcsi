#include "qcsi/features.h"

#include "internal.h"

/* ------------------------------------------------------------------ */
/* Amplitude statistics                                                */
/* ------------------------------------------------------------------ */

qdsp_status_t qcsi_amplitude_stats(const q15_t *window, size_t n_frames,
                                   size_t n_sub, q15_t *out)
{
    size_t k, f;

    if (window == NULL || out == NULL || n_frames == 0u || n_sub == 0u) {
        return QDSP_ERR_ARG;
    }

    for (k = 0u; k < n_sub; ++k) {
        q63_t sum = 0;
        q63_t sum_sq = 0;
        q15_t lo = QDSP_Q15_MAX;
        q15_t hi = QDSP_Q15_MIN;
        int32_t mean;
        q63_t var;

        for (f = 0u; f < n_frames; ++f) {
            q15_t x = window[f * n_sub + k];
            sum += (q63_t)x;
            /* Accumulating squares in 64 bits: n_frames * 2^30 would
               overflow 32 bits for any realistic window length. */
            sum_sq += (q63_t)x * (q63_t)x;
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }

        mean = (int32_t)(sum / (q63_t)n_frames);

        /* Variance as (n*sum_sq - sum^2) / n^2, not as E[x^2] - mean^2.
           The two are algebraically identical but not numerically: the
           second form squares an integer-truncated mean, and that truncation
           error is multiplied by the mean itself. On a CSI amplitude, a
           small ripple sitting on a large static component, the inflation
           is larger than the quantity being measured. Measured on a 1%
           ripple at 0.9 mean, the truncated form overestimated the spread by
           10%; this one is within 1%.

           Range check: sum_sq <= n*2^30 and sum^2 <= n^2*2^30, so with a
           64-bit accumulator the window may be up to ~2^16 frames. */
        var = ((q63_t)n_frames * sum_sq - sum * sum) /
              ((q63_t)n_frames * (q63_t)n_frames);
        if (var < 0) var = 0;   /* only reachable through rounding */

        out[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_MEAN] =
            qdsp_sat_q15(mean);
        /* var is in Q30; its square root is therefore back in Q15, on the
           same scale as the signal, which is where the resolution is. */
        out[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_STD] =
            qdsp_sat_q15((int32_t)qcsi_isqrt64((uint64_t)var));
        out[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_PTP] =
            qdsp_sub_q15(hi, lo);
    }

    return QDSP_OK;
}

qdsp_status_t qcsi_remove_static_component(q15_t *window, size_t n_frames,
                                           size_t n_sub)
{
    size_t k, f;

    if (window == NULL || n_frames == 0u || n_sub == 0u) {
        return QDSP_ERR_ARG;
    }

    for (k = 0u; k < n_sub; ++k) {
        q63_t sum = 0;
        q15_t mean;

        for (f = 0u; f < n_frames; ++f) {
            sum += (q63_t)window[f * n_sub + k];
        }
        mean = qdsp_sat_q15((int32_t)(sum / (q63_t)n_frames));

        for (f = 0u; f < n_frames; ++f) {
            window[f * n_sub + k] = qdsp_sub_q15(window[f * n_sub + k], mean);
        }
    }

    return QDSP_OK;
}

/* ------------------------------------------------------------------ */
/* Doppler spectrum                                                    */
/* ------------------------------------------------------------------ */

qdsp_status_t qcsi_doppler_power(const q15_t *window, size_t n_frames,
                                 size_t n_sub, size_t sub, uint16_t n_fft,
                                 qdsp_cplx_q15 *scratch, q31_t *out)
{
    size_t f;
    uint16_t i;
    qdsp_status_t st;

    if (window == NULL || scratch == NULL || out == NULL ||
        n_frames == 0u || n_sub == 0u || sub >= n_sub ||
        !qdsp_fft_size_is_valid(n_fft) || (size_t)n_fft < n_frames) {
        return QDSP_ERR_ARG;
    }

    for (f = 0u; f < n_frames; ++f) {
        scratch[f].re = window[f * n_sub + sub];
        scratch[f].im = 0;
    }
    /* Zero padding interpolates the spectrum; it adds no information but
       makes the peak easier to locate. */
    for (i = (uint16_t)n_frames; i < n_fft; ++i) {
        scratch[i].re = 0;
        scratch[i].im = 0;
    }

    st = qdsp_fft_q15(scratch, n_fft);
    if (st != QDSP_OK) {
        return st;
    }

    /* Real input, so the second half mirrors the first: only n_fft/2 bins
       are written. Power is kept in Q31 because squaring two Q15 values
       fills 30 bits and clipping it back to Q15 would throw away the small
       bins, which are exactly the ones a classifier looks at. */
    for (i = 0u; i < (uint16_t)(n_fft / 2u); ++i) {
        /* 64-bit intermediate and a saturating narrow. At the limit
           re == im == -32768 the sum is 2^31, one past INT32_MAX: computing
           it in int32_t is signed overflow, which is undefined behaviour and
           is exactly what the UBSan job is there to catch. The saturation is
           reachable only by that one input pair and costs a compare. */
        int64_t re = (int64_t)scratch[i].re;
        int64_t im = (int64_t)scratch[i].im;
        out[i] = qdsp_sat_q31(re * re + im * im);
    }

    return QDSP_OK;
}

size_t qcsi_dominant_doppler_bin(const q31_t *power, size_t n_bins)
{
    size_t i, best;
    q31_t peak;

    if (power == NULL || n_bins < 2u) {
        return 0u;
    }

    /* Bin 0 is skipped: it holds whatever static component survived
       qcsi_remove_static_component(), which is not motion. The search
       therefore starts at bin 1 and so does the running best, rather than at
       bin 0 with a peak of zero: that version returned 0 for an all-zero
       spectrum, which is the same value the argument errors above return and
       is a bin the function promises never to choose. */
    best = 1u;
    peak = power[1];
    for (i = 2u; i < n_bins; ++i) {
        if (power[i] > peak) {
            peak = power[i];
            best = i;
        }
    }

    return best;
}

/* ------------------------------------------------------------------ */
/* Logarithmic compression                                             */
/* ------------------------------------------------------------------ */

/* log2(1 + i/32) in Q22, i = 0..32. Computed once, offline, as
   round(log2(1 + i/32) * 2^22); the last entry is exactly 2^22 = log2(2).
   Thirty-two intervals is what makes linear interpolation between them worth
   less than a Q10 LSB: the curvature of log2 is at most 1/ln2, so the
   interpolation error is bounded by (1/32)^2 / 8 / ln2 = 1.8e-4 in log2
   units, which is 0.13 LSB once converted to nats in Q10. */
static const uint32_t qcsi_log2_frac_q22[33] = {
          0,  186202,  366846,  542252,  712717,  878511, 1039883, 1197064,
    1350264, 1499682, 1645499, 1787884, 1926996, 2062981, 2195978, 2326114,
    2453511, 2578280, 2700529, 2820356, 2937857, 3053120, 3166228, 3277260,
    3386292, 3493394, 3598633, 3702073, 3803775, 3903795, 4002189, 4099009,
    4194304
};

/* ln 2 in Q22, which turns a base-2 logarithm into a natural one. */
#define QCSI_LN2_Q22 2907270

int32_t qcsi_log1p_q10(uint64_t x)
{
    uint64_t y, t;
    uint32_t frac, idx, rem, step, interp;
    int64_t log2_q22;
    int n;

    if (x == 0u) {
        return 0;
    }
    y = x + 1u;

    /* Position of the highest set bit, i.e. floor(log2(y)), by binary
       descent: six compares rather than a loop over 64 bit positions, and no
       dependency on a compiler builtin. */
    t = y;
    n = 0;
    if ((t >> 32) != 0u) { n += 32; t >>= 32; }
    if ((t >> 16) != 0u) { n += 16; t >>= 16; }
    if ((t >>  8) != 0u) { n +=  8; t >>=  8; }
    if ((t >>  4) != 0u) { n +=  4; t >>=  4; }
    if ((t >>  2) != 0u) { n +=  2; t >>=  2; }
    if ((t >>  1) != 0u) { n +=  1; }

    /* Normalise to 1.frac with 31 fractional bits, then drop the leading 1:
       what is left is the mantissa's fractional part, and log2(y) is
       n + log2(1 + frac / 2^31). */
    if (n >= 31) {
        frac = (uint32_t)((y >> (n - 31)) & 0x7FFFFFFFu);
    } else {
        frac = (uint32_t)((y << (31 - n)) & 0x7FFFFFFFu);
    }

    idx = frac >> 26;                    /* 0..31, which table interval */
    rem = frac & 0x03FFFFFFu;            /* how far into it */
    step = qcsi_log2_frac_q22[idx + 1u] - qcsi_log2_frac_q22[idx];
    interp = qcsi_log2_frac_q22[idx] + (uint32_t)(((uint64_t)step * rem) >> 26);

    log2_q22 = (int64_t)((uint64_t)((uint32_t)n) << 22) + (int64_t)interp;
    /* Q22 times Q22 is Q44; the shift by 34 lands it in Q10, with the usual
       half-LSB added first so the result rounds rather than truncates. Both
       operands are non-negative here, so the shift is on a positive value. */
    return (int32_t)((log2_q22 * QCSI_LN2_Q22 + ((int64_t)1 << 33)) >> 34);
}
