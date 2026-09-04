#include "qcsi/phase.h"

extern const int32_t qcsi_cordic_atan[QCSI_CORDIC_ITERATIONS];
extern const int32_t qcsi_cordic_inv_gain;

/* ------------------------------------------------------------------ */
/* CORDIC, vectoring mode                                              */
/* ------------------------------------------------------------------ */

/**
 * Rotate (x, y) towards the positive x axis, accumulating the rotation.
 * When y has been driven to zero, x holds the magnitude scaled by the CORDIC
 * gain and the accumulated rotation is the original angle.
 *
 * The iteration only converges for angles within about +/-99.7 degrees, so
 * the second and third quadrants are folded in first by a 180-degree
 * pre-rotation, which is exact because it costs only sign changes.
 */
void qcsi_polar_q15(q15_t re, q15_t im, q15_t *mag, qcsi_angle_t *angle)
{
    int32_t x = (int32_t)re;
    int32_t y = (int32_t)im;
    int32_t z = 0;
    int i;

    if (x == 0 && y == 0) {
        /* The CORDIC loop below has no fixed point at the origin: with
           x == y == 0 every iteration takes the "y <= 0" branch (since y is
           never > 0) and subtracts qcsi_cordic_atan[i] regardless, so z
           drifts to a large, deterministic but meaningless angle instead of
           the conventional atan2(0, 0) = 0. Magnitude is unaffected (0
           either way), but angle needs an explicit fixed point here. */
        if (angle != NULL) {
            *angle = 0;
        }
        if (mag != NULL) {
            *mag = 0;
        }
        return;
    }

    if (x < 0) {
        /* Fold into the right half-plane. Rotating by pi is exact. */
        x = -x;
        y = -y;
        z = (y >= 0) ? -(int32_t)QCSI_ANGLE_PI : (int32_t)QCSI_ANGLE_PI;
        /* The sign chosen above keeps the final result inside [-pi, pi). */
    }

    for (i = 0; i < QCSI_CORDIC_ITERATIONS; ++i) {
        /* x and y can be negative here, and right-shifting a negative signed
           value is implementation-defined rather than undefined: see the note
           in qcsi_detrend() below for why the two are not the same problem. */
        int32_t xs = x >> i;
        int32_t ys = y >> i;
        if (y > 0) {
            x += ys;
            y -= xs;
            z += qcsi_cordic_atan[i];
        } else {
            x -= ys;
            y += xs;
            z -= qcsi_cordic_atan[i];
        }
    }

    if (angle != NULL) {
        /* The 16-bit type performs the modulo-a-turn reduction for free. */
        *angle = (qcsi_angle_t)z;
    }

    if (mag != NULL) {
        /* Undo the CORDIC gain. x is non-negative here by construction. */
        int32_t m = (int32_t)(((int64_t)x * qcsi_cordic_inv_gain + 16384) >> 15);
        *mag = qdsp_sat_q15(m);
    }
}

qcsi_angle_t qcsi_atan2_q15(q15_t im, q15_t re)
{
    qcsi_angle_t a;
    qcsi_polar_q15(re, im, NULL, &a);
    return a;
}

q15_t qcsi_magnitude_q15(q15_t re, q15_t im)
{
    q15_t m;
    qcsi_polar_q15(re, im, &m, NULL);
    return m;
}

/* ------------------------------------------------------------------ */
/* Offset cancellation                                                 */
/* ------------------------------------------------------------------ */

qdsp_cplx_q15 qcsi_conj_mul(qdsp_cplx_q15 a, qdsp_cplx_q15 b)
{
    qdsp_cplx_q15 r;
    /* (ar + j*ai) * (br - j*bi) */
    q63_t re = qdsp_mac_q15(qdsp_mac_q15(0, a.re, b.re), a.im, b.im);
    q63_t im = qdsp_msu_q15(qdsp_mac_q15(0, a.im, b.re), a.re, b.im);
    r.re = qdsp_acc_to_q15(re);
    r.im = qdsp_acc_to_q15(im);
    return r;
}

qdsp_status_t qcsi_phase_difference(const qdsp_cplx_q15 *a,
                                    const qdsp_cplx_q15 *b,
                                    qcsi_angle_t *out, size_t n)
{
    size_t k;

    if (a == NULL || b == NULL || out == NULL || n == 0u) {
        return QDSP_ERR_ARG;
    }

    for (k = 0u; k < n; ++k) {
        qdsp_cplx_q15 p = qcsi_conj_mul(a[k], b[k]);
        out[k] = qcsi_atan2_q15(p.im, p.re);
    }
    return QDSP_OK;
}

/* ------------------------------------------------------------------ */
/* Unwrapping                                                          */
/* ------------------------------------------------------------------ */

qdsp_status_t qcsi_unwrap(const qcsi_angle_t *in, int32_t *out, size_t n)
{
    size_t k;

    if (in == NULL || out == NULL || n == 0u) {
        return QDSP_ERR_ARG;
    }

    out[0] = (int32_t)in[0];
    for (k = 1u; k < n; ++k) {
        /* The subtraction is done in the 16-bit type on purpose: it wraps,
           and the wrapped result is exactly the shortest-path difference.
           No comparison against pi, no branch, no special case at the seam. */
        qcsi_angle_t d = (qcsi_angle_t)((uint16_t)in[k] - (uint16_t)in[k - 1u]);
        out[k] = out[k - 1u] + (int32_t)d;
    }
    return QDSP_OK;
}

/* ------------------------------------------------------------------ */
/* Detrending                                                          */
/* ------------------------------------------------------------------ */

qdsp_status_t qcsi_detrend(int32_t *phase, size_t n, int32_t *slope_q16)
{
    int64_t sum_y = 0, sum_iy = 0;
    int64_t sum_i, sum_ii, denom, num;
    int64_t slope, intercept_q16;
    size_t k;
    int64_t nn = (int64_t)n;

    if (phase == NULL || n < 2u) {
        return QDSP_ERR_ARG;
    }

    for (k = 0u; k < n; ++k) {
        sum_y += (int64_t)phase[k];
        sum_iy += (int64_t)k * (int64_t)phase[k];
    }

    /* Closed forms for the index sums: no need to accumulate them. */
    sum_i = nn * (nn - 1) / 2;
    sum_ii = (nn - 1) * nn * (2 * nn - 1) / 6;

    denom = nn * sum_ii - sum_i * sum_i;
    if (denom == 0) {
        return QDSP_ERR_RANGE;
    }

    /* Slope in Q16, so a fractional slope survives integer division.
       Multiplication rather than "<< 16" because num and sum_y are signed
       and can be negative, and left-shifting a negative value is undefined
       behaviour in C99 — caught here by the UBSan job, exactly as it was in
       qdsp. Same mistake, same class, second repository. */
    num = nn * sum_iy - sum_i * sum_y;
    slope = (num * 65536) / denom;

    /* intercept = mean(y) - slope * mean(i), also in Q16. */
    intercept_q16 = ((sum_y * 65536) - slope * sum_i) / nn;

    for (k = 0u; k < n; ++k) {
        /* The right shift below is a different case from the left shift
           above, and the distinction is the whole reason both appear in one
           function. Left-shifting a negative value is *undefined*: the
           standard gives the expression no meaning at all, and a sanitizer
           is right to stop the program. Right-shifting one is only
           *implementation-defined*: every implementation has to document what
           it does, and every compiler this library is built with, and every
           one qdsp's fixed.h asserts against at compile time, propagates the
           sign. So the shift is portable in practice and arithmetic, which is
           what rounding a Q16 value towards minus infinity needs; replacing
           it with a division would round towards zero instead and change the
           result on negative phases. */
        int64_t fit = (slope * (int64_t)k + intercept_q16 + 32768) >> 16;
        phase[k] -= (int32_t)fit;
    }

    if (slope_q16 != NULL) {
        *slope_q16 = (int32_t)slope;
    }
    return QDSP_OK;
}
