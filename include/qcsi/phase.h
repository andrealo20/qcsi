/**
 * @file phase.h
 * @brief Fixed-point phase processing for Wi-Fi Channel State Information.
 *
 * Raw CSI phase from a single antenna is not usable. Every packet carries an
 * unknown carrier frequency offset (CFO) and sampling frequency offset (SFO)
 * between transmitter and receiver, which appear as an arbitrary offset plus
 * an arbitrary slope across subcarrier index. Two packets of the same static
 * scene can have completely different raw phase.
 *
 * The signal-processing answer used here is that those offsets are *common*
 * to every antenna of the same receiver, because they come from one shared
 * local oscillator and one shared sampling clock. Multiplying one antenna by
 * the conjugate of another therefore cancels them exactly and leaves the
 * physical phase difference, which is the quantity that actually carries
 * information about the propagation environment.
 *
 * Everything below runs in fixed point, with no floating point on the
 * processing path.
 *
 * ## Angle representation
 *
 * Angles are binary angular measure (BAM): a signed 16-bit value where the
 * full turn is 2^16, so QCSI_ANGLE_PI = 32768 maps to pi radians and the
 * representable range is exactly [-pi, pi).
 *
 * This is not a detail. Because the type wraps modulo a full turn, the
 * difference of two angles is automatically the shortest-path difference,
 * with no branch and no comparison against pi. Phase unwrapping, which is
 * fiddly and error-prone with radians in floating point, reduces to summing
 * those differences. See docs/design.md.
 */
#ifndef QCSI_PHASE_H
#define QCSI_PHASE_H

#include <stddef.h>
#include <stdint.h>

#include "qdsp/fft.h"     /* qdsp_cplx_q15 */
#include "qdsp/fixed.h"
#include "qdsp/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Binary angular measure: full turn = 2^16, so this wraps on its own. */
typedef int16_t qcsi_angle_t;

#define QCSI_ANGLE_PI       32768L   /**< pi radians, as a BAM magnitude */
#define QCSI_ANGLE_HALF_PI  16384
#define QCSI_ANGLE_MAX      32767

/** Number of CORDIC iterations; each one adds about one bit of angle. */
#define QCSI_CORDIC_ITERATIONS 15

/* ------------------------------------------------------------------ */
/* Rectangular to polar                                                */
/* ------------------------------------------------------------------ */

/**
 * Magnitude and angle of a complex Q15 sample, in one CORDIC pass.
 *
 * CORDIC is used rather than a polynomial approximation because it needs
 * only shifts, adds and a small table: no multiplications and no division,
 * which is what makes it the right choice on a core without a hardware
 * multiplier. It also produces magnitude and angle together, so a pipeline
 * that needs both pays for one pass instead of two.
 *
 * @param re,im  input sample.
 * @param mag    output magnitude, saturated to Q15. Note that |z| can reach
 *               sqrt(2) for a full-scale input, which does not fit in Q15;
 *               pre-scale the input if that matters for your data.
 * @param angle  output angle in BAM.
 *
 * Either output pointer may be NULL if that result is not needed.
 */
void qcsi_polar_q15(q15_t re, q15_t im, q15_t *mag, qcsi_angle_t *angle);

/** Angle only, equivalent to atan2(im, re) in BAM. */
qcsi_angle_t qcsi_atan2_q15(q15_t im, q15_t re);

/** Magnitude only, equivalent to hypot(re, im), saturated to Q15. */
q15_t qcsi_magnitude_q15(q15_t re, q15_t im);

/* ------------------------------------------------------------------ */
/* Offset cancellation across antennas                                 */
/* ------------------------------------------------------------------ */

/**
 * a * conj(b), the product whose phase is the phase difference a - b.
 *
 * This is the operation that removes CFO and SFO: both antennas share the
 * same oscillator and the same sampling clock, so the offending terms are
 * identical in a and b and cancel in the difference. The result keeps the
 * product of the magnitudes, which is why the caller usually wants the
 * angle of the result rather than the result itself.
 */
qdsp_cplx_q15 qcsi_conj_mul(qdsp_cplx_q15 a, qdsp_cplx_q15 b);

/**
 * Phase difference between two antenna streams, across n subcarriers.
 *
 * @param a,b  n complex samples each, same subcarrier grid.
 * @param out  n angles, the phase of a[k]*conj(b[k]).
 */
qdsp_status_t qcsi_phase_difference(const qdsp_cplx_q15 *a,
                                    const qdsp_cplx_q15 *b,
                                    qcsi_angle_t *out, size_t n);

/* ------------------------------------------------------------------ */
/* Unwrapping and detrending                                           */
/* ------------------------------------------------------------------ */

/**
 * Unwrap a sequence of wrapped angles into a continuous ramp.
 *
 * out[0] = in[0], and each following point adds the wrapped difference
 * between consecutive inputs. Correct as long as the true step between
 * consecutive subcarriers stays below half a turn, which is the same
 * condition every unwrapping algorithm needs.
 *
 * @param out  n values in BAM units, widened to 32 bits because an unwrapped
 *             ramp is not bounded by one turn.
 */
qdsp_status_t qcsi_unwrap(const qcsi_angle_t *in, int32_t *out, size_t n);

/**
 * Remove the best-fit line from an unwrapped phase sequence, in place.
 *
 * Any residual slope across subcarrier index is timing error, not physics,
 * and any constant offset is arbitrary. Removing both leaves the part of the
 * phase response that depends on the propagation environment.
 *
 * The fit is ordinary least squares, computed with 64-bit accumulators and a
 * Q16 slope, so no floating point is involved.
 *
 * @param slope_q16  optional output: the removed slope, in BAM units per
 *                   subcarrier, Q16. May be NULL.
 */
qdsp_status_t qcsi_detrend(int32_t *phase, size_t n, int32_t *slope_q16);

#ifdef __cplusplus
}
#endif

#endif /* QCSI_PHASE_H */
