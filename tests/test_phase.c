/**
 * Unit tests for the phase pipeline.
 *
 * These test numerical correctness against libm and, more importantly, the
 * *invariance* the pipeline claims: that a phase offset and a phase slope
 * common to both antennas disappear from the result. That property is the
 * whole reason the conjugate product is used, so it is tested directly
 * rather than inferred from a downstream accuracy figure.
 *
 * Note on synthetic data: generating the input here is legitimate because
 * what is being checked is that an operator has the algebraic property it
 * claims, not that the system recognises human activity. A sensing claim
 * would need real recordings; an invariance claim does not.
 */
#include "unity.h"
#include "qcsi/phase.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const double PI = 3.14159265358979323846;

void setUp(void) {}
void tearDown(void) {}

/** BAM -> radians. */
static double bam_to_rad(double bam)
{
    return bam / 32768.0 * PI;
}

/** Build a complex Q15 sample of given magnitude and phase. */
static qdsp_cplx_q15 make_sample(double mag, double phase_rad)
{
    qdsp_cplx_q15 z;
    z.re = qdsp_f32_to_q15((float)(mag * cos(phase_rad)));
    z.im = qdsp_f32_to_q15((float)(mag * sin(phase_rad)));
    return z;
}

/* ------------------------------------------------------------------ */
/* CORDIC                                                              */
/* ------------------------------------------------------------------ */

static void test_atan2_matches_libm_over_full_circle(void)
{
    int i, worst = 0;
    double worst_deg = 0.0;

    for (i = 0; i < 3600; ++i) {
        double th = -PI + 2.0 * PI * (double)i / 3600.0;
        qdsp_cplx_q15 z = make_sample(0.7, th);
        qcsi_angle_t got = qcsi_atan2_q15(z.im, z.re);
        double ref_bam = atan2((double)z.im, (double)z.re) / PI * 32768.0;
        double d = (double)got - ref_bam;

        /* Fold the seam: +pi and -pi are the same angle. */
        if (d > 32768.0) d -= 65536.0;
        if (d < -32768.0) d += 65536.0;

        if (fabs(d) > (double)worst) worst = (int)fabs(d);
        if (fabs(bam_to_rad(d)) > worst_deg) worst_deg = fabs(bam_to_rad(d));
    }

    printf("  [measured] atan2 max error: %d LSB (%.4f deg)\n",
           worst, worst_deg * 180.0 / PI);
    TEST_ASSERT_TRUE_MESSAGE(worst <= 12, "atan2 error above 12 LSB");
}

static void test_atan2_axes_are_exact(void)
{
    TEST_ASSERT_INT16_WITHIN(4, 0, qcsi_atan2_q15(0, 20000));
    TEST_ASSERT_INT16_WITHIN(4, QCSI_ANGLE_HALF_PI, qcsi_atan2_q15(20000, 0));
    TEST_ASSERT_INT16_WITHIN(4, -QCSI_ANGLE_HALF_PI, qcsi_atan2_q15(-20000, 0));
    /* atan2(0, negative) is pi, which in BAM is the wrap point. */
    {
        qcsi_angle_t a = qcsi_atan2_q15(0, -20000);
        int32_t m = (a < 0) ? (int32_t)a + 65536 : (int32_t)a;
        TEST_ASSERT_INT32_WITHIN(4, 32768, m);
    }
}

static void test_magnitude_matches_libm(void)
{
    int i;
    double worst = 0.0;

    for (i = 0; i < 2000; ++i) {
        double mag = 0.05 + 0.6 * ((double)i / 2000.0);
        double th = -PI + 2.0 * PI * ((double)(i * 7 % 2000) / 2000.0);
        qdsp_cplx_q15 z = make_sample(mag, th);
        q15_t got = qcsi_magnitude_q15(z.re, z.im);
        double ref = sqrt((double)z.re * z.re + (double)z.im * z.im);
        double d = fabs((double)got - ref);
        if (d > worst) worst = d;
    }

    printf("  [measured] magnitude max error: %.1f LSB\n", worst);
    TEST_ASSERT_TRUE_MESSAGE(worst <= 8.0, "magnitude error above 8 LSB");
}

static void test_polar_returns_both_in_one_pass(void)
{
    qdsp_cplx_q15 z = make_sample(0.5, 1.0);
    q15_t m = 0;
    qcsi_angle_t a = 0;
    qcsi_polar_q15(z.re, z.im, &m, &a);
    TEST_ASSERT_INT16_WITHIN(8, qcsi_magnitude_q15(z.re, z.im), m);
    TEST_ASSERT_INT16_WITHIN(4, qcsi_atan2_q15(z.im, z.re), a);
}

/* ------------------------------------------------------------------ */
/* Conjugate product                                                   */
/* ------------------------------------------------------------------ */

static void test_conj_mul_gives_phase_difference(void)
{
    int i;
    for (i = 0; i < 200; ++i) {
        double pa = -3.0 + 6.0 * ((double)i / 200.0);
        double pb = 1.3 * sin(0.05 * (double)i);
        qdsp_cplx_q15 a = make_sample(0.6, pa);
        qdsp_cplx_q15 b = make_sample(0.6, pb);
        qdsp_cplx_q15 p = qcsi_conj_mul(a, b);

        qcsi_angle_t got = qcsi_atan2_q15(p.im, p.re);
        double ref = (pa - pb) / PI * 32768.0;
        double d = (double)got - ref;
        while (d > 32768.0) d -= 65536.0;
        while (d < -32768.0) d += 65536.0;

        TEST_ASSERT_TRUE_MESSAGE(fabs(d) < 60.0, "phase difference off");
    }
}

/**
 * The property the whole design rests on: an unknown offset and an unknown
 * slope shared by both antennas leave no trace in the phase difference.
 */
static void test_common_offset_and_slope_cancel(void)
{
    enum { NSUB = 56 };
    static qdsp_cplx_q15 a[NSUB], b[NSUB];
    static qcsi_angle_t clean[NSUB], dirty[NSUB];
    int k, worst = 0;

    /* Physical phase difference between the two antennas: what we want back. */
    double phys[NSUB];
    for (k = 0; k < NSUB; ++k) {
        phys[k] = 0.4 * sin(0.20 * (double)k) + 0.15 * (double)k * 0.05;
    }

    /* Reference: no impairment at all. */
    for (k = 0; k < NSUB; ++k) {
        a[k] = make_sample(0.5, phys[k]);
        b[k] = make_sample(0.5, 0.0);
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_phase_difference(a, b, clean, NSUB));

    /* Same scene, but now with a large CFO offset and an SFO slope applied
       identically to both antennas, which is what a real receiver does. */
    {
        const double cfo = 2.1;        /* radians, common offset */
        const double sfo = 0.037;      /* radians per subcarrier, common slope */
        for (k = 0; k < NSUB; ++k) {
            double common = cfo + sfo * (double)k;
            a[k] = make_sample(0.5, phys[k] + common);
            b[k] = make_sample(0.5, common);
        }
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_phase_difference(a, b, dirty, NSUB));

    for (k = 0; k < NSUB; ++k) {
        int d = (int)dirty[k] - (int)clean[k];
        if (d > 32768) d -= 65536;
        if (d < -32768) d += 65536;
        if (abs(d) > worst) worst = abs(d);
    }

    printf("  [measured] residual after CFO+SFO cancellation: %d LSB (%.3f deg)\n",
           worst, bam_to_rad((double)worst) * 180.0 / PI);
    TEST_ASSERT_TRUE_MESSAGE(worst <= 64,
        "common offset and slope did not cancel");
}

/* ------------------------------------------------------------------ */
/* Unwrapping                                                          */
/* ------------------------------------------------------------------ */

static void test_unwrap_recovers_a_long_ramp(void)
{
    enum { N = 200 };
    static qcsi_angle_t wrapped[N];
    static int32_t out[N];
    int k;
    /* A ramp of 0.4 rad per step crosses the +/-pi seam many times. */
    const double step = 0.4;

    for (k = 0; k < N; ++k) {
        double th = step * (double)k;
        wrapped[k] = (qcsi_angle_t)(int32_t)llround(th / PI * 32768.0);
    }

    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_unwrap(wrapped, out, N));

    for (k = 0; k < N; ++k) {
        double ref = step * (double)k / PI * 32768.0;
        TEST_ASSERT_TRUE_MESSAGE(fabs((double)out[k] - ref) < 4.0,
                                 "unwrapped ramp drifted");
    }
}

static void test_unwrap_is_identity_without_wrapping(void)
{
    static const qcsi_angle_t in[5] = { 100, 200, 150, -100, -300 };
    static int32_t out[5];
    int k;
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_unwrap(in, out, 5));
    for (k = 0; k < 5; ++k) {
        TEST_ASSERT_EQUAL_INT32((int32_t)in[k], out[k]);
    }
}

static void test_unwrap_rejects_bad_arguments(void)
{
    static const qcsi_angle_t in[2] = { 0, 0 };
    static int32_t out[2];
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_unwrap(NULL, out, 2));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_unwrap(in, NULL, 2));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_unwrap(in, out, 0));
}

/* ------------------------------------------------------------------ */
/* Detrending                                                          */
/* ------------------------------------------------------------------ */

static void test_detrend_removes_a_known_line(void)
{
    enum { N = 64 };
    static int32_t y[N];
    int32_t slope = 0;
    int k;
    const double a = 1234.0, b = -57.25;

    for (k = 0; k < N; ++k) {
        y[k] = (int32_t)llround(a + b * (double)k);
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_detrend(y, N, &slope));

    printf("  [measured] recovered slope: %.4f BAM/subcarrier (true %.4f)\n",
           (double)slope / 65536.0, b);
    TEST_ASSERT_TRUE(fabs((double)slope / 65536.0 - b) < 0.01);
    for (k = 0; k < N; ++k) {
        TEST_ASSERT_INT32_WITHIN(2, 0, y[k]);
    }
}

/**
 * Detrending is a linear operator, so adding a line to the input must not
 * change the output: detrend(x + line) == detrend(x).
 *
 * The first version of this test asserted something subtly different and
 * wrong, that a sine survives detrending untouched. It does not: a sine
 * observed over a non-integer number of periods genuinely has a non-zero
 * least-squares linear component, and removing it is correct behaviour. The
 * test failed, the code was right. Testing the linearity property instead
 * checks what the function actually promises.
 */
static void test_detrend_is_linear(void)
{
    enum { N = 64 };
    static int32_t with_line[N], without_line[N];
    int k;
    int worst = 0;

    for (k = 0; k < N; ++k) {
        double resid = 300.0 * sin(0.3 * (double)k);
        without_line[k] = (int32_t)llround(resid);
        with_line[k] = (int32_t)llround(5000.0 - 31.0 * (double)k + resid);
    }

    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_detrend(with_line, N, NULL));
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_detrend(without_line, N, NULL));

    for (k = 0; k < N; ++k) {
        int d = abs(with_line[k] - without_line[k]);
        if (d > worst) worst = d;
    }

    printf("  [measured] linearity residual: %d LSB\n", worst);
    TEST_ASSERT_TRUE_MESSAGE(worst <= 2, "detrend is not linear");
}

static void test_detrend_rejects_bad_arguments(void)
{
    static int32_t y[2] = { 0, 0 };
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_detrend(NULL, 2, NULL));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_detrend(y, 1, NULL));
}

/* ------------------------------------------------------------------ */
/* End to end                                                          */
/* ------------------------------------------------------------------ */

/**
 * The full M0 chain on a synthetic two-antenna frame: conjugate product,
 * unwrap, detrend. What must come out is the physical phase response with
 * its own linear part removed, and nothing of the impairment.
 */
static void test_pipeline_is_invariant_to_impairment(void)
{
    enum { NSUB = 56 };
    static qdsp_cplx_q15 a[NSUB], b[NSUB];
    static qcsi_angle_t diff[NSUB];
    static int32_t ref_out[NSUB], imp_out[NSUB];
    int k, worst = 0;
    double phys[NSUB];

    for (k = 0; k < NSUB; ++k) {
        phys[k] = 0.35 * sin(0.25 * (double)k + 0.4);
    }

    for (k = 0; k < NSUB; ++k) {
        a[k] = make_sample(0.5, phys[k]);
        b[k] = make_sample(0.5, 0.0);
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_phase_difference(a, b, diff, NSUB));
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_unwrap(diff, ref_out, NSUB));
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_detrend(ref_out, NSUB, NULL));

    for (k = 0; k < NSUB; ++k) {
        double common = -2.7 + 0.052 * (double)k;
        a[k] = make_sample(0.5, phys[k] + common);
        b[k] = make_sample(0.5, common);
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_phase_difference(a, b, diff, NSUB));
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_unwrap(diff, imp_out, NSUB));
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_detrend(imp_out, NSUB, NULL));

    for (k = 0; k < NSUB; ++k) {
        int d = abs(imp_out[k] - ref_out[k]);
        if (d > worst) worst = d;
    }

    printf("  [measured] end-to-end residual: %d LSB (%.3f deg)\n",
           worst, bam_to_rad((double)worst) * 180.0 / PI);
    TEST_ASSERT_TRUE_MESSAGE(worst <= 64, "pipeline not invariant to impairment");
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_atan2_matches_libm_over_full_circle);
    RUN_TEST(test_atan2_axes_are_exact);
    RUN_TEST(test_magnitude_matches_libm);
    RUN_TEST(test_polar_returns_both_in_one_pass);
    RUN_TEST(test_conj_mul_gives_phase_difference);
    RUN_TEST(test_common_offset_and_slope_cancel);
    RUN_TEST(test_unwrap_recovers_a_long_ramp);
    RUN_TEST(test_unwrap_is_identity_without_wrapping);
    RUN_TEST(test_unwrap_rejects_bad_arguments);
    RUN_TEST(test_detrend_removes_a_known_line);
    RUN_TEST(test_detrend_is_linear);
    RUN_TEST(test_detrend_rejects_bad_arguments);
    RUN_TEST(test_pipeline_is_invariant_to_impairment);
    return UNITY_END();
}
