/**
 * Unit tests for feature extraction.
 *
 * As in test_phase.c, the inputs are generated, and for the same reason:
 * what is under test is whether an operator computes what it claims —
 * a variance, a spectrum, a peak location — not whether the system can
 * recognise human activity. The second claim needs recorded data and is not
 * made here.
 */
#include "unity.h"
#include "qcsi/features.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const double PI = 3.14159265358979323846;

#define NSUB    8
#define NFRAMES 64
#define NFFT    64

static q15_t window[NFRAMES * NSUB];
static q15_t stats[NSUB * QCSI_STATS_PER_SUBCARRIER];
static qdsp_cplx_q15 scratch[NFFT];
static q31_t power[NFFT / 2];

void setUp(void) {}
void tearDown(void) {}

/** Fill one subcarrier with a constant plus a sinusoid of given bin. */
static void fill_subcarrier(size_t sub, double dc, double amp, double bin)
{
    size_t f;
    for (f = 0; f < NFRAMES; ++f) {
        double v = dc + amp * sin(2.0 * PI * bin * (double)f / (double)NFRAMES);
        window[f * NSUB + sub] = qdsp_f32_to_q15((float)v);
    }
}

/* ------------------------------------------------------------------ */
/* Statistics                                                          */
/* ------------------------------------------------------------------ */

static void test_stats_reject_bad_arguments(void)
{
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_amplitude_stats(NULL, NFRAMES, NSUB, stats));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_amplitude_stats(window, NFRAMES, NSUB, NULL));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_amplitude_stats(window, 0, NSUB, stats));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_amplitude_stats(window, NFRAMES, 0, stats));
}

static void test_constant_subcarrier_has_zero_spread(void)
{
    size_t k;
    for (k = 0; k < NSUB; ++k) {
        fill_subcarrier(k, 0.3, 0.0, 0.0);
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_amplitude_stats(window, NFRAMES, NSUB, stats));

    for (k = 0; k < NSUB; ++k) {
        TEST_ASSERT_INT16_WITHIN(2, qdsp_f32_to_q15(0.3f),
            stats[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_MEAN]);
        TEST_ASSERT_EQUAL_INT16(0,
            stats[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_STD]);
        TEST_ASSERT_EQUAL_INT16(0,
            stats[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_PTP]);
    }
}

static void test_mean_and_ptp_match_the_reference(void)
{
    size_t k;
    for (k = 0; k < NSUB; ++k) {
        fill_subcarrier(k, 0.25, 0.20, 3.0);
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_amplitude_stats(window, NFRAMES, NSUB, stats));

    for (k = 0; k < NSUB; ++k) {
        /* An integer number of periods, so the sinusoid averages out. */
        TEST_ASSERT_INT16_WITHIN(16, qdsp_f32_to_q15(0.25f),
            stats[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_MEAN]);
        /* Peak to peak of a sine of amplitude a is 2a. */
        TEST_ASSERT_INT16_WITHIN(64, qdsp_f32_to_q15(0.40f),
            stats[k * QCSI_STATS_PER_SUBCARRIER + QCSI_STAT_PTP]);
    }
}

static void test_std_matches_a_double_reference(void)
{
    size_t f;
    double sum = 0.0, sum_sq = 0.0, mean, var_ref, std_ref, std_got;

    fill_subcarrier(0, 0.30, 0.25, 5.0);
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_amplitude_stats(window, NFRAMES, NSUB, stats));

    for (f = 0; f < NFRAMES; ++f) {
        double x = (double)window[f * NSUB];
        sum += x;
        sum_sq += x * x;
    }
    mean = sum / (double)NFRAMES;
    var_ref = sum_sq / (double)NFRAMES - mean * mean;
    std_ref = sqrt(var_ref);
    std_got = (double)stats[QCSI_STAT_STD];

    printf("  [measured] std: %.1f (reference %.1f), ratio %.4f\n",
           std_got, std_ref, std_got / std_ref);
    TEST_ASSERT_TRUE_MESSAGE(fabs(std_got - std_ref) / std_ref < 0.01,
                             "standard deviation off by more than 1%");
}

/**
 * A small ripple riding on a large static component is the normal shape of a
 * CSI amplitude, and it is the case that breaks a naive implementation twice
 * over: once if the subtraction is done after truncation, and once if the
 * spread is stored as a squared quantity. Both are checked here.
 */
static void test_small_ripple_on_a_large_mean_survives(void)
{
    const double ripples[] = { 0.02, 0.01, 0.005 };
    size_t i;

    for (i = 0; i < sizeof(ripples) / sizeof(ripples[0]); ++i) {
        double expected = ripples[i] / sqrt(2.0) * 32768.0;
        fill_subcarrier(0, 0.90, ripples[i], 7.0);
        TEST_ASSERT_EQUAL_INT(QDSP_OK,
            qcsi_amplitude_stats(window, NFRAMES, NSUB, stats));

        printf("  [measured] ripple %.3f -> std %d (expected %.1f)\n",
               ripples[i], (int)stats[QCSI_STAT_STD], expected);
        TEST_ASSERT_TRUE_MESSAGE(stats[QCSI_STAT_STD] > 0,
                                 "spread collapsed to zero");
        TEST_ASSERT_TRUE_MESSAGE(
            fabs((double)stats[QCSI_STAT_STD] - expected) / expected < 0.05,
            "spread off by more than 5%");
    }
    TEST_ASSERT_INT16_WITHIN(200, qdsp_f32_to_q15(0.90f), stats[QCSI_STAT_MEAN]);
}

/* ------------------------------------------------------------------ */
/* Static component removal                                            */
/* ------------------------------------------------------------------ */

static void test_static_removal_zeroes_the_mean(void)
{
    size_t k, f;
    for (k = 0; k < NSUB; ++k) {
        fill_subcarrier(k, 0.2 + 0.05 * (double)k, 0.10, 4.0);
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_remove_static_component(window, NFRAMES, NSUB));

    for (k = 0; k < NSUB; ++k) {
        int64_t sum = 0;
        for (f = 0; f < NFRAMES; ++f) {
            sum += window[f * NSUB + k];
        }
        TEST_ASSERT_INT32_WITHIN(NFRAMES, 0, (int32_t)sum);
    }
}

static void test_static_removal_preserves_variation(void)
{
    size_t f;
    static q15_t before[NFRAMES];

    fill_subcarrier(0, 0.5, 0.15, 6.0);
    for (f = 0; f < NFRAMES; ++f) {
        before[f] = window[f * NSUB];
    }
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_remove_static_component(window, NFRAMES, NSUB));

    /* Differences between consecutive samples must be untouched: only a
       constant was removed. */
    for (f = 1; f < NFRAMES; ++f) {
        int32_t d_before = (int32_t)before[f] - (int32_t)before[f - 1];
        int32_t d_after = (int32_t)window[f * NSUB] -
                          (int32_t)window[(f - 1) * NSUB];
        TEST_ASSERT_INT32_WITHIN(2, d_before, d_after);
    }
}

/* ------------------------------------------------------------------ */
/* Doppler                                                             */
/* ------------------------------------------------------------------ */

static void test_doppler_rejects_bad_arguments(void)
{
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_doppler_power(NULL, NFRAMES, NSUB, 0, NFFT, scratch, power));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_doppler_power(window, NFRAMES, NSUB, NSUB, NFFT, scratch, power));
    /* not a power of two */
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_doppler_power(window, NFRAMES, NSUB, 0, 100, scratch, power));
    /* shorter than the window */
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG,
        qcsi_doppler_power(window, NFRAMES, NSUB, 0, 32, scratch, power));
}

static void test_doppler_peaks_at_the_injected_frequency(void)
{
    const double bins[] = { 3.0, 7.0, 11.0, 19.0 };
    size_t i;

    for (i = 0; i < sizeof(bins) / sizeof(bins[0]); ++i) {
        size_t found;
        fill_subcarrier(0, 0.0, 0.45, bins[i]);
        TEST_ASSERT_EQUAL_INT(QDSP_OK,
            qcsi_doppler_power(window, NFRAMES, NSUB, 0, NFFT, scratch, power));
        found = qcsi_dominant_doppler_bin(power, NFFT / 2);
        TEST_ASSERT_EQUAL_UINT32((uint32_t)bins[i], (uint32_t)found);
    }
}

static void test_doppler_ignores_the_static_component(void)
{
    size_t found;
    /* A large DC offset plus a weak tone: without removing the static part
       first, bin 0 would swamp everything. */
    fill_subcarrier(0, 0.7, 0.08, 9.0);
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_remove_static_component(window, NFRAMES, NSUB));
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_doppler_power(window, NFRAMES, NSUB, 0, NFFT, scratch, power));

    found = qcsi_dominant_doppler_bin(power, NFFT / 2);
    printf("  [measured] dominant bin with DC 0.7, tone 0.08 at bin 9: %d\n",
           (int)found);
    TEST_ASSERT_EQUAL_UINT32(9u, (uint32_t)found);
}

static void test_doppler_zero_padding_scales_the_peak_position(void)
{
    size_t found_short, found_padded;
    static qdsp_cplx_q15 big_scratch[128];
    static q31_t big_power[64];

    fill_subcarrier(0, 0.0, 0.4, 5.0);

    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_doppler_power(window, NFRAMES, NSUB, 0, 64, scratch, power));
    found_short = qcsi_dominant_doppler_bin(power, 32);

    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_doppler_power(window, NFRAMES, NSUB, 0, 128, big_scratch, big_power));
    found_padded = qcsi_dominant_doppler_bin(big_power, 64);

    /* Doubling the FFT length doubles the bin index of the same frequency. */
    printf("  [measured] peak bin: %d at N=64, %d at N=128\n",
           (int)found_short, (int)found_padded);
    TEST_ASSERT_UINT32_WITHIN(1u, (uint32_t)(2 * found_short),
                              (uint32_t)found_padded);
}

/**
 * Bin 0 is never a candidate, so it is free to mean "bad arguments" — but
 * only if a flat spectrum returns something else. It used to return 0 for
 * both, which made the two indistinguishable and contradicted the promise
 * that bin 0 is skipped.
 */
static void test_dominant_bin_handles_degenerate_input(void)
{
    static const q31_t flat[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    static const q31_t dc_only[8] = { 1000, 0, 0, 0, 0, 0, 0, 0 };

    TEST_ASSERT_EQUAL_UINT32(0u, (uint32_t)qcsi_dominant_doppler_bin(NULL, 8));
    TEST_ASSERT_EQUAL_UINT32(0u, (uint32_t)qcsi_dominant_doppler_bin(flat, 1));
    /* Nothing to choose between: the first candidate bin, not bin 0. */
    TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)qcsi_dominant_doppler_bin(flat, 8));
    /* All the energy in the bin the function refuses to return. */
    TEST_ASSERT_EQUAL_UINT32(1u,
        (uint32_t)qcsi_dominant_doppler_bin(dc_only, 8));
}

/* ------------------------------------------------------------------ */
/* Logarithmic compression                                             */
/* ------------------------------------------------------------------ */

/**
 * qcsi_log1p_q10() against log1p() from libm.
 *
 * The tolerance is the one the header states, one LSB of Q10, and it is
 * checked over the range the Doppler path actually produces: every small
 * value, where log1p differs most from log, and every power of two up to the
 * largest power a 64-point transform of full-scale Q15 amplitudes can reach.
 */
static void test_log1p_matches_libm(void)
{
    uint64_t x;
    int i;
    double worst = 0.0;
    uint64_t worst_at = 0u;

    for (x = 0u; x < 4096u; ++x) {
        double e = fabs((double)qcsi_log1p_q10(x) - log1p((double)x) * 1024.0);
        if (e > worst) { worst = e; worst_at = x; }
    }
    for (i = 0; i <= 43; ++i) {
        uint64_t p = (uint64_t)1 << i;
        uint64_t cases[3];
        int j;
        cases[0] = p - 1u;
        cases[1] = p;
        cases[2] = p + 1u;
        for (j = 0; j < 3; ++j) {
            double e = fabs((double)qcsi_log1p_q10(cases[j]) -
                            log1p((double)cases[j]) * 1024.0);
            if (e > worst) { worst = e; worst_at = cases[j]; }
        }
    }

    printf("  [measured] log1p worst error %.3f Q10 LSB (%.6f nats), "
           "at x = %llu\n", worst, worst / 1024.0,
           (unsigned long long)worst_at);
    TEST_ASSERT_TRUE_MESSAGE(worst <= 1.0,
                             "log1p is off by more than one Q10 LSB");
}

/**
 * The two properties the Doppler feature relies on: zero maps to zero, so a
 * silent bin stays silent, and the function is monotone, so ordering bins by
 * the compressed value orders them by power.
 */
static void test_log1p_is_zero_at_zero_and_monotone(void)
{
    uint64_t x;
    int32_t prev;

    TEST_ASSERT_EQUAL_INT32(0, qcsi_log1p_q10(0u));
    TEST_ASSERT_TRUE(qcsi_log1p_q10(1u) > 0);

    prev = -1;
    for (x = 0u; x < 100000u; x += 7u) {
        int32_t v = qcsi_log1p_q10(x);
        TEST_ASSERT_TRUE(v >= prev);
        prev = v;
    }
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_stats_reject_bad_arguments);
    RUN_TEST(test_constant_subcarrier_has_zero_spread);
    RUN_TEST(test_mean_and_ptp_match_the_reference);
    RUN_TEST(test_std_matches_a_double_reference);
    RUN_TEST(test_small_ripple_on_a_large_mean_survives);
    RUN_TEST(test_static_removal_zeroes_the_mean);
    RUN_TEST(test_static_removal_preserves_variation);
    RUN_TEST(test_doppler_rejects_bad_arguments);
    RUN_TEST(test_doppler_peaks_at_the_injected_frequency);
    RUN_TEST(test_doppler_ignores_the_static_component);
    RUN_TEST(test_doppler_zero_padding_scales_the_peak_position);
    RUN_TEST(test_dominant_bin_handles_degenerate_input);
    RUN_TEST(test_log1p_matches_libm);
    RUN_TEST(test_log1p_is_zero_at_zero_and_monotone);
    return UNITY_END();
}
