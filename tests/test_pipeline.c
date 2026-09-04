/**
 * Unit tests for the end-to-end pipeline.
 *
 * The stages are already tested individually. What is tested here is what
 * only appears once they are wired together: that the window boundary is
 * handled correctly, that a context carries no state between windows, that
 * two contexts do not interfere, and that the configuration validation
 * catches the mistakes that would otherwise produce a plausible wrong
 * answer rather than an error.
 */
#include "unity.h"
#include "qcsi/pipeline.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define NSUB    30
#define NANT    3
#define NFRAMES 32
#define NFFT    32
#define NDOP    8
#define NCLS    4

static const double PI = 3.14159265358979323846;

static qcsi_pipeline pipe;
static qcsi_pipeline_config cfg;
static qdsp_cplx_q15 frame[NSUB * NANT];

/* Feature count the configuration implies: 5 per subcarrier plus Doppler. */
#define NFEAT (NSUB * 5 + NDOP)

static q15_t weights[NCLS * NFEAT];
/* declared where it is used, in test_init_rejects_a_mismatched_model */

void setUp(void)
{
    cfg.n_sub = NSUB;
    cfg.n_frames = NFRAMES;
    cfg.n_fft = NFFT;
    cfg.n_doppler = NDOP;
    cfg.ant_a = 0;
    cfg.ant_b = 1;
    cfg.n_ant = NANT;
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_pipeline_init(&pipe, &cfg, NULL));
}

void tearDown(void) {}

/**
 * Build one frame with a common impairment on every antenna plus a physical
 * phase difference between antennas 0 and 1, and an amplitude modulated at
 * a chosen Doppler rate.
 */
static void make_frame(int t, double doppler_bin, double phys_scale)
{
    int a, k;
    double common_offset = 2.0 * sin(0.31 * (double)t);
    double common_slope = 0.05 * cos(0.17 * (double)t);
    double mod = 0.45 * (1.0 + 0.3 * sin(2.0 * PI * doppler_bin *
                                         (double)t / (double)NFRAMES));

    for (a = 0; a < NANT; ++a) {
        for (k = 0; k < NSUB; ++k) {
            double phys = (a == 0) ? phys_scale * sin(0.2 * (double)k) : 0.0;
            double ph = phys + common_offset + common_slope * (double)k;
            frame[a * NSUB + k].re = qdsp_f32_to_q15((float)(mod * cos(ph)));
            frame[a * NSUB + k].im = qdsp_f32_to_q15((float)(mod * sin(ph)));
        }
    }
}

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

static void test_init_rejects_bad_configuration(void)
{
    qcsi_pipeline_config bad;
    static qcsi_pipeline p2;

    bad = cfg; bad.n_sub = 0;
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &bad, NULL));
    /* One subcarrier is not merely useless, it is undefined: detrending fits
       a line across subcarriers, and qcsi_detrend() refuses n < 2. The
       pipeline discards that status, so accepting the configuration here
       would leave the phase silently undetrended. */
    bad = cfg; bad.n_sub = 1;
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &bad, NULL));
    bad = cfg; bad.n_sub = QCSI_MAX_SUBCARRIERS + 1;
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &bad, NULL));
    bad = cfg; bad.n_fft = 30;                  /* not a power of two */
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &bad, NULL));
    bad = cfg; bad.n_fft = 16;                  /* shorter than the window */
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &bad, NULL));
    bad = cfg; bad.n_doppler = NFFT;            /* more bins than exist */
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &bad, NULL));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, NULL, NULL));
}

/**
 * Equal antenna indices would make the conjugate product a constant zero
 * phase: a silent wrong answer rather than an error, and the kind of
 * configuration mistake that is easy to make and hard to notice.
 */
static void test_init_rejects_a_degenerate_antenna_pair(void)
{
    qcsi_pipeline_config bad;
    static qcsi_pipeline p2;

    bad = cfg; bad.ant_b = bad.ant_a;
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &bad, NULL));
    bad = cfg; bad.ant_b = NANT;                /* out of range */
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &bad, NULL));
    bad = cfg; bad.n_ant = 1;                   /* nothing to difference */
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &bad, NULL));
}

static void test_init_rejects_a_mismatched_model(void)
{
    static qcsi_pipeline p2;
    qcsi_linear_model m;

    /* A model expecting a different feature count must be refused, not
       silently read past the end of the feature vector. */
    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_init(&m, weights, NULL, NCLS, NFEAT - 1));
    TEST_ASSERT_EQUAL_INT(QDSP_ERR_ARG, qcsi_pipeline_init(&p2, &cfg, &m));

    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_init(&m, weights, NULL, NCLS, NFEAT));
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_pipeline_init(&p2, &cfg, &m));
}

/* ------------------------------------------------------------------ */
/* Window handling                                                     */
/* ------------------------------------------------------------------ */

static void test_window_completes_exactly_on_the_last_frame(void)
{
    int t, r = 0;
    for (t = 0; t < NFRAMES - 1; ++t) {
        make_frame(t, 5.0, 0.3);
        r = qcsi_pipeline_push(&pipe, frame);
        TEST_ASSERT_EQUAL_INT(QCSI_PIPELINE_NEED_MORE, r);
    }
    make_frame(NFRAMES - 1, 5.0, 0.3);
    r = qcsi_pipeline_push(&pipe, frame);
    TEST_ASSERT_EQUAL_INT(QCSI_PIPELINE_FEATURES_READY, r);
}

static void test_features_are_available_and_sized(void)
{
    uint16_t n = 0;
    const q15_t *f;
    int t;

    for (t = 0; t < NFRAMES; ++t) {
        make_frame(t, 5.0, 0.3);
        (void)qcsi_pipeline_push(&pipe, frame);
    }
    f = qcsi_pipeline_features(&pipe, &n);
    TEST_ASSERT_NOT_NULL(f);
    TEST_ASSERT_EQUAL_UINT16(NFEAT, n);

    /* Not every feature may be non-zero, but an all-zero vector would mean
       nothing reached the output. */
    {
        int k, nonzero = 0;
        for (k = 0; k < NFEAT; ++k) {
            if (f[k] != 0) nonzero++;
        }
        printf("  [measured] %d of %d features non-zero\n", nonzero, NFEAT);
        TEST_ASSERT_TRUE(nonzero > NFEAT / 4);
    }
}

/**
 * The second window must not be contaminated by the first. The phase
 * accumulators in particular are running sums, and forgetting to clear them
 * would make every window after the first wrong in a way that still looks
 * like plausible data.
 */
static void test_consecutive_windows_are_independent(void)
{
    static q15_t first[NFEAT];
    uint16_t n = 0;
    const q15_t *f;
    int t;

    for (t = 0; t < NFRAMES; ++t) {
        make_frame(t, 5.0, 0.3);
        (void)qcsi_pipeline_push(&pipe, frame);
    }
    f = qcsi_pipeline_features(&pipe, &n);
    memcpy(first, f, sizeof(first));

    /* Identical input again: identical features. */
    for (t = 0; t < NFRAMES; ++t) {
        make_frame(t, 5.0, 0.3);
        (void)qcsi_pipeline_push(&pipe, frame);
    }
    f = qcsi_pipeline_features(&pipe, &n);
    TEST_ASSERT_EQUAL_INT16_ARRAY(first, f, NFEAT);
}

static void test_reset_clears_a_partial_window(void)
{
    int t, r;
    for (t = 0; t < NFRAMES / 2; ++t) {
        make_frame(t, 5.0, 0.3);
        (void)qcsi_pipeline_push(&pipe, frame);
    }
    qcsi_pipeline_reset(&pipe);

    /* After the reset a full window must be needed again. */
    for (t = 0; t < NFRAMES - 1; ++t) {
        make_frame(t, 5.0, 0.3);
        r = qcsi_pipeline_push(&pipe, frame);
        TEST_ASSERT_EQUAL_INT(QCSI_PIPELINE_NEED_MORE, r);
    }
    make_frame(NFRAMES - 1, 5.0, 0.3);
    TEST_ASSERT_EQUAL_INT(QCSI_PIPELINE_FEATURES_READY,
                          qcsi_pipeline_push(&pipe, frame));
}

/**
 * Every other public entry point tolerates a null context; this one used to
 * dereference it, which is a crash in the one function an error path is most
 * likely to reach for.
 */
static void test_reset_tolerates_a_null_context(void)
{
    qcsi_pipeline_reset(NULL);
}

static void test_two_contexts_do_not_interfere(void)
{
    static qcsi_pipeline other;
    static q15_t solo[NFEAT];
    uint16_t n = 0;
    const q15_t *f;
    int t;

    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_pipeline_init(&other, &cfg, NULL));

    for (t = 0; t < NFRAMES; ++t) {
        make_frame(t, 5.0, 0.3);
        (void)qcsi_pipeline_push(&pipe, frame);
    }
    f = qcsi_pipeline_features(&pipe, &n);
    memcpy(solo, f, sizeof(solo));

    qcsi_pipeline_reset(&pipe);
    for (t = 0; t < NFRAMES; ++t) {
        make_frame(t, 5.0, 0.3);
        (void)qcsi_pipeline_push(&pipe, frame);
        make_frame(t, 11.0, 0.9);          /* different signal, other context */
        (void)qcsi_pipeline_push(&other, frame);
    }
    f = qcsi_pipeline_features(&pipe, &n);
    TEST_ASSERT_EQUAL_INT16_ARRAY(solo, f, NFEAT);
}

/* ------------------------------------------------------------------ */
/* Discrimination                                                      */
/* ------------------------------------------------------------------ */

/**
 * Different Doppler rates must produce different feature vectors. This is
 * not an accuracy claim, that needs recorded data, but without it the
 * pipeline could be wired up perfectly and still carry no information.
 */
static void test_different_motion_gives_different_features(void)
{
    static q15_t slow[NFEAT], fast[NFEAT];
    uint16_t n = 0;
    int t, k, differing = 0;

    for (t = 0; t < NFRAMES; ++t) {
        make_frame(t, 2.0, 0.3);
        (void)qcsi_pipeline_push(&pipe, frame);
    }
    memcpy(slow, qcsi_pipeline_features(&pipe, &n), sizeof(slow));

    for (t = 0; t < NFRAMES; ++t) {
        make_frame(t, 9.0, 0.3);
        (void)qcsi_pipeline_push(&pipe, frame);
    }
    memcpy(fast, qcsi_pipeline_features(&pipe, &n), sizeof(fast));

    for (k = 0; k < NFEAT; ++k) {
        if (slow[k] != fast[k]) differing++;
    }
    printf("  [measured] %d of %d features differ between motion rates\n",
           differing, NFEAT);
    TEST_ASSERT_TRUE_MESSAGE(differing > 0,
        "the pipeline produces the same features for different motion");
}

/* ------------------------------------------------------------------ */
/* Footprint                                                           */
/* ------------------------------------------------------------------ */

static void test_footprint_is_reported_and_plausible(void)
{
    size_t bytes = qcsi_pipeline_footprint();
    printf("  [measured] pipeline context: %lu bytes (%.1f KiB)\n",
           (unsigned long)bytes, (double)bytes / 1024.0);
    TEST_ASSERT_EQUAL_UINT32((uint32_t)sizeof(qcsi_pipeline), (uint32_t)bytes);
    TEST_ASSERT_TRUE(bytes > 0);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_init_rejects_bad_configuration);
    RUN_TEST(test_init_rejects_a_degenerate_antenna_pair);
    RUN_TEST(test_init_rejects_a_mismatched_model);
    RUN_TEST(test_window_completes_exactly_on_the_last_frame);
    RUN_TEST(test_features_are_available_and_sized);
    RUN_TEST(test_consecutive_windows_are_independent);
    RUN_TEST(test_reset_clears_a_partial_window);
    RUN_TEST(test_reset_tolerates_a_null_context);
    RUN_TEST(test_two_contexts_do_not_interfere);
    RUN_TEST(test_different_motion_gives_different_features);
    RUN_TEST(test_footprint_is_reported_and_plausible);
    return UNITY_END();
}
