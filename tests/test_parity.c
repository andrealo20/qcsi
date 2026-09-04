/**
 * Parity between the C pipeline and the floating-point reference.
 *
 * Every other test in this directory checks that an operator computes what it
 * claims. This one checks something the others cannot: that the C and the
 * reference in tools/qcsi_data.py compute the *same* feature vector, in the
 * same order and on the same scale, and that a model fitted against one can be
 * handed to the other.
 *
 * That claim used to be made only in the reference's docstrings, and it was
 * wrong twice over: the phase statistics came out interleaved on one side and
 * blocked on the other, and the Doppler bins were log-compressed on one side
 * and not on the other. Neither showed up, because no test compared the two.
 *
 * The vectors need no dataset: tools/gen_parity_vectors.py synthesises the
 * frames from a fixed seed and runs the reference over them, so this runs in
 * CI on every commit. The tolerances are per block and are set from the
 * measured disagreement; the observed maxima are printed below, so headroom
 * disappearing is visible rather than silent.
 */
#include "unity.h"
#include "qcsi/pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vectors/parity.h"

#define FRAME_SAMPLES (PARITY_N_ANT * PARITY_N_SUB)

static qcsi_pipeline pipe;
static qcsi_linear_model model;
static qcsi_pipeline_config cfg;
static qdsp_cplx_q15 frame[FRAME_SAMPLES];

/* Worst disagreement seen so far, per block, reported at the end. */
static int worst_amplitude;
static int worst_phase;
static int worst_doppler;

void setUp(void)
{
    cfg.n_sub = PARITY_N_SUB;
    cfg.n_frames = PARITY_N_FRAMES;
    cfg.n_fft = PARITY_N_FFT;
    cfg.n_doppler = PARITY_N_DOPPLER;
    cfg.ant_a = PARITY_ANT_A;
    cfg.ant_b = PARITY_ANT_B;
    cfg.n_ant = PARITY_N_ANT;

    TEST_ASSERT_EQUAL_INT(QDSP_OK,
        qcsi_linear_init(&model, parity_weights, parity_bias,
                         PARITY_N_CLASSES, PARITY_N_FEATURES));
    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_pipeline_init(&pipe, &cfg, &model));
}

void tearDown(void) {}

/**
 * Push one window into a pipeline and return what the last frame produced.
 *
 * The header stores re and im interleaved in the order the pipeline reads
 * them, frame[a * n_sub + k]. They are copied into a qdsp_cplx_q15 buffer
 * rather than cast in place: the layouts happen to agree, but reading an
 * int16_t array through a struct type is not something the aliasing rules
 * promise anything about, and the copy costs nothing here.
 */
static int push_window(qcsi_pipeline *p, int w)
{
    int t, i, r = 0;

    for (t = 0; t < PARITY_N_FRAMES; ++t) {
        const int16_t *src =
            &parity_frames[(w * PARITY_N_FRAMES + t) * FRAME_SAMPLES * 2];
        for (i = 0; i < FRAME_SAMPLES; ++i) {
            frame[i].re = src[2 * i];
            frame[i].im = src[2 * i + 1];
        }
        r = qcsi_pipeline_push(p, frame);
    }
    return r;
}

/** Tolerance for feature index i, which depends on the block it is in. */
static int tolerance_at(int i)
{
    if (i < PARITY_N_SUB * 3) {
        return PARITY_TOL_AMPLITUDE;
    }
    if (i < PARITY_N_SUB * 5) {
        return PARITY_TOL_PHASE;
    }
    return PARITY_TOL_DOPPLER;
}

static void record(int i, int diff)
{
    if (i < PARITY_N_SUB * 3) {
        if (diff > worst_amplitude) worst_amplitude = diff;
    } else if (i < PARITY_N_SUB * 5) {
        if (diff > worst_phase) worst_phase = diff;
    } else {
        if (diff > worst_doppler) worst_doppler = diff;
    }
}

/* ------------------------------------------------------------------ */
/* Features                                                            */
/* ------------------------------------------------------------------ */

/**
 * The feature vector, element by element against the reference.
 *
 * Ordering is not checked separately because it does not need to be: a
 * permuted block misses by thousands of LSB, not by the tens the tolerances
 * allow, and so does a block that skipped the log compression.
 */
static void test_features_match_the_reference(void)
{
    int w, i;

    for (w = 0; w < PARITY_N_WINDOWS; ++w) {
        const q15_t *got;
        uint16_t n = 0;

        (void)push_window(&pipe, w);
        got = qcsi_pipeline_features(&pipe, &n);
        TEST_ASSERT_EQUAL_UINT16(PARITY_N_FEATURES, n);

        for (i = 0; i < PARITY_N_FEATURES; ++i) {
            int expected = (int)parity_features[w * PARITY_N_FEATURES + i];
            int diff = abs((int)got[i] - expected);

            record(i, diff);
            if (diff > tolerance_at(i)) {
                char msg[96];
                (void)snprintf(msg, sizeof(msg),
                               "window %d feature %d: got %d, reference %d",
                               w, i, (int)got[i], expected);
                TEST_FAIL_MESSAGE(msg);
            }
        }
    }

    printf("  [measured] worst disagreement: amplitude %d of %d allowed, "
           "phase %d of %d, Doppler %d of %d\n",
           worst_amplitude, PARITY_TOL_AMPLITUDE,
           worst_phase, PARITY_TOL_PHASE,
           worst_doppler, PARITY_TOL_DOPPLER);
}

/* ------------------------------------------------------------------ */
/* Classification                                                      */
/* ------------------------------------------------------------------ */

/**
 * The predicted class, which is the claim that matters.
 *
 * "The features are close" is a weaker statement than "a model fitted against
 * one pipeline works on the other". The generator draws a model whose smallest
 * decision margin is more than eight times PARITY_SCORE_BOUND, the largest
 * amount the feature tolerances can move a score, so a matching prediction
 * here follows from the arithmetic rather than from luck. The margin is
 * asserted as well, otherwise a later regeneration could quietly produce a
 * model for which the prediction proves nothing.
 */
static void test_predictions_match_the_reference(void)
{
    int w;

    for (w = 0; w < PARITY_N_WINDOWS; ++w) {
        q63_t margin, drift;
        int r = push_window(&pipe, w);
        const q15_t *got = qcsi_pipeline_features(&pipe, NULL);

        TEST_ASSERT_EQUAL_INT((int)parity_prediction[w], r);

        margin = qcsi_linear_margin(&model, got, NULL);
        drift = margin - parity_margin[w];
        if (drift < 0) {
            drift = -drift;
        }
        printf("  [measured] window %d: class %d, margin %lld, "
               "%.0f times the score bound\n",
               w, r, (long long)margin,
               (double)margin / (double)PARITY_SCORE_BOUND);
        TEST_ASSERT_TRUE_MESSAGE(margin > 2 * PARITY_SCORE_BOUND,
            "the margin is narrow enough that the feature tolerance could "
            "flip this prediction: the class assertion proves nothing");
        /* The margin itself must land where the reference put it. Two scores
           can each move by at most the bound, so their difference by at most
           twice it; more than that means the features agreed only because the
           tolerances are loose. */
        TEST_ASSERT_TRUE_MESSAGE(drift <= 2 * PARITY_SCORE_BOUND,
            "the decision margin drifted further than the feature tolerances "
            "can account for");
    }
}

/**
 * Without a model the pipeline must produce the same features and say so.
 *
 * The two paths through qcsi_pipeline_push() differ only in what they return,
 * and this is what says the feature path does not depend on a model being
 * attached — which is the mode the tools use to collect training data on
 * target, so it is the mode the reference has to match.
 */
static void test_features_are_the_same_without_a_model(void)
{
    static qcsi_pipeline bare;
    static q15_t with_model[PARITY_N_FEATURES];
    int r;

    (void)push_window(&pipe, 0);
    memcpy(with_model, qcsi_pipeline_features(&pipe, NULL),
           sizeof(with_model));

    TEST_ASSERT_EQUAL_INT(QDSP_OK, qcsi_pipeline_init(&bare, &cfg, NULL));
    r = push_window(&bare, 0);
    TEST_ASSERT_EQUAL_INT(QCSI_PIPELINE_FEATURES_READY, r);
    TEST_ASSERT_EQUAL_INT16_ARRAY(with_model,
                                  qcsi_pipeline_features(&bare, NULL),
                                  PARITY_N_FEATURES);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_features_match_the_reference);
    RUN_TEST(test_predictions_match_the_reference);
    RUN_TEST(test_features_are_the_same_without_a_model);
    return UNITY_END();
}
