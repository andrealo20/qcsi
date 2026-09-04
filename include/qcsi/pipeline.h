/**
 * @file pipeline.h
 * @brief End-to-end CSI frame processing, in the shape firmware needs.
 *
 * Everything else in this library is a stage. This is the single entry point
 * that takes a window of raw CSI frames and returns a class, which is what
 * an application actually wants and what makes the library usable without
 * reading its internals.
 *
 * ## Memory model
 *
 * One context struct holds every buffer, and the caller supplies it. There
 * is no allocation anywhere, and no hidden state: two contexts can run
 * concurrently on different antenna pairs or different windows without
 * interfering. The struct is large: it is dominated by the window and the
 * FFT scratch, so it belongs in static storage rather than on a stack.
 *
 * `qcsi_pipeline_footprint()` reports the byte count, because a number a
 * caller can query is more useful than one buried in a header comment, and
 * it stays correct when the configuration changes.
 *
 * ## Cost
 *
 * Built with -DQDSP_PROFILE=ON, the qdsp operation counters report the
 * multiply-accumulates and roundings per frame. Those figures are exact and
 * architecture-independent. They are not cycle counts, and this library does
 * not claim to produce cycle counts: see docs/design.md.
 */
#ifndef QCSI_PIPELINE_H
#define QCSI_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "qdsp/fft.h"
#include "qdsp/fixed.h"
#include "qdsp/status.h"

#include "qcsi/classify.h"
#include "qcsi/features.h"
#include "qcsi/phase.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compile-time limits. Sized for the Intel 5300 (30 subcarriers, 3
   antennas); raising them costs static memory and nothing else. */
#ifndef QCSI_MAX_SUBCARRIERS
#define QCSI_MAX_SUBCARRIERS 64
#endif
#ifndef QCSI_MAX_WINDOW
#define QCSI_MAX_WINDOW 256
#endif
#ifndef QCSI_MAX_FFT
#define QCSI_MAX_FFT 64
#endif
/* 32 classes is the right default for a microcontroller: the weight table is
   n_classes * n_features * 2 bytes and it is by far the largest thing a
   deployment carries. It is a limit on the model, not on the front end, and
   raising it costs only the score array here plus the caller's own table.

   The SignFi lab_150 experiment behind the accuracies in the README has 150
   classes and does not fit the default: build it with
   -DQCSI_MAX_CLASSES=150, and budget 150 * 166 * 2 = 49800 bytes, about 48.6
   KiB, for the weights themselves. */
#ifndef QCSI_MAX_CLASSES
#define QCSI_MAX_CLASSES 32
#endif
#ifndef QCSI_MAX_FEATURES
#define QCSI_MAX_FEATURES 256
#endif

typedef struct {
    uint16_t n_sub;        /**< subcarriers per antenna */
    uint16_t n_frames;     /**< frames per decision window */
    uint16_t n_fft;        /**< Doppler transform length, power of two */
    uint16_t n_doppler;    /**< Doppler bins kept as features */
    uint8_t  ant_a;        /**< first antenna of the conjugate pair */
    uint8_t  ant_b;        /**< second antenna; measure the pair first */
    uint8_t  n_ant;        /**< antennas present in the input */
} qcsi_pipeline_config;

typedef struct {
    qcsi_pipeline_config cfg;
    qcsi_linear_model    model;

    uint16_t             filled;      /**< frames accumulated so far */
    uint16_t             n_features;  /**< length of the feature vector */

    /* Ring of amplitudes, frame-major, and the phase working buffers. */
    q15_t         amplitude[QCSI_MAX_WINDOW * QCSI_MAX_SUBCARRIERS];
    qcsi_angle_t  phase_wrapped[QCSI_MAX_SUBCARRIERS];
    int32_t       phase_unwrapped[QCSI_MAX_SUBCARRIERS];
    int32_t       phase_sum[QCSI_MAX_SUBCARRIERS];
    int64_t       phase_sum_sq[QCSI_MAX_SUBCARRIERS];

    qdsp_cplx_q15 fft_scratch[QCSI_MAX_FFT];
    q31_t         doppler[QCSI_MAX_FFT / 2];
    q15_t         features[QCSI_MAX_FEATURES];
    q63_t         scores[QCSI_MAX_CLASSES];
} qcsi_pipeline;

/**
 * Configure a pipeline and attach a model.
 *
 * The model may be NULL, in which case the pipeline computes features and
 * qcsi_pipeline_push() returns QCSI_PIPELINE_FEATURES_READY instead of a
 * class. That is the mode to use when collecting training data on target.
 *
 * n_sub must be at least 2: the phase is detrended across subcarriers, and a
 * line fit through a single point is not defined.
 *
 * @return QDSP_OK, or QDSP_ERR_ARG if any dimension is below its minimum or
 *         exceeds its compile-time limit, the antenna indices are out of
 *         range or equal, or n_fft is not a valid FFT size.
 */
qdsp_status_t qcsi_pipeline_init(qcsi_pipeline *p,
                                 const qcsi_pipeline_config *cfg,
                                 const qcsi_linear_model *model);

/** Discard accumulated frames, keeping the configuration and model.
 *  A null pointer is ignored, as everywhere else in the API. */
void qcsi_pipeline_reset(qcsi_pipeline *p);

/** Returned by qcsi_pipeline_push() while the window is still filling. */
#define QCSI_PIPELINE_NEED_MORE     (-100)
/** Returned when a window completed but no model was attached. */
#define QCSI_PIPELINE_FEATURES_READY (-101)

/**
 * Feed one frame of CSI.
 *
 * @param frame  n_sub * n_ant complex samples, subcarrier-major within each
 *               antenna: frame[a * n_sub + k].
 * @return a class index once a window is complete, QCSI_PIPELINE_NEED_MORE
 *         while filling, QCSI_PIPELINE_FEATURES_READY if a window completed
 *         without a model, or a negative qdsp_status_t on bad arguments.
 *
 * The window does not slide: a completed window is consumed and the next
 * one starts empty. A sliding window would produce a decision per frame at
 * n_frames times the cost, which on a microcontroller is a decision the
 * application should make explicitly rather than inherit.
 */
int qcsi_pipeline_push(qcsi_pipeline *p, const qdsp_cplx_q15 *frame);

/** Feature vector from the most recently completed window. */
const q15_t *qcsi_pipeline_features(const qcsi_pipeline *p, uint16_t *n_out);

/** Static footprint of one pipeline context, in bytes. */
size_t qcsi_pipeline_footprint(void);

#ifdef __cplusplus
}
#endif

#endif /* QCSI_PIPELINE_H */
