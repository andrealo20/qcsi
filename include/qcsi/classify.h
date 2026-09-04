/**
 * @file classify.h
 * @brief Quantised linear classifier for fixed-point feature vectors.
 *
 * A linear model, one dot product per class, then argmax. That is a modest
 * choice and a deliberate one: it keeps the whole pipeline's cost visible
 * and interpretable, and when the fixed-point result diverges from the
 * floating-point reference the cause is arithmetic rather than training
 * dynamics.
 *
 * ## Quantisation scheme
 *
 * Weights and features are both Q15, with one scale factor per model rather
 * than per class or per feature. Per-class scales would be more accurate but
 * would mean the class scores are no longer directly comparable without
 * rescaling each one, and an argmax over rescaled values is exactly where a
 * subtle bug would hide. One scale keeps the comparison exact.
 *
 * The dot product accumulates in 64 bits and is never saturated on the way,
 * so the score of a class does not depend on the order its features are
 * summed in. Scores are returned in Q30, the natural format of a sum of Q15
 * products, and are only meaningful relative to one another.
 *
 * ## Memory model
 *
 * As everywhere in qdsp and qcsi, nothing is allocated. The caller owns the
 * weight matrix, which is `n_classes * n_features` Q15 values in
 * class-major order, and the bias vector, which is `n_classes` values in
 * Q30 so that it lands on the same scale as the accumulated dot product.
 */
#ifndef QCSI_CLASSIFY_H
#define QCSI_CLASSIFY_H

#include <stddef.h>
#include <stdint.h>

#include "qdsp/fixed.h"
#include "qdsp/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const q15_t *weights;   /**< n_classes * n_features, class-major, Q15 */
    const q63_t *bias;      /**< n_classes, Q30 */
    uint16_t     n_classes;
    uint16_t     n_features;
} qcsi_linear_model;

/**
 * Validate and store a model description.
 * @return QDSP_OK, or QDSP_ERR_ARG on a null pointer or a zero dimension.
 *         The bias may be NULL, in which case it is treated as zero.
 */
qdsp_status_t qcsi_linear_init(qcsi_linear_model *m, const q15_t *weights,
                               const q63_t *bias, uint16_t n_classes,
                               uint16_t n_features);

/**
 * Score every class for one feature vector.
 *
 * @param features  n_features Q15 values.
 * @param scores    n_classes Q30 values, comparable to one another.
 */
qdsp_status_t qcsi_linear_scores(const qcsi_linear_model *m,
                                 const q15_t *features, q63_t *scores);

/**
 * Predicted class index for one feature vector.
 *
 * Ties are broken towards the lower index, deterministically. A tie is
 * vanishingly unlikely in floating point but perfectly possible once scores
 * are quantised, so leaving it unspecified would make the C and Python
 * implementations disagree at random on exactly the samples a parity test
 * is looking at.
 *
 * @param scratch  optional n_classes buffer for the scores; may be NULL,
 *                 in which case the scores are discarded as they are used.
 * @return the class index, or a negative qdsp_status_t on bad arguments.
 */
int qcsi_linear_predict(const qcsi_linear_model *m, const q15_t *features,
                        q63_t *scratch);

/**
 * Margin between the best and second-best class, in Q30.
 *
 * The margin is what predicts whether quantisation will flip a decision: a
 * sample whose top two scores are close changes class under a tiny
 * perturbation, and one with a wide margin does not. Reporting the margin
 * distribution alongside the accuracy explains *which* samples the
 * fixed-point pipeline loses, rather than only how many.
 */
q63_t qcsi_linear_margin(const qcsi_linear_model *m, const q15_t *features,
                         q63_t *scratch);

#ifdef __cplusplus
}
#endif

#endif /* QCSI_CLASSIFY_H */
