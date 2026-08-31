#include "qcsi/pipeline.h"

#include <string.h>

/* Features produced per window:
 *   n_sub * 3   amplitude statistics (mean, std, peak-to-peak)
 *   n_sub * 2   phase difference statistics (mean, std)
 *   n_doppler   Doppler bins
 */
static uint16_t feature_count(const qcsi_pipeline_config *c)
{
    return (uint16_t)((uint32_t)c->n_sub * 5u + c->n_doppler);
}

qdsp_status_t qcsi_pipeline_init(qcsi_pipeline *p,
                                 const qcsi_pipeline_config *cfg,
                                 const qcsi_linear_model *model)
{
    uint16_t n_feat;

    if (p == NULL || cfg == NULL) {
        return QDSP_ERR_ARG;
    }
    /* build_features() below reads doppler[d + 1] for d in [0, n_doppler),
       i.e. spectrum indices [1, n_doppler] -- it skips bin 0 (the static
       component) and keeps the next n_doppler bins. qcsi_doppler_power()
       only ever writes indices [0, n_fft/2), so the highest index actually
       read, n_doppler, must stay below n_fft/2: n_doppler == n_fft/2 reads
       one bin past what was written (and, at n_fft == QCSI_MAX_FFT, one
       q31_t past the end of the doppler[] array itself). Hence the strict
       "<" rather than "<=" here. */
    if (cfg->n_sub == 0u || cfg->n_sub > QCSI_MAX_SUBCARRIERS ||
        cfg->n_frames < 2u || cfg->n_frames > QCSI_MAX_WINDOW ||
        cfg->n_fft > QCSI_MAX_FFT || !qdsp_fft_size_is_valid(cfg->n_fft) ||
        (uint32_t)cfg->n_fft < cfg->n_frames ||
        cfg->n_doppler == 0u || cfg->n_doppler >= (uint16_t)(cfg->n_fft / 2u)) {
        return QDSP_ERR_ARG;
    }
    /* The antenna pair must be two distinct antennas that exist. Equal
       indices would make the conjugate product a constant zero phase, which
       is a silent wrong answer rather than an error. */
    if (cfg->n_ant < 2u || cfg->ant_a >= cfg->n_ant ||
        cfg->ant_b >= cfg->n_ant || cfg->ant_a == cfg->ant_b) {
        return QDSP_ERR_ARG;
    }

    n_feat = feature_count(cfg);
    if (n_feat > QCSI_MAX_FEATURES) {
        return QDSP_ERR_ARG;
    }
    if (model != NULL) {
        if (model->n_classes > QCSI_MAX_CLASSES ||
            model->n_features != n_feat) {
            return QDSP_ERR_ARG;
        }
        p->model = *model;
    } else {
        p->model.weights = NULL;
        p->model.n_classes = 0u;
        p->model.n_features = 0u;
    }

    p->cfg = *cfg;
    p->n_features = n_feat;
    qcsi_pipeline_reset(p);
    return QDSP_OK;
}

void qcsi_pipeline_reset(qcsi_pipeline *p)
{
    uint16_t k;
    p->filled = 0u;
    for (k = 0u; k < p->cfg.n_sub; ++k) {
        p->phase_sum[k] = 0;
        p->phase_sum_sq[k] = 0;
    }
}

/**
 * Accumulate the phase difference for one frame.
 *
 * Phase statistics are accumulated per frame rather than stored: keeping the
 * whole phase history would cost n_frames * n_sub words on top of the
 * amplitude window, and the mean and standard deviation are all the feature
 * vector needs. This is the difference between a window that fits in RAM and
 * one that does not.
 */
static void accumulate_phase(qcsi_pipeline *p, const qdsp_cplx_q15 *frame)
{
    const qcsi_pipeline_config *c = &p->cfg;
    const qdsp_cplx_q15 *a = &frame[(size_t)c->ant_a * c->n_sub];
    const qdsp_cplx_q15 *b = &frame[(size_t)c->ant_b * c->n_sub];
    uint16_t k;

    (void)qcsi_phase_difference(a, b, p->phase_wrapped, c->n_sub);
    (void)qcsi_unwrap(p->phase_wrapped, p->phase_unwrapped, c->n_sub);
    (void)qcsi_detrend(p->phase_unwrapped, c->n_sub, NULL);

    for (k = 0u; k < c->n_sub; ++k) {
        int32_t v = p->phase_unwrapped[k];
        p->phase_sum[k] += v;
        p->phase_sum_sq[k] += (int64_t)v * (int64_t)v;
    }
}

/** Integer square root, as in features.c: no libm on the processing path. */
static uint32_t isqrt64(uint64_t v)
{
    uint64_t rem = 0u, root = 0u;
    int i;
    for (i = 0; i < 32; ++i) {
        root <<= 1;
        rem = (rem << 2) | (v >> 62);
        v <<= 2;
        if (root < rem) {
            rem -= root | 1u;
            root += 2u;
        }
    }
    return (uint32_t)(root >> 1);
}

static void build_features(qcsi_pipeline *p)
{
    const qcsi_pipeline_config *c = &p->cfg;
    uint16_t k, d;
    uint16_t at = 0u;

    /* Amplitude statistics, three per subcarrier. */
    (void)qcsi_amplitude_stats(p->amplitude, c->n_frames, c->n_sub,
                               &p->features[at]);
    at = (uint16_t)(at + c->n_sub * QCSI_STATS_PER_SUBCARRIER);

    /* Phase statistics, from the running accumulators. The variance uses
       the (n*sum_sq - sum^2)/n^2 form for the reason in docs/design.md: the
       textbook form squares a truncated mean and inflates the result. */
    for (k = 0u; k < c->n_sub; ++k) {
        int64_t n = (int64_t)c->n_frames;
        int64_t sum = (int64_t)p->phase_sum[k];
        int64_t var = (n * p->phase_sum_sq[k] - sum * sum) / (n * n);
        if (var < 0) var = 0;
        p->features[at + k] = qdsp_sat_q15((int32_t)(sum / n));
        p->features[at + c->n_sub + k] =
            qdsp_sat_q15((int32_t)isqrt64((uint64_t)var));
    }
    at = (uint16_t)(at + 2u * c->n_sub);

    /* Doppler: the static component is removed first, since each subcarrier
       carries a large fixed gain that would otherwise dominate bin zero.
       Bins are averaged across subcarriers to keep the feature count down. */
    (void)qcsi_remove_static_component(p->amplitude, c->n_frames, c->n_sub);
    for (d = 0u; d < c->n_doppler; ++d) {
        p->features[at + d] = 0;
    }
    for (k = 0u; k < c->n_sub; ++k) {
        (void)qcsi_doppler_power(p->amplitude, c->n_frames, c->n_sub, k,
                                 c->n_fft, p->fft_scratch, p->doppler);
        for (d = 0u; d < c->n_doppler; ++d) {
            /* Bin 0 is skipped: it holds whatever static component survived
               removal, which is not motion. Power is scaled down before it
               is folded in, so the running sum cannot overflow Q15. */
            int32_t v = (int32_t)(p->doppler[d + 1u] >> 15);
            p->features[at + d] = qdsp_add_q15(p->features[at + d],
                                               qdsp_sat_q15(v / (int32_t)c->n_sub));
        }
    }
}

int qcsi_pipeline_push(qcsi_pipeline *p, const qdsp_cplx_q15 *frame)
{
    const qcsi_pipeline_config *c;
    uint16_t k;

    if (p == NULL || frame == NULL) {
        return (int)QDSP_ERR_ARG;
    }
    c = &p->cfg;

    for (k = 0u; k < c->n_sub; ++k) {
        const qdsp_cplx_q15 z = frame[(size_t)c->ant_a * c->n_sub + k];
        p->amplitude[(size_t)p->filled * c->n_sub + k] =
            qcsi_magnitude_q15(z.re, z.im);
    }
    accumulate_phase(p, frame);

    p->filled = (uint16_t)(p->filled + 1u);
    if (p->filled < c->n_frames) {
        return QCSI_PIPELINE_NEED_MORE;
    }

    build_features(p);
    p->filled = 0u;
    for (k = 0u; k < c->n_sub; ++k) {
        p->phase_sum[k] = 0;
        p->phase_sum_sq[k] = 0;
    }

    if (p->model.weights == NULL) {
        return QCSI_PIPELINE_FEATURES_READY;
    }
    return qcsi_linear_predict(&p->model, p->features, p->scores);
}

const q15_t *qcsi_pipeline_features(const qcsi_pipeline *p, uint16_t *n_out)
{
    if (p == NULL) {
        return NULL;
    }
    if (n_out != NULL) {
        *n_out = p->n_features;
    }
    return p->features;
}

size_t qcsi_pipeline_footprint(void)
{
    return sizeof(qcsi_pipeline);
}
