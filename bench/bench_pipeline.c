/**
 * Cost and footprint of one pipeline, reported per frame.
 *
 * Operation counts are exact and architecture-independent; they are what CI
 * can watch for regressions. They are NOT cycle counts. Real cycle figures
 * need a cross-compiler and either Renode or hardware, neither of which this
 * measurement claims to substitute for.
 *
 * Two passes are run, over identical input, because the front end alone is
 * not what a deployment pays. The classifier is the last stage of the diagram
 * and its cost is proportional to the class count, which the front end's is
 * not: reporting only the front end understates a 150-class model by more
 * than the whole front end.
 */
#include "qcsi/pipeline.h"
#include "qdsp/profile.h"
#include <stdio.h>
#include <math.h>

#define NSUB 30
#define NANT 3
#define NFRAMES 32
#define NFFT 32
#define NDOP 8
#define NFEAT (NSUB*5 + NDOP)
#define NCLS QCSI_MAX_CLASSES

/* The configuration behind the accuracies in the README, for the analytic
   classifier cost only: it needs -DQCSI_MAX_CLASSES=150 to run. */
#define REPORTED_CLASSES  150
#define REPORTED_FEATURES 166

static qcsi_pipeline front;
static qcsi_pipeline full;
static qcsi_linear_model model;
static q15_t weights[NCLS*NFEAT];
static qdsp_cplx_q15 frame[NSUB*NANT];

static void make_frame(int t){
    int a,k;
    for(a=0;a<NANT;++a) for(k=0;k<NSUB;++k){
        double ph=0.3*sin(0.2*k)+2.0*sin(0.31*t);
        frame[a*NSUB+k].re=qdsp_f32_to_q15((float)(0.45*cos(ph)));
        frame[a*NSUB+k].im=qdsp_f32_to_q15((float)(0.45*sin(ph)));
    }
}

static qdsp_profile_t run(qcsi_pipeline *p){
    int t;
    qdsp_profile_reset();
    for(t=0;t<NFRAMES;++t){ make_frame(t); qcsi_pipeline_push(p,frame); }
    return qdsp_profile_get();
}

int main(void){
    qcsi_pipeline_config cfg = {NSUB,NFRAMES,NFFT,NDOP,0,1,NANT};
    qdsp_profile_t pf,pc;
    int i;

    for(i=0;i<NCLS*NFEAT;++i) weights[i]=(q15_t)(((i*37)%2001)-1000);

    if (qcsi_pipeline_init(&front,&cfg,NULL)!=QDSP_OK){puts("init failed");return 1;}
    if (qcsi_linear_init(&model,weights,NULL,NCLS,NFEAT)!=QDSP_OK){
        puts("model failed");return 1;}
    if (qcsi_pipeline_init(&full,&cfg,&model)!=QDSP_OK){puts("init failed");return 1;}

    printf("# qcsi - cost and footprint\n\n");
    printf("Configuration: %d subcarriers, %d antennas, %d-frame window, "
           "%d-point Doppler FFT, %d bins kept, %d features.\n\n",
           NSUB,NANT,NFRAMES,NFFT,NDOP,NFEAT);

    printf("## Static footprint\n\n");
    printf("| Buffer | Bytes |\n|---|---|\n");
    printf("| amplitude window | %lu |\n",(unsigned long)(sizeof(q15_t)*NFRAMES*NSUB));
    printf("| phase accumulators | %lu |\n",
           (unsigned long)((sizeof(int32_t)+sizeof(int64_t)+sizeof(int32_t)+sizeof(qcsi_angle_t))*NSUB));
    printf("| FFT scratch | %lu |\n",(unsigned long)(sizeof(qdsp_cplx_q15)*NFFT));
    printf("| features + scores | %lu |\n",
           (unsigned long)(sizeof(q15_t)*QCSI_MAX_FEATURES+sizeof(q63_t)*QCSI_MAX_CLASSES));
    printf("| **context total** | **%lu** (%.1f KiB) |\n",
           (unsigned long)qcsi_pipeline_footprint(),
           qcsi_pipeline_footprint()/1024.0);
    printf("| weight table, %d classes | %lu |\n",
           NCLS,(unsigned long)(sizeof(q15_t)*NCLS*NFEAT));
    printf("| weight table, %d classes x %d features | %lu |\n\n",
           REPORTED_CLASSES,REPORTED_FEATURES,
           (unsigned long)(sizeof(q15_t)*REPORTED_CLASSES*REPORTED_FEATURES));
    printf("The weight table is the caller's, not part of the context, and at "
           "the class count behind the reported accuracies it is larger than "
           "everything above put together.\n\n");
    printf("Compile-time limits are generous (%d subcarriers, %d-frame window). "
           "Sized exactly for this configuration the context would be far "
           "smaller; the limits are what make the struct static rather than "
           "allocated.\n\n",QCSI_MAX_SUBCARRIERS,QCSI_MAX_WINDOW);

    printf("## Cost per frame\n\n");
    if(!QDSP_PROFILE_ENABLED){
        printf("> Operation counting is off: rebuild with -DQCSI_PROFILE=ON.\n");
        return 0;
    }

    pf=run(&front);
    pc=run(&full);

    printf("| Operation | Per window | Per frame |\n|---|---|---|\n");
    printf("| multiply-accumulate, front end | %llu | %.1f |\n",
           (unsigned long long)pf.mac,(double)pf.mac/NFRAMES);
    printf("| multiply-accumulate, front end + %d-class classifier | %llu | %.1f |\n",
           NCLS,(unsigned long long)pc.mac,(double)pc.mac/NFRAMES);
    printf("| rounding multiply | %llu | %.1f |\n",
           (unsigned long long)pf.mul,(double)pf.mul/NFRAMES);
    printf("| accumulator to q15 | %llu | %.1f |\n",
           (unsigned long long)pf.round,(double)pf.round/NFRAMES);

    printf("\nThe classifier costs exactly n_classes * n_features "
           "multiply-accumulates per window, one dot product per class with "
           "nothing data-dependent in it: %d * %d = %d here, measured %llu, "
           "and %d * %d = %d for the %d-class configuration the reported "
           "accuracies come from.\n",
           NCLS,NFEAT,NCLS*NFEAT,(unsigned long long)(pc.mac-pf.mac),
           REPORTED_CLASSES,REPORTED_FEATURES,REPORTED_CLASSES*REPORTED_FEATURES,
           REPORTED_CLASSES);
    printf("\nExact and architecture-independent. Not cycle counts.\n");
    return 0;
}
