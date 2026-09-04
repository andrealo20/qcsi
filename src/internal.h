/**
 * @file internal.h
 * @brief Helpers shared between translation units, not part of the API.
 *
 * Nothing here is installed or included by a public header. It exists so that
 * a routine needed by two source files has one definition rather than two
 * copies that can drift apart.
 */
#ifndef QCSI_INTERNAL_H
#define QCSI_INTERNAL_H

#include <stdint.h>

/**
 * Integer square root of a 64-bit value, by binary digit-by-digit descent.
 *
 * Used instead of sqrt() from libm because the whole processing path is meant
 * to run without floating point: pulling in the soft-float library for one
 * square root would be self-defeating on a target with no FPU. The loop runs a
 * fixed number of iterations and uses only shifts, adds and compares.
 *
 * static inline rather than a symbol in one .c file: the routine is four lines
 * of arithmetic, both callers are on the processing path, and this way there
 * is exactly one definition to read and to change.
 */
static inline uint32_t qcsi_isqrt64(uint64_t v)
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

#endif /* QCSI_INTERNAL_H */
