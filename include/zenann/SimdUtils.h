#pragma once
#include <cstddef>

#if defined(ENABLE_SIMD)
#include <immintrin.h>
#endif

namespace zenann {

// L2 distance calculation with optional SIMD optimization
inline float l2_distance(const float* a, const float* b, size_t dim) {
#if defined(ENABLE_SIMD) && defined(__AVX2__)
    // SIMD version using AVX2
    const size_t step = 8;            // 8 × 32-bit floats
    __m256 acc       = _mm256_setzero_ps();
    size_t i         = 0;
    for (; i + step - 1 < dim; i += step) {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc         = _mm256_fmadd_ps(diff, diff, acc);   // acc += diff²
    }
    float buf[step];
    _mm256_storeu_ps(buf, acc);
    float d = 0.f;
    for (int j = 0; j < step; ++j) d += buf[j];

    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
#else
    // Naive version (no SIMD or AVX2 not available)
    float d = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
#endif
}

}
