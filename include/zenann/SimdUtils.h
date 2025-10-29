#pragma once
#include <cstddef>

namespace zenann {
inline float l2_naive(const float* a,
                      const float* b,
                      size_t dim) {
    float d = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

}
