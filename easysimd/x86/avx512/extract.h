/* SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright:
 *   2020      Evan Nemerson <evan@nemerson.com>
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_X86_AVX512_EXTRACT_H)
#define EASYSIMD_X86_AVX512_EXTRACT_H

#include "types.h"
#include "mov.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_mask_extractf32x4_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m256 a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[imm8 & 1], src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd__m256_private a_ = easysimd__m256_to_private(a);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * (sizeof(r_.f32) / sizeof(r_.f32[0]));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i + offset] : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_extractf32x4_ps(src, k, a, imm8) _mm256_mask_extractf32x4_ps(src, k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_extractf32x4_ps
  #define _mm256_mask_extractf32x4_ps(src, k, a, imm8) easysimd_mm256_mask_extractf32x4_ps(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_maskz_extractf32x4_ps (easysimd__mmask8 k, easysimd__m256 a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[imm8 & 1], svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m256_private a_ = easysimd__m256_to_private(a);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * (sizeof(r_.f32) / sizeof(r_.f32[0]));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i + offset] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_extractf32x4_ps(k, a, imm8) _mm256_maskz_extractf32x4_ps(k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_extractf32x4_ps
  #define _mm256_maskz_extractf32x4_ps(k, a, imm8) easysimd_mm256_maskz_extractf32x4_ps(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm256_mask_extractf64x2_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m256d a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[imm8 & 1], src.sve_f64);
    return r;
  #else
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      r_;

    const size_t n = sizeof(r_.f64) / sizeof(r_.f64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i + offset] : src_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mask_extractf64x2_pd(src, k, a, imm8) _mm256_mask_extractf64x2_pd(src, k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_extractf64x2_pd
  #define _mm256_mask_extractf64x2_pd(src, k, a, imm8) easysimd_mm256_mask_extractf64x2_pd(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm256_maskz_extractf64x2_pd (easysimd__mmask8 k, easysimd__m256d a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[imm8 & 1], svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);
    easysimd__m128d_private r_;

    const size_t n = sizeof(r_.f64) / sizeof(r_.f64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i + offset] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_maskz_extractf64x2_pd(k, a, imm8) _mm256_maskz_extractf64x2_pd(k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_extractf64x2_pd
  #define _mm256_maskz_extractf64x2_pd(k, a, imm8) easysimd_mm256_maskz_extractf64x2_pd(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm512_extractf32x4_ps (easysimd__m512 a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = a.sve_f32[imm8 & 3];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128i[imm8 & 3];
  /* GCC 6 generates an ICE */
  #elif defined(HEDLEY_GCC_VERSION) && !HEDLEY_GCC_VERSION_CHECK(7,0,0)
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    return a_.m128[imm8 & 3];
  #else
    easysimd__m128_private r_;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 3) * (sizeof(r_.f32) / sizeof(r_.f32[0]));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = a_.f32[i + offset];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_extractf32x4_ps(a, imm8) _mm512_extractf32x4_ps(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_extractf32x4_ps
  #define _mm512_extractf32x4_ps(a, imm8) easysimd_mm512_extractf32x4_ps(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm512_mask_extractf32x4_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m512 a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[imm8 & 3], src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 3) * (sizeof(r_.f32) / sizeof(r_.f32[0]));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i + offset] : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_mask_extractf32x4_ps(src, k, a, imm8) _mm512_mask_extractf32x4_ps(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_mask_extractf32x4_ps(src, k, a, imm8) easysimd_mm_mask_mov_ps(src, k, easysimd_mm512_extractf32x4_ps(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_extractf32x4_ps
  #define _mm512_mask_extractf32x4_ps(src, k, a, imm8) easysimd_mm512_mask_extractf32x4_ps(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm512_maskz_extractf32x4_ps (easysimd__mmask8 k, easysimd__m512 a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[imm8 & 3], svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 3) * (sizeof(r_.f32) / sizeof(r_.f32[0]));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i + offset] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_maskz_extractf32x4_ps(k, a, imm8) _mm512_maskz_extractf32x4_ps(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_maskz_extractf32x4_ps(k, a, imm8) easysimd_mm_maskz_mov_ps(k, easysimd_mm512_extractf32x4_ps(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_extractf32x4_ps
  #define _mm512_maskz_extractf32x4_ps(k, a, imm8) easysimd_mm512_maskz_extractf32x4_ps(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_extractf32x8_ps (easysimd__m512 a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32[(imm8 & 1 ) << 1];
    r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    int id = (imm8 & 1) << 1;
    r.m128[0] = a.m128[id];
    r.m128[1] = a.m128[id | 1];
    return r;
  /* GCC 6 generates an ICE */ 
  #elif defined(HEDLEY_GCC_VERSION) && !HEDLEY_GCC_VERSION_CHECK(7,0,0)
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    return a_.m256[imm8 & 1];
  #else
    easysimd__m256_private r_;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    const size_t n = sizeof(r_.f32) / sizeof(r_.f32[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = a_.f32[i + offset];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_extractf32x8_ps(a, imm8) _mm512_extractf32x8_ps(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_extractf32x8_ps
  #define _mm512_extractf32x8_ps(a, imm8) easysimd_mm512_extractf32x8_ps(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_mask_extractf32x8_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m512 a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)], src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)], src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      src_ = easysimd__m256_to_private(src),
      r_;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    const size_t n = sizeof(r_.f32) / sizeof(r_.f32[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i + offset] : src_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_mask_extractf32x8_ps(src, k, a, imm8) _mm512_mask_extractf32x8_ps(src, k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_extractf32x8_ps
  #define _mm512_mask_extractf32x8_ps(src, k, a, imm8) easysimd_mm512_mask_extractf32x8_ps(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_maskz_extractf32x8_ps (easysimd__mmask8 k, easysimd__m512 a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)], svdup_n_f32(0.0));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)], svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);
    const size_t n = sizeof(r_.f32) / sizeof(r_.f32[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i + offset] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_maskz_extractf32x8_ps(k, a, imm8) _mm512_maskz_extractf32x8_ps(k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_extractf32x8_ps
  #define _mm512_maskz_extractf32x8_ps(k, a, imm8) easysimd_mm512_maskz_extractf32x8_ps(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm512_extractf64x2_pd (easysimd__m512d a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = a.sve_f64[imm8 & 3];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = a.m128d[imm8 & 3].neon_f64;
    return r;
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    return a_.m128d[imm8 & 3];
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_extractf64x2_pd(a, imm8) _mm512_extractf64x2_pd(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_extractf64x2_pd
  #define _mm512_extractf64x2_pd(a, imm8) easysimd_mm512_extractf64x2_pd(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm512_mask_extractf64x2_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m512d a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[imm8 & 3], src.sve_f64);
    return r;
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      r_;

    const size_t n = sizeof(r_.f64) / sizeof(r_.f64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 3) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i + offset] : src_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_mask_extractf64x2_pd(src, k, a, imm8) _mm512_mask_extractf64x2_pd(src, k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_extractf64x2_pd
  #define _mm512_mask_extractf64x2_pd(src, k, a, imm8) easysimd_mm512_mask_extractf64x2_pd(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm512_maskz_extractf64x2_pd (easysimd__mmask8 k, easysimd__m512d a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[imm8 & 3], svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    easysimd__m128d_private r_;

    const size_t n = sizeof(r_.f64) / sizeof(r_.f64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 3) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i + offset] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_maskz_extractf64x2_pd(k, a, imm8) _mm512_maskz_extractf64x2_pd(k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_extractf64x2_pd
  #define _mm512_maskz_extractf64x2_pd(k, a, imm8) easysimd_mm512_maskz_extractf64x2_pd(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm512_extractf64x4_pd (easysimd__m512d a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64[(imm8 & 1) << 1];
    r.sve_f64[EASYSIMD_SV_INDEX_1] = a.sve_f64[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    int id = (imm8 & 1) << 1;
    r.m128d[0] = a.m128d[id];
    r.m128d[1] = a.m128d[id | 1];
    return r;
  #else
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    return a_.m256d[imm8 & 1];
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_extractf64x4_pd(a, imm8) _mm512_extractf64x4_pd(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_extractf64x4_pd
  #define _mm512_extractf64x4_pd(a, imm8) easysimd_mm512_extractf64x4_pd(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm512_mask_extractf64x4_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m512d a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)], src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)], src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      src_ = easysimd__m256d_to_private(src),
      r_;
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    const size_t n = sizeof(r_.f64) / sizeof(r_.f64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i + offset] : src_.f64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_mask_extractf64x4_pd(src, k, a, imm8) _mm512_mask_extractf64x4_pd(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_mask_extractf64x4_pd(src, k, a, imm8) easysimd_mm256_mask_mov_pd(src, k, easysimd_mm512_extractf64x4_pd(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_extractf64x4_pd
  #define _mm512_mask_extractf64x4_pd(src, k, a, imm8) easysimd_mm512_mask_extractf64x4_pd(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm512_maskz_extractf64x4_pd (easysimd__mmask8 k, easysimd__m512d a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)], svdup_n_f64(0.0));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)], svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m256d_private r_;
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);
    const size_t n = sizeof(r_.f64) / sizeof(r_.f64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i + offset] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0))
  #define easysimd_mm512_maskz_extractf64x4_pd(k, a, imm8) _mm512_maskz_extractf64x4_pd(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_maskz_extractf64x4_pd(k, a, imm8) easysimd_mm256_maskz_mov_pd(k, easysimd_mm512_extractf64x4_pd(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_extractf64x4_pd
  #define _mm512_maskz_extractf64x4_pd(k, a, imm8) easysimd_mm512_maskz_extractf64x4_pd(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm512_extracti32x4_epi32 (easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = a.sve_i32[imm8 & 3];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r = a.m128i[imm8 & 3];
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    return a_.m128i[imm8 & 3];
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_CLANG_REV_299346)
  #define easysimd_mm512_extracti32x4_epi32(a, imm8) _mm512_extracti32x4_epi32(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_extracti32x4_epi32
  #define _mm512_extracti32x4_epi32(a, imm8) easysimd_mm512_extracti32x4_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm512_mask_extracti32x4_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[imm8 & 3], src.sve_i32);
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      r_;

    const size_t n = sizeof(r_.i32) / sizeof(r_.i32[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 3) * n;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? a_.i32[i + offset] : src_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_CLANG_REV_299346)
  #define easysimd_mm512_mask_extracti32x4_epi32(src, k, a, imm8) _mm512_mask_extracti32x4_epi32(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_mask_extracti32x4_epi32(src, k, a, imm8) easysimd_mm_mask_mov_epi32(src, k, easysimd_mm512_extracti32x4_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_extracti32x4_epi32
  #define _mm512_mask_extracti32x4_epi32(src, k, a, imm8) easysimd_mm512_mask_extracti32x4_epi32(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm512_maskz_extracti32x4_epi32 (easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[imm8 & 3], svdup_n_s32(0));
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    easysimd__m128i_private r_;

    const size_t n = sizeof(r_.i32) / sizeof(r_.i32[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 3) * n;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? a_.i32[i + offset] : INT32_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_CLANG_REV_299346)
  #define easysimd_mm512_maskz_extracti32x4_epi32(k, a, imm8) _mm512_maskz_extracti32x4_epi32(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_maskz_extracti32x4_epi32(k, a, imm8) easysimd_mm_maskz_mov_epi32(k, easysimd_mm512_extracti32x4_epi32(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_extracti32x4_epi32
  #define _mm512_maskz_extracti32x4_epi32(k, a, imm8) easysimd_mm512_maskz_extracti32x4_epi32(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_extracti32x8_epi32 (easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32[(imm8 & 1) << 1];
    r.sve_i32[EASYSIMD_SV_INDEX_1] = a.sve_i32[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)];
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    const size_t n = sizeof(r_.i32) / sizeof(r_.i32[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[i + offset];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_extracti32x8_epi32
  #define _mm512_extracti32x8_epi32(a, imm8) easysimd_mm512_extracti32x8_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_mask_extracti32x8_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)], src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)], src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    const size_t n = sizeof(r_.i32) / sizeof(r_.i32[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? a_.i32[i + offset] : src_.i32[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_extracti32x8_epi32
  #define _mm512_mask_extracti32x8_epi32(src, k, a, imm8) easysimd_mm512_mask_extracti32x8_epi32(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_maskz_extracti32x8_epi32 (easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)], svdup_n_s32(0));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)], svdup_n_s32(0));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    const size_t n = sizeof(r_.i32) / sizeof(r_.i32[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? a_.i32[i + offset] : INT32_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_extracti32x8_epi32
  #define _mm512_maskz_extracti32x8_epi32(k, a, imm8) easysimd_mm512_maskz_extracti32x8_epi32(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm512_extracti64x2_epi64 (easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = a.sve_i64[imm8 & 3];
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    return a_.m128i[imm8 & 3];
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_CLANG_REV_299346)
  #define easysimd_mm512_extracti64x2_epi64(a, imm8) _mm512_extracti64x2_epi64(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_extracti64x2_epi64
  #define _mm512_extracti64x2_epi64(a, imm8) easysimd_mm512_extracti64x2_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm512_mask_extracti64x2_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[imm8 & 3], src.sve_i64);
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      r_;

    const size_t n = sizeof(r_.i64) / sizeof(r_.i64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 3) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? a_.i64[i + offset] : src_.i64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_CLANG_REV_299346)
  #define easysimd_mm512_mask_extracti64x2_epi64(src, k, a, imm8) _mm512_mask_extracti64x2_epi64(src, k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_extracti64x2_epi64
  #define _mm512_mask_extracti64x2_epi64(src, k, a, imm8) easysimd_mm512_mask_extracti64x2_epi64(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm512_maskz_extracti64x2_epi64 (easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[imm8 & 3], svdup_n_s64(0));
    return r;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    easysimd__m128i_private r_;
    
    const size_t n = sizeof(r_.i64) / sizeof(r_.i64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 3) * n;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? a_.i64[i + offset] : INT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_CLANG_REV_299346)
  #define easysimd_mm512_maskz_extracti64x2_epi64(k, a, imm8) _mm512_maskz_extracti64x2_epi64(k, a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_extracti64x2_epi64
  #define _mm512_maskz_extracti64x2_epi64(k, a, imm8) easysimd_mm512_maskz_extracti64x2_epi64(k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_extracti64x4_epi64 (easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = a.sve_i64[(imm8 & 1) << 1];
    r.sve_i64[EASYSIMD_SV_INDEX_1] = a.sve_i64[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m256i[imm8 & 1];
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    return a_.m256i[imm8 & 1];
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_CLANG_REV_299346)
  #define easysimd_mm512_extracti64x4_epi64(a, imm8) _mm512_extracti64x4_epi64(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_extracti64x4_epi64
  #define _mm512_extracti64x4_epi64(a, imm8) easysimd_mm512_extracti64x4_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_mask_extracti64x4_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)], src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)], src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    const size_t n = sizeof(r_.i64) / sizeof(r_.i64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? a_.i64[i + offset] : src_.i64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_CLANG_REV_299346)
  #define easysimd_mm512_mask_extracti64x4_epi64(src, k, a, imm8) _mm512_mask_extracti64x4_epi64(src, k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_mask_extracti64x4_epi64(src, k, a, imm8) easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm512_extracti64x4_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_extracti64x4_epi64
  #define _mm512_mask_extracti64x4_epi64(src, k, a, imm8) easysimd_mm512_mask_extracti64x4_epi64(src, k, a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_maskz_extracti64x4_epi64 (easysimd__mmask8 k, easysimd__m512i a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0 + ((imm8 & 1) << 1)], svdup_n_s64(0));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1 + ((imm8 & 1) << 1)], svdup_n_s64(0));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    const size_t n = sizeof(r_.i64) / sizeof(r_.i64[0]);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * n;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? a_.i64[i + offset] : INT64_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && !defined(EASYSIMD_BUG_CLANG_REV_299346)
  #define easysimd_mm512_maskz_extracti64x4_epi64(k, a, imm8) _mm512_maskz_extracti64x4_epi64(k, a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_maskz_extracti64x4_epi64(k, a, imm8) easysimd_mm256_maskz_mov_epi64(k, easysimd_mm512_extracti64x4_epi64(a, imm8))
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_extracti64x4_epi64
  #define _mm512_maskz_extracti64x4_epi64(k, a, imm8) easysimd_mm512_maskz_extracti64x4_epi64(k, a, imm8)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_EXTRACT_H) */
