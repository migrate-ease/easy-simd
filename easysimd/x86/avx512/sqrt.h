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
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 */

#if !defined(EASYSIMD_X86_AVX512_SQRT_H)
#define EASYSIMD_X86_AVX512_SQRT_H

#include "types.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_sqrt_ps(easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_sqrt_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svsqrt_f32_z(svptrue_b32(), a.sve_f32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      r_;
    #if defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = ((k >> i) & 1) ? easysimd_math_sqrt(a_.f32[i]) : src_.f32[i];
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_sqrt_ps
  #define _mm_mask_sqrt_ps(src, k, a) easysimd_mm_mask_sqrt_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_sqrt_ps(easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_sqrt_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svsqrt_f32_z(svptrue_b32(), a.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      r_;
    #if defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = ((k >> i) & 1) ? easysimd_math_sqrt(a_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_sqrt_ps
  #define _mm_maskz_sqrt_ps(k, a) easysimd_mm_maskz_sqrt_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_sqrt_pd(easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_sqrt_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svsqrt_f64_z(svptrue_b64(), a.sve_f64), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      a_ = easysimd__m128d_to_private(a),
      r_;
    #if defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = ((k >> i) & 1) ? easysimd_math_sqrt(a_.f64[i]) : src_.f64[i];
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_sqrt_pd
  #define _mm_mask_sqrt_pd(src, k, a) easysimd_mm_mask_sqrt_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_sqrt_pd(easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_sqrt_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svsqrt_f64_z(svptrue_b64(), a.sve_f64), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      r_;
    #if defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = ((k >> i) & 1) ? easysimd_math_sqrt(a_.f64[i]) : EASYSIMD_FLOAT64_C(0.0);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_sqrt_pd
  #define _mm_maskz_sqrt_pd(k, a) easysimd_mm_maskz_sqrt_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_sqrt_ps(easysimd__m256 src, easysimd__mmask8 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_sqrt_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svsqrt_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svsqrt_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      src_ = easysimd__m256_to_private(src),
      a_ = easysimd__m256_to_private(a),
      r_;
    #if defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = ((k >> i) & 1) ? easysimd_math_sqrt(a_.f32[i]) : src_.f32[i];
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_sqrt_ps
  #define _mm256_mask_sqrt_ps(src, k, a) easysimd_mm256_mask_sqrt_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_sqrt_ps(easysimd__mmask8 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_sqrt_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svsqrt_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]), svdup_n_f32(0.0));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svsqrt_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a),
      r_;
    #if defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = ((k >> i) & 1) ? easysimd_math_sqrt(a_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_sqrt_ps
  #define _mm256_maskz_sqrt_ps(k, a) easysimd_mm256_maskz_sqrt_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_sqrt_pd(easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_sqrt_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svsqrt_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svsqrt_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d
      src_ = easysimd__m256d_to_private(src),
      a_ = easysimd__m256d_to_private(a),
      r_;
    #if defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = ((k >> i) & 1) ? easysimd_math_sqrt(a_.f64[i]) : src_.f64[i];
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_sqrt_pd
  #define _mm256_mask_sqrt_pd(src, k, a) easysimd_mm256_mask_sqrt_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_sqrt_pd(easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_sqrt_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svsqrt_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]), svdup_n_f64(0.0));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svsqrt_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a),
      r_;
    #if defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = ((k >> i) & 1) ? easysimd_math_sqrt(a_.f64[i]) : EASYSIMD_FLOAT64_C(0.0);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_sqrt_pd
  #define _mm256_maskz_sqrt_pd(k, a) easysimd_mm256_maskz_sqrt_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_sqrt_ps (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sqrt_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsqrt_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsqrt_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsqrt_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsqrt_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a);

    #if defined(EASYSIMD_X86_AVX_NATIVE)
      r_.m256[0] = easysimd_mm256_sqrt_ps(a_.m256[0]);
      r_.m256[1] = easysimd_mm256_sqrt_ps(a_.m256[1]);
    #elif defined(easysimd_math_sqrtf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_sqrtf(a_.f32[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
#  define _mm512_sqrt_ps(a) easysimd_mm512_sqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_sqrt_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sqrt_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svsqrt_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svsqrt_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svsqrt_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_2]), src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svsqrt_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_3]), src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_sqrt_ps(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sqrt_ps
  #define _mm512_mask_sqrt_ps(src, k, a) easysimd_mm512_mask_sqrt_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_sqrt_pd (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sqrt_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsqrt_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsqrt_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsqrt_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsqrt_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a);

    #if defined(EASYSIMD_X86_AVX_NATIVE)
      r_.m256d[0] = easysimd_mm256_sqrt_pd(a_.m256d[0]);
      r_.m256d[1] = easysimd_mm256_sqrt_pd(a_.m256d[1]);
    #elif defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_sqrt(a_.f64[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
#  define _mm512_sqrt_pd(a) easysimd_mm512_sqrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_sqrt_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sqrt_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svsqrt_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svsqrt_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svsqrt_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_2]), src.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svsqrt_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_3]), src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_sqrt_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sqrt_pd
  #define _mm512_mask_sqrt_pd(src, k, a) easysimd_mm512_mask_sqrt_pd(src, k, a)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SQRT_H) */
