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
 *   2020      Hidayat Khan <huk2209@gmail.com>
 */

#if !defined(EASYSIMD_X86_AVX512_DIV_H)
#define EASYSIMD_X86_AVX512_DIV_H

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
easysimd_mm_mask_div_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_div_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svdiv_f32_z(pg, a.sve_f32, b.sve_f32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (a_.f32[i] / b_.f32[i]) : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_div_ps(src, k, a, b) easysimd_mm_mask_div_ps(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_div_ps (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_div_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svdiv_f32_z(pg, a.sve_f32, b.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (a_.f32[i] / b_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_div_ps(k, a, b) easysimd_mm_maskz_div_ps(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_div_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_div_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdiv_f64_z(pg, a.sve_f64, b.sve_f64), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      src_ = easysimd__m128d_to_private(src),
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (a_.f64[i] / b_.f64[i]) : src_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_div_pd(src, k, a, b) easysimd_mm_mask_div_pd(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_div_pd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_div_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdiv_f64_z(pg, a.sve_f64, b.sve_f64), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (a_.f64[i] / b_.f64[i]) : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_div_pd(k, a, b) easysimd_mm_maskz_div_pd(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_div_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_div_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      src_ = easysimd__m256_to_private(src),
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (a_.f32[i] / b_.f32[i]) : src_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_div_ps(src, k, a, b) easysimd_mm256_mask_div_ps(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_div_ps (easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_div_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svdiv_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svdiv_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (a_.f32[i] / b_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_div_ps(k, a, b) easysimd_mm256_maskz_div_ps(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_div_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_div_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      src_ = easysimd__m256d_to_private(src),
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (a_.f64[i] / b_.f64[i]) : src_.f64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_div_pd(src, k, a, b) easysimd_mm256_mask_div_pd(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_div_pd (easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_div_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svdiv_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svdiv_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (a_.f64[i] / b_.f64[i]) : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_div_pd(k, a, b) easysimd_mm256_maskz_div_pd(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_div_ps (easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0].neon_f32 = vdivq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vdivq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    r.m128[2].neon_f32 = vdivq_f32(a.m128[2].neon_f32, b.m128[2].neon_f32);
    r.m128[3].neon_f32 = vdivq_f32(a.m128[3].neon_f32, b.m128[3].neon_f32);
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_div_ps(a_.m256[i], b_.m256[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 / b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_div_ps(a_.m256[i], b_.m256[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_ps
  #define _mm512_div_ps(a, b) easysimd_mm512_div_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_div_round_ps (easysimd__m512 a, easysimd__m512 b, int rounding) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE_UNKNOWN)
    return _mm512_div_round_ps(a, b, rounding);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]));
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC)
    {
      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_roundevenf(a_.f32[i] / b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_floorf(a_.f32[i] / b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_ceilf(a_.f32[i] / b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_truncf(a_.f32[i] / b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_nearbyintf(a_.f32[i] / b_.f32[i]);
        }
        break;
      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
        break;
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_round_ps
  #define _mm512_div_round_ps(a, b, rounding) easysimd_mm512_div_round_ps(a, b, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_div_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_div_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svdiv_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svdiv_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svdiv_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]), src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svdiv_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]), src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_div_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_div_ps
  #define _mm512_mask_div_ps(src, k, a, b) easysimd_mm512_mask_div_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_div_ps(easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_div_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svdiv_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svdiv_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svdiv_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svdiv_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_div_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_div_ps
  #define _mm512_maskz_div_ps(k, a, b) easysimd_mm512_maskz_div_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_div_pd (easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_div_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0].neon_f64 = vdivq_f64(a.m128d[0].neon_f64, b.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vdivq_f64(a.m128d[1].neon_f64, b.m128d[1].neon_f64);
    r.m128d[2].neon_f64 = vdivq_f64(a.m128d[2].neon_f64, b.m128d[2].neon_f64);
    r.m128d[3].neon_f64 = vdivq_f64(a.m128d[3].neon_f64, b.m128d[3].neon_f64);
    return r;
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_div_pd(a_.m256d[i], b_.m256d[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 / b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_div_pd(a_.m256d[i], b_.m256d[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_pd
  #define _mm512_div_pd(a, b) easysimd_mm512_div_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_div_round_pd (easysimd__m512d a, easysimd__m512d b, int rounding) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE_UNKNOWN)
    return _mm512_div_round_pd(a, b, rounding);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]));
    r.sve_f64[EASYSIMD_SV_INDEX_2] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]));
    r.sve_f64[EASYSIMD_SV_INDEX_3] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]));
    return r;
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC)
    {
      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_roundeven(a_.f64[i] / b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_floor(a_.f64[i] / b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_ceil(a_.f64[i] / b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_trunc(a_.f64[i] / b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_nearbyint(a_.f64[i] / b_.f64[i]);
        }
        break;
      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_pd());
        break;
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_div_round_pd
  #define _mm512_div_round_pd(a, b, rounding) easysimd_mm512_div_round_pd(a, b, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_div_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_div_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svdiv_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svdiv_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svdiv_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]), src.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svdiv_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]), src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_div_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_div_pd
  #define _mm512_mask_div_pd(src, k, a, b) easysimd_mm512_mask_div_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_div_pd(easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_div_pd(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_div_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_div_pd
  #define _mm512_maskz_div_pd(k, a, b) easysimd_mm512_maskz_div_pd(k, a, b)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_DIV_H) */
