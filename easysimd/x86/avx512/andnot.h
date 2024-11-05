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

#if !defined(EASYSIMD_X86_AVX512_ANDNOT_H)
#define EASYSIMD_X86_AVX512_ANDNOT_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_andnot_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_andnot_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svbic_s32_z(pg, b.sve_i32, a.sve_i32), src.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : src_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_andnot_epi32(src, k, a, b) easysimdeasysimd_mm_mask_andnot_epi32(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_andnot_epi32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_andnot_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svbic_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), b.sve_i32, a.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : INT32_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_andnot_epi32(k, a, b) easysimdeasysimd_mm_maskz_andnot_epi32(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_andnot_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_andnot_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svbic_s64_z(pg, b.sve_i64, a.sve_i64), src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? ~(a_.i64[i]) & b_.i64[i] : src_.i64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_andnot_epi64(src, k, a, b) easysimd_mm_mask_andnot_epi64(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_andnot_epi64 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_andnot_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svbic_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), b.sve_i64, a.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? ~(a_.i64[i]) & b_.i64[i] : INT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_andnot_epi64(k, a, b) easysimd_mm_maskz_andnot_epi64(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_andnot_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_andnot_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svbic_s32_z(pg, b.sve_i32, a.sve_i32), src.sve_i32);
    return r;
  #else
    easysimd__m128_private
      r_,
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : src_.i32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_andnot_ps(src, k, a, b) easysimd_mm_mask_andnot_ps(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_andnot_ps (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_andnot_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svbic_s32_z(pg, b.sve_i32, a.sve_i32), svdup_n_s32(0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : INT32_C(0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_andnot_ps(k, a, b) easysimd_mm_maskz_andnot_ps(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_andnot_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_andnot_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svbic_s64_z(pg, b.sve_i64, a.sve_i64), src.sve_i64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      src_ = easysimd__m128d_to_private(src),
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? ~(a_.i64[i]) & b_.i64[i] : src_.i64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_andnot_pd(src, k, a, b) easysimd_mm_mask_andnot_pd(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_andnot_pd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_andnot_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svbic_s64_z(pg, b.sve_i64, a.sve_i64), svdup_n_s64(0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? ~(a_.i64[i]) & b_.i64[i] : INT64_C(0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_andnot_pd(k, a, b) easysimd_mm_maskz_andnot_pd(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_andnot_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_andnot_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : src_.i32[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_andnot_epi32
  #define _mm256_mask_andnot_epi32(src, k, a, b) easysimd_mm256_mask_andnot_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_andnot_epi32 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_andnot_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svbic_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svbic_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : INT32_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_andnot_epi32
  #define _mm256_maskz_andnot_epi32(k, a, b) easysimd_mm256_maskz_andnot_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_andnot_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_andnot_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svbic_s64_z(pg, b.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svbic_s64_z(pg, b.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? ~(a_.i64[i]) & b_.i64[i] : src_.i64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_andnot_epi64
  #define _mm256_mask_andnot_epi64(src, k, a, b) easysimd_mm256_mask_andnot_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_andnot_epi64 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_andnot_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svbic_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), b.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svbic_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), b.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? ~(a_.i64[i]) & b_.i64[i] : INT64_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_andnot_epi64
  #define _mm256_maskz_andnot_epi64(k, a, b) easysimd_mm256_maskz_andnot_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_andnot_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_andnot_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      src_ = easysimd__m256_to_private(src),
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : src_.i32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_andnot_ps(src, k, a, b) easysimd_mm256_mask_andnot_ps(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_andnot_ps (easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_andnot_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svbic_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svbic_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : INT32_C(0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_andnot_ps(k, a, b) easysimd_mm256_maskz_andnot_ps(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_andnot_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_andnot_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svbic_s64_z(pg, b.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svbic_s64_z(pg, b.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      src_ = easysimd__m256d_to_private(src),
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? ~(a_.i64[i]) & b_.i64[i] : src_.i64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_andnot_pd(src, k, a, b) easysimd_mm256_mask_andnot_pd(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_andnot_pd (easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_andnot_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svbic_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), b.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svbic_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), b.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? ~(a_.i64[i]) & b_.i64[i] : INT64_C(0);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_andnot_pd(k, a, b) easysimd_mm256_maskz_andnot_pd(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_andnot_ps (easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_2], a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_3], a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ~(a_.i32[i]) & b_.i32[i];
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_andnot_ps(a, b) _mm512_andnot_ps(a, b)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_andnot_ps(a, b) easysimd_mm512_castsi512_ps(easysimd_mm512_andnot_si512(easysimd_mm512_castps_si512(a), easysimd_mm512_castps_si512(b)))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_andnot_ps
  #define _mm512_andnot_ps(a, b) easysimd_mm512_andnot_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_andnot_ps (easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_2], a.sve_i32[EASYSIMD_SV_INDEX_2]), src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_3], a.sve_i32[EASYSIMD_SV_INDEX_3]), src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      src_ = easysimd__m512_to_private(src),
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : src_.i32[i];
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_mask_andnot_ps(src, k, a, b) _mm512_mask_andnot_ps((src), (k), (a), (b))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_mask_andnot_ps(src, k, a, b) easysimd_mm512_castsi512_ps(easysimd_mm512_mask_andnot_epi32(easysimd_mm512_castps_si512(src), k, easysimd_mm512_castps_si512(a), easysimd_mm512_castps_si512(b)))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_andnot_ps
  #define _mm512_mask_andnot_ps(src, k, a, b) easysimd_mm512_mask_andnot_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_andnot_ps (easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svbic_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svbic_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svbic_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), b.sve_i32[EASYSIMD_SV_INDEX_2], a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svbic_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), b.sve_i32[EASYSIMD_SV_INDEX_3], a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? ~(a_.i32[i]) & b_.i32[i] : INT32_C(0);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_maskz_andnot_ps(k, a, b) _mm512_maskz_andnot_ps((k), (a), (b))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_maskz_andnot_ps(k, a, b) easysimd_mm512_castsi512_ps(easysimd_mm512_maskz_andnot_epi32(k, easysimd_mm512_castps_si512(a), easysimd_mm512_castps_si512(b)))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_andnot_ps
  #define _mm512_maskz_andnot_ps(k, a, b) easysimd_mm512_maskz_andnot_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_andnot_pd (easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svbic_s64_z(pg, b.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svbic_s64_z(pg, b.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svbic_s64_z(pg, b.sve_i64[EASYSIMD_SV_INDEX_2], a.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svbic_s64_z(pg, b.sve_i64[EASYSIMD_SV_INDEX_3], a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ~(a_.i64[i]) & b_.i64[i];
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_andnot_pd(a, b) _mm512_andnot_pd(a, b)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#else
  #define easysimd_mm512_andnot_pd(a, b) easysimd_mm512_castsi512_pd(easysimd_mm512_andnot_si512(easysimd_mm512_castpd_si512(a), easysimd_mm512_castpd_si512(b)))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_andnot_pd
  #define _mm512_andnot_pd(a, b) easysimd_mm512_andnot_pd(a, b)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_mask_andnot_pd(src, k, a, b) _mm512_mask_andnot_pd((src), (k), (a), (b))
#else
  #define easysimd_mm512_mask_andnot_pd(src, k, a, b) easysimd_mm512_castsi512_pd(easysimd_mm512_mask_andnot_epi64(easysimd_mm512_castpd_si512(src), k, easysimd_mm512_castpd_si512(a), easysimd_mm512_castpd_si512(b)))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_andnot_pd
  #define _mm512_mask_andnot_pd(src, k, a, b) easysimd_mm512_mask_andnot_pd(src, k, a, b)
#endif

#if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
  #define easysimd_mm512_maskz_andnot_pd(k, a, b) _mm512_maskz_andnot_pd((k), (a), (b))
#else
  #define easysimd_mm512_maskz_andnot_pd(k, a, b) easysimd_mm512_castsi512_pd(easysimd_mm512_maskz_andnot_epi64(k, easysimd_mm512_castpd_si512(a), easysimd_mm512_castpd_si512(b)))
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_andnot_pd
  #define _mm512_maskz_andnot_pd(k, a, b) easysimd_mm512_maskz_andnot_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_andnot_si512 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_andnot_si512(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_2], a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_3], a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i32 = vbicq_s32(b.m128i[0].neon_i32, a.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vbicq_s32(b.m128i[1].neon_i32, a.m128i[1].neon_i32);
    r.m128i[2].neon_i32 = vbicq_s32(b.m128i[2].neon_i32, a.m128i[2].neon_i32);
    r.m128i[3].neon_i32 = vbicq_s32(b.m128i[3].neon_i32, a.m128i[3].neon_i32);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      r_.m256i[0] = easysimd_mm256_andnot_si256(a_.m256i[0], b_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_andnot_si256(a_.m256i[1], b_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = ~(a_.i32f[i]) & b_.i32f[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#define easysimd_mm512_andnot_epi32(a, b) easysimd_mm512_andnot_si512(a, b)
#define easysimd_mm512_andnot_epi64(a, b) easysimd_mm512_andnot_si512(a, b)
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_andnot_si512
  #define _mm512_andnot_si512(a, b) easysimd_mm512_andnot_si512(a, b)
  #undef _mm512_andnot_epi32
  #define _mm512_andnot_epi32(a, b) easysimd_mm512_andnot_si512(a, b)
  #undef _mm512_andnot_epi64
  #define _mm512_andnot_epi64(a, b) easysimd_mm512_andnot_si512(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_andnot_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_andnot_epi32(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_andnot_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_andnot_epi32
  #define _mm512_mask_andnot_epi32(src, k, a, b) easysimd_mm512_mask_andnot_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_andnot_epi32(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_andnot_epi32(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_andnot_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_andnot_epi32
  #define _mm512_maskz_andnot_epi32(k, a, b) easysimd_mm512_maskz_andnot_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_andnot_epi64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_andnot_epi64(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_andnot_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_andnot_epi64
  #define _mm512_mask_andnot_epi64(src, k, a, b) easysimd_mm512_mask_andnot_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_andnot_epi64(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_andnot_epi64(k, a, b);
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_andnot_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_andnot_epi64
  #define _mm512_maskz_andnot_epi64(k, a, b) easysimd_mm512_maskz_andnot_epi64(k, a, b)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_ANDNOT_H) */
