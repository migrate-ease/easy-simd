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

#if !defined(EASYSIMD_X86_AVX512_MUL_H)
#define EASYSIMD_X86_AVX512_MUL_H

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
easysimd__m128i
easysimd_mm_mask_mul_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mul_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_s64(a.sve_i32, b.sve_i32), src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(int64_t, a_.i32[i * 2]) * HEDLEY_STATIC_CAST(int64_t, b_.i32[i * 2]) : src_.i64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mul_epi32
  #define _mm_mask_mul_epi32(src, k, a, b) easysimd_mm_mask_mul_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_mul_epi32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mul_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_s64(a.sve_i32, b.sve_i32), svdup_n_s64(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(int64_t, a_.i32[i * 2]) * HEDLEY_STATIC_CAST(int64_t, b_.i32[i * 2]) : INT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mul_epi32
  #define _mm_maskz_mul_epi32(k, a, b) easysimd_mm_maskz_mul_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_mul_epu32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mul_epu32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u64 = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_u64(a.sve_u32, b.sve_u32), src.sve_u64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(uint64_t, a_.u32[i * 2]) * HEDLEY_STATIC_CAST(uint64_t, b_.u32[i * 2]) : src_.u64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mul_epu32
  #define _mm_mask_mul_epu32(src, k, a, b) easysimd_mm_mask_mul_epu32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_mul_epu32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mul_epu32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u64 = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_u64(a.sve_u32, b.sve_u32), svdup_n_u64(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(uint64_t, a_.u32[i * 2]) * HEDLEY_STATIC_CAST(uint64_t, b_.u32[i * 2]) : UINT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mul_epu32
  #define _mm_maskz_mul_epu32(k, a, b) easysimd_mm_maskz_mul_epu32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_mul_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mul_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmul_f32_z(pg, a.sve_f32, b.sve_f32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (a_.f32[i] * b_.f32[i]) : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mul_ps
  #define _mm_mask_mul_ps(src, k, a, b) easysimd_mm_mask_mul_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_mul_ps (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mul_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmul_f32_z(pg, a.sve_f32, b.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (a_.f32[i] * b_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mul_ps
  #define _mm_maskz_mul_ps(k, a, b) easysimd_mm_maskz_mul_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_mul_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_mul_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmul_f64_z(pg, a.sve_f64, b.sve_f64), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      src_ = easysimd__m128d_to_private(src),
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (a_.f64[i] * b_.f64[i]) : src_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_mul_pd
  #define _mm_mask_mul_pd(src, k, a, b) easysimd_mm_mask_mul_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_mul_pd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_mul_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmul_f64_z(pg, a.sve_f64, b.sve_f64), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (a_.f64[i] * b_.f64[i]) : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_mul_pd
  #define _mm_maskz_mul_pd(k, a, b) easysimd_mm_maskz_mul_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_mul_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) 
    return _mm256_mask_mul_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(int64_t, a_.i32[i * 2]) * HEDLEY_STATIC_CAST(int64_t, b_.i32[i * 2]) : src_.i64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mul_epi32
  #define _mm256_mask_mul_epi32(src, k, a, b) easysimd_mm256_mask_mul_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_mul_epi32 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_mul_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), svdup_n_s64(0));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), svdup_n_s64(0));
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(int64_t, a_.i32[i * 2]) * HEDLEY_STATIC_CAST(int64_t, b_.i32[i * 2]) : INT64_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mul_epi32
  #define _mm256_maskz_mul_epi32(k, a, b) easysimd_mm256_maskz_mul_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_mul_epu32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) 
    return _mm256_mask_mul_epu32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), src.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), src.sve_u64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(uint64_t, a_.u32[i * 2]) * HEDLEY_STATIC_CAST(uint64_t, b_.u32[i * 2]) : src_.u64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mul_epu32
  #define _mm256_mask_mul_epu32(src, k, a, b) easysimd_mm256_mask_mul_epu32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_mul_epu32 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_mul_epu32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), svdup_n_u64(0));
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), svdup_n_u64(0));
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(uint64_t, a_.u32[i * 2]) * HEDLEY_STATIC_CAST(uint64_t, b_.u32[i * 2]) : UINT64_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mul_epu32
  #define _mm256_maskz_mul_epu32(k, a, b) easysimd_mm256_maskz_mul_epu32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_mul_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) 
    return _mm256_mask_mul_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);

    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      src_ = easysimd__m256_to_private(src),
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (a_.f32[i] * b_.f32[i]) : src_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mul_ps
  #define _mm256_mask_mul_ps(src, k, a, b) easysimd_mm256_mask_mul_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_mul_ps (easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) 
    return _mm256_maskz_mul_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svmul_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svmul_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (a_.f32[i] * b_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mul_ps
  #define _mm256_maskz_mul_ps(k, a, b) easysimd_mm256_maskz_mul_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_mul_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) 
    return _mm256_mask_mul_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      src_ = easysimd__m256d_to_private(src),
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (a_.f64[i] * b_.f64[i]) : src_.f64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_mul_pd
  #define _mm256_mask_mul_pd(src, k, a, b) easysimd_mm256_mask_mul_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_mul_pd (easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) 
    return _mm256_maskz_mul_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svmul_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svmul_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (a_.f64[i] * b_.f64[i]) : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_mul_pd
  #define _mm256_maskz_mul_pd(k, a, b) easysimd_mm256_maskz_mul_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mul_ps (easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mul_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0].neon_f32 = vmulq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vmulq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    r.m128[2].neon_f32 = vmulq_f32(a.m128[2].neon_f32, b.m128[2].neon_f32);
    r.m128[3].neon_f32 = vmulq_f32(a.m128[3].neon_f32, b.m128[3].neon_f32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 * b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_mul_ps(a_.m256[i], b_.m256[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mul_ps
  #define _mm512_mul_ps(a, b) easysimd_mm512_mul_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mul_round_ps (easysimd__m512 a, easysimd__m512 b, int rounding) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE_UNKNOWN)
    return _mm512_mul_round_ps(a, b, rounding);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]));
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
          r_.f32[i] = easysimd_math_roundevenf(a_.f32[i] * b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_floorf(a_.f32[i] * b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_ceilf(a_.f32[i] * b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_truncf(a_.f32[i] * b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_nearbyintf(a_.f32[i] * b_.f32[i]);
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
  #undef _mm512_mul_round_ps
  #define _mm512_mul_round_ps(a, b, rounding) easysimd_mm512_mul_round_ps(a, b, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_mul_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_mul_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmul_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svmul_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svmul_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]), src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svmul_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]), src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_mul_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mul_ps
  #define _mm512_mask_mul_ps(src, k, a, b) easysimd_mm512_mask_mul_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_mul_ps(easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_mul_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512 r;
  r.sve_f32[EASYSIMD_SV_INDEX_0] = svmul_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
  r.sve_f32[EASYSIMD_SV_INDEX_1] = svmul_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
  r.sve_f32[EASYSIMD_SV_INDEX_2] = svmul_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
  r.sve_f32[EASYSIMD_SV_INDEX_3] = svmul_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
  return r;
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_mul_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mul_ps
  #define _mm512_maskz_mul_ps(k, a, b) easysimd_mm512_maskz_mul_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mul_pd (easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mul_pd(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0].neon_f64 = vmulq_f64(a.m128d[0].neon_f64, b.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vmulq_f64(a.m128d[1].neon_f64, b.m128d[1].neon_f64);
    r.m128d[2].neon_f64 = vmulq_f64(a.m128d[2].neon_f64, b.m128d[2].neon_f64);
    r.m128d[3].neon_f64 = vmulq_f64(a.m128d[3].neon_f64, b.m128d[3].neon_f64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 * b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_mul_pd(a_.m256d[i], b_.m256d[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mul_pd
  #define _mm512_mul_pd(a, b) easysimd_mm512_mul_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mul_round_pd (easysimd__m512d a, easysimd__m512d b, int rounding) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE_UNKNOWN)
    return _mm512_mul_round_pd(a, b, rounding);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]));
    r.sve_f64[EASYSIMD_SV_INDEX_2] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]));
    r.sve_f64[EASYSIMD_SV_INDEX_3] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]));
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
          r_.f64[i] = easysimd_math_roundeven(a_.f64[i] * b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_floor(a_.f64[i] * b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_ceil(a_.f64[i] * b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_trunc(a_.f64[i] * b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_nearbyint(a_.f64[i] * b_.f64[i]);
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
  #undef _mm512_mul_round_pd
  #define _mm512_mul_round_pd(a, b, rounding) easysimd_mm512_mul_round_pd(a, b, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_mul_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_mul_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmul_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmul_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svmul_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]), src.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svmul_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]), src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_mul_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mul_pd
  #define _mm512_mask_mul_pd(src, k, a, b) easysimd_mm512_mask_mul_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_mul_pd(easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_mul_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512 r;
  r.sve_f64[EASYSIMD_SV_INDEX_0] = svmul_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
  r.sve_f64[EASYSIMD_SV_INDEX_1] = svmul_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
  r.sve_f64[EASYSIMD_SV_INDEX_2] = svmul_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]);
  r.sve_f64[EASYSIMD_SV_INDEX_3] = svmul_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]);
  return r;
  #else
    return easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_mul_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mul_pd
  #define _mm512_maskz_mul_pd(k, a, b) easysimd_mm512_maskz_mul_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mul_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mul_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    __asm__ __volatile__ (
        "ins %[a0].s[1], %[a0].s[2]             \n\t"
        "ins %[a1].s[1], %[a1].s[2]             \n\t"
        "ins %[a2].s[1], %[a2].s[2]             \n\t"
        "ins %[a3].s[1], %[a3].s[2]             \n\t"
        "ins %[b0].s[1], %[b0].s[2]             \n\t"
        "ins %[b1].s[1], %[b1].s[2]             \n\t"
        "ins %[b2].s[1], %[b2].s[2]             \n\t"
        "ins %[b3].s[1], %[b3].s[2]             \n\t"
        "smull %[a0].2d, %[a0].2s, %[b0].2s     \n\t"
        "smull %[a1].2d, %[a1].2s, %[b1].2s     \n\t"
        "smull %[a2].2d, %[a2].2s, %[b2].2s     \n\t"
        "smull %[a3].2d, %[a3].2s, %[b3].2s     \n\t"
        :[a0]"+w"(a.m128i[0].neon_i32), [a1]"+w"(a.m128i[1].neon_i32), [a2]"+w"(a.m128i[2].neon_i32), [a3]"+w"(a.m128i[3].neon_i32), 
         [b0]"+w"(b.m128i[0].neon_i32), [b1]"+w"(b.m128i[1].neon_i32), [b2]"+w"(b.m128i[2].neon_i32), [b3]"+w"(b.m128i[3].neon_i32)
        :
        :
    );
    return a;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      easysimd__m512i_private x;
      __typeof__(r_.i64) ta, tb;

      /* Get even numbered 32-bit values */
      x.i32 = EASYSIMD_SHUFFLE_VECTOR_(32, 64, a_.i32, b_.i32, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
      /* Cast to 64 bits */
      EASYSIMD_CONVERT_VECTOR_(ta, x.m256i_private[0].i32);
      EASYSIMD_CONVERT_VECTOR_(tb, x.m256i_private[1].i32);
      r_.i64 = ta * tb;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.i32[i << 1]) * HEDLEY_STATIC_CAST(int64_t, b_.i32[i << 1]);
      }
    #endif
      return easysimd__m512i_from_private(r_);
    #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mul_epi32
  #define _mm512_mul_epi32(a, b) easysimd_mm512_mul_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_mul_epi32(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_mul_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
  r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), src.sve_i64[EASYSIMD_SV_INDEX_2]);
  r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), src.sve_i64[EASYSIMD_SV_INDEX_3]);
  return r;
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_mul_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mul_epi32
  #define _mm512_mask_mul_epi32(src, k, a, b) easysimd_mm512_mask_mul_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_mul_epi32(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_mul_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svint64_t svzero = svdup_n_s64(0);
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), svzero);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), svzero);
  r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), svzero);
  r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), svzero);
  return r;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_mul_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mul_epi32
  #define _mm512_maskz_mul_epi32(k, a, b) easysimd_mm512_maskz_mul_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mul_epu32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mul_epu32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    __asm__ __volatile__ (
        "ins %[a0].s[1], %[a0].s[2]             \n\t"
        "ins %[a1].s[1], %[a1].s[2]             \n\t"
        "ins %[a2].s[1], %[a2].s[2]             \n\t"
        "ins %[a3].s[1], %[a3].s[2]             \n\t"
        "ins %[b0].s[1], %[b0].s[2]             \n\t"
        "ins %[b1].s[1], %[b1].s[2]             \n\t"
        "ins %[b2].s[1], %[b2].s[2]             \n\t"
        "ins %[b3].s[1], %[b3].s[2]             \n\t"
        "umull %[a0].2d, %[a0].2s, %[b0].2s     \n\t"
        "umull %[a1].2d, %[a1].2s, %[b1].2s     \n\t"
        "umull %[a2].2d, %[a2].2s, %[b2].2s     \n\t"
        "umull %[a3].2d, %[a3].2s, %[b3].2s     \n\t"
        :[a0]"+w"(a.m128i[0].neon_u32), [a1]"+w"(a.m128i[1].neon_u32), [a2]"+w"(a.m128i[2].neon_u32), [a3]"+w"(a.m128i[3].neon_u32), 
         [b0]"+w"(b.m128i[0].neon_u32), [b1]"+w"(b.m128i[1].neon_u32), [b2]"+w"(b.m128i[2].neon_u32), [b3]"+w"(b.m128i[3].neon_u32)
        :
        :
    );
    return a;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      easysimd__m512i_private x;
      __typeof__(r_.u64) ta, tb;

      x.u32 = EASYSIMD_SHUFFLE_VECTOR_(32, 64, a_.u32, b_.u32, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
      EASYSIMD_CONVERT_VECTOR_(ta, x.m256i_private[0].u32);
      EASYSIMD_CONVERT_VECTOR_(tb, x.m256i_private[1].u32);
      r_.u64 = ta * tb;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = HEDLEY_STATIC_CAST(uint64_t, a_.u32[i << 1]) * HEDLEY_STATIC_CAST(uint64_t, b_.u32[i << 1]);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mul_epu32
  #define _mm512_mul_epu32(a, b) easysimd_mm512_mul_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_mul_epu32(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_mul_epu32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), src.sve_u64[EASYSIMD_SV_INDEX_0]);
  r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), src.sve_u64[EASYSIMD_SV_INDEX_1]);
  r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), src.sve_u64[EASYSIMD_SV_INDEX_2]);
  r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), src.sve_u64[EASYSIMD_SV_INDEX_3]);
  return r;
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_mul_epu32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_mul_epu32
  #define _mm512_mask_mul_epu32(src, k, a, b) easysimd_mm512_mask_mul_epu32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_mul_epu32(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_mul_epu32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  svuint64_t svzero = svdup_n_u64(0);
  r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), svzero);
  r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), svzero);
  r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), svzero);
  r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), svzero);
  return r;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_mul_epu32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_mul_epu32
  #define _mm512_maskz_mul_epu32(k, a, b) easysimd_mm512_maskz_mul_epu32(k, a, b)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_MUL_H) */
