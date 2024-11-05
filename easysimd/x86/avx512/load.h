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
 */

#if !defined(EASYSIMD_X86_AVX512_LOAD_H)
#define EASYSIMD_X86_AVX512_LOAD_H

#include "types.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_load_epi32(easysimd__m128i src, easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_load_epi32(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svld1_s32(svptrue_b32(), (int32_t const *)mem_addr), src.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(easysimd__m128i));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : src_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_load_epi32(src, k, mem_addr) easysimd_mm_mask_load_epi32(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_load_epi32(easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_load_epi32(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (int32_t const *)mem_addr);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(easysimd__m128i));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : INT32_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_load_epi32(k, mem_addr) easysimd_mm_maskz_load_epi32(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_load_epi64(easysimd__m128i src, easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_load_epi64(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svld1_s64(svptrue_b64(), (int64_t const *)mem_addr), src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(easysimd__m128i));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? r_.i64[i] : src_.i64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_load_epi64(src, k, mem_addr) easysimd_mm_mask_load_epi64(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_load_epi64(easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_load_epi64(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int64_t const *)mem_addr);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(easysimd__m128i));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? r_.i64[i] : INT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_load_epi64(k, mem_addr) easysimd_mm_maskz_load_epi64(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_load_ps (easysimd__m128 src, easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_load_ps(src, k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;
  easysimd_svbool_t pg = svptrue_b32();
  r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svld1_f32(pg, (float32_t const *)mem_addr), src.sve_f32);
  return r;
#else
  easysimd__m128_private
    src_ = easysimd__m128_to_private(src),
    r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : src_.f32[i];
  }
  return easysimd__m128_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_load_ps(src, k, mem_addr) easysimd_mm_mask_load_ps(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_load_ps (easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_maskz_load_ps(k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;
  easysimd_svbool_t pg = svptrue_b32();
  r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svld1_f32(pg, (float32_t const *)mem_addr), svdup_n_f32(0.0));
  return r;
#else
  easysimd__m128_private r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
  }
  return easysimd__m128_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_load_ps(k, mem_addr) easysimd_mm_maskz_load_ps(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_load_pd (easysimd__m128d src, easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_load_pd(src, k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128d r;
  easysimd_svbool_t pg = svptrue_b64();
  r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svld1_f64(pg, (float64_t const *)mem_addr), src.sve_f64);
  return r;
#else
  easysimd__m128d_private
    src_ = easysimd__m128d_to_private(src),
    r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128d), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : src_.f64[i];
  }
  return easysimd__m128d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_load_pd(src, k, mem_addr) easysimd_mm_mask_load_pd(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_load_pd (easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_maskz_load_pd(k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128d r;
  easysimd_svbool_t pg = svptrue_b64();
  r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svld1_f64(pg, (float64_t const *)mem_addr), svdup_n_f64(0.0));
  return r;
#else
  easysimd__m128d_private r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128d), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
  }
  return easysimd__m128d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_load_pd(k, mem_addr) easysimd_mm_maskz_load_pd(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_load_epi32 (easysimd__m256i src, easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_load_epi32(src, k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  easysimd_svbool_t pg = svptrue_b32();
  r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                          svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_i32[EASYSIMD_SV_INDEX_0]);
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                          svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_i32[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd__m256i_private
    src_ = easysimd__m256i_to_private(src),
    r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256i), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
    r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : src_.i32[i];
  }
  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_load_epi32(src, k, mem_addr) easysimd_mm256_mask_load_epi32(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_load_epi32 (easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_maskz_load_epi32(k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
  return r;
#else
  easysimd__m256i_private r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256i), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
    r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : INT32_C(0);
  }
  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_load_epi32(k, mem_addr) easysimd_mm256_maskz_load_epi32(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_load_epi64 (easysimd__m256i src, easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_load_epi64(src, k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  easysimd_svbool_t pg = svptrue_b64();
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                                          svld1_s64(pg, HEDLEY_STATIC_CAST(const int64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)), src.sve_i64[EASYSIMD_SV_INDEX_0]);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                                          svld1_s64(pg, HEDLEY_STATIC_CAST(const int64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)), src.sve_i64[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd__m256i_private
    src_ = easysimd__m256i_to_private(src),
    r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256i), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
    r_.i64[i] = ((k >> i) & 1) ? r_.i64[i] : src_.i64[i];
  }
  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_load_epi64(src, k, mem_addr) easysimd_mm256_mask_load_epi64(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_load_epi64 (easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_maskz_load_epi64(k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), HEDLEY_STATIC_CAST(const int64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6));
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), HEDLEY_STATIC_CAST(const int64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6));
  return r;
#else
  easysimd__m256i_private r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256i), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
    r_.i64[i] = ((k >> i) & 1) ? r_.i64[i] : INT64_C(0);
  }
  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_load_epi64(k, mem_addr) easysimd_mm256_maskz_load_epi64(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_load_ps (easysimd__m256 src, easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_load_ps(src, k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  easysimd_svbool_t pg = svptrue_b32();
  r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                          svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_0]);
  r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                          svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd__m256_private
    src_ = easysimd__m256_to_private(src),
    r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : src_.f32[i];
  }
  return easysimd__m256_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_load_ps(src, k, mem_addr) easysimd_mm256_mask_load_ps(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_load_ps (easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_maskz_load_ps(k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
  r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
  return r;
#else
  easysimd__m256_private r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
  }
  return easysimd__m256_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_load_ps(k, mem_addr) easysimd_mm256_maskz_load_ps(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_load_pd (easysimd__m256d src, easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_load_pd(src, k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256d r;
  easysimd_svbool_t pg = svptrue_b64();
  r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                                          svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)), src.sve_f64[EASYSIMD_SV_INDEX_0]);
  r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                                          svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)), src.sve_f64[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd__m256d_private
    src_ = easysimd__m256d_to_private(src),
    r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256d), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : src_.f64[i];
  }
  return easysimd__m256d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_load_pd(src, k, mem_addr) easysimd_mm256_mask_load_pd(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_load_pd (easysimd__mmask8 k, void const* mem_addr) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_maskz_load_pd(k, mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256d r;
  r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6));
  r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6));
  return r;
#else
  easysimd__m256d_private r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256d), sizeof(r_));

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
  }
  return easysimd__m256d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_load_pd(k, mem_addr) easysimd_mm256_maskz_load_pd(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_load_ps (void const * mem_addr) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0].neon_f32 = vld1q_f32((const float *)mem_addr + 0);
    r.m128[1].neon_f32 = vld1q_f32((const float *)mem_addr + 4);
    r.m128[2].neon_f32 = vld1q_f32((const float *)mem_addr + 8);
    r.m128[3].neon_f32 = vld1q_f32((const float *)mem_addr + 12);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_load_ps(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512));
  #else
    easysimd__m512 r;
    easysimd_memcpy(&r, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512), sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
#  define _mm512_load_ps(mem_addr) easysimd_mm512_load_ps(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_load_ps (easysimd__m512 src, easysimd__mmask16 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_load_ps(src, k, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),
                                            svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),
                                            svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      src_ = easysimd__m512_to_private(src);
    easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512), sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : src_.f32[i];
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_load_ps
  #define _mm512_mask_load_ps(src, k, mem_addr) easysimd_mm512_mask_load_ps(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_load_ps (easysimd__mmask16 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_load_ps(k, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #else
    easysimd__m512_private r_;
    easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512), sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_load_ps
  #define _mm512_maskz_load_ps(k, mem_addr) easysimd_mm512_maskz_load_ps(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_load_pd (void const * mem_addr) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6));
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 6));
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 6));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0].neon_f64 = vld1q_f64((const double *)mem_addr + 0);
    r.m128d[1].neon_f64 = vld1q_f64((const double *)mem_addr + 2);
    r.m128d[2].neon_f64 = vld1q_f64((const double *)mem_addr + 4);
    r.m128d[3].neon_f64 = vld1q_f64((const double *)mem_addr + 6);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_load_pd(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512d));
  #else
    easysimd__m512d r;
    easysimd_memcpy(&r, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512d), sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
#  define _mm512_load_pd(mem_addr) easysimd_mm512_load_pd(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_load_si512 (void const * mem_addr) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 0);
    r.m128i[1].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 4);
    r.m128i[2].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 8);
    r.m128i[3].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 12);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_load_si512(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512i));
  #else
    easysimd__m512i r;
    easysimd_memcpy(&r, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m512i), sizeof(r));
    return r;
  #endif
}
#define easysimd_mm512_load_epi8(mem_addr) easysimd_mm512_load_si512(mem_addr)
#define easysimd_mm512_load_epi16(mem_addr) easysimd_mm512_load_si512(mem_addr)
#define easysimd_mm512_load_epi32(mem_addr) easysimd_mm512_load_si512(mem_addr)
#define easysimd_mm512_load_epi64(mem_addr) easysimd_mm512_load_si512(mem_addr)
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_load_epi8
  #undef _mm512_load_epi16
  #undef _mm512_load_epi32
  #undef _mm512_load_epi64
  #undef _mm512_load_si512
  #define _mm512_load_si512(a) easysimd_mm512_load_si512(a)
  #define _mm512_load_epi8(a) easysimd_mm512_load_si512(a)
  #define _mm512_load_epi16(a) easysimd_mm512_load_si512(a)
  #define _mm512_load_epi32(a) easysimd_mm512_load_si512(a)
  #define _mm512_load_epi64(a) easysimd_mm512_load_si512(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_LOAD_H) */
