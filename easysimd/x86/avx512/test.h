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
 *   2020      Hidayat Khan <huk2209@gmail.com>
 *   2020      Christopher Moore <moore@free.fr>
 *   2021      Andrew Rodriguez <anrodriguez@linkedin.com>
 */

#if !defined(EASYSIMD_X86_AVX512_TEST_H)
#define EASYSIMD_X86_AVX512_TEST_H

#include "types.h"
#include "mov_mask.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_test_epi8_mask (easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_test_epi8_mask(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask16 rk = 0;
  svbool_t pg = svptrue_b8();
  svint8_t svzero =  svdup_n_s8(0);
  EASYSIMD_B8_TO_MASK(rk, svcmpne_s8(pg, svand_s8_x(pg, a.sve_i8, b.sve_i8), svzero), EASYSIMD_SV_INDEX_0);
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask16 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
    r |= HEDLEY_STATIC_CAST(easysimd__mmask16, !!(a_.i8[i] & b_.i8[i]) << i);
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_test_epi8_mask
#define _mm_test_epi8_mask(a, b) easysimd_mm_test_epi8_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_test_epi16_mask (easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_test_epi16_mask(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b16();
  svint16_t svzero =  svdup_n_s16(0);
  EASYSIMD_B16_TO_MASK(rk, svcmpne_s16(pg, svand_s16_x(pg, a.sve_i16, b.sve_i16), svzero), EASYSIMD_SV_INDEX_0);
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
    r |= HEDLEY_STATIC_CAST(easysimd__mmask16, !!(a_.i16[i] & b_.i16[i]) << i);
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_test_epi16_mask
#define _mm_test_epi16_mask(a, b) easysimd_mm_test_epi16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_test_epi32_mask (easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_test_epi32_mask(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b32();
  svint32_t svzero =  svdup_n_s32(0);
  EASYSIMD_B32_TO_MASK(rk, svcmpne_s32(pg, svand_s32_x(pg, a.sve_i32, b.sve_i32), svzero), EASYSIMD_SV_INDEX_0);
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
    r |= HEDLEY_STATIC_CAST(easysimd__mmask32, !!(a_.i32[i] & b_.i32[i]) << i);
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_test_epi32_mask
#define _mm_test_epi32_mask(a, b) easysimd_mm_test_epi32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_test_epi64_mask (easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_test_epi64_mask(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b64();
  svint64_t svzero =  svdup_n_s64(0);
  EASYSIMD_B64_TO_MASK(rk, svcmpne_s64(pg, svand_s64_x(pg, a.sve_i64, b.sve_i64), svzero), EASYSIMD_SV_INDEX_0);
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
    r |= HEDLEY_STATIC_CAST(easysimd__mmask64, !!(a_.i64[i] & b_.i64[i]) << i);
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_test_epi64_mask
#define _mm_test_epi64_mask(a, b) easysimd_mm_test_epi64_mask(a, b)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_mask_test_epi8_mask (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_test_epi8_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask16 rk = 0;
  svbool_t pg = svptrue_b8();
  svint8_t svzero =  svdup_n_s8(0);
  EASYSIMD_B8_TO_MASK(rk, svcmpne_s8(pg, svand_s8_x(pg, a.sve_i8, b.sve_i8), svzero), EASYSIMD_SV_INDEX_0);
  rk &= k;
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask16 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask16, !!(a_.i8[i] & b_.i8[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_test_epi8_mask
#define _mm_mask_test_epi8_mask(k, a, b) easysimd_mm_mask_test_epi8_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_test_epi16_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_test_epi16_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b16();
  svint16_t svzero =  svdup_n_s16(0);
  EASYSIMD_B16_TO_MASK(rk, svcmpne_s16(pg, svand_s16_x(pg, a.sve_i16, b.sve_i16), svzero), EASYSIMD_SV_INDEX_0);
  rk &= k;
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask16, !!(a_.i16[i] & b_.i16[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_test_epi16_mask
#define _mm_mask_test_epi16_mask(k, a, b) easysimd_mm_mask_test_epi16_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_test_epi32_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_test_epi32_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b32();
  svint32_t svzero =  svdup_n_s32(0);
  EASYSIMD_B32_TO_MASK(rk, svcmpne_s32(pg, svand_s32_x(pg, a.sve_i32, b.sve_i32), svzero), EASYSIMD_SV_INDEX_0);
  rk &= k;
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask32, !!(a_.i32[i] & b_.i32[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_test_epi32_mask
#define _mm_mask_test_epi32_mask(k, a, b) easysimd_mm_mask_test_epi32_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_test_epi64_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_test_epi64_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b64();
  svint64_t svzero =  svdup_n_s64(0);
  EASYSIMD_B64_TO_MASK(rk, svcmpne_s64(pg, svand_s64_x(pg, a.sve_i64, b.sve_i64), svzero), EASYSIMD_SV_INDEX_0);
  rk &= k;
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask64, !!(a_.i64[i] & b_.i64[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_test_epi64_mask
#define _mm_mask_test_epi64_mask(k, a, b) easysimd_mm_mask_test_epi64_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_test_epi16_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_test_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 res = 0;
    svbool_t pg = svptrue_b16();
    svuint16_t svzero = svdup_n_u16(0);
    EASYSIMD_B16_TO_MASK(res, svcmpne_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(res, svcmpne_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
    return res;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
      r |= (!(!(a_.i16[i] & b_.i16[i]))) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_test_epi16_mask
  #define _mm256_test_epi16_mask(a, b) easysimd_mm256_test_epi16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_test_epi32_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_test_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 res = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(res, svcmpne_n_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), 0), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(res, svcmpne_n_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), 0), EASYSIMD_SV_INDEX_1);
    return res;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      r |= HEDLEY_STATIC_CAST(easysimd__mmask16, !!(a_.i32[i] & b_.i32[i]) << i);
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_test_epi32_mask
#define _mm256_test_epi32_mask(a, b) easysimd_mm256_test_epi32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_test_epi64_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_test_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 res = 0;
    svbool_t pg = svptrue_b64();
    svuint64_t svzero = svdup_n_u64(0);
    EASYSIMD_B64_TO_MASK(res, svcmpne_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(res, svcmpne_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
    return res;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r |= (!(!(a_.i64[i] & b_.i64[i]))) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_test_epi64_mask
  #define _mm256_test_epi64_mask(a, b) easysimd_mm256_test_epi64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_mask_test_epi8_mask (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_test_epi8_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask32 rk = 0;
  svbool_t pg = svptrue_b8();
  svint8_t svzero =  svdup_n_s8(0);
  EASYSIMD_B8_TO_MASK(rk, svcmpne_s8(pg, svand_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
  EASYSIMD_B8_TO_MASK(rk, svcmpne_s8(pg, svand_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
  rk &= k;
  return rk;
#else
  easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);
  easysimd__mmask32 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask32, !!(a_.i8[i] & b_.i8[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_test_epi8_mask
#define _mm256_mask_test_epi8_mask(k, a, b) easysimd_mm256_mask_test_epi8_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_mask_test_epi16_mask (easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_test_epi16_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask16 rk = 0;
  svbool_t pg = svptrue_b16();
  svint16_t svzero =  svdup_n_s16(0);
  EASYSIMD_B16_TO_MASK(rk, svcmpne_s16(pg, svand_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
  EASYSIMD_B16_TO_MASK(rk, svcmpne_s16(pg, svand_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
  rk &= k;
  return rk;
#else
  easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);
  easysimd__mmask16 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask16, !!(a_.i16[i] & b_.i16[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_test_epi16_mask
#define _mm256_mask_test_epi16_mask(k, a, b) easysimd_mm256_mask_test_epi16_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_test_epi32_mask (easysimd__mmask8 k1, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_test_epi32_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    svint32_t svzero =  svdup_n_s32(0);
    EASYSIMD_B32_TO_MASK(rk, svcmpne_s32(pg, svand_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmpne_s32(pg, svand_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm256_test_epi32_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_test_epi32_mask
  #define _mm256_mask_test_epi32_mask(k1, a, b) easysimd_mm256_mask_test_epi32_mask(k1, a, b)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_test_epi64_mask (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_test_epi64_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b64();
  svint64_t svzero =  svdup_n_s64(0);
  EASYSIMD_B64_TO_MASK(rk, svcmpne_s64(pg, svand_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
  EASYSIMD_B64_TO_MASK(rk, svcmpne_s64(pg, svand_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
  rk &= k;
  return rk;
#else
  easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask8, !!(a_.i64[i] & b_.i64[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_test_epi64_mask
#define _mm256_mask_test_epi64_mask(k, a, b) easysimd_mm256_mask_test_epi64_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_test_epi16_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_test_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 res = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(res, svcmpne_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), 0), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(res, svcmpne_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), 0), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(res, svcmpne_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), 0), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(res, svcmpne_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), 0), EASYSIMD_SV_INDEX_3);
    return res;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask32 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
      r |= HEDLEY_STATIC_CAST(easysimd__mmask32, !!(a_.i16[i] & b_.i16[i]) << i);
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_test_epi16_mask
  #define _mm512_test_epi16_mask(a, b) easysimd_mm512_test_epi16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_test_epi32_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_test_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint32_t g_mask_epi32[4] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08};
    uint32x4_t mask_and = vld1q_u32(g_mask_epi32);
    easysimd__m512i tmp;
    tmp.m128i[0].neon_u32 = vandq_u32(vtstq_u32(a.m128i[0].neon_u32, b.m128i[0].neon_u32), mask_and);
    tmp.m128i[1].neon_u32 = vandq_u32(vtstq_u32(a.m128i[1].neon_u32, b.m128i[1].neon_u32), mask_and);
    tmp.m128i[2].neon_u32 = vandq_u32(vtstq_u32(a.m128i[2].neon_u32, b.m128i[2].neon_u32), mask_and);
    tmp.m128i[3].neon_u32 = vandq_u32(vtstq_u32(a.m128i[3].neon_u32, b.m128i[3].neon_u32), mask_and);
    uint32_t r0 = vaddvq_u32(tmp.m128i[0].neon_u32);
    uint32_t r1 = vaddvq_u32(tmp.m128i[1].neon_u32);
    uint32_t r2 = vaddvq_u32(tmp.m128i[2].neon_u32);
    uint32_t r3 = vaddvq_u32(tmp.m128i[3].neon_u32);
    easysimd__mmask16 res = r0 | (r1 << 4) | (r2 << 8) | (r3 << 12);
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdup_n_s32_z(svcmpne_n_s32(pg, svand_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), INT32_C(0)), ~INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32_z(svcmpne_n_s32(pg, svand_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), INT32_C(0)), ~INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svdup_n_s32_z(svcmpne_n_s32(pg, svand_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), INT32_C(0)), ~INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svdup_n_s32_z(svcmpne_n_s32(pg, svand_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), INT32_C(0)), ~INT32_C(0));
    return easysimd_mm512_movepi32_mask(r);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      r |= HEDLEY_STATIC_CAST(easysimd__mmask16, !!(a_.i32[i] & b_.i32[i]) << i);
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_test_epi32_mask
#define _mm512_test_epi32_mask(a, b) easysimd_mm512_test_epi32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_test_epi64_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_test_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint64_t g_mask_epi64[2] __attribute__((aligned(16))) = {0x01, 0x02};
    uint64x2_t mask_and = vld1q_u64(g_mask_epi64);
    easysimd__m512i tmp;
    tmp.m128i[0].neon_u64 = vandq_u64(vtstq_u64(a.m128i[0].neon_u64, b.m128i[0].neon_u64), mask_and);
    tmp.m128i[1].neon_u64 = vandq_u64(vtstq_u64(a.m128i[1].neon_u64, b.m128i[1].neon_u64), mask_and);
    tmp.m128i[2].neon_u64 = vandq_u64(vtstq_u64(a.m128i[2].neon_u64, b.m128i[2].neon_u64), mask_and);
    tmp.m128i[3].neon_u64 = vandq_u64(vtstq_u64(a.m128i[3].neon_u64, b.m128i[3].neon_u64), mask_and);
    uint32_t r0 = vaddvq_u32(tmp.m128i[0].neon_u32);
    uint32_t r1 = vaddvq_u32(tmp.m128i[1].neon_u32);
    uint32_t r2 = vaddvq_u32(tmp.m128i[2].neon_u32);
    uint32_t r3 = vaddvq_u32(tmp.m128i[3].neon_u32);
    easysimd__mmask8 res = r0 | (r1 << 2) | (r2 << 4) | (r3 << 6);
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svdup_n_s64_z(svcmpne_n_s64(pg, svand_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), INT64_C(0)), ~INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svdup_n_s64_z(svcmpne_n_s64(pg, svand_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), INT64_C(0)), ~INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svdup_n_s64_z(svcmpne_n_s64(pg, svand_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]), INT64_C(0)), ~INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svdup_n_s64_z(svcmpne_n_s64(pg, svand_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]), INT64_C(0)), ~INT64_C(0));
    return easysimd_mm512_movepi64_mask(r);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r |= HEDLEY_STATIC_CAST(easysimd__mmask8, !!(a_.i64[i] & b_.i64[i]) << i);
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_test_epi64_mask
  #define _mm512_test_epi64_mask(a, b) easysimd_mm512_test_epi64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_test_epi8_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_test_epi8_mask(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint8_t g_mask_epi8[16] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    uint8x16_t mask_and = vld1q_u8(g_mask_epi8);
    easysimd__m512i tmp;
    tmp.m128i[0].neon_u8 = vandq_u8(vtstq_u8(a.m128i[0].neon_u8, b.m128i[0].neon_u8), mask_and);
    tmp.m128i[1].neon_u8 = vandq_u8(vtstq_u8(a.m128i[1].neon_u8, b.m128i[1].neon_u8), mask_and);
    tmp.m128i[2].neon_u8 = vandq_u8(vtstq_u8(a.m128i[2].neon_u8, b.m128i[2].neon_u8), mask_and);
    tmp.m128i[3].neon_u8 = vandq_u8(vtstq_u8(a.m128i[3].neon_u8, b.m128i[3].neon_u8), mask_and);
    uint64_t r0 = vaddv_u8(vget_low_u8(tmp.m128i[0].neon_u8)) | (vaddv_u8(vget_high_u8(tmp.m128i[0].neon_u8)) << 8);
    uint64_t r1 = vaddv_u8(vget_low_u8(tmp.m128i[1].neon_u8)) | (vaddv_u8(vget_high_u8(tmp.m128i[1].neon_u8)) << 8);
    uint64_t r2 = vaddv_u8(vget_low_u8(tmp.m128i[2].neon_u8)) | (vaddv_u8(vget_high_u8(tmp.m128i[2].neon_u8)) << 8);
    uint64_t r3 = vaddv_u8(vget_low_u8(tmp.m128i[3].neon_u8)) | (vaddv_u8(vget_high_u8(tmp.m128i[3].neon_u8)) << 8);
    easysimd__mmask64 res = r0 | (r1 << 16) | (r2 << 32) | (r3 << 48);
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svdup_n_s8_z(svcmpne_n_s8(pg, svand_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), INT8_C(0)), ~INT8_C(0));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svdup_n_s8_z(svcmpne_n_s8(pg, svand_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), INT8_C(0)), ~INT8_C(0));
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svdup_n_s8_z(svcmpne_n_s8(pg, svand_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), INT8_C(0)), ~INT8_C(0));
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svdup_n_s8_z(svcmpne_n_s8(pg, svand_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), INT8_C(0)), ~INT8_C(0));
    return easysimd_mm512_movepi8_mask(r);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask64 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
      r |= HEDLEY_STATIC_CAST(easysimd__mmask64, HEDLEY_STATIC_CAST(uint64_t, !!(a_.i8[i] & b_.i8[i])) << i);
    }
    return r;

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_test_epi8_mask
  #define _mm512_test_epi8_mask(a, b) easysimd_mm512_test_epi8_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_mask_test_epi16_mask (easysimd__mmask32 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_test_epi16_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b16();
    svuint16_t svzero = svdup_n_u16(0);
    EASYSIMD_B16_TO_MASK(rk, svcmpne_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmpne_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(rk, svcmpne_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), svzero), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(rk, svcmpne_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), svzero), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_test_epi16_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_test_epi16_mask
  #define _mm512_mask_test_epi16_mask(k1, a, b) easysimd_mm512_mask_test_epi16_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_mask_test_epi32_mask (easysimd__mmask16 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_test_epi32_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b32();
    svuint32_t svzero = svdup_n_u32(0);
    EASYSIMD_B32_TO_MASK(rk, svcmpne_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmpne_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(rk, svcmpne_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), svzero), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(rk, svcmpne_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), svzero), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_test_epi32_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_test_epi32_mask
  #define _mm512_mask_test_epi32_mask(k1, a, b) easysimd_mm512_mask_test_epi32_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_mask_test_epi64_mask (easysimd__mmask8 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_test_epi64_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    svuint64_t svzero = svdup_n_u64(0);
    EASYSIMD_B64_TO_MASK(rk, svcmpne_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmpne_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(rk, svcmpne_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]), svzero), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(rk, svcmpne_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]), svzero), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_test_epi64_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_test_epi64_mask
  #define _mm512_mask_test_epi64_mask(k1, a, b) easysimd_mm512_mask_test_epi64_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_mask_test_epi8_mask (easysimd__mmask64 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_test_epi8_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 rk = 0;
    svbool_t pg = svptrue_b8();
    svuint8_t svzero = svdup_n_u8(0);
    EASYSIMD_B8_TO_MASK(rk, svcmpne_u8(pg, svand_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), svzero), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmpne_u8(pg, svand_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), svzero), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(rk, svcmpne_u8(pg, svand_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]), svzero), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(rk, svcmpne_u8(pg, svand_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]), svzero), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_test_epi8_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_test_epi8_mask
  #define _mm512_mask_test_epi8_mask(k1, a, b) easysimd_mm512_mask_test_epi8_mask(k1, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_TEST_H) */
