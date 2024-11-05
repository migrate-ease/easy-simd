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
 *   2021      Andrew Rodriguez <anrodriguez@linkedin.com>
 */

#if !defined(EASYSIMD_X86_AVX512_TESTN_H)
#define EASYSIMD_X86_AVX512_TESTN_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_testn_epi8_mask (easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_testn_epi8_mask(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask16 rk = 0;
  svbool_t pg = svptrue_b8();
  svint8_t svzero =  svdup_n_s8(0);
  EASYSIMD_B8_TO_MASK(rk, svcmpeq_s8(pg, svand_s8_x(pg, a.sve_i8, b.sve_i8), svzero), EASYSIMD_SV_INDEX_0);
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask16 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
    r |= HEDLEY_STATIC_CAST(easysimd__mmask16, !(a_.i8[i] & b_.i8[i]) << i);
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_testn_epi8_mask
#define _mm_testn_epi8_mask(a, b) easysimd_mm_testn_epi8_mask(a, b)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_testn_epi16_mask (easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_testn_epi16_mask(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b16();
  svint16_t svzero =  svdup_n_s16(0);
  EASYSIMD_B16_TO_MASK(rk, svcmpeq_s16(pg, svand_s16_x(pg, a.sve_i16, b.sve_i16), svzero), EASYSIMD_SV_INDEX_0);
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
    r |= HEDLEY_STATIC_CAST(easysimd__mmask16, !(a_.i16[i] & b_.i16[i]) << i);
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_testn_epi16_mask
#define _mm_testn_epi16_mask(a, b) easysimd_mm_testn_epi16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_testn_epi32_mask (easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_testn_epi32_mask(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b32();
  svint32_t svzero =  svdup_n_s32(0);
  EASYSIMD_B32_TO_MASK(rk, svcmpeq_s32(pg, svand_s32_x(pg, a.sve_i32, b.sve_i32), svzero), EASYSIMD_SV_INDEX_0);
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
    r |= HEDLEY_STATIC_CAST(easysimd__mmask32, !(a_.i32[i] & b_.i32[i]) << i);
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_testn_epi32_mask
#define _mm_testn_epi32_mask(a, b) easysimd_mm_testn_epi32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_testn_epi64_mask (easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_testn_epi64_mask(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b64();
  svint64_t svzero =  svdup_n_s64(0);
  EASYSIMD_B64_TO_MASK(rk, svcmpeq_s64(pg, svand_s64_x(pg, a.sve_i64, b.sve_i64), svzero), EASYSIMD_SV_INDEX_0);
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
    r |= HEDLEY_STATIC_CAST(easysimd__mmask64, !(a_.i64[i] & b_.i64[i]) << i);
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_testn_epi64_mask
#define _mm_testn_epi64_mask(a, b) easysimd_mm_testn_epi64_mask(a, b)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_mask_testn_epi8_mask (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_testn_epi8_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask16 rk = 0;
  svbool_t pg = svptrue_b8();
  svint8_t svzero =  svdup_n_s8(0);
  EASYSIMD_B8_TO_MASK(rk, svcmpeq_s8(pg, svand_s8_x(pg, a.sve_i8, b.sve_i8), svzero), EASYSIMD_SV_INDEX_0);
  rk &= k;
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask16 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask16, !(a_.i8[i] & b_.i8[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_testn_epi8_mask
#define _mm_mask_testn_epi8_mask(k, a, b) easysimd_mm_mask_testn_epi8_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_testn_epi16_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_testn_epi16_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b16();
  svint16_t svzero =  svdup_n_s16(0);
  EASYSIMD_B16_TO_MASK(rk, svcmpeq_s16(pg, svand_s16_x(pg, a.sve_i16, b.sve_i16), svzero), EASYSIMD_SV_INDEX_0);
  rk &= k;
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask16, !(a_.i16[i] & b_.i16[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_testn_epi16_mask
#define _mm_mask_testn_epi16_mask(k, a, b) easysimd_mm_mask_testn_epi16_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_testn_epi32_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_testn_epi32_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b32();
  svint32_t svzero =  svdup_n_s32(0);
  EASYSIMD_B32_TO_MASK(rk, svcmpeq_s32(pg, svand_s32_x(pg, a.sve_i32, b.sve_i32), svzero), EASYSIMD_SV_INDEX_0);
  rk &= k;
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask32, !(a_.i32[i] & b_.i32[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_testn_epi32_mask
#define _mm_mask_testn_epi32_mask(k, a, b) easysimd_mm_mask_testn_epi32_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_testn_epi64_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_testn_epi64_mask(k, a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__mmask8 rk = 0;
  svbool_t pg = svptrue_b64();
  svint64_t svzero =  svdup_n_s64(0);
  EASYSIMD_B64_TO_MASK(rk, svcmpeq_s64(pg, svand_s64_x(pg, a.sve_i64, b.sve_i64), svzero), EASYSIMD_SV_INDEX_0);
  rk &= k;
  return rk;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);
  easysimd__mmask8 r = 0;

  EASYSIMD_VECTORIZE_REDUCTION(|:r)
  for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
    r |= (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(easysimd__mmask64, !(a_.i64[i] & b_.i64[i]) << i) : 0;
  }
  return r;

#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_testn_epi64_mask
#define _mm_mask_testn_epi64_mask(k, a, b) easysimd_mm_mask_testn_epi64_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_testn_epi64_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_testn_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 res = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(res, svcmpeq_n_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), 0), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(res, svcmpeq_n_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), 0), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(res, svcmpeq_n_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]), 0), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(res, svcmpeq_n_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]), 0), EASYSIMD_SV_INDEX_3);
    return res;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r |= (!(a_.i64[i] & b_.i64[i])) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_testn_epi64_mask
  #define _mm512_testn_epi64_mask(a, b) easysimd_mm512_testn_epi64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_testn_epi64_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_testn_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 res = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(res, svcmpeq_n_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), 0), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(res, svcmpeq_n_u64(pg, svand_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), 0), EASYSIMD_SV_INDEX_1);
    return res;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r |= (!(a_.i64[i] & b_.i64[i])) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testn_epi64_mask
  #define _mm256_testn_epi64_mask(a, b) easysimd_mm256_testn_epi64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_testn_epi32_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_testn_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 res = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(res, svcmpeq_n_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), 0), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(res, svcmpeq_n_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), 0), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(res, svcmpeq_n_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), 0), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(res, svcmpeq_n_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), 0), EASYSIMD_SV_INDEX_3);
    return res;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      r |= (!(a_.i32[i] & b_.i32[i])) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_testn_epi32_mask
  #define _mm512_testn_epi32_mask(a, b) easysimd_mm512_testn_epi32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_testn_epi32_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_testn_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 res = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(res, svcmpeq_n_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), 0), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(res, svcmpeq_n_u32(pg, svand_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), 0), EASYSIMD_SV_INDEX_1);
    return res;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      r |= (!(a_.i32[i] & b_.i32[i])) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testn_epi32_mask
  #define _mm256_testn_epi32_mask(a, b) easysimd_mm256_testn_epi32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_testn_epi16_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_testn_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 res = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(res, svcmpeq_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), 0), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(res, svcmpeq_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), 0), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(res, svcmpeq_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), 0), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(res, svcmpeq_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), 0), EASYSIMD_SV_INDEX_3);
    return res;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask32 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
      r |= (!(a_.i16[i] & b_.i16[i])) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_testn_epi16_mask
  #define _mm512_testn_epi16_mask(a, b) easysimd_mm512_testn_epi16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_testn_epi16_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_testn_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 res = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(res, svcmpeq_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), 0), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(res, svcmpeq_n_u16(pg, svand_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), 0), EASYSIMD_SV_INDEX_1);
    return res;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask32 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
      r |= (!(a_.i16[i] & b_.i16[i])) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testn_epi16_mask
  #define _mm256_testn_epi16_mask(a, b) easysimd_mm256_testn_epi16_mask(a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_TESTN_H) */
