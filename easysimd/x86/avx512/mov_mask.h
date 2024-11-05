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

#if !defined(EASYSIMD_X86_AVX512_MOV_MASK_H)
#define EASYSIMD_X86_AVX512_MOV_MASK_H

#include "types.h"
#include "../avx2.h"

#include "cast.h"
#include "set.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_movepi8_mask (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_movepi8_mask(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return HEDLEY_STATIC_CAST(easysimd__mmask16, easysimd_mm_movemask_epi8(a));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B8_TO_MASK(k, svcmplt_s8(svptrue_b8(), a.sve_i8, svdup_n_s8(0)), EASYSIMD_SV_INDEX_0);
    return (easysimd__mmask16)k;
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
      r |= (a_.i8[i] < 0) ? (UINT64_C(1) << i) : 0;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_movepi8_mask
  #define _mm_movepi8_mask(a) easysimd_mm_movepi8_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_movepi16_mask (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_movepi16_mask(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    /* There is no 32-bit _mm_movemask_* function, so we use
     * _mm_movemask_epi8 then extract the odd bits. */
    uint_fast16_t r = HEDLEY_STATIC_CAST(uint_fast16_t, easysimd_mm_movemask_epi8(a));
    r = (    (r >> 1)) & UINT32_C(0x5555);
    r = (r | (r >> 1)) & UINT32_C(0x3333);
    r = (r | (r >> 2)) & UINT32_C(0x0f0f);
    r = (r | (r >> 4)) & UINT32_C(0x00ff);
    return HEDLEY_STATIC_CAST(easysimd__mmask8, r);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B16_TO_MASK(k, svcmplt_s16(svptrue_b16(), a.sve_i16, svdup_n_s16(0)), EASYSIMD_SV_INDEX_0);
    return (easysimd__mmask8)k;
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
      r |= (a_.i16[i] < 0) ? (UINT32_C(1) << i) : 0;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_movepi16_mask
  #define _mm_movepi16_mask(a) easysimd_mm_movepi16_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_movepi32_mask (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_movepi32_mask(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B32_TO_MASK(k, svcmplt_s32(svptrue_b32(), a.sve_i32, svdup_n_s32(0)), EASYSIMD_SV_INDEX_0);
    return (easysimd__mmask8)k;
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return HEDLEY_STATIC_CAST(easysimd__mmask8, easysimd_mm_movemask_ps(easysimd_mm_castsi128_ps(a)));
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      r |= (a_.i32[i] < 0) ? (UINT32_C(1) << i) : 0;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_movepi32_mask
  #define _mm_movepi32_mask(a) easysimd_mm_movepi32_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_movepi64_mask (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_movepi64_mask(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return HEDLEY_STATIC_CAST(easysimd__mmask8, easysimd_mm_movemask_pd(easysimd_mm_castsi128_pd(a)));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64, svdup_n_s64(0)), EASYSIMD_SV_INDEX_0);
    return (easysimd__mmask8)k;
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r |= (a_.i64[i] < 0) ? (UINT32_C(1) << i) : 0;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_movepi64_mask
  #define _mm_movepi64_mask(a) easysimd_mm_movepi64_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_movepi8_mask (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_movepi8_mask(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B8_TO_MASK(k, svcmplt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_0], svdup_n_s8(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(k, svcmplt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_1], svdup_n_s8(0)), EASYSIMD_SV_INDEX_1);
    return (easysimd__mmask32)k;
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    easysimd__mmask32 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
        r |= HEDLEY_STATIC_CAST(easysimd__mmask32, easysimd_mm_movepi8_mask(a_.m128i[i])) << (i * 16);
      }
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
        r |= (a_.i8[i] < 0) ? (UINT64_C(1) << i) : 0;
      }
    #endif

    return HEDLEY_STATIC_CAST(easysimd__mmask32, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_movepi8_mask
  #define _mm256_movepi8_mask(a) easysimd_mm256_movepi8_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_movepi16_mask (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_movepi16_mask(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B16_TO_MASK(k, svcmplt_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_0], svdup_n_s16(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(k, svcmplt_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_1], svdup_n_s16(0)), EASYSIMD_SV_INDEX_1);
    return (easysimd__mmask16)k;
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    easysimd__mmask16 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
        r |= HEDLEY_STATIC_CAST(easysimd__mmask16, easysimd_mm_movepi16_mask(a_.m128i[i])) << (i * 8);
      }
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
        r |= (a_.i16[i] < 0) ? (UINT32_C(1) << i) : 0;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_movepi16_mask
  #define _mm256_movepi16_mask(a) easysimd_mm256_movepi16_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_movepi32_mask (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_movepi32_mask(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B32_TO_MASK(k, svcmplt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], svdup_n_s32(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmplt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], svdup_n_s32(0)), EASYSIMD_SV_INDEX_1);
    return (easysimd__mmask8)k;
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    easysimd__mmask8 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
        r |= HEDLEY_STATIC_CAST(easysimd__mmask16, easysimd_mm_movepi32_mask(a_.m128i[i])) << (i * 4);
      }
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
        r |= (a_.i32[i] < 0) ? (UINT32_C(1) << i) : 0;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_movepi32_mask
  #define _mm256_movepi32_mask(a) easysimd_mm256_movepi32_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_movepi64_mask (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_movepi64_mask(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_0], svdup_n_s64(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_1], svdup_n_s64(0)), EASYSIMD_SV_INDEX_1);
    return (easysimd__mmask8)k;
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    easysimd__mmask8 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
        r |= HEDLEY_STATIC_CAST(easysimd__mmask8, easysimd_mm_movepi64_mask(a_.m128i[i])) << (i * 2);
      }
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
        r |= (a_.i64[i] < 0) ? (UINT32_C(1) << i) : 0;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_movepi64_mask
  #define _mm256_movepi64_mask(a) easysimd_mm256_movepi64_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_movepi8_mask (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_movepi8_mask(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B8_TO_MASK(k, svcmplt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_0], svdup_n_s8(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(k, svcmplt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_1], svdup_n_s8(0)), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(k, svcmplt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_2], svdup_n_s8(0)), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(k, svcmplt_s8(svptrue_b8(), a.sve_i8[EASYSIMD_SV_INDEX_3], svdup_n_s8(0)), EASYSIMD_SV_INDEX_3);
    return (easysimd__mmask64)k;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    easysimd__mmask64 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        r |= HEDLEY_STATIC_CAST(easysimd__mmask64, easysimd_mm256_movepi8_mask(a_.m256i[i])) << (i * 32);
      }
    #else
      r = 0;

      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
        r |= (a_.i8[i] < 0) ? (UINT64_C(1) << i) : 0;
      }
    #endif

    return HEDLEY_STATIC_CAST(easysimd__mmask64, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_movepi8_mask
  #define _mm512_movepi8_mask(a) easysimd_mm512_movepi8_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_movepi16_mask (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_movepi16_mask(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B16_TO_MASK(k, svcmplt_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_0], svdup_n_s16(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(k, svcmplt_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_1], svdup_n_s16(0)), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(k, svcmplt_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_2], svdup_n_s16(0)), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(k, svcmplt_s16(svptrue_b16(), a.sve_i16[EASYSIMD_SV_INDEX_3], svdup_n_s16(0)), EASYSIMD_SV_INDEX_3);
    return (easysimd__mmask32)k;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    easysimd__mmask32 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        r |= HEDLEY_STATIC_CAST(easysimd__mmask32, easysimd_mm256_movepi16_mask(a_.m256i[i])) << (i * 16);
      }
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
        r |= (a_.i16[i] < 0) ? (UINT32_C(1) << i) : 0;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_movepi16_mask
  #define _mm512_movepi16_mask(a) easysimd_mm512_movepi16_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_movepi32_mask (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_movepi32_mask(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B32_TO_MASK(k, svcmplt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], svdup_n_s32(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmplt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], svdup_n_s32(0)), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(k, svcmplt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_2], svdup_n_s32(0)), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(k, svcmplt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_3], svdup_n_s32(0)), EASYSIMD_SV_INDEX_3);
    return (easysimd__mmask16)k;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    easysimd__mmask16 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        r |= HEDLEY_STATIC_CAST(easysimd__mmask16, easysimd_mm256_movepi32_mask(a_.m256i[i])) << (i * 8);
      }
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
        r |= (a_.i32[i] < 0) ? (UINT32_C(1) << i) : 0;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_movepi32_mask
  #define _mm512_movepi32_mask(a) easysimd_mm512_movepi32_mask(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_movepi64_mask (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_movepi64_mask(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_0], svdup_n_s64(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_1], svdup_n_s64(0)), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_2], svdup_n_s64(0)), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_3], svdup_n_s64(0)), EASYSIMD_SV_INDEX_3);
    return (easysimd__mmask8)k;
  #else
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);
    easysimd__mmask8 r = 0;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        r |= easysimd_mm256_movepi64_mask(a_.m256i[i]) << (i * 4);
      }
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
        r |= (a_.i64[i] < 0) ? (UINT32_C(1) << i) : 0;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_movepi64_mask
  #define _mm512_movepi64_mask(a) easysimd_mm512_movepi64_mask(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_MOV_MASK_H) */
