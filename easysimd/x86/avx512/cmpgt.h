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

#if !defined(EASYSIMD_X86_AVX512_CMPGT_H)
#define EASYSIMD_X86_AVX512_CMPGT_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"
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
easysimd_mm_cmpgt_epi8_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmpgt_epi8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8, b.sve_i8), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
      r |= (a_.i8[i] > b_.i8[i]) ? (UINT8_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpgt_epi8_mask
  #define _mm_cmpgt_epi8_mask(a, b) easysimd_mm_cmpgt_epi8_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmpgt_epi16_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmpgt_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16, b.sve_i16), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
      r |= (a_.i16[i] > b_.i16[i]) ? (UINT16_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpgt_epi16_mask
  #define _mm_cmpgt_epi16_mask(a, b) easysimd_mm_cmpgt_epi16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmpgt_epi32_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmpgt_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32, b.sve_i32), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      r |= (a_.i32[i] > b_.i32[i]) ? (UINT32_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpgt_epi32_mask
  #define _mm_cmpgt_epi32_mask(a, b) easysimd_mm_cmpgt_epi32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmpgt_epi64_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmpgt_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64, b.sve_i64), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r |= (a_.i64[i] > b_.i64[i]) ? (UINT64_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpgt_epi64_mask
  #define _mm_cmpgt_epi64_mask(a, b) easysimd_mm_cmpgt_epi64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_mask_cmpgt_epi8_mask (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmpgt_epi8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8, b.sve_i8), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return easysimd_mm_cmpgt_epi8_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmpgt_epi8_mask
  #define _mm_mask_cmpgt_epi8_mask(k, a, b) easysimd_mm_mask_cmpgt_epi8_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmpgt_epi16_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmpgt_epi16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16, b.sve_i16), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return easysimd_mm_cmpgt_epi16_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmpgt_epi16_mask
  #define _mm_mask_cmpgt_epi16_mask(k, a, b) easysimd_mm_mask_cmpgt_epi16_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmpgt_epi32_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmpgt_epi32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32, b.sve_i32), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return easysimd_mm_cmpgt_epi32_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmpgt_epi32_mask
  #define _mm_mask_cmpgt_epi32_mask(k, a, b) easysimd_mm_mask_cmpgt_epi32_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmpgt_epi64_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmpgt_epi64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64, b.sve_i64), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return easysimd_mm_cmpgt_epi64_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmpgt_epi64_mask
  #define _mm_mask_cmpgt_epi64_mask(k, a, b) easysimd_mm_mask_cmpgt_epi64_mask(k, a, b)
#endif

/* mm128 epu */

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_cmpgt_epu8_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmpgt_epu8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8, b.sve_u8), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
      r |= (a_.u8[i] > b_.u8[i]) ? (UINT8_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpgt_epu8_mask
  #define _mm_cmpgt_epu8_mask(a, b) easysimd_mm_cmpgt_epu8_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmpgt_epu16_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmpgt_epu16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16, b.sve_u16), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
      r |= (a_.u16[i] > b_.u16[i]) ? (UINT16_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpgt_epu16_mask
  #define _mm_cmpgt_epu16_mask(a, b) easysimd_mm_cmpgt_epu16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmpgt_epu32_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmpgt_epu32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32, b.sve_u32), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
      r |= (a_.u32[i] > b_.u32[i]) ? (UINT32_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpgt_epu32_mask
  #define _mm_cmpgt_epu32_mask(a, b) easysimd_mm_cmpgt_epu32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_cmpgt_epu64_mask (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cmpgt_epu64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64, b.sve_u64), EASYSIMD_SV_INDEX_0);
    return rk;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r |= (a_.u64[i] > b_.u64[i]) ? (UINT64_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpgt_epu64_mask
  #define _mm_cmpgt_epu64_mask(a, b) easysimd_mm_cmpgt_epu64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm_mask_cmpgt_epu8_mask (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmpgt_epu8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8, b.sve_u8), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return easysimd_mm_cmpgt_epu8_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmpgt_epu8_mask
  #define _mm_mask_cmpgt_epu8_mask(k, a, b) easysimd_mm_mask_cmpgt_epu8_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmpgt_epu16_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmpgt_epu16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16, b.sve_u16), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return easysimd_mm_cmpgt_epu16_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmpgt_epu16_mask
  #define _mm_mask_cmpgt_epu16_mask(k, a, b) easysimd_mm_mask_cmpgt_epu16_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmpgt_epu32_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmpgt_epu32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32, b.sve_u32), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return easysimd_mm_cmpgt_epu32_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmpgt_epu32_mask
  #define _mm_mask_cmpgt_epu32_mask(k, a, b) easysimd_mm_mask_cmpgt_epu32_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm_mask_cmpgt_epu64_mask (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cmpgt_epu64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64, b.sve_u64), EASYSIMD_SV_INDEX_0);
    rk &= k;
    return rk;
  #else
    return easysimd_mm_cmpgt_epu64_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cmpgt_epu64_mask
  #define _mm_mask_cmpgt_epu64_mask(k, a, b) easysimd_mm_mask_cmpgt_epu64_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_cmpgt_epi8_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmpgt_epi8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return rk;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask32 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
      r |= (a_.i8[i] > b_.i8[i]) ? (UINT8_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epi8_mask
  #define _mm256_cmpgt_epi8_mask(a, b) easysimd_mm256_cmpgt_epi8_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_cmpgt_epi16_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmpgt_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return rk;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
      r |= (a_.i16[i] > b_.i16[i]) ? (UINT16_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epi16_mask
  #define _mm256_cmpgt_epi16_mask(a, b) easysimd_mm256_cmpgt_epi16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_cmpgt_epi32_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmpgt_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return rk;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      r |= (a_.i32[i] > b_.i32[i]) ? (UINT32_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epi32_mask
  #define _mm256_cmpgt_epi32_mask(a, b) easysimd_mm256_cmpgt_epi32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_cmpgt_epi64_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmpgt_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return rk;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r |= (a_.i64[i] > b_.i64[i]) ? (UINT64_C(1) << i) : 0;
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epi64_mask
  #define _mm256_cmpgt_epi64_mask(a, b) easysimd_mm256_cmpgt_epi64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_mask_cmpgt_epi8_mask (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmpgt_epi8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return easysimd_mm256_cmpgt_epi8_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmpgt_epi8_mask
  #define _mm256_mask_cmpgt_epi8_mask(k, a, b) easysimd_mm256_mask_cmpgt_epi8_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_mask_cmpgt_epi16_mask (easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmpgt_epi16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return easysimd_mm256_cmpgt_epi16_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmpgt_epi16_mask
  #define _mm256_mask_cmpgt_epi16_mask(k, a, b) easysimd_mm256_mask_cmpgt_epi16_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_cmpgt_epi32_mask (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmpgt_epi32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return easysimd_mm256_cmpgt_epi32_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmpgt_epi32_mask
  #define _mm256_mask_cmpgt_epi32_mask(k, a, b) easysimd_mm256_mask_cmpgt_epi32_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_cmpgt_epi64_mask (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmpgt_epi64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return easysimd_mm256_cmpgt_epi64_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmpgt_epi64_mask
  #define _mm256_mask_cmpgt_epi64_mask(k, a, b) easysimd_mm256_mask_cmpgt_epi64_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_cmpgt_epu8_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmpgt_epu8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 k = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(k, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(k, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask32 r = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
      r |= (a_.u8[i] > b_.u8[i]) ? (UINT64_C(1) << i) : 0;
    }

    return r;
  #endif
} 
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epu8_mask
  #define _mm256_cmpgt_epu8_mask(a, b) easysimd_mm256_cmpgt_epu8_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_cmpgt_epu16_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmpgt_epu16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 k = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(k, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(k, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
      r |= (a_.u16[i] > b_.u16[i]) ? (UINT32_C(1) << i) : 0;
    }

    return r;
  #endif
} 
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epu16_mask
  #define _mm256_cmpgt_epu16_mask(a, b) easysimd_mm256_cmpgt_epu16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_cmpgt_epu32_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)  && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmpgt_epu32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(k, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
      r |= (a_.u32[i] > b_.u32[i]) ? (UINT32_C(1) << i) : 0;
    }

    return r;
  #endif
} 
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epu32_mask
  #define _mm256_cmpgt_epu32_mask(a, b) easysimd_mm256_cmpgt_epu32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_cmpgt_epu64_mask (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)  && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cmpgt_epu64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 k = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(k, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(k, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    return k;
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r |= (a_.u64[i] > b_.u64[i]) ? (UINT32_C(1) << i) : 0;
    }

    return r;
  #endif
} 
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epu64_mask
  #define _mm256_cmpgt_epu64_mask(a, b) easysimd_mm256_cmpgt_epu64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm256_mask_cmpgt_epu8_mask (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmpgt_epu8_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return easysimd_mm256_cmpgt_epu8_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmpgt_epu8_mask
  #define _mm256_mask_cmpgt_epu8_mask(k, a, b) easysimd_mm256_mask_cmpgt_epu8_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm256_mask_cmpgt_epu16_mask (easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmpgt_epu16_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return easysimd_mm256_cmpgt_epu16_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmpgt_epu16_mask
  #define _mm256_mask_cmpgt_epu16_mask(k, a, b) easysimd_mm256_mask_cmpgt_epu16_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_cmpgt_epu32_mask (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmpgt_epu32_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return easysimd_mm256_cmpgt_epu32_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmpgt_epu32_mask
  #define _mm256_mask_cmpgt_epu32_mask(k, a, b) easysimd_mm256_mask_cmpgt_epu32_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm256_mask_cmpgt_epu64_mask (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cmpgt_epu64_mask(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    rk &= k;
    return rk;
  #else
    return easysimd_mm256_cmpgt_epu64_mask(a, b) & k;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cmpgt_epu64_mask
  #define _mm256_mask_cmpgt_epu64_mask(k, a, b) easysimd_mm256_mask_cmpgt_epu64_mask(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_cmpgt_epi8_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cmpgt_epi8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return rk;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask64 r;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256) && !defined(HEDLEY_INTEL_VERSION)
      r = 0;

      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        const uint32_t t = HEDLEY_STATIC_CAST(uint32_t, easysimd_mm256_movemask_epi8(easysimd_mm256_cmpgt_epi8(a_.m256i[i], b_.m256i[i])));
        r |= HEDLEY_STATIC_CAST(uint64_t, t) << HEDLEY_STATIC_CAST(uint64_t, i * 32);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      easysimd__m512i_private tmp;

      tmp.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(tmp.i8), a_.i8 > b_.i8);
      r = easysimd_mm512_movepi8_mask(easysimd__m512i_from_private(tmp));
    #else
      r = 0;

      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
        r |= (a_.i8[i] > b_.i8[i]) ? (UINT64_C(1) << i) : 0;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpgt_epi8_mask
  #define _mm512_cmpgt_epi8_mask(a, b) easysimd_mm512_cmpgt_epi8_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_cmpgt_epu8_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cmpgt_epu8_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return rk;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask64 r = 0;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      easysimd__m512i_private tmp;

      tmp.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(tmp.i8), a_.u8 > b_.u8);
      r = easysimd_mm512_movepi8_mask(easysimd__m512i_from_private(tmp));
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
        r |= (a_.u8[i] > b_.u8[i]) ? (UINT64_C(1) << i) : 0;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpgt_epu8_mask
  #define _mm512_cmpgt_epu8_mask(a, b) easysimd_mm512_cmpgt_epu8_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_cmpgt_epi16_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cmpgt_epi16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return rk;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask64 r;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      easysimd__m512i_private tmp;

      tmp.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(tmp.i16), a_.i16 > b_.i16);
      r = easysimd_mm512_movepi16_mask(easysimd__m512i_from_private(tmp));
    #else
      r = 0;

      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i16) / sizeof(a_.i16[0])) ; i++) {
        r |= (a_.i16[i] > b_.i16[i]) ? (UINT32_C(1) << i) : 0;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X166_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpgt_epi16_mask
  #define _mm512_cmpgt_epi16_mask(a, b) easysimd_mm512_cmpgt_epi16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_mask_cmpgt_epi8_mask (easysimd__mmask64 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_cmpgt_epi8_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_cmpgt_epi8_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmpgt_epi8_mask
  #define _mm512_mask_cmpgt_epi8_mask(k1, a, b) easysimd_mm512_mask_cmpgt_epi8_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_mask_cmpgt_epi16_mask (easysimd__mmask32 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_cmpgt_epi16_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_cmpgt_epi16_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmpgt_epi16_mask
  #define _mm512_mask_cmpgt_epi16_mask(k1, a, b) easysimd_mm512_mask_cmpgt_epi16_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_cmpgt_epi32_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cmpgt_epi32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return rk;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__mmask16 mask;
    easysimd__mmask16 *k = &mask;
    easysimd__m512i res;
    res.m128i[0].neon_u32 = vcgtq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    res.m128i[1].neon_u32 = vcgtq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    res.m128i[2].neon_u32 = vcgtq_s32(a.m128i[2].neon_i32, b.m128i[2].neon_i32);
    res.m128i[3].neon_u32 = vcgtq_s32(a.m128i[3].neon_i32, b.m128i[3].neon_i32);
    EXTRACT_HB_32x16(res, k);
    return mask;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
      r_.m256i[i] = easysimd_mm256_cmpgt_epi32(a_.m256i[i], b_.m256i[i]);
    }

    return easysimd_mm512_movepi32_mask(easysimd__m512i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpgt_epi32_mask
  #define _mm512_cmpgt_epi32_mask(a, b) easysimd_mm512_cmpgt_epi32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_mask_cmpgt_epi32_mask (easysimd__mmask16 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cmpgt_epi32_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_cmpgt_epi32_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmpgt_epi32_mask
  #define _mm512_mask_cmpgt_epi32_mask(k1, a, b) easysimd_mm512_mask_cmpgt_epi32_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_cmpgt_epi64_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cmpgt_epi64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return rk;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
      r_.m256i[i] = easysimd_mm256_cmpgt_epi64(a_.m256i[i], b_.m256i[i]);
    }

    return easysimd_mm512_movepi64_mask(easysimd__m512i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpgt_epi64_mask
  #define _mm512_cmpgt_epi64_mask(a, b) easysimd_mm512_cmpgt_epi64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_mask_cmpgt_epi64_mask (easysimd__mmask8 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cmpgt_epi64_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_cmpgt_epi64_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmpgt_epi64_mask
  #define _mm512_mask_cmpgt_epi64_mask(k1, a, b) easysimd_mm512_mask_cmpgt_epi64_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_cmpgt_epu16_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cmpgt_epu16_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return rk;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask32 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
      r |= (a_.u16[i] > b_.u16[i]) ? (UINT32_C(1) << i) : 0;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X166_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpgt_epu16_mask
  #define _mm512_cmpgt_epu16_mask(a, b) easysimd_mm512_cmpgt_epu16_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_cmpgt_epu32_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cmpgt_epu32_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return rk;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask16 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
      r |= (a_.u32[i] > b_.u32[i]) ? (UINT32_C(1) << i) : 0;
    }

    return r;

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpgt_epu32_mask
  #define _mm512_cmpgt_epu32_mask(a, b) easysimd_mm512_cmpgt_epu32_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_cmpgt_epu64_mask (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cmpgt_epu64_mask(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    return rk;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    easysimd__mmask8 r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r |= (a_.u64[i] > b_.u64[i]) ? (UINT64_C(1) << i) : 0;
    }

    return r;

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cmpgt_epu64_mask
  #define _mm512_cmpgt_epu64_mask(a, b) easysimd_mm512_cmpgt_epu64_mask(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_mm512_mask_cmpgt_epu8_mask (easysimd__mmask64 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_cmpgt_epu8_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask64 rk = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B8_TO_MASK(rk, svcmpgt_u8(pg, a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_cmpgt_epu8_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmpgt_epu8_mask
  #define _mm512_mask_cmpgt_epu8_mask(k1, a, b) easysimd_mm512_mask_cmpgt_epu8_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_mm512_mask_cmpgt_epu16_mask (easysimd__mmask32 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_cmpgt_epu16_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask32 rk = 0;
    svbool_t pg = svptrue_b16();
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B16_TO_MASK(rk, svcmpgt_u16(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_cmpgt_epu16_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmpgt_epu16_mask
  #define _mm512_mask_cmpgt_epu16_mask(k1, a, b) easysimd_mm512_mask_cmpgt_epu16_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_mm512_mask_cmpgt_epu32_mask (easysimd__mmask16 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cmpgt_epu32_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask16 rk = 0;
    svbool_t pg = svptrue_b32();
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B32_TO_MASK(rk, svcmpgt_u32(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_cmpgt_epu32_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmpgt_epu32_mask
  #define _mm512_mask_cmpgt_epu32_mask(k1, a, b) easysimd_mm512_mask_cmpgt_epu32_mask(k1, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_mm512_mask_cmpgt_epu64_mask (easysimd__mmask8 k1, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cmpgt_epu64_mask(k1, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__mmask8 rk = 0;
    svbool_t pg = svptrue_b64();
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), EASYSIMD_SV_INDEX_1);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]), EASYSIMD_SV_INDEX_2);
    EASYSIMD_B64_TO_MASK(rk, svcmpgt_u64(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]), EASYSIMD_SV_INDEX_3);
    rk &= k1;
    return rk;
  #else
    return easysimd_mm512_cmpgt_epu64_mask(a, b) & k1;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cmpgt_epu64_mask
  #define _mm512_mask_cmpgt_epu64_mask(k1, a, b) easysimd_mm512_mask_cmpgt_epu64_mask(k1, a, b)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_CMPGT_H) */
