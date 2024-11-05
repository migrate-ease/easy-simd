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
 */

#if !defined(EASYSIMD_X86_AVX512_PACKS_H)
#define EASYSIMD_X86_AVX512_PACKS_H

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
easysimd_mm_mask_packs_epi16 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_packs_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svuzp1_s8(svqxtnb_s16(a.sve_i16), svqxtnb_s16(b.sve_i16)), src.sve_i8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      int16_t v = (i < (sizeof(a_.i16) / sizeof(a_.i16[0]))) ? a_.i16[i] : b_.i16[i & 7];
      r_.i8[i] = ((k >> i) & 1) ? ((v < INT8_MIN) ? INT8_MIN : ((v > INT8_MAX) ? INT8_MAX : HEDLEY_STATIC_CAST(int8_t, v))) : src_.i8[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_packs_epi16(src, k, a, b) easysimd_mm_mask_packs_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_packs_epi16 (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_packs_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svuzp1_s8(svqxtnb_s16(a.sve_i16), svqxtnb_s16(b.sve_i16)), svdup_n_s8(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      int16_t v = (i < (sizeof(a_.i16) / sizeof(a_.i16[0]))) ? a_.i16[i] : b_.i16[i & 7];
      r_.i8[i] = ((k >> i) & 1) ? ((v < INT8_MIN) ? INT8_MIN : ((v > INT8_MAX) ? INT8_MAX : HEDLEY_STATIC_CAST(int8_t, v))) : INT8_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_packs_epi16(k, a, b) easysimd_mm_maskz_packs_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_packs_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_packs_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svuzp1_s16(svqxtnb_s32(a.sve_i32), svqxtnb_s32(b.sve_i32)), src.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      int32_t v = (i < (sizeof(a_.i32) / sizeof(a_.i32[0]))) ? a_.i32[i] : b_.i32[i & 3];
      r_.i16[i] = ((k >> i) & 1) ? ((v < INT16_MIN) ? INT16_MIN : ((v > INT16_MAX) ? INT16_MAX : HEDLEY_STATIC_CAST(int16_t, v))) : src_.i16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_packs_epi32(src, k, a, b) easysimd_mm_mask_packs_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_packs_epi32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_packs_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svuzp1_s16(svqxtnb_s32(a.sve_i32), svqxtnb_s32(b.sve_i32)), svdup_n_s16(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      int32_t v = (i < (sizeof(a_.i32) / sizeof(a_.i32[0]))) ? a_.i32[i] : b_.i32[i & 3];
      r_.i16[i] = ((k >> i) & 1) ? ((v < INT16_MIN) ? INT16_MIN : ((v > INT16_MAX) ? INT16_MAX : HEDLEY_STATIC_CAST(int16_t, v))) : INT16_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_packs_epi32(k, a, b) easysimd_mm_maskz_packs_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_packs_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_packs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svuzp1_s8(svqxtnb_s16(a.sve_i16[EASYSIMD_SV_INDEX_0]), svqxtnb_s16(b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svuzp1_s8(svqxtnb_s16(a.sve_i16[EASYSIMD_SV_INDEX_1]), svqxtnb_s16(b.sve_i16[EASYSIMD_SV_INDEX_1]));
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svuzp1_s8(svqxtnb_s16(a.sve_i16[EASYSIMD_SV_INDEX_2]), svqxtnb_s16(b.sve_i16[EASYSIMD_SV_INDEX_2]));
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svuzp1_s8(svqxtnb_s16(a.sve_i16[EASYSIMD_SV_INDEX_3]), svqxtnb_s16(b.sve_i16[EASYSIMD_SV_INDEX_3]));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      r_.m256i[0] = easysimd_mm256_packs_epi16(a_.m256i[0], b_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_packs_epi16(a_.m256i[1], b_.m256i[1]);
    #else
      const size_t halfway_point = (sizeof(r_.i8) / sizeof(r_.i8[0])) / 2;
      const size_t quarter_point = (sizeof(r_.i8) / sizeof(r_.i8[0])) / 4;
      const size_t octet_point = (sizeof(r_.i8) / sizeof(r_.i8[0])) / 8;

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < octet_point ; i++) {
        r_.i8[i] = (a_.i16[i] > INT8_MAX) ? INT8_MAX : ((a_.i16[i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, a_.i16[i]));
        r_.i8[i + octet_point] = (b_.i16[i] > INT8_MAX) ? INT8_MAX : ((b_.i16[i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, b_.i16[i]));
        r_.i8[quarter_point + i] = (a_.i16[octet_point + i] > INT8_MAX) ? INT8_MAX : ((a_.i16[octet_point + i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, a_.i16[octet_point + i]));
        r_.i8[quarter_point + i + octet_point] = (b_.i16[octet_point + i] > INT8_MAX) ? INT8_MAX : ((b_.i16[octet_point + i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, b_.i16[octet_point + i]));
        r_.i8[halfway_point + i] = (a_.i16[quarter_point + i] > INT8_MAX) ? INT8_MAX : ((a_.i16[quarter_point + i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, a_.i16[quarter_point + i]));
        r_.i8[halfway_point + i + octet_point] = (b_.i16[quarter_point + i] > INT8_MAX) ? INT8_MAX : ((b_.i16[quarter_point + i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, b_.i16[quarter_point + i]));
        r_.i8[halfway_point + quarter_point + i]     = (a_.i16[quarter_point + octet_point + i] > INT8_MAX) ? INT8_MAX : ((a_.i16[quarter_point + octet_point + i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, a_.i16[quarter_point + octet_point + i]));
        r_.i8[halfway_point + quarter_point + i + octet_point] = (b_.i16[quarter_point + octet_point + i] > INT8_MAX) ? INT8_MAX : ((b_.i16[quarter_point + octet_point + i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, b_.i16[quarter_point + octet_point + i]));
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_packs_epi16
  #define _mm512_packs_epi16(a, b) easysimd_mm512_packs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_packs_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_packs_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svuzp1_s16(svqxtnb_s32(a.sve_i32[EASYSIMD_SV_INDEX_0]), svqxtnb_s32(b.sve_i32[EASYSIMD_SV_INDEX_0]));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svuzp1_s16(svqxtnb_s32(a.sve_i32[EASYSIMD_SV_INDEX_1]), svqxtnb_s32(b.sve_i32[EASYSIMD_SV_INDEX_1]));
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svuzp1_s16(svqxtnb_s32(a.sve_i32[EASYSIMD_SV_INDEX_2]), svqxtnb_s32(b.sve_i32[EASYSIMD_SV_INDEX_2]));
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svuzp1_s16(svqxtnb_s32(a.sve_i32[EASYSIMD_SV_INDEX_3]), svqxtnb_s32(b.sve_i32[EASYSIMD_SV_INDEX_3]));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i16 = vcombine_s16(vqmovn_s32(a.m128i[0].neon_i32), vqmovn_s32(b.m128i[0].neon_i32));
    r.m128i[1].neon_i16 = vcombine_s16(vqmovn_s32(a.m128i[1].neon_i32), vqmovn_s32(b.m128i[1].neon_i32));
    r.m128i[2].neon_i16 = vcombine_s16(vqmovn_s32(a.m128i[2].neon_i32), vqmovn_s32(b.m128i[2].neon_i32));
    r.m128i[3].neon_i16 = vcombine_s16(vqmovn_s32(a.m128i[3].neon_i32), vqmovn_s32(b.m128i[3].neon_i32));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      r_.m256i[0] = easysimd_mm256_packs_epi32(a_.m256i[0], b_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_packs_epi32(a_.m256i[1], b_.m256i[1]);
    #else
      const size_t halfway_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 2;
      const size_t quarter_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 4;
      const size_t octet_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 8;

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < octet_point ; i++) {
        r_.i16[i] = (a_.i32[i] > INT16_MAX) ? INT16_MAX : ((a_.i32[i] < INT16_MIN) ? INT16_MIN : HEDLEY_STATIC_CAST(int16_t, a_.i32[i]));
        r_.i16[i + octet_point] = (b_.i32[i] > INT16_MAX) ? INT16_MAX : ((b_.i32[i] < INT16_MIN) ? INT16_MIN : HEDLEY_STATIC_CAST(int16_t, b_.i32[i]));
        r_.i16[quarter_point + i] = (a_.i32[octet_point + i] > INT16_MAX) ? INT16_MAX : ((a_.i32[octet_point + i] < INT16_MIN) ? INT16_MIN : HEDLEY_STATIC_CAST(int16_t, a_.i32[octet_point + i]));
        r_.i16[quarter_point + i + octet_point] = (b_.i32[octet_point + i] > INT16_MAX) ? INT16_MAX : ((b_.i32[octet_point + i] < INT16_MIN) ? INT16_MIN : HEDLEY_STATIC_CAST(int16_t, b_.i32[octet_point + i]));
        r_.i16[halfway_point + i] = (a_.i32[quarter_point + i] > INT16_MAX) ? INT16_MAX : ((a_.i32[quarter_point +i] < INT16_MIN) ? INT16_MIN : HEDLEY_STATIC_CAST(int16_t, a_.i32[quarter_point + i]));
        r_.i16[halfway_point + i + octet_point] = (b_.i32[quarter_point + i] > INT16_MAX) ? INT16_MAX : ((b_.i32[quarter_point + i] < INT16_MIN) ? INT16_MIN : HEDLEY_STATIC_CAST(int16_t, b_.i32[quarter_point +i]));
        r_.i16[halfway_point + quarter_point + i] = (a_.i32[quarter_point + octet_point + i] > INT16_MAX) ? INT16_MAX : ((a_.i32[quarter_point + octet_point + i] < INT16_MIN) ? INT16_MIN : HEDLEY_STATIC_CAST(int16_t, a_.i32[quarter_point + octet_point + i]));
        r_.i16[halfway_point + quarter_point + i + octet_point] = (b_.i32[quarter_point + octet_point + i] > INT16_MAX) ? INT16_MAX : ((b_.i32[quarter_point + octet_point + i] < INT16_MIN) ? INT16_MIN : HEDLEY_STATIC_CAST(int16_t, b_.i32[quarter_point + octet_point + i]));
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_packs_epi32
  #define _mm512_packs_epi32(a, b) easysimd_mm512_packs_epi32(a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_PACKS_H) */
