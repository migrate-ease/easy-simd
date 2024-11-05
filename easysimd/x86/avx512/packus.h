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

#if !defined(EASYSIMD_X86_AVX512_PACKUS_H)
#define EASYSIMD_X86_AVX512_PACKUS_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_packus_epi16 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_packus_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svuzp1_u8(svqxtunb_s16(a.sve_i16), svqxtunb_s16(b.sve_i16)), src.sve_u8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      int16_t v = (i < (sizeof(a_.i16) / sizeof(a_.i16[0]))) ? a_.i16[i] : b_.i16[i & 7];
      r_.u8[i] = ((k >> i) & 1) ? ((v < 0) ? UINT8_C(0) : ((v > UINT8_MAX) ? UINT8_MAX : HEDLEY_STATIC_CAST(uint8_t, v))) : src_.u8[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_packus_epi16(src, k, a, b) easysimd_mm_mask_packus_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_packus_epi16 (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_packus_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svuzp1_u8(svqxtunb_s16(a.sve_i16), svqxtunb_s16(b.sve_i16)), svdup_n_u8(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      int16_t v = (i < (sizeof(a_.i16) / sizeof(a_.i16[0]))) ? a_.i16[i] : b_.i16[i & 7];
      r_.u8[i] = ((k >> i) & 1) ? ((v < 0) ? UINT8_C(0) : ((v > UINT8_MAX) ? UINT8_MAX : HEDLEY_STATIC_CAST(uint8_t, v))) : UINT8_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_packus_epi16(k, a, b) easysimd_mm_maskz_packus_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_packus_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_packus_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svuzp1_u16(svqxtunb_s32(a.sve_i32), svqxtunb_s32(b.sve_i32)), src.sve_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      int32_t v = (i < (sizeof(a_.i32) / sizeof(a_.i32[0]))) ? a_.i32[i] : b_.i32[i & 3];
      r_.u16[i] = ((k >> i) & 1) ? ((v < 0) ? UINT16_C(0) : ((v > UINT16_MAX) ? UINT16_MAX : HEDLEY_STATIC_CAST(uint16_t, v))) : src_.u16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_packus_epi32
  #define _mm_mask_packus_epi32(src, k, a, b) easysimd_mm_mask_packus_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_packus_epi32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_packus_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svuzp1_u16(svqxtunb_s32(a.sve_i32), svqxtunb_s32(b.sve_i32)), svdup_n_u16(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      int32_t v = (i < (sizeof(a_.i32) / sizeof(a_.i32[0]))) ? a_.i32[i] : b_.i32[i & 3];
      r_.u16[i] = ((k >> i) & 1) ? ((v < 0) ? UINT16_C(0) : ((v > UINT16_MAX) ? UINT16_MAX : HEDLEY_STATIC_CAST(uint16_t, v))) : UINT16_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_packus_epi32
  #define _mm_maskz_packus_epi32(k, a, b) easysimd_mm_maskz_packus_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_packus_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_packus_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svuzp1_u8(svqxtunb_s16(a.sve_i16[EASYSIMD_SV_INDEX_0]), svqxtunb_s16(b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svuzp1_u8(svqxtunb_s16(a.sve_i16[EASYSIMD_SV_INDEX_1]), svqxtunb_s16(b.sve_i16[EASYSIMD_SV_INDEX_1]));
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svuzp1_u8(svqxtunb_s16(a.sve_i16[EASYSIMD_SV_INDEX_2]), svqxtunb_s16(b.sve_i16[EASYSIMD_SV_INDEX_2]));
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svuzp1_u8(svqxtunb_s16(a.sve_i16[EASYSIMD_SV_INDEX_3]), svqxtunb_s16(b.sve_i16[EASYSIMD_SV_INDEX_3]));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      r_.m256i[0] = easysimd_mm256_packus_epi16(a_.m256i[0], b_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_packus_epi16(a_.m256i[1], b_.m256i[1]);
    #else
      const size_t halfway_point = (sizeof(r_.i8) / sizeof(r_.i8[0])) / 2;
      const size_t quarter_point = (sizeof(r_.i8) / sizeof(r_.i8[0])) / 4;
      const size_t octet_point = (sizeof(r_.i8) / sizeof(r_.i8[0])) / 8;

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < octet_point ; i++) {
        r_.u8[i]     = (a_.i16[i] > UINT8_MAX) ? UINT8_MAX : ((a_.i16[i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, a_.i16[i]));
        r_.u8[i + octet_point] = (b_.i16[i] > UINT8_MAX) ? UINT8_MAX : ((b_.i16[i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, b_.i16[i]));
        r_.u8[quarter_point + i]     = (a_.i16[octet_point + i] > UINT8_MAX) ? UINT8_MAX : ((a_.i16[octet_point + i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, a_.i16[octet_point + i]));
        r_.u8[quarter_point + i + octet_point] = (b_.i16[octet_point + i] > UINT8_MAX) ? UINT8_MAX : ((b_.i16[octet_point + i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, b_.i16[octet_point + i]));
        r_.u8[halfway_point + i]     = (a_.i16[quarter_point + i] > UINT8_MAX) ? UINT8_MAX : ((a_.i16[quarter_point + i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, a_.i16[quarter_point + i]));
        r_.u8[halfway_point + i + octet_point] = (b_.i16[quarter_point + i] > UINT8_MAX) ? UINT8_MAX : ((b_.i16[quarter_point + i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, b_.i16[quarter_point + i]));
        r_.u8[halfway_point + quarter_point + i]     = (a_.i16[quarter_point + octet_point + i] > UINT8_MAX) ? UINT8_MAX : ((a_.i16[quarter_point + octet_point + i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, a_.i16[quarter_point + octet_point + i]));
        r_.u8[halfway_point + quarter_point + i + octet_point] = (b_.i16[quarter_point + octet_point + i] > UINT8_MAX) ? UINT8_MAX : ((b_.i16[quarter_point + octet_point + i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, b_.i16[quarter_point + octet_point + i]));
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_packus_epi16
  #define _mm512_packus_epi16(a, b) easysimd_mm512_packus_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_packus_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_packus_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svuzp1_u16(svqxtunb_s32(a.sve_i32[EASYSIMD_SV_INDEX_0]), svqxtunb_s32(b.sve_i32[EASYSIMD_SV_INDEX_0]));
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svuzp1_u16(svqxtunb_s32(a.sve_i32[EASYSIMD_SV_INDEX_1]), svqxtunb_s32(b.sve_i32[EASYSIMD_SV_INDEX_1]));
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svuzp1_u16(svqxtunb_s32(a.sve_i32[EASYSIMD_SV_INDEX_2]), svqxtunb_s32(b.sve_i32[EASYSIMD_SV_INDEX_2]));
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svuzp1_u16(svqxtunb_s32(a.sve_i32[EASYSIMD_SV_INDEX_3]), svqxtunb_s32(b.sve_i32[EASYSIMD_SV_INDEX_3]));
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_packus_epi32(a_.m256i[i], b_.m256i[i]);
      }
    #else
      const size_t halfway_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 2;
      const size_t quarter_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 4;
      const size_t octet_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 8;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < octet_point ; i++) {
        r_.u16[i] = (a_.i32[i] > UINT16_MAX) ? UINT16_MAX : ((a_.i32[i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, a_.i32[i]));
        r_.u16[i + octet_point] = (b_.i32[i] > UINT16_MAX) ? UINT16_MAX : ((b_.i32[i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, b_.i32[i]));
        r_.u16[quarter_point + i]     = (a_.i32[octet_point + i] > UINT16_MAX) ? UINT16_MAX : ((a_.i32[octet_point + i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, a_.i32[octet_point + i]));
        r_.u16[quarter_point + i + octet_point] = (b_.i32[octet_point + i] > UINT16_MAX) ? UINT16_MAX : ((b_.i32[octet_point + i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, b_.i32[octet_point + i]));
        r_.u16[halfway_point + i] = (a_.i32[quarter_point + i] > UINT16_MAX) ? UINT16_MAX : ((a_.i32[quarter_point +i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, a_.i32[quarter_point + i]));
        r_.u16[halfway_point + i + octet_point] = (b_.i32[quarter_point + i] > UINT16_MAX) ? UINT16_MAX : ((b_.i32[quarter_point + i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, b_.i32[quarter_point +i]));
        r_.u16[halfway_point + quarter_point + i]     = (a_.i32[quarter_point + octet_point + i] > UINT16_MAX) ? UINT16_MAX : ((a_.i32[quarter_point + octet_point + i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, a_.i32[quarter_point + octet_point + i]));
        r_.u16[halfway_point + quarter_point + i + octet_point] = (b_.i32[quarter_point + octet_point + i] > UINT16_MAX) ? UINT16_MAX : ((b_.i32[quarter_point + octet_point + i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, b_.i32[quarter_point + octet_point + i]));
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_packus_epi32
  #define _mm512_packus_epi32(a, b) easysimd_mm512_packus_epi32(a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_PACKUS_H) */
