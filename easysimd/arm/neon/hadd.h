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

/* TODO: the 128-bit versions only require AVX-512 because of the final
 * conversions from larger types down to smaller ones.  We could get
 * the same results from AVX/AVX2 instructions with some shuffling
 * to extract the low half of each input element to the low half
 * of a 256-bit vector, then cast that to a 128-bit vector. */

#if !defined(EASYSIMD_ARM_NEON_HADD_H)
#define EASYSIMD_ARM_NEON_HADD_H

#include "addl.h"
#include "shr_n.h"
#include "movn.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vhadd_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhadd_s8(a, b);
  #else
    return easysimd_vmovn_s16(easysimd_vshrq_n_s16(easysimd_vaddl_s8(a, b), 1));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhadd_s8
  #define vhadd_s8(a, b) easysimd_vhadd_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vhadd_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhadd_s16(a, b);
  #else
    return easysimd_vmovn_s32(easysimd_vshrq_n_s32(easysimd_vaddl_s16(a, b), 1));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhadd_s16
  #define vhadd_s16(a, b) easysimd_vhadd_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vhadd_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhadd_s32(a, b);
  #else
    return easysimd_vmovn_s64(easysimd_vshrq_n_s64(easysimd_vaddl_s32(a, b), 1));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhadd_s32
  #define vhadd_s32(a, b) easysimd_vhadd_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vhadd_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhadd_u8(a, b);
  #else
    return easysimd_vmovn_u16(easysimd_vshrq_n_u16(easysimd_vaddl_u8(a, b), 1));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhadd_u8
  #define vhadd_u8(a, b) easysimd_vhadd_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vhadd_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhadd_u16(a, b);
  #else
    return easysimd_vmovn_u32(easysimd_vshrq_n_u32(easysimd_vaddl_u16(a, b), 1));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhadd_u16
  #define vhadd_u16(a, b) easysimd_vhadd_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vhadd_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhadd_u32(a, b);
  #else
    return easysimd_vmovn_u64(easysimd_vshrq_n_u64(easysimd_vaddl_u32(a, b), 1));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhadd_u32
  #define vhadd_u32(a, b) easysimd_vhadd_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vhaddq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhaddq_s8(a, b);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
      r_.m128i = _mm256_cvtepi16_epi8(_mm256_srai_epi16(_mm256_add_epi16(_mm256_cvtepi8_epi16(a_.m128i), _mm256_cvtepi8_epi16(b_.m128i)), 1));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(int8_t, (HEDLEY_STATIC_CAST(int16_t, a_.values[i]) + HEDLEY_STATIC_CAST(int16_t, b_.values[i])) >> 1);
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhaddq_s8
  #define vhaddq_s8(a, b) easysimd_vhaddq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vhaddq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhaddq_s16(a, b);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm256_cvtepi32_epi16(_mm256_srai_epi32(_mm256_add_epi32(_mm256_cvtepi16_epi32(a_.m128i), _mm256_cvtepi16_epi32(b_.m128i)), 1));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(int16_t, (HEDLEY_STATIC_CAST(int32_t, a_.values[i]) + HEDLEY_STATIC_CAST(int32_t, b_.values[i])) >> 1);
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhaddq_s16
  #define vhaddq_s16(a, b) easysimd_vhaddq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vhaddq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhaddq_s32(a, b);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm256_cvtepi64_epi32(_mm256_srai_epi64(_mm256_add_epi64(_mm256_cvtepi32_epi64(a_.m128i), _mm256_cvtepi32_epi64(b_.m128i)), 1));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(int32_t, (HEDLEY_STATIC_CAST(int64_t, a_.values[i]) + HEDLEY_STATIC_CAST(int64_t, b_.values[i])) >> 1);
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhaddq_s32
  #define vhaddq_s32(a, b) easysimd_vhaddq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vhaddq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhaddq_u8(a, b);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
      r_.m128i = _mm256_cvtepi16_epi8(_mm256_srli_epi16(_mm256_add_epi16(_mm256_cvtepu8_epi16(a_.m128i), _mm256_cvtepu8_epi16(b_.m128i)), 1));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint8_t, (HEDLEY_STATIC_CAST(uint16_t, a_.values[i]) + HEDLEY_STATIC_CAST(uint16_t, b_.values[i])) >> 1);
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhaddq_u8
  #define vhaddq_u8(a, b) easysimd_vhaddq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vhaddq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhaddq_u16(a, b);
  #else
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a),
      b_ = easysimd_uint16x8_to_private(b);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm256_cvtepi32_epi16(_mm256_srli_epi32(_mm256_add_epi32(_mm256_cvtepu16_epi32(a_.m128i), _mm256_cvtepu16_epi32(b_.m128i)), 1));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint16_t, (HEDLEY_STATIC_CAST(uint32_t, a_.values[i]) + HEDLEY_STATIC_CAST(uint32_t, b_.values[i])) >> 1);
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhaddq_u16
  #define vhaddq_u16(a, b) easysimd_vhaddq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vhaddq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vhaddq_u32(a, b);
  #else
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a),
      b_ = easysimd_uint32x4_to_private(b);

    #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r_.m128i = _mm256_cvtepi64_epi32(_mm256_srli_epi64(_mm256_add_epi64(_mm256_cvtepu32_epi64(a_.m128i), _mm256_cvtepu32_epi64(b_.m128i)), 1));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint32_t, (HEDLEY_STATIC_CAST(uint64_t, a_.values[i]) + HEDLEY_STATIC_CAST(uint64_t, b_.values[i])) >> 1);
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vhaddq_u32
  #define vhaddq_u32(a, b) easysimd_vhaddq_u32((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_HADD_H) */
