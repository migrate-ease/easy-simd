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

#if !defined(EASYSIMD_ARM_NEON_RSHL_H)
#define EASYSIMD_ARM_NEON_RSHL_H

#include "types.h"

/* Notes from the implementer (Christopher Moore aka rosbif)
 *
 * I have tried to exactly reproduce the documented behaviour of the
 * ARM NEON rshl and rshlq intrinsics.
 * This is complicated for the following reasons:-
 *
 * a) Negative shift counts shift right.
 *
 * b) Only the low byte of the shift count is used but the shift count
 * is not limited to 8-bit values (-128 to 127).
 *
 * c) Overflow must be avoided when rounding, together with sign change
 * warning/errors in the C versions.
 *
 * d) Intel SIMD is not nearly as complete as NEON and AltiVec.
 * There were no intrisics with a vector shift count before AVX2 which
 * only has 32 and 64-bit logical ones and only a 32-bit arithmetic
 * one. The others need AVX512. There are no 8-bit shift intrinsics at
 * all, even with a scalar shift count. It is surprising to use AVX2
 * and even AVX512 to implement a 64-bit vector operation.
 *
 * e) Many shift implementations, and the C standard, do not treat a
 * shift count >= the object's size in bits as one would expect.
 * (Personally I feel that > is silly but == can be useful.)
 *
 * Note that even the C17/18 standard does not define the behaviour of
 * a right shift of a negative value.
 * However Evan and I agree that all compilers likely to be used
 * implement this as an arithmetic right shift with sign extension.
 * If this is not the case it could be replaced by a logical right shift
 * if negative values are complemented before and after the shift.
 *
 * Some of the SIMD translations may be slower than the portable code,
 * particularly those for vectors with only one or two elements.
 * But I had fun writing them ;-)
 *
 */

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vrshld_s64(int64_t a, int64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrshld_s64(a, b);
  #else
    b = HEDLEY_STATIC_CAST(int8_t, b);
    return
      (easysimd_math_llabs(b) >= 64)
        ? 0
        : (b >= 0)
          ? (a << b)
          : ((a + (INT64_C(1) << (-b - 1))) >> -b);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrshld_s64
  #define vrshld_s64(a, b) easysimd_vrshld_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vrshld_u64(uint64_t a, int64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrshld_u64(a, HEDLEY_STATIC_CAST(uint64_t, b));
  #else
    b = HEDLEY_STATIC_CAST(int8_t, b);
    return
      (b >=  64) ? 0 :
      (b >=   0) ? (a << b) :
      (b >= -64) ? (((b == -64) ? 0 : (a >> -b)) + ((a >> (-b - 1)) & 1)) : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrshld_u64
  #define vrshld_u64(a, b) easysimd_vrshld_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vrshl_s8 (const easysimd_int8x8_t a, const easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshl_s8(a, b);
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi16(zero, zero);
      __m128i a128 = _mm_cvtepi8_epi16(_mm_movpi64_epi64(a_.m64));
      __m128i b128 = _mm_cvtepi8_epi16(_mm_movpi64_epi64(b_.m64));
      __m128i a128_shr = _mm_srav_epi16(a128, _mm_xor_si128(b128, ff));
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi16(a128, b128),
                                    _mm_srai_epi16(_mm_sub_epi16(a128_shr, ff), 1),
                                    _mm_cmpgt_epi16(zero, b128));
      r_.m64 = _mm_movepi64_pi64(_mm_cvtepi16_epi8(r128));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m256i zero = _mm256_setzero_si256();
      const __m256i ff   = _mm256_cmpeq_epi32(zero, zero);
      __m256i a256 = _mm256_cvtepi8_epi32(_mm_movpi64_epi64(a_.m64));
      __m256i b256 = _mm256_cvtepi8_epi32(_mm_movpi64_epi64(b_.m64));
      __m256i a256_shr = _mm256_srav_epi32(a256, _mm256_xor_si256(b256, ff));
      __m256i r256 = _mm256_blendv_epi8(_mm256_sllv_epi32(a256, b256),
                                        _mm256_srai_epi32(_mm256_sub_epi32(a256_shr, ff), 1),
                                        _mm256_cmpgt_epi32(zero, b256));
      r256 = _mm256_shuffle_epi8(r256, _mm256_set1_epi32(0x0C080400));
      r_.m64 = _mm_set_pi32(_mm256_extract_epi32(r256, 4), _mm256_extract_epi32(r256, 0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(int8_t,
                                          (easysimd_math_abs(b_.values[i]) >= 8) ? 0 :
                                          (b_.values[i] >= 0) ? (a_.values[i] << b_.values[i]) :
                                          ((a_.values[i] + (1 << (-b_.values[i] - 1))) >> -b_.values[i]));
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshl_s8
  #define vrshl_s8(a, b) easysimd_vrshl_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vrshl_s16 (const easysimd_int16x4_t a, const easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshl_s16(a, b);
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);

    #if defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi32(zero, zero);
      __m128i a128 = _mm_cvtepi16_epi32(_mm_movpi64_epi64(a_.m64));
      __m128i b128 = _mm_cvtepi16_epi32(_mm_movpi64_epi64(b_.m64));
      b128 = _mm_srai_epi32(_mm_slli_epi32(b128, 24), 24);
      __m128i a128_shr = _mm_srav_epi32(a128, _mm_xor_si128(b128, ff));
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi32(a128, b128),
                                    _mm_srai_epi32(_mm_sub_epi32(a128_shr, ff), 1),
                                    _mm_cmpgt_epi32(zero, b128));
      r_.m64 = _mm_movepi64_pi64(_mm_shuffle_epi8(r128, _mm_set1_epi64x(0x0D0C090805040100)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        b_.values[i] = HEDLEY_STATIC_CAST(int8_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(int16_t,
                                          (easysimd_math_abs(b_.values[i]) >= 16) ? 0 :
                                          (b_.values[i] >= 0) ? (a_.values[i] << b_.values[i]) :
                                          ((a_.values[i] + (1 << (-b_.values[i] - 1))) >> -b_.values[i]));
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshl_s16
  #define vrshl_s16(a, b) easysimd_vrshl_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vrshl_s32 (const easysimd_int32x2_t a, const easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshl_s32(a, b);
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b);

    #if defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi32(zero, zero);
      __m128i a128 = _mm_movpi64_epi64(a_.m64);
      __m128i b128 = _mm_movpi64_epi64(b_.m64);
      b128 = _mm_srai_epi32(_mm_slli_epi32(b128, 24), 24);
      __m128i a128_shr = _mm_srav_epi32(a128, _mm_xor_si128(b128, ff));
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi32(a128, b128),
                                    _mm_srai_epi32(_mm_sub_epi32(a128_shr, ff), 1),
                                    _mm_cmpgt_epi32(zero, b128));
      r_.m64 = _mm_movepi64_pi64(r128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        b_.values[i] = HEDLEY_STATIC_CAST(int8_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(int32_t,
                                          (easysimd_math_abs(b_.values[i]) >= 32) ? 0 :
                                          (b_.values[i] >= 0) ? (a_.values[i] << b_.values[i]) :
                                          ((a_.values[i] + (1 << (-b_.values[i] - 1))) >> -b_.values[i]));
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshl_s32
  #define vrshl_s32(a, b) easysimd_vrshl_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vrshl_s64 (const easysimd_int64x1_t a, const easysimd_int64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshl_s64(a, b);
  #else
    easysimd_int64x1_private
      r_,
      a_ = easysimd_int64x1_to_private(a),
      b_ = easysimd_int64x1_to_private(b);

    #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi64(zero, zero);
      __m128i a128 = _mm_movpi64_epi64(a_.m64);
      __m128i b128 = _mm_movpi64_epi64(b_.m64);
      b128 = _mm_srai_epi64(_mm_slli_epi64(b128, 56), 56);
      __m128i a128_shr = _mm_srav_epi64(a128, _mm_xor_si128(b128, ff));
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi64(a128, b128),
                                    _mm_srai_epi64(_mm_sub_epi64(a128_shr, ff), 1),
                                    _mm_cmpgt_epi64(zero, b128));
      r_.m64 = _mm_movepi64_pi64(r128);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ones = _mm_set1_epi64x(1);
      __m128i a128 = _mm_movpi64_epi64(a_.m64);
      __m128i b128 = _mm_movpi64_epi64(b_.m64);
      __m128i maska = _mm_cmpgt_epi64(zero, a128);
      __m128i b128_abs = _mm_and_si128(_mm_abs_epi8(b128), _mm_set1_epi64x(0xFF));
      __m128i a128_rnd = _mm_and_si128(_mm_srlv_epi64(a128, _mm_sub_epi64(b128_abs, ones)), ones);
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi64(a128, b128_abs),
                                    _mm_add_epi64(_mm_xor_si128(_mm_srlv_epi64(_mm_xor_si128(a128, maska), b128_abs), maska), a128_rnd),
                                    _mm_cmpgt_epi64(zero, _mm_slli_epi64(b128, 56)));
      r_.m64 = _mm_movepi64_pi64(r128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vrshld_s64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshl_s64
  #define vrshl_s64(a, b) easysimd_vrshl_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vrshl_u8 (const easysimd_uint8x8_t a, const easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshl_u8(a, b);
  #else
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a);
    easysimd_int8x8_private b_ = easysimd_int8x8_to_private(b);

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi16(zero, zero);
      __m128i a128 = _mm_cvtepu8_epi16(_mm_movpi64_epi64(a_.m64));
      __m128i b128 = _mm_cvtepi8_epi16(_mm_movpi64_epi64(b_.m64));
      __m128i a128_shr = _mm_srlv_epi16(a128, _mm_xor_si128(b128, ff));
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi16(a128, b128),
                                    _mm_srli_epi16(_mm_sub_epi16(a128_shr, ff), 1),
                                    _mm_cmpgt_epi16(zero, b128));
      r_.m64 = _mm_movepi64_pi64(_mm_cvtepi16_epi8(r128));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m256i zero = _mm256_setzero_si256();
      const __m256i ff   = _mm256_cmpeq_epi32(zero, zero);
      __m256i a256 = _mm256_cvtepu8_epi32(_mm_movpi64_epi64(a_.m64));
      __m256i b256 = _mm256_cvtepi8_epi32(_mm_movpi64_epi64(b_.m64));
      __m256i a256_shr = _mm256_srlv_epi32(a256, _mm256_xor_si256(b256, ff));
      __m256i r256 = _mm256_blendv_epi8(_mm256_sllv_epi32(a256, b256),
                                        _mm256_srli_epi32(_mm256_sub_epi32(a256_shr, ff), 1),
                                        _mm256_cmpgt_epi32(zero, b256));
      r256 = _mm256_shuffle_epi8(r256, _mm256_set1_epi32(0x0C080400));
      r_.m64 = _mm_set_pi32(_mm256_extract_epi32(r256, 4), _mm256_extract_epi32(r256, 0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint8_t,
                                          (b_.values[i] >=  8) ? 0 :
                                          (b_.values[i] >=  0) ? (a_.values[i] << b_.values[i]) :
                                          (b_.values[i] >= -8) ? (((b_.values[i] == -8) ? 0 : (a_.values[i] >> -b_.values[i])) + ((a_.values[i] >> (-b_.values[i] - 1)) & 1)) :
                                          0);
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshl_u8
  #define vrshl_u8(a, b) easysimd_vrshl_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vrshl_u16 (const easysimd_uint16x4_t a, const easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshl_u16(a, b);
  #else
    easysimd_uint16x4_private
      r_,
      a_ = easysimd_uint16x4_to_private(a);
    easysimd_int16x4_private b_ = easysimd_int16x4_to_private(b);

    #if defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi32(zero, zero);
      __m128i a128 = _mm_cvtepu16_epi32(_mm_movpi64_epi64(a_.m64));
      __m128i b128 = _mm_cvtepi16_epi32(_mm_movpi64_epi64(b_.m64));
      b128 = _mm_srai_epi32(_mm_slli_epi32(b128, 24), 24);
      __m128i a128_shr = _mm_srlv_epi32(a128, _mm_xor_si128(b128, ff));
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi32(a128, b128),
                                    _mm_srli_epi32(_mm_sub_epi32(a128_shr, ff), 1),
                                    _mm_cmpgt_epi32(zero, b128));
      r_.m64 = _mm_movepi64_pi64(_mm_shuffle_epi8(r128, _mm_set1_epi64x(0x0D0C090805040100)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        b_.values[i] = HEDLEY_STATIC_CAST(int8_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(uint16_t,
                                          (b_.values[i] >=  16) ? 0 :
                                          (b_.values[i] >=   0) ? (a_.values[i] << b_.values[i]) :
                                          (b_.values[i] >= -16) ? (((b_.values[i] == -16) ? 0 : (a_.values[i] >> -b_.values[i])) + ((a_.values[i] >> (-b_.values[i] - 1)) & 1)) :
                                          0);
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshl_u16
  #define vrshl_u16(a, b) easysimd_vrshl_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vrshl_u32 (const easysimd_uint32x2_t a, const easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshl_u32(a, b);
  #else
    easysimd_uint32x2_private
      r_,
      a_ = easysimd_uint32x2_to_private(a);
    easysimd_int32x2_private b_ = easysimd_int32x2_to_private(b);

    #if defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi32(zero, zero);
      __m128i a128 = _mm_movpi64_epi64(a_.m64);
      __m128i b128 = _mm_movpi64_epi64(b_.m64);
      b128 = _mm_srai_epi32(_mm_slli_epi32(b128, 24), 24);
      __m128i a128_shr = _mm_srlv_epi32(a128, _mm_xor_si128(b128, ff));
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi32(a128, b128),
                                    _mm_srli_epi32(_mm_sub_epi32(a128_shr, ff), 1),
                                    _mm_cmpgt_epi32(zero, b128));
      r_.m64 = _mm_movepi64_pi64(r128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        b_.values[i] = HEDLEY_STATIC_CAST(int8_t, b_.values[i]);
        r_.values[i] =
          (b_.values[i] >=  32) ? 0 :
          (b_.values[i] >=   0) ? (a_.values[i] << b_.values[i]) :
          (b_.values[i] >= -32) ? (((b_.values[i] == -32) ? 0 : (a_.values[i] >> -b_.values[i])) + ((a_.values[i] >> (-b_.values[i] - 1)) & 1)) :
          0;
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshl_u32
  #define vrshl_u32(a, b) easysimd_vrshl_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vrshl_u64 (const easysimd_uint64x1_t a, const easysimd_int64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshl_u64(a, b);
  #else
    easysimd_uint64x1_private
      r_,
      a_ = easysimd_uint64x1_to_private(a);
    easysimd_int64x1_private b_ = easysimd_int64x1_to_private(b);

    #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi64(zero, zero);
      __m128i a128 = _mm_movpi64_epi64(a_.m64);
      __m128i b128 = _mm_movpi64_epi64(b_.m64);
      b128 = _mm_srai_epi64(_mm_slli_epi64(b128, 56), 56);
      __m128i a128_shr = _mm_srlv_epi64(a128, _mm_xor_si128(b128, ff));
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi64(a128, b128),
                                    _mm_srli_epi64(_mm_sub_epi64(a128_shr, ff), 1),
                                    _mm_cmpgt_epi64(zero, b128));
      r_.m64 = _mm_movepi64_pi64(r128);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      const __m128i ones = _mm_set1_epi64x(1);
      const __m128i a128 = _mm_movpi64_epi64(a_.m64);
      __m128i b128 = _mm_movpi64_epi64(b_.m64);
      __m128i b128_abs = _mm_and_si128(_mm_abs_epi8(b128), _mm_set1_epi64x(0xFF));
      __m128i a128_shr = _mm_srlv_epi64(a128, _mm_sub_epi64(b128_abs, ones));
      __m128i r128 = _mm_blendv_epi8(_mm_sllv_epi64(a128, b128_abs),
                                    _mm_srli_epi64(_mm_add_epi64(a128_shr, ones), 1),
                                    _mm_cmpgt_epi64(_mm_setzero_si128(), _mm_slli_epi64(b128, 56)));
      r_.m64 = _mm_movepi64_pi64(r128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vrshld_u64(a_.values[i], b_.values[i]);
      }
    #endif

  return easysimd_uint64x1_from_private(r_);
#endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshl_u64
  #define vrshl_u64(a, b) easysimd_vrshl_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vrshlq_s8 (const easysimd_int8x16_t a, const easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshlq_s8(a, b);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      const __m256i zero = _mm256_setzero_si256();
      const __m256i ff   = _mm256_cmpeq_epi16(zero, zero);
      __m256i a256 = _mm256_cvtepi8_epi16(a_.m128i);
      __m256i b256 = _mm256_cvtepi8_epi16(b_.m128i);
      __m256i a256_shr = _mm256_srav_epi16(a256, _mm256_xor_si256(b256, ff));
      __m256i r256 = _mm256_blendv_epi8(_mm256_sllv_epi16(a256, b256),
                                        _mm256_srai_epi16(_mm256_sub_epi16(a256_shr, ff), 1),
                                        _mm256_cmpgt_epi16(zero, b256));
      r_.m128i = _mm256_cvtepi16_epi8(r256);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(int8_t,
                                          (easysimd_math_abs(b_.values[i]) >= 8) ? 0 :
                                          (b_.values[i] >= 0) ? (a_.values[i] << b_.values[i]) :
                                          ((a_.values[i] + (1 << (-b_.values[i] - 1))) >> -b_.values[i]));
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshlq_s8
  #define vrshlq_s8(a, b) easysimd_vrshlq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vrshlq_s16 (const easysimd_int16x8_t a, const easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshlq_s16(a, b);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi16(zero, zero);
      __m128i B = _mm_srai_epi16(_mm_slli_epi16(b_.m128i, 8), 8);
      __m128i a_shr = _mm_srav_epi16(a_.m128i, _mm_xor_si128(B, ff));
      r_.m128i = _mm_blendv_epi8(_mm_sllv_epi16(a_.m128i, B),
                            _mm_srai_epi16(_mm_sub_epi16(a_shr, ff), 1),
                            _mm_cmpgt_epi16(zero, B));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
      const __m256i zero = _mm256_setzero_si256();
      const __m256i ff   = _mm256_cmpeq_epi32(zero, zero);
      __m256i a256 = _mm256_cvtepi16_epi32(a_.m128i);
      __m256i b256 = _mm256_cvtepi16_epi32(b_.m128i);
      b256 = _mm256_srai_epi32(_mm256_slli_epi32(b256, 24), 24);
      __m256i a256_shr = _mm256_srav_epi32(a256, _mm256_xor_si256(b256, ff));
      __m256i r256 = _mm256_blendv_epi8(_mm256_sllv_epi32(a256, b256),
                                        _mm256_srai_epi32(_mm256_sub_epi32(a256_shr, ff), 1),
                                        _mm256_cmpgt_epi32(zero, b256));
      r256 = _mm256_shuffle_epi8(r256, _mm256_set1_epi64x(0x0D0C090805040100));
      r_.m128i = _mm_set_epi64x(_mm256_extract_epi64(r256, 2), _mm256_extract_epi64(r256, 0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        b_.values[i] = HEDLEY_STATIC_CAST(int8_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(int16_t,
                                          (easysimd_math_abs(b_.values[i]) >= 16) ? 0 :
                                          (b_.values[i] >= 0) ? (a_.values[i] << b_.values[i]) :
                                          ((a_.values[i] + (1 << (-b_.values[i] - 1))) >> -b_.values[i]));
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshlq_s16
  #define vrshlq_s16(a, b) easysimd_vrshlq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vrshlq_s32 (const easysimd_int32x4_t a, const easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshlq_s32(a, b);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi32(zero, zero);
      __m128i B = _mm_srai_epi32(_mm_slli_epi32(b_.m128i, 24), 24);
      __m128i a_shr = _mm_srav_epi32(a_.m128i, _mm_xor_si128(B, ff));
      r_.m128i = _mm_blendv_epi8(_mm_sllv_epi32(a_.m128i, B),
                            _mm_srai_epi32(_mm_sub_epi32(a_shr, ff), 1),
                            _mm_cmpgt_epi32(zero, B));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        b_.values[i] = HEDLEY_STATIC_CAST(int8_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(int32_t,
                                          (easysimd_math_abs(b_.values[i]) >= 32) ? 0 :
                                          (b_.values[i] >=  0) ? (a_.values[i] << b_.values[i]) :
                                          ((a_.values[i] + (1 << (-b_.values[i] - 1))) >> -b_.values[i]));
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshlq_s32
  #define vrshlq_s32(a, b) easysimd_vrshlq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vrshlq_s64 (const easysimd_int64x2_t a, const easysimd_int64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshlq_s64(a, b);
  #else
    easysimd_int64x2_private
      r_,
      a_ = easysimd_int64x2_to_private(a),
      b_ = easysimd_int64x2_to_private(b);

    #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi32(zero, zero);
      __m128i B = _mm_srai_epi64(_mm_slli_epi64(b_.m128i, 56), 56);
      __m128i a_shr = _mm_srav_epi64(a_.m128i, _mm_xor_si128(B, ff));
      r_.m128i = _mm_blendv_epi8(_mm_sllv_epi64(a_.m128i, B),
                            _mm_srai_epi64(_mm_sub_epi64(a_shr, ff), 1),
                            _mm_cmpgt_epi64(zero, B));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ones = _mm_set1_epi64x(1);
      __m128i maska = _mm_cmpgt_epi64(zero, a_.m128i);
      __m128i b_abs = _mm_and_si128(_mm_abs_epi8(b_.m128i), _mm_set1_epi64x(0xFF));
      __m128i a_rnd = _mm_and_si128(_mm_srlv_epi64(a_.m128i, _mm_sub_epi64(b_abs, ones)), ones);
      r_.m128i = _mm_blendv_epi8(_mm_sllv_epi64(a_.m128i, b_abs),
                            _mm_add_epi64(_mm_xor_si128(_mm_srlv_epi64(_mm_xor_si128(a_.m128i, maska), b_abs), maska), a_rnd),
                            _mm_cmpgt_epi64(zero, _mm_slli_epi64(b_.m128i, 56)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vrshld_s64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshlq_s64
  #define vrshlq_s64(a, b) easysimd_vrshlq_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vrshlq_u8 (const easysimd_uint8x16_t a, const easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshlq_u8(a, b);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a);
    easysimd_int8x16_private b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      const __m256i zero = _mm256_setzero_si256();
      const __m256i ff   = _mm256_cmpeq_epi32(zero, zero);
      __m256i a256 = _mm256_cvtepu8_epi16(a_.m128i);
      __m256i b256 = _mm256_cvtepi8_epi16(b_.m128i);
      __m256i a256_shr = _mm256_srlv_epi16(a256, _mm256_xor_si256(b256, ff));
      __m256i r256 = _mm256_blendv_epi8(_mm256_sllv_epi16(a256, b256),
                                        _mm256_srli_epi16(_mm256_sub_epi16(a256_shr, ff), 1),
                                        _mm256_cmpgt_epi16(zero, b256));
      r_.m128i = _mm256_cvtepi16_epi8(r256);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint8_t,
                                          (b_.values[i] >=  8) ? 0 :
                                          (b_.values[i] >=  0) ? (a_.values[i] << b_.values[i]) :
                                          (b_.values[i] >= -8) ? (((b_.values[i] == -8) ? 0 : (a_.values[i] >> -b_.values[i])) + ((a_.values[i] >> (-b_.values[i] - 1)) & 1)) :
                                          0);
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshlq_u8
  #define vrshlq_u8(a, b) easysimd_vrshlq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vrshlq_u16 (const easysimd_uint16x8_t a, const easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshlq_u16(a, b);
  #else
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a);
    easysimd_int16x8_private b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi16(zero, zero);
      __m128i B = _mm_srai_epi16(_mm_slli_epi16(b_.m128i, 8), 8);
      __m128i a_shr = _mm_srlv_epi16(a_.m128i, _mm_xor_si128(B, ff));
      r_.m128i = _mm_blendv_epi8(_mm_sllv_epi16(a_.m128i, B),
                            _mm_srli_epi16(_mm_sub_epi16(a_shr, ff), 1),
                            _mm_cmpgt_epi16(zero, B));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
      const __m256i zero = _mm256_setzero_si256();
      const __m256i ff   = _mm256_cmpeq_epi32(zero, zero);
      __m256i a256 = _mm256_cvtepu16_epi32(a_.m128i);
      __m256i b256 = _mm256_cvtepi16_epi32(b_.m128i);
      b256 = _mm256_srai_epi32(_mm256_slli_epi32(b256, 24), 24);
      __m256i a256_shr = _mm256_srlv_epi32(a256, _mm256_xor_si256(b256, ff));
      __m256i r256 = _mm256_blendv_epi8(_mm256_sllv_epi32(a256, b256),
                                        _mm256_srli_epi32(_mm256_sub_epi32(a256_shr, ff), 1),
                                        _mm256_cmpgt_epi32(zero, b256));
      r256 = _mm256_shuffle_epi8(r256, _mm256_set1_epi64x(0x0D0C090805040100));
      r_.m128i = _mm_set_epi64x(_mm256_extract_epi64(r256, 2), _mm256_extract_epi64(r256, 0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        b_.values[i] = HEDLEY_STATIC_CAST(int8_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(uint16_t,
                                          (b_.values[i] >=  16) ? 0 :
                                          (b_.values[i] >=   0) ? (a_.values[i] << b_.values[i]) :
                                          (b_.values[i] >= -16) ? (((b_.values[i] == -16) ? 0 : (a_.values[i] >> -b_.values[i])) + ((a_.values[i] >> (-b_.values[i] - 1)) & 1)) :
                                          0);
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshlq_u16
  #define vrshlq_u16(a, b) easysimd_vrshlq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vrshlq_u32 (const easysimd_uint32x4_t a, const easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshlq_u32(a, b);
  #else
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a);
    easysimd_int32x4_private b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi32(zero, zero);
      __m128i B = _mm_srai_epi32(_mm_slli_epi32(b_.m128i, 24), 24);
      __m128i a_shr = _mm_srlv_epi32(a_.m128i, _mm_xor_si128(B, ff));
      r_.m128i = _mm_blendv_epi8(_mm_sllv_epi32(a_.m128i, B),
                            _mm_srli_epi32(_mm_sub_epi32(a_shr, ff), 1),
                            _mm_cmpgt_epi32(zero, B));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        b_.values[i] = HEDLEY_STATIC_CAST(int8_t, b_.values[i]);
        r_.values[i] =
          (b_.values[i] >=  32) ? 0 :
          (b_.values[i] >=   0) ? (a_.values[i] << b_.values[i]) :
          (b_.values[i] >= -32) ? (((b_.values[i] == -32) ? 0 : (a_.values[i] >> -b_.values[i])) + ((a_.values[i] >> (-b_.values[i] - 1)) & 1)) :
          0;
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshlq_u32
  #define vrshlq_u32(a, b) easysimd_vrshlq_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vrshlq_u64 (const easysimd_uint64x2_t a, const easysimd_int64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrshlq_u64(a, b);
  #else
    easysimd_uint64x2_private
      r_,
      a_ = easysimd_uint64x2_to_private(a);
    easysimd_int64x2_private b_ = easysimd_int64x2_to_private(b);

    #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      const __m128i zero = _mm_setzero_si128();
      const __m128i ff   = _mm_cmpeq_epi64(zero, zero);
      __m128i B = _mm_srai_epi64(_mm_slli_epi64(b_.m128i, 56), 56);
      __m128i a_shr = _mm_srlv_epi64(a_.m128i, _mm_xor_si128(B, ff));
      r_.m128i = _mm_blendv_epi8(_mm_sllv_epi64(a_.m128i, B),
                            _mm_srli_epi64(_mm_sub_epi64(a_shr, ff), 1),
                            _mm_cmpgt_epi64(zero, B));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      const __m128i ones = _mm_set1_epi64x(1);
      __m128i b_abs = _mm_and_si128(_mm_abs_epi8(b_.m128i), _mm_set1_epi64x(0xFF));
      __m128i a_shr = _mm_srlv_epi64(a_.m128i, _mm_sub_epi64(b_abs, ones));
      r_.m128i = _mm_blendv_epi8(_mm_sllv_epi64(a_.m128i, b_abs),
                            _mm_srli_epi64(_mm_add_epi64(a_shr, ones), 1),
                            _mm_cmpgt_epi64(_mm_setzero_si128(), _mm_slli_epi64(b_.m128i, 56)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vrshld_u64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshlq_u64
  #define vrshlq_u64(a, b) easysimd_vrshlq_u64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_RSHL_H) */
