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
 *   2021      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_ARM_SVE_QADD_H)
#define EASYSIMD_ARM_SVE_QADD_H

#include "types.h"
#include "sel.h"
#include "dup.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svqadd_s8(easysimd_svint8_t op1, easysimd_svint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_s8(op1, op2);
  #else
    easysimd_svint8_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vqaddq_s8(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_adds_epi8(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_adds_epi8(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_adds_epi8(op1.m128i[i], op2.m128i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = easysimd_math_adds_i8(op1.values[i], op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_s8
  #define svqadd_s8(op1, op2) easysimd_svqadd_s8(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svqadd_n_s8(easysimd_svint8_t op1, int8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_n_s8(op1, op2);
  #else
    return easysimd_svqadd_s8(op1, easysimd_svdup_n_s8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_n_s8
  #define svqadd_n_s8(op1, op2) easysimd_svqadd_n_s8(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svqadd_s16(easysimd_svint16_t op1, easysimd_svint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_s16(op1, op2);
  #else
    easysimd_svint16_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vqaddq_s16(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_adds_epi16(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_adds_epi16(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_adds_epi16(op1.m128i[i], op2.m128i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = easysimd_math_adds_i16(op1.values[i], op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_s16
  #define svqadd_s16(op1, op2) easysimd_svqadd_s16(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svqadd_n_s16(easysimd_svint16_t op1, int16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_n_s16(op1, op2);
  #else
    return easysimd_svqadd_s16(op1, easysimd_svdup_n_s16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_n_s16
  #define svqadd_n_s16(op1, op2) easysimd_svqadd_n_s16(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svqadd_s32(easysimd_svint32_t op1, easysimd_svint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_s32(op1, op2);
  #else
    easysimd_svint32_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vqaddq_s32(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512VL_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm512_cvtsepi64_epi32(_mm512_add_epi64(_mm512_cvtepi32_epi64(op1.m256i[i]), _mm512_cvtepi32_epi64(op2.m256i[i])));
      }
    #elif defined(EASYSIMD_X86_AVX512VL_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm256_cvtsepi64_epi32(_mm256_add_epi64(_mm256_cvtepi32_epi64(op1.m128i[i]), _mm256_cvtepi32_epi64(op2.m128i[i])));
      }
    #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
      r.altivec =
        vec_packs(
          vec_unpackh(op1.altivec) + vec_unpackh(op2.altivec),
          vec_unpackl(op1.altivec) + vec_unpackl(op2.altivec)
        );
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = easysimd_math_adds_i32(op1.values[i], op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_s32
  #define svqadd_s32(op1, op2) easysimd_svqadd_s32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svqadd_n_s32(easysimd_svint32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_n_s32(op1, op2);
  #else
    return easysimd_svqadd_s32(op1, easysimd_svdup_n_s32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_n_s32
  #define svqadd_n_s32(op1, op2) easysimd_svqadd_n_s32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svqadd_s64(easysimd_svint64_t op1, easysimd_svint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_s64(op1, op2);
  #else
    easysimd_svint64_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vqaddq_s64(op1.neon, op2.neon);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = easysimd_math_adds_i64(op1.values[i], op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_s64
  #define svqadd_s64(op1, op2) easysimd_svqadd_s64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svqadd_n_s64(easysimd_svint64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_n_s64(op1, op2);
  #else
    return easysimd_svqadd_s64(op1, easysimd_svdup_n_s64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_n_s64
  #define svqadd_n_s64(op1, op2) easysimd_svqadd_n_s64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svqadd_u8(easysimd_svuint8_t op1, easysimd_svuint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_u8(op1, op2);
  #else
    easysimd_svuint8_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vqaddq_u8(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_adds_epu8(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_adds_epu8(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_adds_epu8(op1.m128i[i], op2.m128i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = easysimd_math_adds_u8(op1.values[i], op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_u8
  #define svqadd_u8(op1, op2) easysimd_svqadd_u8(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svqadd_n_u8(easysimd_svuint8_t op1, uint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_n_u8(op1, op2);
  #else
    return easysimd_svqadd_u8(op1, easysimd_svdup_n_u8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_n_u8
  #define svqadd_n_u8(op1, op2) easysimd_svqadd_n_u8(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svqadd_u16(easysimd_svuint16_t op1, easysimd_svuint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_u16(op1, op2);
  #else
    easysimd_svuint16_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vqaddq_u16(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_adds_epu16(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_adds_epu16(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_adds_epu16(op1.m128i[i], op2.m128i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = easysimd_math_adds_u16(op1.values[i], op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_u16
  #define svqadd_u16(op1, op2) easysimd_svqadd_u16(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svqadd_n_u16(easysimd_svuint16_t op1, uint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_n_u16(op1, op2);
  #else
    return easysimd_svqadd_u16(op1, easysimd_svdup_n_u16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_n_u16
  #define svqadd_n_u16(op1, op2) easysimd_svqadd_n_u16(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svqadd_u32(easysimd_svuint32_t op1, easysimd_svuint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_u32(op1, op2);
  #else
    easysimd_svuint32_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vqaddq_u32(op1.neon, op2.neon);
    #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
      r.altivec =
        vec_packs(
          vec_unpackh(op1.altivec) + vec_unpackh(op2.altivec),
          vec_unpackl(op1.altivec) + vec_unpackl(op2.altivec)
        );
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = easysimd_math_adds_u32(op1.values[i], op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_u32
  #define svqadd_u32(op1, op2) easysimd_svqadd_u32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svqadd_n_u32(easysimd_svuint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_n_u32(op1, op2);
  #else
    return easysimd_svqadd_u32(op1, easysimd_svdup_n_u32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_n_u32
  #define svqadd_n_u32(op1, op2) easysimd_svqadd_n_u32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svqadd_u64(easysimd_svuint64_t op1, easysimd_svuint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_u64(op1, op2);
  #else
    easysimd_svuint64_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vqaddq_u64(op1.neon, op2.neon);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = easysimd_math_adds_u64(op1.values[i], op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_u64
  #define svqadd_u64(op1, op2) easysimd_svqadd_u64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svqadd_n_u64(easysimd_svuint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svqadd_n_u64(op1, op2);
  #else
    return easysimd_svqadd_u64(op1, easysimd_svdup_n_u64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svqadd_n_u64
  #define svqadd_n_u64(op1, op2) easysimd_svqadd_n_u64(op1, op2)
#endif

#if defined(__cplusplus)
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svqadd(   easysimd_svint8_t op1,    easysimd_svint8_t op2) { return easysimd_svqadd_s8   (op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svqadd(  easysimd_svint16_t op1,   easysimd_svint16_t op2) { return easysimd_svqadd_s16  (op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svqadd(  easysimd_svint32_t op1,   easysimd_svint32_t op2) { return easysimd_svqadd_s32  (op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svqadd(  easysimd_svint64_t op1,   easysimd_svint64_t op2) { return easysimd_svqadd_s64  (op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svqadd(  easysimd_svuint8_t op1,   easysimd_svuint8_t op2) { return easysimd_svqadd_u8   (op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svqadd( easysimd_svuint16_t op1,  easysimd_svuint16_t op2) { return easysimd_svqadd_u16  (op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svqadd( easysimd_svuint32_t op1,  easysimd_svuint32_t op2) { return easysimd_svqadd_u32  (op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svqadd( easysimd_svuint64_t op1,  easysimd_svuint64_t op2) { return easysimd_svqadd_u64  (op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svqadd(   easysimd_svint8_t op1,            int8_t op2) { return easysimd_svqadd_n_s8 (op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svqadd(  easysimd_svint16_t op1,           int16_t op2) { return easysimd_svqadd_n_s16(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svqadd(  easysimd_svint32_t op1,           int32_t op2) { return easysimd_svqadd_n_s32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svqadd(  easysimd_svint64_t op1,           int64_t op2) { return easysimd_svqadd_n_s64(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svqadd(  easysimd_svuint8_t op1,           uint8_t op2) { return easysimd_svqadd_n_u8 (op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svqadd( easysimd_svuint16_t op1,          uint16_t op2) { return easysimd_svqadd_n_u16(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svqadd( easysimd_svuint32_t op1,          uint32_t op2) { return easysimd_svqadd_n_u32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svqadd( easysimd_svuint64_t op1,          uint64_t op2) { return easysimd_svqadd_n_u64(op1, op2); }
#elif defined(EASYSIMD_GENERIC_)
  #define easysimd_svqadd_x(op1, op2) \
    (EASYSIMD_GENERIC_((op2), \
         easysimd_svint8_t: easysimd_svqadd_s8, \
        easysimd_svint16_t: easysimd_svqadd_s16, \
        easysimd_svint32_t: easysimd_svqadd_s32, \
        easysimd_svint64_t: easysimd_svqadd_s64, \
        easysimd_svuint8_t: easysimd_svqadd_u8, \
       easysimd_svuint16_t: easysimd_svqadd_u16, \
       easysimd_svuint32_t: easysimd_svqadd_u32, \
       easysimd_svuint64_t: easysimd_svqadd_u64, \
                 int8_t: easysimd_svqadd_n_s8, \
                int16_t: easysimd_svqadd_n_s16, \
                int32_t: easysimd_svqadd_n_s32, \
                int64_t: easysimd_svqadd_n_s64, \
                uint8_t: easysimd_svqadd_n_u8, \
               uint16_t: easysimd_svqadd_n_u16, \
               uint32_t: easysimd_svqadd_n_u32, \
               uint64_t: easysimd_svqadd_n_u64)((pg), (op1), (op2)))
#endif
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef svqadd
  #define svqadd(op1, op2) easysimd_svqadd((pg), (op1), (op2))
#endif

HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_SVE_QADD_H */
