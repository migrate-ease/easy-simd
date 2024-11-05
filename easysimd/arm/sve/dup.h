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

#if !defined(EASYSIMD_ARM_SVE_DUP_H)
#define EASYSIMD_ARM_SVE_DUP_H

#include "types.h"
#include "reinterpret.h"
#include "sel.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svdup_n_s8(int8_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s8(op);
  #else
    easysimd_svint8_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vdupq_n_s8(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_set1_epi8(op);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_set1_epi8(op);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_set1_epi8(op);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s8
  #define svdup_n_s8(op) easysimd_svdup_n_s8((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svdup_s8(int8_t op) {
  return easysimd_svdup_n_s8(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s8
  #define svdup_s8(op) easysimd_svdup_n_s8((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svdup_n_s8_z(easysimd_svbool_t pg, int8_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s8_z(pg, op);
  #else
    return easysimd_x_svsel_s8_z(pg, easysimd_svdup_n_s8(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s8_z
  #define svdup_n_s8_z(pg, op) easysimd_svdup_n_s8_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svdup_s8_z(easysimd_svbool_t pg, int8_t op) {
  return easysimd_svdup_n_s8_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s8_z
  #define svdup_s8_z(pg, op) easysimd_svdup_n_s8_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svdup_n_s8_m(easysimd_svint8_t inactive, easysimd_svbool_t pg, int8_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s8_m(inactive, pg, op);
  #else
    return easysimd_svsel_s8(pg, easysimd_svdup_n_s8(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s8_m
  #define svdup_n_s8_m(inactive, pg, op) easysimd_svdup_n_s8_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svdup_s8_m(easysimd_svint8_t inactive, easysimd_svbool_t pg, int8_t op) {
  return easysimd_svdup_n_s8_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s8_m
  #define svdup_s8_m(inactive, pg, op) easysimd_svdup_n_s8_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svdup_n_s16(int16_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s16(op);
  #else
    easysimd_svint16_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vdupq_n_s16(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_set1_epi16(op);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_set1_epi16(op);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_set1_epi16(op);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s16
  #define svdup_n_s16(op) easysimd_svdup_n_s16((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svdup_s16(int16_t op) {
  return easysimd_svdup_n_s16(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s16
  #define svdup_s16(op) easysimd_svdup_n_s16((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svdup_n_s16_z(easysimd_svbool_t pg, int16_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s16_z(pg, op);
  #else
    return easysimd_x_svsel_s16_z(pg, easysimd_svdup_n_s16(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s16_z
  #define svdup_n_s16_z(pg, op) easysimd_svdup_n_s16_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svdup_s16_z(easysimd_svbool_t pg, int8_t op) {
  return easysimd_svdup_n_s16_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s16_z
  #define svdup_s16_z(pg, op) easysimd_svdup_n_s16_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svdup_n_s16_m(easysimd_svint16_t inactive, easysimd_svbool_t pg, int16_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s16_m(inactive, pg, op);
  #else
    return easysimd_svsel_s16(pg, easysimd_svdup_n_s16(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s16_m
  #define svdup_n_s16_m(inactive, pg, op) easysimd_svdup_n_s16_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svdup_s16_m(easysimd_svint16_t inactive, easysimd_svbool_t pg, int16_t op) {
  return easysimd_svdup_n_s16_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s16_m
  #define svdup_s16_m(inactive, pg, op) easysimd_svdup_n_s16_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svdup_n_s32(int32_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s32(op);
  #else
    easysimd_svint32_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vdupq_n_s32(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_set1_epi32(op);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_set1_epi32(op);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_set1_epi32(op);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s32
  #define svdup_n_s32(op) easysimd_svdup_n_s32((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svdup_s32(int8_t op) {
  return easysimd_svdup_n_s32(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s32
  #define svdup_s32(op) easysimd_svdup_n_s32((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svdup_n_s32_z(easysimd_svbool_t pg, int32_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s32_z(pg, op);
  #else
    return easysimd_x_svsel_s32_z(pg, easysimd_svdup_n_s32(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s32_z
  #define svdup_n_s32_z(pg, op) easysimd_svdup_n_s32_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svdup_s32_z(easysimd_svbool_t pg, int32_t op) {
  return easysimd_svdup_n_s32_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s32_z
  #define svdup_s32_z(pg, op) easysimd_svdup_n_s32_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svdup_n_s32_m(easysimd_svint32_t inactive, easysimd_svbool_t pg, int32_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s32_m(inactive, pg, op);
  #else
    return easysimd_svsel_s32(pg, easysimd_svdup_n_s32(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s32_m
  #define svdup_n_s32_m(inactive, pg, op) easysimd_svdup_n_s32_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svdup_s32_m(easysimd_svint32_t inactive, easysimd_svbool_t pg, int32_t op) {
  return easysimd_svdup_n_s32_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s32_m
  #define svdup_s32_m(inactive, pg, op) easysimd_svdup_n_s32_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svdup_n_s64(int64_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s64(op);
  #else
    easysimd_svint64_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vdupq_n_s64(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_set1_epi64(op);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_set1_epi64x(op);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_set1_epi64x(op);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s64
  #define svdup_n_s64(op) easysimd_svdup_n_s64((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svdup_s64(int64_t op) {
  return easysimd_svdup_n_s64(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s64
  #define svdup_s64(op) easysimd_svdup_n_s64((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svdup_n_s64_z(easysimd_svbool_t pg, int64_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s64_z(pg, op);
  #else
    return easysimd_x_svsel_s64_z(pg, easysimd_svdup_n_s64(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s64_z
  #define svdup_n_s64_z(pg, op) easysimd_svdup_n_s64_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svdup_s64_z(easysimd_svbool_t pg, int64_t op) {
  return easysimd_svdup_n_s64_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s64_z
  #define svdup_s64_z(pg, op) easysimd_svdup_n_f64_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svdup_n_s64_m(easysimd_svint64_t inactive, easysimd_svbool_t pg, int64_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_s64_m(inactive, pg, op);
  #else
    return easysimd_svsel_s64(pg, easysimd_svdup_n_s64(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_s64_m
  #define svdup_n_s64_m(inactive, pg, op) easysimd_svdup_n_s64_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svdup_s64_m(easysimd_svint64_t inactive, easysimd_svbool_t pg, int64_t op) {
  return easysimd_svdup_n_s64_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_s64_m
  #define svdup_s64_m(inactive, pg, op) easysimd_svdup_n_s64_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svdup_n_u8(uint8_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u8(op);
  #else
    easysimd_svuint8_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vdupq_n_u8(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_set1_epi8(HEDLEY_STATIC_CAST(int8_t, op));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_set1_epi8(HEDLEY_STATIC_CAST(int8_t, op));
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, op));
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u8
  #define svdup_n_u8(op) easysimd_svdup_n_u8((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svdup_u8(uint8_t op) {
  return easysimd_svdup_n_u8(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u8
  #define svdup_u8(op) easysimd_svdup_n_u8((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svdup_n_u8_z(easysimd_svbool_t pg, uint8_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u8_z(pg, op);
  #else
    return easysimd_x_svsel_u8_z(pg, easysimd_svdup_n_u8(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u8_z
  #define svdup_n_u8_z(pg, op) easysimd_svdup_n_u8_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svdup_u8_z(easysimd_svbool_t pg, uint8_t op) {
  return easysimd_svdup_n_u8_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u8_z
  #define svdup_u8_z(pg, op) easysimd_svdup_n_u8_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svdup_n_u8_m(easysimd_svuint8_t inactive, easysimd_svbool_t pg, uint8_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u8_m(inactive, pg, op);
  #else
    return easysimd_svsel_u8(pg, easysimd_svdup_n_u8(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u8_m
  #define svdup_n_u8_m(inactive, pg, op) easysimd_svdup_n_u8_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svdup_u8_m(easysimd_svuint8_t inactive, easysimd_svbool_t pg, uint8_t op) {
  return easysimd_svdup_n_u8_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u8_m
  #define svdup_u8_m(inactive, pg, op) easysimd_svdup_n_u8_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svdup_n_u16(uint16_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u16(op);
  #else
    easysimd_svuint16_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vdupq_n_u16(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_set1_epi16(HEDLEY_STATIC_CAST(int16_t, op));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_set1_epi16(HEDLEY_STATIC_CAST(int16_t, op));
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_set1_epi16(HEDLEY_STATIC_CAST(int16_t, op));
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u16
  #define svdup_n_u16(op) easysimd_svdup_n_u16((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svdup_u16(uint16_t op) {
  return easysimd_svdup_n_u16(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u16
  #define svdup_u16(op) easysimd_svdup_n_u16((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svdup_n_u16_z(easysimd_svbool_t pg, uint16_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u16_z(pg, op);
  #else
    return easysimd_x_svsel_u16_z(pg, easysimd_svdup_n_u16(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u16_z
  #define svdup_n_u16_z(pg, op) easysimd_svdup_n_u16_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svdup_u16_z(easysimd_svbool_t pg, uint8_t op) {
  return easysimd_svdup_n_u16_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u16_z
  #define svdup_u16_z(pg, op) easysimd_svdup_n_u16_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svdup_n_u16_m(easysimd_svuint16_t inactive, easysimd_svbool_t pg, uint16_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u16_m(inactive, pg, op);
  #else
    return easysimd_svsel_u16(pg, easysimd_svdup_n_u16(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u16_m
  #define svdup_n_u16_m(inactive, pg, op) easysimd_svdup_n_u16_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svdup_u16_m(easysimd_svuint16_t inactive, easysimd_svbool_t pg, uint16_t op) {
  return easysimd_svdup_n_u16_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u16_m
  #define svdup_u16_m(inactive, pg, op) easysimd_svdup_n_u16_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svdup_n_u32(uint32_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u32(op);
  #else
    easysimd_svuint32_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vdupq_n_u32(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_set1_epi32(HEDLEY_STATIC_CAST(int32_t, op));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_set1_epi32(HEDLEY_STATIC_CAST(int32_t, op));
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_set1_epi32(HEDLEY_STATIC_CAST(int32_t, op));
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u32
  #define svdup_n_u32(op) easysimd_svdup_n_u32((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svdup_u32(uint8_t op) {
  return easysimd_svdup_n_u32(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u32
  #define svdup_u32(op) easysimd_svdup_n_u32((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svdup_n_u32_z(easysimd_svbool_t pg, uint32_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u32_z(pg, op);
  #else
    return easysimd_x_svsel_u32_z(pg, easysimd_svdup_n_u32(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u32_z
  #define svdup_n_u32_z(pg, op) easysimd_svdup_n_u32_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svdup_u32_z(easysimd_svbool_t pg, uint32_t op) {
  return easysimd_svdup_n_u32_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u32_z
  #define svdup_u32_z(pg, op) easysimd_svdup_n_u32_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svdup_n_u32_m(easysimd_svuint32_t inactive, easysimd_svbool_t pg, uint32_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u32_m(inactive, pg, op);
  #else
    return easysimd_svsel_u32(pg, easysimd_svdup_n_u32(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u32_m
  #define svdup_n_u32_m(inactive, pg, op) easysimd_svdup_n_u32_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svdup_u32_m(easysimd_svuint32_t inactive, easysimd_svbool_t pg, uint32_t op) {
  return easysimd_svdup_n_u32_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u32_m
  #define svdup_u32_m(inactive, pg, op) easysimd_svdup_n_u32_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svdup_n_u64(uint64_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u64(op);
  #else
    easysimd_svuint64_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vdupq_n_u64(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_set1_epi64(HEDLEY_STATIC_CAST(int64_t, op));
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_set1_epi64x(HEDLEY_STATIC_CAST(int64_t, op));
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_set1_epi64x(HEDLEY_STATIC_CAST(int64_t, op));
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u64
  #define svdup_n_u64(op) easysimd_svdup_n_u64((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svdup_u64(uint64_t op) {
  return easysimd_svdup_n_u64(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u64
  #define svdup_u64(op) easysimd_svdup_n_u64((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svdup_n_u64_z(easysimd_svbool_t pg, uint64_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u64_z(pg, op);
  #else
    return easysimd_x_svsel_u64_z(pg, easysimd_svdup_n_u64(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u64_z
  #define svdup_n_u64_z(pg, op) easysimd_svdup_n_u64_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svdup_u64_z(easysimd_svbool_t pg, uint64_t op) {
  return easysimd_svdup_n_u64_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u64_z
  #define svdup_u64_z(pg, op) easysimd_svdup_n_f64_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svdup_n_u64_m(easysimd_svuint64_t inactive, easysimd_svbool_t pg, uint64_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_u64_m(inactive, pg, op);
  #else
    return easysimd_svsel_u64(pg, easysimd_svdup_n_u64(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_u64_m
  #define svdup_n_u64_m(inactive, pg, op) easysimd_svdup_n_u64_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svdup_u64_m(easysimd_svuint64_t inactive, easysimd_svbool_t pg, uint64_t op) {
  return easysimd_svdup_n_u64_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_u64_m
  #define svdup_u64_m(inactive, pg, op) easysimd_svdup_n_u64_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svdup_n_f32(easysimd_float32 op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_f32(op);
  #else
    easysimd_svfloat32_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vdupq_n_f32(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512 = _mm512_set1_ps(op);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256) / sizeof(r.m256[0])) ; i++) {
        r.m256[i] = _mm256_set1_ps(op);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128) / sizeof(r.m128[0])) ; i++) {
        r.m128[i] = _mm_set1_ps(op);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_f32
  #define svdup_n_f32(op) easysimd_svdup_n_f32((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svdup_f32(int8_t op) {
  return easysimd_svdup_n_f32(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_f32
  #define svdup_f32(op) easysimd_svdup_n_f32((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svdup_n_f32_z(easysimd_svbool_t pg, easysimd_float32 op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_f32_z(pg, op);
  #else
    return easysimd_x_svsel_f32_z(pg, easysimd_svdup_n_f32(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_f32_z
  #define svdup_n_f32_z(pg, op) easysimd_svdup_n_f32_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svdup_f32_z(easysimd_svbool_t pg, easysimd_float32 op) {
  return easysimd_svdup_n_f32_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_f32_z
  #define svdup_f32_z(pg, op) easysimd_svdup_n_f32_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svdup_n_f32_m(easysimd_svfloat32_t inactive, easysimd_svbool_t pg, easysimd_float32_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_f32_m(inactive, pg, op);
  #else
    return easysimd_svsel_f32(pg, easysimd_svdup_n_f32(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_f32_m
  #define svdup_n_f32_m(inactive, pg, op) easysimd_svdup_n_f32_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svdup_f32_m(easysimd_svfloat32_t inactive, easysimd_svbool_t pg, easysimd_float32_t op) {
  return easysimd_svdup_n_f32_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_f32_m
  #define svdup_f32_m(inactive, pg, op) easysimd_svdup_n_f32_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svdup_n_f64(easysimd_float64 op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_f64(op);
  #else
    easysimd_svfloat64_t r;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r.neon = vdupq_n_f64(op);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512d = _mm512_set1_pd(op);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256d) / sizeof(r.m256d[0])) ; i++) {
        r.m256d[i] = _mm256_set1_pd(op);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128d) / sizeof(r.m128d[0])) ; i++) {
        r.m128d[i] = _mm_set1_pd(op);
      }
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_f64
  #define svdup_n_f64(op) easysimd_svdup_n_f64((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svdup_f64(easysimd_float64 op) {
  return easysimd_svdup_n_f64(op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_f64
  #define svdup_f64(op) easysimd_svdup_n_f64((op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svdup_n_f64_z(easysimd_svbool_t pg, easysimd_float64 op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_f64_z(pg, op);
  #else
    return easysimd_x_svsel_f64_z(pg, easysimd_svdup_n_f64(op));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_f64_z
  #define svdup_n_f64_z(pg, op) easysimd_svdup_n_f64_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svdup_f64_z(easysimd_svbool_t pg, easysimd_float64 op) {
  return easysimd_svdup_n_f64_z(pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_f64_z
  #define svdup_f64_z(pg, op) easysimd_svdup_n_f64_z((pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svdup_n_f64_m(easysimd_svfloat64_t inactive, easysimd_svbool_t pg, easysimd_float64_t op) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svdup_n_f64_m(inactive, pg, op);
  #else
    return easysimd_svsel_f64(pg, easysimd_svdup_n_f64(op), inactive);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_n_f64_m
  #define svdup_n_f64_m(inactive, pg, op) easysimd_svdup_n_f64_m((inactive), (pg), (op))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svdup_f64_m(easysimd_svfloat64_t inactive, easysimd_svbool_t pg, easysimd_float64_t op) {
  return easysimd_svdup_n_f64_m(inactive, pg, op);
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svdup_f64_m
  #define svdup_f64_m(inactive, pg, op) easysimd_svdup_n_f64_m((inactive), (pg), (op))
#endif

#if defined(__cplusplus)
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svdup_n  (                          int8_t  op) { return easysimd_svdup_n_s8    (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svdup    (                          int8_t  op) { return easysimd_svdup_n_s8    (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svdup_n_z(easysimd_svbool_t pg,        int8_t  op) { return easysimd_svdup_n_s8_z  (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svdup_z  (easysimd_svbool_t pg,        int8_t  op) { return easysimd_svdup_n_s8_z  (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svdup_n  (                         int16_t  op) { return easysimd_svdup_n_s16   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svdup    (                         int16_t  op) { return easysimd_svdup_n_s16   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svdup_n_z(easysimd_svbool_t pg,       int16_t  op) { return easysimd_svdup_n_s16_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svdup_z  (easysimd_svbool_t pg,       int16_t  op) { return easysimd_svdup_n_s16_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svdup_n  (                         int32_t  op) { return easysimd_svdup_n_s32   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svdup    (                         int32_t  op) { return easysimd_svdup_n_s32   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svdup_n_z(easysimd_svbool_t pg,       int32_t  op) { return easysimd_svdup_n_s32_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svdup_z  (easysimd_svbool_t pg,       int32_t  op) { return easysimd_svdup_n_s32_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svdup_n  (                         int64_t  op) { return easysimd_svdup_n_s64   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svdup    (                         int64_t  op) { return easysimd_svdup_n_s64   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svdup_n_z(easysimd_svbool_t pg,       int64_t  op) { return easysimd_svdup_n_s64_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svdup_z  (easysimd_svbool_t pg,       int64_t  op) { return easysimd_svdup_n_s64_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svdup_n  (                         uint8_t  op) { return easysimd_svdup_n_u8    (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svdup    (                         uint8_t  op) { return easysimd_svdup_n_u8    (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svdup_n_z(easysimd_svbool_t pg,       uint8_t  op) { return easysimd_svdup_n_u8_z  (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svdup_z  (easysimd_svbool_t pg,       uint8_t  op) { return easysimd_svdup_n_u8_z  (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svdup_n  (                        uint16_t  op) { return easysimd_svdup_n_u16   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svdup    (                        uint16_t  op) { return easysimd_svdup_n_u16   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svdup_n_z(easysimd_svbool_t pg,      uint16_t  op) { return easysimd_svdup_n_u16_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svdup_z  (easysimd_svbool_t pg,      uint16_t  op) { return easysimd_svdup_n_u16_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svdup_n  (                        uint32_t  op) { return easysimd_svdup_n_u32   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svdup    (                        uint32_t  op) { return easysimd_svdup_n_u32   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svdup_n_z(easysimd_svbool_t pg,      uint32_t  op) { return easysimd_svdup_n_u32_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svdup_z  (easysimd_svbool_t pg,      uint32_t  op) { return easysimd_svdup_n_u32_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svdup_n  (                        uint64_t  op) { return easysimd_svdup_n_u64   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svdup    (                        uint64_t  op) { return easysimd_svdup_n_u64   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svdup_n_z(easysimd_svbool_t pg,      uint64_t  op) { return easysimd_svdup_n_u64_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svdup_z  (easysimd_svbool_t pg,      uint64_t  op) { return easysimd_svdup_n_u64_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svdup_n  (                   easysimd_float32  op) { return easysimd_svdup_n_f32   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svdup    (                   easysimd_float32  op) { return easysimd_svdup_n_f32   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svdup_n_z(easysimd_svbool_t pg, easysimd_float32  op) { return easysimd_svdup_n_f32_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svdup_z  (easysimd_svbool_t pg, easysimd_float32  op) { return easysimd_svdup_n_f32_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svdup_n  (                   easysimd_float64  op) { return easysimd_svdup_n_f64   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svdup    (                   easysimd_float64  op) { return easysimd_svdup_n_f64   (    op); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svdup_n_z(easysimd_svbool_t pg, easysimd_float64  op) { return easysimd_svdup_n_f64_z (pg, op); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svdup_z  (easysimd_svbool_t pg, easysimd_float64  op) { return easysimd_svdup_n_f64_z (pg, op); }

  #if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
    EASYSIMD_FUNCTION_ATTRIBUTES    svint8_t svdup_n  (                    int8_t  op) { return svdup_n_s8    (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES    svint8_t svdup    (                    int8_t  op) { return svdup_n_s8    (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES    svint8_t svdup_n_z(svbool_t pg,        int8_t  op) { return svdup_n_s8_z  (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES    svint8_t svdup_z  (svbool_t pg,        int8_t  op) { return svdup_n_s8_z  (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint16_t svdup_n  (                   int16_t  op) { return svdup_n_s16   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint16_t svdup    (                   int16_t  op) { return svdup_n_s16   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint16_t svdup_n_z(svbool_t pg,       int16_t  op) { return svdup_n_s16_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint16_t svdup_z  (svbool_t pg,       int16_t  op) { return svdup_n_s16_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint32_t svdup_n  (                   int32_t  op) { return svdup_n_s32   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint32_t svdup    (                   int32_t  op) { return svdup_n_s32   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint32_t svdup_n_z(svbool_t pg,       int32_t  op) { return svdup_n_s32_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint32_t svdup_z  (svbool_t pg,       int32_t  op) { return svdup_n_s32_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint64_t svdup_n  (                   int64_t  op) { return svdup_n_s64   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint64_t svdup    (                   int64_t  op) { return svdup_n_s64   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint64_t svdup_n_z(svbool_t pg,       int64_t  op) { return svdup_n_s64_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svint64_t svdup_z  (svbool_t pg,       int64_t  op) { return svdup_n_s64_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svuint8_t svdup_n  (                   uint8_t  op) { return svdup_n_u8    (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svuint8_t svdup    (                   uint8_t  op) { return svdup_n_u8    (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svuint8_t svdup_n_z(svbool_t pg,       uint8_t  op) { return svdup_n_u8_z  (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES   svuint8_t svdup_z  (svbool_t pg,       uint8_t  op) { return svdup_n_u8_z  (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint16_t svdup_n  (                  uint16_t  op) { return svdup_n_u16   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint16_t svdup    (                  uint16_t  op) { return svdup_n_u16   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint16_t svdup_n_z(svbool_t pg,      uint16_t  op) { return svdup_n_u16_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint16_t svdup_z  (svbool_t pg,      uint16_t  op) { return svdup_n_u16_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint32_t svdup_n  (                  uint32_t  op) { return svdup_n_u32   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint32_t svdup    (                  uint32_t  op) { return svdup_n_u32   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint32_t svdup_n_z(svbool_t pg,      uint32_t  op) { return svdup_n_u32_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint32_t svdup_z  (svbool_t pg,      uint32_t  op) { return svdup_n_u32_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint64_t svdup_n  (                  uint64_t  op) { return svdup_n_u64   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint64_t svdup    (                  uint64_t  op) { return svdup_n_u64   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint64_t svdup_n_z(svbool_t pg,      uint64_t  op) { return svdup_n_u64_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES  svuint64_t svdup_z  (svbool_t pg,      uint64_t  op) { return svdup_n_u64_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES svfloat32_t svdup_n  (             easysimd_float32  op) { return svdup_n_f32   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES svfloat32_t svdup    (             easysimd_float32  op) { return svdup_n_f32   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES svfloat32_t svdup_n_z(svbool_t pg, easysimd_float32  op) { return svdup_n_f32_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES svfloat32_t svdup_z  (svbool_t pg, easysimd_float32  op) { return svdup_n_f32_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES svfloat64_t svdup_n  (             easysimd_float64  op) { return svdup_n_f64   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES svfloat64_t svdup    (             easysimd_float64  op) { return svdup_n_f64   (    op); }
    EASYSIMD_FUNCTION_ATTRIBUTES svfloat64_t svdup_n_z(svbool_t pg, easysimd_float64  op) { return svdup_n_f64_z (pg, op); }
    EASYSIMD_FUNCTION_ATTRIBUTES svfloat64_t svdup_z  (svbool_t pg, easysimd_float64  op) { return svdup_n_f64_z (pg, op); }
  #endif
#elif defined(EASYSIMD_GENERIC_)
  #define easysimd_svdup_n(op) \
    (EASYSIMD_GENERIC_((op), \
         int8_t: easysimd_svdup_n_s8, \
        int16_t: easysimd_svdup_n_s16, \
        int32_t: easysimd_svdup_n_s32, \
        int64_t: easysimd_svdup_n_s64, \
        uint8_t: easysimd_svdup_n_u8, \
       uint16_t: easysimd_svdup_n_u16, \
       uint32_t: easysimd_svdup_n_u32, \
       uint64_t: easysimd_svdup_n_u64, \
      float32_t: easysimd_svdup_n_f32, \
      float64_t: easysimd_svdup_n_f64)((op)))
  #define easysimd_svdup(op) easysimd_svdup_n((op))

  #define easysimd_svdup_n_z(pg, op) \
    (EASYSIMD_GENERIC_((op), \
         int8_t: easysimd_svdup_n_s8_z, \
        int16_t: easysimd_svdup_n_s16_z, \
        int32_t: easysimd_svdup_n_s32_z, \
        int64_t: easysimd_svdup_n_s64_z, \
        uint8_t: easysimd_svdup_n_s8_z, \
       uint16_t: easysimd_svdup_n_u16_z, \
       uint32_t: easysimd_svdup_n_u32_z, \
       uint64_t: easysimd_svdup_n_u64_z, \
      float32_t: easysimd_svdup_n_u32_z, \
      float64_t: easysimd_svdup_n_f64_z)((pg), (op)))
  #define easysimd_svdup_z(pg, op) easysimd_svdup_n_z((pg), (op))
#endif
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef svdup
  #undef svdup_z
  #undef svdup_n
  #undef svdup_n_z
  #define svdup_n(op) easysimd_svdup_n((op))
  #define svdup_n_z(pg, op) easysimd_svdup_n_z((pg), (op))
  #define svdup(op) easysimd_svdup((op))
  #define svdup_z(pg, op) easysimd_svdup_z((pg), (op))
#endif

#if defined(EASYSIMD_ARM_SVE_NATIVE)
  #define easysimd_svdupq_n_s32(op1, op2, op3, op4)        svdupq_n_s32((op1), (op2), (op3), (op4))
  #define easysimd_svdupq_n_f32(op1, op2, op3, op4)        svdupq_n_f32((op1), (op2), (op3), (op4))
  #define easysimd_svdupq_n_s64(op1, op2)                  svdupq_n_s64((op1), (op2))
  #define easysimd_svdupq_n_f64(op1, op2)                  svdupq_n_f64((op1), (op2))
#endif

HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_SVE_DUP_H */
