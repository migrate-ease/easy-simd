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

#if !defined(EASYSIMD_ARM_SVE_AND_H)
#define EASYSIMD_ARM_SVE_AND_H

#include "types.h"
#include "dup.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svand_s8_x(easysimd_svbool_t pg, easysimd_svint8_t op1, easysimd_svint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s8_x(pg, op1, op2);
  #else
    easysimd_svint8_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vandq_s8(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_and_si512(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_and_si256(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_and_si256(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values & op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] & op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s8_x
  #define svand_s8_x(pg, op1, op2) easysimd_svand_s8_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svand_s8_z(easysimd_svbool_t pg, easysimd_svint8_t op1, easysimd_svint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s8_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_s8_z(pg, easysimd_svand_s8_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s8_z
  #define svand_s8_z(pg, op1, op2) easysimd_svand_s8_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svand_s8_m(easysimd_svbool_t pg, easysimd_svint8_t op1, easysimd_svint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s8_m(pg, op1, op2);
  #else
    return easysimd_svsel_s8(pg, easysimd_svand_s8_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s8_m
  #define svand_s8_m(pg, op1, op2) easysimd_svand_s8_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svand_n_s8_z(easysimd_svbool_t pg, easysimd_svint8_t op1, int8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s8_z(pg, op1, op2);
  #else
    return easysimd_svand_s8_z(pg, op1, easysimd_svdup_n_s8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s8_z
  #define svand_n_s8_z(pg, op1, op2) easysimd_svand_n_s8_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svand_n_s8_m(easysimd_svbool_t pg, easysimd_svint8_t op1, int8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s8_m(pg, op1, op2);
  #else
    return easysimd_svand_s8_m(pg, op1, easysimd_svdup_n_s8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s8_m
  #define svand_n_s8_m(pg, op1, op2) easysimd_svand_n_s8_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svand_n_s8_x(easysimd_svbool_t pg, easysimd_svint8_t op1, int8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s8_x(pg, op1, op2);
  #else
    return easysimd_svand_s8_x(pg, op1, easysimd_svdup_n_s8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s8_x
  #define svand_n_s8_x(pg, op1, op2) easysimd_svand_n_s8_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svand_s16_x(easysimd_svbool_t pg, easysimd_svint16_t op1, easysimd_svint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s16_x(pg, op1, op2);
  #else
    easysimd_svint16_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vandq_s16(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_and_si512(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_and_si256(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_and_si256(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values & op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] & op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s16_x
  #define svand_s16_x(pg, op1, op2) easysimd_svand_s16_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svand_s16_z(easysimd_svbool_t pg, easysimd_svint16_t op1, easysimd_svint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s16_z(pg, op1, op2);
  #else
    return easysimd_x_svsel_s16_z(pg, easysimd_svand_s16_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s16_z
  #define svand_s16_z(pg, op1, op2) easysimd_svand_s16_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svand_s16_m(easysimd_svbool_t pg, easysimd_svint16_t op1, easysimd_svint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s16_m(pg, op1, op2);
  #else
    return easysimd_svsel_s16(pg, easysimd_svand_s16_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s16_m
  #define svand_s16_m(pg, op1, op2) easysimd_svand_s16_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svand_n_s16_z(easysimd_svbool_t pg, easysimd_svint16_t op1, int16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s16_z(pg, op1, op2);
  #else
    return easysimd_svand_s16_z(pg, op1, easysimd_svdup_n_s16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s16_z
  #define svand_n_s16_z(pg, op1, op2) easysimd_svand_n_s16_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svand_n_s16_m(easysimd_svbool_t pg, easysimd_svint16_t op1, int16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s16_m(pg, op1, op2);
  #else
    return easysimd_svand_s16_m(pg, op1, easysimd_svdup_n_s16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s16_m
  #define svand_n_s16_m(pg, op1, op2) easysimd_svand_n_s16_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svand_n_s16_x(easysimd_svbool_t pg, easysimd_svint16_t op1, int16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s16_x(pg, op1, op2);
  #else
    return easysimd_svand_s16_x(pg, op1, easysimd_svdup_n_s16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s16_x
  #define svand_n_s16_x(pg, op1, op2) easysimd_svand_n_s16_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svand_s32_x(easysimd_svbool_t pg, easysimd_svint32_t op1, easysimd_svint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s32_x(pg, op1, op2);
  #else
    easysimd_svint32_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vandq_s32(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_and_si512(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_and_si256(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_and_si256(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values & op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] & op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s32_x
  #define svand_s32_x(pg, op1, op2) easysimd_svand_s32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svand_s32_z(easysimd_svbool_t pg, easysimd_svint32_t op1, easysimd_svint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s32_z(pg, op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && ((EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512) || defined(EASYSIMD_X86_AVX512VL_NATIVE))
    easysimd_svint32_t r;

    #if EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512
      r.m512i = _mm512_maskz_and_epi32(easysimd_svbool_to_mmask16(pg), op1.m512i, op2.m512i);
    #else
      r.m256i[0] = _mm256_maskz_and_epi32(easysimd_svbool_to_mmask8(pg), op1.m256i[0], op2.m256i[0]);
    #endif

    return r;
  #else
    return easysimd_x_svsel_s32_z(pg, easysimd_svand_s32_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s32_z
  #define svand_s32_z(pg, op1, op2) easysimd_svand_s32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svand_s32_m(easysimd_svbool_t pg, easysimd_svint32_t op1, easysimd_svint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s32_m(pg, op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && ((EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512) || defined(EASYSIMD_X86_AVX512VL_NATIVE))
    easysimd_svint32_t r;

    #if EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512
      r.m512i = _mm512_mask_and_epi32(op1.m512i, easysimd_svbool_to_mmask16(pg), op1.m512i, op2.m512i);
    #else
      r.m256i[0] = _mm256_mask_and_epi32(op1.m256i[0], easysimd_svbool_to_mmask8(pg), op1.m256i[0], op2.m256i[0]);
    #endif

    return r;
  #else
    return easysimd_svsel_s32(pg, easysimd_svand_s32_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s32_m
  #define svand_s32_m(pg, op1, op2) easysimd_svand_s32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svand_n_s32_z(easysimd_svbool_t pg, easysimd_svint32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s32_z(pg, op1, op2);
  #else
    return easysimd_svand_s32_z(pg, op1, easysimd_svdup_n_s32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s32_z
  #define svand_n_s32_z(pg, op1, op2) easysimd_svand_n_s32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svand_n_s32_m(easysimd_svbool_t pg, easysimd_svint32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s32_m(pg, op1, op2);
  #else
    return easysimd_svand_s32_m(pg, op1, easysimd_svdup_n_s32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s32_m
  #define svand_n_s32_m(pg, op1, op2) easysimd_svand_n_s32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svand_n_s32_x(easysimd_svbool_t pg, easysimd_svint32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s32_x(pg, op1, op2);
  #else
    return easysimd_svand_s32_x(pg, op1, easysimd_svdup_n_s32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s32_x
  #define svand_n_s32_x(pg, op1, op2) easysimd_svand_n_s32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svand_s64_x(easysimd_svbool_t pg, easysimd_svint64_t op1, easysimd_svint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s64_x(pg, op1, op2);
  #else
    easysimd_svint64_t r;
    HEDLEY_STATIC_CAST(void, pg);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vandq_s64(op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_and_si512(op1.m512i, op2.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_and_si256(op1.m256i[0], op2.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_and_si256(op1.m256i[i], op2.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(op1.m128i[i], op2.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = op1.values & op2.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = op1.values[i] & op2.values[i];
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s64_x
  #define svand_s64_x(pg, op1, op2) easysimd_svand_s64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svand_s64_z(easysimd_svbool_t pg, easysimd_svint64_t op1, easysimd_svint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s64_z(pg, op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && ((EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512) || defined(EASYSIMD_X86_AVX512VL_NATIVE))
    easysimd_svint64_t r;

    #if EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512
      r.m512i = _mm512_maskz_and_epi64(easysimd_svbool_to_mmask8(pg), op1.m512i, op2.m512i);
    #else
      r.m256i[0] = _mm256_maskz_and_epi64(easysimd_svbool_to_mmask4(pg), op1.m256i[0], op2.m256i[0]);
    #endif

    return r;
  #else
    return easysimd_x_svsel_s64_z(pg, easysimd_svand_s64_x(pg, op1, op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s64_z
  #define svand_s64_z(pg, op1, op2) easysimd_svand_s64_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svand_s64_m(easysimd_svbool_t pg, easysimd_svint64_t op1, easysimd_svint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s64_m(pg, op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && ((EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512) || defined(EASYSIMD_X86_AVX512VL_NATIVE))
    easysimd_svint64_t r;

    #if EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512
      r.m512i = _mm512_mask_and_epi64(op1.m512i, easysimd_svbool_to_mmask8(pg), op1.m512i, op2.m512i);
    #else
      r.m256i[0] = _mm256_mask_and_epi64(op1.m256i[0], easysimd_svbool_to_mmask4(pg), op1.m256i[0], op2.m256i[0]);
    #endif

    return r;
  #else
    return easysimd_svsel_s64(pg, easysimd_svand_s64_x(pg, op1, op2), op1);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_s64_m
  #define svand_s64_m(pg, op1, op2) easysimd_svand_s64_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svand_n_s64_z(easysimd_svbool_t pg, easysimd_svint64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s64_z(pg, op1, op2);
  #else
    return easysimd_svand_s64_z(pg, op1, easysimd_svdup_n_s64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s64_z
  #define svand_n_s64_z(pg, op1, op2) easysimd_svand_n_s64_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svand_n_s64_m(easysimd_svbool_t pg, easysimd_svint64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s64_m(pg, op1, op2);
  #else
    return easysimd_svand_s64_m(pg, op1, easysimd_svdup_n_s64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s64_m
  #define svand_n_s64_m(pg, op1, op2) easysimd_svand_n_s64_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svand_n_s64_x(easysimd_svbool_t pg, easysimd_svint64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_s64_x(pg, op1, op2);
  #else
    return easysimd_svand_s64_x(pg, op1, easysimd_svdup_n_s64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_s64_x
  #define svand_n_s64_x(pg, op1, op2) easysimd_svand_n_s64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svand_u8_z(easysimd_svbool_t pg, easysimd_svuint8_t op1, easysimd_svuint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u8_z(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u8_s8(easysimd_svand_s8_z(pg, easysimd_svreinterpret_s8_u8(op1), easysimd_svreinterpret_s8_u8(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u8_z
  #define svand_u8_z(pg, op1, op2) easysimd_svand_u8_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svand_u8_m(easysimd_svbool_t pg, easysimd_svuint8_t op1, easysimd_svuint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u8_m(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u8_s8(easysimd_svand_s8_m(pg, easysimd_svreinterpret_s8_u8(op1), easysimd_svreinterpret_s8_u8(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u8_m
  #define svand_u8_m(pg, op1, op2) easysimd_svand_u8_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svand_u8_x(easysimd_svbool_t pg, easysimd_svuint8_t op1, easysimd_svuint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u8_x(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u8_s8(easysimd_svand_s8_x(pg, easysimd_svreinterpret_s8_u8(op1), easysimd_svreinterpret_s8_u8(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u8_x
  #define svand_u8_x(pg, op1, op2) easysimd_svand_u8_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svand_n_u8_z(easysimd_svbool_t pg, easysimd_svuint8_t op1, uint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u8_z(pg, op1, op2);
  #else
    return easysimd_svand_u8_z(pg, op1, easysimd_svdup_n_u8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u8_z
  #define svand_n_u8_z(pg, op1, op2) easysimd_svand_n_u8_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svand_n_u8_m(easysimd_svbool_t pg, easysimd_svuint8_t op1, uint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u8_m(pg, op1, op2);
  #else
    return easysimd_svand_u8_m(pg, op1, easysimd_svdup_n_u8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u8_m
  #define svand_n_u8_m(pg, op1, op2) easysimd_svand_n_u8_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svand_n_u8_x(easysimd_svbool_t pg, easysimd_svuint8_t op1, uint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u8_x(pg, op1, op2);
  #else
    return easysimd_svand_u8_x(pg, op1, easysimd_svdup_n_u8(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u8_x
  #define svand_n_u8_x(pg, op1, op2) easysimd_svand_n_u8_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svand_u16_z(easysimd_svbool_t pg, easysimd_svuint16_t op1, easysimd_svuint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u16_z(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u16_s16(easysimd_svand_s16_z(pg, easysimd_svreinterpret_s16_u16(op1), easysimd_svreinterpret_s16_u16(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u16_z
  #define svand_u16_z(pg, op1, op2) easysimd_svand_u16_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svand_u16_m(easysimd_svbool_t pg, easysimd_svuint16_t op1, easysimd_svuint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u16_m(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u16_s16(easysimd_svand_s16_m(pg, easysimd_svreinterpret_s16_u16(op1), easysimd_svreinterpret_s16_u16(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u16_m
  #define svand_u16_m(pg, op1, op2) easysimd_svand_u16_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svand_u16_x(easysimd_svbool_t pg, easysimd_svuint16_t op1, easysimd_svuint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u16_x(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u16_s16(easysimd_svand_s16_x(pg, easysimd_svreinterpret_s16_u16(op1), easysimd_svreinterpret_s16_u16(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u16_x
  #define svand_u16_x(pg, op1, op2) easysimd_svand_u16_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svand_n_u16_z(easysimd_svbool_t pg, easysimd_svuint16_t op1, uint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u16_z(pg, op1, op2);
  #else
    return easysimd_svand_u16_z(pg, op1, easysimd_svdup_n_u16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u16_z
  #define svand_n_u16_z(pg, op1, op2) easysimd_svand_n_u16_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svand_n_u16_m(easysimd_svbool_t pg, easysimd_svuint16_t op1, uint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u16_m(pg, op1, op2);
  #else
    return easysimd_svand_u16_m(pg, op1, easysimd_svdup_n_u16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u16_m
  #define svand_n_u16_m(pg, op1, op2) easysimd_svand_n_u16_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svand_n_u16_x(easysimd_svbool_t pg, easysimd_svuint16_t op1, uint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u16_x(pg, op1, op2);
  #else
    return easysimd_svand_u16_x(pg, op1, easysimd_svdup_n_u16(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u16_x
  #define svand_n_u16_x(pg, op1, op2) easysimd_svand_n_u16_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svand_u32_z(easysimd_svbool_t pg, easysimd_svuint32_t op1, easysimd_svuint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u32_z(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u32_s32(easysimd_svand_s32_z(pg, easysimd_svreinterpret_s32_u32(op1), easysimd_svreinterpret_s32_u32(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u32_z
  #define svand_u32_z(pg, op1, op2) easysimd_svand_u32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svand_u32_m(easysimd_svbool_t pg, easysimd_svuint32_t op1, easysimd_svuint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u32_m(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u32_s32(easysimd_svand_s32_m(pg, easysimd_svreinterpret_s32_u32(op1), easysimd_svreinterpret_s32_u32(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u32_m
  #define svand_u32_m(pg, op1, op2) easysimd_svand_u32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svand_u32_x(easysimd_svbool_t pg, easysimd_svuint32_t op1, easysimd_svuint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u32_x(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u32_s32(easysimd_svand_s32_x(pg, easysimd_svreinterpret_s32_u32(op1), easysimd_svreinterpret_s32_u32(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u32_x
  #define svand_u32_x(pg, op1, op2) easysimd_svand_u32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svand_n_u32_z(easysimd_svbool_t pg, easysimd_svuint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u32_z(pg, op1, op2);
  #else
    return easysimd_svand_u32_z(pg, op1, easysimd_svdup_n_u32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u32_z
  #define svand_n_u32_z(pg, op1, op2) easysimd_svand_n_u32_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svand_n_u32_m(easysimd_svbool_t pg, easysimd_svuint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u32_m(pg, op1, op2);
  #else
    return easysimd_svand_u32_m(pg, op1, easysimd_svdup_n_u32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u32_m
  #define svand_n_u32_m(pg, op1, op2) easysimd_svand_n_u32_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svand_n_u32_x(easysimd_svbool_t pg, easysimd_svuint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u32_x(pg, op1, op2);
  #else
    return easysimd_svand_u32_x(pg, op1, easysimd_svdup_n_u32(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u32_x
  #define svand_n_u32_x(pg, op1, op2) easysimd_svand_n_u32_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svand_u64_z(easysimd_svbool_t pg, easysimd_svuint64_t op1, easysimd_svuint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u64_z(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u64_s64(easysimd_svand_s64_z(pg, easysimd_svreinterpret_s64_u64(op1), easysimd_svreinterpret_s64_u64(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u64_z
  #define svand_u64_z(pg, op1, op2) easysimd_svand_u64_z(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svand_u64_m(easysimd_svbool_t pg, easysimd_svuint64_t op1, easysimd_svuint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u64_m(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u64_s64(easysimd_svand_s64_m(pg, easysimd_svreinterpret_s64_u64(op1), easysimd_svreinterpret_s64_u64(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u64_m
  #define svand_u64_m(pg, op1, op2) easysimd_svand_u64_m(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svand_u64_x(easysimd_svbool_t pg, easysimd_svuint64_t op1, easysimd_svuint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u64_x(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u64_s64(easysimd_svand_s64_x(pg, easysimd_svreinterpret_s64_u64(op1), easysimd_svreinterpret_s64_u64(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_u64_x
  #define svand_u64_x(pg, op1, op2) easysimd_svand_u64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svand_n_u64_z(easysimd_svbool_t pg, easysimd_svuint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u64_z(pg, op1, op2);
  #else
    return easysimd_svand_u64_z(pg, op1, easysimd_svdup_n_u64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u64_z
  #define svand_n_u64_x(pg, op1, op2) easysimd_svand_n_u64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svand_n_u64_m(easysimd_svbool_t pg, easysimd_svuint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u64_m(pg, op1, op2);
  #else
    return easysimd_svand_u64_m(pg, op1, easysimd_svdup_n_u64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u64_m
  #define svand_n_u64_x(pg, op1, op2) easysimd_svand_n_u64_x(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svand_n_u64_x(easysimd_svbool_t pg, easysimd_svuint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_n_u64_x(pg, op1, op2);
  #else
    return easysimd_svand_u64_x(pg, op1, easysimd_svdup_n_u64(op2));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svand_n_u64_x
  #define svand_n_u64_x(pg, op1, op2) easysimd_svand_n_u64_x(pg, op1, op2)
#endif

#if defined(__cplusplus)
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svand_z(easysimd_svbool_t pg,    easysimd_svint8_t op1,    easysimd_svint8_t op2) { return easysimd_svand_s8_z (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svand_z(easysimd_svbool_t pg,   easysimd_svint16_t op1,   easysimd_svint16_t op2) { return easysimd_svand_s16_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svand_z(easysimd_svbool_t pg,   easysimd_svint32_t op1,   easysimd_svint32_t op2) { return easysimd_svand_s32_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svand_z(easysimd_svbool_t pg,   easysimd_svint64_t op1,   easysimd_svint64_t op2) { return easysimd_svand_s64_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svand_z(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   easysimd_svuint8_t op2) { return easysimd_svand_u8_z (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svand_z(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  easysimd_svuint16_t op2) { return easysimd_svand_u16_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svand_z(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  easysimd_svuint32_t op2) { return easysimd_svand_u32_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svand_z(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  easysimd_svuint64_t op2) { return easysimd_svand_u64_z(pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svand_m(easysimd_svbool_t pg,    easysimd_svint8_t op1,    easysimd_svint8_t op2) { return easysimd_svand_s8_m (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svand_m(easysimd_svbool_t pg,   easysimd_svint16_t op1,   easysimd_svint16_t op2) { return easysimd_svand_s16_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svand_m(easysimd_svbool_t pg,   easysimd_svint32_t op1,   easysimd_svint32_t op2) { return easysimd_svand_s32_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svand_m(easysimd_svbool_t pg,   easysimd_svint64_t op1,   easysimd_svint64_t op2) { return easysimd_svand_s64_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svand_m(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   easysimd_svuint8_t op2) { return easysimd_svand_u8_m (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svand_m(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  easysimd_svuint16_t op2) { return easysimd_svand_u16_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svand_m(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  easysimd_svuint32_t op2) { return easysimd_svand_u32_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svand_m(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  easysimd_svuint64_t op2) { return easysimd_svand_u64_m(pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svand_x(easysimd_svbool_t pg,    easysimd_svint8_t op1,    easysimd_svint8_t op2) { return easysimd_svand_s8_x (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svand_x(easysimd_svbool_t pg,   easysimd_svint16_t op1,   easysimd_svint16_t op2) { return easysimd_svand_s16_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svand_x(easysimd_svbool_t pg,   easysimd_svint32_t op1,   easysimd_svint32_t op2) { return easysimd_svand_s32_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svand_x(easysimd_svbool_t pg,   easysimd_svint64_t op1,   easysimd_svint64_t op2) { return easysimd_svand_s64_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svand_x(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   easysimd_svuint8_t op2) { return easysimd_svand_u8_x (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svand_x(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  easysimd_svuint16_t op2) { return easysimd_svand_u16_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svand_x(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  easysimd_svuint32_t op2) { return easysimd_svand_u32_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svand_x(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  easysimd_svuint64_t op2) { return easysimd_svand_u64_x(pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svand_z(easysimd_svbool_t pg,    easysimd_svint8_t op1,    int8_t op2) { return  easysimd_svand_n_s8_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svand_z(easysimd_svbool_t pg,   easysimd_svint16_t op1,   int16_t op2) { return easysimd_svand_n_s16_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svand_z(easysimd_svbool_t pg,   easysimd_svint32_t op1,   int32_t op2) { return easysimd_svand_n_s32_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svand_z(easysimd_svbool_t pg,   easysimd_svint64_t op1,   int64_t op2) { return easysimd_svand_n_s64_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svand_z(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   uint8_t op2) { return  easysimd_svand_n_u8_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svand_z(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  uint16_t op2) { return easysimd_svand_n_u16_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svand_z(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  uint32_t op2) { return easysimd_svand_n_u32_z(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svand_z(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  uint64_t op2) { return easysimd_svand_n_u64_z(pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svand_m(easysimd_svbool_t pg,    easysimd_svint8_t op1,    int8_t op2) { return  easysimd_svand_n_s8_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svand_m(easysimd_svbool_t pg,   easysimd_svint16_t op1,   int16_t op2) { return easysimd_svand_n_s16_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svand_m(easysimd_svbool_t pg,   easysimd_svint32_t op1,   int32_t op2) { return easysimd_svand_n_s32_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svand_m(easysimd_svbool_t pg,   easysimd_svint64_t op1,   int64_t op2) { return easysimd_svand_n_s64_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svand_m(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   uint8_t op2) { return  easysimd_svand_n_u8_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svand_m(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  uint16_t op2) { return easysimd_svand_n_u16_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svand_m(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  uint32_t op2) { return easysimd_svand_n_u32_m(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svand_m(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  uint64_t op2) { return easysimd_svand_n_u64_m(pg, op1, op2); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svand_x(easysimd_svbool_t pg,    easysimd_svint8_t op1,    int8_t op2) { return  easysimd_svand_n_s8_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svand_x(easysimd_svbool_t pg,   easysimd_svint16_t op1,   int16_t op2) { return easysimd_svand_n_s16_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svand_x(easysimd_svbool_t pg,   easysimd_svint32_t op1,   int32_t op2) { return easysimd_svand_n_s32_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svand_x(easysimd_svbool_t pg,   easysimd_svint64_t op1,   int64_t op2) { return easysimd_svand_n_s64_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svand_x(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   uint8_t op2) { return  easysimd_svand_n_u8_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svand_x(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  uint16_t op2) { return easysimd_svand_n_u16_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svand_x(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  uint32_t op2) { return easysimd_svand_n_u32_x(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svand_x(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  uint64_t op2) { return easysimd_svand_n_u64_x(pg, op1, op2); }
#elif defined(EASYSIMD_GENERIC_)
  #define easysimd_svand_z(pg, op1, op2) \
    (EASYSIMD_GENERIC_((op2), \
         easysimd_svint8_t: easysimd_svand_s8_z, \
        easysimd_svint16_t: easysimd_svand_s16_z, \
        easysimd_svint32_t: easysimd_svand_s32_z, \
        easysimd_svint64_t: easysimd_svand_s64_z, \
        easysimd_svuint8_t: easysimd_svand_u8_z, \
       easysimd_svuint16_t: easysimd_svand_u16_z, \
       easysimd_svuint32_t: easysimd_svand_u32_z, \
       easysimd_svuint64_t: easysimd_svand_u64_z, \
                 int8_t: easysimd_svand_n_s8_z, \
                int16_t: easysimd_svand_n_s16_z, \
                int32_t: easysimd_svand_n_s32_z, \
                int64_t: easysimd_svand_n_s64_z, \
                uint8_t: easysimd_svand_n_u8_z, \
               uint16_t: easysimd_svand_n_u16_z, \
               uint32_t: easysimd_svand_n_u32_z, \
               uint64_t: easysimd_svand_n_u64_z)((pg), (op1), (op2)))

  #define easysimd_svand_m(pg, op1, op2) \
    (EASYSIMD_GENERIC_((op2), \
         easysimd_svint8_t: easysimd_svand_s8_m, \
        easysimd_svint16_t: easysimd_svand_s16_m, \
        easysimd_svint32_t: easysimd_svand_s32_m, \
        easysimd_svint64_t: easysimd_svand_s64_m, \
        easysimd_svuint8_t: easysimd_svand_u8_m, \
       easysimd_svuint16_t: easysimd_svand_u16_m, \
       easysimd_svuint32_t: easysimd_svand_u32_m, \
       easysimd_svuint64_t: easysimd_svand_u64_m, \
                 int8_t: easysimd_svand_n_s8_m, \
                int16_t: easysimd_svand_n_s16_m, \
                int32_t: easysimd_svand_n_s32_m, \
                int64_t: easysimd_svand_n_s64_m, \
                uint8_t: easysimd_svand_n_u8_m, \
               uint16_t: easysimd_svand_n_u16_m, \
               uint32_t: easysimd_svand_n_u32_m, \
               uint64_t: easysimd_svand_n_u64_m)((pg), (op1), (op2)))

  #define easysimd_svand_x(pg, op1, op2) \
    (EASYSIMD_GENERIC_((op2), \
         easysimd_svint8_t: easysimd_svand_s8_x, \
        easysimd_svint16_t: easysimd_svand_s16_x, \
        easysimd_svint32_t: easysimd_svand_s32_x, \
        easysimd_svint64_t: easysimd_svand_s64_x, \
        easysimd_svuint8_t: easysimd_svand_u8_x, \
       easysimd_svuint16_t: easysimd_svand_u16_x, \
       easysimd_svuint32_t: easysimd_svand_u32_x, \
       easysimd_svuint64_t: easysimd_svand_u64_x, \
                 int8_t: easysimd_svand_n_s8_x, \
                int16_t: easysimd_svand_n_s16_x, \
                int32_t: easysimd_svand_n_s32_x, \
                int64_t: easysimd_svand_n_s64_x, \
                uint8_t: easysimd_svand_n_u8_x, \
               uint16_t: easysimd_svand_n_u16_x, \
               uint32_t: easysimd_svand_n_u32_x, \
               uint64_t: easysimd_svand_n_u64_x)((pg), (op1), (op2)))
#endif
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef svand_x
  #undef svand_z
  #undef svand_m
  #define svand_x(pg, op1, op2) easysimd_svand_x((pg), (op1), (op2))
  #define svand_z(pg, op1, op2) easysimd_svand_z((pg), (op1), (op2))
  #define svand_m(pg, op1, op2) easysimd_svand_m((pg), (op1), (op2))
#endif

HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_SVE_AND_H */
