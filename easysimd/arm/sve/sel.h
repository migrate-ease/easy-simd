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

#if !defined(EASYSIMD_ARM_SVE_SEL_H)
#define EASYSIMD_ARM_SVE_SEL_H

#include "types.h"
#include "reinterpret.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_x_svsel_s8_z(easysimd_svbool_t pg, easysimd_svint8_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s8_z(pg, op1, op1);
  #else
    easysimd_svint8_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vandq_s8(pg.neon_i8, op1.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_maskz_mov_epi8(easysimd_svbool_to_mmask64(pg), op1.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_maskz_mov_epi8(easysimd_svbool_to_mmask32(pg), op1.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_and_si256(pg.m256i[i], op1.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(pg.m128i[i], op1.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = pg.values_i8 & op1.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = pg.values_i8[i] & op1.values[i];
      }
    #endif

    return r;
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint8_t
easysimd_svsel_s8(easysimd_svbool_t pg, easysimd_svint8_t op1, easysimd_svint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_s8(pg, op1, op2);
  #else
    easysimd_svint8_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vbslq_s8(pg.neon_u8, op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_mask_mov_epi8(op2.m512i, easysimd_svbool_to_mmask64(pg), op1.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_mask_mov_epi8(op2.m256i[0], easysimd_svbool_to_mmask32(pg), op1.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_blendv_epi8(op2.m256i[i], op1.m256i[i], pg.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE4_1_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_blendv_epi8(op2.m128i[i], op1.m128i[i], pg.m128i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_or_si128(_mm_and_si128(pg.m128i[i], op1.m128i[i]), _mm_andnot_si128(pg.m128i[i], op2.m128i[i]));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = (pg.values_i8 & op1.values) | (~pg.values_i8 & op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = (pg.values_i8[i] & op1.values[i]) | (~pg.values_i8[i] & op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_s8
  #define svsel_s8(pg, op1, op2) easysimd_svsel_s8(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_x_svsel_s16_z(easysimd_svbool_t pg, easysimd_svint16_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s16_z(pg, op1, op1);
  #else
    easysimd_svint16_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vandq_s16(pg.neon_i16, op1.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_maskz_mov_epi16(easysimd_svbool_to_mmask32(pg), op1.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_maskz_mov_epi16(easysimd_svbool_to_mmask16(pg), op1.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_and_si256(pg.m256i[i], op1.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(pg.m128i[i], op1.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = pg.values_i16 & op1.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = pg.values_i16[i] & op1.values[i];
      }
    #endif

    return r;
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint16_t
easysimd_svsel_s16(easysimd_svbool_t pg, easysimd_svint16_t op1, easysimd_svint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_s16(pg, op1, op2);
  #else
    easysimd_svint16_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vbslq_s16(pg.neon_u16, op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_mask_mov_epi16(op2.m512i, easysimd_svbool_to_mmask32(pg), op1.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_mask_mov_epi16(op2.m256i[0], easysimd_svbool_to_mmask16(pg), op1.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_blendv_epi8(op2.m256i[i], op1.m256i[i], pg.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE4_1_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_blendv_epi8(op2.m128i[i], op1.m128i[i], pg.m128i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_or_si128(_mm_and_si128(pg.m128i[i], op1.m128i[i]), _mm_andnot_si128(pg.m128i[i], op2.m128i[i]));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = (pg.values_i16 & op1.values) | (~pg.values_i16 & op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = (pg.values_i16[i] & op1.values[i]) | (~pg.values_i16[i] & op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_s16
  #define svsel_s16(pg, op1, op2) easysimd_svsel_s16(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_x_svsel_s32_z(easysimd_svbool_t pg, easysimd_svint32_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s32_z(pg, op1, op1);
  #else
    easysimd_svint32_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vandq_s32(pg.neon_i32, op1.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_maskz_mov_epi32(easysimd_svbool_to_mmask16(pg), op1.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_maskz_mov_epi32(easysimd_svbool_to_mmask8(pg), op1.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_and_si256(pg.m256i[i], op1.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(pg.m128i[i], op1.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = pg.values_i32 & op1.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = pg.values_i32[i] & op1.values[i];
      }
    #endif

    return r;
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint32_t
easysimd_svsel_s32(easysimd_svbool_t pg, easysimd_svint32_t op1, easysimd_svint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_s32(pg, op1, op2);
  #else
    easysimd_svint32_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vbslq_s32(pg.neon_u32, op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_mask_mov_epi32(op2.m512i, easysimd_svbool_to_mmask16(pg), op1.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_mask_mov_epi32(op2.m256i[0], easysimd_svbool_to_mmask8(pg), op1.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_blendv_epi8(op2.m256i[i], op1.m256i[i], pg.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE4_1_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_blendv_epi8(op2.m128i[i], op1.m128i[i], pg.m128i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_or_si128(_mm_and_si128(pg.m128i[i], op1.m128i[i]), _mm_andnot_si128(pg.m128i[i], op2.m128i[i]));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = (pg.values_i32 & op1.values) | (~pg.values_i32 & op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = (pg.values_i32[i] & op1.values[i]) | (~pg.values_i32[i] & op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_s32
  #define svsel_s32(pg, op1, op2) easysimd_svsel_s32(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_x_svsel_s64_z(easysimd_svbool_t pg, easysimd_svint64_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_s64_z(pg, op1, op1);
  #else
    easysimd_svint64_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vandq_s64(pg.neon_i64, op1.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_maskz_mov_epi64(easysimd_svbool_to_mmask8(pg), op1.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_maskz_mov_epi64(easysimd_svbool_to_mmask4(pg), op1.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_and_si256(pg.m256i[i], op1.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_and_si128(pg.m128i[i], op1.m128i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = pg.values_i64 & op1.values;
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = pg.values_i64[i] & op1.values[i];
      }
    #endif

    return r;
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svint64_t
easysimd_svsel_s64(easysimd_svbool_t pg, easysimd_svint64_t op1, easysimd_svint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_s64(pg, op1, op2);
  #else
    easysimd_svint64_t r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r.neon = vbslq_s64(pg.neon_u64, op1.neon, op2.neon);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
      r.m512i = _mm512_mask_mov_epi64(op2.m512i, easysimd_svbool_to_mmask8(pg), op1.m512i);
    #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
      r.m256i[0] = _mm256_mask_mov_epi64(op2.m256i[0], easysimd_svbool_to_mmask4(pg), op1.m256i[0]);
    #elif defined(EASYSIMD_X86_AVX2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m256i) / sizeof(r.m256i[0])) ; i++) {
        r.m256i[i] = _mm256_blendv_epi8(op2.m256i[i], op1.m256i[i], pg.m256i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE4_1_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_blendv_epi8(op2.m128i[i], op1.m128i[i], pg.m128i[i]);
      }
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.m128i) / sizeof(r.m128i[0])) ; i++) {
        r.m128i[i] = _mm_or_si128(_mm_and_si128(pg.m128i[i], op1.m128i[i]), _mm_andnot_si128(pg.m128i[i], op2.m128i[i]));
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r.values = (pg.values_i64 & op1.values) | (~pg.values_i64 & op2.values);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, sizeof(r.values) / sizeof(r.values[0])) ; i++) {
        r.values[i] = (pg.values_i64[i] & op1.values[i]) | (~pg.values_i64[i] & op2.values[i]);
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_s64
  #define svsel_s64(pg, op1, op2) easysimd_svsel_s64(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_x_svsel_u8_z(easysimd_svbool_t pg, easysimd_svuint8_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u8_z(pg, op1, op1);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && ((EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512) || defined(EASYSIMD_X86_AVX512VL_NATIVE))
    easysimd_svuint8_t r;

    #if EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512
      r.m512i = _mm512_maskz_mov_epi8(easysimd_svbool_to_mmask64(pg), op1.m512i);
    #else
      r.m256i[0] = _mm256_maskz_mov_epi8(easysimd_svbool_to_mmask32(pg), op1.m256i[0]);
    #endif

    return r;
  #else
    return easysimd_svreinterpret_u8_s8(easysimd_x_svsel_s8_z(pg, easysimd_svreinterpret_s8_u8(op1)));
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint8_t
easysimd_svsel_u8(easysimd_svbool_t pg, easysimd_svuint8_t op1, easysimd_svuint8_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_u8(pg, op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && ((EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512) || defined(EASYSIMD_X86_AVX512VL_NATIVE))
    easysimd_svuint8_t r;

    #if EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512
      r.m512i = _mm512_mask_mov_epi8(op2.m512i, easysimd_svbool_to_mmask64(pg), op1.m512i);
    #else
      r.m256i[0] = _mm256_mask_mov_epi8(op2.m256i[0], easysimd_svbool_to_mmask32(pg), op1.m256i[0]);
    #endif

    return r;
  #else
    return easysimd_svreinterpret_u8_s8(easysimd_svsel_s8(pg, easysimd_svreinterpret_s8_u8(op1), easysimd_svreinterpret_s8_u8(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_u8
  #define svsel_u8(pg, op1, op2) easysimd_svsel_u8(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_x_svsel_u16_z(easysimd_svbool_t pg, easysimd_svuint16_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u16_z(pg, op1, op1);
  #else
    return easysimd_svreinterpret_u16_s16(easysimd_x_svsel_s16_z(pg, easysimd_svreinterpret_s16_u16(op1)));
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint16_t
easysimd_svsel_u16(easysimd_svbool_t pg, easysimd_svuint16_t op1, easysimd_svuint16_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_u16(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u16_s16(easysimd_svsel_s16(pg, easysimd_svreinterpret_s16_u16(op1), easysimd_svreinterpret_s16_u16(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_u16
  #define svsel_u16(pg, op1, op2) easysimd_svsel_u16(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_x_svsel_u32_z(easysimd_svbool_t pg, easysimd_svuint32_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u32_z(pg, op1, op1);
  #else
    return easysimd_svreinterpret_u32_s32(easysimd_x_svsel_s32_z(pg, easysimd_svreinterpret_s32_u32(op1)));
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint32_t
easysimd_svsel_u32(easysimd_svbool_t pg, easysimd_svuint32_t op1, easysimd_svuint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_u32(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u32_s32(easysimd_svsel_s32(pg, easysimd_svreinterpret_s32_u32(op1), easysimd_svreinterpret_s32_u32(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_u32
  #define svsel_u32(pg, op1, op2) easysimd_svsel_u32(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_x_svsel_u64_z(easysimd_svbool_t pg, easysimd_svuint64_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svand_u64_z(pg, op1, op1);
  #else
    return easysimd_svreinterpret_u64_s64(easysimd_x_svsel_s64_z(pg, easysimd_svreinterpret_s64_u64(op1)));
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svuint64_t
easysimd_svsel_u64(easysimd_svbool_t pg, easysimd_svuint64_t op1, easysimd_svuint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_u64(pg, op1, op2);
  #else
    return easysimd_svreinterpret_u64_s64(easysimd_svsel_s64(pg, easysimd_svreinterpret_s64_u64(op1), easysimd_svreinterpret_s64_u64(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_u64
  #define svsel_u64(pg, op1, op2) easysimd_svsel_u64(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_x_svsel_f32_z(easysimd_svbool_t pg, easysimd_svfloat32_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return easysimd_svreinterpret_f32_s32(svand_s32_z(pg, easysimd_svreinterpret_s32_f32(op1), easysimd_svreinterpret_s32_f32(op1)));
  #else
    return easysimd_svreinterpret_f32_s32(easysimd_x_svsel_s32_z(pg, easysimd_svreinterpret_s32_f32(op1)));
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat32_t
easysimd_svsel_f32(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_f32(pg, op1, op2);
  #else
    return easysimd_svreinterpret_f32_s32(easysimd_svsel_s32(pg, easysimd_svreinterpret_s32_f32(op1), easysimd_svreinterpret_s32_f32(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_f32
  #define svsel_f32(pg, op1, op2) easysimd_svsel_f32(pg, op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_x_svsel_f64_z(easysimd_svbool_t pg, easysimd_svfloat64_t op1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return easysimd_svreinterpret_f64_s64(svand_s64_z(pg, easysimd_svreinterpret_s64_f64(op1), easysimd_svreinterpret_s64_f64(op1)));
  #else
    return easysimd_svreinterpret_f64_s64(easysimd_x_svsel_s64_z(pg, easysimd_svreinterpret_s64_f64(op1)));
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svfloat64_t
easysimd_svsel_f64(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svsel_f64(pg, op1, op2);
  #else
    return easysimd_svreinterpret_f64_s64(easysimd_svsel_s64(pg, easysimd_svreinterpret_s64_f64(op1), easysimd_svreinterpret_s64_f64(op2)));
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svsel_f64
  #define svsel_f64(pg, op1, op2) easysimd_svsel_f64(pg, op1, op2)
#endif

#if defined(__cplusplus)
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_x_svsel_z(easysimd_svbool_t pg,    easysimd_svint8_t op1) { return easysimd_x_svsel_s8_z (pg, op1); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_x_svsel_z(easysimd_svbool_t pg,   easysimd_svint16_t op1) { return easysimd_x_svsel_s16_z(pg, op1); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_x_svsel_z(easysimd_svbool_t pg,   easysimd_svint32_t op1) { return easysimd_x_svsel_s32_z(pg, op1); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_x_svsel_z(easysimd_svbool_t pg,   easysimd_svint64_t op1) { return easysimd_x_svsel_s64_z(pg, op1); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_x_svsel_z(easysimd_svbool_t pg,   easysimd_svuint8_t op1) { return easysimd_x_svsel_u8_z (pg, op1); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_x_svsel_z(easysimd_svbool_t pg,  easysimd_svuint16_t op1) { return easysimd_x_svsel_u16_z(pg, op1); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_x_svsel_z(easysimd_svbool_t pg,  easysimd_svuint32_t op1) { return easysimd_x_svsel_u32_z(pg, op1); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_x_svsel_z(easysimd_svbool_t pg,  easysimd_svuint64_t op1) { return easysimd_x_svsel_u64_z(pg, op1); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_x_svsel_z(easysimd_svbool_t pg, easysimd_svfloat32_t op1) { return easysimd_x_svsel_f32_z(pg, op1); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_x_svsel_z(easysimd_svbool_t pg, easysimd_svfloat64_t op1) { return easysimd_x_svsel_f64_z(pg, op1); }

  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint8_t easysimd_svsel(easysimd_svbool_t pg,    easysimd_svint8_t op1,    easysimd_svint8_t op2) { return easysimd_svsel_s8 (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint16_t easysimd_svsel(easysimd_svbool_t pg,   easysimd_svint16_t op1,   easysimd_svint16_t op2) { return easysimd_svsel_s16(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint32_t easysimd_svsel(easysimd_svbool_t pg,   easysimd_svint32_t op1,   easysimd_svint32_t op2) { return easysimd_svsel_s32(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svint64_t easysimd_svsel(easysimd_svbool_t pg,   easysimd_svint64_t op1,   easysimd_svint64_t op2) { return easysimd_svsel_s64(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint8_t easysimd_svsel(easysimd_svbool_t pg,   easysimd_svuint8_t op1,   easysimd_svuint8_t op2) { return easysimd_svsel_u8 (pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint16_t easysimd_svsel(easysimd_svbool_t pg,  easysimd_svuint16_t op1,  easysimd_svuint16_t op2) { return easysimd_svsel_u16(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint32_t easysimd_svsel(easysimd_svbool_t pg,  easysimd_svuint32_t op1,  easysimd_svuint32_t op2) { return easysimd_svsel_u32(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svuint64_t easysimd_svsel(easysimd_svbool_t pg,  easysimd_svuint64_t op1,  easysimd_svuint64_t op2) { return easysimd_svsel_u64(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat32_t easysimd_svsel(easysimd_svbool_t pg, easysimd_svfloat32_t op1, easysimd_svfloat32_t op2) { return easysimd_svsel_f32(pg, op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svfloat64_t easysimd_svsel(easysimd_svbool_t pg, easysimd_svfloat64_t op1, easysimd_svfloat64_t op2) { return easysimd_svsel_f64(pg, op1, op2); }
#elif defined(EASYSIMD_GENERIC_)
  #define easysimd_x_svsel_z(pg, op1) \
    (EASYSIMD_GENERIC_((op1), \
         easysimd_svint8_t: easysimd_x_svsel_s8_z, \
        easysimd_svint16_t: easysimd_x_svsel_s16_z, \
        easysimd_svint32_t: easysimd_x_svsel_s32_z, \
        easysimd_svint64_t: easysimd_x_svsel_s64_z, \
        easysimd_svuint8_t: easysimd_x_svsel_u8_z, \
       easysimd_svuint16_t: easysimd_x_svsel_u16_z, \
       easysimd_svuint32_t: easysimd_x_svsel_u32_z, \
       easysimd_svuint64_t: easysimd_x_svsel_u64_z, \
      easysimd_svfloat32_t: easysimd_x_svsel_f32_z, \
      easysimd_svfloat64_t: easysimd_x_svsel_f64_z)((pg), (op1)))

  #define easysimd_svsel(pg, op1, op2) \
    (EASYSIMD_GENERIC_((op1), \
         easysimd_svint8_t: easysimd_svsel_s8, \
        easysimd_svint16_t: easysimd_svsel_s16, \
        easysimd_svint32_t: easysimd_svsel_s32, \
        easysimd_svint64_t: easysimd_svsel_s64, \
        easysimd_svuint8_t: easysimd_svsel_u8, \
       easysimd_svuint16_t: easysimd_svsel_u16, \
       easysimd_svuint32_t: easysimd_svsel_u32, \
       easysimd_svuint64_t: easysimd_svsel_u64, \
      easysimd_svfloat32_t: easysimd_svsel_f32, \
      easysimd_svfloat64_t: easysimd_svsel_f64)((pg), (op1), (op2)))
#endif
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef svsel
  #define svsel(pg, op1) easysimd_svsel((pg), (op1))
#endif

HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_SVE_SEL_H */
