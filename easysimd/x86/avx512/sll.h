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

#if !defined(EASYSIMD_X86_AVX512_SLL_H)
#define EASYSIMD_X86_AVX512_SLL_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"
#include "setzero.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sll_epi16 (easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_sll_epi16(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    if (shift > 15) {
        return easysimd_mm512_setzero_si512();
    }
    uint16_t shift_ = HEDLEY_STATIC_CAST(uint16_t, shift);

    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svlsl_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svlsl_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], shift_);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svlsl_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], shift_);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svlsl_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], shift_);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      r_.m256i[0] = easysimd_mm256_sll_epi16(a_.m256i[0], count);
      r_.m256i[1] = easysimd_mm256_sll_epi16(a_.m256i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);
      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
      if (shift > 15)
        return easysimd_mm512_setzero_si512();

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i16 = a_.i16 << HEDLEY_STATIC_CAST(int16_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (shift));
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sll_epi16
  #define _mm512_sll_epi16(a, count) easysimd_mm512_sll_epi16(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_sll_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_sll_epi16(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    uint16_t shift = HEDLEY_STATIC_CAST(uint16_t, count.i64[0]);
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], shift), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], shift), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], shift), src.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], shift), src.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
    easysimd__m128i_private
      count_ = easysimd__m128i_to_private(count);
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
    if (shift > 15)
      return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_setzero_si512());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i16 = a_.i16 << HEDLEY_STATIC_CAST(int16_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (shift));
      }
    #endif

    return easysimd_mm512_mask_mov_epi16(src, k, easysimd__m512i_from_private(r_));

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sll_epi16
  #define _mm512_mask_sll_epi16(src, k, a, count) easysimd_mm512_mask_sll_epi16(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_sll_epi16 (easysimd__mmask32 k, easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_sll_epi16(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    uint16_t shift = HEDLEY_STATIC_CAST(uint16_t, count.i64[0]);
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], shift);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_i16[EASYSIMD_SV_INDEX_2], shift);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_i16[EASYSIMD_SV_INDEX_3], shift);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
    easysimd__m128i_private
      count_ = easysimd__m128i_to_private(count);
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
    if (shift > 15)
      return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_setzero_si512());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i16 = a_.i16 << HEDLEY_STATIC_CAST(int16_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (shift));
      }
    #endif
    return easysimd_mm512_maskz_mov_epi16(k, easysimd__m512i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_sll_epi16
  #define _mm512_maskz_sll_epi16(k, a, count) easysimd_mm512_maskz_sll_epi16(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_sll_epi16 (easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_sll_epi16(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    uint16_t shift = HEDLEY_STATIC_CAST(uint16_t, count.i64[0]);
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], shift), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], shift), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    easysimd__m128i_private
      count_ = easysimd__m128i_to_private(count);

    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
    if (shift > 15)
      return easysimd_mm256_mask_mov_epi16(src, k, easysimd_mm256_setzero_si256());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i16 = a_.i16 << HEDLEY_STATIC_CAST(int16_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (shift));
      }
    #endif

    return easysimd_mm256_mask_mov_epi16(src, k, easysimd__m256i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_sll_epi16
  #define _mm256_mask_sll_epi16(src, k, a, count) easysimd_mm256_mask_sll_epi16(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_sll_epi16 (easysimd__mmask16 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_maskz_sll_epi16(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    uint16_t shift = HEDLEY_STATIC_CAST(uint16_t, count.i64[0]);
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    easysimd__m128i_private
      count_ = easysimd__m128i_to_private(count);

    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
    if (shift > 15)
      return easysimd_mm256_maskz_mov_epi16(k, easysimd_mm256_setzero_si256());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i16 = a_.i16 << HEDLEY_STATIC_CAST(int16_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (shift));
      }
    #endif

    return easysimd_mm256_maskz_mov_epi16(k, easysimd__m256i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_sll_epi16
  #define _mm256_maskz_sll_epi16(k, a, count) easysimd_mm256_maskz_sll_epi16(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_sll_epi16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_sll_epi16(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b16();
    uint16_t shift = HEDLEY_STATIC_CAST(uint16_t, count.i64[0]);
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, 0), svlsl_n_s16_x(pg, a.sve_i16, shift), src.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    if (count_.u64[0] > 15)
      return easysimd_mm_mask_mov_epi16(src, k, easysimd_mm_setzero_si128());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u16 = (a_.u16 << count_.u64[0]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (a_.u16[i] << count_.u64[0]));
      }
    #endif

    return easysimd_mm_mask_mov_epi16(src, k, easysimd__m128i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_sll_epi16
  #define _mm_mask_sll_epi16(src, k, a, count) easysimd_mm_mask_sll_epi16(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_sll_epi16 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_sll_epi16(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    uint16_t shift = HEDLEY_STATIC_CAST(uint16_t, count.i64[0]);
    r.sve_i16 = svlsl_n_s16_z(EASYSIMD_MASK_TO_B16(k, 0), a.sve_i16, shift);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    if (count_.u64[0] > 15)
      return easysimd_mm_maskz_mov_epi16(k, easysimd_mm_setzero_si128());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u16 = (a_.u16 << count_.u64[0]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (a_.u16[i] << count_.u64[0]));
      }
    #endif

    return easysimd_mm_maskz_mov_epi16(k, easysimd__m128i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_sll_epi16
  #define _mm_maskz_sll_epi16(k, a, count) easysimd_mm_maskz_sll_epi16(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sll_epi32 (easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sll_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    if (shift > 31) {
        return easysimd_mm512_setzero_si512();
    }
    uint32_t shift_ = HEDLEY_STATIC_CAST(uint32_t, shift);

    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], shift_);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], shift_);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], shift_);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      r_.m256i[0] = easysimd_mm256_sll_epi32(a_.m256i[0], count);
      r_.m256i[1] = easysimd_mm256_sll_epi32(a_.m256i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);
      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
      if (shift > 31)
        return easysimd_mm512_setzero_si512();

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i32 = a_.i32 << HEDLEY_STATIC_CAST(int32_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, a_.i32[i] << (shift));
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sll_epi32
  #define _mm512_sll_epi32(a, count) easysimd_mm512_sll_epi32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_sll_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sll_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    uint32_t shift = HEDLEY_STATIC_CAST(uint32_t, b.i64[0]);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], shift), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], shift), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], shift), src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], shift), src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
    easysimd__m128i_private
      b_ = easysimd__m128i_to_private(b);
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, b_.i64[0]);
    if (shift > 31)
      return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_setzero_si512());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i32 = a_.i32 << HEDLEY_STATIC_CAST(int32_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, a_.i32[i] << (shift));
      }
    #endif
    
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd__m512i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sll_epi32
  #define _mm512_mask_sll_epi32(src, k, a, b) easysimd_mm512_mask_sll_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_sll_epi32(easysimd__mmask16 k, easysimd__m512i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_sll_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    uint32_t shift = HEDLEY_STATIC_CAST(uint32_t, b.i64[0]);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], shift);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], shift);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], shift);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
    easysimd__m128i_private
      b_ = easysimd__m128i_to_private(b);
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, b_.i64[0]);
    if (shift > 31)
      return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_setzero_si512());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i32 = a_.i32 << HEDLEY_STATIC_CAST(int32_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, a_.i32[i] << (shift));
      }
    #endif
    return easysimd_mm512_maskz_mov_epi32(k, easysimd__m512i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_sll_epi32
  #define _mm512_maskz_sll_epi32(k, a, b) easysimd_mm512_maskz_sll_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_sll_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_sll_epi32(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    uint32_t shift = HEDLEY_STATIC_CAST(uint32_t, count.i64[0]);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], shift), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsl_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], shift), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    easysimd__m128i_private
      count_ = easysimd__m128i_to_private(count);

    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
    if (shift > 31)
      return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_setzero_si256());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i32 = a_.i32 << HEDLEY_STATIC_CAST(int32_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, a_.i32[i] << (shift));
      }
    #endif

    return easysimd_mm256_mask_mov_epi32(src, k, easysimd__m256i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_sll_epi32
  #define _mm256_mask_sll_epi32(src, k, a, count) easysimd_mm256_mask_sll_epi32(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_sll_epi32 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_sll_epi32(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    uint32_t shift = HEDLEY_STATIC_CAST(uint32_t, count.i64[0]);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    easysimd__m128i_private
      count_ = easysimd__m128i_to_private(count);

    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
    if (shift > 31)
      return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_setzero_si256());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i32 = a_.i32 << HEDLEY_STATIC_CAST(int32_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, a_.i32[i] << (shift));
      }
    #endif
    return easysimd_mm256_maskz_mov_epi32(k, easysimd__m256i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_sll_epi32
  #define _mm256_maskz_sll_epi32(k, a, count) easysimd_mm256_maskz_sll_epi32(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_sll_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_sll_epi32(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    uint32_t shift = HEDLEY_STATIC_CAST(uint32_t, count.i64[0]);
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, 0), svlsl_n_s32_x(pg, a.sve_i32, shift), src.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    if (count_.u64[0] > 31)
      return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_setzero_si128());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u32 = (a_.u32 << count_.u64[0]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (a_.u32[i] << count_.u64[0]));
      }
    #endif
    return easysimd_mm_mask_mov_epi32(src, k, easysimd__m128i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_sll_epi32
  #define _mm_mask_sll_epi32(src, k, a, count) easysimd_mm_mask_sll_epi32(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_sll_epi32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_sll_epi32(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    uint32_t shift = HEDLEY_STATIC_CAST(uint32_t, count.i64[0]);
    r.sve_i32 = svlsl_n_s32_z(EASYSIMD_MASK_TO_B32(k, 0), a.sve_i32, shift);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    if (count_.u64[0] > 31)
      return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_setzero_si128());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u32 = (a_.u32 << count_.u64[0]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (a_.u32[i] << count_.u64[0]));
      }
    #endif
    return easysimd_mm_maskz_mov_epi32(k, easysimd__m128i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_sll_epi32
  #define _mm_maskz_sll_epi32(k, a, count) easysimd_mm_maskz_sll_epi32(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sll_epi64 (easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sll_epi64(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    if (shift > 63) {
        return easysimd_mm512_setzero_si512();
    }

    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], shift);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    if (HEDLEY_LIKELY((count.neon_i64[0]) >= 0 && (count.neon_i64[0]) < 64)) {
        r.m128i[0].neon_i64 = vshlq_s64(a.m128i[0].neon_i64, vdupq_n_s64(count.neon_i64[0]));
        r.m128i[1].neon_i64 = vshlq_s64(a.m128i[1].neon_i64, vdupq_n_s64(count.neon_i64[0]));
        r.m128i[2].neon_i64 = vshlq_s64(a.m128i[2].neon_i64, vdupq_n_s64(count.neon_i64[0]));
        r.m128i[3].neon_i64 = vshlq_s64(a.m128i[3].neon_i64, vdupq_n_s64(count.neon_i64[0]));
    } else {
        r.m128i[0].neon_i64 = vdupq_n_s64(0);
        r.m128i[1].neon_i64 = vdupq_n_s64(0);
        r.m128i[2].neon_i64 = vdupq_n_s64(0);
        r.m128i[3].neon_i64 = vdupq_n_s64(0);
    } 
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      r_.m256i[0] = easysimd_mm256_sll_epi64(a_.m256i[0], count);
      r_.m256i[1] = easysimd_mm256_sll_epi64(a_.m256i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);
      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
      if (shift > 63)
        return easysimd_mm512_setzero_si512();

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i64 = a_.i64 << HEDLEY_STATIC_CAST(int64_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
          r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.i64[i] << (shift));
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sll_epi64
  #define _mm512_sll_epi64(a, count) easysimd_mm512_sll_epi64(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_sll_epi64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_sll_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, b.i64[0]);
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], shift), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], shift), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], shift), src.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], shift), src.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    easysimd__m128i_private
      b_ = easysimd__m128i_to_private(b);
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, b_.i64[0]);
    if (shift > 63)
      return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_setzero_si512());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i64 = a_.i64 << HEDLEY_STATIC_CAST(int64_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.i64[i] << (shift));
      }
    #endif
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd__m512i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_sll_epi64
  #define _mm512_mask_sll_epi64(src, k, a, b) easysimd_mm512_mask_sll_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_sll_epi64(easysimd__mmask8 k, easysimd__m512i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_sll_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, b.i64[0]);
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], shift);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    easysimd__m128i_private
      b_ = easysimd__m128i_to_private(b);
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, b_.i64[0]);
    if (shift > 63)
      return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_setzero_si512());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i64 = a_.i64 << HEDLEY_STATIC_CAST(int64_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.i64[i] << (shift));
      }
    #endif
    return easysimd_mm512_maskz_mov_epi64(k, easysimd__m512i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_sll_epi64
  #define _mm512_maskz_sll_epi64(k, a, b) easysimd_mm512_maskz_sll_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_sll_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_sll_epi64(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], shift), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsl_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], shift), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
    if (shift > 63)
      return easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_setzero_si256());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i64 = a_.i64 << HEDLEY_STATIC_CAST(int64_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.i64[i] << (shift));
      }
    #endif
    return easysimd_mm256_mask_mov_epi64(src, k, easysimd__m256i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_sll_epi64
  #define _mm256_mask_sll_epi64(src, k, a, count) easysimd_mm256_mask_sll_epi64(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_sll_epi64 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_sll_epi64(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
    if (shift > 63)
      return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_setzero_si256());

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i64 = a_.i64 << HEDLEY_STATIC_CAST(int64_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.i64[i] << (shift));
      }
    #endif
    return easysimd_mm256_maskz_mov_epi64(k, easysimd__m256i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_sll_epi64
  #define _mm256_maskz_sll_epi64(k, a, count) easysimd_mm256_maskz_sll_epi64(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_sll_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_sll_epi64(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b64();
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, 0), svlsl_n_s64_x(pg, a.sve_i64, shift), src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    if (count_.u64[0] > 63)
      return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_setzero_si128());

    const int_fast16_t s = HEDLEY_STATIC_CAST(int_fast16_t, count_.u64[0]);

    #if !defined(EASYSIMD_BUG_GCC_94488)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = a_.u64[i] << s;
    }

    return easysimd_mm_mask_mov_epi64(src, k, easysimd__m128i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_sll_epi64
  #define _mm_mask_sll_epi64(src, k, a, count) easysimd_mm_mask_sll_epi64(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_sll_epi64 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_sll_epi64(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    r.sve_i64 = svlsl_n_s64_z(EASYSIMD_MASK_TO_B64(k, 0), a.sve_i64, shift);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    if (count_.u64[0] > 63)
      return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_setzero_si128());

    const int_fast16_t s = HEDLEY_STATIC_CAST(int_fast16_t, count_.u64[0]);

    #if !defined(EASYSIMD_BUG_GCC_94488)
      EASYSIMD_VECTORIZE
    #endif
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = a_.u64[i] << s;
    }
    return easysimd_mm_maskz_mov_epi64(k, easysimd__m128i_from_private(r_));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_sll_epi64
  #define _mm_maskz_sll_epi64(k, a, count) easysimd_mm_maskz_sll_epi64(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_bslli_epi128 (easysimd__m512i a, const int imm8)
  EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    sveuint8_t svid = svindex_u8(imm8, 1);
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svrev_s8(svtbl_s8(svrev_s8(a.sve_i8[EASYSIMD_SV_INDEX_0]), svid));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svrev_s8(svtbl_s8(svrev_s8(a.sve_i8[EASYSIMD_SV_INDEX_1]), svid));
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svrev_s8(svtbl_s8(svrev_s8(a.sve_i8[EASYSIMD_SV_INDEX_2]), svid));
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svrev_s8(svtbl_s8(svrev_s8(a.sve_i8[EASYSIMD_SV_INDEX_3]), svid));
    return r;
  #elif 0 //defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    if (HEDLEY_LIKELY(imm8 > 0 && imm8 <= 15)) {
        r.m128i[0].neon_i8 = vextq_s8(vdupq_n_s8(0), a.m128i[0].neon_i8, 16 - imm8);
        r.m128i[1].neon_i8 = vextq_s8(vdupq_n_s8(0), a.m128i[1].neon_i8, 16 - imm8);
        r.m128i[2].neon_i8 = vextq_s8(vdupq_n_s8(0), a.m128i[2].neon_i8, 16 - imm8);
        r.m128i[3].neon_i8 = vextq_s8(vdupq_n_s8(0), a.m128i[3].neon_i8, 16 - imm8);
    } else if (imm8 == 0) {
        r = a;
    } else {
        r.m128i[0].neon_i8 = vdupq_n_s8(0);
        r.m128i[1].neon_i8 = vdupq_n_s8(0);
        r.m128i[2].neon_i8 = vdupq_n_s8(0);
        r.m128i[3].neon_i8 = vdupq_n_s8(0);
    }
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t h = 0 ; h < (sizeof(r_.m128i_private) / sizeof(r_.m128i_private[0])) ; h++) {
      for (size_t i = 0 ; i < (sizeof(r_.m128i_private[h].i8) / sizeof(r_.m128i_private[h].i8[0])) ; i++) {
        const int e = imm8 + HEDLEY_STATIC_CAST(int, i);
        r_.m128i_private[h].i8[15 - i] = ((15 - e) >= 0) ? a_.m128i_private[h].i8[15 - e] : 0;
      }
    }
    return easysimd__m512i_from_private(r_);

  #endif  
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_bslli_epi128
  #define _mm512_bslli_epi128(a, imm8) easysimd_mm512_bslli_epi128(a, imm8)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SLL_H) */
