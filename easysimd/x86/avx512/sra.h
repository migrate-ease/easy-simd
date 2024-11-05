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

#if !defined(EASYSIMD_X86_AVX512_SRA_H)
#define EASYSIMD_X86_AVX512_SRA_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sra_epi16 (easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_sra_epi16(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    uint64_t shift = (uint64_t )count.i64[0];
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svasr_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svasr_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], shift);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svasr_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], shift);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svasr_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], shift);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_sra_epi16(a_.m256i[i], count);
      }
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);
      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);

      if (shift > 15) shift = 15;

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i16 = a_.i16 >> HEDLEY_STATIC_CAST(int16_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = a_.i16[i] >> shift;
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sra_epi16
  #define _mm512_sra_epi16(a, count) easysimd_mm512_sra_epi16(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sra_epi32 (easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sra_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    uint64_t shift = (uint64_t )count.i64[0];
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svasr_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svasr_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], shift);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svasr_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], shift);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svasr_n_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], shift);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_sra_epi32(a_.m256i[i], count);
      }
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);
      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);

      if (shift > 31) shift = 31;

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i32 = a_.i32 >> HEDLEY_STATIC_CAST(int32_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.i32[i] = a_.i32[i] >> shift;
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sra_epi32
  #define _mm512_sra_epi32(a, count) easysimd_mm512_sra_epi32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_sra_epi64 (easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_sra_epi64(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    uint64_t shift = (uint64_t )count.i64[0];
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svasr_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svasr_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svasr_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], shift);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svasr_n_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], shift);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);
      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);

      if (shift > 63) shift = 63;

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i64 = a_.i64 >> HEDLEY_STATIC_CAST(int64_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
          r_.i64[i] = a_.i64[i] >> shift;
        }
      #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_sra_epi64
  #define _mm512_sra_epi64(a, count) easysimd_mm512_sra_epi64(a, count)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SRA_H) */
