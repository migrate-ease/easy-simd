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

#if !defined(EASYSIMD_X86_AVX512_SRL_H)
#define EASYSIMD_X86_AVX512_SRL_H

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
easysimd_mm512_srl_epi16 (easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_srl_epi16(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    if (shift > 15) {
      return easysimd_mm512_setzero_si512();
    }
    uint16_t shift_ = HEDLEY_STATIC_CAST(uint16_t, shift);

    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], shift_);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], shift_);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_srl_epi16(a_.m256i[i], count);
      }
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

      if (HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]) > 15)
        return easysimd_mm512_setzero_si512();

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.u16 = a_.u16 >> count_.i64[0];
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.u16[i] = a_.u16[i] >> count_.i64[0];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_srl_epi16
  #define _mm512_srl_epi16(a, count) easysimd_mm512_srl_epi16(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_srl_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_srl_epi16(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint16_t shift_ = HEDLEY_STATIC_CAST(uint16_t, count.u64[0]);
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_), src.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_), src.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], shift_), src.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], shift_), src.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else

    return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_srl_epi16(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_srl_epi16
  #define _mm512_mask_srl_epi16(src, k, a, count) easysimd_mm512_mask_srl_epi16(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_srl_epi16 (easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_srl_epi16(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint16_t shift_ = HEDLEY_STATIC_CAST(uint16_t, count.u64[0]);
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift_), src.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift_), src.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else

    return easysimd_mm256_mask_mov_epi16(src, k, easysimd_mm256_srl_epi16(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_srl_epi16
  #define _mm256_mask_srl_epi16(src, k, a, count) easysimd_mm256_mask_srl_epi16(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_srl_epi16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_srl_epi16(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint16_t shift_ = HEDLEY_STATIC_CAST(uint16_t, count.u64[0]);
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_u16 = svsel_u16(EASYSIMD_MASK_TO_B16(k, 0), svlsr_n_u16_z(pg, a.sve_u16, shift_), src.sve_u16);
    return r;
  #else

    return easysimd_mm_mask_mov_epi16(src, k, easysimd_mm_srl_epi16(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_srl_epi16
  #define _mm_mask_srl_epi16(src, k, a, count) easysimd_mm_mask_srl_epi16(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_srl_epi16 (easysimd__mmask32 k, easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_srl_epi16(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint16_t shift_ = HEDLEY_STATIC_CAST(uint16_t, count.u64[0]);
    easysimd__m512i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svlsr_n_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_u16[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svlsr_n_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_u16[EASYSIMD_SV_INDEX_1], shift_);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svlsr_n_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_u16[EASYSIMD_SV_INDEX_2], shift_);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svlsr_n_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_u16[EASYSIMD_SV_INDEX_3], shift_);
    return r;
  #else

    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_srl_epi16(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_srl_epi16
  #define _mm512_maskz_srl_epi16(k, a, count) easysimd_mm512_maskz_srl_epi16(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_srl_epi16 (easysimd__mmask16 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_srl_epi16(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint16_t shift_ = HEDLEY_STATIC_CAST(uint16_t, count.u64[0]);
    easysimd__m256i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svlsr_n_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_u16[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svlsr_n_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_u16[EASYSIMD_SV_INDEX_1], shift_);
    return r;
  #else

    return easysimd_mm256_maskz_mov_epi16(k, easysimd_mm256_srl_epi16(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_srl_epi16
  #define _mm256_maskz_srl_epi16(k, a, count) easysimd_mm256_maskz_srl_epi16(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_srl_epi16 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_srl_epi16(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint16_t shift_ = HEDLEY_STATIC_CAST(uint16_t, count.u64[0]);
    easysimd__m128i r;
    r.sve_u16 = svlsr_n_u16_z(EASYSIMD_MASK_TO_B16(k, 0), a.sve_u16, shift_);
    return r;
  #else

    return easysimd_mm_maskz_mov_epi16(k, easysimd_mm_srl_epi16(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_srl_epi16
  #define _mm_maskz_srl_epi16(k, a, count) easysimd_mm_maskz_srl_epi16(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_srl_epi32 (easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_srl_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    if (shift > 31) {
      return easysimd_mm512_setzero_si512();
    }
    uint32_t shift_ = HEDLEY_STATIC_CAST(uint32_t, shift);

    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], shift_);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], shift_);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_srl_epi32(a_.m256i[i], count);
      }
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

      if (HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]) > 31)
        return easysimd_mm512_setzero_si512();

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.u32 = a_.u32 >> count_.i64[0];
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.u32[i] = a_.u32[i] >> count_.i64[0];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_srl_epi32
  #define _mm512_srl_epi32(a, count) easysimd_mm512_srl_epi32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_srl_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_srl_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint32_t shift_ = HEDLEY_STATIC_CAST(uint32_t, b.u64[0]);
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_), src.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_), src.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], shift_), src.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], shift_), src.sve_u32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_srl_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_srl_epi32
  #define _mm512_mask_srl_epi32(src, k, a, b) easysimd_mm512_mask_srl_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_srl_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_srl_epi32(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint32_t shift_ = HEDLEY_STATIC_CAST(uint32_t, count.u64[0]);
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift_), src.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift_), src.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else

    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_srl_epi32(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_srl_epi32
  #define _mm256_mask_srl_epi32(src, k, a, count) easysimd_mm256_mask_srl_epi32(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_srl_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_srl_epi32(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint32_t shift_ = HEDLEY_STATIC_CAST(uint32_t, count.u64[0]);
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_u32 = svsel_u32(EASYSIMD_MASK_TO_B32(k, 0), svlsr_n_u32_z(pg, a.sve_u32, shift_), src.sve_u32);
    return r;
  #else

    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_srl_epi32(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_srl_epi32
  #define _mm_mask_srl_epi32(src, k, a, count) easysimd_mm_mask_srl_epi32(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_srl_epi32(easysimd__mmask16 k, easysimd__m512i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_srl_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint32_t shift_ = HEDLEY_STATIC_CAST(uint32_t, b.u64[0]);
    easysimd__m512i r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svlsr_n_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svlsr_n_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_u32[EASYSIMD_SV_INDEX_1], shift_);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svlsr_n_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_u32[EASYSIMD_SV_INDEX_2], shift_);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svlsr_n_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_u32[EASYSIMD_SV_INDEX_3], shift_);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_srl_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_srl_epi32
  #define _mm512_maskz_srl_epi32(k, a, b) easysimd_mm512_maskz_srl_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_srl_epi32 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_srl_epi32(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint32_t shift_ = HEDLEY_STATIC_CAST(uint32_t, count.u64[0]);
    easysimd__m256i r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svlsr_n_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svlsr_n_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_u32[EASYSIMD_SV_INDEX_1], shift_);
    return r;
  #else

    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_srl_epi32(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_srl_epi32
  #define _mm256_maskz_srl_epi32(k, a, count) easysimd_mm256_maskz_srl_epi32(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_srl_epi32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_srl_epi32(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint32_t shift_ = HEDLEY_STATIC_CAST(uint32_t, count.u64[0]);
    easysimd__m128i r;
    r.sve_u32 = svlsr_n_u32_z(EASYSIMD_MASK_TO_B32(k, 0), a.sve_u32, shift_);
    return r;
  #else

    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_srl_epi32(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_srl_epi32
  #define _mm_maskz_srl_epi32(k, a, count) easysimd_mm_maskz_srl_epi32(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_srl_epi64 (easysimd__m512i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_srl_epi64(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    if (shift > 63) {
      return easysimd_mm512_setzero_si512();
    }

    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], shift);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], shift);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], shift);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], shift);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_srl_epi64(a_.m256i[i], count);
      }
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

      if (HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]) > 63)
        return easysimd_mm512_setzero_si512();

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.u64 = a_.u64 >> count_.i64[0];
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
          r_.u64[i] = a_.u64[i] >> count_.i64[0];
        }
      #endif
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_srl_epi64
  #define _mm512_srl_epi64(a, count) easysimd_mm512_srl_epi64(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_srl_epi64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_srl_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift_ = HEDLEY_STATIC_CAST(uint64_t, b.u64[0]);
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], shift_), src.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], shift_), src.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], shift_), src.sve_u64[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], shift_), src.sve_u64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_srl_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_srl_epi64
  #define _mm512_mask_srl_epi64(src, k, a, b) easysimd_mm512_mask_srl_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_srl_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_srl_epi64(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift_ = HEDLEY_STATIC_CAST(uint64_t, count.u64[0]);
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], shift_), src.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], shift_), src.sve_u64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else

    return easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_srl_epi64(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_srl_epi64
  #define _mm256_mask_srl_epi64(src, k, a, count) easysimd_mm256_mask_srl_epi64(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_srl_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_srl_epi64(src, k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift_ = HEDLEY_STATIC_CAST(uint64_t, count.u64[0]);
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_u64 = svsel_u64(EASYSIMD_MASK_TO_B64(k, 0), svlsr_n_u64_z(pg, a.sve_u64, shift_), src.sve_u64);
    return r;
  #else

    return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_srl_epi64(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_srl_epi64
  #define _mm_mask_srl_epi64(src, k, a, count) easysimd_mm_mask_srl_epi64(src, k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_srl_epi64(easysimd__mmask8 k, easysimd__m512i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_srl_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift_ = HEDLEY_STATIC_CAST(uint64_t, b.u64[0]);
    easysimd__m512i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svlsr_n_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svlsr_n_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_u64[EASYSIMD_SV_INDEX_1], shift_);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svlsr_n_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_u64[EASYSIMD_SV_INDEX_2], shift_);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svlsr_n_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_u64[EASYSIMD_SV_INDEX_3], shift_);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_srl_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_srl_epi64
  #define _mm512_maskz_srl_epi64(k, a, b) easysimd_mm512_maskz_srl_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_srl_epi64 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_srl_epi64(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift_ = HEDLEY_STATIC_CAST(uint64_t, count.u64[0]);
    easysimd__m256i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svlsr_n_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64[EASYSIMD_SV_INDEX_0], shift_);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svlsr_n_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_u64[EASYSIMD_SV_INDEX_1], shift_);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_srl_epi64(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_srl_epi64
  #define _mm256_maskz_srl_epi64(k, a, count) easysimd_mm256_maskz_srl_epi64(k, a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_srl_epi64 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_srl_epi64(k, a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t shift_ = HEDLEY_STATIC_CAST(uint64_t, count.u64[0]);
    easysimd__m128i r;
    r.sve_u64 = svlsr_n_u64_z(EASYSIMD_MASK_TO_B64(k, 0), a.sve_u64, shift_);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_srl_epi64(a, count));

  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_srl_epi64
  #define _mm_maskz_srl_epi64(k, a, count) easysimd_mm_maskz_srl_epi64(k, a, count)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_bsrli_epi128 (easysimd__m512i a, const int imm8)
  EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    sveuint8_t svid = svindex_u8(imm8, 1);
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svtbl_s8(a.sve_i8[EASYSIMD_SV_INDEX_0], svid);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svtbl_s8(a.sve_i8[EASYSIMD_SV_INDEX_1], svid);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svtbl_s8(a.sve_i8[EASYSIMD_SV_INDEX_2], svid);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svtbl_s8(a.sve_i8[EASYSIMD_SV_INDEX_3], svid);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t h = 0 ; h < (sizeof(r_.m128i_private) / sizeof(r_.m128i_private[0])) ; h++) {
      for (size_t i = 0 ; i < (sizeof(r_.m128i_private[h].i8) / sizeof(r_.m128i_private[h].i8[0])) ; i++) {
        const int e = imm8 + HEDLEY_STATIC_CAST(int, i);
        r_.m128i_private[h].i8[i] = (e < 16) ? a_.m128i_private[h].i8[e] : 0;
      }
    }
    return easysimd__m512i_from_private(r_);

  #endif  
}
#if defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm512_bsrli_epi128(a, imm8) ({ \
    easysimd__m512i r; \
    if (HEDLEY_LIKELY(imm8 > 0 && imm8 <= 15)) { \
        r.m128i[0].neon_i8 = vextq_s8(a.m128i[0].neon_i8, vdupq_n_s8(0), imm8); \
        r.m128i[1].neon_i8 = vextq_s8(a.m128i[1].neon_i8, vdupq_n_s8(0), imm8); \
        r.m128i[2].neon_i8 = vextq_s8(a.m128i[2].neon_i8, vdupq_n_s8(0), imm8); \
        r.m128i[3].neon_i8 = vextq_s8(a.m128i[3].neon_i8, vdupq_n_s8(0), imm8); \
    } else if (imm8 == 0) { \
        r = a; \
    } else { \
      r.m128i[0].neon_i8 = vdupq_n_s8(0); \
      r.m128i[1].neon_i8 = vdupq_n_s8(0); \
      r.m128i[2].neon_i8 = vdupq_n_s8(0); \
      r.m128i[3].neon_i8 = vdupq_n_s8(0); \
    } \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_bsrli_epi128
  #define _mm512_bsrli_epi128(a, imm8) easysimd_mm512_bsrli_epi128(a, imm8)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SRL_H) */
