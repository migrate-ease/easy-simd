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
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 */

#if !defined(EASYSIMD_X86_AVX512_ADDS_H)
#define EASYSIMD_X86_AVX512_ADDS_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_adds_epi8(easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_adds_epi8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svqadd_s8_z(pg, a.sve_i8, b.sve_i8), src.sve_i8);
    return r;
  #else
    return easysimd_mm_mask_mov_epi8(src, k, easysimd_mm_adds_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_adds_epi8
  #define _mm_mask_adds_epi8(src, k, a, b) easysimd_mm_mask_adds_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_adds_epi8(easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_adds_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svqadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8, b.sve_i8);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi8(k, easysimd_mm_adds_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_adds_epi8
  #define _mm_maskz_adds_epi8(k, a, b) easysimd_mm_maskz_adds_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_adds_epu8(easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_adds_epu8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_u8 = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svqadd_u8_z(pg, a.sve_u8, b.sve_u8), src.sve_u8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = ((k >> i) & 1) ? easysimd_math_adds_u8(a_.u8[i], b_.u8[i]) : src_.u8[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_adds_epu8
  #define _mm_mask_adds_epu8(src, k, a, b) easysimd_mm_mask_adds_epu8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_adds_epu8(easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_adds_epu8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svqadd_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_u8, b.sve_u8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = ((k >> i) & 1) ? easysimd_math_adds_u8(a_.u8[i], b_.u8[i]) : UINT8_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_adds_epu8
  #define _mm_maskz_adds_epu8(k, a, b) easysimd_mm_maskz_adds_epu8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_adds_epi16(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_adds_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svqadd_s16_z(pg, a.sve_i16, b.sve_i16), src.sve_i16);
    return r;
  #else
    return easysimd_mm_mask_mov_epi16(src, k, easysimd_mm_adds_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_adds_epi16
  #define _mm_mask_adds_epi16(src, k, a, b) easysimd_mm_mask_adds_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_adds_epi16(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_adds_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svqadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16, b.sve_i16);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi16(k, easysimd_mm_adds_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_adds_epi16
  #define _mm_maskz_adds_epi16(k, a, b) easysimd_mm_maskz_adds_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_adds_epu16(easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_adds_epu16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_u16 = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svqadd_u16_z(pg, a.sve_u16, b.sve_u16), src.sve_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = ((k >> i) & 1) ? easysimd_math_adds_u16(a_.u16[i], b_.u16[i]) : src_.u16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_adds_epu16
  #define _mm_mask_adds_epu16(src, k, a, b) easysimd_mm_mask_adds_epu16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_adds_epu16(easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_adds_epu16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svqadd_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_u16, b.sve_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = ((k >> i) & 1) ? easysimd_math_adds_u16(a_.u16[i], b_.u16[i]) : UINT16_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_adds_epu16
  #define _mm_maskz_adds_epu16(k, a, b) easysimd_mm_maskz_adds_epu16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_adds_epi8(easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_adds_epi8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svqadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svqadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), src.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi8(src, k, easysimd_mm256_adds_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_adds_epi8
  #define _mm256_mask_adds_epi8(src, k, a, b) easysimd_mm256_mask_adds_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_adds_epi8(easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_maskz_adds_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svqadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svqadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi8(k, easysimd_mm256_adds_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_adds_epi8
  #define _mm256_maskz_adds_epi8(k, a, b) easysimd_mm256_maskz_adds_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_adds_epu8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_adds_epu8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svqadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), src.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svqadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), src.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = ((k >> i) & 1) ? easysimd_math_adds_u8(a_.u8[i], b_.u8[i]) : src_.u8[i];
      }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_adds_epu8
  #define _mm256_mask_adds_epu8(src, k, a, b) easysimd_mm256_mask_adds_epu8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_adds_epu8 (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_adds_epu8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svqadd_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svqadd_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = ((k >> i) & 1) ? easysimd_math_adds_u8(a_.u8[i], b_.u8[i]) : UINT8_C(0);
      }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_adds_epu8
  #define _mm256_maskz_adds_epu8(k, a, b) easysimd_mm256_maskz_adds_epu8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_adds_epi16(easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_adds_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svqadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svqadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi16(src, k, easysimd_mm256_adds_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_adds_epi16
  #define _mm256_mask_adds_epi16(src, k, a, b) easysimd_mm256_mask_adds_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_adds_epi16(easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_maskz_adds_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svqadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svqadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi16(k, easysimd_mm256_adds_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_adds_epi16
  #define _mm256_maskz_adds_epi16(k, a, b) easysimd_mm256_maskz_adds_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_adds_epu16(easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_adds_epu16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svqadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), src.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svqadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), src.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = ((k >> i) & 1) ? easysimd_math_adds_u16(a_.u16[i], b_.u16[i]) : src_.u16[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_adds_epu16
  #define _mm256_mask_adds_epu16(src, k, a, b) easysimd_mm256_mask_adds_epu16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_adds_epu16(easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_adds_epu16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svqadd_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svqadd_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = ((k >> i) & 1) ? easysimd_math_adds_u16(a_.u16[i], b_.u16[i]) : UINT16_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_adds_epu16
  #define _mm256_maskz_adds_epu16(k, a, b) easysimd_mm256_maskz_adds_epu16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_adds_epi8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_adds_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svqadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svqadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svqadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svqadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i8 = vqaddq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vqaddq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    r.m128i[2].neon_i8 = vqaddq_s8(a.m128i[2].neon_i8, b.m128i[2].neon_i8);
    r.m128i[3].neon_i8 = vqaddq_s8(a.m128i[3].neon_i8, b.m128i[3].neon_i8);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if !defined(HEDLEY_INTEL_VERSION)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_adds_epi8(a_.m256i[i], b_.m256i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = easysimd_math_adds_i8(a_.i8[i], b_.i8[i]);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_adds_epi8
  #define _mm512_adds_epi8(a, b) easysimd_mm512_adds_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_adds_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_adds_epi8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svqadd_s8(a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svqadd_s8(a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), src.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), svqadd_s8(a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), src.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), svqadd_s8(a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), src.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_adds_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_adds_epi8
  #define _mm512_mask_adds_epi8(src, k, a, b) easysimd_mm512_mask_adds_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_adds_epi8 (easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_adds_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svqadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svqadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svqadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svqadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_adds_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_adds_epi8
  #define _mm512_maskz_adds_epi8(k, a, b) easysimd_mm512_maskz_adds_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_adds_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_adds_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svqadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svqadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svqadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svqadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i16 = vqaddq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vqaddq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    r.m128i[2].neon_i16 = vqaddq_s16(a.m128i[2].neon_i16, b.m128i[2].neon_i16);
    r.m128i[3].neon_i16 = vqaddq_s16(a.m128i[3].neon_i16, b.m128i[3].neon_i16);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if !defined(HEDLEY_INTEL_VERSION)
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
          r_.m256i[i] = easysimd_mm256_adds_epi16(a_.m256i[i], b_.m256i[i]);
        }
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = easysimd_math_adds_i16(a_.i16[i], b_.i16[i]);
        }
      #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_adds_epi16
  #define _mm512_adds_epi16(a, b) easysimd_mm512_adds_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_adds_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_adds_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svqadd_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svqadd_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svqadd_s16(a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]), src.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svqadd_s16(a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]), src.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_adds_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_adds_epi16
  #define _mm512_mask_adds_epi16(src, k, a, b) easysimd_mm512_mask_adds_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_adds_epi16 (easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_adds_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svqadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svqadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svqadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svqadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_adds_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_adds_epi16
  #define _mm512_maskz_adds_epi16(k, a, b) easysimd_mm512_maskz_adds_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_adds_epu8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_adds_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svqadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svqadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svqadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]);
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svqadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u8 = vqaddq_u8(a.m128i[0].neon_u8, b.m128i[0].neon_u8);
    r.m128i[1].neon_u8 = vqaddq_u8(a.m128i[1].neon_u8, b.m128i[1].neon_u8);
    r.m128i[2].neon_u8 = vqaddq_u8(a.m128i[2].neon_u8, b.m128i[2].neon_u8);
    r.m128i[3].neon_u8 = vqaddq_u8(a.m128i[3].neon_u8, b.m128i[3].neon_u8);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if !defined(HEDLEY_INTEL_VERSION)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
        r_.m128i[i] = easysimd_mm_adds_epu8(a_.m128i[i], b_.m128i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = easysimd_math_adds_u8(a_.u8[i], b_.u8[i]);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_adds_epu8
  #define _mm512_adds_epu8(a, b) easysimd_mm512_adds_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_adds_epu8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_adds_epu8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svqadd_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), src.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svqadd_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), src.sve_u8[EASYSIMD_SV_INDEX_1]);
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), svqadd_u8(a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]), src.sve_u8[EASYSIMD_SV_INDEX_2]);
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), svqadd_u8(a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]), src.sve_u8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_adds_epu8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_adds_epu8
  #define _mm512_mask_adds_epu8(src, k, a, b) easysimd_mm512_mask_adds_epu8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_adds_epu8 (easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_adds_epu8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svqadd_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svqadd_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svqadd_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]);
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svqadd_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_adds_epu8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_adds_epu8
  #define _mm512_maskz_adds_epu8(k, a, b) easysimd_mm512_maskz_adds_epu8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_adds_epu16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_adds_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svqadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svqadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svqadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svqadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u16 = vqaddq_u16(a.m128i[0].neon_u16, b.m128i[0].neon_u16);
    r.m128i[1].neon_u16 = vqaddq_u16(a.m128i[1].neon_u16, b.m128i[1].neon_u16);
    r.m128i[2].neon_u16 = vqaddq_u16(a.m128i[2].neon_u16, b.m128i[2].neon_u16);
    r.m128i[3].neon_u16 = vqaddq_u16(a.m128i[3].neon_u16, b.m128i[3].neon_u16);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if !defined(HEDLEY_INTEL_VERSION)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_adds_epu16(a_.m256i[i], b_.m256i[i]);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = easysimd_math_adds_u16(a_.u16[i], b_.u16[i]);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_adds_epu16
  #define _mm512_adds_epu16(a, b) easysimd_mm512_adds_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_adds_epu16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_adds_epu16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svqadd_u16(a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), src.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svqadd_u16(a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), src.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svqadd_u16(a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), src.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svqadd_u16(a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), src.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_adds_epu16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_adds_epu16
  #define _mm512_mask_adds_epu16(src, k, a, b) easysimd_mm512_mask_adds_epu16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_adds_epu16 (easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_adds_epu16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svqadd_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svqadd_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svqadd_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svqadd_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_adds_epu16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_adds_epu16
  #define _mm512_maskz_adds_epu16(k, a, b) easysimd_mm512_maskz_adds_epu16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_adds_epi32(easysimd__m128i a, easysimd__m128i b) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    r_.neon_i32 = vqaddq_s32(a_.neon_i32, b_.neon_i32);
  #else
    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      /* https://stackoverflow.com/a/56544654/501126 */
      const __m128i int_max = _mm_set1_epi32(INT32_MAX);

      /* normal result (possibly wraps around) */
      const __m128i sum = _mm_add_epi32(a_.ni, b_.ni);

      /* If result saturates, it has the same sign as both a and b */
      const __m128i sign_bit = _mm_srli_epi32(a_.ni, 31); /* shift sign to lowest bit */

      #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
        const __m128i overflow = _mm_ternarylogic_epi32(a_.ni, b_.ni, sum, 0x42);
      #else
        const __m128i sign_xor = _mm_xor_si128(a_.ni, b_.ni);
        const __m128i overflow = _mm_andnot_si128(sign_xor, _mm_xor_si128(a_.ni, sum));
      #endif

      #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
        r_.ni = _mm_mask_add_epi32(sum, _mm_movepi32_mask(overflow), int_max, sign_bit);
      #else
        const __m128i saturated = _mm_add_epi32(int_max, sign_bit);

        #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
          r_.ni =
            _mm_castps_si128(
              _mm_blendv_ps(
                _mm_castsi128_ps(sum),
                _mm_castsi128_ps(saturated),
                _mm_castsi128_ps(overflow)
              )
            );
        #else
          const __m128i overflow_mask = _mm_srai_epi32(overflow, 31);
          r_.ni =
            _mm_or_si128(
              _mm_and_si128(overflow_mask, saturated),
              _mm_andnot_si128(overflow_mask, sum)
            );
        #endif
      #endif
    #elif defined(EASYSIMD_VECTOR_SCALAR)
      uint32_t au EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.i32);
      uint32_t bu EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.i32);
      uint32_t ru EASYSIMD_VECTOR(16) = au + bu;

      au = (au >> 31) + INT32_MAX;

      uint32_t m EASYSIMD_VECTOR(16) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (au ^ bu) | ~(bu ^ ru)) < 0);
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (au & ~m) | (ru & m));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = easysimd_math_adds_i32(a_.i32[i], b_.i32[i]);
      }
    #endif
  #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_adds_epi32(easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
      r_.m128i[i] = easysimd_x_mm_adds_epi32(a_.m128i[i], b_.m128i[i]);
    }
  #elif defined(EASYSIMD_VECTOR_SCALAR)
    uint32_t au EASYSIMD_VECTOR(32) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.i32);
    uint32_t bu EASYSIMD_VECTOR(32) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.i32);
    uint32_t ru EASYSIMD_VECTOR(32) = au + bu;

    au = (au >> 31) + INT32_MAX;

    uint32_t m EASYSIMD_VECTOR(32) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (au ^ bu) | ~(bu ^ ru)) < 0);
    r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (au & ~m) | (ru & m));
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = easysimd_math_adds_i32(a_.i32[i], b_.i32[i]);
    }
  #endif

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_x_mm512_adds_epi32(easysimd__m512i a, easysimd__m512i b) {
  easysimd__m512i_private
    r_,
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    for (size_t i = 0 ; i < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; i++) {
      r_.m128i[i] = easysimd_x_mm_adds_epi32(a_.m128i[i], b_.m128i[i]);
    }
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
    for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
      r_.m256i[i] = easysimd_x_mm256_adds_epi32(a_.m256i[i], b_.m256i[i]);
    }
  #elif defined(EASYSIMD_VECTOR_SCALAR)
    uint32_t au EASYSIMD_VECTOR(64) = HEDLEY_REINTERPRET_CAST(__typeof__(au), a_.i32);
    uint32_t bu EASYSIMD_VECTOR(64) = HEDLEY_REINTERPRET_CAST(__typeof__(bu), b_.i32);
    uint32_t ru EASYSIMD_VECTOR(64) = au + bu;

    au = (au >> 31) + INT32_MAX;

    uint32_t m EASYSIMD_VECTOR(64) = HEDLEY_REINTERPRET_CAST(__typeof__(m), HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (au ^ bu) | ~(bu ^ ru)) < 0);
    r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (au & ~m) | (ru & m));
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = easysimd_math_adds_i32(a_.i32[i], b_.i32[i]);
    }
  #endif

  return easysimd__m512i_from_private(r_);
}

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_ADDS_H) */
