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

#if !defined(EASYSIMD_X86_AVX512_PERMUTEX2VAR_H)
#define EASYSIMD_X86_AVX512_PERMUTEX2VAR_H

#include "types.h"
#include "and.h"
#include "andnot.h"
#include "blend.h"
#include "mov.h"
#include "or.h"
#include "set1.h"
#include "slli.h"
#include "srli.h"
#include "test.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

/* The following generic code avoids many, nearly identical, repetitions of fairly complex code.
 * If the compiler optimizes well, in particular extracting invariant code from loops
 * and simplifying code involving constants passed as arguments, it should not be
 * significantly slower than specific code.
 * Note that when the original vector contains few elements, these implementations
 * may not be faster than portable code.
 */
#if defined(EASYSIMD_X86_SSSE3_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
 #define EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC
#endif

#if defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC)
EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_permutex2var128 (const easysimd__m128i *a, const easysimd__m128i idx, const easysimd__m128i *b, const unsigned int log2_index_size, const unsigned int log2_data_length) {
  const int idx_mask = (1 << (5 - log2_index_size + log2_data_length)) - 1;

  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    __m128i ra, rb, t, test, select, index;
    const __m128i sixteen = _mm_set1_epi8(16);

    /* Avoid the mullo intrinsics which have high latency (and the 32-bit one requires SSE4.1) */
    switch (log2_index_size) {
    default:  /* Avoid uninitialized variable warning/error */
    case 0:
      index = _mm_and_si128(idx, _mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, idx_mask)));
      break;
    case 1:
      index = _mm_and_si128(idx, _mm_set1_epi16(HEDLEY_STATIC_CAST(int16_t, idx_mask)));
      index = _mm_slli_epi32(index, 1);
      t = _mm_slli_epi32(index, 8);
      index = _mm_or_si128(index, t);
      index = _mm_add_epi16(index, _mm_set1_epi16(0x0100));
      break;
    case 2:
      index = _mm_and_si128(idx, _mm_set1_epi32(HEDLEY_STATIC_CAST(int32_t, idx_mask)));
      index = _mm_slli_epi32(index, 2);
      t = _mm_slli_epi32(index, 8);
      index = _mm_or_si128(index, t);
      t = _mm_slli_epi32(index, 16);
      index = _mm_or_si128(index, t);
      index = _mm_add_epi32(index, _mm_set1_epi32(0x03020100));
      break;
    }

    test = index;
    index = _mm_and_si128(index, _mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, (1 << (4 + log2_data_length)) - 1)));
    test = _mm_cmpgt_epi8(test, index);

    ra = _mm_shuffle_epi8(a[0], index);
    rb = _mm_shuffle_epi8(b[0], index);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      EASYSIMD_VECTORIZE
      for (int i = 1 ; i < (1 << log2_data_length) ; i++) {
        select = _mm_cmplt_epi8(index, sixteen);
        index = _mm_sub_epi8(index, sixteen);
        ra = _mm_blendv_epi8(_mm_shuffle_epi8(a[i], index), ra, select);
        rb = _mm_blendv_epi8(_mm_shuffle_epi8(b[i], index), rb, select);
      }

      return _mm_blendv_epi8(ra, rb, test);
    #else
      EASYSIMD_VECTORIZE
      for (int i = 1 ; i < (1 << log2_data_length) ; i++) {
        select = _mm_cmplt_epi8(index, sixteen);
        index = _mm_sub_epi8(index, sixteen);
        ra = _mm_or_si128(_mm_andnot_si128(select, _mm_shuffle_epi8(a[i], index)), _mm_and_si128(select, ra));
        rb = _mm_or_si128(_mm_andnot_si128(select, _mm_shuffle_epi8(b[i], index)), _mm_and_si128(select, rb));
      }

      return _mm_or_si128(_mm_andnot_si128(test, ra), _mm_and_si128(test, rb));
    #endif
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint8x16_t index, r;
    uint16x8_t index16;
    uint32x4_t index32;
    uint8x16x2_t table2_a, table2_b;
    uint8x16x4_t table4_a, table4_b;

    switch (log2_index_size) {
    case 0:
      index = vandq_u8(easysimd__m128i_to_neon_u8(idx), vdupq_n_u8(HEDLEY_STATIC_CAST(uint8_t, idx_mask)));
      break;
    case 1:
      index16 = vandq_u16(easysimd__m128i_to_neon_u16(idx), vdupq_n_u16(HEDLEY_STATIC_CAST(uint16_t, idx_mask)));
      index16 = vmulq_n_u16(index16, 0x0202);
      index16 = vaddq_u16(index16, vdupq_n_u16(0x0100));
      index = vreinterpretq_u8_u16(index16);
      break;
    case 2:
      index32 = vandq_u32(easysimd__m128i_to_neon_u32(idx), vdupq_n_u32(HEDLEY_STATIC_CAST(uint32_t, idx_mask)));
      index32 = vmulq_n_u32(index32, 0x04040404);
      index32 = vaddq_u32(index32, vdupq_n_u32(0x03020100));
      index = vreinterpretq_u8_u32(index32);
      break;
    }

    uint8x16_t mask = vdupq_n_u8(HEDLEY_STATIC_CAST(uint8_t, (1 << (4 + log2_data_length)) - 1));

    switch (log2_data_length) {
    case 0:
      r = vqtbx1q_u8(vqtbl1q_u8(easysimd__m128i_to_neon_u8(b[0]), vandq_u8(index, mask)), easysimd__m128i_to_neon_u8(a[0]), index);
      break;
    case 1:
      table2_a.val[0] = easysimd__m128i_to_neon_u8(a[0]);
      table2_a.val[1] = easysimd__m128i_to_neon_u8(a[1]);
      table2_b.val[0] = easysimd__m128i_to_neon_u8(b[0]);
      table2_b.val[1] = easysimd__m128i_to_neon_u8(b[1]);
      r = vqtbx2q_u8(vqtbl2q_u8(table2_b, vandq_u8(index, mask)), table2_a, index);
      break;
    case 2:
      table4_a.val[0] = easysimd__m128i_to_neon_u8(a[0]);
      table4_a.val[1] = easysimd__m128i_to_neon_u8(a[1]);
      table4_a.val[2] = easysimd__m128i_to_neon_u8(a[2]);
      table4_a.val[3] = easysimd__m128i_to_neon_u8(a[3]);
      table4_b.val[0] = easysimd__m128i_to_neon_u8(b[0]);
      table4_b.val[1] = easysimd__m128i_to_neon_u8(b[1]);
      table4_b.val[2] = easysimd__m128i_to_neon_u8(b[2]);
      table4_b.val[3] = easysimd__m128i_to_neon_u8(b[3]);
      r = vqtbx4q_u8(vqtbl4q_u8(table4_b, vandq_u8(index, mask)), table4_a, index);
      break;
    }

    return easysimd__m128i_from_neon_u8(r);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_x_permutex2var (easysimd__m128i *r, const easysimd__m128i *a, const easysimd__m128i *idx, const easysimd__m128i *b, const unsigned int log2_index_size, const unsigned int log2_data_length) {
  EASYSIMD_VECTORIZE
  for (int i = 0 ; i < (1 << log2_data_length) ; i++) {
    r[i] = easysimd_x_permutex2var128(a, idx[i], b, log2_index_size, log2_data_length);
  }
}
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_permutex2var_epi16 (easysimd__m128i a, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_permutex2var_epi16(a, idx, b);
  #elif defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC)
    easysimd__m128i r;

    easysimd_x_permutex2var(&r, &a, &idx, &b, 1, 0);

    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      idx_ = easysimd__m128i_to_private(idx),
      b_ = easysimd__m128i_to_private(b),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((idx_.i16[i] & 8) ? b_ : a_).i16[idx_.i16[i] & 7];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutex2var_epi16
  #define _mm_permutex2var_epi16(a, idx, b) easysimd_mm_permutex2var_epi16(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_permutex2var_epi16 (easysimd__m128i a, easysimd__mmask8 k, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_permutex2var_epi16(a, k, idx, b);
  #else
    return easysimd_mm_mask_mov_epi16(a, k, easysimd_mm_permutex2var_epi16(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permutex2var_epi16
#define _mm_mask_permutex2var_epi16(a, k, idx, b) easysimd_mm_mask_permutex2var_epi16(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask2_permutex2var_epi16 (easysimd__m128i a, easysimd__m128i idx, easysimd__mmask8 k, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask2_permutex2var_epi16(a, idx, k, b);
  #else
    return easysimd_mm_mask_mov_epi16(idx, k, easysimd_mm_permutex2var_epi16(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask2_permutex2var_epi16
#define _mm_mask2_permutex2var_epi16(a, idx, k, b) easysimd_mm_mask2_permutex2var_epi16(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_permutex2var_epi16 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_permutex2var_epi16(k, a, idx, b);
  #else
    return easysimd_mm_maskz_mov_epi16(k, easysimd_mm_permutex2var_epi16(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permutex2var_epi16
#define _mm_maskz_permutex2var_epi16(k, a, idx, b) easysimd_mm_maskz_permutex2var_epi16(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_permutex2var_epi32 (easysimd__m128i a, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_permutex2var_epi32(a, idx, b);
  #elif defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC) /* This may not be faster than the portable version */
    easysimd__m128i r;

    easysimd_x_permutex2var(&r, &a, &idx, &b, 2, 0);

    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      idx_ = easysimd__m128i_to_private(idx),
      b_ = easysimd__m128i_to_private(b),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((idx_.i32[i] & 4) ? b_ : a_).i32[idx_.i32[i] & 3];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutex2var_epi32
  #define _mm_permutex2var_epi32(a, idx, b) easysimd_mm_permutex2var_epi32(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_permutex2var_epi32 (easysimd__m128i a, easysimd__mmask8 k, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_permutex2var_epi32(a, k, idx, b);
  #else
    return easysimd_mm_mask_mov_epi32(a, k, easysimd_mm_permutex2var_epi32(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permutex2var_epi32
#define _mm_mask_permutex2var_epi32(a, k, idx, b) easysimd_mm_mask_permutex2var_epi32(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask2_permutex2var_epi32 (easysimd__m128i a, easysimd__m128i idx, easysimd__mmask8 k, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask2_permutex2var_epi32(a, idx, k, b);
  #else
    return easysimd_mm_mask_mov_epi32(idx, k, easysimd_mm_permutex2var_epi32(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask2_permutex2var_epi32
#define _mm_mask2_permutex2var_epi32(a, idx, k, b) easysimd_mm_mask2_permutex2var_epi32(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_permutex2var_epi32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_permutex2var_epi32(k, a, idx, b);
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_permutex2var_epi32(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permutex2var_epi32
#define _mm_maskz_permutex2var_epi32(k, a, idx, b) easysimd_mm_maskz_permutex2var_epi32(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_permutex2var_epi64 (easysimd__m128i a, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_permutex2var_epi64(a, idx, b);
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      idx_ = easysimd__m128i_to_private(idx),
      b_ = easysimd__m128i_to_private(b),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((idx_.i64[i] & 2) ? b_ : a_).i64[idx_.i64[i] & 1];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutex2var_epi64
  #define _mm_permutex2var_epi64(a, idx, b) easysimd_mm_permutex2var_epi64(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_permutex2var_epi64 (easysimd__m128i a, easysimd__mmask8 k, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_permutex2var_epi64(a, k, idx, b);
  #else
    return easysimd_mm_mask_mov_epi64(a, k, easysimd_mm_permutex2var_epi64(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permutex2var_epi64
#define _mm_mask_permutex2var_epi64(a, k, idx, b) easysimd_mm_mask_permutex2var_epi64(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask2_permutex2var_epi64 (easysimd__m128i a, easysimd__m128i idx, easysimd__mmask8 k, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask2_permutex2var_epi64(a, idx, k, b);
  #else
    return easysimd_mm_mask_mov_epi64(idx, k, easysimd_mm_permutex2var_epi64(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask2_permutex2var_epi64
#define _mm_mask2_permutex2var_epi64(a, idx, k, b) easysimd_mm_mask2_permutex2var_epi64(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_permutex2var_epi64 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_permutex2var_epi64(k, a, idx, b);
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_permutex2var_epi64(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permutex2var_epi64
#define _mm_maskz_permutex2var_epi64(k, a, idx, b) easysimd_mm_maskz_permutex2var_epi64(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_permutex2var_epi8 (easysimd__m128i a, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_permutex2var_epi8(a, idx, b);
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cvtepi32_epi8(_mm512_permutex2var_epi32(_mm512_cvtepu8_epi32(a), _mm512_cvtepu8_epi32(idx), _mm512_cvtepu8_epi32(b)));
  #elif defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC)
    easysimd__m128i r;

    easysimd_x_permutex2var(&r, &a, &idx, &b, 0, 0);

    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      idx_ = easysimd__m128i_to_private(idx),
      b_ = easysimd__m128i_to_private(b),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((idx_.i8[i] & 0x10) ? b_ : a_).i8[idx_.i8[i] & 0x0F];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutex2var_epi8
  #define _mm_permutex2var_epi8(a, idx, b) easysimd_mm_permutex2var_epi8(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_permutex2var_epi8 (easysimd__m128i a, easysimd__mmask16 k, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_permutex2var_epi8(a, k, idx, b);
  #else
    return easysimd_mm_mask_mov_epi8(a, k, easysimd_mm_permutex2var_epi8(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permutex2var_epi8
#define _mm_mask_permutex2var_epi8(a, k, idx, b) easysimd_mm_mask_permutex2var_epi8(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask2_permutex2var_epi8 (easysimd__m128i a, easysimd__m128i idx, easysimd__mmask16 k, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask2_permutex2var_epi8(a, idx, k, b);
  #else
    return easysimd_mm_mask_mov_epi8(idx, k, easysimd_mm_permutex2var_epi8(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask2_permutex2var_epi8
#define _mm_mask2_permutex2var_epi8(a, idx, k, b) easysimd_mm_mask2_permutex2var_epi8(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_permutex2var_epi8 (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i idx, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_permutex2var_epi8(k, a, idx, b);
  #else
    return easysimd_mm_maskz_mov_epi8(k, easysimd_mm_permutex2var_epi8(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permutex2var_epi8
#define _mm_maskz_permutex2var_epi8(k, a, idx, b) easysimd_mm_maskz_permutex2var_epi8(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_permutex2var_pd (easysimd__m128d a, easysimd__m128i idx, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_permutex2var_pd(a, idx, b);
  #else
    return easysimd_mm_castsi128_pd(easysimd_mm_permutex2var_epi64(easysimd_mm_castpd_si128(a), idx, easysimd_mm_castpd_si128(b)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutex2var_pd
  #define _mm_permutex2var_pd(a, idx, b) easysimd_mm_permutex2var_pd(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_permutex2var_pd (easysimd__m128d a, easysimd__mmask8 k, easysimd__m128i idx, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_permutex2var_pd(a, k, idx, b);
  #else
    return easysimd_mm_mask_mov_pd(a, k, easysimd_mm_permutex2var_pd(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permutex2var_pd
#define _mm_mask_permutex2var_pd(a, k, idx, b) easysimd_mm_mask_permutex2var_pd(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask2_permutex2var_pd (easysimd__m128d a, easysimd__m128i idx, easysimd__mmask8 k, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask2_permutex2var_pd(a, idx, k, b);
  #else
    return easysimd_mm_mask_mov_pd(easysimd_mm_castsi128_pd(idx), k, easysimd_mm_permutex2var_pd(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask2_permutex2var_pd
#define _mm_mask2_permutex2var_pd(a, idx, k, b) easysimd_mm_mask2_permutex2var_pd(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_permutex2var_pd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128i idx, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_permutex2var_pd(k, a, idx, b);
  #else
    return easysimd_mm_maskz_mov_pd(k, easysimd_mm_permutex2var_pd(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permutex2var_pd
#define _mm_maskz_permutex2var_pd(k, a, idx, b) easysimd_mm_maskz_permutex2var_pd(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_permutex2var_ps (easysimd__m128 a, easysimd__m128i idx, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_permutex2var_ps(a, idx, b);
  #else
    return easysimd_mm_castsi128_ps(easysimd_mm_permutex2var_epi32(easysimd_mm_castps_si128(a), idx, easysimd_mm_castps_si128(b)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutex2var_ps
  #define _mm_permutex2var_ps(a, idx, b) easysimd_mm_permutex2var_ps(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_permutex2var_ps (easysimd__m128 a, easysimd__mmask8 k, easysimd__m128i idx, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_permutex2var_ps(a, k, idx, b);
  #else
    return easysimd_mm_mask_mov_ps(a, k, easysimd_mm_permutex2var_ps(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_permutex2var_ps
#define _mm_mask_permutex2var_ps(a, k, idx, b) easysimd_mm_mask_permutex2var_ps(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask2_permutex2var_ps (easysimd__m128 a, easysimd__m128i idx, easysimd__mmask8 k, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask2_permutex2var_ps(a, idx, k, b);
  #else
    return easysimd_mm_mask_mov_ps(easysimd_mm_castsi128_ps(idx), k, easysimd_mm_permutex2var_ps(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask2_permutex2var_ps
#define _mm_mask2_permutex2var_ps(a, idx, k, b) easysimd_mm_mask2_permutex2var_ps(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_permutex2var_ps (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128i idx, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_permutex2var_ps(k, a, idx, b);
  #else
    return easysimd_mm_maskz_mov_ps(k, easysimd_mm_permutex2var_ps(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_permutex2var_ps
#define _mm_maskz_permutex2var_ps(k, a, idx, b) easysimd_mm_maskz_permutex2var_ps(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permutex2var_epi16 (easysimd__m256i a, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutex2var_epi16(a, idx, b);
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    __m256i hilo, hilo2, hi, lo, idx2, ta, tb, select;
    const __m256i ones = _mm256_set1_epi16(1);

    idx2 = _mm256_srli_epi32(idx, 1);

    ta = _mm256_permutevar8x32_epi32(a, idx2);
    tb = _mm256_permutevar8x32_epi32(b, idx2);
    select = _mm256_slli_epi32(idx2, 28);
    hilo = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ta),
                                                _mm256_castsi256_ps(tb),
                                                _mm256_castsi256_ps(select)));
    idx2 = _mm256_srli_epi32(idx2, 16);

    ta = _mm256_permutevar8x32_epi32(a, idx2);
    tb = _mm256_permutevar8x32_epi32(b, idx2);
    select = _mm256_slli_epi32(idx2, 28);
    hilo2 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ta),
                                                 _mm256_castsi256_ps(tb),
                                                 _mm256_castsi256_ps(select)));

    lo = HEDLEY_REINTERPRET_CAST(__m256i, _mm256_blend_epi16(_mm256_slli_epi32(hilo2, 16), hilo, 0x55));
    hi = HEDLEY_REINTERPRET_CAST(__m256i, _mm256_blend_epi16(hilo2, _mm256_srli_epi32(hilo, 16), 0x55));

    select = _mm256_cmpeq_epi16(_mm256_and_si256(idx, ones), ones);
    return _mm256_blendv_epi8(lo, hi, select);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      idx_ = easysimd__m256i_to_private(idx),
      b_ = easysimd__m256i_to_private(b),
      r_;

    #if defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC)
      easysimd_x_permutex2var(r_.m128i, a_.m128i, idx_.m128i, b_.m128i, 1, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = ((idx_.i16[i] & 0x10) ? b_ : a_).i16[idx_.i16[i] & 0x0F];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutex2var_epi16
  #define _mm256_permutex2var_epi16(a, idx, b) easysimd_mm256_permutex2var_epi16(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_permutex2var_epi16 (easysimd__m256i a, easysimd__mmask16 k, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutex2var_epi16(a, k, idx, b);
  #else
    return easysimd_mm256_mask_mov_epi16(a, k, easysimd_mm256_permutex2var_epi16(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutex2var_epi16
#define _mm256_mask_permutex2var_epi16(a, k, idx, b) easysimd_mm256_mask_permutex2var_epi16(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask2_permutex2var_epi16 (easysimd__m256i a, easysimd__m256i idx, easysimd__mmask16 k, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask2_permutex2var_epi16(a, idx, k, b);
  #else
    return easysimd_mm256_mask_mov_epi16(idx, k, easysimd_mm256_permutex2var_epi16(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask2_permutex2var_epi16
#define _mm256_mask2_permutex2var_epi16(a, idx, k, b) easysimd_mm256_mask2_permutex2var_epi16(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_permutex2var_epi16 (easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutex2var_epi16(k, a, idx, b);
  #else
    return easysimd_mm256_maskz_mov_epi16(k, easysimd_mm256_permutex2var_epi16(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutex2var_epi16
#define _mm256_maskz_permutex2var_epi16(k, a, idx, b) easysimd_mm256_maskz_permutex2var_epi16(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permutex2var_epi32 (easysimd__m256i a, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutex2var_epi32(a, idx, b);
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    __m256i ta, tb, select;
    ta = _mm256_permutevar8x32_epi32(a, idx);
    tb = _mm256_permutevar8x32_epi32(b, idx);
    select = _mm256_slli_epi32(idx, 28);
    return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ta),
                                                _mm256_castsi256_ps(tb),
                                                _mm256_castsi256_ps(select)));
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      idx_ = easysimd__m256i_to_private(idx),
      b_ = easysimd__m256i_to_private(b),
      r_;

    #if defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC)
      easysimd_x_permutex2var(r_.m128i, a_.m128i, idx_.m128i, b_.m128i, 2, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ((idx_.i32[i] & 8) ? b_ : a_).i32[idx_.i32[i] & 7];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutex2var_epi32
  #define _mm256_permutex2var_epi32(a, idx, b) easysimd_mm256_permutex2var_epi32(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_permutex2var_epi32 (easysimd__m256i a, easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutex2var_epi32(a, k, idx, b);
  #else
    return easysimd_mm256_mask_mov_epi32(a, k, easysimd_mm256_permutex2var_epi32(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutex2var_epi32
#define _mm256_mask_permutex2var_epi32(a, k, idx, b) easysimd_mm256_mask_permutex2var_epi32(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask2_permutex2var_epi32 (easysimd__m256i a, easysimd__m256i idx, easysimd__mmask8 k, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask2_permutex2var_epi32(a, idx, k, b);
  #else
    return easysimd_mm256_mask_mov_epi32(idx, k, easysimd_mm256_permutex2var_epi32(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask2_permutex2var_epi32
#define _mm256_mask2_permutex2var_epi32(a, idx, k, b) easysimd_mm256_mask2_permutex2var_epi32(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_permutex2var_epi32 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutex2var_epi32(k, a, idx, b);
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_permutex2var_epi32(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutex2var_epi32
#define _mm256_maskz_permutex2var_epi32(k, a, idx, b) easysimd_mm256_maskz_permutex2var_epi32(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permutex2var_epi64 (easysimd__m256i a, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutex2var_epi64(a, idx, b);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      idx_ = easysimd__m256i_to_private(idx),
      b_ = easysimd__m256i_to_private(b),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((idx_.i64[i] & 4) ? b_ : a_).i64[idx_.i64[i] & 3];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutex2var_epi64
  #define _mm256_permutex2var_epi64(a, idx, b) easysimd_mm256_permutex2var_epi64(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_permutex2var_epi64 (easysimd__m256i a, easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutex2var_epi64(a, k, idx, b);
  #else
    return easysimd_mm256_mask_mov_epi64(a, k, easysimd_mm256_permutex2var_epi64(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutex2var_epi64
#define _mm256_mask_permutex2var_epi64(a, k, idx, b) easysimd_mm256_mask_permutex2var_epi64(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask2_permutex2var_epi64 (easysimd__m256i a, easysimd__m256i idx, easysimd__mmask8 k, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask2_permutex2var_epi64(a, idx, k, b);
  #else
    return easysimd_mm256_mask_mov_epi64(idx, k, easysimd_mm256_permutex2var_epi64(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask2_permutex2var_epi64
#define _mm256_mask2_permutex2var_epi64(a, idx, k, b) easysimd_mm256_mask2_permutex2var_epi64(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_permutex2var_epi64 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutex2var_epi64(k, a, idx, b);
  #else
    return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_permutex2var_epi64(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutex2var_epi64
#define _mm256_maskz_permutex2var_epi64(k, a, idx, b) easysimd_mm256_maskz_permutex2var_epi64(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permutex2var_epi8 (easysimd__m256i a, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutex2var_epi8(a, idx, b);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cvtepi16_epi8(_mm512_permutex2var_epi16(_mm512_cvtepu8_epi16(a), _mm512_cvtepu8_epi16(idx), _mm512_cvtepu8_epi16(b)));
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    __m256i t0, t1, index, select0x10, select0x20, a01, b01;
    const __m256i mask = _mm256_set1_epi8(0x3F);
    const __m256i a0 = _mm256_permute4x64_epi64(a, (1 << 6) + (0 << 4) + (1 << 2) + (0 << 0));
    const __m256i a1 = _mm256_permute4x64_epi64(a, (3 << 6) + (2 << 4) + (3 << 2) + (2 << 0));
    const __m256i b0 = _mm256_permute4x64_epi64(b, (1 << 6) + (0 << 4) + (1 << 2) + (0 << 0));
    const __m256i b1 = _mm256_permute4x64_epi64(b, (3 << 6) + (2 << 4) + (3 << 2) + (2 << 0));

    index = _mm256_and_si256(idx, mask);
    t0 = _mm256_shuffle_epi8(a0, index);
    t1 = _mm256_shuffle_epi8(a1, index);
    select0x10 = _mm256_slli_epi64(index, 3);
    a01 = _mm256_blendv_epi8(t0, t1, select0x10);
    t0 = _mm256_shuffle_epi8(b0, index);
    t1 = _mm256_shuffle_epi8(b1, index);
    b01 = _mm256_blendv_epi8(t0, t1, select0x10);
    select0x20 = _mm256_slli_epi64(index, 2);
    return _mm256_blendv_epi8(a01, b01, select0x20);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      idx_ = easysimd__m256i_to_private(idx),
      b_ = easysimd__m256i_to_private(b),
      r_;

    #if defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC)
      easysimd_x_permutex2var(r_.m128i, a_.m128i, idx_.m128i, b_.m128i, 0, 1);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = ((idx_.i8[i] & 0x20) ? b_ : a_).i8[idx_.i8[i] & 0x1F];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutex2var_epi8
  #define _mm256_permutex2var_epi8(a, idx, b) easysimd_mm256_permutex2var_epi8(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_permutex2var_epi8 (easysimd__m256i a, easysimd__mmask32 k, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutex2var_epi8(a, k, idx, b);
  #else
    return easysimd_mm256_mask_mov_epi8(a, k, easysimd_mm256_permutex2var_epi8(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutex2var_epi8
#define _mm256_mask_permutex2var_epi8(a, k, idx, b) easysimd_mm256_mask_permutex2var_epi8(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask2_permutex2var_epi8 (easysimd__m256i a, easysimd__m256i idx, easysimd__mmask32 k, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask2_permutex2var_epi8(a, idx, k, b);
  #else
    return easysimd_mm256_mask_mov_epi8(idx, k, easysimd_mm256_permutex2var_epi8(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask2_permutex2var_epi8
#define _mm256_mask2_permutex2var_epi8(a, idx, k, b) easysimd_mm256_mask2_permutex2var_epi8(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_permutex2var_epi8 (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i idx, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutex2var_epi8(k, a, idx, b);
  #else
    return easysimd_mm256_maskz_mov_epi8(k, easysimd_mm256_permutex2var_epi8(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutex2var_epi8
#define _mm256_maskz_permutex2var_epi8(k, a, idx, b) easysimd_mm256_maskz_permutex2var_epi8(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_permutex2var_pd (easysimd__m256d a, easysimd__m256i idx, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutex2var_pd(a, idx, b);
  #else
    return easysimd_mm256_castsi256_pd(easysimd_mm256_permutex2var_epi64(easysimd_mm256_castpd_si256(a), idx, easysimd_mm256_castpd_si256(b)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutex2var_pd
  #define _mm256_permutex2var_pd(a, idx, b) easysimd_mm256_permutex2var_pd(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_permutex2var_pd (easysimd__m256d a, easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutex2var_pd(a, k, idx, b);
  #else
    return easysimd_mm256_mask_mov_pd(a, k, easysimd_mm256_permutex2var_pd(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutex2var_pd
#define _mm256_mask_permutex2var_pd(a, k, idx, b) easysimd_mm256_mask_permutex2var_pd(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask2_permutex2var_pd (easysimd__m256d a, easysimd__m256i idx, easysimd__mmask8 k, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask2_permutex2var_pd(a, idx, k, b);
  #else
    return easysimd_mm256_mask_mov_pd(easysimd_mm256_castsi256_pd(idx), k, easysimd_mm256_permutex2var_pd(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask2_permutex2var_pd
#define _mm256_mask2_permutex2var_pd(a, idx, k, b) easysimd_mm256_mask2_permutex2var_pd(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_permutex2var_pd (easysimd__mmask8 k, easysimd__m256d a, easysimd__m256i idx, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutex2var_pd(k, a, idx, b);
  #else
    return easysimd_mm256_maskz_mov_pd(k, easysimd_mm256_permutex2var_pd(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutex2var_pd
#define _mm256_maskz_permutex2var_pd(k, a, idx, b) easysimd_mm256_maskz_permutex2var_pd(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_permutex2var_ps (easysimd__m256 a, easysimd__m256i idx, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_permutex2var_ps(a, idx, b);
  #else
    return easysimd_mm256_castsi256_ps(easysimd_mm256_permutex2var_epi32(easysimd_mm256_castps_si256(a), idx, easysimd_mm256_castps_si256(b)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutex2var_ps
  #define _mm256_permutex2var_ps(a, idx, b) easysimd_mm256_permutex2var_ps(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_permutex2var_ps (easysimd__m256 a, easysimd__mmask8 k, easysimd__m256i idx, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_permutex2var_ps(a, k, idx, b);
  #else
    return easysimd_mm256_mask_mov_ps(a, k, easysimd_mm256_permutex2var_ps(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_permutex2var_ps
#define _mm256_mask_permutex2var_ps(a, k, idx, b) easysimd_mm256_mask_permutex2var_ps(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask2_permutex2var_ps (easysimd__m256 a, easysimd__m256i idx, easysimd__mmask8 k, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask2_permutex2var_ps(a, idx, k, b);
  #else
    return easysimd_mm256_mask_mov_ps(easysimd_mm256_castsi256_ps(idx), k, easysimd_mm256_permutex2var_ps(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask2_permutex2var_ps
#define _mm256_mask2_permutex2var_ps(a, idx, k, b) easysimd_mm256_mask2_permutex2var_ps(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_permutex2var_ps (easysimd__mmask8 k, easysimd__m256 a, easysimd__m256i idx, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_permutex2var_ps(k, a, idx, b);
  #else
    return easysimd_mm256_maskz_mov_ps(k, easysimd_mm256_permutex2var_ps(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_permutex2var_ps
#define _mm256_maskz_permutex2var_ps(k, a, idx, b) easysimd_mm256_maskz_permutex2var_ps(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_permutex2var_epi16 (easysimd__m512i a, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_permutex2var_epi16(a, idx, b);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      idx_ = easysimd__m512i_to_private(idx),
      b_ = easysimd__m512i_to_private(b),
      r_;

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      __m256i hilo, hilo1, hilo2, hi, lo, idx1, idx2, ta, tb, select;
      const __m256i ones = _mm256_set1_epi16(1);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i_private) / sizeof(r_.m256i_private[0])) ; i++) {
        idx1 = idx_.m256i[i];
        idx2 = _mm256_srli_epi32(idx1, 1);

        select = _mm256_slli_epi32(idx2, 27);
        ta = _mm256_permutevar8x32_epi32(a_.m256i[0], idx2);
        tb = _mm256_permutevar8x32_epi32(b_.m256i[0], idx2);
        hilo = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ta),
                                                    _mm256_castsi256_ps(tb),
                                                    _mm256_castsi256_ps(select)));
        ta  = _mm256_permutevar8x32_epi32(a_.m256i[1], idx2);
        tb  = _mm256_permutevar8x32_epi32(b_.m256i[1], idx2);
        hilo1 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ta),
                                                     _mm256_castsi256_ps(tb),
                                                     _mm256_castsi256_ps(select)));
        select = _mm256_add_epi32(select, select);
        hilo1 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(hilo),
                                                     _mm256_castsi256_ps(hilo1),
                                                     _mm256_castsi256_ps(select)));

        idx2 = _mm256_srli_epi32(idx2, 16);

        select = _mm256_slli_epi32(idx2, 27);
        ta = _mm256_permutevar8x32_epi32(a_.m256i[0], idx2);
        tb = _mm256_permutevar8x32_epi32(b_.m256i[0], idx2);
        hilo = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ta),
                                                    _mm256_castsi256_ps(tb),
                                                    _mm256_castsi256_ps(select)));
        ta  = _mm256_permutevar8x32_epi32(a_.m256i[1], idx2);
        tb  = _mm256_permutevar8x32_epi32(b_.m256i[1], idx2);
        hilo2 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ta),
                                                     _mm256_castsi256_ps(tb),
                                                     _mm256_castsi256_ps(select)));
        select = _mm256_add_epi32(select, select);
        hilo2 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(hilo),
                                                     _mm256_castsi256_ps(hilo2),
                                                     _mm256_castsi256_ps(select)));

        lo = HEDLEY_REINTERPRET_CAST(__m256i, _mm256_blend_epi16(_mm256_slli_epi32(hilo2, 16), hilo1, 0x55));
        hi = HEDLEY_REINTERPRET_CAST(__m256i, _mm256_blend_epi16(hilo2, _mm256_srli_epi32(hilo1, 16), 0x55));

        select = _mm256_cmpeq_epi16(_mm256_and_si256(idx1, ones), ones);
        r_.m256i[i] = _mm256_blendv_epi8(lo, hi, select);
      }
    #elif defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC)
      easysimd_x_permutex2var(r_.m128i, a_.m128i, idx_.m128i, b_.m128i, 1, 2);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = ((idx_.i16[i] & 0x20) ? b_ : a_).i16[idx_.i16[i] & 0x1F];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutex2var_epi16
  #define _mm512_permutex2var_epi16(a, idx, b) easysimd_mm512_permutex2var_epi16(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_permutex2var_epi16 (easysimd__m512i a, easysimd__mmask32 k, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_permutex2var_epi16(a, k, idx, b);
  #else
    return easysimd_mm512_mask_mov_epi16(a, k, easysimd_mm512_permutex2var_epi16(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutex2var_epi16
#define _mm512_mask_permutex2var_epi16(a, k, idx, b) easysimd_mm512_mask_permutex2var_epi16(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask2_permutex2var_epi16 (easysimd__m512i a, easysimd__m512i idx, easysimd__mmask32 k, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask2_permutex2var_epi16(a, idx, k, b);
  #else
    return easysimd_mm512_mask_mov_epi16(idx, k, easysimd_mm512_permutex2var_epi16(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask2_permutex2var_epi16
#define _mm512_mask2_permutex2var_epi16(a, idx, k, b) easysimd_mm512_mask2_permutex2var_epi16(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_permutex2var_epi16 (easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_permutex2var_epi16(k, a, idx, b);
  #else
    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_permutex2var_epi16(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutex2var_epi16
#define _mm512_maskz_permutex2var_epi16(k, a, idx, b) easysimd_mm512_maskz_permutex2var_epi16(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_permutex2var_epi32 (easysimd__m512i a, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_permutex2var_epi32(a, idx, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    int32_t *ptr_a = (int32_t *)&a;
    int32_t *ptr_b = (int32_t *)&b;
    int32_t *ptr_i = (int32_t *)&idx;
    int32_t *ptr_r = (int32_t *)&r;
    int i;
    for (i = 0; i < 16; ++i) {
        int id = ptr_i[i] & 0x0f;
        ptr_r[i] = ((ptr_i[i] & 0x10)) ? ptr_b[id] : ptr_a[id];
    }
    return r;
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      idx_ = easysimd__m512i_to_private(idx),
      b_ = easysimd__m512i_to_private(b),
      r_;

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      __m256i index, t0, t1, a01, b01, select;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i_private) / sizeof(r_.m256i_private[0])) ; i++) {
        index = idx_.m256i[i];
        t0 = _mm256_permutevar8x32_epi32(a_.m256i[0], index);
        t1 = _mm256_permutevar8x32_epi32(a_.m256i[1], index);
        select = _mm256_slli_epi32(index, 28);
        a01 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(t0),
                                                   _mm256_castsi256_ps(t1),
                                                   _mm256_castsi256_ps(select)));
        t0 = _mm256_permutevar8x32_epi32(b_.m256i[0], index);
        t1 = _mm256_permutevar8x32_epi32(b_.m256i[1], index);
        b01 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(t0),
                                                   _mm256_castsi256_ps(t1),
                                                   _mm256_castsi256_ps(select)));
        select = _mm256_slli_epi32(index, 27);
        r_.m256i[i] = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(a01),
                                                           _mm256_castsi256_ps(b01),
                                                           _mm256_castsi256_ps(select)));
      }
    #elif defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC)
      easysimd_x_permutex2var(r_.m128i, a_.m128i, idx_.m128i, b_.m128i, 2, 2);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ((idx_.i32[i] & 0x10) ? b_ : a_).i32[idx_.i32[i] & 0x0F];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutex2var_epi32
  #define _mm512_permutex2var_epi32(a, idx, b) easysimd_mm512_permutex2var_epi32(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_permutex2var_epi32 (easysimd__m512i a, easysimd__mmask16 k, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_permutex2var_epi32(a, k, idx, b);
  #else
    return easysimd_mm512_mask_mov_epi32(a, k, easysimd_mm512_permutex2var_epi32(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutex2var_epi32
#define _mm512_mask_permutex2var_epi32(a, k, idx, b) easysimd_mm512_mask_permutex2var_epi32(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask2_permutex2var_epi32 (easysimd__m512i a, easysimd__m512i idx, easysimd__mmask16 k, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask2_permutex2var_epi32(a, idx, k, b);
  #else
    return easysimd_mm512_mask_mov_epi32(idx, k, easysimd_mm512_permutex2var_epi32(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask2_permutex2var_epi32
#define _mm512_mask2_permutex2var_epi32(a, idx, k, b) easysimd_mm512_mask2_permutex2var_epi32(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_permutex2var_epi32 (easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_permutex2var_epi32(k, a, idx, b);
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_permutex2var_epi32(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutex2var_epi32
#define _mm512_maskz_permutex2var_epi32(k, a, idx, b) easysimd_mm512_maskz_permutex2var_epi32(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_permutex2var_epi64 (easysimd__m512i a, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_permutex2var_epi64(a, idx, b);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      idx_ = easysimd__m512i_to_private(idx),
      b_ = easysimd__m512i_to_private(b),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((idx_.i64[i] & 8) ? b_ : a_).i64[idx_.i64[i] & 7];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutex2var_epi64
  #define _mm512_permutex2var_epi64(a, idx, b) easysimd_mm512_permutex2var_epi64(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_permutex2var_epi64 (easysimd__m512i a, easysimd__mmask8 k, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_permutex2var_epi64(a, k, idx, b);
  #else
    return easysimd_mm512_mask_mov_epi64(a, k, easysimd_mm512_permutex2var_epi64(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutex2var_epi64
#define _mm512_mask_permutex2var_epi64(a, k, idx, b) easysimd_mm512_mask_permutex2var_epi64(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask2_permutex2var_epi64 (easysimd__m512i a, easysimd__m512i idx, easysimd__mmask8 k, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask2_permutex2var_epi64(a, idx, k, b);
  #else
    return easysimd_mm512_mask_mov_epi64(idx, k, easysimd_mm512_permutex2var_epi64(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask2_permutex2var_epi64
#define _mm512_mask2_permutex2var_epi64(a, idx, k, b) easysimd_mm512_mask2_permutex2var_epi64(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_permutex2var_epi64 (easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_permutex2var_epi64(k, a, idx, b);
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_permutex2var_epi64(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutex2var_epi64
#define _mm512_maskz_permutex2var_epi64(k, a, idx, b) easysimd_mm512_maskz_permutex2var_epi64(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_permutex2var_epi8 (easysimd__m512i a, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_permutex2var_epi8(a, idx, b);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    __m512i hilo, hi, lo, hi2, lo2, idx2;
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i low_bytes = _mm512_set1_epi16(0x00FF);

    idx2 = _mm512_srli_epi16(idx, 1);
    hilo = _mm512_permutex2var_epi16(a, idx2, b);
    __mmask64 mask = _mm512_test_epi8_mask(idx, ones);
    lo = _mm512_and_si512(hilo, low_bytes);
    hi = _mm512_srli_epi16(hilo, 8);

    idx2 = _mm512_srli_epi16(idx, 9);
    hilo = _mm512_permutex2var_epi16(a, idx2, b);
    lo2 = _mm512_slli_epi16(hilo, 8);
    hi2 = _mm512_andnot_si512(low_bytes, hilo);

    lo = _mm512_or_si512(lo, lo2);
    hi = _mm512_or_si512(hi, hi2);

    return _mm512_mask_blend_epi8(mask, lo, hi);
  #else
    easysimd__m512i_private
      a_ = easysimd__m512i_to_private(a),
      idx_ = easysimd__m512i_to_private(idx),
      b_ = easysimd__m512i_to_private(b),
      r_;

    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      __m256i t0, t1, index, select0x10, select0x20, select0x40, t01, t23, a0123, b0123;
      const __m256i mask = _mm256_set1_epi8(0x7F);
      const __m256i a0 = _mm256_permute4x64_epi64(a_.m256i[0], (1 << 6) + (0 << 4) + (1 << 2) + (0 << 0));
      const __m256i a1 = _mm256_permute4x64_epi64(a_.m256i[0], (3 << 6) + (2 << 4) + (3 << 2) + (2 << 0));
      const __m256i a2 = _mm256_permute4x64_epi64(a_.m256i[1], (1 << 6) + (0 << 4) + (1 << 2) + (0 << 0));
      const __m256i a3 = _mm256_permute4x64_epi64(a_.m256i[1], (3 << 6) + (2 << 4) + (3 << 2) + (2 << 0));
      const __m256i b0 = _mm256_permute4x64_epi64(b_.m256i[0], (1 << 6) + (0 << 4) + (1 << 2) + (0 << 0));
      const __m256i b1 = _mm256_permute4x64_epi64(b_.m256i[0], (3 << 6) + (2 << 4) + (3 << 2) + (2 << 0));
      const __m256i b2 = _mm256_permute4x64_epi64(b_.m256i[1], (1 << 6) + (0 << 4) + (1 << 2) + (0 << 0));
      const __m256i b3 = _mm256_permute4x64_epi64(b_.m256i[1], (3 << 6) + (2 << 4) + (3 << 2) + (2 << 0));

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i_private) / sizeof(r_.m256i_private[0])) ; i++) {
        index = _mm256_and_si256(idx_.m256i[i], mask);
        t0 = _mm256_shuffle_epi8(a0, index);
        t1 = _mm256_shuffle_epi8(a1, index);
        select0x10 = _mm256_slli_epi64(index, 3);
        t01 = _mm256_blendv_epi8(t0, t1, select0x10);
        t0 = _mm256_shuffle_epi8(a2, index);
        t1 = _mm256_shuffle_epi8(a3, index);
        t23 = _mm256_blendv_epi8(t0, t1, select0x10);
        select0x20 = _mm256_slli_epi64(index, 2);
        a0123 = _mm256_blendv_epi8(t01, t23, select0x20);
        t0 = _mm256_shuffle_epi8(b0, index);
        t1 = _mm256_shuffle_epi8(b1, index);
        t01 = _mm256_blendv_epi8(t0, t1, select0x10);
        t0 = _mm256_shuffle_epi8(b2, index);
        t1 = _mm256_shuffle_epi8(b3, index);
        t23 = _mm256_blendv_epi8(t0, t1, select0x10);
        b0123 = _mm256_blendv_epi8(t01, t23, select0x20);
        select0x40 = _mm256_slli_epi64(index, 1);
        r_.m256i[i] = _mm256_blendv_epi8(a0123, b0123, select0x40);
      }
    #elif defined(EASYSIMD_X_PERMUTEX2VAR_USE_GENERIC)
      easysimd_x_permutex2var(r_.m128i, a_.m128i, idx_.m128i, b_.m128i, 0, 2);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = ((idx_.i8[i] & 0x40) ? b_ : a_).i8[idx_.i8[i] & 0x3F];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutex2var_epi8
  #define _mm512_permutex2var_epi8(a, idx, b) easysimd_mm512_permutex2var_epi8(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_permutex2var_epi8 (easysimd__m512i a, easysimd__mmask64 k, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_mask_permutex2var_epi8(a, k, idx, b);
  #else
    return easysimd_mm512_mask_mov_epi8(a, k, easysimd_mm512_permutex2var_epi8(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutex2var_epi8
#define _mm512_mask_permutex2var_epi8(a, k, idx, b) easysimd_mm512_mask_permutex2var_epi8(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask2_permutex2var_epi8 (easysimd__m512i a, easysimd__m512i idx, easysimd__mmask64 k, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_mask2_permutex2var_epi8(a, idx, k, b);
  #else
    return easysimd_mm512_mask_mov_epi8(idx, k, easysimd_mm512_permutex2var_epi8(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask2_permutex2var_epi8
#define _mm512_mask2_permutex2var_epi8(a, idx, k, b) easysimd_mm512_mask2_permutex2var_epi8(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_permutex2var_epi8 (easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i idx, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    return _mm512_maskz_permutex2var_epi8(k, a, idx, b);
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_permutex2var_epi8(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutex2var_epi8
#define _mm512_maskz_permutex2var_epi8(k, a, idx, b) easysimd_mm512_maskz_permutex2var_epi8(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_permutex2var_pd (easysimd__m512d a, easysimd__m512i idx, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_permutex2var_pd(a, idx, b);
  #else
    return easysimd_mm512_castsi512_pd(easysimd_mm512_permutex2var_epi64(easysimd_mm512_castpd_si512(a), idx, easysimd_mm512_castpd_si512(b)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutex2var_pd
  #define _mm512_permutex2var_pd(a, idx, b) easysimd_mm512_permutex2var_pd(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_permutex2var_pd (easysimd__m512d a, easysimd__mmask8 k, easysimd__m512i idx, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_permutex2var_pd(a, k, idx, b);
  #else
    return easysimd_mm512_mask_mov_pd(a, k, easysimd_mm512_permutex2var_pd(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutex2var_pd
#define _mm512_mask_permutex2var_pd(a, k, idx, b) easysimd_mm512_mask_permutex2var_pd(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask2_permutex2var_pd (easysimd__m512d a, easysimd__m512i idx, easysimd__mmask8 k, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask2_permutex2var_pd(a, idx, k, b);
  #else
    return easysimd_mm512_mask_mov_pd(easysimd_mm512_castsi512_pd(idx), k, easysimd_mm512_permutex2var_pd(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask2_permutex2var_pd
#define _mm512_mask2_permutex2var_pd(a, idx, k, b) easysimd_mm512_mask2_permutex2var_pd(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_permutex2var_pd (easysimd__mmask8 k, easysimd__m512d a, easysimd__m512i idx, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_permutex2var_pd(k, a, idx, b);
  #else
    return easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_permutex2var_pd(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutex2var_pd
#define _mm512_maskz_permutex2var_pd(k, a, idx, b) easysimd_mm512_maskz_permutex2var_pd(k, a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_permutex2var_ps (easysimd__m512 a, easysimd__m512i idx, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_permutex2var_ps(a, idx, b);
  #else
    return easysimd_mm512_castsi512_ps(easysimd_mm512_permutex2var_epi32(easysimd_mm512_castps_si512(a), idx, easysimd_mm512_castps_si512(b)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_permutex2var_ps
  #define _mm512_permutex2var_ps(a, idx, b) easysimd_mm512_permutex2var_ps(a, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_permutex2var_ps (easysimd__m512 a, easysimd__mmask16 k, easysimd__m512i idx, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_permutex2var_ps(a, k, idx, b);
  #else
    return easysimd_mm512_mask_mov_ps(a, k, easysimd_mm512_permutex2var_ps(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_permutex2var_ps
#define _mm512_mask_permutex2var_ps(a, k, idx, b) easysimd_mm512_mask_permutex2var_ps(a, k, idx, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask2_permutex2var_ps (easysimd__m512 a, easysimd__m512i idx, easysimd__mmask16 k, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask2_permutex2var_ps(a, idx, k, b);
  #else
    return easysimd_mm512_mask_mov_ps(easysimd_mm512_castsi512_ps(idx), k, easysimd_mm512_permutex2var_ps(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask2_permutex2var_ps
#define _mm512_mask2_permutex2var_ps(a, idx, k, b) easysimd_mm512_mask2_permutex2var_ps(a, idx, k, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_permutex2var_ps (easysimd__mmask16 k, easysimd__m512 a, easysimd__m512i idx, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_permutex2var_ps(k, a, idx, b);
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_permutex2var_ps(a, idx, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_permutex2var_ps
#define _mm512_maskz_permutex2var_ps(k, a, idx, b) easysimd_mm512_maskz_permutex2var_ps(k, a, idx, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_PERMUTEX2VAR_H) */
