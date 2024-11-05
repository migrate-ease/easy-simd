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

#if !defined(EASYSIMD_X86_AVX512_MIN_H)
#define EASYSIMD_X86_AVX512_MIN_H

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
easysimd_mm_min_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_min_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svmin_s64_z(svptrue_b64(), a.sve_i64, b.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
       r_.i64[i] = a_.i64[i] < b_.i64[i] ? a_.i64[i] : b_.i64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_min_epi64(a, b) easysimd_mm_min_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_min_epu64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_min_epu64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u64 = svmin_u64_z(svptrue_b64(), a.sve_u64, b.sve_u64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
       r_.u64[i] = a_.u64[i] < b_.u64[i] ? a_.u64[i] : b_.u64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_min_epu64(a, b) easysimd_mm_min_epu64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_min_epi8 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_epi8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svmin_s8_z(svptrue_b8(), a.sve_i8, b.sve_i8), src.sve_i8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? (a_.i8[i] < b_.i8[i] ? a_.i8[i] : b_.i8[i]) : src_.i8[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_min_epi8
  #define _mm_mask_min_epi8(src, k, a, b) easysimd_mm_mask_min_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_min_epi8 (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svmin_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8, b.sve_i8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? (a_.i8[i] < b_.i8[i] ? a_.i8[i] : b_.i8[i]) : INT8_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_min_epi8
  #define _mm_maskz_min_epi8(k, a, b) easysimd_mm_maskz_min_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_min_epi16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svmin_s16_z(svptrue_b16(), a.sve_i16, b.sve_i16), src.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? (a_.i16[i] < b_.i16[i] ? a_.i16[i] : b_.i16[i]) : src_.i16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_min_epi16
  #define _mm_mask_min_epi16(src, k, a, b) easysimd_mm_mask_min_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_min_epi16 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svmin_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16, b.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? (a_.i16[i] < b_.i16[i] ? a_.i16[i] : b_.i16[i]) : INT16_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_min_epi16
  #define _mm_maskz_min_epi16(k, a, b) easysimd_mm_maskz_min_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_min_epi32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_s32_z(svptrue_b32(), a.sve_i32, b.sve_i32), src.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? (a_.i32[i] < b_.i32[i] ? a_.i32[i] : b_.i32[i]) : src_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_min_epi32
  #define _mm_mask_min_epi32(src, k, a, b) easysimd_mm_mask_min_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_min_epi32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svmin_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32, b.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? (a_.i32[i] < b_.i32[i] ? a_.i32[i] : b_.i32[i]) : INT32_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_min_epi32
  #define _mm_maskz_min_epi32(k, a, b) easysimd_mm_maskz_min_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_min_epi64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_s64_z(svptrue_b64(), a.sve_i64, b.sve_i64), src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? (a_.i64[i] < b_.i64[i] ? a_.i64[i] : b_.i64[i]) : src_.i64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_min_epi64
  #define _mm_mask_min_epi64(src, k, a, b) easysimd_mm_mask_min_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_min_epi64 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svmin_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64, b.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? (a_.i64[i] < b_.i64[i] ? a_.i64[i] : b_.i64[i]) : INT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_min_epi64
  #define _mm_maskz_min_epi64(k, a, b) easysimd_mm_maskz_min_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_min_epu8 (easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_epu8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svmin_u8_z(svptrue_b8(), a.sve_u8, b.sve_u8), src.sve_u8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = ((k >> i) & 1) ? (a_.u8[i] < b_.u8[i] ? a_.u8[i] : b_.u8[i]) : src_.u8[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_min_epu8
  #define _mm_mask_min_epu8(src, k, a, b) easysimd_mm_mask_min_epu8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_min_epu8 (easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_epu8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svmin_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_u8, b.sve_u8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = ((k >> i) & 1) ? (a_.u8[i] < b_.u8[i] ? a_.u8[i] : b_.u8[i]) : UINT8_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_min_epu8
  #define _mm_maskz_min_epu8(k, a, b) easysimd_mm_maskz_min_epu8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_min_epu16 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_epu16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svmin_u16_z(svptrue_b16(), a.sve_u16, b.sve_u16), src.sve_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = ((k >> i) & 1) ? (a_.u16[i] < b_.u16[i] ? a_.u16[i] : b_.u16[i]) : src_.u16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_min_epu16
  #define _mm_mask_min_epu16(src, k, a, b) easysimd_mm_mask_min_epu16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_min_epu16 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_epu16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svmin_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_u16, b.sve_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = ((k >> i) & 1) ? (a_.u16[i] < b_.u16[i] ? a_.u16[i] : b_.u16[i]) : UINT16_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_min_epu16
  #define _mm_maskz_min_epu16(k, a, b) easysimd_mm_maskz_min_epu16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_min_epu32 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_epu32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u32 = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_u32_z(svptrue_b32(), a.sve_u32, b.sve_u32), src.sve_u32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] < b_.u32[i] ? a_.u32[i] : b_.u32[i]) : src_.u32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_min_epu32
  #define _mm_mask_min_epu32(src, k, a, b) easysimd_mm_mask_min_epu32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_min_epu32 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_epu32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u32 = svmin_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32, b.sve_u32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] < b_.u32[i] ? a_.u32[i] : b_.u32[i]) : UINT32_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_min_epu32
  #define _mm_maskz_min_epu32(k, a, b) easysimd_mm_maskz_min_epu32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_min_epu64 (easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_epu64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u64 = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_u64_z(svptrue_b64(), a.sve_u64, b.sve_u64), src.sve_u64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src),
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] < b_.u64[i] ? a_.u64[i] : b_.u64[i]) : src_.u64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_min_epu64
  #define _mm_mask_min_epu64(src, k, a, b) easysimd_mm_mask_min_epu64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_min_epu64 (easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_epu64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u64 = svmin_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64, b.sve_u64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] < b_.u64[i] ? a_.u64[i] : b_.u64[i]) : UINT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_min_epu64
  #define _mm_maskz_min_epu64(k, a, b) easysimd_mm_maskz_min_epu64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_min_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_f32_z(pg, a.sve_f32, b.sve_f32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? ((a_.f32[i] < b_.f32[i]) ? a_.f32[i] : b_.f32[i]) : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_min_ps(src, k, a, b) easysimd_mm_mask_min_ps(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_min_ps (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_f32_z(pg, a.sve_f32, b.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? ((a_.f32[i] < b_.f32[i]) ? a_.f32[i] : b_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_min_ps(k, a, b) easysimd_mm_maskz_min_ps(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_min_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_min_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_f64_z(pg, a.sve_f64, b.sve_f64), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      src_ = easysimd__m128d_to_private(src),
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? ((a_.f64[i] < b_.f64[i]) ? a_.f64[i] : b_.f64[i]) : src_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_min_pd(src, k, a, b) easysimd_mm_mask_min_pd(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_min_pd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_min_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_f64_z(pg, a.sve_f64, b.sve_f64), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? ((a_.f64[i] < b_.f64[i]) ? a_.f64[i] : b_.f64[i]) : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_min_pd(k, a, b) easysimd_mm_maskz_min_pd(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_min_epi8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_epi8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svmin_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svmin_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), src.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? (a_.i8[i] < b_.i8[i] ? a_.i8[i] : b_.i8[i]) : src_.i8[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_min_epi8
  #define _mm256_mask_min_epi8(src, k, a, b) easysimd_mm256_mask_min_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_min_epi8 (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svmin_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svmin_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? (a_.i8[i] < b_.i8[i] ? a_.i8[i] : b_.i8[i]) : INT8_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_min_epi8
  #define _mm256_maskz_min_epi8(k, a, b) easysimd_mm256_maskz_min_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_min_epi16 (easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svmin_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svmin_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? (a_.i16[i] < b_.i16[i] ? a_.i16[i] : b_.i16[i]) : src_.i16[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_min_epi16
  #define _mm256_mask_min_epi16(src, k, a, b) easysimd_mm256_mask_min_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_min_epi16 (easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svmin_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svmin_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? (a_.i16[i] < b_.i16[i] ? a_.i16[i] : b_.i16[i]) : INT16_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_min_epi16
  #define _mm256_maskz_min_epi16(k, a, b) easysimd_mm256_maskz_min_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_min_epi32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svmin_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? (a_.i32[i] < b_.i32[i] ? a_.i32[i] : b_.i32[i]) : src_.i32[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_min_epi32
  #define _mm256_mask_min_epi32(src, k, a, b) easysimd_mm256_mask_min_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_min_epi32 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svmin_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svmin_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? (a_.i32[i] < b_.i32[i] ? a_.i32[i] : b_.i32[i]) : INT32_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_min_epi32
  #define _mm256_maskz_min_epi32(k, a, b) easysimd_mm256_maskz_min_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_min_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_min_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svmin_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svmin_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] < b_.i64[i] ? a_.i64[i] : b_.i64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_epi64
  #define _mm256_min_epi64(a, b) easysimd_mm256_min_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_min_epi64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmin_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? (a_.i64[i] < b_.i64[i] ? a_.i64[i] : b_.i64[i]) : src_.i64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_min_epi64
  #define _mm256_mask_min_epi64(src, k, a, b) easysimd_mm256_mask_min_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_min_epi64 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svmin_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svmin_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? (a_.i64[i] < b_.i64[i] ? a_.i64[i] : b_.i64[i]) : INT64_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_min_epi64
  #define _mm256_maskz_min_epi64(k, a, b) easysimd_mm256_maskz_min_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_min_epu8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_epu8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svmin_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), src.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svmin_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), src.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = ((k >> i) & 1) ? (a_.u8[i] < b_.u8[i] ? a_.u8[i] : b_.u8[i]) : src_.u8[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_min_epu8
  #define _mm256_mask_min_epu8(src, k, a, b) easysimd_mm256_mask_min_epu8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_min_epu8 (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_epu8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svmin_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svmin_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = ((k >> i) & 1) ? (a_.u8[i] < b_.u8[i] ? a_.u8[i] : b_.u8[i]) : UINT8_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_min_epu8
  #define _mm256_maskz_min_epu8(k, a, b) easysimd_mm256_maskz_min_epu8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_min_epu16 (easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_epu16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svmin_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), src.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svmin_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), src.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = ((k >> i) & 1) ? (a_.u16[i] < b_.u16[i] ? a_.u16[i] : b_.u16[i]) : src_.u16[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_min_epu16
  #define _mm256_mask_min_epu16(src, k, a, b) easysimd_mm256_mask_min_epu16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_min_epu16 (easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_epu16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svmin_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svmin_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = ((k >> i) & 1) ? (a_.u16[i] < b_.u16[i] ? a_.u16[i] : b_.u16[i]) : UINT16_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_min_epu16
  #define _mm256_maskz_min_epu16(k, a, b) easysimd_mm256_maskz_min_epu16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_min_epu32 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_epu32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), src.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svmin_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), src.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] < b_.u32[i] ? a_.u32[i] : b_.u32[i]) : src_.u32[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_min_epu32
  #define _mm256_mask_min_epu32(src, k, a, b) easysimd_mm256_mask_min_epu32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_min_epu32 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_epu32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svmin_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svmin_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = ((k >> i) & 1) ? (a_.u32[i] < b_.u32[i] ? a_.u32[i] : b_.u32[i]) : UINT32_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_min_epu32
  #define _mm256_maskz_min_epu32(k, a, b) easysimd_mm256_maskz_min_epu32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_min_epu64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_min_epu64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svmin_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svmin_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = (a_.u64[i] < b_.u64[i]) ? a_.u64[i] : b_.u64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_epu64
  #define _mm256_min_epu64(a, b) easysimd_mm256_min_epu64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_min_epu64 (easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_epu64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), src.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmin_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), src.sve_u64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] < b_.u64[i] ? a_.u64[i] : b_.u64[i]) : src_.u64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_min_epu64
  #define _mm256_mask_min_epu64(src, k, a, b) easysimd_mm256_mask_min_epu64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_min_epu64 (easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_epu64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svmin_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svmin_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = ((k >> i) & 1) ? (a_.u64[i] < b_.u64[i] ? a_.u64[i] : b_.u64[i]) : UINT64_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_min_epu64
  #define _mm256_maskz_min_epu64(k, a, b) easysimd_mm256_maskz_min_epu64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_min_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svmin_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      src_ = easysimd__m256_to_private(src),
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? ((a_.f32[i] < b_.f32[i]) ? a_.f32[i] : b_.f32[i]) : src_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_min_ps(src, k, a, b) easysimd_mm256_mask_min_ps(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_min_ps (easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svmin_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svmin_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? ((a_.f32[i] < b_.f32[i]) ? a_.f32[i] : b_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_min_ps(k, a, b) easysimd_mm256_maskz_min_ps(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_min_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_min_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmin_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      src_ = easysimd__m256d_to_private(src),
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? ((a_.f64[i] < b_.f64[i]) ? a_.f64[i] : b_.f64[i]) : src_.f64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_min_pd(src, k, a, b) easysimd_mm256_mask_min_pd(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_min_pd (easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_min_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svmin_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svmin_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? ((a_.f64[i] < b_.f64[i]) ? a_.f64[i] : b_.f64[i]) : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_min_pd(k, a, b) easysimd_mm256_maskz_min_pd(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_min_epi8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_min_epi8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i8 = vminq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vminq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    r.m128i[2].neon_i8 = vminq_s8(a.m128i[2].neon_i8, b.m128i[2].neon_i8);
    r.m128i[3].neon_i8 = vminq_s8(a.m128i[3].neon_i8, b.m128i[3].neon_i8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svmin_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svmin_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svmin_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svmin_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = (a_.i8[i] < b_.i8[i]) ? a_.i8[i] : b_.i8[i];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
#  define _mm512_min_epi8(a, b) easysimd_mm512_min_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_min_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_min_epi8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svmin_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svmin_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), src.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), svmin_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), src.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), svmin_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), src.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_min_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_epi8
  #define _mm512_mask_min_epi8(src, k, a, b) easysimd_mm512_mask_min_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_min_epi8 (easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_min_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svmin_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svmin_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svmin_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svmin_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_min_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_epi8
  #define _mm512_maskz_min_epi8(k, a, b) easysimd_mm512_maskz_min_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_min_epu8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_min_epu8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u8 = vminq_u8(a.m128i[0].neon_u8, b.m128i[0].neon_u8);
    r.m128i[1].neon_u8 = vminq_u8(a.m128i[1].neon_u8, b.m128i[1].neon_u8);
    r.m128i[2].neon_u8 = vminq_u8(a.m128i[2].neon_u8, b.m128i[2].neon_u8);
    r.m128i[3].neon_u8 = vminq_u8(a.m128i[3].neon_u8, b.m128i[3].neon_u8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svmin_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svmin_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svmin_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]);
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svmin_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_min_epu8(a_.m256i[i], b_.m256i[i]);
      }
    #else   
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = (a_.u8[i] < b_.u8[i]) ? a_.u8[i] : b_.u8[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_min_epu8
  #define _mm512_min_epu8(a, b) easysimd_mm512_min_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_min_epu8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_min_epu8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svmin_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]), src.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svmin_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]), src.sve_u8[EASYSIMD_SV_INDEX_1]);
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), svmin_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]), src.sve_u8[EASYSIMD_SV_INDEX_2]);
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svsel_u8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), svmin_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]), src.sve_u8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_min_epu8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_epu8
  #define _mm512_mask_min_epu8(src, k, a, b) easysimd_mm512_mask_min_epu8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_min_epu8 (easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_min_epu8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svmin_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svmin_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    r.sve_u8[EASYSIMD_SV_INDEX_2] = svmin_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), a.sve_u8[EASYSIMD_SV_INDEX_2], b.sve_u8[EASYSIMD_SV_INDEX_2]);
    r.sve_u8[EASYSIMD_SV_INDEX_3] = svmin_u8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), a.sve_u8[EASYSIMD_SV_INDEX_3], b.sve_u8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_min_epu8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_epu8
  #define _mm512_maskz_min_epu8(k, a, b) easysimd_mm512_maskz_min_epu8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_min_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_min_epi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i16 = vminq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vminq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    r.m128i[2].neon_i16 = vminq_s16(a.m128i[2].neon_i16, b.m128i[2].neon_i16);
    r.m128i[3].neon_i16 = vminq_s16(a.m128i[3].neon_i16, b.m128i[3].neon_i16);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svmin_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svmin_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svmin_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svmin_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private 
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? a_.i16[i] : b_.i16[i];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
#  define _mm512_min_epi16(a, b) easysimd_mm512_min_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_min_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_min_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svmin_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svmin_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svmin_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]), src.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svmin_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]), src.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_min_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_epi16
  #define _mm512_mask_min_epi16(src, k, a, b) easysimd_mm512_mask_min_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_min_epi16 (easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_min_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svmin_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svmin_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svmin_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svmin_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_min_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_epi16
  #define _mm512_maskz_min_epi16(k, a, b) easysimd_mm512_maskz_min_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_min_epu16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_min_epu16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u16 = vminq_u16(a.m128i[0].neon_u16, b.m128i[0].neon_u16);
    r.m128i[1].neon_u16 = vminq_u16(a.m128i[1].neon_u16, b.m128i[1].neon_u16);
    r.m128i[2].neon_u16 = vminq_u16(a.m128i[2].neon_u16, b.m128i[2].neon_u16);
    r.m128i[3].neon_u16 = vminq_u16(a.m128i[3].neon_u16, b.m128i[3].neon_u16);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svmin_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svmin_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svmin_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svmin_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(a_.m256i) / sizeof(a_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_min_epu16(a_.m256i[i], b_.m256i[i]);
      }
    #else    
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = (a_.u16[i] < b_.u16[i]) ? a_.u16[i] : b_.u16[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_min_epu16
  #define _mm512_min_epu16(a, b) easysimd_mm512_min_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_min_epu16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_min_epu16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svmin_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]), src.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svmin_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]), src.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svmin_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]), src.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svsel_u16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svmin_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]), src.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_min_epu16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_epu16
  #define _mm512_mask_min_epu16(src, k, a, b) easysimd_mm512_mask_min_epu16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_min_epu16 (easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_min_epu16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svmin_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svmin_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    r.sve_u16[EASYSIMD_SV_INDEX_2] = svmin_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_u16[EASYSIMD_SV_INDEX_2], b.sve_u16[EASYSIMD_SV_INDEX_2]);
    r.sve_u16[EASYSIMD_SV_INDEX_3] = svmin_u16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_u16[EASYSIMD_SV_INDEX_3], b.sve_u16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_min_epu16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_epu16
  #define _mm512_maskz_min_epu16(k, a, b) easysimd_mm512_maskz_min_epu16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_min_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_min_epi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i32 = vminq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vminq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    r.m128i[2].neon_i32 = vminq_s32(a.m128i[2].neon_i32, b.m128i[2].neon_i32);
    r.m128i[3].neon_i32 = vminq_s32(a.m128i[3].neon_i32, b.m128i[3].neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svmin_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svmin_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svmin_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svmin_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    #if defined(EASYSIMD_X86_AVX2_NATIVE)   
      r_.m256i[0] = easysimd_mm256_min_epi32(a_.m256i[0], b_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_min_epi32(a_.m256i[1], b_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] < b_.i32[i] ? a_.i32[i] : b_.i32[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_min_epi32
  #define _mm512_min_epi32(a, b) easysimd_mm512_min_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_min_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_min_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svmin_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svmin_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svmin_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_min_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_epi32
  #define _mm512_mask_min_epi32(src, k, a, b) easysimd_mm512_mask_min_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
  easysimd_mm512_maskz_min_epi32(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_min_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svmin_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svmin_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svmin_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svmin_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_min_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_epi32
  #define _mm512_maskz_min_epi32(k, a, b) easysimd_mm512_maskz_min_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_min_epu32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_min_epu32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u32 = vminq_u32(a.m128i[0].neon_u32, b.m128i[0].neon_u32);
    r.m128i[1].neon_u32 = vminq_u32(a.m128i[1].neon_u32, b.m128i[1].neon_u32);
    r.m128i[2].neon_u32 = vminq_u32(a.m128i[2].neon_u32, b.m128i[2].neon_u32);
    r.m128i[3].neon_u32 = vminq_u32(a.m128i[3].neon_u32, b.m128i[3].neon_u32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svmin_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svmin_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svmin_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svmin_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    #if defined(EASYSIMD_X86_AVX2_NATIVE)
      r_.m256i[0] = easysimd_mm256_min_epu32(a_.m256i[0], b_.m256i[0]);
      r_.m256i[1] = easysimd_mm256_min_epu32(a_.m256i[1], b_.m256i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = (a_.u32[i] < b_.u32[i]) ? a_.u32[i] : b_.u32[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_min_epu32
  #define _mm512_min_epu32(a, b) easysimd_mm512_min_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_min_epu32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_min_epu32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]), src.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svmin_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]), src.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svmin_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]), src.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svsel_u32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svmin_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]), src.sve_u32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_min_epu32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_epu32
  #define _mm512_mask_min_epu32(src, k, a, b) easysimd_mm512_mask_min_epu32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_min_epu32(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_min_epu32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svmin_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svmin_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svmin_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svmin_u32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_min_epu32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_epu32
  #define _mm512_maskz_min_epu32(k, a, b) easysimd_mm512_maskz_min_epu32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_min_epi64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_min_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svmin_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svmin_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svmin_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svmin_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] < b_.i64[i] ? a_.i64[i] : b_.i64[i];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_min_epi64
  #define _mm512_min_epi64(a, b) easysimd_mm512_min_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_min_epi64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_min_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmin_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svmin_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]), src.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svmin_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]), src.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_min_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_epi64
  #define _mm512_mask_min_epi64(src, k, a, b) easysimd_mm512_mask_min_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_min_epi64(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_min_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svmin_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svmin_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svmin_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svmin_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_min_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_epi64
  #define _mm512_maskz_min_epi64(k, a, b) easysimd_mm512_maskz_min_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_min_epu64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_min_epu64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svmin_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svmin_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svmin_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svmin_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = (a_.u64[i] < b_.u64[i]) ? a_.u64[i] : b_.u64[i];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_min_epu64
  #define _mm512_min_epu64(a, b) easysimd_mm512_min_epu64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_min_epu64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_min_epu64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]), src.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmin_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]), src.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svmin_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]), src.sve_u64[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svsel_u64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svmin_u64_x(pg, a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]), src.sve_u64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_min_epu64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_epu64
  #define _mm512_mask_min_epu64(src, k, a, b) easysimd_mm512_mask_min_epu64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_min_epu64(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_min_epu64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svmin_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svmin_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_u64[EASYSIMD_SV_INDEX_2] = svmin_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_u64[EASYSIMD_SV_INDEX_2], b.sve_u64[EASYSIMD_SV_INDEX_2]);
    r.sve_u64[EASYSIMD_SV_INDEX_3] = svmin_u64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_u64[EASYSIMD_SV_INDEX_3], b.sve_u64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_min_epu64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_epu64
  #define _mm512_maskz_min_epu64(k, a, b) easysimd_mm512_maskz_min_epu64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_min_ps (easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_min_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0].neon_f32 = vminq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vminq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    r.m128[2].neon_f32 = vminq_f32(a.m128[2].neon_f32, b.m128[2].neon_f32);
    r.m128[3].neon_f32 = vminq_f32(a.m128[3].neon_f32, b.m128[3].neon_f32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svmin_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svmin_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svmin_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svmin_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);
    #if defined(EASYSIMD_X86_AVX2_NATIVE)     
      r_.m256[0] = easysimd_mm256_min_ps(a_.m256[0], b_.m256[0]);
      r_.m256[1] = easysimd_mm256_min_ps(a_.m256[1], b_.m256[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i] < b_.f32[i] ? a_.f32[i] : b_.f32[i];
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_min_ps
  #define _mm512_min_ps(a, b) easysimd_mm512_min_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_min_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_min_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svmin_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svmin_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svmin_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]), src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svmin_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]), src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_min_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_ps
  #define _mm512_mask_min_ps(src, k, a, b) easysimd_mm512_mask_min_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_min_ps(easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_min_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svmin_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svmin_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svmin_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svmin_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_min_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_ps
  #define _mm512_maskz_min_ps(k, a, b) easysimd_mm512_maskz_min_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_min_pd (easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_min_pd(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0].neon_f64 = vminq_f64(a.m128d[0].neon_f64, b.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vminq_f64(a.m128d[1].neon_f64, b.m128d[1].neon_f64);
    r.m128d[2].neon_f64 = vminq_f64(a.m128d[2].neon_f64, b.m128d[2].neon_f64);
    r.m128d[3].neon_f64 = vminq_f64(a.m128d[3].neon_f64, b.m128d[3].neon_f64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svmin_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svmin_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svmin_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svmin_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[i] < b_.f64[i] ? a_.f64[i] : b_.f64[i];
      }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_min_pd
  #define _mm512_min_pd(a, b) easysimd_mm512_min_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_min_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_min_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svmin_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svmin_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svmin_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]), src.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svmin_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]), src.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_min_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_min_pd
  #define _mm512_mask_min_pd(src, k, a, b) easysimd_mm512_mask_min_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_min_pd(easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_min_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svmin_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svmin_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svmin_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svmin_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_min_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_min_pd
  #define _mm512_maskz_min_pd(k, a, b) easysimd_mm512_maskz_min_pd(k, a, b)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_MIN_H) */
