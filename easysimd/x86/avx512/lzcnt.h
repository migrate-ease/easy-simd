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
 */

#if !defined(EASYSIMD_X86_AVX512_LZCNT_H)
#define EASYSIMD_X86_AVX512_LZCNT_H

#include "types.h"
#include "mov.h"
#if HEDLEY_MSVC_VERSION_CHECK(14,0,0)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
  #if defined(_M_AMD64) || defined(_M_ARM64)
  #pragma intrinsic(_BitScanReverse64)
  #endif
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

#if \
    ( HEDLEY_HAS_BUILTIN(__builtin_clz) || \
      HEDLEY_GCC_VERSION_CHECK(3,4,0) || \
      HEDLEY_ARM_VERSION_CHECK(4,1,0) ) && \
    defined(__INT_MAX__) && defined(__LONG_MAX__) && defined(__LONG_LONG_MAX__) && \
    defined(__INT32_MAX__) && defined(__INT64_MAX__)
  #if __INT_MAX__ == __INT32_MAX__
    #define easysimd_x_clz32(v) __builtin_clz(HEDLEY_STATIC_CAST(unsigned int, (v)))
  #elif __LONG_MAX__ == __INT32_MAX__
    #define easysimd_x_clz32(v) __builtin_clzl(HEDLEY_STATIC_CAST(unsigned long, (v)))
  #elif __LONG_LONG_MAX__ == __INT32_MAX__
    #define easysimd_x_clz32(v) __builtin_clzll(HEDLEY_STATIC_CAST(unsigned long long, (v)))
  #endif

  #if __INT_MAX__ == __INT64_MAX__
    #define easysimd_x_clz64(v) __builtin_clz(HEDLEY_STATIC_CAST(unsigned int, (v)))
  #elif __LONG_MAX__ == __INT64_MAX__
    #define easysimd_x_clz64(v) __builtin_clzl(HEDLEY_STATIC_CAST(unsigned long, (v)))
  #elif __LONG_LONG_MAX__ == __INT64_MAX__
    #define easysimd_x_clz64(v) __builtin_clzll(HEDLEY_STATIC_CAST(unsigned long long, (v)))
  #endif
#elif HEDLEY_MSVC_VERSION_CHECK(14,0,0)
  static int easysimd_x_clz32(uint32_t x) {
    unsigned long r;
    _BitScanReverse(&r, x);
    return 31 - HEDLEY_STATIC_CAST(int, r);
  }
  #define easysimd_x_clz32 easysimd_x_clz32

  static int easysimd_x_clz64(uint64_t x) {
    unsigned long r;

    #if defined(_M_AMD64) || defined(_M_ARM64)
      _BitScanReverse64(&r, x);
      return 63 - HEDLEY_STATIC_CAST(int, r);
    #else
      uint32_t high = HEDLEY_STATIC_CAST(uint32_t, x >> 32);
      if (high != 0)
        return _BitScanReverse(&r, HEDLEY_STATIC_CAST(unsigned long, high));
      else
        return _BitScanReverse(&r, HEDLEY_STATIC_CAST(unsigned long, x & ~UINT32_C(0))) + 32;
    #endif
  }
  #define easysimd_x_clz64 easysimd_x_clz64
#endif

#if !defined(easysimd_x_clz32) || !defined(easysimd_x_clz64)
  static uint8_t easysimd_x_avx512cd_lz_lookup(const uint8_t value) {
    static const uint8_t lut[256] = {
      7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    return lut[value];
  };

  #if !defined(easysimd_x_clz32)
    static int easysimd_x_clz32(uint32_t x) {
      size_t s = sizeof(x) * 8;
      uint32_t r;

      while ((s -= 8) != 0) {
        r = x >> s;
        if (r != 0)
          return easysimd_x_avx512cd_lz_lookup(HEDLEY_STATIC_CAST(uint8_t, r)) +
            (((sizeof(x) - 1) * 8) - s);
      }

      if (x == 0)
        return (int) ((sizeof(x) * 8) - 1);
      else
        return easysimd_x_avx512cd_lz_lookup(HEDLEY_STATIC_CAST(uint8_t, x)) +
          ((sizeof(x) - 1) * 8);
    }
  #endif

  #if !defined(easysimd_x_clz64)
    static int easysimd_x_clz64(uint64_t x) {
      size_t s = sizeof(x) * 8;
      uint64_t r;

      while ((s -= 8) != 0) {
        r = x >> s;
        if (r != 0)
          return easysimd_x_avx512cd_lz_lookup(HEDLEY_STATIC_CAST(uint8_t, r)) +
            (((sizeof(x) - 1) * 8) - s);
      }

      if (x == 0)
        return (int) ((sizeof(x) * 8) - 1);
      else
        return easysimd_x_avx512cd_lz_lookup(HEDLEY_STATIC_CAST(uint8_t, x)) +
          ((sizeof(x) - 1) * 8);
    }
  #endif
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_lzcnt_epi32(easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_lzcnt_epi32(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    /* https://stackoverflow.com/a/58827596/501126 */
    a = _mm_andnot_si128(_mm_srli_epi32(a, 8), a);
    a = _mm_castps_si128(_mm_cvtepi32_ps(a));
    a = _mm_srli_epi32(a, 23);
    a = _mm_subs_epu16(_mm_set1_epi32(158), a);
    a = _mm_min_epi16(a, _mm_set1_epi32(32));
    return a;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32 = svclz_s32_x(svptrue_b32(), a.sve_i32);
    return a;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i32) / sizeof(r_.i32[0])); i++) {
        r_.i32[i] = (HEDLEY_UNLIKELY(a_.i32[i] == 0) ? HEDLEY_STATIC_CAST(int32_t, sizeof(int32_t) * CHAR_BIT) : HEDLEY_STATIC_CAST(int32_t, easysimd_x_clz32(HEDLEY_STATIC_CAST(uint32_t, a_.i32[i]))));
      }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_lzcnt_epi32
  #define _mm_lzcnt_epi32(a) easysimd_mm_lzcnt_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_lzcnt_epi32(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_mask_lzcnt_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32 = svclz_s32_m(src.sve_u32, EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32);
    return a;
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_lzcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_lzcnt_epi32
  #define _mm_mask_lzcnt_epi32(src, k, a) easysimd_mm_mask_lzcnt_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_lzcnt_epi32(easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_maskz_lzcnt_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32 = svclz_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32);
    return a;
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_lzcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_lzcnt_epi32
  #define _mm_maskz_lzcnt_epi32(k, a) easysimd_mm_maskz_lzcnt_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_lzcnt_epi64(easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_lzcnt_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u64 = svclz_s64_x(svptrue_b64(), a.sve_i64);
    return a;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i64) / sizeof(r_.i64[0])); i++) {
        r_.i64[i] = (HEDLEY_UNLIKELY(a_.i64[i] == 0) ? HEDLEY_STATIC_CAST(int64_t, sizeof(int64_t) * CHAR_BIT) : HEDLEY_STATIC_CAST(int64_t, easysimd_x_clz64(HEDLEY_STATIC_CAST(uint64_t, a_.i64[i]))));
      }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_lzcnt_epi64
  #define _mm_lzcnt_epi64(a) easysimd_mm_lzcnt_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_lzcnt_epi64(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_mask_lzcnt_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u64 = svclz_s64_m(src.sve_u64, EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64);
    return a;
  #else
    return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_lzcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_lzcnt_epi64
  #define _mm_mask_lzcnt_epi64(src, k, a) easysimd_mm_mask_lzcnt_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_lzcnt_epi64(easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm_maskz_lzcnt_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u64 = svclz_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64);
    return a;
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_lzcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_lzcnt_epi64
  #define _mm_maskz_lzcnt_epi64(k, a) easysimd_mm_maskz_lzcnt_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_lzcnt_epi32(easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_lzcnt_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svclz_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svclz_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i32) / sizeof(r_.i32[0])); i++) {
        r_.i32[i] = (HEDLEY_UNLIKELY(a_.i32[i] == 0) ? HEDLEY_STATIC_CAST(int32_t, sizeof(int32_t) * CHAR_BIT) : HEDLEY_STATIC_CAST(int32_t, easysimd_x_clz32(HEDLEY_STATIC_CAST(uint32_t, a_.i32[i]))));
      }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_lzcnt_epi32
  #define _mm256_lzcnt_epi32(a) easysimd_mm256_lzcnt_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_lzcnt_epi32(easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_mask_lzcnt_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svclz_s32_m(src.sve_u32[EASYSIMD_SV_INDEX_0], EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svclz_s32_m(src.sve_u32[EASYSIMD_SV_INDEX_1], EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_lzcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_lzcnt_epi32
  #define _mm256_mask_lzcnt_epi32(src, k, a) easysimd_mm256_mask_lzcnt_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_lzcnt_epi32(easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_maskz_lzcnt_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svclz_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svclz_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_lzcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_lzcnt_epi32
  #define _mm256_maskz_lzcnt_epi32(k, a) easysimd_mm256_maskz_lzcnt_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_lzcnt_epi64(easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_lzcnt_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svclz_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svclz_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i64) / sizeof(r_.i64[0])); i++) {
        r_.i64[i] = (HEDLEY_UNLIKELY(a_.i64[i] == 0) ? HEDLEY_STATIC_CAST(int64_t, sizeof(int64_t) * CHAR_BIT) : HEDLEY_STATIC_CAST(int64_t, easysimd_x_clz64(HEDLEY_STATIC_CAST(uint64_t, a_.i64[i]))));
      }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_lzcnt_epi64
  #define _mm256_lzcnt_epi64(a) easysimd_mm256_lzcnt_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_lzcnt_epi64(easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_mask_lzcnt_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svclz_s64_m(src.sve_u64[EASYSIMD_SV_INDEX_0], EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svclz_s64_m(src.sve_u64[EASYSIMD_SV_INDEX_1], EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    return easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_lzcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_lzcnt_epi64
  #define _mm256_mask_lzcnt_epi64(src, k, a) easysimd_mm256_mask_lzcnt_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_lzcnt_epi64(easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm256_maskz_lzcnt_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svclz_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svclz_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_lzcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_lzcnt_epi64
  #define _mm256_maskz_lzcnt_epi64(k, a) easysimd_mm256_maskz_lzcnt_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_lzcnt_epi32(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_lzcnt_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svclz_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svclz_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    a.sve_u32[EASYSIMD_SV_INDEX_2] = svclz_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_2]);
    a.sve_u32[EASYSIMD_SV_INDEX_3] = svclz_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i32) / sizeof(r_.i32[0])); i++) {
        r_.i32[i] = (HEDLEY_UNLIKELY(a_.i32[i] == 0) ? HEDLEY_STATIC_CAST(int32_t, sizeof(int32_t) * CHAR_BIT) : HEDLEY_STATIC_CAST(int32_t, easysimd_x_clz32(HEDLEY_STATIC_CAST(uint32_t, a_.i32[i]))));
      }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_lzcnt_epi32
  #define _mm512_lzcnt_epi32(a) easysimd_mm512_lzcnt_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_lzcnt_epi32(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_mask_lzcnt_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svclz_s32_m(src.sve_u32[EASYSIMD_SV_INDEX_0], EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svclz_s32_m(src.sve_u32[EASYSIMD_SV_INDEX_1], EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    a.sve_u32[EASYSIMD_SV_INDEX_2] = svclz_s32_m(src.sve_u32[EASYSIMD_SV_INDEX_2], EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2]);
    a.sve_u32[EASYSIMD_SV_INDEX_3] = svclz_s32_m(src.sve_u32[EASYSIMD_SV_INDEX_3], EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_lzcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_lzcnt_epi32
  #define _mm512_mask_lzcnt_epi32(src, k, a) easysimd_mm512_mask_lzcnt_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_lzcnt_epi32(easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_maskz_lzcnt_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svclz_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svclz_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    a.sve_u32[EASYSIMD_SV_INDEX_2] = svclz_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2]);
    a.sve_u32[EASYSIMD_SV_INDEX_3] = svclz_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_lzcnt_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_lzcnt_epi32
  #define _mm512_maskz_lzcnt_epi32(k, a) easysimd_mm512_maskz_lzcnt_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_lzcnt_epi64(easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_lzcnt_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svclz_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svclz_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    a.sve_u64[EASYSIMD_SV_INDEX_2] = svclz_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_u64[EASYSIMD_SV_INDEX_3] = svclz_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i64) / sizeof(r_.i64[0])); i++) {
        r_.i64[i] = (HEDLEY_UNLIKELY(a_.i64[i] == 0) ? HEDLEY_STATIC_CAST(int64_t, sizeof(int64_t) * CHAR_BIT) : HEDLEY_STATIC_CAST(int64_t, easysimd_x_clz64(HEDLEY_STATIC_CAST(uint64_t, a_.i64[i]))));
      }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_lzcnt_epi64
  #define _mm512_lzcnt_epi64(a) easysimd_mm512_lzcnt_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_lzcnt_epi64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_mask_lzcnt_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svclz_s64_m(src.sve_u64[EASYSIMD_SV_INDEX_0], EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svclz_s64_m(src.sve_u64[EASYSIMD_SV_INDEX_1], EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    a.sve_u64[EASYSIMD_SV_INDEX_2] = svclz_s64_m(src.sve_u64[EASYSIMD_SV_INDEX_2], EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_u64[EASYSIMD_SV_INDEX_3] = svclz_s64_m(src.sve_u64[EASYSIMD_SV_INDEX_3], EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_lzcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_lzcnt_epi64
  #define _mm512_mask_lzcnt_epi64(src, k, a) easysimd_mm512_mask_lzcnt_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_lzcnt_epi64(easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512CD_NATIVE)
    return _mm512_maskz_lzcnt_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u64[EASYSIMD_SV_INDEX_0] = svclz_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    a.sve_u64[EASYSIMD_SV_INDEX_1] = svclz_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1]);
    a.sve_u64[EASYSIMD_SV_INDEX_2] = svclz_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2]);
    a.sve_u64[EASYSIMD_SV_INDEX_3] = svclz_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3]);
    return a;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_lzcnt_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_lzcnt_epi64
  #define _mm512_maskz_lzcnt_epi64(k, a) easysimd_mm512_maskz_lzcnt_epi64(k, a)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_LZCNT_H) */
