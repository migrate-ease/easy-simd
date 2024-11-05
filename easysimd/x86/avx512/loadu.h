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

#if !defined(EASYSIMD_X86_AVX512_LOADU_H)
#define EASYSIMD_X86_AVX512_LOADU_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_loadu_epi8(easysimd__m128i src, easysimd__mmask16 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_loadu_epi8(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svld1_s8(svptrue_b8(), (const int8_t *)mem_addr), src.sve_i8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? r_.i8[i] : src_.i8[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_loadu_epi8
  #define _mm_mask_loadu_epi8(src, k, mem_addr) easysimd_mm_mask_loadu_epi8(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_loadu_epi8(easysimd__mmask16 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_loadu_epi8(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svld1_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), (const int8_t *)mem_addr);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    uint8_t const* data_addr = (uint8_t const*)mem_addr;
    static easysimd__m128i mask = {
      .u8 = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80}
    };
    uint8x8_t mk = vcreate_u8(k);
    uint8x16_t k_vec = vcombine_u8(vdup_lane_u8(mk, 0), vdup_lane_u8(mk, 1));
    r.neon_u8 = vandq_u8(vtstq_u8(k_vec, mask.neon_u8), vld1q_u8(data_addr));
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? r_.i8[i] : INT8_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_loadu_epi8
  #define _mm_maskz_loadu_epi8(k, mem_addr) easysimd_mm_maskz_loadu_epi8(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_loadu_epi16(easysimd__m128i src, easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_loadu_epi16(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svld1_s16(svptrue_b16(), (const int16_t *)mem_addr), src.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? r_.i16[i] : src_.i16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_loadu_epi16
  #define _mm_mask_loadu_epi16(src, k, mem_addr) easysimd_mm_mask_loadu_epi16(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_loadu_epi16(easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_loadu_epi16(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svld1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), (const int16_t *)mem_addr);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? r_.i16[i] : INT16_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_loadu_epi16
  #define _mm_maskz_loadu_epi16(k, mem_addr) easysimd_mm_maskz_loadu_epi16(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_loadu_epi32(easysimd__m128i src, easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_loadu_epi32(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svld1_s32(svptrue_b32(), (const int32_t *)mem_addr), src.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : src_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_loadu_epi32
  #define _mm_mask_loadu_epi32(src, k, mem_addr) easysimd_mm_mask_loadu_epi32(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_loadu_epi32(easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_loadu_epi32(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (const int32_t *)mem_addr);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : INT32_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_loadu_epi32
  #define _mm_maskz_loadu_epi32(k, mem_addr) easysimd_mm_maskz_loadu_epi32(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_loadu_epi64(easysimd__m128i src, easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_loadu_epi64(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svld1_s64(svptrue_b64(), (const int64_t *)mem_addr), src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      r_,
      src_ = easysimd__m128i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? r_.i64[i] : src_.i64[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_loadu_epi64
  #define _mm_mask_loadu_epi64(src, k, mem_addr) easysimd_mm_mask_loadu_epi64(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_loadu_epi64(easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_loadu_epi64(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (const int64_t *)mem_addr);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? r_.i64[i] : INT64_C(0);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_loadu_epi64
  #define _mm_maskz_loadu_epi64(k, mem_addr) easysimd_mm_maskz_loadu_epi64(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_loadu_ps (easysimd__m128 src, easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_loadu_ps(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svld1_f32(pg, (float32_t const *)mem_addr), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : src_.f32[i];
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_loadu_ps(src, k, mem_addr) easysimd_mm_mask_loadu_ps(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_loadu_ps (easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_loadu_ps(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svld1_f32(pg, (float32_t const *)mem_addr), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_loadu_ps(k, mem_addr) easysimd_mm_maskz_loadu_ps(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_loadu_pd (easysimd__m128d src, easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_loadu_pd(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svld1_f64(pg, (float64_t const *)mem_addr), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : src_.f64[i];
    }
    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_loadu_pd(src, k, mem_addr) easysimd_mm_mask_loadu_pd(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_loadu_pd (easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_loadu_pd(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svld1_f64(pg, (float64_t const *)mem_addr), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
    }
    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_loadu_pd(k, mem_addr) easysimd_mm_maskz_loadu_pd(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_loadu_epi8(easysimd__m256i src, easysimd__mmask32 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_loadu_epi8(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0),
                                          svld1_s8(pg, (const int8_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 3)), src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1),
                                          svld1_s8(pg, (const int8_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 3)), src.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? r_.i8[i] : src_.i8[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_loadu_epi8
  #define _mm256_mask_loadu_epi8(src, k, mem_addr) easysimd_mm256_mask_loadu_epi8(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_loadu_epi8(easysimd__mmask32 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_maskz_loadu_epi8(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svld1_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), (const int8_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 3));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svld1_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), (const int8_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 3));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? r_.i8[i] : INT8_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_loadu_epi8
  #define _mm256_maskz_loadu_epi8(mask, mem_addr) easysimd_mm256_maskz_loadu_epi8(mask, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_loadu_epi16(easysimd__m256i src, easysimd__mmask16 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_loadu_epi16(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0),
                                            svld1_s16(pg, (const int16_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 4)), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1),
                                            svld1_s16(pg, (const int16_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 4)), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? r_.i16[i] : src_.i16[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_loadu_epi16
  #define _mm256_mask_loadu_epi16(src, k, mem_addr) easysimd_mm256_mask_loadu_epi16(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_loadu_epi16(easysimd__mmask16 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_maskz_loadu_epi16(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svld1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), (const int16_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 4));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svld1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), (const int16_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 4));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((k >> i) & 1) ? r_.i16[i] : INT16_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_loadu_epi16
  #define _mm256_maskz_loadu_epi16(k, mem_addr) easysimd_mm256_maskz_loadu_epi16(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_loadu_epi32(easysimd__m256i src, easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_loadu_epi32(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svld1_s32(pg, (const int32_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svld1_s32(pg, (const int32_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : src_.i32[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_loadu_epi32
  #define _mm256_mask_loadu_epi32(src, k, mem_addr) easysimd_mm256_mask_loadu_epi32(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_loadu_epi32(easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_loadu_epi32(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (const int32_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), (const int32_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((k >> i) & 1) ? r_.i32[i] : INT32_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_loadu_epi32
  #define _mm256_maskz_loadu_epi32(k, mem_addr) easysimd_mm256_maskz_loadu_epi32(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_loadu_epi64(easysimd__m256i src, easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_loadu_epi64(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                                            svld1_s64(pg, (const int64_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                                            svld1_s64(pg, (const int64_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? r_.i64[i] : src_.i64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_loadu_epi64
  #define _mm256_mask_loadu_epi64(src, k, mem_addr) easysimd_mm256_mask_loadu_epi64(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_loadu_epi64(easysimd__mmask8 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_loadu_epi64(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (const int64_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), (const int64_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ((k >> i) & 1) ? r_.i64[i] : INT64_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_loadu_epi64
  #define _mm256_maskz_loadu_epi64(k, mem_addr) easysimd_mm256_maskz_loadu_epi64(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_loadu_ps (easysimd__m256 src, easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_loadu_ps(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svld1_f32(pg, (float32_t const *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svld1_f32(pg, (float32_t const *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      src_ = easysimd__m256_to_private(src),
      r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : src_.f32[i];
    }
    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_loadu_ps(src, k, mem_addr) easysimd_mm256_mask_loadu_ps(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_loadu_ps (easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_loadu_ps(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (float32_t const *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), (float32_t const *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #else
    easysimd__m256_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }
    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_loadu_ps(k, mem_addr) easysimd_mm256_maskz_loadu_ps(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_loadu_pd (easysimd__m256d src, easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_loadu_pd(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0),
                                            svld1_f64(pg, (float64_t const *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1),
                                            svld1_f64(pg, (float64_t const *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      src_ = easysimd__m256d_to_private(src),
      r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : src_.f64[i];
    }
    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_mask_loadu_pd(src, k, mem_addr) easysimd_mm256_mask_loadu_pd(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_loadu_pd (easysimd__mmask8 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_loadu_pd(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (float64_t const *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), (float64_t const *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6));
    return r;
  #else
    easysimd__m256d_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? r_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
    }
    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm256_maskz_loadu_pd(k, mem_addr) easysimd_mm256_maskz_loadu_pd(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_loadu_ps (void const * mem_addr) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0].neon_f32 = vld1q_f32(((float32_t const*)mem_addr) + 0);
    r.m128[1].neon_f32 = vld1q_f32(((float32_t const*)mem_addr) + 4);
    r.m128[2].neon_f32 = vld1q_f32(((float32_t const*)mem_addr) + 8);
    r.m128[3].neon_f32 = vld1q_f32(((float32_t const*)mem_addr) + 12);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if defined(EASYSIMD_BUG_CLANG_REV_298042)
      return _mm512_loadu_ps(EASYSIMD_ALIGN_CAST(const float *, mem_addr));
    #else
      return _mm512_loadu_ps(mem_addr);
    #endif
  #else
    easysimd__m512 r;
    easysimd_memcpy(&r, mem_addr, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_loadu_ps
  #define _mm512_loadu_ps(mem_addr) easysimd_mm512_loadu_ps(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_loadu_ps (easysimd__m512 src, easysimd__mmask16 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_loadu_ps(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0),
                                            svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1),
                                            svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2),
                                            svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3),
                                            svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private
      r_,
      src_ = easysimd__m512_to_private(src);
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : src_.f32[i];
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_loadu_ps
  #define _mm512_mask_loadu_ps(src, k, mem_addr) easysimd_mm512_mask_loadu_ps(src, k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_loadu_ps (easysimd__mmask16 k, void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_loadu_ps(k, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svld1_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), HEDLEY_STATIC_CAST(const float32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #else
    easysimd__m512_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? r_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_loadu_ps
  #define _mm512_maskz_loadu_ps(k, mem_addr) easysimd_mm512_maskz_loadu_ps(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_loadu_pd (void const * mem_addr) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6));
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 6));
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 6));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0].neon_f64 = vld1q_f64(((float64_t const*)mem_addr) + 0);
    r.m128d[1].neon_f64 = vld1q_f64(((float64_t const*)mem_addr) + 2);
    r.m128d[2].neon_f64 = vld1q_f64(((float64_t const*)mem_addr) + 4);
    r.m128d[3].neon_f64 = vld1q_f64(((float64_t const*)mem_addr) + 6);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    #if defined(EASYSIMD_BUG_CLANG_REV_298042)
      return _mm512_loadu_pd(EASYSIMD_ALIGN_CAST(const double *, mem_addr));
    #else
      return _mm512_loadu_pd(mem_addr);
    #endif
  #else
    easysimd__m512d r;
    easysimd_memcpy(&r, mem_addr, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_loadu_pd
  #define _mm512_loadu_pd(mem_addr) easysimd_mm512_loadu_pd(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_loadu_si512 (void const * mem_addr) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 0);
    r.m128i[1].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 4);
    r.m128i[2].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 8);
    r.m128i[3].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 12);
    return r;
  #elif defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_loadu_si512(HEDLEY_REINTERPRET_CAST(void const*, mem_addr));
  #else
    easysimd__m512i r;

    #if HEDLEY_GNUC_HAS_ATTRIBUTE(may_alias,3,3,0)
      HEDLEY_DIAGNOSTIC_PUSH
      EASYSIMD_DIAGNOSTIC_DISABLE_PACKED_
      struct easysimd_mm512_loadu_si512_s {
        __typeof__(r) v;
      } __attribute__((__packed__, __may_alias__));
      r = HEDLEY_REINTERPRET_CAST(const struct easysimd_mm512_loadu_si512_s *, mem_addr)->v;
      HEDLEY_DIAGNOSTIC_POP
    #else
      easysimd_memcpy(&r, mem_addr, sizeof(r));
    #endif

    return r;
  #endif
}
#define easysimd_mm512_loadu_epi8(mem_addr) easysimd_mm512_loadu_si512(mem_addr)
#define easysimd_mm512_loadu_epi16(mem_addr) easysimd_mm512_loadu_si512(mem_addr)
#define easysimd_mm512_loadu_epi32(mem_addr) easysimd_mm512_loadu_si512(mem_addr)
#define easysimd_mm512_loadu_epi64(mem_addr) easysimd_mm512_loadu_si512(mem_addr)
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_loadu_epi8
  #undef _mm512_loadu_epi16
  #define _mm512_loadu_epi8(a) easysimd_mm512_loadu_si512(a)
  #define _mm512_loadu_epi16(a) easysimd_mm512_loadu_si512(a)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_loadu_epi32
  #undef _mm512_loadu_epi64
  #undef _mm512_loadu_si512
  #define _mm512_loadu_si512(a) easysimd_mm512_loadu_si512(a)
  #define _mm512_loadu_epi32(a) easysimd_mm512_loadu_si512(a)
  #define _mm512_loadu_epi64(a) easysimd_mm512_loadu_si512(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i easysimd_mm512_mask_loadu_epi8(easysimd__m512i src, easysimd__mmask64 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_loadu_epi8(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    int8_t const* data_addr = (int8_t const*)mem_addr;
    uint8_t g_mask_epi8[16] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    uint8x16_t mask = vld1q_u8(g_mask_epi8);
    uint8x16_t k_vec[4];
    k_vec[0] = vcombine_u8(vdup_n_u8(k & 0xff), vdup_n_u8((k >> 8) & 0xff));
    k_vec[1] = vcombine_u8(vdup_n_u8((k >> 16) & 0xff), vdup_n_u8((k >> 24) & 0xff));
    k_vec[2] = vcombine_u8(vdup_n_u8((k >> 32) & 0xff), vdup_n_u8((k >> 40) & 0xff));
    k_vec[3] = vcombine_u8(vdup_n_u8((k >> 48) & 0xff), vdup_n_u8((k >> 56) & 0xff));
    r.m128i[0].neon_i8 = vbslq_s8(vtstq_u8(k_vec[0], mask), vld1q_s8(data_addr + 0), src.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vbslq_s8(vtstq_u8(k_vec[1], mask), vld1q_s8(data_addr + 16), src.m128i[1].neon_i8);
    r.m128i[2].neon_i8 = vbslq_s8(vtstq_u8(k_vec[2], mask), vld1q_s8(data_addr + 32), src.m128i[2].neon_i8);
    r.m128i[3].neon_i8 = vbslq_s8(vtstq_u8(k_vec[3], mask), vld1q_s8(data_addr + 48), src.m128i[3].neon_i8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0),
                                          svld1_s8(pg, HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 3)),
                                          src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1),
                                          svld1_s8(pg, HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 3)),
                                          src.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2),
                                          svld1_s8(pg, HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 3)),
                                          src.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3),
                                          svld1_s8(pg, HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 3)),
                                          src.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private
      r_,
      src_ = easysimd__m512i_to_private(src);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = ((k >> i) & 1) ? ((int8_t*)mem_addr)[i] : src_.i8[i];
    }
    return easysimd__m512i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_loadu_epi8
  #define _mm512_mask_loadu_epi8(src, k, mem_addr) easysimd_mm512_mask_loadu_epi8(src, k, , mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i easysimd_mm512_maskz_loadu_epi8(easysimd__mmask64 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_loadu_epi8(k, mem_addr);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint8_t const* data_addr = (uint8_t const*)mem_addr;
    static easysimd__m128i mask = {
      .u8 = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80}};
    uint8x16_t k_vec[4];
    uint8x8_t mk = vcreate_u8(k);
    k_vec[0] = vcombine_u8(vdup_lane_u8(mk, 0), vdup_lane_u8(mk, 1));
    k_vec[1] = vcombine_u8(vdup_lane_u8(mk, 2), vdup_lane_u8(mk, 3));
    k_vec[2] = vcombine_u8(vdup_lane_u8(mk, 4), vdup_lane_u8(mk, 5));
    k_vec[3] = vcombine_u8(vdup_lane_u8(mk, 6), vdup_lane_u8(mk, 7));
    r.m128i[0].neon_u8 = vandq_u8(vtstq_u8(k_vec[0], mask.neon_u8), vld1q_u8(data_addr));
    r.m128i[1].neon_u8 = vandq_u8(vtstq_u8(k_vec[1], mask.neon_u8), vld1q_u8(data_addr + 16));
    r.m128i[2].neon_u8 = vandq_u8(vtstq_u8(k_vec[2], mask.neon_u8), vld1q_u8(data_addr + 32));
    r.m128i[3].neon_u8 = vandq_u8(vtstq_u8(k_vec[3], mask.neon_u8), vld1q_u8(data_addr + 48));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svld1_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 3));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svld1_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 3));
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svld1_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 3));
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svld1_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 3));
    return r;
  #else
    easysimd__m512i_private r_;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = ((k >> i) & 1) ? ((int8_t*)mem_addr)[i] : INT8_C(0);
    }
    return easysimd__m512i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_loadu_epi8
  #define _mm512_maskz_loadu_epi8(k, mem_addr) easysimd_mm512_maskz_loadu_epi8(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i easysimd_mm512_maskz_loadu_epi16(easysimd__mmask32 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_loadu_epi16(k, mem_addr);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint16_t const* data_addr = (uint16_t const*)mem_addr;
    static easysimd__m128i mask = {
      .u16 = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80}};
    uint16x8_t mk = vmovl_u8(vcreate_u8(k));
    uint16x8_t k_vec[4];
    k_vec[0] = vdupq_laneq_u16(mk, 0);
    k_vec[1] = vdupq_laneq_u16(mk, 1);
    k_vec[2] = vdupq_laneq_u16(mk, 2);
    k_vec[3] = vdupq_laneq_u16(mk, 3);
    r.m128i[0].neon_u16 = vandq_u16(vtstq_u16(k_vec[0], mask.neon_u16), vld1q_u16(data_addr));
    r.m128i[1].neon_u16 = vandq_u16(vtstq_u16(k_vec[1], mask.neon_u16), vld1q_u16(data_addr + 8));
    r.m128i[2].neon_u16 = vandq_u16(vtstq_u16(k_vec[2], mask.neon_u16), vld1q_u16(data_addr + 16));
    r.m128i[3].neon_u16 = vandq_u16(vtstq_u16(k_vec[3], mask.neon_u16), vld1q_u16(data_addr + 24));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svld1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), HEDLEY_STATIC_CAST(const int16_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 4));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svld1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), HEDLEY_STATIC_CAST(const int16_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 4));
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svld1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), HEDLEY_STATIC_CAST(const int16_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 4));
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svld1_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), HEDLEY_STATIC_CAST(const int16_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 4));
    return r;
  #else
    easysimd__m512i_private r_;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = ((k >> i) & 1) ? ((int16_t*)mem_addr)[i] : INT16_C(0);
    }
    return easysimd__m512i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_loadu_epi16
  #define _mm512_maskz_loadu_epi16(k, mem_addr) easysimd_mm512_maskz_loadu_epi16(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i easysimd_mm512_maskz_loadu_epi32(easysimd__mmask32 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_loadu_epi32(k, mem_addr);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint32_t const* data_addr = (uint32_t const*)mem_addr;
    static easysimd__m128i mask = {
      .u32 = {0x01, 0x02, 0x04, 0x08}};
    uint32x4_t k_vec[4];
    k_vec[0] = vdupq_n_u32((k >>  0) & 0xFFFFFFFF);
    k_vec[1] = vdupq_n_u32((k >>  4) & 0xFFFFFFFF);
    k_vec[2] = vdupq_n_u32((k >>  8) & 0xFFFFFFFF);
    k_vec[3] = vdupq_n_u32((k >> 12) & 0xFFFFFFFF);
    r.m128i[0].neon_u32 = vandq_u32(vtstq_u32(k_vec[0], mask.neon_u32), vld1q_u32(data_addr));
    r.m128i[1].neon_u32 = vandq_u32(vtstq_u32(k_vec[1], mask.neon_u32), vld1q_u32(data_addr + 4));
    r.m128i[2].neon_u32 = vandq_u32(vtstq_u32(k_vec[2], mask.neon_u32), vld1q_u32(data_addr + 8));
    r.m128i[3].neon_u32 = vandq_u32(vtstq_u32(k_vec[3], mask.neon_u32), vld1q_u32(data_addr + 12));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svld1_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #else
    easysimd__m512i_private r_;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ((k >> i) & 1) ? ((int32_t*)mem_addr)[i] : INT32_C(0);
    }
    return easysimd__m512i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_loadu_epi32
  #define _mm512_maskz_loadu_epi32(k, mem_addr) easysimd_mm512_maskz_loadu_epi32(k, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i easysimd_mm512_mask_loadu_epi32(easysimd__m512i src, easysimd__mmask32 k, void const* mem_addr) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_loadu_epi32(src, k, mem_addr);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    uint32_t const* data_addr = (uint32_t const*)mem_addr;
    uint32_t g_mask_epi32[8] __attribute__((aligned(32))) = {0x01, 0x02, 0x04, 0x08};
    uint32x4_t mask = vld1q_u32(g_mask_epi32);
    uint32x4_t k_vec[4];
    k_vec[0] = vdupq_n_u32((k >>  0) & 0xFFFFFFFF);
    k_vec[1] = vdupq_n_u32((k >>  4) & 0xFFFFFFFF);
    k_vec[2] = vdupq_n_u32((k >>  8) & 0xFFFFFFFF);
    k_vec[3] = vdupq_n_u32((k >> 12) & 0xFFFFFFFF);
    r.m128i[0].neon_u32 = vbslq_u32(vtstq_u32(k_vec[0], mask), vld1q_u32(data_addr     ), src.m128[0].neon_u32);
    r.m128i[1].neon_u32 = vbslq_u32(vtstq_u32(k_vec[1], mask), vld1q_u32(data_addr +  4), src.m128[1].neon_u32);
    r.m128i[2].neon_u32 = vbslq_u32(vtstq_u32(k_vec[2], mask), vld1q_u32(data_addr +  8), src.m128[2].neon_u32);
    r.m128i[3].neon_u32 = vbslq_u32(vtstq_u32(k_vec[3], mask), vld1q_u32(data_addr + 12), src.m128[3].neon_u32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svld1_s32(svptrue_b32(), (const int32_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svld1_s32(svptrue_b32(), (const int32_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svld1_s32(svptrue_b32(), (const int32_t *)mem_addr + EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svld1_s32(svptrue_b32(), (const int32_t *)mem_addr + EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5)), src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private 
      r_,
      src_ = easysimd__m512_to_private(src);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ((k >> i) & 1) ? ((int32_t*)mem_addr)[i] : src_.i32[i];
    }
    return easysimd__m512i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_loadu_epi32
  #define _mm512_mask_loadu_epi32(src, k, mem_addr) easysimd_mm512_mask_loadu_epi32(src, k, mem_addr)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_LOADU_H) */
