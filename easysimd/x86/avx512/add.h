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

#if !defined(EASYSIMD_X86_AVX512_ADD_H)
#define EASYSIMD_X86_AVX512_ADD_H

#include "types.h"
#include "../avx2.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_add_epi8(easysimd__m128i src, easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_add_epi8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8 = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svadd_s8_z(pg, a.sve_i8, b.sve_i8), src.sve_i8);
    return r;
  #else
    return easysimd_mm_mask_mov_epi8(src, k, easysimd_mm_add_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_add_epi8
  #define _mm_mask_add_epi8(src, k, a, b) easysimd_mm_mask_add_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_add_epi8(easysimd__mmask16 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_add_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8, b.sve_i8);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi8(k, easysimd_mm_add_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_add_epi8
  #define _mm_maskz_add_epi8(k, a, b) easysimd_mm_maskz_add_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_add_epi16(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_mask_add_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svadd_s16_z(pg, a.sve_i16, b.sve_i16), src.sve_i16);
    return r;
  #else
    return easysimd_mm_mask_mov_epi16(src, k, easysimd_mm_add_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_add_epi16
  #define _mm_mask_add_epi16(src, k, a, b) easysimd_mm_mask_add_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_add_epi16(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm_maskz_add_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16, b.sve_i16);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi16(k, easysimd_mm_add_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_add_epi16
  #define _mm_maskz_add_epi16(k, a, b) easysimd_mm_maskz_add_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_add_epi32(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_add_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_s32_z(pg, a.sve_i32, b.sve_i32), src.sve_i32);
    return r;
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_add_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_add_epi32
  #define _mm_mask_add_epi32(src, k, a, b) easysimd_mm_mask_add_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_add_epi32(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_add_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svadd_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32, b.sve_i32);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_add_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_add_epi32
  #define _mm_maskz_add_epi32(k, a, b) easysimd_mm_maskz_add_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_add_epi64(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_add_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svadd_s64_z(pg, a.sve_i64, b.sve_i64), src.sve_i64);
    return r;
  #else
    return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_add_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_add_epi64
  #define _mm_mask_add_epi64(src, k, a, b) easysimd_mm_mask_add_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_add_epi64(easysimd__mmask8 k, easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_add_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svadd_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64, b.sve_i64);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_add_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_add_epi64
  #define _mm_maskz_add_epi64(k, a, b) easysimd_mm_maskz_add_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_add_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_add_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_f32_z(pg, a.sve_f32, b.sve_f32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i] + b_.f32[i] : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_mask_add_ps(src, k, a, b) easysimd_mm_mask_add_ps(src, k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_add_ps (easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_add_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_f32_z(pg, a.sve_f32, b.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i] + b_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
#  define _mm_maskz_add_ps(k, a, b) easysimd_mm_maskz_add_ps(k, (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_add_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_add_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svadd_f64_z(pg, a.sve_f64, b.sve_f64), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      src_ = easysimd__m128d_to_private(src),
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i] + b_.f64[i] : src_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_add_pd(src, k, a, b) easysimd_mm_mask_add_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_add_pd (easysimd__mmask8 k, easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_add_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svadd_f64_z(pg, a.sve_f64, b.sve_f64), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i] + b_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_add_pd(k, a, b) easysimd_mm_maskz_add_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_add_ss(easysimd__m128 src, easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(8,1,0))
    return _mm_mask_add_ss(src, k, a, b);
  #elif 1
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      r_ = easysimd__m128_to_private(a);

    r_.f32[0] = (k & 1) ? (a_.f32[0] + b_.f32[0]) : src_.f32[0];

    return easysimd__m128_from_private(r_);
  #else
    return easysimd_mm_move_ss(a, easysimd_mm_mask_mov_ps(src, k, easysimd_mm_add_ps(a, b)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_add_ss
  #define _mm_mask_add_ss(src, k, a, b) easysimd_mm_mask_add_ss(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_add_ss(easysimd__mmask8 k, easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(8,1,0))
    return _mm_maskz_add_ss(k, a, b);
  #elif 1
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      r_ = easysimd__m128_to_private(a);

    r_.f32[0] = (k & 1) ? (a_.f32[0] + b_.f32[0]) : 0.0f;

    return easysimd__m128_from_private(r_);
  #else
    return easysimd_mm_move_ss(a, easysimd_mm_maskz_mov_ps(k, easysimd_mm_add_ps(a, b)));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_add_ss
  #define _mm_maskz_add_ss(k, a, b) easysimd_mm_maskz_add_ss(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_add_epi8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_add_epi8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), src.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      src_ = easysimd__m256i_to_private(src),
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? a_.i8[i] + b_.i8[i] : src_.i8[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_add_epi8
  #define _mm256_mask_add_epi8(src, k, a, b) easysimd_mm256_mask_add_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_add_epi8 (easysimd__mmask32 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_add_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = ((k >> i) & 1) ? a_.i8[i] + b_.i8[i] : INT8_C(0);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_add_epi8
  #define _mm256_maskz_add_epi8(k, a, b) easysimd_mm256_maskz_add_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_add_epi16(easysimd__m256i src, easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_mask_add_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi16(src, k, easysimd_mm256_add_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_add_epi16
  #define _mm256_mask_add_epi16(src, k, a, b) easysimd_mm256_mask_add_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_add_epi16(easysimd__mmask16 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm256_maskz_add_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi16(k, easysimd_mm256_add_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_add_epi16
  #define _mm256_maskz_add_epi16(k, a, b) easysimd_mm256_maskz_add_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_add_epi32(easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_add_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svadd_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi32(src, k, easysimd_mm256_add_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_add_epi32
  #define _mm256_mask_add_epi32(src, k, a, b) easysimd_mm256_mask_add_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_add_epi32(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_add_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svadd_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svadd_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi32(k, easysimd_mm256_add_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_add_epi32
  #define _mm256_maskz_add_epi32(k, a, b) easysimd_mm256_maskz_add_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_add_epi64(easysimd__m256i src, easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_add_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svadd_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svadd_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_mask_mov_epi64(src, k, easysimd_mm256_add_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_add_epi64
  #define _mm256_mask_add_epi64(src, k, a, b) easysimd_mm256_mask_add_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_add_epi64(easysimd__mmask8 k, easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_add_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svadd_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svadd_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    return easysimd_mm256_maskz_mov_epi64(k, easysimd_mm256_add_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_add_epi64
  #define _mm256_maskz_add_epi64(k, a, b) easysimd_mm256_maskz_add_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_add_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_add_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      src_ = easysimd__m256_to_private(src),
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i] + b_.f32[i] : src_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_add_ps
  #define _mm256_mask_add_ps(src, k, a, b) easysimd_mm256_mask_add_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskz_add_ps (easysimd__mmask8 k, easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_add_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? a_.f32[i] + b_.f32[i] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_add_ps
  #define _mm256_maskz_add_ps(k, a, b) easysimd_mm256_maskz_add_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_add_pd (easysimd__m256d src, easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_add_pd(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      src_ = easysimd__m256d_to_private(src),
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i] + b_.f64[i] : src_.f64[i];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_add_pd
  #define _mm256_mask_add_pd(src, k, a, b) easysimd_mm256_mask_add_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskz_add_pd (easysimd__mmask8 k, easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_add_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svadd_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svadd_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? a_.f64[i] + b_.f64[i] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_add_pd
  #define _mm256_maskz_add_pd(k, a, b) easysimd_mm256_maskz_add_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_add_epi8 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_add_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i8 = vaddq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vaddq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    r.m128i[2].neon_i8 = vaddq_s8(a.m128i[2].neon_i8, b.m128i[2].neon_i8);
    r.m128i[3].neon_i8 = vaddq_s8(a.m128i[3].neon_i8, b.m128i[3].neon_i8);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = a_.i8 + b_.i8;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_add_epi8(a_.m256i[i], b_.m256i[i]);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_add_epi8
  #define _mm512_add_epi8(a, b) easysimd_mm512_add_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_add_epi8 (easysimd__m512i src, easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_add_epi8(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), svadd_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), src.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), svadd_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), src.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), svadd_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]), src.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svsel_s8(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), svadd_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]), src.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi8(src, k, easysimd_mm512_add_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_add_epi8
  #define _mm512_mask_add_epi8(src, k, a, b) easysimd_mm512_mask_add_epi8(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_add_epi8 (easysimd__mmask64 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_add_epi8(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_0), a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_1), a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    r.sve_i8[EASYSIMD_SV_INDEX_2] = svadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_2), a.sve_i8[EASYSIMD_SV_INDEX_2], b.sve_i8[EASYSIMD_SV_INDEX_2]);
    r.sve_i8[EASYSIMD_SV_INDEX_3] = svadd_s8_z(EASYSIMD_MASK_TO_B8(k, EASYSIMD_SV_INDEX_3), a.sve_i8[EASYSIMD_SV_INDEX_3], b.sve_i8[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi8(k, easysimd_mm512_add_epi8(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_add_epi8
  #define _mm512_maskz_add_epi8(k, a, b) easysimd_mm512_maskz_add_epi8(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_add_epi16 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_add_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i16 = vaddq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vaddq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    r.m128i[2].neon_i16 = vaddq_s16(a.m128i[2].neon_i16, b.m128i[2].neon_i16);
    r.m128i[3].neon_i16 = vaddq_s16(a.m128i[3].neon_i16, b.m128i[3].neon_i16);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = a_.i16 + b_.i16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_add_epi16(a_.m256i[i], b_.m256i[i]);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_add_epi16
  #define _mm512_add_epi16(a, b) easysimd_mm512_add_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_add_epi16 (easysimd__m512i src, easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_add_epi16(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), svadd_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), src.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), svadd_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), src.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), svadd_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]), src.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svsel_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), svadd_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]), src.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi16(src, k, easysimd_mm512_add_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_add_epi16
  #define _mm512_mask_add_epi16(src, k, a, b) easysimd_mm512_mask_add_epi16(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_add_epi16 (easysimd__mmask32 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_add_epi16(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    r.sve_i16[EASYSIMD_SV_INDEX_2] = svadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), a.sve_i16[EASYSIMD_SV_INDEX_2], b.sve_i16[EASYSIMD_SV_INDEX_2]);
    r.sve_i16[EASYSIMD_SV_INDEX_3] = svadd_s16_z(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), a.sve_i16[EASYSIMD_SV_INDEX_3], b.sve_i16[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi16(k, easysimd_mm512_add_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_add_epi16
  #define _mm512_maskz_add_epi16(k, a, b) easysimd_mm512_maskz_add_epi16(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_add_epi32 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_add_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svadd_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svadd_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svadd_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svadd_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i32 = vaddq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vaddq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    r.m128i[2].neon_i32 = vaddq_s32(a.m128i[2].neon_i32, b.m128i[2].neon_i32);
    r.m128i[3].neon_i32 = vaddq_s32(a.m128i[3].neon_i32, b.m128i[3].neon_i32);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_add_epi32(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 + b_.i32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_add_epi32(a_.m256i[i], b_.m256i[i]);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_add_epi32
  #define _mm512_add_epi32(a, b) easysimd_mm512_add_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_add_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_add_epi32(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svadd_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svadd_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]), src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svadd_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]), src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi32(src, k, easysimd_mm512_add_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_add_epi32
  #define _mm512_mask_add_epi32(src, k, a, b) easysimd_mm512_mask_add_epi32(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_add_epi32(easysimd__mmask16 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_add_epi32(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svadd_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svadd_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svadd_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_i32[EASYSIMD_SV_INDEX_2], b.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svadd_s32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_i32[EASYSIMD_SV_INDEX_3], b.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi32(k, easysimd_mm512_add_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_add_epi32
  #define _mm512_maskz_add_epi32(k, a, b) easysimd_mm512_maskz_add_epi32(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_add_epi64 (easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_add_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svadd_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svadd_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svadd_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svadd_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i64 = vaddq_s64(a.m128i[0].neon_i64, b.m128i[0].neon_i64);
    r.m128i[1].neon_i64 = vaddq_s64(a.m128i[1].neon_i64, b.m128i[1].neon_i64);
    r.m128i[2].neon_i64 = vaddq_s64(a.m128i[2].neon_i64, b.m128i[2].neon_i64);
    r.m128i[3].neon_i64 = vaddq_s64(a.m128i[3].neon_i64, b.m128i[3].neon_i64);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(256)
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_add_epi64(a_.m256i[i], b_.m256i[i]);
      }
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_CLANG_BAD_VI64_OPS)
      r_.i64 = a_.i64 + b_.i64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256i) / sizeof(r_.m256i[0])) ; i++) {
        r_.m256i[i] = easysimd_mm256_add_epi64(a_.m256i[i], b_.m256i[i]);
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_add_epi64
  #define _mm512_add_epi64(a, b) easysimd_mm512_add_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_add_epi64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_add_epi64(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svadd_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), svadd_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svadd_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]), src.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), svadd_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]), src.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_epi64(src, k, easysimd_mm512_add_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_add_epi64
  #define _mm512_mask_add_epi64(src, k, a, b) easysimd_mm512_mask_add_epi64(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_add_epi64(easysimd__mmask8 k, easysimd__m512i a, easysimd__m512i b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_add_epi64(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svadd_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svadd_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svadd_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_i64[EASYSIMD_SV_INDEX_2], b.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svadd_s64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_i64[EASYSIMD_SV_INDEX_3], b.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_epi64(k, easysimd_mm512_add_epi64(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_add_epi64
  #define _mm512_maskz_add_epi64(k, a, b) easysimd_mm512_maskz_add_epi64(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_add_ps (easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_add_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0].neon_f32 = vaddq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vaddq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    r.m128[2].neon_f32 = vaddq_f32(a.m128[2].neon_f32, b.m128[2].neon_f32);
    r.m128[3].neon_f32 = vaddq_f32(a.m128[3].neon_f32, b.m128[3].neon_f32);
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 + b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256) / sizeof(r_.m256[0])) ; i++) {
        r_.m256[i] = easysimd_mm256_add_ps(a_.m256[i], b_.m256[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_add_ps
  #define _mm512_add_ps(a, b) easysimd_mm512_add_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_add_round_ps (easysimd__m512 a, easysimd__m512 b, int rounding) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE_UNKNOWN)
    return _mm512_add_round_ps(a, b, rounding);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]));
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC)
    {
      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_roundevenf(a_.f32[i] + b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_floorf(a_.f32[i] + b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_ceilf(a_.f32[i] + b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_truncf(a_.f32[i] + b_.f32[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_nearbyintf(a_.f32[i] + b_.f32[i]);
        }
        break;
      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
        break;
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_add_round_ps
  #define _mm512_add_round_ps(a, b, rounding) easysimd_mm512_add_round_ps(a, b, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_mask_add_ps(easysimd__m512 src, easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_add_ps(src, k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svadd_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svadd_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svadd_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]), src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svadd_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]), src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_mask_mov_ps(src, k, easysimd_mm512_add_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_add_ps
  #define _mm512_mask_add_ps(src, k, a, b) easysimd_mm512_mask_add_ps(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_maskz_add_ps(easysimd__mmask16 k, easysimd__m512 a, easysimd__m512 b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_add_ps(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svadd_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svadd_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_ps(k, easysimd_mm512_add_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_add_ps
  #define _mm512_maskz_add_ps(k, a, b) easysimd_mm512_maskz_add_ps(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_add_pd (easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_add_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    r.m128d[0].neon_f64 = vaddq_f64(a.m128d[0].neon_f64, b.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vaddq_f64(a.m128d[1].neon_f64, b.m128d[1].neon_f64);
    r.m128d[2].neon_f64 = vaddq_f64(a.m128d[2].neon_f64, b.m128d[2].neon_f64);
    r.m128d[3].neon_f64 = vaddq_f64(a.m128d[3].neon_f64, b.m128d[3].neon_f64);
    return r;
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 + b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.m256d) / sizeof(r_.m256d[0])) ; i++) {
        r_.m256d[i] = easysimd_mm256_add_pd(a_.m256d[i], b_.m256d[i]);
      }
    #endif

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_add_pd
  #define _mm512_add_pd(a, b) easysimd_mm512_add_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_add_round_pd (easysimd__m512d a, easysimd__m512d b, int rounding) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE_UNKNOWN)
    return _mm512_add_round_pd(a, b, rounding);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]));
    r.sve_f64[EASYSIMD_SV_INDEX_2] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]));
    r.sve_f64[EASYSIMD_SV_INDEX_3] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]));
    return r;
  #else
    easysimd__m512d_private
      r_,
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC)
    {
      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_roundeven(a_.f64[i] + b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_floor(a_.f64[i] + b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_ceil(a_.f64[i] + b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_trunc(a_.f64[i] + b_.f64[i]);
        }
        break;
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_nearbyint(a_.f64[i] + b_.f64[i]);
        }
        break;
      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_pd());
        break;
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_add_round_pd
  #define _mm512_add_round_pd(a, b, rounding) easysimd_mm512_add_round_pd(a, b, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_mask_add_pd(easysimd__m512d src, easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_add_pd(src, k, a, b);
  #else
    return easysimd_mm512_mask_mov_pd(src, k, easysimd_mm512_add_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_add_pd
  #define _mm512_mask_add_pd(src, k, a, b) easysimd_mm512_mask_add_pd(src, k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_maskz_add_pd(easysimd__mmask8 k, easysimd__m512d a, easysimd__m512d b) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_add_pd(k, a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svadd_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svadd_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svadd_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svadd_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    return easysimd_mm512_maskz_mov_pd(k, easysimd_mm512_add_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_add_pd
  #define _mm512_maskz_add_pd(k, a, b) easysimd_mm512_maskz_add_pd(k, a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_addn_ps(easysimd__m512 a, easysimd__m512 b)
{
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b32();
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svneg_f32_z(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]));
    a.sve_f32[EASYSIMD_SV_INDEX_1] = svneg_f32_z(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]));
    a.sve_f32[EASYSIMD_SV_INDEX_2] = svneg_f32_z(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]));
    a.sve_f32[EASYSIMD_SV_INDEX_3] = svneg_f32_z(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]));
    return a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    a.m128[0].neon_f32 = vnegq_f32(vaddq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32));
    a.m128[1].neon_f32 = vnegq_f32(vaddq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32));
    a.m128[2].neon_f32 = vnegq_f32(vaddq_f32(a.m128[2].neon_f32, b.m128[2].neon_f32));
    a.m128[3].neon_f32 = vnegq_f32(vaddq_f32(a.m128[3].neon_f32, b.m128[3].neon_f32));
    return a;
  #else
    easysimd__m512_private
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      a_.f32[i] = -(a_.f32[i] + b_.f32[i]);
    }

    return easysimd__m512_from_private(a_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_addn_round_ps(easysimd__m512 a, easysimd__m512 b, int rounding)
{
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b32();
    a.sve_f32[EASYSIMD_SV_INDEX_0] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svneg_f32_z(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0])));
    a.sve_f32[EASYSIMD_SV_INDEX_1] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svneg_f32_z(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1])));
    a.sve_f32[EASYSIMD_SV_INDEX_2] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svneg_f32_z(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2])));
    a.sve_f32[EASYSIMD_SV_INDEX_3] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svneg_f32_z(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3])));
    return a;
  #else
    easysimd__m512_private
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC)
    {
      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          a_.f32[i] = easysimd_math_roundevenf(-(a_.f32[i] + b_.f32[i]));
        }
        break;
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          a_.f32[i] = easysimd_math_floorf(-(a_.f32[i] + b_.f32[i]));
        }
        break;
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          a_.f32[i] = easysimd_math_ceilf(-(a_.f32[i] + b_.f32[i]));
        }
        break;
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          a_.f32[i] = easysimd_math_truncf(-(a_.f32[i] + b_.f32[i]));
        }
        break;
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          a_.f32[i] = easysimd_math_nearbyintf(-(a_.f32[i] + b_.f32[i]));
        }
        break;
      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
        break;
    }

    return easysimd__m512_from_private(a_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_addn_pd(easysimd__m512d a, easysimd__m512d b)
{
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b64();
    a.sve_f64[EASYSIMD_SV_INDEX_0] = svneg_f64_z(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]));
    a.sve_f64[EASYSIMD_SV_INDEX_1] = svneg_f64_z(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]));
    a.sve_f64[EASYSIMD_SV_INDEX_2] = svneg_f64_z(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2]));
    a.sve_f64[EASYSIMD_SV_INDEX_3] = svneg_f64_z(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3]));
    return a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    a.m128d[0].neon_f64 = vnegq_f64(vaddq_f64(a.m128d[0].neon_f64, b.m128d[0].neon_f64));
    a.m128d[1].neon_f64 = vnegq_f64(vaddq_f64(a.m128d[1].neon_f64, b.m128d[1].neon_f64));
    a.m128d[2].neon_f64 = vnegq_f64(vaddq_f64(a.m128d[2].neon_f64, b.m128d[2].neon_f64));
    a.m128d[3].neon_f64 = vnegq_f64(vaddq_f64(a.m128d[3].neon_f64, b.m128d[3].neon_f64));
    return a;
  #else
    easysimd__m512d_private
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      a_.f64[i] = -(a_.f64[i] + b_.f64[i]);
    }

    return easysimd__m512d_from_private(a_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_addn_round_pd(easysimd__m512d a, easysimd__m512d b, int rounding)
{
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b64();
    a.sve_f64[EASYSIMD_SV_INDEX_0] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svneg_f64_z(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0])));
    a.sve_f64[EASYSIMD_SV_INDEX_1] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svneg_f64_z(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1])));
    a.sve_f64[EASYSIMD_SV_INDEX_2] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svneg_f64_z(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2], b.sve_f64[EASYSIMD_SV_INDEX_2])));
    a.sve_f64[EASYSIMD_SV_INDEX_3] = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(pg, svneg_f64_z(pg, svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3], b.sve_f64[EASYSIMD_SV_INDEX_3])));
    return a;
  #else
    easysimd__m512d_private
      a_ = easysimd__m512d_to_private(a),
      b_ = easysimd__m512d_to_private(b);

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC)
    {
      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
          a_.f64[i] = easysimd_math_roundeven(-(a_.f64[i] + b_.f64[i]));
        }
        break;
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
          a_.f64[i] = easysimd_math_floor(-(a_.f64[i] + b_.f64[i]));
        }
        break;
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
          a_.f64[i] = easysimd_math_ceil(-(a_.f64[i] + b_.f64[i]));
        }
        break;
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
          a_.f64[i] = easysimd_math_trunc(-(a_.f64[i] + b_.f64[i]));
        }
        break;
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
          a_.f64[i] = easysimd_math_nearbyint(-(a_.f64[i] + b_.f64[i]));
        }
        break;
      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_pd());
        break;
      }

    return easysimd__m512d_from_private(a_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_addsetc_epi32 (easysimd__m512i a, easysimd__m512i b, easysimd__mmask16 *k2_res) {
  #ifdef EASYSIMD_X86_AVX_NATIVE_UNKNOWN
    return _mm512_addsetc_epi32(a, b, k2_res);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    const size_t n = sizeof(a.u32) / sizeof(a.u32[0]);
    size_t i = 0;
    easysimd_svbool_t pg = svwhilelt_b32(i, n);
    sveuint32_t svfactor = svdupq_n_u32(1, 2, 4, 8);
    do {
      sveuint32_t
        svr,
        sva = svld1_u32(pg, &(a.u32[i])),
        svb = svld1_u32(pg, &(b.u32[i]));
      svr = svadd_u32_z(pg, sva, svb);
      svst1_u32(pg, &(r.u32[i]), svr);

      sveuint32_t
        svx = svlsr_n_u32_z(pg, sva, 31),
        svy = svlsr_n_u32_z(pg, svb, 31),
        svz = svlsr_n_u32_z(pg, svr, 31);
      sveuint32_t
        svt1 = svbic_u32_z(pg, svx, svz),
        svt2 = svand_u32_z(pg, svx, svy),
        svt3 = svbic_u32_z(pg, svy, svz);
      sveuint32_t
        svtmp1 = svorr_u32_z(pg, svt1, svt2),
        svtmp = svorr_u32_z(pg, svtmp1, svt3),
        svk = svmul_u32_z(pg, svtmp, svfactor);
      uint32_t ret = (uint32_t)svaddv_u32(pg, svk);
      *k2_res += (uint16_t)(ret << i);

      i += svcntw();
      pg = svwhilelt_b32(i, n);
    } while (svptest_any(svptrue_b32(), pg));
    return r;
  #elif 0 //defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r, p;
    r.m128i[0].neon_u32 = vaddq_u32(a.m128i[0].neon_u32, b.m128i[0].neon_u32);
    r.m128i[1].neon_u32 = vaddq_u32(a.m128i[1].neon_u32, b.m128i[1].neon_u32);
    r.m128i[2].neon_u32 = vaddq_u32(a.m128i[2].neon_u32, b.m128i[2].neon_u32);
    r.m128i[3].neon_u32 = vaddq_u32(a.m128i[3].neon_u32, b.m128i[3].neon_u32);
    p.m128i[0].neon_u32 = vcltq_u32(r.m128i[0].neon_u32, b.m128i[0].neon_u32);
    p.m128i[1].neon_u32 = vcltq_u32(r.m128i[1].neon_u32, b.m128i[1].neon_u32);
    p.m128i[2].neon_u32 = vcltq_u32(r.m128i[2].neon_u32, b.m128i[2].neon_u32);
    p.m128i[3].neon_u32 = vcltq_u32(r.m128i[3].neon_u32, b.m128i[3].neon_u32);
    EXTRACT_HB_32x16(p, k2_res);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] + b_.u32[i];
      *k2_res |= (uint16_t)((~(r_.u32[i] >> 31) & (a_.u32[i] >> 31)) | ((a_.u32[i] >> 31) & (b_.u32[i] >> 31)) | (~(r_.u32[i] >> 31) & (b_.u32[i] >> 31))) << i;
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_addsetc_epi32
  #define _mm512_addsetc_epi32(a, b, k2_res) easysimd_mm512_addsetc_epi32(a, b, k2_res)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_addsets_epi32 (easysimd__m512i a, easysimd__m512i b, easysimd__mmask16 *sign) {
  #if defined(EASYSIMD_X86_AVX_NATIVE_UNKNOWN)
    return _mm512_addsets_epi32(a, b, sign);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    easysimd_svbool_t pg = svptrue_b32();
    sveuint32_t svfactor = svdupq_n_u32(1, 2, 4, 8);
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svadd_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svadd_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    r.sve_u32[EASYSIMD_SV_INDEX_2] = svadd_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_2], b.sve_u32[EASYSIMD_SV_INDEX_2]);
    r.sve_u32[EASYSIMD_SV_INDEX_3] = svadd_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_3], b.sve_u32[EASYSIMD_SV_INDEX_3]);

    uint32_t
      r1 = ((uint32_t)svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, r.sve_u32[EASYSIMD_SV_INDEX_0], 31), svfactor))),
      r2 = ((uint32_t)svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, r.sve_u32[EASYSIMD_SV_INDEX_1], 31), svfactor))),
      r3 = ((uint32_t)svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, r.sve_u32[EASYSIMD_SV_INDEX_2], 31), svfactor))),
      r4 = ((uint32_t)svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, r.sve_u32[EASYSIMD_SV_INDEX_3], 31), svfactor)));
    *sign = (uint16_t)((r1 << (EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5))) + (r2 << (EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5))) + \
                       (r3 << (EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5))) + (r4 << (EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5))));
    return r;
  #elif 0 //defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r, p;
    r.m128i[0].neon_i32 = vaddq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vaddq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    r.m128i[2].neon_i32 = vaddq_s32(a.m128i[2].neon_i32, b.m128i[2].neon_i32);
    r.m128i[3].neon_i32 = vaddq_s32(a.m128i[3].neon_i32, b.m128i[3].neon_i32);
    p.m128i[0].neon_u32 = vandq_u32(a.m128i[0].neon_u32, b.m128i[0].neon_u32);
    p.m128i[1].neon_u32 = vandq_u32(a.m128i[1].neon_u32, b.m128i[1].neon_u32);
    p.m128i[2].neon_u32 = vandq_u32(a.m128i[2].neon_u32, b.m128i[2].neon_u32);
    p.m128i[3].neon_u32 = vandq_u32(a.m128i[3].neon_u32, b.m128i[3].neon_u32);
    EXTRACT_HB_32x16(p, sign);
    return r;
  #else
    easysimd__m512i_private
      r_,
      a_ = easysimd__m512i_to_private(a),
      b_ = easysimd__m512i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] + b_.u32[i];
      *sign |= (uint16_t)(r_.u32[i] >> 31) << i;
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_addsets_epi32
  #define _mm512_addsets_epi32(a, b, sign) easysimd_mm512_addsets_epi32(a, b, sign)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_addsets_ps (easysimd__m512 a, easysimd__m512 b, easysimd__mmask16 *sign) {
  #if defined(EASYSIMD_X86_AVX_NATIVE_UNKNOWN)
    return _mm512_addsets_ps(a, b, sign);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    sveuint32_t svfactor = svdupq_n_u32(1, 2, 4, 8);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]);

    uint32_t
      r1 = svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, svreinterpret_u32_f32(r.sve_f32[EASYSIMD_SV_INDEX_0]), 31), svfactor)),
      r2 = svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, svreinterpret_u32_f32(r.sve_f32[EASYSIMD_SV_INDEX_1]), 31), svfactor)),
      r3 = svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, svreinterpret_u32_f32(r.sve_f32[EASYSIMD_SV_INDEX_2]), 31), svfactor)),
      r4 = svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, svreinterpret_u32_f32(r.sve_f32[EASYSIMD_SV_INDEX_3]), 31), svfactor));
    *sign = (uint16_t)((r1 << (EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5))) + (r2 << (EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5))) + \
                       (r3 << (EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5))) + (r4 << (EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5))));
    return r;
  #elif 0 //defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    easysimd__m512i p;
    r.m128[0].neon_f32 = vaddq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vaddq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    r.m128[2].neon_f32 = vaddq_f32(a.m128[2].neon_f32, b.m128[2].neon_f32);
    r.m128[3].neon_f32 = vaddq_f32(a.m128[3].neon_f32, b.m128[3].neon_f32);
    p.m128[0].neon_u32 = vandq_u32(vreinterpretq_u32_f32(a.m128[0].neon_f32), vreinterpretq_u32_f32(b.m128[0].neon_f32));
    p.m128[1].neon_u32 = vandq_u32(vreinterpretq_u32_f32(a.m128[1].neon_f32), vreinterpretq_u32_f32(b.m128[1].neon_f32));
    p.m128[2].neon_u32 = vandq_u32(vreinterpretq_u32_f32(a.m128[2].neon_f32), vreinterpretq_u32_f32(b.m128[2].neon_f32));
    p.m128[3].neon_u32 = vandq_u32(vreinterpretq_u32_f32(a.m128[3].neon_f32), vreinterpretq_u32_f32(b.m128[3].neon_f32));
    EXTRACT_HB_32x16(p, sign);
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = a_.f32[i] + b_.f32[i];
      *sign |= (uint16_t)(r_.u32[i] >> 31) << i;
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_addsets_ps
  #define _mm512_addsets_ps(a, b, sign) easysimd_mm512_addsets_ps(a, b, sign)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_addsets_round_ps(easysimd__m512 a, easysimd__m512 b, easysimd__mmask16 *sign, int rounding)
{
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    sveuint32_t svfactor = svdupq_n_u32(1, 2, 4, 8);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_2], b.sve_f32[EASYSIMD_SV_INDEX_2]));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(pg, svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_3], b.sve_f32[EASYSIMD_SV_INDEX_3]));

    uint32_t
      r1 = svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, svreinterpret_u32_f32(r.sve_f32[EASYSIMD_SV_INDEX_0]), 31), svfactor)),
      r2 = svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, svreinterpret_u32_f32(r.sve_f32[EASYSIMD_SV_INDEX_1]), 31), svfactor)),
      r3 = svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, svreinterpret_u32_f32(r.sve_f32[EASYSIMD_SV_INDEX_2]), 31), svfactor)),
      r4 = svaddv_u32(pg, svmul_u32_z(pg, svlsr_n_u32_z(pg, svreinterpret_u32_f32(r.sve_f32[EASYSIMD_SV_INDEX_3]), 31), svfactor));
    *sign = (uint16_t)((r1 << (EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5))) + (r2 << (EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5))) + \
                       (r3 << (EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 5))) + (r4 << (EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 5))));
    return r;
  #else
    easysimd__m512_private
      r_,
      a_ = easysimd__m512_to_private(a),
      b_ = easysimd__m512_to_private(b);
    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC)
    {
      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_roundevenf((a_.f32[i] + b_.f32[i]));
          *sign |= (uint16_t)(r_.u32[i] >> 31) << i;
        }
        break;
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_floorf((a_.f32[i] + b_.f32[i]));
          *sign |= (uint16_t)(r_.u32[i] >> 31) << i;
        }
        break;
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_ceilf((a_.f32[i] + b_.f32[i]));
          *sign |= (uint16_t)(r_.u32[i] >> 31) << i;
        }
        break;
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_truncf((a_.f32[i] + b_.f32[i]));
          *sign |= (uint16_t)(r_.u32[i] >> 31) << i;
        }
        break;
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_nearbyintf((a_.f32[i] + b_.f32[i]));
          *sign |= (uint16_t)(r_.u32[i] >> 31) << i;
        }
        break;
      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm512_setzero_ps());
        break;
    }
    return easysimd__m512_from_private(r_);

  #endif  
}

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_ADD_H) */
