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
 *   2020-2021 Evan Nemerson <evan@nemerson.com>
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 *   2020      Hidayat Khan <huk2209@gmail.com>
 *   2021      Andrew Rodriguez <anrodriguez@linkedin.com>
 */

#if !defined(EASYSIMD_X86_AVX512_CVT_H)
#define EASYSIMD_X86_AVX512_CVT_H

#include "types.h"
#include "mov.h"
#include "../../easysimd-f16.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_cvtepi32_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cvtepi32_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svcvt_f32_s32_z(pg, a.sve_i32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (easysimd_float32) a_.i32[i] : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_cvtepi32_ps(src, k, a) easysimd_mm_mask_cvtepi32_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_cvtepi32_ps (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_cvtepi32_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svcvt_f32_s32_z(pg, a.sve_i32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? (easysimd_float32) a_.i32[i] : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_cvtepi32_ps(k, a) easysimd_mm_maskz_cvtepi32_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_cvtepi32_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cvtepi32_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_f64_s32_z(pg, svtbl_s32(a.sve_i32, svdupq_n_u32(0, 0, 1, 0))), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (easysimd_float64) a_.i32[i] : src_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_cvtepi32_pd(src, k, a) easysimd_mm_mask_cvtepi32_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_cvtepi32_pd (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_cvtepi32_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_f64_s32_z(pg, svtbl_s32(a.sve_i32, svdupq_n_u32(0, 0, 1, 0))), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (easysimd_float64) a_.i32[i] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_cvtepi32_pd(k, a) easysimd_mm_maskz_cvtepi32_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_cvtepu32_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cvtepu32_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_f64_u32_z(pg, svtbl_u32(a.sve_u32, svdupq_n_u32(0, 0, 1, 0))), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (easysimd_float64) a_.u32[i] : src_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_cvtepu32_pd(src, k, a) easysimd_mm_mask_cvtepu32_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_cvtepu32_pd (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_cvtepu32_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_f64_u32_z(pg, svtbl_u32(a.sve_u32, svdupq_n_u32(0, 0, 1, 0))), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((k >> i) & 1) ? (easysimd_float64) a_.u32[i] : EASYSIMD_FLOAT64_C(0.0);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_cvtepu32_pd(k, a) easysimd_mm_maskz_cvtepu32_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtepi64_ps (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_cvtepi64_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp, r;
    tmp.sve_f32 = svcvt_f32_s64_z(svptrue_b32(), a.sve_i64);
    r.sve_f32 = svuzp1_f32(tmp.sve_f32, svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    easysimd_memset(&r_, 0, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.i64[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepi64_ps
  #define _mm_cvtepi64_ps(a) easysimd_mm_cvtepi64_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_cvtepi64_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_mask_cvtepi64_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(svcvt_f32_s64_z(pg, a.sve_i64), svdup_n_f32(0.0)), svdupq_n_f32(src.f32[0], src.f32[1], 0, 0));
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    easysimd_memset(&r_, 0, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.i64[i]) : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepi64_ps
  #define _mm_mask_cvtepi64_ps(src, k, a) easysimd_mm_mask_cvtepi64_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_cvtepi64_ps (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_maskz_cvtepi64_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(svcvt_f32_s64_z(pg, a.sve_i64), svdup_n_f32(0.0)), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    easysimd_memset(&r_, 0, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.i64[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_cvtepi64_ps
  #define _mm_maskz_cvtepi64_ps(k, a) easysimd_mm_maskz_cvtepi64_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtepu64_ps (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_cvtepu64_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp, r;
    tmp.sve_f32 = svcvt_f32_u64_z(svptrue_b32(), a.sve_u64);
    r.sve_f32 = svuzp1_f32(tmp.sve_f32, svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    easysimd_memset(&r_, 0, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.u64[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepu64_ps
  #define _mm_cvtepu64_ps(a) easysimd_mm_cvtepu64_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_cvtepu64_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_mask_cvtepu64_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(svcvt_f32_u64_z(pg, a.sve_u64), svdup_n_f32(0.0)), svdupq_n_f32(src.f32[0], src.f32[1], 0, 0));
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    easysimd_memset(&r_, 0, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.u64[i]) : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepu64_ps
  #define _mm_mask_cvtepu64_ps(src, k, a) easysimd_mm_mask_cvtepu64_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_cvtepu64_ps (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_maskz_cvtepu64_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(svcvt_f32_u64_z(pg, a.sve_u64), svdup_n_f32(0.0)), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    easysimd_memset(&r_, 0, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.u64[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_cvtepu64_ps
  #define _mm_maskz_cvtepu64_ps(k, a) easysimd_mm_maskz_cvtepu64_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cvtepi64_pd (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_cvtepi64_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svcvt_f64_s64_z(svptrue_b64(), a.sve_i64);
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      /* https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx */
      __m128i xH = _mm_srai_epi32(a_.ni, 16);
      #if defined(EASYSIMD_X86_SSE4_2_NATIVE)
        xH = _mm_blend_epi16(xH, _mm_setzero_si128(), 0x33);
      #else
        xH = _mm_and_si128(xH, _mm_set_epi16(~INT16_C(0), ~INT16_C(0), INT16_C(0), INT16_C(0), ~INT16_C(0), ~INT16_C(0), INT16_C(0), INT16_C(0)));
      #endif
      xH = _mm_add_epi64(xH, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.0)));
      const __m128i e = _mm_castpd_si128(_mm_set1_pd(0x0010000000000000));
      #if defined(EASYSIMD_X86_SSE4_2_NATIVE)
        __m128i xL = _mm_blend_epi16(a_.n, e, 0x88);
      #else
        __m128i m = _mm_set_epi16(INT16_C(0), ~INT16_C(0), ~INT16_C(0), ~INT16_C(0), INT16_C(0), ~INT16_C(0), ~INT16_C(0), ~INT16_C(0));
        __m128i xL = _mm_or_si128(_mm_and_si128(m, a_.ni), _mm_andnot_si128(m, e));
      #endif
      __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(442726361368656609280.0));
      return _mm_add_pd(f, _mm_castsi128_pd(xL));
    #elif defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.f64, a_.i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = HEDLEY_STATIC_CAST(easysimd_float64, a_.i64[i]);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepi64_pd
  #define _mm_cvtepi64_pd(a) easysimd_mm_cvtepi64_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_cvtepi64_pd(easysimd__m128d src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_mask_cvtepi64_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_f64_s64_z(svptrue_b64(), a.sve_i64), src.sve_f64);
    return r;
  #else
    return easysimd_mm_mask_mov_pd(src, k, easysimd_mm_cvtepi64_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepi64_pd
  #define _mm_mask_cvtepi64_pd(src, k, a) easysimd_mm_mask_cvtepi64_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_cvtepi64_pd(easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_maskz_cvtepi64_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_f64_s64_z(svptrue_b64(), a.sve_i64), svdup_n_f64(0.0));
    return r;
  #else
    return easysimd_mm_maskz_mov_pd(k, easysimd_mm_cvtepi64_pd(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_cvtepi64_pd
  #define _mm_maskz_cvtepi64_pd(k, a) easysimd_mm_maskz_cvtepi64_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cvtepu64_pd (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_cvtepu64_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svcvt_f64_u64_z(svptrue_b64(), a.sve_u64);
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = HEDLEY_STATIC_CAST(easysimd_float64, a_.u64[i]);
      }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepu64_pd
  #define _mm_cvtepu64_pd(a) easysimd_mm_cvtepu64_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_cvtepu64_pd (easysimd__m128d src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_mask_cvtepu64_pd(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_f64_u64_z(svptrue_b64(), a.sve_u64), src.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float64, a_.u64[i]) : src_.f64[i];
      }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepu64_pd
  #define _mm_mask_cvtepu64_pd(src, k, a) easysimd_mm_mask_cvtepu64_pd(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskz_cvtepu64_pd (easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_maskz_cvtepu64_pd(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_f64_u64_z(svptrue_b64(), a.sve_u64), svdup_n_f64(0.0));
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float64, a_.u64[i]) : EASYSIMD_FLOAT64_C(0.0);
      }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_cvtepu64_pd
  #define _mm_maskz_cvtepu64_pd(k, a) easysimd_mm_maskz_cvtepu64_pd(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_cvtph_ps(easysimd__m128 src, easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cvtph_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd__m128i tmp;
    easysimd_svbool_t pg = svptrue_b32();
    tmp.sve_u16= svtbl_u16(a.sve_u16, svdupq_n_u16(0, 0, 1, 0, 2, 0, 3, 0));
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svcvt_f32_f16_z(pg, tmp.sve_f16), src.sve_f32);
    return r;
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? easysimd_float16_to_float32(easysimd_uint16_as_float16(a_.u16[i])) : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_cvtph_ps(src, k, a) easysimd_mm_mask_cvtph_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_cvtph_ps(easysimd__mmask8 k, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_cvtph_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd__m128i tmp;
    easysimd_svbool_t pg = svptrue_b32();
    tmp.sve_u16= svtbl_u16(a.sve_u16, svdupq_n_u16(0, 0, 1, 0, 2, 0, 3, 0));
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svcvt_f32_f16_z(pg, tmp.sve_f16), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    easysimd__m128_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? easysimd_float16_to_float32(easysimd_uint16_as_float16(a_.u16[i])) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_cvtph_ps(k, a) easysimd_mm_maskz_cvtph_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_cvtpd_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_mask_cvtpd_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(svcvt_f32_f64_z(pg, a.sve_f64), svdup_n_f32(0.0)), svdupq_n_f32(src.f32[0], src.f32[1], 0, 0));
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    easysimd_memset(&r_, 0, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[i]) : src_.f32[i];
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_mask_cvtpd_ps(src, k, a) easysimd_mm_mask_cvtpd_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskz_cvtpd_ps (easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_maskz_cvtpd_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(svcvt_f32_f64_z(pg, a.sve_f64), svdup_n_f32(0.0)), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    easysimd_memset(&r_, 0, sizeof(r_));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[i]) : EASYSIMD_FLOAT32_C(0.0);
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_maskz_cvtpd_ps(k, a) easysimd_mm_maskz_cvtpd_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_mask_cvtepi64_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cvtepi64_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp1, tmp2, r;
    svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.i64[i]) : src_.f32[i];
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvtepi64_ps
  #define _mm256_mask_cvtepi64_ps(src, k, a) easysimd_mm256_mask_cvtepi64_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_maskz_cvtepi64_ps (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_cvtepi64_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp1, tmp2, r;
    svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.i64[i]) : EASYSIMD_FLOAT32_C(0.0);
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_cvtepi64_ps
  #define _mm256_maskz_cvtepi64_ps(k, a) easysimd_mm256_maskz_cvtepi64_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_mask_cvtepu64_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cvtepu64_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp1, tmp2, r;
    svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.u64[i]) : src_.f32[i];
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvtepu64_ps
  #define _mm256_mask_cvtepu64_ps(src, k, a) easysimd_mm256_mask_cvtepu64_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_maskz_cvtepu64_ps (easysimd__mmask8 k, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_cvtepu64_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp1, tmp2, r;
    svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.u64[i]) : EASYSIMD_FLOAT32_C(0.0);
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_cvtepu64_ps
  #define _mm256_maskz_cvtepu64_ps(k, a) easysimd_mm256_maskz_cvtepu64_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_mask_cvtpd_ps (easysimd__m128 src, easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_mask_cvtpd_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp1, tmp2, r;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), src.sve_f32);
    return r;
  #else
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      r_;
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[i]) : src_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvtpd_ps
  #define _mm256_mask_cvtpd_ps(src, k, a) easysimd_mm256_mask_cvtpd_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_maskz_cvtpd_ps (easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_maskz_cvtpd_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp1, tmp2, r;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_cvtpd_ps
  #define _mm256_maskz_cvtpd_ps(k, a) easysimd_mm256_maskz_cvtpd_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
  easysimd_mm512_cvtepi32_ps (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cvtepi32_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svcvt_f32_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svcvt_f32_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svcvt_f32_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svcvt_f32_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    r.m128[0].neon_f32 = vcvtq_f32_s32(a.m128[0].neon_i32);
    r.m128[1].neon_f32 = vcvtq_f32_s32(a.m128[1].neon_i32);
    r.m128[2].neon_f32 = vcvtq_f32_s32(a.m128[2].neon_i32);
    r.m128[3].neon_f32 = vcvtq_f32_s32(a.m128[3].neon_i32);
    return r;
  #else
    easysimd__m512_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.i32[i]);
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepi32_ps
  #define _mm512_cvtepi32_ps(a) easysimd_mm512_cvtepi32_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
  easysimd_mm512_mask_cvtepi32_ps (easysimd__m512 src, easysimd__mmask16 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cvtepi32_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svcvt_f32_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]), src.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svcvt_f32_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svcvt_f32_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_2]), src.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svcvt_f32_s32_x(pg, a.sve_i32[EASYSIMD_SV_INDEX_3]), src.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512_private r_,
                        src_ = easysimd__m512i_to_private(src);
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = (k >> i) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.i32[i]) : src_.f32[i];
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepi32_ps
  #define _mm512_mask_cvtepi32_ps(src, k, a) easysimd_mm512_mask_cvtepi32_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_cvtepi32_pd (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cvtepi32_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svcvt_f64_s64_z(pg, svld1sw_s64(pg, (const int32_t *)&(a.i32[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)])));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svcvt_f64_s64_z(pg, svld1sw_s64(pg, (const int32_t *)&(a.i32[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)])));
    r.sve_f64[EASYSIMD_SV_INDEX_2] = svcvt_f64_s64_z(pg, svld1sw_s64(pg, (const int32_t *)&(a.i32[EASYSIMD_SV_INDEX_2 * (__ARM_FEATURE_SVE_BITS >> 6)])));
    r.sve_f64[EASYSIMD_SV_INDEX_3] = svcvt_f64_s64_z(pg, svld1sw_s64(pg, (const int32_t *)&(a.i32[EASYSIMD_SV_INDEX_3 * (__ARM_FEATURE_SVE_BITS >> 6)])));
    return r;
  #elif 0 //defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    __asm__ __volatile__ (
      "scvtf v0.4s, %[a0].4s           \n\t"
      "scvtf v1.4s, %[a1].4s           \n\t"
      "fcvtl %[r0].2d, v0.2s           \n\t"
      "fcvtl %[r2].2d, v1.2s           \n\t"
      "mov v0.d[0], v0.d[1]            \n\t"
      "mov v1.d[0], v1.d[1]            \n\t"
      "fcvtl %[r1].2d, v0.2s           \n\t"
      "fcvtl %[r3].2d, v1.2s           \n\t"
      :[r0]"=w"(r.m128d[0].neon_f64), [r1]"=w"(r.m128d[1].neon_f64), [r2]"=w"(r.m128d[2].neon_f64), [r3]"=w"(r.m128d[3].neon_f64)
      :[a0]"w"(a.m128d[0].neon_i32), [a1]"w"(a.m128d[1].neon_i32)
      :"v0", "v1"
    );
    return r;
  #else
    easysimd__m512d_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = HEDLEY_STATIC_CAST(easysimd_float64, a_.i32[i]);
    }

    return easysimd__m512d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepi32_pd
  #define _mm512_cvtepi32_pd(a) easysimd_mm256_cvtepi32_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_cvtepi16_epi8 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cvtepi16_epi8(a);
  #else
    easysimd__m256i_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i8, a_.i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = HEDLEY_STATIC_CAST(int8_t, a_.i16[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepi16_epi8
  #define _mm512_cvtepi16_epi8(a) easysimd_mm512_cvtepi16_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_mask_cvtepi16_epi8 (easysimd__m256i src, easysimd__mmask32 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_mask_cvtepi16_epi8(src, k, a);
  #else
    return easysimd_mm256_mask_mov_epi8(src, k, easysimd_mm512_cvtepi16_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepi16_epi8
  #define _mm512_mask_cvtepi16_epi8(src, k, a) easysimd_mm512_mask_cvtepi16_epi8(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_maskz_cvtepi16_epi8 (easysimd__mmask32 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_maskz_cvtepi16_epi8(k, a);
  #else
    return easysimd_mm256_maskz_mov_epi8(k, easysimd_mm512_cvtepi16_epi8(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_cvtepi16_epi8
  #define _mm512_maskz_cvtepi16_epi8(k, a) easysimd_mm512_maskz_cvtepi16_epi8(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvtepi8_epi16 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cvtepi8_epi16(a);
  #else
    easysimd__m512i_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i16, a_.i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i8[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepi8_epi16
  #define _mm512_cvtepi8_epi16(a) easysimd_mm512_cvtepi8_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_cvtepi64_ps (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_cvtepi64_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128 tmp1, tmp2;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32);

    tmp1.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2]);
    tmp2.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32);
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.i64[i]);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepi64_ps
  #define _mm512_cvtepi64_ps(a) easysimd_mm512_cvtepi64_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_mask_cvtepi64_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_mask_cvtepi64_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128 tmp1, tmp2;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), src.sve_f32[EASYSIMD_SV_INDEX_0]);

    tmp1.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2]);
    tmp2.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      src_ = easysimd__m256_to_private(src),
      r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.i64[i]) : src_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepi64_ps
  #define _mm512_mask_cvtepi64_ps(src, k, a) easysimd_mm512_mask_cvtepi64_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_maskz_cvtepi64_ps (easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_maskz_cvtepi64_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128 tmp1, tmp2;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), svdup_n_f32(0.0));

    tmp1.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_2]);
    tmp2.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_3]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.i64[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_cvtepi64_ps
  #define _mm512_maskz_cvtepi64_ps(k, a) easysimd_mm512_mask_cvtepi64_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_cvtepu64_ps (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_cvtepu64_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128 tmp1, tmp2;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32);

    tmp1.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2]);
    tmp2.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32);
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.u64[i]);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepu64_ps
  #define _mm512_cvtepu64_ps(a) easysimd_mm512_cvtepu64_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_mask_cvtepu64_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_mask_cvtepu64_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128 tmp1, tmp2;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), src.sve_f32[EASYSIMD_SV_INDEX_0]);

    tmp1.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2]);
    tmp2.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      src_ = easysimd__m256_to_private(src),
      r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.u64[i]) : src_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepu64_ps
  #define _mm512_mask_cvtepu64_ps(src, k, a) easysimd_mm512_mask_cvtepu64_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_maskz_cvtepu64_ps (easysimd__mmask8 k, easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_maskz_cvtepu64_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128 tmp1, tmp2;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), svdup_n_f32(0.0));

    tmp1.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_2]);
    tmp2.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_3]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.u64[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_cvtepu64_ps
  #define _mm512_maskz_cvtepu64_ps(k, a) easysimd_mm512_maskz_cvtepu64_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm512_cvtepi64_epi32 (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cvtepi64_epi32(a);
  #else
    easysimd__m256i_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i32, a_.i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, a_.i64[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepi64_epi32
  #define _mm512_cvtepi64_epi32(a) easysimd_mm512_cvtepi64_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_cvtepu32_ps (easysimd__m512i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cvtepu32_ps(a);
  #else
    easysimd__m512_private r_;
    easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      for (size_t i = 0 ; i < (sizeof(r_.m128) / sizeof(r_.m128[0])) ; i++) {
        /* https://stackoverflow.com/a/34067907/501126 */
        const __m128 tmp = _mm_cvtepi32_ps(_mm_srli_epi32(a_.m128i[i], 1));
        r_.m128[i] =
          _mm_add_ps(
            _mm_add_ps(tmp, tmp),
            _mm_cvtepi32_ps(_mm_and_si128(a_.m128i[i], _mm_set1_epi32(1)))
          );
      }
    #elif defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.f32, a_.u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.f32[i] = HEDLEY_STATIC_CAST(float, a_.u32[i]);
      }
    #endif

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepu32_epi32
  #define _mm512_cvtepu32_epi32(a) easysimd_mm512_cvtepu32_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_cvtpd_ps (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cvtpd_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128 tmp1, tmp2;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32);

    tmp1.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2]);
    tmp2.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32);
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[i]);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtpd_ps
  #define _mm512_cvtpd_ps(a) easysimd_mm512_cvtpd_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_mask_cvtpd_ps (easysimd__m256 src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cvtpd_ps(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128 tmp1, tmp2;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), src.sve_f32[EASYSIMD_SV_INDEX_0]);

    tmp1.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2]);
    tmp2.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      src_ = easysimd__m256_to_private(src),
      r_;
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[i]) : src_.f32[i];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtpd_ps
  #define _mm512_mask_cvtpd_ps(src, k, a) easysimd_mm512_mask_cvtpd_ps(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm512_maskz_cvtpd_ps (easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_cvtpd_ps(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128 tmp1, tmp2;
    easysimd_svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), svdup_n_f32(0.0));

    tmp1.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_2]);
    tmp2.sve_f32 = svcvt_f32_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_3]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32), svdup_n_f32(0.0));
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((k >> i) & 1) ? HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_cvtpd_ps
  #define _mm512_maskz_cvtpd_ps(k, a) easysimd_mm512_maskz_cvtpd_ps(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_cvtph_ps(easysimd__m256i a) {
  #if defined(EASYSIMD_X86_F16C_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cvtph_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512 r;
    easysimd__m128i tmp;
    easysimd_svbool_t pg = svptrue_b32();
    tmp.sve_u16 = svtbl_u16(a.sve_u16[EASYSIMD_SV_INDEX_0], svdupq_n_u16(0, 0, 1, 0, 2, 0, 3, 0));
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svcvt_f32_f16_z(pg, tmp.sve_f16);

    tmp.sve_u16 = svtbl_u16(a.sve_u16[EASYSIMD_SV_INDEX_0], svdupq_n_u16(4, 0, 5, 0, 6, 0, 7, 0));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svcvt_f32_f16_z(pg, tmp.sve_f16);

    tmp.sve_u16 = svtbl_u16(a.sve_u16[EASYSIMD_SV_INDEX_1], svdupq_n_u16(0, 0, 1, 0, 2, 0, 3, 0));
    r.sve_f32[EASYSIMD_SV_INDEX_2] = svcvt_f32_f16_z(pg, tmp.sve_f16);

    tmp.sve_u16 = svtbl_u16(a.sve_u16[EASYSIMD_SV_INDEX_1], svdupq_n_u16(4, 0, 5, 0, 6, 0, 7, 0));
    r.sve_f32[EASYSIMD_SV_INDEX_3] = svcvt_f32_f16_z(pg, tmp.sve_f16);
    return r;
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    easysimd__m512_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_float16_to_float32(easysimd_uint16_as_float16(a_.u16[i]));
    }

    return easysimd__m512_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_F16C_ENABLE_NATIVE_ALIASES)
  #define _mm512_cvtph_ps(a) easysimd_mm512_cvtph_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvtepi16_epi32 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm512_cvtepi16_epi32(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_i32 = vmovl_s16(vget_low_s16(a.m128i[0].neon_i16));
    r.m128i[1].neon_i32 = vmovl_s16(vget_high_s16(a.m128i[0].neon_i16));
    r.m128i[2].neon_i32 = vmovl_s16(vget_low_s16(a.m128i[1].neon_i16));
    r.m128i[3].neon_i32 = vmovl_s16(vget_high_s16(a.m128i[1].neon_i16));
    return r;
  #else
    easysimd__m512i_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i32, a_.i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i16[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepi16_epi32
  #define _mm512_cvtepi16_epi32(a) easysimd_mm512_cvtepi16_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvtepu16_epi32 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm512_cvtepu16_epi32(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u32 = vmovl_u16(vget_low_u16(a.m128i[0].neon_u16));
    r.m128i[1].neon_u32 = vmovl_u16(vget_high_u16(a.m128i[0].neon_u16));
    r.m128i[2].neon_u32 = vmovl_u16(vget_low_u16(a.m128i[1].neon_u16));
    r.m128i[3].neon_u32 = vmovl_u16(vget_high_u16(a.m128i[1].neon_u16));
    return r;
  #else
    easysimd__m512i_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i32, a_.u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.u16[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepu16_epi32
  #define _mm512_cvtepu16_epi32(a) easysimd_mm512_cvtepu16_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvtepu32_epi64 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm512_cvtepu32_epi64(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u64 = vmovl_u32(vget_low_u32(a.m128i[0].neon_u32));
    r.m128i[1].neon_u64 = vmovl_u32(vget_high_u32(a.m128i[0].neon_u32));
    r.m128i[2].neon_u64 = vmovl_u32(vget_low_u32(a.m128i[1].neon_u32));
    r.m128i[3].neon_u64 = vmovl_u32(vget_high_u32(a.m128i[1].neon_u32));
    return r;
  #else
    easysimd__m512i_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i64, a_.u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.u32[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepu32_epi64
  #define _mm512_cvtepu32_epi64(a) easysimd_mm512_cvtepu32_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvtepu8_epi16 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512BW_NATIVE)
    return _mm512_cvtepu8_epi16(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    r.m128i[0].neon_u16 = vmovl_u8(vget_low_u8(a.m128i[0].neon_u8));
    r.m128i[1].neon_u16 = vmovl_u8(vget_high_u8(a.m128i[0].neon_u8));
    r.m128i[2].neon_u16 = vmovl_u8(vget_low_u8(a.m128i[1].neon_u8));
    r.m128i[3].neon_u16 = vmovl_u8(vget_high_u8(a.m128i[1].neon_u8));
    return r;
  #else
    easysimd__m512i_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i16, a_.u8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.u8[i];
      }
    #endif

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepu8_epi16
  #define _mm512_cvtepu8_epi16(a) easysimd_mm512_cvtepu8_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvtepu16_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm512_cvtepu16_epi64(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i res;
    res.m128i[0].neon_u64 = vmovl_u32(vget_low_u32(vmovl_u16(vget_low_u16(a.neon_u16))));
    res.m128i[2].neon_u64 = vmovl_u32(vget_low_u32(vmovl_u16(vget_high_u16(a.neon_u16))));
    res.m128i[1].neon_u64 = vmovl_u32(vget_high_u32(vmovl_u16(vget_low_u16(a.neon_u16))));
    res.m128i[3].neon_u64 = vmovl_u32(vget_high_u32(vmovl_u16(vget_high_u16(a.neon_u16))));
    return res;
  #else
    easysimd__m512i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u16) / sizeof(a_.u16[0])) ; i++) {
      r_.i64[i] = a_.u16[i];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepu16_epi64
  #define _mm512_cvtepu16_epi64(a) easysimd_mm512_cvtepu16_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvtepu8_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm512_cvtepu8_epi32(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i res;
    res.m128i[0].neon_u32 = vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(a.neon_u8))));
    res.m128i[2].neon_u32 = vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(a.neon_u8))));
    res.m128i[1].neon_u32 = vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(a.neon_u8))));
    res.m128i[3].neon_u32 = vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(a.neon_u8))));
    return res;
  #else
    easysimd__m512i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
      r_.i32[i] = a_.u8[i];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepu8_epi32
  #define _mm512_cvtepu8_epi32(a) easysimd_mm512_cvtepu8_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvtepu8_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cvtepu8_epi64(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i res;
    uint16x8_t temp_vec = vmovl_u8(vget_low_u8(a.neon_u8));
    res.m128i[0].neon_u64 = vmovl_u32(vget_low_u32(vmovl_u16(vget_low_u16(temp_vec))));
    res.m128i[2].neon_u64 = vmovl_u32(vget_low_u32(vmovl_u16(vget_high_u16(temp_vec))));
    res.m128i[1].neon_u64 = vmovl_u32(vget_high_u32(vmovl_u16(vget_low_u16(temp_vec))));
    res.m128i[3].neon_u64 = vmovl_u32(vget_high_u32(vmovl_u16(vget_high_u16(temp_vec))));
    return res;
  #else
    easysimd__m512i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0; i < (sizeof(r_.i64) / sizeof(r_.i64[0])); i++) {
      r_.i64[i] = a_.u8[i];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvtepu8_epi64
  #define _mm512_cvtepu8_epi64(a) easysimd_mm512_cvtepu8_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_mask_cvtepi16_storeu_epi8(void *base_addr, easysimd__mmask8 k, easysimd__m128i a) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_cvtepi16_storeu_epi8(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1b_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), (int8_t *)base_addr, a.sve_i16);
#else
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i16) / sizeof(a_.i16[0])); i++) {
    *((int8_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i16[i] & 0xFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepi16_storeu_epi8
  #define _mm_mask_cvtepi16_storeu_epi8(base_addr, k, a) easysimd_mm_mask_cvtepi16_storeu_epi8(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_mask_cvtepi32_storeu_epi8(void *base_addr, easysimd__mmask8 k, easysimd__m128i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_cvtepi32_storeu_epi8(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1b_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (int8_t *)base_addr, a.sve_i32);
#else
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i32) / sizeof(a_.i32[0])); i++) {
    *((int8_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i32[i] & 0xFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepi32_storeu_epi8
  #define _mm_mask_cvtepi32_storeu_epi8(base_addr, k, a) easysimd_mm_mask_cvtepi32_storeu_epi8(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_mask_cvtepi64_storeu_epi8(void *base_addr, easysimd__mmask8 k, easysimd__m128i a) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm_mask_cvtepi64_storeu_epi8(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1b_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int8_t *)base_addr, a.sve_i64);
#else
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i64) / sizeof(a_.i64[0])); i++) {
    *((int8_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i64[i] & 0xFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepi64_storeu_epi8
  #define _mm_mask_cvtepi64_storeu_epi8(base_addr, k, a) easysimd_mm_mask_cvtepi64_storeu_epi8(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_mask_cvtepi32_storeu_epi16(void *base_addr, easysimd__mmask8 k, easysimd__m128i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_cvtepi32_storeu_epi16(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1h_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (int16_t *)base_addr, a.sve_i32);
#else
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i32) / sizeof(a_.i32[0])); i++) {
    *((int16_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i32[i] & 0xFFFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepi32_storeu_epi16
  #define _mm_mask_cvtepi32_storeu_epi16(base_addr, k, a) easysimd_mm_mask_cvtepi32_storeu_epi16(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_mask_cvtepi64_storeu_epi16(void *base_addr, easysimd__mmask8 k, easysimd__m128i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_cvtepi64_storeu_epi16(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1h_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int16_t *)base_addr, a.sve_i64);
#else
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i64) / sizeof(a_.i64[0])); i++) {
    *((int16_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i64[i] & 0xFFFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepi64_storeu_epi16
  #define _mm_mask_cvtepi64_storeu_epi16(base_addr, k, a) easysimd_mm_mask_cvtepi64_storeu_epi16(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_mask_cvtepi64_storeu_epi32(void *base_addr, easysimd__mmask8 k, easysimd__m128i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm_mask_cvtepi64_storeu_epi32(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1w_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int32_t *)base_addr, a.sve_i64);
#else
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i64) / sizeof(a_.i64[0])); i++) {
    *((int32_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i64[i] & 0xFFFFFFFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvtepi64_storeu_epi32
  #define _mm_mask_cvtepi64_storeu_epi32(base_addr, k, a) easysimd_mm_mask_cvtepi64_storeu_epi32(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_cvtepi16_storeu_epi8(void *base_addr, easysimd__mmask16 k, easysimd__m256i a) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_cvtepi16_storeu_epi8(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1b_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 16) * EASYSIMD_SV_INDEX_0, a.sve_i16[EASYSIMD_SV_INDEX_0]);
  svst1b_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 16) * EASYSIMD_SV_INDEX_1, a.sve_i16[EASYSIMD_SV_INDEX_1]);
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i16) / sizeof(a_.i16[0])); i++) {
    *((int8_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i16[i] & 0xFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvtepi16_storeu_epi8
  #define _mm256_mask_cvtepi16_storeu_epi8(base_addr, k, a) easysimd_mm256_mask_cvtepi16_storeu_epi8(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_cvtepi32_storeu_epi8(void *base_addr, easysimd__mmask8 k, easysimd__m256i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_cvtepi32_storeu_epi8(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1b_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_0, a.sve_i32[EASYSIMD_SV_INDEX_0]);
  svst1b_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_1, a.sve_i32[EASYSIMD_SV_INDEX_1]);
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i32) / sizeof(a_.i32[0])); i++) {
    *((int8_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i32[i] & 0xFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvtepi32_storeu_epi8
  #define _mm256_mask_cvtepi32_storeu_epi8(base_addr, k, a) easysimd_mm256_mask_cvtepi32_storeu_epi8(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_cvtepi64_storeu_epi8(void *base_addr, easysimd__mmask8 k, easysimd__m256i a) {
#if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm256_mask_cvtepi64_storeu_epi8(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1b_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_0, a.sve_i64[EASYSIMD_SV_INDEX_0]);
  svst1b_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_1, a.sve_i64[EASYSIMD_SV_INDEX_1]);
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i64) / sizeof(a_.i64[0])); i++) {
    *((int8_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i64[i] & 0xFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvtepi64_storeu_epi8
  #define _mm256_mask_cvtepi64_storeu_epi8(base_addr, k, a) easysimd_mm256_mask_cvtepi64_storeu_epi8(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_cvtepi32_storeu_epi16(void *base_addr, easysimd__mmask8 k, easysimd__m256i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_cvtepi32_storeu_epi16(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1h_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_0, a.sve_i32[EASYSIMD_SV_INDEX_0]);
  svst1h_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_1, a.sve_i32[EASYSIMD_SV_INDEX_1]);
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i32) / sizeof(a_.i32[0])); i++) {
    *((int16_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i32[i] & 0xFFFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvtepi32_storeu_epi16
  #define _mm256_mask_cvtepi32_storeu_epi16(base_addr, k, a) easysimd_mm256_mask_cvtepi32_storeu_epi16(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_cvtepi64_storeu_epi16(void *base_addr, easysimd__mmask8 k, easysimd__m256i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_cvtepi64_storeu_epi16(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1h_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_0, a.sve_i64[EASYSIMD_SV_INDEX_0]);
  svst1h_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_1, a.sve_i64[EASYSIMD_SV_INDEX_1]);
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i64) / sizeof(a_.i64[0])); i++) {
    *((int16_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i64[i] & 0xFFFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvtepi64_storeu_epi16
  #define _mm256_mask_cvtepi64_storeu_epi16(base_addr, k, a) easysimd_mm256_mask_cvtepi64_storeu_epi16(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_mask_cvtepi64_storeu_epi32(void *base_addr, easysimd__mmask8 k, easysimd__m256i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  return _mm256_mask_cvtepi64_storeu_epi32(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1w_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int32_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_0, a.sve_i64[EASYSIMD_SV_INDEX_0]);
  svst1w_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), (int32_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_1, a.sve_i64[EASYSIMD_SV_INDEX_1]);
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i64) / sizeof(a_.i64[0])); i++) {
    *((int32_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i64[i] & 0xFFFFFFFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvtepi64_storeu_epi32
  #define _mm256_mask_cvtepi64_storeu_epi32(base_addr, k, a) easysimd_mm256_mask_cvtepi64_storeu_epi32(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_cvtepi16_storeu_epi8(void *base_addr, easysimd__mmask32 k, easysimd__m512i a) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  return _mm512_mask_cvtepi16_storeu_epi8(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1b_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_0), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 16) * EASYSIMD_SV_INDEX_0, a.sve_i16[EASYSIMD_SV_INDEX_0]);
  svst1b_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_1), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 16) * EASYSIMD_SV_INDEX_1, a.sve_i16[EASYSIMD_SV_INDEX_1]);
  svst1b_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_2), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 16) * EASYSIMD_SV_INDEX_2, a.sve_i16[EASYSIMD_SV_INDEX_2]);
  svst1b_s16(EASYSIMD_MASK_TO_B16(k, EASYSIMD_SV_INDEX_3), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 16) * EASYSIMD_SV_INDEX_3, a.sve_i16[EASYSIMD_SV_INDEX_3]);
#else
  easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i16) / sizeof(a_.i16[0])); i++) {
    *((int8_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i16[i] & 0xFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepi16_storeu_epi8
  #define _mm512_mask_cvtepi16_storeu_epi8(base_addr, k, a) easysimd_mm512_mask_cvtepi16_storeu_epi8(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_cvtepi32_storeu_epi8(void *base_addr, easysimd__mmask16 k, easysimd__m512i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_mask_cvtepi32_storeu_epi8(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1b_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_0, a.sve_i32[EASYSIMD_SV_INDEX_0]);
  svst1b_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_1, a.sve_i32[EASYSIMD_SV_INDEX_1]);
  svst1b_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_2, a.sve_i32[EASYSIMD_SV_INDEX_2]);
  svst1b_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_3, a.sve_i32[EASYSIMD_SV_INDEX_3]);
#else
  easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i32) / sizeof(a_.i32[0])); i++) {
    *((int8_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i32[i] & 0xFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepi32_storeu_epi8
  #define _mm512_mask_cvtepi32_storeu_epi8(base_addr, k, a) easysimd_mm512_mask_cvtepi32_storeu_epi8(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_cvtepi64_storeu_epi8(void *base_addr, easysimd__mmask8 k, easysimd__m512i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_mask_cvtepi64_storeu_epi8(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1b_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_0, a.sve_i64[EASYSIMD_SV_INDEX_0]);
  svst1b_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_1, a.sve_i64[EASYSIMD_SV_INDEX_1]);
  svst1b_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_2, a.sve_i64[EASYSIMD_SV_INDEX_2]);
  svst1b_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), (int8_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_3, a.sve_i64[EASYSIMD_SV_INDEX_3]);
#else
  easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i64) / sizeof(a_.i64[0])); i++) {
    *((int8_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i64[i] & 0xFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepi64_storeu_epi8
  #define _mm512_mask_cvtepi64_storeu_epi8(base_addr, k, a) easysimd_mm512_mask_cvtepi64_storeu_epi8(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_cvtepi32_storeu_epi16(void *base_addr, easysimd__mmask16 k, easysimd__m512i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_mask_cvtepi32_storeu_epi16(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1h_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_0, a.sve_i32[EASYSIMD_SV_INDEX_0]);
  svst1h_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_1, a.sve_i32[EASYSIMD_SV_INDEX_1]);
  svst1h_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_2, a.sve_i32[EASYSIMD_SV_INDEX_2]);
  svst1h_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_3, a.sve_i32[EASYSIMD_SV_INDEX_3]);
#else
  easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i32) / sizeof(a_.i32[0])); i++) {
    *((int16_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i32[i] & 0xFFFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepi32_storeu_epi16
  #define _mm512_mask_cvtepi32_storeu_epi16(base_addr, k, a) easysimd_mm512_mask_cvtepi32_storeu_epi16(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_cvtepi64_storeu_epi16(void *base_addr, easysimd__mmask8 k, easysimd__m512i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_mask_cvtepi64_storeu_epi16(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1h_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_0, a.sve_i64[EASYSIMD_SV_INDEX_0]);
  svst1h_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_1, a.sve_i64[EASYSIMD_SV_INDEX_1]);
  svst1h_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_2, a.sve_i64[EASYSIMD_SV_INDEX_2]);
  svst1h_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), (int16_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_3, a.sve_i64[EASYSIMD_SV_INDEX_3]);
#else
  easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i64) / sizeof(a_.i64[0])); i++) {
    *((int16_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i64[i] & 0xFFFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepi64_storeu_epi16
  #define _mm512_mask_cvtepi64_storeu_epi16(base_addr, k, a) easysimd_mm512_mask_cvtepi64_storeu_epi16(base_addr, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm512_mask_cvtepi64_storeu_epi32(void *base_addr, easysimd__mmask8 k, easysimd__m512i a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_mask_cvtepi64_storeu_epi32(base_addr, k, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svst1w_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), (int32_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_0, a.sve_i64[EASYSIMD_SV_INDEX_0]);
  svst1w_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), (int32_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_1, a.sve_i64[EASYSIMD_SV_INDEX_1]);
  svst1w_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), (int32_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_2, a.sve_i64[EASYSIMD_SV_INDEX_2]);
  svst1w_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), (int32_t *)base_addr + (__ARM_FEATURE_SVE_BITS / 64) * EASYSIMD_SV_INDEX_3, a.sve_i64[EASYSIMD_SV_INDEX_3]);
#else
  easysimd__m512i_private a_ = easysimd__m512i_to_private(a);

  for(size_t i = 0; i < (sizeof(a_.i64) / sizeof(a_.i64[0])); i++) {
    *((int32_t *)base_addr + i) = (k >> i) & 0x01 ? a_.i64[i] & 0xFFFFFFFF : 0;
  }
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvtepi64_storeu_epi32
  #define _mm512_mask_cvtepi64_storeu_epi32(base_addr, k, a) easysimd_mm512_mask_cvtepi64_storeu_epi32(base_addr, k, a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_AVX512_CVT_H) */
