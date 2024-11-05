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
 *   2021      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_X86_AVX512_CVTT_H)
#define EASYSIMD_X86_AVX512_CVTT_H

#include "types.h"
#include "mov.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvttpd_epi64 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_cvttpd_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svcvt_s64_f64_x(svptrue_b64(), a.sve_f64);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
      r_.ni =
        _mm_set_epi64x(
          _mm_cvttsd_si64(_mm_unpackhi_pd(a_.nd, a_.nd)),
          _mm_cvttsd_si64(a_.nd)
        );
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i64 = vcvtq_s64_f64(a_.neon_f64);
    #elif defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i64, a_.f64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.f64[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvttpd_epi64
  #define _mm_cvttpd_epi64(a) easysimd_mm_cvttpd_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_cvttpd_epi64(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_mask_cvttpd_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsel_s64(EASYSIMD_MASK_TO_B64(k, 0), svcvt_s64_f64_x(svptrue_b64(), a.sve_f64), src.sve_i64);
    return r;
  #else
    return easysimd_mm_mask_mov_epi64(src, k, easysimd_mm_cvttpd_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvttpd_epi64
  #define _mm_mask_cvttpd_epi64(src, k, a) easysimd_mm_mask_cvttpd_epi64(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_cvttpd_epi64(easysimd__mmask8 k, easysimd__m128d a) { 
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm_maskz_cvttpd_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svcvt_s64_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi64(k, easysimd_mm_cvttpd_epi64(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_cvttpd_epi64
  #define _mm_maskz_cvttpd_epi64(k, a) easysimd_mm_maskz_cvttpd_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvttpd_epi64 (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_cvttpd_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.f64[i]);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvttpd_epi64
  #define _mm256_cvttpd_epi64(a) easysimd_mm256_cvttpd_epi64(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_cvttpd_epi64(easysimd__m256i src, easysimd__mmask8 k, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_mask_cvttpd_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k ,EASYSIMD_SV_INDEX_1), svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private r_,
                         src_ = easysimd__m256i_to_private(src);
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(int64_t, a_.f64[i]) : src_.i64[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvttpd_epi64
  #define _mm256_mask_cvttpd_epi64(src, k, a) easysimd_mm256_mask_cvttpd_epi64(src, k, a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_cvttpd_epi64(easysimd__mmask8 k, easysimd__m256d a) { 
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm256_maskz_cvttpd_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svcvt_s64_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svcvt_s64_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(int64_t, a_.f64[i]) : 0;
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_cvttpd_epi64
  #define _mm256_maskz_cvttpd_epi64(k, a) easysimd_mm256_maskz_cvttpd_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvttpd_epi64 (easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_cvttpd_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private r_;
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.f64[i]);
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvttpd_epi64
  #define _mm512_cvttpd_epi64(a) easysimd_mm512_cvttpd_epi64(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_cvttpd_epi64(easysimd__m512i src, easysimd__mmask8 k, easysimd__m512d a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_mask_cvttpd_epi64(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_0]), src.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(EASYSIMD_MASK_TO_B64(k ,EASYSIMD_SV_INDEX_1), svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_1]), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svsel_s64(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_2]), src.sve_i64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svsel_s64(EASYSIMD_MASK_TO_B64(k ,EASYSIMD_SV_INDEX_3), svcvt_s64_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_3]), src.sve_i64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private r_,
                         src_ = easysimd__m512i_to_private(src);
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(int64_t, a_.f64[i]) : src_.i64[i];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvttpd_epi64
  #define _mm512_mask_cvttpd_epi64(src, k, a) easysimd_mm512_mask_cvttpd_epi64(src, k, a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_cvttpd_epi64(easysimd__mmask8 k, easysimd__m512d a) { 
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    return _mm512_maskz_cvttpd_epi64(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svcvt_s64_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_0), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svcvt_s64_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_1), a.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_i64[EASYSIMD_SV_INDEX_2] = svcvt_s64_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_2), a.sve_f64[EASYSIMD_SV_INDEX_2]);
    r.sve_i64[EASYSIMD_SV_INDEX_3] = svcvt_s64_f64_z(EASYSIMD_MASK_TO_B64(k, EASYSIMD_SV_INDEX_3), a.sve_f64[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private r_;
    easysimd__m512d_private a_ = easysimd__m512d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(int64_t, a_.f64[i]) : 0;
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_cvttpd_epi64
  #define _mm512_maskz_cvttpd_epi64(k, a) easysimd_mm512_maskz_cvttpd_epi64(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_cvttps_epi32(easysimd__m128i src, easysimd__mmask8 k, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_mask_cvttps_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(k, 0), svcvt_s32_f32_x(svptrue_b32(), a.sve_f32), src.sve_i32);
    return r;
  #else
    return easysimd_mm_mask_mov_epi32(src, k, easysimd_mm_cvttps_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_cvttps_epi32
  #define _mm_mask_cvttps_epi32(src, k, a) easysimd_mm_mask_cvttps_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskz_cvttps_epi32(easysimd__mmask8 k, easysimd__m128 a) { 
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm_maskz_cvttps_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svcvt_s32_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32);
    return r;
  #else
    return easysimd_mm_maskz_mov_epi32(k, easysimd_mm_cvttps_epi32(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskz_cvttps_epi32
  #define _mm_maskz_cvttps_epi32(k, a) easysimd_mm_maskz_cvttps_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_cvttps_epi32(easysimd__m256i src, easysimd__mmask8 k, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_mask_cvttps_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svcvt_s32_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svcvt_s32_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private r_,
                         src_ = easysimd__m256i_to_private(src);
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(int32_t, a_.f32[i]) : src_.i32[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_cvttps_epi32
  #define _mm256_mask_cvttps_epi32(src, k, a) easysimd_mm256_mask_cvttps_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskz_cvttps_epi32(easysimd__mmask8 k, easysimd__m256 a) { 
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm256_maskz_cvttps_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svcvt_s32_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svcvt_s32_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(int32_t, a_.f32[i]) : 0;
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskz_cvttps_epi32
  #define _mm256_maskz_cvttps_epi32(k, a) easysimd_mm256_maskz_cvttps_epi32(k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_cvttps_epi32 (easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_cvttps_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svcvt_s32_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svcvt_s32_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svcvt_s32_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svcvt_s32_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private r_;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, a_.f32[i]);
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_cvttps_epi32
  #define _mm512_cvttps_epi32(a) easysimd_mm512_cvttps_epi32(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_mask_cvttps_epi32(easysimd__m512i src, easysimd__mmask16 k, easysimd__m512 a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_mask_cvttps_epi32(src, k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), svcvt_s32_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]), src.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), svcvt_s32_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), svcvt_s32_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_2]), src.sve_i32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svsel_s32(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), svcvt_s32_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_3]), src.sve_i32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private r_,
                         src_ = easysimd__m512i_to_private(src);
    easysimd__m512_private a_ = easysimd__m512_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(int32_t, a_.f32[i]) : src_.i32[i];
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_mask_cvttps_epi32
  #define _mm512_mask_cvttps_epi32(src, k, a) easysimd_mm512_mask_cvttps_epi32(src, k, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_maskz_cvttps_epi32(easysimd__mmask16 k, easysimd__m512 a) { 
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    return _mm512_maskz_cvttps_epi32(k, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m512i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svcvt_s32_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_0), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svcvt_s32_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_1), a.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_2] = svcvt_s32_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_2), a.sve_f32[EASYSIMD_SV_INDEX_2]);
    r.sve_i32[EASYSIMD_SV_INDEX_3] = svcvt_s32_f32_z(EASYSIMD_MASK_TO_B32(k, EASYSIMD_SV_INDEX_3), a.sve_f32[EASYSIMD_SV_INDEX_3]);
    return r;
  #else
    easysimd__m512i_private r_;
    easysimd__m512_private a_ = easysimd__m512_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = (k >> i) & 0x01 ? HEDLEY_STATIC_CAST(int32_t, a_.f32[i]) : 0;
    }

    return easysimd__m512i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_maskz_cvttps_epi32
  #define _mm512_maskz_cvttps_epi32(k, a) easysimd_mm512_maskz_cvttps_epi32(k, a)
#endif


EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_CVTT_H) */
