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

#include "../easysimd-common.h"
#include "../easysimd-math.h"
#include "../easysimd-f16.h"

#if !defined(EASYSIMD_X86_F16C_H)
#define EASYSIMD_X86_F16C_H

#include "avx.h"

#if !defined(EASYSIMD_X86_PF16C_NATIVE) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
#  define EASYSIMD_X86_PF16C_ENABLE_NATIVE_ALIASES
#endif
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtps_ph(easysimd__m128 a, const int sae) {
  #if defined(EASYSIMD_X86_F16C_NATIVE)
    EASYSIMD_LCC_DISABLE_DEPRECATED_WARNINGS
    switch (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      case EASYSIMD_MM_FROUND_NO_EXC:
        return _mm_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC);
      default:
        return _mm_cvtps_ph(a, 0);
    }
    EASYSIMD_LCC_REVERT_DEPRECATED_WARNINGS
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_f16 = svuzp1_f16(svcvt_f16_f32_z(svptrue_b16(), a.sve_f32), svdup_n_f16(0.0));
    //r.sve_f16 = easysimd_fun_list_round_f16(svptrue_b16(), r.sve_f16, sae);
    return r;
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);
    easysimd__m128i_private r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128());

    HEDLEY_STATIC_CAST(void, sae);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
      r_.neon_f16 = vcombine_f16(vcvt_f16_f32(a_.neon_f32), vdup_n_f16(EASYSIMD_FLOAT16_C(0.0)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
        r_.u16[i] = easysimd_float16_as_uint16(easysimd_float16_from_float32(a_.f32[i]));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_F16C_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtps_ph(a, sae) easysimd_mm_cvtps_ph(a, sae)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtph_ps(easysimd__m128i a) {
  #if defined(EASYSIMD_X86_F16C_NATIVE)
    return _mm_cvtph_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd__m128i tmp;
    easysimd_svbool_t pg = svptrue_b32();
    tmp.sve_u16= svtbl_u16(a.sve_u16, svdupq_n_u16(0, 0, 1, 0, 2, 0, 3, 0));
    r.sve_f32 = svcvt_f32_f16_z(pg, tmp.sve_f16);
    return r;
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    easysimd__m128_private r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
      r_.neon_f32 = vcvt_f32_f16(vget_low_f16(a_.neon_f16));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_float16_to_float32(easysimd_uint16_as_float16(a_.u16[i]));
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_F16C_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtph_ps(a) easysimd_mm_cvtph_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm256_cvtps_ph(easysimd__m256 a, const int sae) {
  #if defined(EASYSIMD_X86_F16C_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    EASYSIMD_LCC_DISABLE_DEPRECATED_WARNINGS
    switch (sae & EASYSIMD_MM_FROUND_NO_EXC) {
      case EASYSIMD_MM_FROUND_NO_EXC:
        return _mm256_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC);
      default:
        return _mm256_cvtps_ph(a, 0);
    }
    EASYSIMD_LCC_REVERT_DEPRECATED_WARNINGS
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_f16 = svuzp1_f16(svcvt_f16_f32_z(svptrue_b16(), a.sve_f32[EASYSIMD_SV_INDEX_0]), svcvt_f16_f32_z(svptrue_b16(), a.sve_f32[EASYSIMD_SV_INDEX_1]));
    //r.sve_f16 = easysimd_fun_list_round_f16(svptrue_b16(), r.sve_f16, sae);
    return r;
  #else
    easysimd__m256_private a_ = easysimd__m256_to_private(a);
    easysimd__m128i_private r_;

    HEDLEY_STATIC_CAST(void, sae);

    #if defined(EASYSIMD_X86_F16C_NATIVE)
      return _mm_castps_si128(_mm_movelh_ps(
        _mm_castsi128_ps(_mm_cvtps_ph(a_.m128[0], EASYSIMD_MM_FROUND_NO_EXC)),
        _mm_castsi128_ps(_mm_cvtps_ph(a_.m128[1], EASYSIMD_MM_FROUND_NO_EXC))
      ));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
        r_.u16[i] = easysimd_float16_as_uint16(easysimd_float16_from_float32(a_.f32[i]));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_F16C_ENABLE_NATIVE_ALIASES)
  #define _mm256_cvtps_ph(a, sae) easysimd_mm256_cvtps_ph(a, sae)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_cvtph_ps(easysimd__m128i a) {
  #if defined(EASYSIMD_X86_F16C_NATIVE) && defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cvtph_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd__m128i tmp;
    easysimd_svbool_t pg = svptrue_b32();
    tmp.sve_u16 = svtbl_u16(a.sve_u16, svdupq_n_u16(0, 0, 1, 0, 2, 0, 3, 0));
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svcvt_f32_f16_z(pg, tmp.sve_f16);

    tmp.sve_u16 = svtbl_u16(a.sve_u16, svdupq_n_u16(4, 0, 5, 0, 6, 0, 7, 0));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svcvt_f32_f16_z(pg, tmp.sve_f16);
    return r;
  #elif defined(EASYSIMD_X86_F16C_NATIVE)
    return _mm256_setr_m128(
      _mm_cvtph_ps(a),
      _mm_cvtph_ps(_mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(a), 0xee)))
    );
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    easysimd__m256_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_float16_to_float32(easysimd_uint16_as_float16(a_.u16[i]));
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_F16C_ENABLE_NATIVE_ALIASES)
  #define _mm256_cvtph_ps(a) easysimd_mm256_cvtph_ps(a)
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_F16C_H) */
