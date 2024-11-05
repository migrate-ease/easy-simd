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

#if !defined(EASYSIMD_X86_AVX512_SETR4_H)
#define EASYSIMD_X86_AVX512_SETR4_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_setr4_epi32 (int32_t d, int32_t c, int32_t b, int32_t a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_setr4_epi32(d, c, b, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  r.sve_i32[EASYSIMD_SV_INDEX_0] = svdupq_n_s32(d, c, b, a);
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svdupq_n_s32(d, c, b, a);
  r.sve_i32[EASYSIMD_SV_INDEX_2] = svdupq_n_s32(d, c, b, a);
  r.sve_i32[EASYSIMD_SV_INDEX_3] = svdupq_n_s32(d, c, b, a);
  return r;
#else
  easysimd__m512i_private r_;

  r_.i32[ 0] = d;
  r_.i32[ 1] = c;
  r_.i32[ 2] = b;
  r_.i32[ 3] = a;
  r_.i32[ 4] = d;
  r_.i32[ 5] = c;
  r_.i32[ 6] = b;
  r_.i32[ 7] = a;
  r_.i32[ 8] = d;
  r_.i32[ 9] = c;
  r_.i32[10] = b;
  r_.i32[11] = a;
  r_.i32[12] = d;
  r_.i32[13] = c;
  r_.i32[14] = b;
  r_.i32[15] = a;

  return easysimd__m512i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_setr4_epi32
  #define _mm512_setr4_epi32(d,c,b,a) easysimd_mm512_setr4_epi32(d,c,b,a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_setr4_epi64 (int64_t d, int64_t c, int64_t b, int64_t a) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm512_setr4_epi64(d, c, b, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svdupq_n_s64(d, c);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svdupq_n_s64(b, a);
  r.sve_i64[EASYSIMD_SV_INDEX_2] = svdupq_n_s64(d, c);
  r.sve_i64[EASYSIMD_SV_INDEX_3] = svdupq_n_s64(b, a);
  return r;
#else
  easysimd__m512i_private r_;

  r_.i64[0] = d;
  r_.i64[1] = c;
  r_.i64[2] = b;
  r_.i64[3] = a;
  r_.i64[4] = d;
  r_.i64[5] = c;
  r_.i64[6] = b;
  r_.i64[7] = a;

  return easysimd__m512i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_setr4_epi64
  #define _mm512_setr4_epi64(d,c,b,a) easysimd_mm512_setr4_epi64(d,c,b,a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_mm512_setr4_ps (easysimd_float32 d, easysimd_float32 c, easysimd_float32 b, easysimd_float32 a) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512 r;
  r.sve_f32[EASYSIMD_SV_INDEX_0] = svdupq_n_f32(d, c, b, a);
  r.sve_f32[EASYSIMD_SV_INDEX_1] = r.sve_f32[EASYSIMD_SV_INDEX_0];
  r.sve_f32[EASYSIMD_SV_INDEX_2] = r.sve_f32[EASYSIMD_SV_INDEX_0];
  r.sve_f32[EASYSIMD_SV_INDEX_3] = r.sve_f32[EASYSIMD_SV_INDEX_0];
  return r;
#else
  easysimd__m512_private r_;

  r_.f32[ 0] = d;
  r_.f32[ 1] = c;
  r_.f32[ 2] = b;
  r_.f32[ 3] = a;
  r_.f32[ 4] = d;
  r_.f32[ 5] = c;
  r_.f32[ 6] = b;
  r_.f32[ 7] = a;
  r_.f32[ 8] = d;
  r_.f32[ 9] = c;
  r_.f32[10] = b;
  r_.f32[11] = a;
  r_.f32[12] = d;
  r_.f32[13] = c;
  r_.f32[14] = b;
  r_.f32[15] = a;

  return easysimd__m512_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_setr4_ps
  #define _mm512_setr4_ps(d,c,b,a) easysimd_mm512_setr4_ps(d,c,b,a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_mm512_setr4_pd (easysimd_float64 d, easysimd_float64 c, easysimd_float64 b, easysimd_float64 a) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512d r;
  r.sve_f64[EASYSIMD_SV_INDEX_0] = svdupq_n_f64(d, c);
  r.sve_f64[EASYSIMD_SV_INDEX_1] = svdupq_n_f64(b, a);
  r.sve_f64[EASYSIMD_SV_INDEX_2] = r.sve_f64[EASYSIMD_SV_INDEX_0];
  r.sve_f64[EASYSIMD_SV_INDEX_3] = r.sve_f64[EASYSIMD_SV_INDEX_1];
  return r;

#else
  easysimd__m512d_private r_;

  r_.f64[0] = d;
  r_.f64[1] = c;
  r_.f64[2] = b;
  r_.f64[3] = a;
  r_.f64[4] = d;
  r_.f64[5] = c;
  r_.f64[6] = b;
  r_.f64[7] = a;

  return easysimd__m512d_from_private(r_);
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _mm512_setr4_pd
  #define _mm512_setr4_pd(d,c,b,a) easysimd_mm512_setr4_pd(d,c,b,a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_SETR4_H) */
