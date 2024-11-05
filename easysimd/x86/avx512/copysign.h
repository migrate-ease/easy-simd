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

#if !defined(EASYSIMD_X86_AVX512_COPYSIGN_H)
#define EASYSIMD_X86_AVX512_COPYSIGN_H

#include "types.h"
#include "mov.h"
#include "and.h"
#include "andnot.h"
#include "xor.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_x_mm512_copysign_ps(easysimd__m512 dest, easysimd__m512 src) {
  easysimd__m512_private
    r_,
    dest_ = easysimd__m512_to_private(dest),
    src_ = easysimd__m512_to_private(src);

  #if defined(easysimd_math_copysignf)
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_copysignf(dest_.f32[i], src_.f32[i]);
    }
  #else
    easysimd__m512 sgnbit = easysimd_mm512_xor_ps(easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0)), easysimd_mm512_set1_ps(-EASYSIMD_FLOAT32_C(0.0)));
    return easysimd_mm512_xor_ps(easysimd_mm512_and_ps(sgnbit, src), easysimd_mm512_andnot_ps(sgnbit, dest));
  #endif

  return easysimd__m512_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_x_mm512_copysign_pd(easysimd__m512d dest, easysimd__m512d src) {
  easysimd__m512d_private
    r_,
    dest_ = easysimd__m512d_to_private(dest),
    src_ = easysimd__m512d_to_private(src);

  #if defined(easysimd_math_copysign)
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_copysign(dest_.f64[i], src_.f64[i]);
    }
  #else
    easysimd__m512d sgnbit = easysimd_mm512_xor_pd(easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0)), easysimd_mm512_set1_pd(-EASYSIMD_FLOAT64_C(0.0)));
    return easysimd_mm512_xor_pd(easysimd_mm512_and_pd(sgnbit, src), easysimd_mm512_andnot_pd(sgnbit, dest));
  #endif

  return easysimd__m512d_from_private(r_);
}

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_COPYSIGN_H) */
