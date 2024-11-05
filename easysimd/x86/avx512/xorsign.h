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

/* This is a SIMDe extension which is not part of AVX-512.  It exists
 * because a lot of numerical methods in SIMDe have algoriths which do
 * something like:
 *
 *   float sgn = input < 0 ? -1 : 1;
 *   ...
 *   return res * sgn;
 *
 * Which can be replaced with a much more efficient call to xorsign:
 *
 *   return easysimd_x_mm512_xorsign_ps(res, input);
 *
 * While this was originally intended for use in SIMDe, please feel
 * free to use it in your code.
 */

#if !defined(EASYSIMD_X86_AVX512_XORSIGN_H)
#define EASYSIMD_X86_AVX512_XORSIGN_H

#include "types.h"
#include "mov.h"
#include "and.h"
#include "xor.h"
#include "set1.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512
easysimd_x_mm512_xorsign_ps(easysimd__m512 dest, easysimd__m512 src) {
  return easysimd_mm512_xor_ps(easysimd_mm512_and_ps(easysimd_mm512_set1_ps(-0.0f), src), dest);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512d
easysimd_x_mm512_xorsign_pd(easysimd__m512d dest, easysimd__m512d src) {
  return easysimd_mm512_xor_pd(easysimd_mm512_and_pd(easysimd_mm512_set1_pd(-0.0), src), dest);
}

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_XORSIGN_H) */
