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
 *   2021      Zhi An Ng <zhin@google.com> (Copyright owned by Google, LLC)
 */

#if !defined(EASYSIMD_ARM_NEON_QDMULH_N_H)
#define EASYSIMD_ARM_NEON_QDMULH_N_H

#include "qdmulh.h"
#include "dup_n.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqdmulh_n_s16(a, b) vqdmulh_n_s16((a), (b))
#else
  #define easysimd_vqdmulh_n_s16(a, b) easysimd_vqdmulh_s16((a), easysimd_vdup_n_s16(b))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqdmulh_n_s16
  #define vqdmulh_n_s16(a, b) easysimd_vqdmulh_n_s16((a), (b))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqdmulh_n_s32(a, b) vqdmulh_n_s32((a), (b))
#else
  #define easysimd_vqdmulh_n_s32(a, b) easysimd_vqdmulh_s32((a), easysimd_vdup_n_s32(b))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqdmulh_n_s32
  #define vqdmulh_n_s32(a, b) easysimd_vqdmulh_n_s32((a), (b))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqdmulhq_n_s16(a, b) vqdmulhq_n_s16((a), (b))
#else
  #define easysimd_vqdmulhq_n_s16(a, b) easysimd_vqdmulhq_s16((a), easysimd_vdupq_n_s16(b))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqdmulhq_n_s16
  #define vqdmulhq_n_s16(a, b) easysimd_vqdmulhq_n_s16((a), (b))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqdmulhq_n_s32(a, b) vqdmulhq_n_s32((a), (b))
#else
  #define easysimd_vqdmulhq_n_s32(a, b) easysimd_vqdmulhq_s32((a), easysimd_vdupq_n_s32(b))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqdmulhq_n_s32
  #define vqdmulhq_n_s32(a, b) easysimd_vqdmulhq_n_s32((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QDMULH_N_H) */
