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
 *   2021      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_ARM_NEON_QSHRN_N_H)
#define EASYSIMD_ARM_NEON_QSHRN_N_H

#include "types.h"
#include "shr_n.h"
#include "qmovn.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vqshrns_n_s32(a, n) vqshrns_n_s32(a, n)
#else
  #define easysimd_vqshrns_n_s32(a, n) easysimd_vqmovns_s32(easysimd_x_vshrs_n_s32(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqshrns_n_s32
  #define vqshrns_n_s32(a, n) easysimd_vqshrns_n_s32(a, n)
#endif

#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vqshrns_n_u32(a, n) vqshrns_n_u32(a, n)
#else
  #define easysimd_vqshrns_n_u32(a, n) easysimd_vqmovns_u32(easysimd_x_vshrs_n_u32(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqshrns_n_u32
  #define vqshrns_n_u32(a, n) easysimd_vqshrns_n_u32(a, n)
#endif

#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vqshrnd_n_s64(a, n) vqshrnd_n_s64(a, n)
#else
  #define easysimd_vqshrnd_n_s64(a, n) easysimd_vqmovnd_s64(easysimd_vshrd_n_s64(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqshrnd_n_s64
  #define vqshrnd_n_s64(a, n) easysimd_vqshrnd_n_s64(a, n)
#endif

#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vqshrnd_n_u64(a, n) vqshrnd_n_u64(a, n)
#else
  #define easysimd_vqshrnd_n_u64(a, n) easysimd_vqmovnd_u64(easysimd_vshrd_n_u64(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqshrnd_n_u64
  #define vqshrnd_n_u64(a, n) easysimd_vqshrnd_n_u64(a, n)
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshrn_n_s16(a, n) vqshrn_n_s16((a), (n))
#else
  #define easysimd_vqshrn_n_s16(a, n) easysimd_vqmovn_s16(easysimd_vshrq_n_s16(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshrn_n_s16
  #define vqshrn_n_s16(a, n) easysimd_vqshrn_n_s16((a), (n))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshrn_n_s32(a, n) vqshrn_n_s32((a), (n))
#else
  #define easysimd_vqshrn_n_s32(a, n) easysimd_vqmovn_s32(easysimd_vshrq_n_s32(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshrn_n_s32
  #define vqshrn_n_s32(a, n) easysimd_vqshrn_n_s32((a), (n))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshrn_n_s64(a, n) vqshrn_n_s64((a), (n))
#else
  #define easysimd_vqshrn_n_s64(a, n) easysimd_vqmovn_s64(easysimd_vshrq_n_s64(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshrn_n_s64
  #define vqshrn_n_s64(a, n) easysimd_vqshrn_n_s64((a), (n))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshrn_n_u16(a, n) vqshrn_n_u16((a), (n))
#else
  #define easysimd_vqshrn_n_u16(a, n) easysimd_vqmovn_u16(easysimd_vshrq_n_u16(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshrn_n_u16
  #define vqshrn_n_u16(a, n) easysimd_vqshrn_n_u16((a), (n))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshrn_n_u32(a, n) vqshrn_n_u32((a), (n))
#else
  #define easysimd_vqshrn_n_u32(a, n) easysimd_vqmovn_u32(easysimd_vshrq_n_u32(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshrn_n_u32
  #define vqshrn_n_u32(a, n) easysimd_vqshrn_n_u32((a), (n))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshrn_n_u64(a, n) vqshrn_n_u64((a), (n))
#else
  #define easysimd_vqshrn_n_u64(a, n) easysimd_vqmovn_u64(easysimd_vshrq_n_u64(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshrn_n_u64
  #define vqshrn_n_u64(a, n) easysimd_vqshrn_n_u64((a), (n))
#endif


EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QSHRN_N_H) */
