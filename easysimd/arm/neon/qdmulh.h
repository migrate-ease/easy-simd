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
 *   2020      Sean Maher <seanptmaher@gmail.com> (Copyright owned by Google, LLC)
 */

#if !defined(EASYSIMD_ARM_NEON_QDMULH_H)
#define EASYSIMD_ARM_NEON_QDMULH_H

#include "types.h"

#include "combine.h"
#include "get_high.h"
#include "get_low.h"
#include "qdmull.h"
#include "reinterpret.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vqdmulhs_s32(int32_t a, int32_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqdmulhs_s32(a, b);
  #else
    int64_t tmp = easysimd_vqdmulls_s32(a, b);
    return HEDLEY_STATIC_CAST(int32_t, tmp >> 32);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqdmulhs_s32
  #define vqdmulhs_s32(a) easysimd_vqdmulhs_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vqdmulh_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqdmulh_s16(a, b);
  #else
    easysimd_int16x4_private r_;

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      easysimd_int16x8_private tmp_ =
        easysimd_int16x8_to_private(
          easysimd_vreinterpretq_s16_s32(
            easysimd_vqdmull_s16(a, b)
          )
        );

      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 1, 3, 5, 7);
    #else
      easysimd_int32x4_private tmp = easysimd_int32x4_to_private(easysimd_vqdmull_s16(a, b));

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(int16_t, tmp.values[i] >> 16);
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqdmulh_s16
  #define vqdmulh_s16(a, b) easysimd_vqdmulh_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vqdmulh_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqdmulh_s32(a, b);
  #else
    easysimd_int32x2_private r_;

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      easysimd_int32x4_private tmp_ =
        easysimd_int32x4_to_private(
          easysimd_vreinterpretq_s32_s64(
            easysimd_vqdmull_s32(a, b)
          )
        );

      r_.values = __builtin_shufflevector(tmp_.values, tmp_.values, 1, 3);
    #else
      easysimd_int32x2_private a_ = easysimd_int32x2_to_private(a);
      easysimd_int32x2_private b_ = easysimd_int32x2_to_private(b);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vqdmulhs_s32(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqdmulh_s32
  #define vqdmulh_s32(a, b) easysimd_vqdmulh_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vqdmulhq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqdmulhq_s16(a, b);
  #else
    return easysimd_vcombine_s16(easysimd_vqdmulh_s16(easysimd_vget_low_s16(a), easysimd_vget_low_s16(b)),
                              easysimd_vqdmulh_s16(easysimd_vget_high_s16(a), easysimd_vget_high_s16(b)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqdmulhq_s16
  #define vqdmulhq_s16(a, b) easysimd_vqdmulhq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vqdmulhq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqdmulhq_s32(a, b);
  #else
    return easysimd_vcombine_s32(easysimd_vqdmulh_s32(easysimd_vget_low_s32(a), easysimd_vget_low_s32(b)),
                              easysimd_vqdmulh_s32(easysimd_vget_high_s32(a), easysimd_vget_high_s32(b)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqdmulhq_s32
  #define vqdmulhq_s32(a, b) easysimd_vqdmulhq_s32((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QDMULH_H) */
