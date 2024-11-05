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

#if !defined(EASYSIMD_ARM_NEON_DOT_H)
#define EASYSIMD_ARM_NEON_DOT_H

#include "types.h"

#include "add.h"
#include "combine.h"
#include "dup_n.h"
#include "get_low.h"
#include "get_high.h"
#include "paddl.h"
#include "movn.h"
#include "mull.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vdot_s32(easysimd_int32x2_t r, easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    return vdot_s32(r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return easysimd_vadd_s32(r, easysimd_vmovn_s64(easysimd_vpaddlq_s32(easysimd_vpaddlq_s16(easysimd_vmull_s8(a, b)))));
  #else
    easysimd_int32x2_private r_;
    easysimd_int8x8_private
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);
    for (int i = 0 ; i < 2 ; i++) {
      int32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for (int j = 0 ; j < 4 ; j++) {
        const int idx = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(int32_t, a_.values[idx]) * HEDLEY_STATIC_CAST(int32_t, b_.values[idx]);
      }
      r_.values[i] = acc;
    }
    return easysimd_vadd_s32(r, easysimd_int32x2_from_private(r_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdot_s32
  #define vdot_s32(r, a, b) easysimd_vdot_s32((r), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vdot_u32(easysimd_uint32x2_t r, easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    return vdot_u32(r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return easysimd_vadd_u32(r, easysimd_vmovn_u64(easysimd_vpaddlq_u32(easysimd_vpaddlq_u16(easysimd_vmull_u8(a, b)))));
  #else
    easysimd_uint32x2_private r_;
    easysimd_uint8x8_private
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    for (int i = 0 ; i < 2 ; i++) {
      uint32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for (int j = 0 ; j < 4 ; j++) {
        const int idx = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(uint32_t, a_.values[idx]) * HEDLEY_STATIC_CAST(uint32_t, b_.values[idx]);
      }
      r_.values[i] = acc;
    }
    return easysimd_vadd_u32(r, easysimd_uint32x2_from_private(r_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdot_u32
  #define vdot_u32(r, a, b) easysimd_vdot_u32((r), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vdotq_s32(easysimd_int32x4_t r, easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return easysimd_vaddq_s32(r,
                           easysimd_vcombine_s32(easysimd_vmovn_s64(easysimd_vpaddlq_s32(easysimd_vpaddlq_s16(easysimd_vmull_s8(easysimd_vget_low_s8(a), easysimd_vget_low_s8(b))))),
                                                              easysimd_vmovn_s64(easysimd_vpaddlq_s32(easysimd_vpaddlq_s16(easysimd_vmull_s8(easysimd_vget_high_s8(a), easysimd_vget_high_s8(b)))))));
  #else
    easysimd_int32x4_private r_;
    easysimd_int8x16_private
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);
    for (int i = 0 ; i < 4 ; i++) {
      int32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for (int j = 0 ; j < 4 ; j++) {
        const int idx = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(int32_t, a_.values[idx]) * HEDLEY_STATIC_CAST(int32_t, b_.values[idx]);
      }
      r_.values[i] = acc;
    }
    return easysimd_vaddq_s32(r, easysimd_int32x4_from_private(r_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdotq_s32
  #define vdotq_s32(r, a, b) easysimd_vdotq_s32((r), (a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vdotq_u32(easysimd_uint32x4_t r, easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(__ARM_FEATURE_DOTPROD)
    return vdotq_u32(r, a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return easysimd_vaddq_u32(r,
                           easysimd_vcombine_u32(easysimd_vmovn_u64(easysimd_vpaddlq_u32(easysimd_vpaddlq_u16(easysimd_vmull_u8(easysimd_vget_low_u8(a), easysimd_vget_low_u8(b))))),
                                              easysimd_vmovn_u64(easysimd_vpaddlq_u32(easysimd_vpaddlq_u16(easysimd_vmull_u8(easysimd_vget_high_u8(a), easysimd_vget_high_u8(b)))))));
  #else
    easysimd_uint32x4_private r_;
    easysimd_uint8x16_private
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);
    for (int i = 0 ; i < 4 ; i++) {
      uint32_t acc = 0;
      EASYSIMD_VECTORIZE_REDUCTION(+:acc)
      for (int j = 0 ; j < 4 ; j++) {
        const int idx = j + (i << 2);
        acc += HEDLEY_STATIC_CAST(uint32_t, a_.values[idx]) * HEDLEY_STATIC_CAST(uint32_t, b_.values[idx]);
      }
      r_.values[i] = acc;
    }
    return easysimd_vaddq_u32(r, easysimd_uint32x4_from_private(r_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(__ARM_FEATURE_DOTPROD))
  #undef vdotq_u32
  #define vdotq_u32(r, a, b) easysimd_vdotq_u32((r), (a), (b))
#endif


EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_DOT_H) */
