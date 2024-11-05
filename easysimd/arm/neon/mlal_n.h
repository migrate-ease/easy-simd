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
 */

#if !defined(EASYSIMD_ARM_NEON_MLAL_N_H)
#define EASYSIMD_ARM_NEON_MLAL_N_H

#include "movl.h"
#include "dup_n.h"
#include "mla.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmlal_n_s16(easysimd_int32x4_t a, easysimd_int16x4_t b, int16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlal_n_s16(a, b, c);
  #else
    return easysimd_vmlaq_s32(a, easysimd_vmovl_s16(b), easysimd_vdupq_n_s32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlal_n_s16
  #define vmlal_n_s16(a, b, c) easysimd_vmlal_n_s16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vmlal_n_s32(easysimd_int64x2_t a, easysimd_int32x2_t b, int32_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlal_n_s32(a, b, c);
  #else
    easysimd_int64x2_private
      r_,
      a_ = easysimd_int64x2_to_private(a),
      b_ = easysimd_int64x2_to_private(easysimd_vmovl_s32(b)),
      c_ = easysimd_int64x2_to_private(easysimd_vdupq_n_s64(c));

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = (b_.values * c_.values) + a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (b_.values[i] * c_.values[i]) + a_.values[i];
      }
    #endif

    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlal_n_s32
  #define vmlal_n_s32(a, b, c) easysimd_vmlal_n_s32((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmlal_n_u16(easysimd_uint32x4_t a, easysimd_uint16x4_t b, uint16_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlal_n_u16(a, b, c);
  #else
    return easysimd_vmlaq_u32(a, easysimd_vmovl_u16(b), easysimd_vdupq_n_u32(c));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlal_n_u16
  #define vmlal_n_u16(a, b, c) easysimd_vmlal_n_u16((a), (b), (c))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vmlal_n_u32(easysimd_uint64x2_t a, easysimd_uint32x2_t b, uint32_t c) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmlal_n_u32(a, b, c);
  #else
    easysimd_uint64x2_private
      r_,
      a_ = easysimd_uint64x2_to_private(a),
      b_ = easysimd_uint64x2_to_private(easysimd_vmovl_u32(b)),
      c_ = easysimd_uint64x2_to_private(easysimd_vdupq_n_u64(c));

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.values = (b_.values * c_.values) + a_.values;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (b_.values[i] * c_.values[i]) + a_.values[i];
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmlal_n_u32
  #define vmlal_n_u32(a, b, c) easysimd_vmlal_n_u32((a), (b), (c))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MLAL_N_H) */
