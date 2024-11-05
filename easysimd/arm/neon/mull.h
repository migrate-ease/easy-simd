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

#if !defined(EASYSIMD_ARM_NEON_MULL_H)
#define EASYSIMD_ARM_NEON_MULL_H

#include "types.h"
#include "mul.h"
#include "movl.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vmull_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmull_s8(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_vmulq_s16(easysimd_vmovl_s8(a), easysimd_vmovl_s8(b));
  #else
    easysimd_int16x8_private r_;
    easysimd_int8x8_private
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100761)
      __typeof__(r_.values) av, bv;
      EASYSIMD_CONVERT_VECTOR_(av, a_.values);
      EASYSIMD_CONVERT_VECTOR_(bv, b_.values);
      r_.values = av * bv;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(int16_t, a_.values[i]) * HEDLEY_STATIC_CAST(int16_t, b_.values[i]);
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmull_s8
  #define vmull_s8(a, b) easysimd_vmull_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vmull_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmull_s16(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_vmulq_s32(easysimd_vmovl_s16(a), easysimd_vmovl_s16(b));
  #else
    easysimd_int32x4_private r_;
    easysimd_int16x4_private
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100761)
      __typeof__(r_.values) av, bv;
      EASYSIMD_CONVERT_VECTOR_(av, a_.values);
      EASYSIMD_CONVERT_VECTOR_(bv, b_.values);
      r_.values = av * bv;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(int32_t, a_.values[i]) * HEDLEY_STATIC_CAST(int32_t, b_.values[i]);
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmull_s16
  #define vmull_s16(a, b) easysimd_vmull_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vmull_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmull_s32(a, b);
  #else
    easysimd_int64x2_private r_;
    easysimd_int32x2_private
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      __typeof__(r_.values) av, bv;
      EASYSIMD_CONVERT_VECTOR_(av, a_.values);
      EASYSIMD_CONVERT_VECTOR_(bv, b_.values);
      r_.values = av * bv;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(int64_t, a_.values[i]) * HEDLEY_STATIC_CAST(int64_t, b_.values[i]);
      }
    #endif

    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmull_s32
  #define vmull_s32(a, b) easysimd_vmull_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vmull_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmull_u8(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_vmulq_u16(easysimd_vmovl_u8(a), easysimd_vmovl_u8(b));
  #else
    easysimd_uint16x8_private r_;
    easysimd_uint8x8_private
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100761)
      __typeof__(r_.values) av, bv;
      EASYSIMD_CONVERT_VECTOR_(av, a_.values);
      EASYSIMD_CONVERT_VECTOR_(bv, b_.values);
      r_.values = av * bv;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint16_t, a_.values[i]) * HEDLEY_STATIC_CAST(uint16_t, b_.values[i]);
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmull_u8
  #define vmull_u8(a, b) easysimd_vmull_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vmull_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmull_u16(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_vmulq_u32(easysimd_vmovl_u16(a), easysimd_vmovl_u16(b));
  #else
    easysimd_uint32x4_private r_;
    easysimd_uint16x4_private
      a_ = easysimd_uint16x4_to_private(a),
      b_ = easysimd_uint16x4_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_GCC_100761)
      __typeof__(r_.values) av, bv;
      EASYSIMD_CONVERT_VECTOR_(av, a_.values);
      EASYSIMD_CONVERT_VECTOR_(bv, b_.values);
      r_.values = av * bv;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint32_t, a_.values[i]) * HEDLEY_STATIC_CAST(uint32_t, b_.values[i]);
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmull_u16
  #define vmull_u16(a, b) easysimd_vmull_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vmull_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vmull_u32(a, b);
  #else
    easysimd_uint64x2_private r_;
    easysimd_uint32x2_private
      a_ = easysimd_uint32x2_to_private(a),
      b_ = easysimd_uint32x2_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      __typeof__(r_.values) av, bv;
      EASYSIMD_CONVERT_VECTOR_(av, a_.values);
      EASYSIMD_CONVERT_VECTOR_(bv, b_.values);
      r_.values = av * bv;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint64_t, a_.values[i]) * HEDLEY_STATIC_CAST(uint64_t, b_.values[i]);
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vmull_u32
  #define vmull_u32(a, b) easysimd_vmull_u32((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_MULL_H) */
