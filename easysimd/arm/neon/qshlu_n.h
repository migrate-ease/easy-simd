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
 *   2021      Atharva Nimbalkar <atharvakn@gmail.com>
 */

#if !defined(EASYSIMD_ARM_NEON_QSHLU_N_H)
#define EASYSIMD_ARM_NEON_QSHLU_N_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vqshlub_n_s8(int8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  uint8_t r = HEDLEY_STATIC_CAST(uint8_t, a << n);
  r |= (((r >> n) != HEDLEY_STATIC_CAST(uint8_t, a)) ? UINT8_MAX : 0);
  return (a < 0) ? 0 : r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vqshlub_n_s8(a, n) HEDLEY_STATIC_CAST(uint8_t, vqshlub_n_s8(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqshlub_n_s8
  #define vqshlub_n_s8(a, n) easysimd_vqshlub_n_s8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vqshlus_n_s32(int32_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 31) {
  uint32_t r = HEDLEY_STATIC_CAST(uint32_t, a << n);
  r |= (((r >> n) != HEDLEY_STATIC_CAST(uint32_t, a)) ? UINT32_MAX : 0);
  return (a < 0) ? 0 : r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vqshlus_n_s32(a, n) HEDLEY_STATIC_CAST(uint32_t, vqshlus_n_s32(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqshlus_n_s32
  #define vqshlus_n_s32(a, n) easysimd_vqshlus_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vqshlud_n_s64(int64_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 63) {
  uint32_t r = HEDLEY_STATIC_CAST(uint32_t, a << n);
  r |= (((r >> n) != HEDLEY_STATIC_CAST(uint32_t, a)) ? UINT32_MAX : 0);
  return (a < 0) ? 0 : r;
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vqshlud_n_s64(a, n) HEDLEY_STATIC_CAST(uint64_t, vqshlud_n_s64(a, n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqshlud_n_s64
  #define vqshlud_n_s64(a, n) easysimd_vqshlud_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vqshlu_n_s8(easysimd_int8x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
    easysimd_int8x8_private a_ = easysimd_int8x8_to_private(a);
    easysimd_uint8x8_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      __typeof__(r_.values) shifted = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values) << n;

      __typeof__(r_.values) overflow = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (shifted >> n) != HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values));

      r_.values = (shifted & ~overflow) | overflow;

      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values >= 0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint8_t, a_.values[i] << n);
        r_.values[i] |= (((r_.values[i] >> n) != HEDLEY_STATIC_CAST(uint8_t, a_.values[i])) ? UINT8_MAX : 0);
        r_.values[i] = (a_.values[i] < 0) ? 0 : r_.values[i];
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshlu_n_s8(a, n) vqshlu_n_s8(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshlu_n_s8
  #define vqshlu_n_s8(a, n) easysimd_vqshlu_n_s8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vqshlu_n_s16(easysimd_int16x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 15) {
    easysimd_int16x4_private a_ = easysimd_int16x4_to_private(a);
    easysimd_uint16x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      __typeof__(r_.values) shifted = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values) << n;

      __typeof__(r_.values) overflow = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (shifted >> n) != HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values));

      r_.values = (shifted & ~overflow) | overflow;

      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values >= 0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint16_t, a_.values[i] << n);
        r_.values[i] |= (((r_.values[i] >> n) != HEDLEY_STATIC_CAST(uint16_t, a_.values[i])) ? UINT16_MAX : 0);
        r_.values[i] = (a_.values[i] < 0) ? 0 : r_.values[i];
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshlu_n_s16(a, n) vqshlu_n_s16(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshlu_n_s16
  #define vqshlu_n_s16(a, n) easysimd_vqshlu_n_s16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vqshlu_n_s32(easysimd_int32x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 31) {
    easysimd_int32x2_private a_ = easysimd_int32x2_to_private(a);
    easysimd_uint32x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      __typeof__(r_.values) shifted = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values) << n;

      __typeof__(r_.values) overflow = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (shifted >> n) != HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values));

      r_.values = (shifted & ~overflow) | overflow;

      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values >= 0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint32_t, a_.values[i] << n);
        r_.values[i] |= (((r_.values[i] >> n) != HEDLEY_STATIC_CAST(uint32_t, a_.values[i])) ? UINT32_MAX : 0);
        r_.values[i] = (a_.values[i] < 0) ? 0 : r_.values[i];
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshlu_n_s32(a, n) vqshlu_n_s32(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshlu_n_s32
  #define vqshlu_n_s32(a, n) easysimd_vqshlu_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vqshlu_n_s64(easysimd_int64x1_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 63) {
    easysimd_int64x1_private a_ = easysimd_int64x1_to_private(a);
    easysimd_uint64x1_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      __typeof__(r_.values) shifted = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values) << n;

      __typeof__(r_.values) overflow = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (shifted >> n) != HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values));

      r_.values = (shifted & ~overflow) | overflow;

      r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values >= 0));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = HEDLEY_STATIC_CAST(uint64_t, a_.values[i] << n);
        r_.values[i] |= (((r_.values[i] >> n) != HEDLEY_STATIC_CAST(uint64_t, a_.values[i])) ? UINT64_MAX : 0);
        r_.values[i] = (a_.values[i] < 0) ? 0 : r_.values[i];
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshlu_n_s64(a, n) vqshlu_n_s64(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshlu_n_s64
  #define vqshlu_n_s64(a, n) easysimd_vqshlu_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vqshluq_n_s8(easysimd_int8x16_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  easysimd_int8x16_private a_ = easysimd_int8x16_to_private(a);
  easysimd_uint8x16_private r_;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    __typeof__(r_.values) shifted = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values) << n;

    __typeof__(r_.values) overflow = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (shifted >> n) != HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values));

    r_.values = (shifted & ~overflow) | overflow;

    r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values >= 0));
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint8_t, a_.values[i] << n);
      r_.values[i] |= (((r_.values[i] >> n) != HEDLEY_STATIC_CAST(uint8_t, a_.values[i])) ? UINT8_MAX : 0);
      r_.values[i] = (a_.values[i] < 0) ? 0 : r_.values[i];
    }
  #endif

  return easysimd_uint8x16_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshluq_n_s8(a, n) vqshluq_n_s8(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshluq_n_s8
  #define vqshluq_n_s8(a, n) easysimd_vqshluq_n_s8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vqshluq_n_s16(easysimd_int16x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 15) {
  easysimd_int16x8_private a_ = easysimd_int16x8_to_private(a);
  easysimd_uint16x8_private r_;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    __typeof__(r_.values) shifted = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values) << n;

    __typeof__(r_.values) overflow = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (shifted >> n) != HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values));

    r_.values = (shifted & ~overflow) | overflow;

    r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values >= 0));
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint16_t, a_.values[i] << n);
      r_.values[i] |= (((r_.values[i] >> n) != HEDLEY_STATIC_CAST(uint16_t, a_.values[i])) ? UINT16_MAX : 0);
      r_.values[i] = (a_.values[i] < 0) ? 0 : r_.values[i];
    }
  #endif

  return easysimd_uint16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshluq_n_s16(a, n) vqshluq_n_s16(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshluq_n_s16
  #define vqshluq_n_s16(a, n) easysimd_vqshluq_n_s16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vqshluq_n_s32(easysimd_int32x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 31) {
  easysimd_int32x4_private a_ = easysimd_int32x4_to_private(a);
  easysimd_uint32x4_private r_;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    __typeof__(r_.values) shifted = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values) << n;

    __typeof__(r_.values) overflow = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (shifted >> n) != HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values));

    r_.values = (shifted & ~overflow) | overflow;

    r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values >= 0));
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint32_t, a_.values[i] << n);
      r_.values[i] |= (((r_.values[i] >> n) != HEDLEY_STATIC_CAST(uint32_t, a_.values[i])) ? UINT32_MAX : 0);
      r_.values[i] = (a_.values[i] < 0) ? 0 : r_.values[i];
    }
  #endif

  return easysimd_uint32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshluq_n_s32(a, n) vqshluq_n_s32(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshluq_n_s32
  #define vqshluq_n_s32(a, n) easysimd_vqshluq_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vqshluq_n_s64(easysimd_int64x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 63) {
  easysimd_int64x2_private a_ = easysimd_int64x2_to_private(a);
  easysimd_uint64x2_private r_;

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    __typeof__(r_.values) shifted = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values) << n;

    __typeof__(r_.values) overflow = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (shifted >> n) != HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), a_.values));

    r_.values = (shifted & ~overflow) | overflow;

    r_.values &= HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values >= 0));
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = HEDLEY_STATIC_CAST(uint64_t, a_.values[i] << n);
      r_.values[i] |= (((r_.values[i] >> n) != HEDLEY_STATIC_CAST(uint64_t, a_.values[i])) ? UINT64_MAX : 0);
      r_.values[i] = (a_.values[i] < 0) ? 0 : r_.values[i];
    }
  #endif

  return easysimd_uint64x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vqshluq_n_s64(a, n) vqshluq_n_s64(a, n)
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqshluq_n_s64
  #define vqshluq_n_s64(a, n) easysimd_vqshluq_n_s64((a), (n))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QSHLU_N_H) */
