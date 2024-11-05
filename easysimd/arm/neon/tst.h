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
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_ARM_NEON_TST_H)
#define EASYSIMD_ARM_NEON_TST_H

#include "and.h"
#include "ceqz.h"
#include "cgt.h"
#include "combine.h"
#include "dup_n.h"
#include "get_low.h"
#include "mvn.h"
#include "reinterpret.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vtstd_s64(int64_t a, int64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint64_t, vtstd_s64(a, b));
  #else
    return ((a & b) != 0) ? UINT64_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vtstd_s64
  #define vtstd_s64(a, b) easysimd_vtstd_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vtstd_u64(uint64_t a, uint64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint64_t, vtstd_u64(a, b));
  #else
    return ((a & b) != 0) ? UINT64_MAX : 0;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vtstd_u64
  #define vtstd_u64(a, b) easysimd_vtstd_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vtstq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtstq_s8(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvnq_u8(easysimd_vceqzq_s8(easysimd_vandq_s8(a, b)));
  #else
    easysimd_int8x16_private
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);
    easysimd_uint8x16_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT8_MAX : 0;
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtstq_s8
  #define vtstq_s8(a, b) easysimd_vtstq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vtstq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtstq_s16(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvnq_u16(easysimd_vceqzq_s16(easysimd_vandq_s16(a, b)));
  #else
    easysimd_int16x8_private
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);
    easysimd_uint16x8_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT16_MAX : 0;
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtstq_s16
  #define vtstq_s16(a, b) easysimd_vtstq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vtstq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtstq_s32(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvnq_u32(easysimd_vceqzq_s32(easysimd_vandq_s32(a, b)));
  #else
    easysimd_int32x4_private
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);
    easysimd_uint32x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtstq_s32
  #define vtstq_s32(a, b) easysimd_vtstq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vtstq_s64(easysimd_int64x2_t a, easysimd_int64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vtstq_s64(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vceqzq_u64(easysimd_vceqzq_s64(easysimd_vandq_s64(a, b)));
  #else
    easysimd_int64x2_private
      a_ = easysimd_int64x2_to_private(a),
      b_ = easysimd_int64x2_to_private(b);
    easysimd_uint64x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vtstd_s64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vtstq_s64
  #define vtstq_s64(a, b) easysimd_vtstq_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vtstq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtstq_u8(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvnq_u8(easysimd_vceqzq_u8(easysimd_vandq_u8(a, b)));
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT8_MAX : 0;
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtstq_u8
  #define vtstq_u8(a, b) easysimd_vtstq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vtstq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtstq_u16(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvnq_u16(easysimd_vceqzq_u16(easysimd_vandq_u16(a, b)));
  #else
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a),
      b_ = easysimd_uint16x8_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT16_MAX : 0;
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtstq_u16
  #define vtstq_u16(a, b) easysimd_vtstq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vtstq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtstq_u32(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvnq_u32(easysimd_vceqzq_u32(easysimd_vandq_u32(a, b)));
  #else
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a),
      b_ = easysimd_uint32x4_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtstq_u32
  #define vtstq_u32(a, b) easysimd_vtstq_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vtstq_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vtstq_u64(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vceqzq_u64(easysimd_vceqzq_u64(easysimd_vandq_u64(a, b)));
  #else
    easysimd_uint64x2_private
      r_,
      a_ = easysimd_uint64x2_to_private(a),
      b_ = easysimd_uint64x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vtstd_u64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vtstq_u64
  #define vtstq_u64(a, b) easysimd_vtstq_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vtst_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtst_s8(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvn_u8(easysimd_vceqz_s8(easysimd_vand_s8(a, b)));
  #else
    easysimd_int8x8_private
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);
    easysimd_uint8x8_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT8_MAX : 0;
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtst_s8
  #define vtst_s8(a, b) easysimd_vtst_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vtst_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtst_s16(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvn_u16(easysimd_vceqz_s16(easysimd_vand_s16(a, b)));
  #else
    easysimd_int16x4_private
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);
    easysimd_uint16x4_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT16_MAX : 0;
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtst_s16
  #define vtst_s16(a, b) easysimd_vtst_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vtst_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtst_s32(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvn_u32(easysimd_vceqz_s32(easysimd_vand_s32(a, b)));
  #else
    easysimd_int32x2_private
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b);
    easysimd_uint32x2_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtst_s32
  #define vtst_s32(a, b) easysimd_vtst_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vtst_s64(easysimd_int64x1_t a, easysimd_int64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vtst_s64(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vceqz_u64(easysimd_vceqz_s64(easysimd_vand_s64(a, b)));
  #else
    easysimd_int64x1_private
      a_ = easysimd_int64x1_to_private(a),
      b_ = easysimd_int64x1_to_private(b);
    easysimd_uint64x1_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vtstd_s64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vtst_s64
  #define vtst_s64(a, b) easysimd_vtst_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vtst_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtst_u8(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvn_u8(easysimd_vceqz_u8(easysimd_vand_u8(a, b)));
  #else
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT8_MAX : 0;
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtst_u8
  #define vtst_u8(a, b) easysimd_vtst_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vtst_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtst_u16(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvn_u16(easysimd_vceqz_u16(easysimd_vand_u16(a, b)));
  #else
    easysimd_uint16x4_private
      r_,
      a_ = easysimd_uint16x4_to_private(a),
      b_ = easysimd_uint16x4_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT16_MAX : 0;
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtst_u16
  #define vtst_u16(a, b) easysimd_vtst_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vtst_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vtst_u32(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmvn_u32(easysimd_vceqz_u32(easysimd_vand_u32(a, b)));
  #else
    easysimd_uint32x2_private
      r_,
      a_ = easysimd_uint32x2_to_private(a),
      b_ = easysimd_uint32x2_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_100762)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = ((a_.values[i] & b_.values[i]) != 0) ? UINT32_MAX : 0;
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vtst_u32
  #define vtst_u32(a, b) easysimd_vtst_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vtst_u64(easysimd_uint64x1_t a, easysimd_uint64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vtst_u64(a, b);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vceqz_u64(easysimd_vceqz_u64(easysimd_vand_u64(a, b)));
  #else
    easysimd_uint64x1_private
      r_,
      a_ = easysimd_uint64x1_to_private(a),
      b_ = easysimd_uint64x1_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.values = HEDLEY_REINTERPRET_CAST(__typeof__(r_.values), (a_.values & b_.values) != 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = easysimd_vtstd_u64(a_.values[i], b_.values[i]);
      }
    #endif

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vtst_u64
  #define vtst_u64(a, b) easysimd_vtst_u64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_TST_H) */
