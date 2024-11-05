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

#if !defined(EASYSIMD_ARM_NEON_RSHR_N_H)
#define EASYSIMD_ARM_NEON_RSHR_N_H

#include "combine.h"
#include "dup_n.h"
#include "get_low.h"
#include "reinterpret.h"
#include "shr_n.h"
#include "sub.h"
#include "tst.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_x_vrshrs_n_s32(int32_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  return (a >> ((n == 32) ? 31 : n)) + ((a & HEDLEY_STATIC_CAST(int32_t, UINT32_C(1) << (n - 1))) != 0);
}

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_x_vrshrs_n_u32(uint32_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  return ((n == 32) ? 0 : (a >> n)) + ((a & (UINT32_C(1) << (n - 1))) != 0);
}

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vrshrd_n_s64(int64_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  return (a >> ((n == 64) ? 63 : n)) + ((a & HEDLEY_STATIC_CAST(int64_t, UINT64_C(1) << (n - 1))) != 0);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vrshrd_n_s64(a, n) vrshrd_n_s64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrshrd_n_s64
  #define vrshrd_n_s64(a, n) easysimd_vrshrd_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vrshrd_n_u64(uint64_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  return ((n == 64) ? 0 : (a >> n)) + ((a & (UINT64_C(1) << (n - 1))) != 0);
}
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_vrshrd_n_u64(a, n) vrshrd_n_u64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrshrd_n_u64
  #define vrshrd_n_u64(a, n) easysimd_vrshrd_n_u64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vrshrq_n_s8 (const easysimd_int8x16_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 8) {
  easysimd_int8x16_private
    r_,
    a_ = easysimd_int8x16_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(int8_t, (a_.values[i] + (1 << (n - 1))) >> n);
  }

  return easysimd_int8x16_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshrq_n_s8(a, n) vrshrq_n_s8((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshrq_n_s8(a, n) easysimd_vsubq_s8(easysimd_vshrq_n_s8((a), (n)), easysimd_vreinterpretq_s8_u8( \
    easysimd_vtstq_u8(easysimd_vreinterpretq_u8_s8(a), \
                   easysimd_vdupq_n_u8(HEDLEY_STATIC_CAST(uint8_t, 1 << ((n) - 1))))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshrq_n_s8
  #define vrshrq_n_s8(a, n) easysimd_vrshrq_n_s8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vrshrq_n_s16 (const easysimd_int16x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 16) {
  easysimd_int16x8_private
    r_,
    a_ = easysimd_int16x8_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(int16_t, (a_.values[i] + (1 << (n - 1))) >> n);
  }

  return easysimd_int16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshrq_n_s16(a, n) vrshrq_n_s16((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshrq_n_s16(a, n) easysimd_vsubq_s16(easysimd_vshrq_n_s16((a), (n)), easysimd_vreinterpretq_s16_u16( \
    easysimd_vtstq_u16(easysimd_vreinterpretq_u16_s16(a),                              \
                    easysimd_vdupq_n_u16(HEDLEY_STATIC_CAST(uint16_t, 1 << ((n) - 1))))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshrq_n_s16
  #define vrshrq_n_s16(a, n) easysimd_vrshrq_n_s16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vrshrq_n_s32 (const easysimd_int32x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  easysimd_int32x4_private
    r_,
    a_ = easysimd_int32x4_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = (a_.values[i] >> ((n == 32) ? 31 : n)) + ((a_.values[i] & HEDLEY_STATIC_CAST(int32_t, UINT32_C(1) << (n - 1))) != 0);
  }

  return easysimd_int32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshrq_n_s32(a, n) vrshrq_n_s32((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshrq_n_s32(a, n) easysimd_vsubq_s32(easysimd_vshrq_n_s32((a), (n)), \
    easysimd_vreinterpretq_s32_u32(easysimd_vtstq_u32(easysimd_vreinterpretq_u32_s32(a), \
      easysimd_vdupq_n_u32(UINT32_C(1) << ((n) - 1)))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshrq_n_s32
  #define vrshrq_n_s32(a, n) easysimd_vrshrq_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vrshrq_n_s64 (const easysimd_int64x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  easysimd_int64x2_private
    r_,
    a_ = easysimd_int64x2_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = (a_.values[i] >> ((n == 64) ? 63 : n)) + ((a_.values[i] & HEDLEY_STATIC_CAST(int64_t, UINT64_C(1) << (n - 1))) != 0);
  }

  return easysimd_int64x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshrq_n_s64(a, n) vrshrq_n_s64((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshrq_n_s64(a, n) easysimd_vsubq_s64(easysimd_vshrq_n_s64((a), (n)), \
    easysimd_vreinterpretq_s64_u64(easysimd_vtstq_u64(easysimd_vreinterpretq_u64_s64(a), \
      easysimd_vdupq_n_u64(UINT64_C(1) << ((n) - 1)))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshrq_n_s64
  #define vrshrq_n_s64(a, n) easysimd_vrshrq_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vrshrq_n_u8 (const easysimd_uint8x16_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 8) {
  easysimd_uint8x16_private
    r_,
    a_ = easysimd_uint8x16_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(uint8_t, (a_.values[i] + (1 << (n - 1))) >> n);
  }

  return easysimd_uint8x16_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshrq_n_u8(a, n) vrshrq_n_u8((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshrq_n_u8(a, n) easysimd_vsubq_u8(easysimd_vshrq_n_u8((a), (n)), \
    easysimd_vtstq_u8((a), easysimd_vdupq_n_u8(HEDLEY_STATIC_CAST(uint8_t, 1 << ((n) - 1)))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshrq_n_u8
  #define vrshrq_n_u8(a, n) easysimd_vrshrq_n_u8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vrshrq_n_u16 (const easysimd_uint16x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 16) {
  easysimd_uint16x8_private
    r_,
    a_ = easysimd_uint16x8_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(uint16_t, (a_.values[i] + (1 << (n - 1))) >> n);
  }

  return easysimd_uint16x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshrq_n_u16(a, n) vrshrq_n_u16((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshrq_n_u16(a, n) easysimd_vsubq_u16(easysimd_vshrq_n_u16((a), (n)), \
    easysimd_vtstq_u16((a), easysimd_vdupq_n_u16(HEDLEY_STATIC_CAST(uint16_t, 1 << ((n) - 1)))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshrq_n_u16
  #define vrshrq_n_u16(a, n) easysimd_vrshrq_n_u16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vrshrq_n_u32 (const easysimd_uint32x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  easysimd_uint32x4_private
    r_,
    a_ = easysimd_uint32x4_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = ((n == 32) ? 0 : (a_.values[i] >> n)) + ((a_.values[i] & (UINT32_C(1) << (n - 1))) != 0);
  }

  return easysimd_uint32x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshrq_n_u32(a, n) vrshrq_n_u32((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshrq_n_u32(a, n) easysimd_vsubq_u32(easysimd_vshrq_n_u32((a), (n)), \
    easysimd_vtstq_u32((a), easysimd_vdupq_n_u32(UINT32_C(1) << ((n) - 1))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshrq_n_u32
  #define vrshrq_n_u32(a, n) easysimd_vrshrq_n_u32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vrshrq_n_u64 (const easysimd_uint64x2_t a, const int n)
  EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  easysimd_uint64x2_private
    r_,
    a_ = easysimd_uint64x2_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = ((n == 64) ? 0 : (a_.values[i] >> n)) + ((a_.values[i] & (UINT64_C(1) << (n - 1))) != 0);
  }

  return easysimd_uint64x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshrq_n_u64(a, n) vrshrq_n_u64((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshrq_n_u64(a, n) easysimd_vsubq_u64(easysimd_vshrq_n_u64((a), (n)), \
    easysimd_vtstq_u64((a), easysimd_vdupq_n_u64(UINT64_C(1) << ((n) - 1))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshrq_n_u64
  #define vrshrq_n_u64(a, n) easysimd_vrshrq_n_u64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vrshr_n_s8 (const easysimd_int8x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 8) {
  easysimd_int8x8_private
    r_,
    a_ = easysimd_int8x8_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(int8_t, (a_.values[i] + (1 << (n - 1))) >> n);
  }

  return easysimd_int8x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshr_n_s8(a, n) vrshr_n_s8((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshr_n_s8(a, n) easysimd_vsub_s8(easysimd_vshr_n_s8((a), (n)), easysimd_vreinterpret_s8_u8( \
    easysimd_vtst_u8(easysimd_vreinterpret_u8_s8(a),                              \
                  easysimd_vdup_n_u8(HEDLEY_STATIC_CAST(uint8_t, 1 << ((n) - 1))))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshr_n_s8
  #define vrshr_n_s8(a, n) easysimd_vrshr_n_s8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vrshr_n_s16 (const easysimd_int16x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 16) {
  easysimd_int16x4_private
    r_,
    a_ = easysimd_int16x4_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(int16_t, (a_.values[i] + (1 << (n - 1))) >> n);
  }

  return easysimd_int16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshr_n_s16(a, n) vrshr_n_s16((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshr_n_s16(a, n) easysimd_vsub_s16(easysimd_vshr_n_s16((a), (n)), easysimd_vreinterpret_s16_u16( \
    easysimd_vtst_u16(easysimd_vreinterpret_u16_s16(a), \
                   easysimd_vdup_n_u16(HEDLEY_STATIC_CAST(uint16_t, 1 << ((n) - 1))))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshr_n_s16
  #define vrshr_n_s16(a, n) easysimd_vrshr_n_s16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vrshr_n_s32 (const easysimd_int32x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  easysimd_int32x2_private
    r_,
    a_ = easysimd_int32x2_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = (a_.values[i] >> ((n == 32) ? 31 : n)) + ((a_.values[i] & HEDLEY_STATIC_CAST(int32_t, UINT32_C(1) << (n - 1))) != 0);
  }

  return easysimd_int32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshr_n_s32(a, n) vrshr_n_s32((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshr_n_s32(a, n) easysimd_vsub_s32(easysimd_vshr_n_s32((a), (n)), \
    easysimd_vreinterpret_s32_u32(easysimd_vtst_u32(easysimd_vreinterpret_u32_s32(a), \
      easysimd_vdup_n_u32(UINT32_C(1) << ((n) - 1)))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshr_n_s32
  #define vrshr_n_s32(a, n) easysimd_vrshr_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vrshr_n_s64 (const easysimd_int64x1_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  easysimd_int64x1_private
    r_,
    a_ = easysimd_int64x1_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = (a_.values[i] >> ((n == 64) ? 63 : n)) + ((a_.values[i] & HEDLEY_STATIC_CAST(int64_t, UINT64_C(1) << (n - 1))) != 0);
  }

  return easysimd_int64x1_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshr_n_s64(a, n) vrshr_n_s64((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshr_n_s64(a, n) easysimd_vsub_s64(easysimd_vshr_n_s64((a), (n)), \
    easysimd_vreinterpret_s64_u64(easysimd_vtst_u64(easysimd_vreinterpret_u64_s64(a), \
      easysimd_vdup_n_u64(UINT64_C(1) << ((n) - 1)))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshr_n_s64
  #define vrshr_n_s64(a, n) easysimd_vrshr_n_s64((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vrshr_n_u8 (const easysimd_uint8x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 8) {
  easysimd_uint8x8_private
    r_,
    a_ = easysimd_uint8x8_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(uint8_t, (a_.values[i] + (1 << (n - 1))) >> n);
  }

  return easysimd_uint8x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshr_n_u8(a, n) vrshr_n_u8((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshr_n_u8(a, n) easysimd_vsub_u8(easysimd_vshr_n_u8((a), (n)), \
    easysimd_vtst_u8((a), easysimd_vdup_n_u8(HEDLEY_STATIC_CAST(uint8_t, 1 << ((n) - 1)))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshr_n_u8
  #define vrshr_n_u8(a, n) easysimd_vrshr_n_u8((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vrshr_n_u16 (const easysimd_uint16x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 16) {
  easysimd_uint16x4_private
    r_,
    a_ = easysimd_uint16x4_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(uint16_t, (a_.values[i] + (1 << (n - 1))) >> n);
  }

  return easysimd_uint16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshr_n_u16(a, n) vrshr_n_u16((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshr_n_u16(a, n) easysimd_vsub_u16(easysimd_vshr_n_u16((a), (n)), \
    easysimd_vtst_u16((a), easysimd_vdup_n_u16(HEDLEY_STATIC_CAST(uint16_t, 1 << ((n) - 1)))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshr_n_u16
  #define vrshr_n_u16(a, n) easysimd_vrshr_n_u16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vrshr_n_u32 (const easysimd_uint32x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  easysimd_uint32x2_private
    r_,
    a_ = easysimd_uint32x2_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = ((n == 32) ? 0 : (a_.values[i] >> n))  + ((a_.values[i] & (UINT32_C(1) << (n - 1))) != 0);
  }

  return easysimd_uint32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshr_n_u32(a, n) vrshr_n_u32((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshr_n_u32(a, n) easysimd_vsub_u32(easysimd_vshr_n_u32((a), (n)), \
    easysimd_vtst_u32((a), easysimd_vdup_n_u32(UINT32_C(1) << ((n) - 1))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshr_n_u32
  #define vrshr_n_u32(a, n) easysimd_vrshr_n_u32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vrshr_n_u64 (const easysimd_uint64x1_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 64) {
  easysimd_uint64x1_private
    r_,
    a_ = easysimd_uint64x1_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = ((n == 64) ? 0 : (a_.values[i] >> n))  + ((a_.values[i] & (UINT64_C(1) << (n - 1))) != 0);
  }

  return easysimd_uint64x1_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vrshr_n_u64(a, n) vrshr_n_u64((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vrshr_n_u64(a, n) easysimd_vsub_u64(easysimd_vshr_n_u64((a), (n)), \
    easysimd_vtst_u64((a), easysimd_vdup_n_u64(UINT64_C(1) << ((n) - 1))))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrshr_n_u64
  #define vrshr_n_u64(a, n) easysimd_vrshr_n_u64((a), (n))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_RSHR_N_H) */
