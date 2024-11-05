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
 *   2021      Zhi An Ng <zhin@google.com> (Copyright owned by Google, LLC)
 */

#if !defined(EASYSIMD_ARM_NEON_SHRN_N_H)
#define EASYSIMD_ARM_NEON_SHRN_N_H

#include "types.h"
#include "reinterpret.h"
#include "movn.h"
#include "shr_n.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vshrn_n_s16 (const easysimd_int16x8_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 8) {
  easysimd_int8x8_private r_;
  easysimd_int16x8_private a_ = easysimd_int16x8_to_private(a);
  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(int8_t, (a_.values[i] >> n) & UINT8_MAX);
  }
  return easysimd_int8x8_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrn_n_s16(a, n) vshrn_n_s16((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vshrn_n_s16(a, n) easysimd_vmovn_s16(easysimd_vshrq_n_s16((a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrn_n_s16
  #define vshrn_n_s16(a, n) easysimd_vshrn_n_s16((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vshrn_n_s32 (const easysimd_int32x4_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 16) {
  easysimd_int16x4_private r_;
  easysimd_int32x4_private a_ = easysimd_int32x4_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(int16_t, (a_.values[i] >> n) & UINT16_MAX);
  }

  return easysimd_int16x4_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrn_n_s32(a, n) vshrn_n_s32((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vshrn_n_s32(a, n) easysimd_vmovn_s32(easysimd_vshrq_n_s32((a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrn_n_s32
  #define vshrn_n_s32(a, n) easysimd_vshrn_n_s32((a), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vshrn_n_s64 (const easysimd_int64x2_t a, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 1, 32) {
  easysimd_int32x2_private r_;
  easysimd_int64x2_private a_ = easysimd_int64x2_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
    r_.values[i] = HEDLEY_STATIC_CAST(int32_t, (a_.values[i] >> n) & UINT32_MAX);
  }

  return easysimd_int32x2_from_private(r_);
}
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_vshrn_n_s64(a, n) vshrn_n_s64((a), (n))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_vshrn_n_s64(a, n) easysimd_vmovn_s64(easysimd_vshrq_n_s64((a), (n)))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrn_n_s64
  #define vshrn_n_s64(a, n) easysimd_vshrn_n_s64((a), (n))
#endif

#define easysimd_vshrn_n_u16(a, n) \
  easysimd_vreinterpret_u8_s8(     \
      easysimd_vshrn_n_s16(easysimd_vreinterpretq_s16_u16(a), (n)))

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #undef easysimd_vshrn_n_u16
  #define easysimd_vshrn_n_u16(a, n) vshrn_n_u16((a), (n))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrn_n_u16
  #define vshrn_n_u16(a, n) easysimd_vshrn_n_u16((a), (n))
#endif

#define easysimd_vshrn_n_u32(a, n) \
  easysimd_vreinterpret_u16_s16( \
      easysimd_vshrn_n_s32(easysimd_vreinterpretq_s32_u32(a), (n)))

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #undef easysimd_vshrn_n_u32
  #define easysimd_vshrn_n_u32(a, n) vshrn_n_u32((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrn_n_u32
  #define vshrn_n_u32(a, n) easysimd_vshrn_n_u32((a), (n))
#endif

#define easysimd_vshrn_n_u64(a, n) \
  easysimd_vreinterpret_u32_s32( \
      easysimd_vshrn_n_s64(easysimd_vreinterpretq_s64_u64(a), (n)))

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #undef easysimd_vshrn_n_u64
  #define easysimd_vshrn_n_u64(a, n) vshrn_n_u64((a), (n))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vshrn_n_u64
  #define vshrn_n_u64(a, n) easysimd_vshrn_n_u64((a), (n))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_SHRN_N_H) */
