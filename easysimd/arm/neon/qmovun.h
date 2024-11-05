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

#if !defined(EASYSIMD_ARM_NEON_QMOVUN_H)
#define EASYSIMD_ARM_NEON_QMOVUN_H

#include "types.h"
#include "dup_n.h"
#include "min.h"
#include "max.h"
#include "movn.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
uint8_t
easysimd_vqmovunh_s16(int16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint8_t, vqmovunh_s16(a));
  #else
    return (a > UINT8_MAX) ? UINT8_MAX : ((a < 0) ? 0 : HEDLEY_STATIC_CAST(uint8_t, a));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqmovunh_s16
  #define vqmovunh_s16(a) easysimd_vqmovunh_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vqmovuns_s32(int32_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint16_t, vqmovuns_s32(a));
  #else
    return (a > UINT16_MAX) ? UINT16_MAX : ((a < 0) ? 0 : HEDLEY_STATIC_CAST(uint16_t, a));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqmovuns_s32
  #define vqmovuns_s32(a) easysimd_vqmovuns_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vqmovund_s64(int64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint32_t, vqmovund_s64(a));
  #else
    return (a > UINT32_MAX) ? UINT32_MAX : ((a < 0) ? 0 : HEDLEY_STATIC_CAST(uint32_t, a));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqmovund_s64
  #define vqmovund_s64(a) easysimd_vqmovund_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vqmovun_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqmovun_s16(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmovn_u16(easysimd_vreinterpretq_u16_s16(easysimd_vmaxq_s16(easysimd_vdupq_n_s16(0), easysimd_vminq_s16(easysimd_vdupq_n_s16(UINT8_MAX), a))));
  #else
    easysimd_uint8x8_private r_;
    easysimd_int16x8_private a_ = easysimd_int16x8_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = easysimd_vqmovunh_s16(a_.values[i]);
    }

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqmovun_s16
  #define vqmovun_s16(a) easysimd_vqmovun_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vqmovun_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqmovun_s32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmovn_u32(easysimd_vreinterpretq_u32_s32(easysimd_vmaxq_s32(easysimd_vdupq_n_s32(0), easysimd_vminq_s32(easysimd_vdupq_n_s32(UINT16_MAX), a))));
  #else
    easysimd_uint16x4_private r_;
    easysimd_int32x4_private a_ = easysimd_int32x4_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = easysimd_vqmovuns_s32(a_.values[i]);
    }

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqmovun_s32
  #define vqmovun_s32(a) easysimd_vqmovun_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vqmovun_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqmovun_s64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
    return easysimd_vmovn_u64(easysimd_vreinterpretq_u64_s64(easysimd_x_vmaxq_s64(easysimd_vdupq_n_s64(0), easysimd_x_vminq_s64(easysimd_vdupq_n_s64(UINT32_MAX), a))));
  #else
    easysimd_uint32x2_private r_;
    easysimd_int64x2_private a_ = easysimd_int64x2_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = easysimd_vqmovund_s64(a_.values[i]);
    }

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqmovun_s64
  #define vqmovun_s64(a) easysimd_vqmovun_s64((a))
#endif


EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QMOVUN_H) */
