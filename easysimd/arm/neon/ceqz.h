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

#if !defined(EASYSIMD_ARM_NEON_CEQZ_H)
#define EASYSIMD_ARM_NEON_CEQZ_H

#include "ceq.h"
#include "dup_n.h"
#include "types.h"
#include "reinterpret.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vceqz_f16(easysimd_float16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vceqz_f16(a);
  #else
    return easysimd_vceq_f16(a, easysimd_vdup_n_f16(EASYSIMD_FLOAT16_VALUE(0.0)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES)
  #undef vceqz_f16
  #define vceqz_f16(a) easysimd_vceqz_f16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vceqz_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_f32(a);
  #else
    return easysimd_vceq_f32(a, easysimd_vdup_n_f32(0.0f));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqz_f32
  #define vceqz_f32(a) easysimd_vceqz_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vceqz_f64(easysimd_float64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_f64(a);
  #else
    return easysimd_vceq_f64(a, easysimd_vdup_n_f64(0.0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vceqz_f64
  #define vceqz_f64(a) easysimd_vceqz_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vceqz_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_s8(a);
  #else
    return easysimd_vceq_s8(a, easysimd_vdup_n_s8(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqz_s8
  #define vceqz_s8(a) easysimd_vceqz_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vceqz_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_s16(a);
  #else
    return easysimd_vceq_s16(a, easysimd_vdup_n_s16(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqz_s16
  #define vceqz_s16(a) easysimd_vceqz_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vceqz_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_s32(a);
  #else
    return easysimd_vceq_s32(a, easysimd_vdup_n_s32(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqz_s32
  #define vceqz_s32(a) easysimd_vceqz_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vceqz_s64(easysimd_int64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_s64(a);
  #else
    return easysimd_vceq_s64(a, easysimd_vdup_n_s64(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqz_s64
  #define vceqz_s64(a) easysimd_vceqz_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vceqz_u8(easysimd_uint8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_u8(a);
  #else
    return easysimd_vceq_u8(a, easysimd_vdup_n_u8(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqz_u8
  #define vceqz_u8(a) easysimd_vceqz_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vceqz_u16(easysimd_uint16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_u16(a);
  #else
    return easysimd_vceq_u16(a, easysimd_vdup_n_u16(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqz_u16
  #define vceqz_u16(a) easysimd_vceqz_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vceqz_u32(easysimd_uint32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_u32(a);
  #else
    return easysimd_vceq_u32(a, easysimd_vdup_n_u32(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqz_u32
  #define vceqz_u32(a) easysimd_vceqz_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vceqz_u64(easysimd_uint64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqz_u64(a);
  #else
    return easysimd_vceq_u64(a, easysimd_vdup_n_u64(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqz_u64
  #define vceqz_u64(a) easysimd_vceqz_u64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vceqzq_f16(easysimd_float16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vceqzq_f16(a);
  #else
    return easysimd_vceqq_f16(a, easysimd_vdupq_n_f16(EASYSIMD_FLOAT16_VALUE(0.0)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_f16
  #define vceqzq_f16(a) easysimd_vceqzq_f16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vceqzq_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_f32(a);
  #else
    return easysimd_vceqq_f32(a, easysimd_vdupq_n_f32(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_f32
  #define vceqzq_f32(a) easysimd_vceqzq_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vceqzq_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_f64(a);
  #else
    return easysimd_vceqq_f64(a, easysimd_vdupq_n_f64(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_f64
  #define vceqzq_f64(a) easysimd_vceqzq_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vceqzq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_s8(a);
  #else
    return easysimd_vceqq_s8(a, easysimd_vdupq_n_s8(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_s8
  #define vceqzq_s8(a) easysimd_vceqzq_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vceqzq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_s16(a);
  #else
    return easysimd_vceqq_s16(a, easysimd_vdupq_n_s16(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_s16
  #define vceqzq_s16(a) easysimd_vceqzq_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vceqzq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_s32(a);
  #else
    return easysimd_vceqq_s32(a, easysimd_vdupq_n_s32(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_s32
  #define vceqzq_s32(a) easysimd_vceqzq_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vceqzq_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_s64(a);
  #else
    return easysimd_vceqq_s64(a, easysimd_vdupq_n_s64(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_s64
  #define vceqzq_s64(a) easysimd_vceqzq_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vceqzq_u8(easysimd_uint8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_u8(a);
  #else
    return easysimd_vceqq_u8(a, easysimd_vdupq_n_u8(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_u8
  #define vceqzq_u8(a) easysimd_vceqzq_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vceqzq_u16(easysimd_uint16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_u16(a);
  #else
    return easysimd_vceqq_u16(a, easysimd_vdupq_n_u16(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_u16
  #define vceqzq_u16(a) easysimd_vceqzq_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vceqzq_u32(easysimd_uint32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_u32(a);
  #else
    return easysimd_vceqq_u32(a, easysimd_vdupq_n_u32(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_u32
  #define vceqzq_u32(a) easysimd_vceqzq_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vceqzq_u64(easysimd_uint64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzq_u64(a);
  #else
    return easysimd_vceqq_u64(a, easysimd_vdupq_n_u64(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzq_u64
  #define vceqzq_u64(a) easysimd_vceqzq_u64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vceqzd_s64(int64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return HEDLEY_STATIC_CAST(uint64_t, vceqzd_s64(a));
  #else
    return easysimd_vceqd_s64(a, INT64_C(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzd_s64
  #define vceqzd_s64(a) easysimd_vceqzd_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vceqzd_u64(uint64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzd_u64(a);
  #else
    return easysimd_vceqd_u64(a, UINT64_C(0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzd_u64
  #define vceqzd_u64(a) easysimd_vceqzd_u64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint16_t
easysimd_vceqzh_f16(easysimd_float16 a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vceqzh_f16(a);
  #else
    return easysimd_vceqh_f16(a, EASYSIMD_FLOAT16_VALUE(0.0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vceqzh_f16
  #define vceqzh_f16(a) easysimd_vceqzh_f16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_vceqzs_f32(easysimd_float32_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzs_f32(a);
  #else
    return easysimd_vceqs_f32(a, EASYSIMD_FLOAT32_C(0.0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzs_f32
  #define vceqzs_f32(a) easysimd_vceqzs_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_vceqzd_f64(easysimd_float64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vceqzd_f64(a);
  #else
    return easysimd_vceqd_f64(a, EASYSIMD_FLOAT64_C(0.0));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vceqzd_f64
  #define vceqzd_f64(a) easysimd_vceqzd_f64((a))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_CEQZ_H) */
