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

#if !defined(EASYSIMD_ARM_NEON_ST1_H)
#define EASYSIMD_ARM_NEON_ST1_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_f16(easysimd_float16_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float16x4_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    vst1_f16(ptr, val);
  #else
    easysimd_float16x4_private val_ = easysimd_float16x4_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_f16
  #define vst1_f16(a, b) easysimd_vst1_f16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_f32(easysimd_float32_t ptr[HEDLEY_ARRAY_PARAM(2)], easysimd_float32x2_t val) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1_f32(ptr, val);
  #else
    easysimd_float32x2_private val_ = easysimd_float32x2_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_f32
  #define vst1_f32(a, b) easysimd_vst1_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_f64(easysimd_float64_t ptr[HEDLEY_ARRAY_PARAM(1)], easysimd_float64x1_t val) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1_f64(ptr, val);
  #else
    easysimd_float64x1_private val_ = easysimd_float64x1_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst1_f64
  #define vst1_f64(a, b) easysimd_vst1_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_s8(int8_t ptr[HEDLEY_ARRAY_PARAM(8)], easysimd_int8x8_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1_s8(ptr, val);
  #else
    easysimd_int8x8_private val_ = easysimd_int8x8_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_s8
  #define vst1_s8(a, b) easysimd_vst1_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_s16(int16_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int16x4_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1_s16(ptr, val);
  #else
    easysimd_int16x4_private val_ = easysimd_int16x4_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_s16
  #define vst1_s16(a, b) easysimd_vst1_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_s32(int32_t ptr[HEDLEY_ARRAY_PARAM(2)], easysimd_int32x2_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1_s32(ptr, val);
  #else
    easysimd_int32x2_private val_ = easysimd_int32x2_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_s32
  #define vst1_s32(a, b) easysimd_vst1_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_s64(int64_t ptr[HEDLEY_ARRAY_PARAM(1)], easysimd_int64x1_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1_s64(ptr, val);
  #else
    easysimd_int64x1_private val_ = easysimd_int64x1_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_s64
  #define vst1_s64(a, b) easysimd_vst1_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_u8(uint8_t ptr[HEDLEY_ARRAY_PARAM(8)], easysimd_uint8x8_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1_u8(ptr, val);
  #else
    easysimd_uint8x8_private val_ = easysimd_uint8x8_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_u8
  #define vst1_u8(a, b) easysimd_vst1_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_u16(uint16_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint16x4_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1_u16(ptr, val);
  #else
    easysimd_uint16x4_private val_ = easysimd_uint16x4_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_u16
  #define vst1_u16(a, b) easysimd_vst1_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_u32(uint32_t ptr[HEDLEY_ARRAY_PARAM(2)], easysimd_uint32x2_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1_u32(ptr, val);
  #else
    easysimd_uint32x2_private val_ = easysimd_uint32x2_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_u32
  #define vst1_u32(a, b) easysimd_vst1_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1_u64(uint64_t ptr[HEDLEY_ARRAY_PARAM(1)], easysimd_uint64x1_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1_u64(ptr, val);
  #else
    easysimd_uint64x1_private val_ = easysimd_uint64x1_to_private(val);
    easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1_u64
  #define vst1_u64(a, b) easysimd_vst1_u64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_f16(easysimd_float16_t ptr[HEDLEY_ARRAY_PARAM(8)], easysimd_float16x8_t val) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    vst1q_f16(ptr, val);
  #else
    easysimd_float16x8_private val_ = easysimd_float16x8_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst1q_f16
  #define vst1q_f16(a, b) easysimd_vst1q_f16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_f32(easysimd_float32_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_float32x4_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_f32(ptr, val);
  #else
    easysimd_float32x4_private val_ = easysimd_float32x4_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1q_f32
  #define vst1q_f32(a, b) easysimd_vst1q_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_f64(easysimd_float64_t ptr[HEDLEY_ARRAY_PARAM(2)], easysimd_float64x2_t val) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_f64(ptr, val);
  #else
    easysimd_float64x2_private val_ = easysimd_float64x2_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vst1q_f64
  #define vst1q_f64(a, b) easysimd_vst1q_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_s8(int8_t ptr[HEDLEY_ARRAY_PARAM(16)], easysimd_int8x16_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_s8(ptr, val);
  #else
    easysimd_int8x16_private val_ = easysimd_int8x16_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1q_s8
  #define vst1q_s8(a, b) easysimd_vst1q_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_s16(int16_t ptr[HEDLEY_ARRAY_PARAM(8)], easysimd_int16x8_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_s16(ptr, val);
  #else
    easysimd_int16x8_private val_ = easysimd_int16x8_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1q_s16
  #define vst1q_s16(a, b) easysimd_vst1q_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_s32(int32_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_int32x4_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_s32(ptr, val);
  #else
    easysimd_int32x4_private val_ = easysimd_int32x4_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1q_s32
  #define vst1q_s32(a, b) easysimd_vst1q_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_s64(int64_t ptr[HEDLEY_ARRAY_PARAM(2)], easysimd_int64x2_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_s64(ptr, val);
  #else
    easysimd_int64x2_private val_ = easysimd_int64x2_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1q_s64
  #define vst1q_s64(a, b) easysimd_vst1q_s64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_u8(uint8_t ptr[HEDLEY_ARRAY_PARAM(16)], easysimd_uint8x16_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_u8(ptr, val);
  #else
    easysimd_uint8x16_private val_ = easysimd_uint8x16_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1q_u8
  #define vst1q_u8(a, b) easysimd_vst1q_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_u16(uint16_t ptr[HEDLEY_ARRAY_PARAM(8)], easysimd_uint16x8_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_u16(ptr, val);
  #else
    easysimd_uint16x8_private val_ = easysimd_uint16x8_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1q_u16
  #define vst1q_u16(a, b) easysimd_vst1q_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_u32(uint32_t ptr[HEDLEY_ARRAY_PARAM(4)], easysimd_uint32x4_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_u32(ptr, val);
  #else
    easysimd_uint32x4_private val_ = easysimd_uint32x4_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1q_u32
  #define vst1q_u32(a, b) easysimd_vst1q_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_vst1q_u64(uint64_t ptr[HEDLEY_ARRAY_PARAM(2)], easysimd_uint64x2_t val) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_u64(ptr, val);
  #else
    easysimd_uint64x2_private val_ = easysimd_uint64x2_to_private(val);

      easysimd_memcpy(ptr, &val_, sizeof(val_));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vst1q_u64
  #define vst1q_u64(a, b) easysimd_vst1q_u64((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_ST1_H) */
