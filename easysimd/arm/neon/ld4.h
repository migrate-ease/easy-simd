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
 *   2020      Sean Maher <seanptmaher@gmail.com>
 */

#if !defined(EASYSIMD_ARM_NEON_LD4_H)
#define EASYSIMD_ARM_NEON_LD4_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
#if HEDLEY_GCC_VERSION_CHECK(7,0,0)
  EASYSIMD_DIAGNOSTIC_DISABLE_MAYBE_UNINITIAZILED_
#endif
EASYSIMD_BEGIN_DECLS_

#if !defined(EASYSIMD_BUG_INTEL_857088)

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2x4_t
easysimd_vld4_f32(easysimd_float32 const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4_f32(ptr);
  #else
    easysimd_float32x2_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_float32x2_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_float32x2x4_t s_ = { { easysimd_float32x2_from_private(a_[0]), easysimd_float32x2_from_private(a_[1]),
                                 easysimd_float32x2_from_private(a_[2]), easysimd_float32x2_from_private(a_[3]) } };
    return (s_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_f32
  #define vld4_f32(a) easysimd_vld4_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1x4_t
easysimd_vld4_f64(easysimd_float64 const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld4_f64(ptr);
  #else
    easysimd_float64x1_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_float64x1_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_float64x1x4_t s_ = { { easysimd_float64x1_from_private(a_[0]), easysimd_float64x1_from_private(a_[1]),
                                 easysimd_float64x1_from_private(a_[2]), easysimd_float64x1_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4_f64
  #define vld4_f64(a) easysimd_vld4_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8x4_t
easysimd_vld4_s8(int8_t const ptr[HEDLEY_ARRAY_PARAM(32)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4_s8(ptr);
  #else
    easysimd_int8x8_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_int8x8_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_int8x8x4_t s_ = { { easysimd_int8x8_from_private(a_[0]), easysimd_int8x8_from_private(a_[1]),
                              easysimd_int8x8_from_private(a_[2]), easysimd_int8x8_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_s8
  #define vld4_s8(a) easysimd_vld4_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4x4_t
easysimd_vld4_s16(int16_t const ptr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4_s16(ptr);
  #else
    easysimd_int16x4_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_int16x4_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_int16x4x4_t s_ = { { easysimd_int16x4_from_private(a_[0]), easysimd_int16x4_from_private(a_[1]),
                               easysimd_int16x4_from_private(a_[2]), easysimd_int16x4_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_s16
  #define vld4_s16(a) easysimd_vld4_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2x4_t
easysimd_vld4_s32(int32_t const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4_s32(ptr);
  #else
    easysimd_int32x2_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_int32x2_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_int32x2x4_t s_ = { { easysimd_int32x2_from_private(a_[0]), easysimd_int32x2_from_private(a_[1]),
                               easysimd_int32x2_from_private(a_[2]), easysimd_int32x2_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_s32
  #define vld4_s32(a) easysimd_vld4_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1x4_t
easysimd_vld4_s64(int64_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4_s64(ptr);
  #else
    easysimd_int64x1_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_int64x1_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_int64x1x4_t s_ = { { easysimd_int64x1_from_private(a_[0]), easysimd_int64x1_from_private(a_[1]),
                               easysimd_int64x1_from_private(a_[2]), easysimd_int64x1_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4_s64
  #define vld4_s64(a) easysimd_vld4_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8x4_t
easysimd_vld4_u8(uint8_t const ptr[HEDLEY_ARRAY_PARAM(32)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4_u8(ptr);
  #else
    easysimd_uint8x8_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_uint8x8_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_uint8x8x4_t s_ = { { easysimd_uint8x8_from_private(a_[0]), easysimd_uint8x8_from_private(a_[1]),
                               easysimd_uint8x8_from_private(a_[2]), easysimd_uint8x8_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_u8
  #define vld4_u8(a) easysimd_vld4_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4x4_t
easysimd_vld4_u16(uint16_t const ptr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4_u16(ptr);
  #else
    easysimd_uint16x4_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_uint16x4_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_uint16x4x4_t s_ = { { easysimd_uint16x4_from_private(a_[0]), easysimd_uint16x4_from_private(a_[1]),
                                easysimd_uint16x4_from_private(a_[2]), easysimd_uint16x4_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_u16
  #define vld4_u16(a) easysimd_vld4_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2x4_t
easysimd_vld4_u32(uint32_t const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4_u32(ptr);
  #else
    easysimd_uint32x2_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_uint32x2_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_uint32x2x4_t s_ = { { easysimd_uint32x2_from_private(a_[0]), easysimd_uint32x2_from_private(a_[1]),
                                easysimd_uint32x2_from_private(a_[2]), easysimd_uint32x2_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4_u32
  #define vld4_u32(a) easysimd_vld4_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1x4_t
easysimd_vld4_u64(uint64_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4_u64(ptr);
  #else
    easysimd_uint64x1_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_uint64x1_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_uint64x1x4_t s_ = { { easysimd_uint64x1_from_private(a_[0]), easysimd_uint64x1_from_private(a_[1]),
                                easysimd_uint64x1_from_private(a_[2]), easysimd_uint64x1_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4_u64
  #define vld4_u64(a) easysimd_vld4_u64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4x4_t
easysimd_vld4q_f32(easysimd_float32 const ptr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4q_f32(ptr);
  #else
    easysimd_float32x4_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_float32x4_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_float32x4x4_t s_ = { { easysimd_float32x4_from_private(a_[0]), easysimd_float32x4_from_private(a_[1]),
                                 easysimd_float32x4_from_private(a_[2]), easysimd_float32x4_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_f32
  #define vld4q_f32(a) easysimd_vld4q_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2x4_t
easysimd_vld4q_f64(easysimd_float64 const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld4q_f64(ptr);
  #else
    easysimd_float64x2_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_float64x2_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_float64x2x4_t s_ = { { easysimd_float64x2_from_private(a_[0]), easysimd_float64x2_from_private(a_[1]),
                                 easysimd_float64x2_from_private(a_[2]), easysimd_float64x2_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4q_f64
  #define vld4q_f64(a) easysimd_vld4q_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16x4_t
easysimd_vld4q_s8(int8_t const ptr[HEDLEY_ARRAY_PARAM(64)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4q_s8(ptr);
  #else
    easysimd_int8x16_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_int8x16_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_int8x16x4_t s_ = { { easysimd_int8x16_from_private(a_[0]), easysimd_int8x16_from_private(a_[1]),
                               easysimd_int8x16_from_private(a_[2]), easysimd_int8x16_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_s8
  #define vld4q_s8(a) easysimd_vld4q_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8x4_t
easysimd_vld4q_s16(int16_t const ptr[HEDLEY_ARRAY_PARAM(32)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4q_s16(ptr);
  #else
    easysimd_int16x8_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_int16x8_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_int16x8x4_t s_ = { { easysimd_int16x8_from_private(a_[0]), easysimd_int16x8_from_private(a_[1]),
                               easysimd_int16x8_from_private(a_[2]), easysimd_int16x8_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_s16
  #define vld4q_s16(a) easysimd_vld4q_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4x4_t
easysimd_vld4q_s32(int32_t const ptr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4q_s32(ptr);
  #else
    easysimd_int32x4_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_int32x4_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_int32x4x4_t s_ = { { easysimd_int32x4_from_private(a_[0]), easysimd_int32x4_from_private(a_[1]),
                               easysimd_int32x4_from_private(a_[2]), easysimd_int32x4_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_s32
  #define vld4q_s32(a) easysimd_vld4q_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2x4_t
easysimd_vld4q_s64(int64_t const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld4q_s64(ptr);
  #else
    easysimd_int64x2_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_int64x2_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_int64x2x4_t s_ = { { easysimd_int64x2_from_private(a_[0]), easysimd_int64x2_from_private(a_[1]),
                               easysimd_int64x2_from_private(a_[2]), easysimd_int64x2_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4q_s64
  #define vld4q_s64(a) easysimd_vld4q_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16x4_t
easysimd_vld4q_u8(uint8_t const ptr[HEDLEY_ARRAY_PARAM(64)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4q_u8(ptr);
  #else
    easysimd_uint8x16_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_uint8x16_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_uint8x16x4_t s_ = { { easysimd_uint8x16_from_private(a_[0]), easysimd_uint8x16_from_private(a_[1]),
                                easysimd_uint8x16_from_private(a_[2]), easysimd_uint8x16_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_u8
  #define vld4q_u8(a) easysimd_vld4q_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8x4_t
easysimd_vld4q_u16(uint16_t const ptr[HEDLEY_ARRAY_PARAM(32)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4q_u16(ptr);
  #else
    easysimd_uint16x8_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_uint16x8_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_uint16x8x4_t s_ = { { easysimd_uint16x8_from_private(a_[0]), easysimd_uint16x8_from_private(a_[1]),
                                easysimd_uint16x8_from_private(a_[2]), easysimd_uint16x8_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_u16
  #define vld4q_u16(a) easysimd_vld4q_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4x4_t
easysimd_vld4q_u32(uint32_t const ptr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld4q_u32(ptr);
  #else
    easysimd_uint32x4_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_uint32x4_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_uint32x4x4_t s_ = { { easysimd_uint32x4_from_private(a_[0]), easysimd_uint32x4_from_private(a_[1]),
                                easysimd_uint32x4_from_private(a_[2]), easysimd_uint32x4_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld4q_u32
  #define vld4q_u32(a) easysimd_vld4q_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2x4_t
easysimd_vld4q_u64(uint64_t const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld4q_u64(ptr);
  #else
    easysimd_uint64x2_private a_[4];
    for (size_t i = 0; i < (sizeof(easysimd_uint64x2_t) / sizeof(*ptr)) * 4 ; i++) {
      a_[i % 4].values[i / 4] = ptr[i];
    }
    easysimd_uint64x2x4_t s_ = { { easysimd_uint64x2_from_private(a_[0]), easysimd_uint64x2_from_private(a_[1]),
                                easysimd_uint64x2_from_private(a_[2]), easysimd_uint64x2_from_private(a_[3]) } };
    return s_;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld4q_u64
  #define vld4q_u64(a) easysimd_vld4q_u64((a))
#endif

#endif /* !defined(EASYSIMD_BUG_INTEL_857088) */

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_LD4_H) */
