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
 *   2021      Zhi An Ng <zhin@google.com> (Copyright owned by Google, LLC)
 */

#if !defined(EASYSIMD_ARM_NEON_LD2_H)
#define EASYSIMD_ARM_NEON_LD2_H

#include "get_low.h"
#include "get_high.h"
#include "ld1.h"
#include "uzp.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
#if HEDLEY_GCC_VERSION_CHECK(7,0,0)
  EASYSIMD_DIAGNOSTIC_DISABLE_MAYBE_UNINITIAZILED_
#endif
EASYSIMD_BEGIN_DECLS_

#if !defined(EASYSIMD_BUG_INTEL_857088)

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8x2_t
easysimd_vld2_s8(int8_t const ptr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2_s8(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_int8x16_private a_ = easysimd_int8x16_to_private(easysimd_vld1q_s8(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.values, a_.values, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    easysimd_int8x8x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_int8x8_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_int8x8x2_t r = { {
      easysimd_int8x8_from_private(r_[0]),
      easysimd_int8x8_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2_s8
  #define vld2_s8(a) easysimd_vld2_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4x2_t
easysimd_vld2_s16(int16_t const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2_s16(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_int16x8_private a_ = easysimd_int16x8_to_private(easysimd_vld1q_s16(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.values, a_.values, 0, 2, 4, 6, 1, 3, 5, 7);
    easysimd_int16x4x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_int16x4_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_int16x4x2_t r = { {
      easysimd_int16x4_from_private(r_[0]),
      easysimd_int16x4_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2_s16
  #define vld2_s16(a) easysimd_vld2_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2x2_t
easysimd_vld2_s32(int32_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2_s32(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_int32x4_private a_ = easysimd_int32x4_to_private(easysimd_vld1q_s32(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.values, a_.values, 0, 2, 1, 3);
    easysimd_int32x2x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_int32x2_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_int32x2x2_t r = { {
      easysimd_int32x2_from_private(r_[0]),
      easysimd_int32x2_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2_s32
  #define vld2_s32(a) easysimd_vld2_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1x2_t
easysimd_vld2_s64(int64_t const ptr[HEDLEY_ARRAY_PARAM(2)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2_s64(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_int64x2_private a_ = easysimd_int64x2_to_private(easysimd_vld1q_s64(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.values, a_.values, 0, 1);
    easysimd_int64x1x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_int64x1_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_int64x1x2_t r = { {
      easysimd_int64x1_from_private(r_[0]),
      easysimd_int64x1_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2_s64
  #define vld2_s64(a) easysimd_vld2_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8x2_t
easysimd_vld2_u8(uint8_t const ptr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2_u8(ptr);

  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_uint8x16_private a_ = easysimd_uint8x16_to_private(easysimd_vld1q_u8(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.values, a_.values, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    easysimd_uint8x8x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_uint8x8_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_uint8x8x2_t r = { {
      easysimd_uint8x8_from_private(r_[0]),
      easysimd_uint8x8_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2_u8
  #define vld2_u8(a) easysimd_vld2_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4x2_t
easysimd_vld2_u16(uint16_t const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2_u16(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_uint16x8_private a_ = easysimd_uint16x8_to_private(easysimd_vld1q_u16(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.values, a_.values, 0, 2, 4, 6, 1, 3, 5, 7);
    easysimd_uint16x4x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_uint16x4_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_uint16x4x2_t r = { {
      easysimd_uint16x4_from_private(r_[0]),
      easysimd_uint16x4_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2_u16
  #define vld2_u16(a) easysimd_vld2_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2x2_t
easysimd_vld2_u32(uint32_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2_u32(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_uint32x4_private a_ = easysimd_uint32x4_to_private(easysimd_vld1q_u32(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.values, a_.values, 0, 2, 1, 3);
    easysimd_uint32x2x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_uint32x2_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_uint32x2x2_t r = { {
      easysimd_uint32x2_from_private(r_[0]),
      easysimd_uint32x2_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2_u32
  #define vld2_u32(a) easysimd_vld2_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1x2_t
easysimd_vld2_u64(uint64_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2_u64(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_uint64x2_private a_ = easysimd_uint64x2_to_private(easysimd_vld1q_u64(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.values, a_.values, 0, 1);
    easysimd_uint64x1x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_uint64x1_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_uint64x1x2_t r = { {
      easysimd_uint64x1_from_private(r_[0]),
      easysimd_uint64x1_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2_u64
  #define vld2_u64(a) easysimd_vld2_u64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2x2_t
easysimd_vld2_f32(easysimd_float32_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2_f32(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_float32x4_private a_ = easysimd_float32x4_to_private(easysimd_vld1q_f32(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.values, a_.values, 0, 2, 1, 3);
    easysimd_float32x2x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_float32x2_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_float32x2x2_t r = { {
      easysimd_float32x2_from_private(r_[0]),
      easysimd_float32x2_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2_f32
  #define vld2_f32(a) easysimd_vld2_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1x2_t
easysimd_vld2_f64(easysimd_float64_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld2_f64(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_SHUFFLE_VECTOR_)
    easysimd_float64x2_private a_ = easysimd_float64x2_to_private(easysimd_vld1q_f64(ptr));
    a_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.values, a_.values, 0, 1);
    easysimd_float64x1x2_t r;
    easysimd_memcpy(&r, &a_, sizeof(r));
    return r;
  #else
    easysimd_float64x1_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_float64x1x2_t r = { {
      easysimd_float64x1_from_private(r_[0]),
      easysimd_float64x1_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld2_f64
  #define vld2_f64(a) easysimd_vld2_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16x2_t
easysimd_vld2q_s8(int8_t const ptr[HEDLEY_ARRAY_PARAM(32)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2q_s8(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return
      easysimd_vuzpq_s8(
        easysimd_vld1q_s8(&(ptr[0])),
        easysimd_vld1q_s8(&(ptr[16]))
      );
  #else
    easysimd_int8x16_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_int8x16x2_t r = { {
      easysimd_int8x16_from_private(r_[0]),
      easysimd_int8x16_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2q_s8
  #define vld2q_s8(a) easysimd_vld2q_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4x2_t
easysimd_vld2q_s32(int32_t const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2q_s32(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return
      easysimd_vuzpq_s32(
        easysimd_vld1q_s32(&(ptr[0])),
        easysimd_vld1q_s32(&(ptr[4]))
      );
  #else
    easysimd_int32x4_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_int32x4x2_t r = { {
      easysimd_int32x4_from_private(r_[0]),
      easysimd_int32x4_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2q_s32
  #define vld2q_s32(a) easysimd_vld2q_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8x2_t
easysimd_vld2q_s16(int16_t const ptr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2q_s16(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return
      easysimd_vuzpq_s16(
        easysimd_vld1q_s16(&(ptr[0])),
        easysimd_vld1q_s16(&(ptr[8]))
      );
  #else
    easysimd_int16x8_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_int16x8x2_t r = { {
      easysimd_int16x8_from_private(r_[0]),
      easysimd_int16x8_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2q_s16
  #define vld2q_s16(a) easysimd_vld2q_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2x2_t
easysimd_vld2q_s64(int64_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld2q_s64(ptr);
  #else
    easysimd_int64x2_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_int64x2x2_t r = { {
      easysimd_int64x2_from_private(r_[0]),
      easysimd_int64x2_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld2q_s64
  #define vld2q_s64(a) easysimd_vld2q_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16x2_t
easysimd_vld2q_u8(uint8_t const ptr[HEDLEY_ARRAY_PARAM(32)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2q_u8(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return
      easysimd_vuzpq_u8(
        easysimd_vld1q_u8(&(ptr[ 0])),
        easysimd_vld1q_u8(&(ptr[16]))
      );
  #else
    easysimd_uint8x16_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_uint8x16x2_t r = { {
      easysimd_uint8x16_from_private(r_[0]),
      easysimd_uint8x16_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2q_u8
  #define vld2q_u8(a) easysimd_vld2q_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8x2_t
easysimd_vld2q_u16(uint16_t const ptr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2q_u16(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return
      easysimd_vuzpq_u16(
        easysimd_vld1q_u16(&(ptr[0])),
        easysimd_vld1q_u16(&(ptr[8]))
      );
  #else
    easysimd_uint16x8_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_uint16x8x2_t r = { {
      easysimd_uint16x8_from_private(r_[0]),
      easysimd_uint16x8_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2q_u16
  #define vld2q_u16(a) easysimd_vld2q_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4x2_t
easysimd_vld2q_u32(uint32_t const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2q_u32(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return
      easysimd_vuzpq_u32(
        easysimd_vld1q_u32(&(ptr[0])),
        easysimd_vld1q_u32(&(ptr[4]))
      );
  #else
    easysimd_uint32x4_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_uint32x4x2_t r = { {
      easysimd_uint32x4_from_private(r_[0]),
      easysimd_uint32x4_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2q_u32
  #define vld2q_u32(a) easysimd_vld2q_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2x2_t
easysimd_vld2q_u64(uint64_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld2q_u64(ptr);
  #else
    easysimd_uint64x2_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_uint64x2x2_t r = { {
      easysimd_uint64x2_from_private(r_[0]),
      easysimd_uint64x2_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld2q_u64
  #define vld2q_u64(a) easysimd_vld2q_u64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4x2_t
easysimd_vld2q_f32(easysimd_float32_t const ptr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld2q_f32(ptr);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return
      easysimd_vuzpq_f32(
        easysimd_vld1q_f32(&(ptr[0])),
        easysimd_vld1q_f32(&(ptr[4]))
      );
  #else
    easysimd_float32x4_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])); i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_float32x4x2_t r = { {
      easysimd_float32x4_from_private(r_[0]),
      easysimd_float32x4_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld2q_f32
  #define vld2q_f32(a) easysimd_vld2q_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2x2_t
easysimd_vld2q_f64(easysimd_float64_t const ptr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld2q_f64(ptr);
  #else
    easysimd_float64x2_private r_[2];

    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_[0])) ; i++) {
      for (size_t j = 0 ; j < (sizeof(r_[0].values) / sizeof(r_[0].values[0])) ; j++) {
        r_[i].values[j] = ptr[i + (j * (sizeof(r_) / sizeof(r_[0])))];
      }
    }

    easysimd_float64x2x2_t r = { {
      easysimd_float64x2_from_private(r_[0]),
      easysimd_float64x2_from_private(r_[1]),
    } };

    return r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld2q_f64
  #define vld2q_f64(a) easysimd_vld2q_f64((a))
#endif

#endif /* !defined(EASYSIMD_BUG_INTEL_857088) */

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_LD2_H) */
