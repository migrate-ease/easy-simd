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
 *   2020      Sean Maher <seanptmaher@gmail.com> (Copyright owned by Google, LLC)
 *   2020      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_ARM_NEON_DUP_N_H)
#define EASYSIMD_ARM_NEON_DUP_N_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float16x4_t
easysimd_vdup_n_f16(easysimd_float16 value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vdup_n_f16(value);
  #else
    easysimd_float16x4_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = value;
    }

    return easysimd_float16x4_from_private(r_);
  #endif
}
#define easysimd_vmov_n_f16 easysimd_vdup_n_f16
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_f16
  #define vdup_n_f16(value) easysimd_vdup_n_f16((value))
  #undef vmov_n_f16
  #define vmov_n_f16(value) easysimd_vmov_n_f16((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vdup_n_f32(float value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdup_n_f32(value);
  #else
    easysimd_float32x2_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = value;
    }

    return easysimd_float32x2_from_private(r_);
  #endif
}
#define easysimd_vmov_n_f32 easysimd_vdup_n_f32
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_f32
  #define vdup_n_f32(value) easysimd_vdup_n_f32((value))
  #undef vmov_n_f32
  #define vmov_n_f32(value) easysimd_vmov_n_f32((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vdup_n_f64(double value) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vdup_n_f64(value);
  #else
    easysimd_float64x1_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = value;
    }

    return easysimd_float64x1_from_private(r_);
  #endif
}
#define easysimd_vmov_n_f64 easysimd_vdup_n_f64
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_f64
  #define vdup_n_f64(value) easysimd_vdup_n_f64((value))
  #undef vmov_n_f64
  #define vmov_n_f64(value) easysimd_vmov_n_f64((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vdup_n_s8(int8_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdup_n_s8(value);
  #else
    easysimd_int8x8_private r_;

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_set1_pi8(value);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_int8x8_from_private(r_);
  #endif
}
#define easysimd_vmov_n_s8 easysimd_vdup_n_s8
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_s8
  #define vdup_n_s8(value) easysimd_vdup_n_s8((value))
  #undef vmov_n_s8
  #define vmov_n_s8(value) easysimd_vmov_n_s8((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vdup_n_s16(int16_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdup_n_s16(value);
  #else
    easysimd_int16x4_private r_;

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_set1_pi16(value);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_int16x4_from_private(r_);
  #endif
}
#define easysimd_vmov_n_s16 easysimd_vdup_n_s16
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_s16
  #define vdup_n_s16(value) easysimd_vdup_n_s16((value))
  #undef vmov_n_s16
  #define vmov_n_s16(value) easysimd_vmov_n_s16((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vdup_n_s32(int32_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdup_n_s32(value);
  #else
    easysimd_int32x2_private r_;

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_set1_pi32(value);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_int32x2_from_private(r_);
  #endif
}
#define easysimd_vmov_n_s32 easysimd_vdup_n_s32
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_s32
  #define vdup_n_s32(value) easysimd_vdup_n_s32((value))
  #undef vmov_n_s32
  #define vmov_n_s32(value) easysimd_vmov_n_s32((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vdup_n_s64(int64_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdup_n_s64(value);
  #else
    easysimd_int64x1_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = value;
    }

    return easysimd_int64x1_from_private(r_);
  #endif
}
#define easysimd_vmov_n_s64 easysimd_vdup_n_s64
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_s64
  #define vdup_n_s64(value) easysimd_vdup_n_s64((value))
  #undef vmov_n_s64
  #define vmov_n_s64(value) easysimd_vmov_n_s64((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vdup_n_u8(uint8_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdup_n_u8(value);
  #else
    easysimd_uint8x8_private r_;

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_set1_pi8(HEDLEY_STATIC_CAST(int8_t, value));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#define easysimd_vmov_n_u8 easysimd_vdup_n_u8
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_u8
  #define vdup_n_u8(value) easysimd_vdup_n_u8((value))
  #undef vmov_n_u8
  #define vmov_n_u8(value) easysimd_vmov_n_u8((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vdup_n_u16(uint16_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdup_n_u16(value);
  #else
    easysimd_uint16x4_private r_;

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_set1_pi16(HEDLEY_STATIC_CAST(int16_t, value));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_uint16x4_from_private(r_);
  #endif
}
#define easysimd_vmov_n_u16 easysimd_vdup_n_u16
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_u16
  #define vdup_n_u16(value) easysimd_vdup_n_u16((value))
  #undef vmov_n_u16
  #define vmov_n_u16(value) easysimd_vmov_n_u16((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vdup_n_u32(uint32_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdup_n_u32(value);
  #else
    easysimd_uint32x2_private r_;

    #if defined(EASYSIMD_X86_MMX_NATIVE)
      r_.m64 = _mm_set1_pi32(HEDLEY_STATIC_CAST(int32_t, value));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_uint32x2_from_private(r_);
  #endif
}
#define easysimd_vmov_n_u32 easysimd_vdup_n_u32
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_u32
  #define vdup_n_u32(value) easysimd_vdup_n_u32((value))
  #undef vmov_n_u32
  #define vmov_n_u32(value) easysimd_vmov_n_u32((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vdup_n_u64(uint64_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdup_n_u64(value);
  #else
    easysimd_uint64x1_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = value;
    }

    return easysimd_uint64x1_from_private(r_);
  #endif
}
#define easysimd_vmov_n_u64 easysimd_vdup_n_u64
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdup_n_u64
  #define vdup_n_u64(value) easysimd_vdup_n_u64((value))
  #undef vmov_n_u64
  #define vmov_n_u64(value) easysimd_vmov_n_u64((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float16x8_t
easysimd_vdupq_n_f16(easysimd_float16 value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_ARM_NEON_FP16)
    return vdupq_n_f16(value);
  #else
    easysimd_float16x8_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = value;
    }

    return easysimd_float16x8_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_f32 easysimd_vdupq_n_f32
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_f16
  #define vdupq_n_f16(value) easysimd_vdupq_n_f16((value))
  #undef vmovq_n_f16
  #define vmovq_n_f16(value) easysimd_vmovq_n_f16((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vdupq_n_f32(float value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdupq_n_f32(value);
  #else
    easysimd_float32x4_private r_;

    #if defined(EASYSIMD_X86_SSE_NATIVE)
      r_.m128 = _mm_set1_ps(value);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_float32x4_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_f32 easysimd_vdupq_n_f32
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_f32
  #define vdupq_n_f32(value) easysimd_vdupq_n_f32((value))
  #undef vmovq_n_f32
  #define vmovq_n_f32(value) easysimd_vmovq_n_f32((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vdupq_n_f64(double value) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vdupq_n_f64(value);
  #else
    easysimd_float64x2_private r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128d = _mm_set1_pd(value);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_float64x2_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_f64 easysimd_vdupq_n_f64
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_f64
  #define vdupq_n_f64(value) easysimd_vdupq_n_f64((value))
  #undef vmovq_n_f64
  #define vmovq_n_f64(value) easysimd_vmovq_n_f64((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vdupq_n_s8(int8_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdupq_n_s8(value);
  #else
    easysimd_int8x16_private r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_set1_epi8(value);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_s8 easysimd_vdupq_n_s8
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_s8
  #define vdupq_n_s8(value) easysimd_vdupq_n_s8((value))
  #undef vmovq_n_s8
  #define vmovq_n_s8(value) easysimd_vmovq_n_s8((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vdupq_n_s16(int16_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdupq_n_s16(value);
  #else
    easysimd_int16x8_private r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_set1_epi16(value);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_int16x8_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_s16 easysimd_vdupq_n_s16
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_s16
  #define vdupq_n_s16(value) easysimd_vdupq_n_s16((value))
  #undef vmovq_n_s16
  #define vmovq_n_s16(value) easysimd_vmovq_n_s16((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vdupq_n_s32(int32_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdupq_n_s32(value);
  #else
    easysimd_int32x4_private r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_set1_epi32(value);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_s32 easysimd_vdupq_n_s32
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_s32
  #define vdupq_n_s32(value) easysimd_vdupq_n_s32((value))
  #undef vmovq_n_s32
  #define vmovq_n_s32(value) easysimd_vmovq_n_s32((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vdupq_n_s64(int64_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdupq_n_s64(value);
  #else
    easysimd_int64x2_private r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE) && (!defined(HEDLEY_MSVC_VERSION) || HEDLEY_MSVC_VERSION_CHECK(19,0,0))
      r_.m128i = _mm_set1_epi64x(value);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_int64x2_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_s64 easysimd_vdupq_n_s64
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_s64
  #define vdupq_n_s64(value) easysimd_vdupq_n_s64((value))
  #undef vmovq_n_s64
  #define vmovq_n_s64(value) easysimd_vmovq_n_s64((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vdupq_n_u8(uint8_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdupq_n_u8(value);
  #else
    easysimd_uint8x16_private r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, value));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_u8 easysimd_vdupq_n_u8
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_u8
  #define vdupq_n_u8(value) easysimd_vdupq_n_u8((value))
  #undef vmovq_n_u8
  #define vmovq_n_u8(value) easysimd_vmovq_n_u8((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vdupq_n_u16(uint16_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdupq_n_u16(value);
  #else
    easysimd_uint16x8_private r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_set1_epi16(HEDLEY_STATIC_CAST(int16_t, value));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_u16 easysimd_vdupq_n_u16
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_u16
  #define vdupq_n_u16(value) easysimd_vdupq_n_u16((value))
  #undef vmovq_n_u16
  #define vmovq_n_u16(value) easysimd_vmovq_n_u16((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vdupq_n_u32(uint32_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdupq_n_u32(value);
  #else
    easysimd_uint32x4_private r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_set1_epi32(HEDLEY_STATIC_CAST(int32_t, value));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_u32 easysimd_vdupq_n_u32
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_u32
  #define vdupq_n_u32(value) easysimd_vdupq_n_u32((value))
  #undef vmovq_n_u32
  #define vmovq_n_u32(value) easysimd_vmovq_n_u32((value))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vdupq_n_u64(uint64_t value) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vdupq_n_u64(value);
  #else
    easysimd_uint64x2_private r_;

    #if defined(EASYSIMD_X86_SSE2_NATIVE) && (!defined(HEDLEY_MSVC_VERSION) || HEDLEY_MSVC_VERSION_CHECK(19,0,0))
      r_.m128i = _mm_set1_epi64x(HEDLEY_STATIC_CAST(int64_t, value));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = value;
      }
    #endif

    return easysimd_uint64x2_from_private(r_);
  #endif
}
#define easysimd_vmovq_n_u64 easysimd_vdupq_n_u64
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vdupq_n_u64
  #define vdupq_n_u64(value) easysimd_vdupq_n_u64((value))
  #undef vmovq_n_u64
  #define vmovq_n_u64(value) easysimd_vmovq_n_u64((value))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_DUP_N_H) */
