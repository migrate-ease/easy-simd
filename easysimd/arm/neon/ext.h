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

#if !defined(EASYSIMD_ARM_NEON_EXT_H)
#define EASYSIMD_ARM_NEON_EXT_H
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vext_f32(easysimd_float32x2_t a, easysimd_float32x2_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_float32x2_t r;
    EASYSIMD_CONSTIFY_2_(vext_f32, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_float32x2_private
      a_ = easysimd_float32x2_to_private(a),
      b_ = easysimd_float32x2_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 1];
    }
    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_f32(a, b, n) easysimd_float32x2_from_m64(_mm_alignr_pi8(easysimd_float32x2_to_m64(b), easysimd_float32x2_to_m64(a), n * sizeof(easysimd_float32)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vext_f32(a, b, n) (__extension__ ({ \
      easysimd_float32x2_private easysimd_vext_f32_r_; \
      easysimd_vext_f32_r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 8, easysimd_float32x2_to_private(a).values, easysimd_float32x2_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1))); \
      easysimd_float32x2_from_private(easysimd_vext_f32_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vext_f32
  #define vext_f32(a, b, n) easysimd_vext_f32((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vext_f64(easysimd_float64x1_t a, easysimd_float64x1_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 0) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    (void) n;
    return vext_f64(a, b, 0);
  #else
    easysimd_float64x1_private
      a_ = easysimd_float64x1_to_private(a),
      b_ = easysimd_float64x1_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 0];
    }
    return easysimd_float64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_f64(a, b, n) easysimd_float64x1_from_m64(_mm_alignr_pi8(easysimd_float64x1_to_m64(b), easysimd_float64x1_to_m64(a), n * sizeof(easysimd_float64)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vext_f64(a, b, n) (__extension__ ({ \
      easysimd_float64x1_private easysimd_vext_f64_r_; \
      easysimd_vext_f64_r_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 8, easysimd_float64x1_to_private(a).values, easysimd_float64x1_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, (n))); \
      easysimd_float64x1_from_private(easysimd_vext_f64_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vext_f64
  #define vext_f64(a, b, n) easysimd_vext_f64((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vext_s8(easysimd_int8x8_t a, easysimd_int8x8_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int8x8_t r;
    EASYSIMD_CONSTIFY_8_(vext_s8, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_int8x8_private
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 7];
    }
    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_s8(a, b, n) easysimd_int8x8_from_m64(_mm_alignr_pi8(easysimd_int8x8_to_m64(b), easysimd_int8x8_to_m64(a), n * sizeof(int8_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vext_s8(a, b, n) (__extension__ ({ \
      easysimd_int8x8_private easysimd_vext_s8_r_; \
      easysimd_vext_s8_r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 8, easysimd_int8x8_to_private(a).values, easysimd_int8x8_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 4)), HEDLEY_STATIC_CAST(int8_t, ((n) + 5)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 6)), HEDLEY_STATIC_CAST(int8_t, ((n) + 7))); \
      easysimd_int8x8_from_private(easysimd_vext_s8_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vext_s8
  #define vext_s8(a, b, n) easysimd_vext_s8((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vext_s16(easysimd_int16x4_t a, easysimd_int16x4_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int16x4_t r;
    EASYSIMD_CONSTIFY_4_(vext_s16, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_int16x4_private
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 3];
    }
    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_s16(a, b, n) easysimd_int16x4_from_m64(_mm_alignr_pi8(easysimd_int16x4_to_m64(b), easysimd_int16x4_to_m64(a), n * sizeof(int16_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)  && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vext_s16(a, b, n) (__extension__ ({ \
      easysimd_int16x4_private easysimd_vext_s16_r_; \
      easysimd_vext_s16_r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 8, easysimd_int16x4_to_private(a).values, easysimd_int16x4_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3))); \
      easysimd_int16x4_from_private(easysimd_vext_s16_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vext_s16
  #define vext_s16(a, b, n) easysimd_vext_s16((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vext_s32(easysimd_int32x2_t a, easysimd_int32x2_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int32x2_t r;
    EASYSIMD_CONSTIFY_2_(vext_s32, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_int32x2_private
      a_ = easysimd_int32x2_to_private(a),
      b_ = easysimd_int32x2_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 1];
    }
    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_s32(a, b, n) easysimd_int32x2_from_m64(_mm_alignr_pi8(easysimd_int32x2_to_m64(b), easysimd_int32x2_to_m64(a), n * sizeof(int32_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vext_s32(a, b, n) (__extension__ ({ \
      easysimd_int32x2_private easysimd_vext_s32_r_; \
      easysimd_vext_s32_r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 8, easysimd_int32x2_to_private(a).values, easysimd_int32x2_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1))); \
      easysimd_int32x2_from_private(easysimd_vext_s32_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vext_s32
  #define vext_s32(a, b, n) easysimd_vext_s32((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vext_s64(easysimd_int64x1_t a, easysimd_int64x1_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 0) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    (void) n;
    return vext_s64(a, b, 0);
  #else
    easysimd_int64x1_private
      a_ = easysimd_int64x1_to_private(a),
      b_ = easysimd_int64x1_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 0];
    }
    return easysimd_int64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_s64(a, b, n) easysimd_int64x1_from_m64(_mm_alignr_pi8(easysimd_int64x1_to_m64(b), easysimd_int64x1_to_m64(a), n * sizeof(int64_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vext_s64(a, b, n) (__extension__ ({ \
      easysimd_int64x1_private easysimd_vext_s64_r_; \
      easysimd_vext_s64_r_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 8, easysimd_int64x1_to_private(a).values, easysimd_int64x1_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0))); \
      easysimd_int64x1_from_private(easysimd_vext_s64_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vext_s64
  #define vext_s64(a, b, n) easysimd_vext_s64((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vext_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint8x8_t r;
    EASYSIMD_CONSTIFY_8_(vext_u8, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_uint8x8_private
      a_ = easysimd_uint8x8_to_private(a),
      b_ = easysimd_uint8x8_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 7];
    }
    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_u8(a, b, n) easysimd_uint8x8_from_m64(_mm_alignr_pi8(easysimd_uint8x8_to_m64(b), easysimd_uint8x8_to_m64(a), n * sizeof(uint8_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vext_u8(a, b, n) (__extension__ ({ \
      easysimd_uint8x8_private easysimd_vext_u8_r_; \
      easysimd_vext_u8_r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 8, easysimd_uint8x8_to_private(a).values, easysimd_uint8x8_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 4)), HEDLEY_STATIC_CAST(int8_t, ((n) + 5)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 6)), HEDLEY_STATIC_CAST(int8_t, ((n) + 7))); \
      easysimd_uint8x8_from_private(easysimd_vext_u8_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vext_u8
  #define vext_u8(a, b, n) easysimd_vext_u8((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vext_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint16x4_t r;
    EASYSIMD_CONSTIFY_4_(vext_u16, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_uint16x4_private
      a_ = easysimd_uint16x4_to_private(a),
      b_ = easysimd_uint16x4_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 3];
    }
    return easysimd_uint16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_u16(a, b, n) easysimd_uint16x4_from_m64(_mm_alignr_pi8(easysimd_uint16x4_to_m64(b), easysimd_uint16x4_to_m64(a), n * sizeof(uint16_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vext_u16(a, b, n) (__extension__ ({ \
      easysimd_uint16x4_private easysimd_vext_u16_r_; \
      easysimd_vext_u16_r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 8, easysimd_uint16x4_to_private(a).values, easysimd_uint16x4_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3))); \
      easysimd_uint16x4_from_private(easysimd_vext_u16_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vext_u16
  #define vext_u16(a, b, n) easysimd_vext_u16((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vext_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint32x2_t r;
    EASYSIMD_CONSTIFY_2_(vext_u32, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_uint32x2_private
      a_ = easysimd_uint32x2_to_private(a),
      b_ = easysimd_uint32x2_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 1];
    }
    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_u32(a, b, n) easysimd_uint32x2_from_m64(_mm_alignr_pi8(easysimd_uint32x2_to_m64(b), easysimd_uint32x2_to_m64(a), n * sizeof(uint32_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32) && !defined(EASYSIMD_BUG_GCC_100760)
  #define easysimd_vext_u32(a, b, n) (__extension__ ({ \
      easysimd_uint32x2_private easysimd_vext_u32_r_; \
      easysimd_vext_u32_r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 8, easysimd_uint32x2_to_private(a).values, easysimd_uint32x2_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1))); \
      easysimd_uint32x2_from_private(easysimd_vext_u32_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vext_u32
  #define vext_u32(a, b, n) easysimd_vext_u32((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vext_u64(easysimd_uint64x1_t a, easysimd_uint64x1_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 0) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    (void) n;
    return vext_u64(a, b, 0);
  #else
    easysimd_uint64x1_private
      a_ = easysimd_uint64x1_to_private(a),
      b_ = easysimd_uint64x1_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 0];
    }
    return easysimd_uint64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vext_u64(a, b, n) easysimd_uint64x1_from_m64(_mm_alignr_pi8(easysimd_uint64x1_to_m64(b), easysimd_uint64x1_to_m64(a), n * sizeof(uint64_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vext_u64(a, b, n) (__extension__ ({ \
      easysimd_uint64x1_private easysimd_vext_u64_r_; \
      easysimd_vext_u64_r_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 8, easysimd_uint64x1_to_private(a).values, easysimd_uint64x1_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0))); \
      easysimd_uint64x1_from_private(easysimd_vext_u64_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vext_u64
  #define vext_u64(a, b, n) easysimd_vext_u64((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vextq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_float32x4_t r;
    EASYSIMD_CONSTIFY_4_(vextq_f32, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_float32x4_private
      a_ = easysimd_float32x4_to_private(a),
      b_ = easysimd_float32x4_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 3];
    }
    return easysimd_float32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_f32(a, b, n) easysimd_float32x4_from_m128(_mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(easysimd_float32x4_to_m128(b)), _mm_castps_si128(easysimd_float32x4_to_m128(a)), n * sizeof(easysimd_float32))))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_f32(a, b, n) (__extension__ ({ \
      easysimd_float32x4_private easysimd_vextq_f32_r_; \
      easysimd_vextq_f32_r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, easysimd_float32x4_to_private(a).values, easysimd_float32x4_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3))); \
      easysimd_float32x4_from_private(easysimd_vextq_f32_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vextq_f32
  #define vextq_f32(a, b, n) easysimd_vextq_f32((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vextq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd_float64x2_t r;
    EASYSIMD_CONSTIFY_2_(vextq_f64, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_float64x2_private
      a_ = easysimd_float64x2_to_private(a),
      b_ = easysimd_float64x2_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 1];
    }
    return easysimd_float64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_f64(a, b, n) easysimd_float64x2_from_m128d(_mm_castsi128_pd(_mm_alignr_epi8(_mm_castpd_si128(easysimd_float64x2_to_m128d(b)), _mm_castpd_si128(easysimd_float64x2_to_m128d(a)), n * sizeof(easysimd_float64))))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_f64(a, b, n) (__extension__ ({ \
      easysimd_float64x2_private easysimd_vextq_f64_r_; \
      easysimd_vextq_f64_r_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 16, easysimd_float64x2_to_private(a).values, easysimd_float64x2_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1))); \
      easysimd_float64x2_from_private(easysimd_vextq_f64_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vextq_f64
  #define vextq_f64(a, b, n) easysimd_vextq_f64((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vextq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 15) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int8x16_t r;
    EASYSIMD_CONSTIFY_16_(vextq_s8, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_int8x16_private
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 15];
    }
    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_s8(a, b, n) easysimd_int8x16_from_m128i(_mm_alignr_epi8(easysimd_int8x16_to_m128i(b), easysimd_int8x16_to_m128i(a), n * sizeof(int8_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_s8(a, b, n) (__extension__ ({ \
      easysimd_int8x16_private easysimd_vextq_s8_r_; \
      easysimd_vextq_s8_r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 16, easysimd_int8x16_to_private(a).values, easysimd_int8x16_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 4)), HEDLEY_STATIC_CAST(int8_t, ((n) + 5)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 6)), HEDLEY_STATIC_CAST(int8_t, ((n) + 7)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 8)), HEDLEY_STATIC_CAST(int8_t, ((n) + 9)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 10)), HEDLEY_STATIC_CAST(int8_t, ((n) + 11)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 12)), HEDLEY_STATIC_CAST(int8_t, ((n) + 13)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 14)), HEDLEY_STATIC_CAST(int8_t, ((n) + 15))); \
      easysimd_int8x16_from_private(easysimd_vextq_s8_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vextq_s8
  #define vextq_s8(a, b, n) easysimd_vextq_s8((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vextq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int16x8_t r;
    EASYSIMD_CONSTIFY_8_(vextq_s16, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_int16x8_private
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 7];
    }
    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_s16(a, b, n) easysimd_int16x8_from_m128i(_mm_alignr_epi8(easysimd_int16x8_to_m128i(b), easysimd_int16x8_to_m128i(a), n * sizeof(int16_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_s16(a, b, n) (__extension__ ({ \
      easysimd_int16x8_private easysimd_vextq_s16_r_; \
      easysimd_vextq_s16_r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 16, easysimd_int16x8_to_private(a).values, easysimd_int16x8_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 4)), HEDLEY_STATIC_CAST(int8_t, ((n) + 5)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 6)), HEDLEY_STATIC_CAST(int8_t, ((n) + 7))); \
      easysimd_int16x8_from_private(easysimd_vextq_s16_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vextq_s16
  #define vextq_s16(a, b, n) easysimd_vextq_s16((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vextq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int32x4_t r;
    EASYSIMD_CONSTIFY_4_(vextq_s32, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_int32x4_private
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 3];
    }
    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_s32(a, b, n) easysimd_int32x4_from_m128i(_mm_alignr_epi8(easysimd_int32x4_to_m128i(b), easysimd_int32x4_to_m128i(a), n * sizeof(int32_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_s32(a, b, n) (__extension__ ({ \
      easysimd_int32x4_private easysimd_vextq_s32_r_; \
      easysimd_vextq_s32_r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, easysimd_int32x4_to_private(a).values, easysimd_int32x4_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3))); \
      easysimd_int32x4_from_private(easysimd_vextq_s32_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vextq_s32
  #define vextq_s32(a, b, n) easysimd_vextq_s32((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vextq_s64(easysimd_int64x2_t a, easysimd_int64x2_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_int64x2_t r;
    EASYSIMD_CONSTIFY_2_(vextq_s64, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_int64x2_private
      a_ = easysimd_int64x2_to_private(a),
      b_ = easysimd_int64x2_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 1];
    }
    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_s64(a, b, n) easysimd_int64x2_from_m128i(_mm_alignr_epi8(easysimd_int64x2_to_m128i(b), easysimd_int64x2_to_m128i(a), n * sizeof(int64_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_s64(a, b, n) (__extension__ ({ \
      easysimd_int64x2_private easysimd_vextq_s64_r_; \
      easysimd_vextq_s64_r_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 16, easysimd_int64x2_to_private(a).values, easysimd_int64x2_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1))); \
      easysimd_int64x2_from_private(easysimd_vextq_s64_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vextq_s64
  #define vextq_s64(a, b, n) easysimd_vextq_s64((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vextq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 15) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint8x16_t r;
    EASYSIMD_CONSTIFY_16_(vextq_u8, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_uint8x16_private
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 15];
    }
    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_u8(a, b, n) easysimd_uint8x16_from_m128i(_mm_alignr_epi8(easysimd_uint8x16_to_m128i(b), easysimd_uint8x16_to_m128i(a), n * sizeof(uint8_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_u8(a, b, n) (__extension__ ({ \
      easysimd_uint8x16_private easysimd_vextq_u8_r_; \
      easysimd_vextq_u8_r_.values = EASYSIMD_SHUFFLE_VECTOR_(8, 16, easysimd_uint8x16_to_private(a).values, easysimd_uint8x16_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 4)), HEDLEY_STATIC_CAST(int8_t, ((n) + 5)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 6)), HEDLEY_STATIC_CAST(int8_t, ((n) + 7)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 8)), HEDLEY_STATIC_CAST(int8_t, ((n) + 9)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 10)), HEDLEY_STATIC_CAST(int8_t, ((n) + 11)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 12)), HEDLEY_STATIC_CAST(int8_t, ((n) + 13)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 14)), HEDLEY_STATIC_CAST(int8_t, ((n) + 15))); \
      easysimd_uint8x16_from_private(easysimd_vextq_u8_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vextq_u8
  #define vextq_u8(a, b, n) easysimd_vextq_u8((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vextq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 7) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint16x8_t r;
    EASYSIMD_CONSTIFY_8_(vextq_u16, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_uint16x8_private
      a_ = easysimd_uint16x8_to_private(a),
      b_ = easysimd_uint16x8_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 7];
    }
    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_u16(a, b, n) easysimd_uint16x8_from_m128i(_mm_alignr_epi8(easysimd_uint16x8_to_m128i(b), easysimd_uint16x8_to_m128i(a), n * sizeof(uint16_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_u16(a, b, n) (__extension__ ({ \
      easysimd_uint16x8_private easysimd_vextq_u16_r_; \
      easysimd_vextq_u16_r_.values = EASYSIMD_SHUFFLE_VECTOR_(16, 16, easysimd_uint16x8_to_private(a).values, easysimd_uint16x8_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 4)), HEDLEY_STATIC_CAST(int8_t, ((n) + 5)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 6)), HEDLEY_STATIC_CAST(int8_t, ((n) + 7))); \
      easysimd_uint16x8_from_private(easysimd_vextq_u16_r_); \
    }))
#elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
  #define easysimd_vextq_u16(a, b, n) (__extension__ ({ \
    easysimd_uint16x8_private r_; \
    r_.values = __builtin_shufflevector( \
        easysimd_uint16x8_to_private(a).values, \
        easysimd_uint16x8_to_private(b).values, \
        n + 0, n + 1, n + 2, n + 3, n + 4, n + 5, n + 6, n + 7); \
    easysimd_uint16x8_from_private(r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vextq_u16
  #define vextq_u16(a, b, n) easysimd_vextq_u16((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vextq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 3) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint32x4_t r;
    EASYSIMD_CONSTIFY_4_(vextq_u32, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_uint32x4_private
      a_ = easysimd_uint32x4_to_private(a),
      b_ = easysimd_uint32x4_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 3];
    }
    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_u32(a, b, n) easysimd_uint32x4_from_m128i(_mm_alignr_epi8(easysimd_uint32x4_to_m128i(b), easysimd_uint32x4_to_m128i(a), n * sizeof(uint32_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_u32(a, b, n) (__extension__ ({ \
      easysimd_uint32x4_private easysimd_vextq_u32_r_; \
      easysimd_vextq_u32_r_.values = EASYSIMD_SHUFFLE_VECTOR_(32, 16, easysimd_uint32x4_to_private(a).values, easysimd_uint32x4_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1)), \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 2)), HEDLEY_STATIC_CAST(int8_t, ((n) + 3))); \
      easysimd_uint32x4_from_private(easysimd_vextq_u32_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vextq_u32
  #define vextq_u32(a, b, n) easysimd_vextq_u32((a), (b), (n))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vextq_u64(easysimd_uint64x2_t a, easysimd_uint64x2_t b, const int n)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(n, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd_uint64x2_t r;
    EASYSIMD_CONSTIFY_2_(vextq_u64, r, (HEDLEY_UNREACHABLE(), a), n, a, b);
    return r;
  #else
    easysimd_uint64x2_private
      a_ = easysimd_uint64x2_to_private(a),
      b_ = easysimd_uint64x2_to_private(b),
      r_ = a_;
    const size_t n_ = HEDLEY_STATIC_CAST(size_t, n);
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      size_t src = i + n_;
      r_.values[i] = (src < (sizeof(r_.values) / sizeof(r_.values[0]))) ? a_.values[src] : b_.values[src & 1];
    }
    return easysimd_uint64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_BUG_GCC_SIZEOF_IMMEDIATE)
  #define easysimd_vextq_u64(a, b, n) easysimd_uint64x2_from_m128i(_mm_alignr_epi8(easysimd_uint64x2_to_m128i(b), easysimd_uint64x2_to_m128i(a), n * sizeof(uint64_t)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_VEXT_REV32)
  #define easysimd_vextq_u64(a, b, n) (__extension__ ({ \
      easysimd_uint64x2_private easysimd_vextq_u64_r_; \
      easysimd_vextq_u64_r_.values = EASYSIMD_SHUFFLE_VECTOR_(64, 16, easysimd_uint64x2_to_private(a).values, easysimd_uint64x2_to_private(b).values, \
        HEDLEY_STATIC_CAST(int8_t, ((n) + 0)), HEDLEY_STATIC_CAST(int8_t, ((n) + 1))); \
      easysimd_uint64x2_from_private(easysimd_vextq_u64_r_); \
    }))
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vextq_u64
  #define vextq_u64(a, b, n) easysimd_vextq_u64((a), (b), (n))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_EXT_H) */
