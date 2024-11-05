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
 *   2021      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_ARM_NEON_LD1_DUP_H)
#define EASYSIMD_ARM_NEON_LD1_DUP_H

#include "dup_n.h"
#include "reinterpret.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vld1_dup_f32(easysimd_float32 const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1_dup_f32(ptr);
  #else
    return easysimd_vdup_n_f32(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_f32
  #define vld1_dup_f32(a) easysimd_vld1_dup_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vld1_dup_f64(easysimd_float64 const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld1_dup_f64(ptr);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return easysimd_vreinterpret_f64_s64(vld1_dup_s64(HEDLEY_REINTERPRET_CAST(int64_t const*, ptr)));
  #else
    return easysimd_vdup_n_f64(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_f64
  #define vld1_dup_f64(a) easysimd_vld1_dup_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vld1_dup_s8(int8_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1_dup_s8(ptr);
  #else
    return easysimd_vdup_n_s8(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_s8
  #define vld1_dup_s8(a) easysimd_vld1_dup_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vld1_dup_s16(int16_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1_dup_s16(ptr);
  #else
    return easysimd_vdup_n_s16(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_s16
  #define vld1_dup_s16(a) easysimd_vld1_dup_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vld1_dup_s32(int32_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1_dup_s32(ptr);
  #else
    return easysimd_vdup_n_s32(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_s32
  #define vld1_dup_s32(a) easysimd_vld1_dup_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vld1_dup_s64(int64_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1_dup_s64(ptr);
  #else
    return easysimd_vdup_n_s64(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_s64
  #define vld1_dup_s64(a) easysimd_vld1_dup_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vld1_dup_u8(uint8_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1_dup_u8(ptr);
  #else
    return easysimd_vdup_n_u8(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_u8
  #define vld1_dup_u8(a) easysimd_vld1_dup_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vld1_dup_u16(uint16_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1_dup_u16(ptr);
  #else
    return easysimd_vdup_n_u16(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_u16
  #define vld1_dup_u16(a) easysimd_vld1_dup_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vld1_dup_u32(uint32_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1_dup_u32(ptr);
  #else
    return easysimd_vdup_n_u32(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_u32
  #define vld1_dup_u32(a) easysimd_vld1_dup_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x1_t
easysimd_vld1_dup_u64(uint64_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1_dup_u64(ptr);
  #else
    return easysimd_vdup_n_u64(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1_dup_u64
  #define vld1_dup_u64(a) easysimd_vld1_dup_u64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vld1q_dup_f32(easysimd_float32 const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1q_dup_f32(ptr);
  #elif \
      defined(EASYSIMD_X86_SSE_NATIVE)
    easysimd_float32x4_private r_;
    r_.m128 = _mm_load_ps1(ptr);

    return easysimd_float32x4_from_private(r_);
  #else
    return easysimd_vdupq_n_f32(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_f32
  #define vld1q_dup_f32(a) easysimd_vld1q_dup_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vld1q_dup_f64(easysimd_float64 const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vld1q_dup_f64(ptr);
  #else
    return easysimd_vdupq_n_f64(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_f64
  #define vld1q_dup_f64(a) easysimd_vld1q_dup_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vld1q_dup_s8(int8_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1q_dup_s8(ptr);
  #else
    return easysimd_vdupq_n_s8(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_s8
  #define vld1q_dup_s8(a) easysimd_vld1q_dup_s8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vld1q_dup_s16(int16_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1q_dup_s16(ptr);
  #else
    return easysimd_vdupq_n_s16(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_s16
  #define vld1q_dup_s16(a) easysimd_vld1q_dup_s16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vld1q_dup_s32(int32_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1q_dup_s32(ptr);
  #elif \
      defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_int32x4_private r_;

    r_.m128i = _mm_castps_si128(_mm_load_ps1(HEDLEY_REINTERPRET_CAST(float const *, ptr)));

    return easysimd_int32x4_from_private(r_);
  #else
    return easysimd_vdupq_n_s32(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_s32
  #define vld1q_dup_s32(a) easysimd_vld1q_dup_s32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vld1q_dup_s64(int64_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1q_dup_s64(ptr);
  #elif \
      defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_int64x2_private r_;

    r_.m128i = _mm_set1_epi64x(*ptr);

    return easysimd_int64x2_from_private(r_);
  #else
    return easysimd_vdupq_n_s64(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_s64
  #define vld1q_dup_s64(a) easysimd_vld1q_dup_s64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vld1q_dup_u8(uint8_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1q_dup_u8(ptr);
  #else
    return easysimd_vdupq_n_u8(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_u8
  #define vld1q_dup_u8(a) easysimd_vld1q_dup_u8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vld1q_dup_u16(uint16_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1q_dup_u16(ptr);
  #else
    return easysimd_vdupq_n_u16(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_u16
  #define vld1q_dup_u16(a) easysimd_vld1q_dup_u16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vld1q_dup_u32(uint32_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1q_dup_u32(ptr);
  #elif \
      defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_uint32x4_private r_;

    r_.m128i = _mm_castps_si128(_mm_load_ps1(HEDLEY_REINTERPRET_CAST(float const *, ptr)));

    return easysimd_uint32x4_from_private(r_);
  #else
    return easysimd_vdupq_n_u32(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_u32
  #define vld1q_dup_u32(a) easysimd_vld1q_dup_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint64x2_t
easysimd_vld1q_dup_u64(uint64_t const * ptr) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vld1q_dup_u64(ptr);
  #elif \
      defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_uint64x2_private r_;

    r_.m128i = _mm_set1_epi64x(*HEDLEY_REINTERPRET_CAST(int64_t const *, ptr));

    return easysimd_uint64x2_from_private(r_);
  #else
    return easysimd_vdupq_n_u64(*ptr);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vld1q_dup_u64
  #define vld1q_dup_u64(a) easysimd_vld1q_dup_u64((a))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_LD1_DUP_H) */
