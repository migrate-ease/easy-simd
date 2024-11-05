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

#if !defined(EASYSIMD_ARM_NEON_ABD_H)
#define EASYSIMD_ARM_NEON_ABD_H

#include "abs.h"
#include "subl.h"
#include "movn.h"
#include "movl.h"
#include "reinterpret.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vabds_f32(easysimd_float32_t a, easysimd_float32_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vabds_f32(a, b);
  #else
    easysimd_float32_t r = a - b;
    return r < 0 ? -r : r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vabds_f32
  #define vabds_f32(a, b) easysimd_vabds_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vabdd_f64(easysimd_float64_t a, easysimd_float64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vabdd_f64(a, b);
  #else
    easysimd_float64_t r = a - b;
    return r < 0 ? -r : r;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vabdd_f64
  #define vabdd_f64(a, b) easysimd_vabdd_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vabd_f32(easysimd_float32x2_t a, easysimd_float32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabd_f32(a, b);
  #else
    return easysimd_vabs_f32(easysimd_vsub_f32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabd_f32
  #define vabd_f32(a, b) easysimd_vabd_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vabd_f64(easysimd_float64x1_t a, easysimd_float64x1_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vabd_f64(a, b);
  #else
    return easysimd_vabs_f64(easysimd_vsub_f64(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vabd_f64
  #define vabd_f64(a, b) easysimd_vabd_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vabd_s8(easysimd_int8x8_t a, easysimd_int8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabd_s8(a, b);
  #elif defined(EASYSIMD_X86_MMX_NATIVE)
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a),
      b_ = easysimd_int8x8_to_private(b);

    const __m64 m = _mm_cmpgt_pi8(b_.m64, a_.m64);
    r_.m64 =
      _mm_xor_si64(
        _mm_add_pi8(
          _mm_sub_pi8(a_.m64, b_.m64),
          m
        ),
        m
      );

    return easysimd_int8x8_from_private(r_);
  #else
    return easysimd_vmovn_s16(easysimd_vabsq_s16(easysimd_vsubl_s8(a, b)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabd_s8
  #define vabd_s8(a, b) easysimd_vabd_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vabd_s16(easysimd_int16x4_t a, easysimd_int16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabd_s16(a, b);
  #elif defined(EASYSIMD_X86_MMX_NATIVE) && defined(EASYSIMD_X86_SSE_NATIVE)
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a),
      b_ = easysimd_int16x4_to_private(b);

    r_.m64 = _mm_sub_pi16(_mm_max_pi16(a_.m64, b_.m64), _mm_min_pi16(a_.m64, b_.m64));

    return easysimd_int16x4_from_private(r_);
  #else
    return easysimd_vmovn_s32(easysimd_vabsq_s32(easysimd_vsubl_s16(a, b)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabd_s16
  #define vabd_s16(a, b) easysimd_vabd_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vabd_s32(easysimd_int32x2_t a, easysimd_int32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabd_s32(a, b);
  #else
    return easysimd_vmovn_s64(easysimd_vabsq_s64(easysimd_vsubl_s32(a, b)));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabd_s32
  #define vabd_s32(a, b) easysimd_vabd_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vabd_u8(easysimd_uint8x8_t a, easysimd_uint8x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabd_u8(a, b);
  #else
    return easysimd_vmovn_u16(
      easysimd_vreinterpretq_u16_s16(
        easysimd_vabsq_s16(
          easysimd_vsubq_s16(
            easysimd_vreinterpretq_s16_u16(easysimd_vmovl_u8(a)),
            easysimd_vreinterpretq_s16_u16(easysimd_vmovl_u8(b))))));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabd_u8
  #define vabd_u8(a, b) easysimd_vabd_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x4_t
easysimd_vabd_u16(easysimd_uint16x4_t a, easysimd_uint16x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabd_u16(a, b);
  #else
    return easysimd_vmovn_u32(
      easysimd_vreinterpretq_u32_s32(
        easysimd_vabsq_s32(
          easysimd_vsubq_s32(
            easysimd_vreinterpretq_s32_u32(easysimd_vmovl_u16(a)),
            easysimd_vreinterpretq_s32_u32(easysimd_vmovl_u16(b))))));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabd_u16
  #define vabd_u16(a, b) easysimd_vabd_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vabd_u32(easysimd_uint32x2_t a, easysimd_uint32x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabd_u32(a, b);
  #else
    return easysimd_vmovn_u64(
      easysimd_vreinterpretq_u64_s64(
        easysimd_vabsq_s64(
          easysimd_vsubq_s64(
            easysimd_vreinterpretq_s64_u64(easysimd_vmovl_u32(a)),
            easysimd_vreinterpretq_s64_u64(easysimd_vmovl_u32(b))))));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabd_u32
  #define vabd_u32(a, b) easysimd_vabd_u32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vabdq_f32(easysimd_float32x4_t a, easysimd_float32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdq_f32(a, b);
  #else
    return easysimd_vabsq_f32(easysimd_vsubq_f32(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdq_f32
  #define vabdq_f32(a, b) easysimd_vabdq_f32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vabdq_f64(easysimd_float64x2_t a, easysimd_float64x2_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vabdq_f64(a, b);
  #else
    return easysimd_vabsq_f64(easysimd_vsubq_f64(a, b));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vabdq_f64
  #define vabdq_f64(a, b) easysimd_vabdq_f64((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vabdq_s8(easysimd_int8x16_t a, easysimd_int8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdq_s8(a, b);
  #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
    return vec_max(a, b) - vec_min(a, b);
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a),
      b_ = easysimd_int8x16_to_private(b);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.m128i = _mm_sub_epi8(_mm_max_epi8(a_.m128i, b_.m128i), _mm_min_epi8(a_.m128i, b_.m128i));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      const __m128i m = _mm_cmpgt_epi8(b_.m128i, a_.m128i);
      r_.m128i =
        _mm_xor_si128(
          _mm_add_epi8(
            _mm_sub_epi8(a_.m128i, b_.m128i),
            m
          ),
          m
        );
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        int16_t tmp = HEDLEY_STATIC_CAST(int16_t, a_.values[i]) - HEDLEY_STATIC_CAST(int16_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(int8_t, tmp < 0 ? -tmp : tmp);
      }
    #endif

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdq_s8
  #define vabdq_s8(a, b) easysimd_vabdq_s8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vabdq_s16(easysimd_int16x8_t a, easysimd_int16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdq_s16(a, b);
  #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
    return vec_max(a, b) - vec_min(a, b);
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a),
      b_ = easysimd_int16x8_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      /* https://github.com/simd-everywhere/simde/issues/855#issuecomment-881658604 */
      r_.m128i = _mm_sub_epi16(_mm_max_epi16(a_.m128i, b_.m128i), _mm_min_epi16(a_.m128i, b_.m128i));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] =
          (a_.values[i] < b_.values[i]) ?
            (b_.values[i] - a_.values[i]) :
            (a_.values[i] - b_.values[i]);
      }

    #endif
    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdq_s16
  #define vabdq_s16(a, b) easysimd_vabdq_s16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vabdq_s32(easysimd_int32x4_t a, easysimd_int32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdq_s32(a, b);
  #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
    return vec_max(a, b) - vec_min(a, b);
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a),
      b_ = easysimd_int32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.m128i = _mm_sub_epi32(_mm_max_epi32(a_.m128i, b_.m128i), _mm_min_epi32(a_.m128i, b_.m128i));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      const __m128i m = _mm_cmpgt_epi32(b_.m128i, a_.m128i);
      r_.m128i =
        _mm_xor_si128(
          _mm_add_epi32(
            _mm_sub_epi32(a_.m128i, b_.m128i),
            m
          ),
          m
        );
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        int64_t tmp = HEDLEY_STATIC_CAST(int64_t, a_.values[i]) - HEDLEY_STATIC_CAST(int64_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(int32_t, tmp < 0 ? -tmp : tmp);
      }
    #endif

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdq_s32
  #define vabdq_s32(a, b) easysimd_vabdq_s32((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vabdq_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdq_u8(a, b);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      b_ = easysimd_uint8x16_to_private(b);

    #if defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.m128i = _mm_sub_epi8(_mm_max_epu8(a_.m128i, b_.m128i), _mm_min_epu8(a_.m128i, b_.m128i));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        int16_t tmp = HEDLEY_STATIC_CAST(int16_t, a_.values[i]) - HEDLEY_STATIC_CAST(int16_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(uint8_t, tmp < 0 ? -tmp : tmp);
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdq_u8
  #define vabdq_u8(a, b) easysimd_vabdq_u8((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint16x8_t
easysimd_vabdq_u16(easysimd_uint16x8_t a, easysimd_uint16x8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdq_u16(a, b);
  #else
    easysimd_uint16x8_private
      r_,
      a_ = easysimd_uint16x8_to_private(a),
      b_ = easysimd_uint16x8_to_private(b);

    #if defined(EASYSIMD_X86_SSE4_2_NATIVE)
      r_.m128i = _mm_sub_epi16(_mm_max_epu16(a_.m128i, b_.m128i), _mm_min_epu16(a_.m128i, b_.m128i));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        int32_t tmp = HEDLEY_STATIC_CAST(int32_t, a_.values[i]) - HEDLEY_STATIC_CAST(int32_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(uint16_t, tmp < 0 ? -tmp : tmp);
      }
    #endif

    return easysimd_uint16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdq_u16
  #define vabdq_u16(a, b) easysimd_vabdq_u16((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vabdq_u32(easysimd_uint32x4_t a, easysimd_uint32x4_t b) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vabdq_u32(a, b);
  #else
    easysimd_uint32x4_private
      r_,
      a_ = easysimd_uint32x4_to_private(a),
      b_ = easysimd_uint32x4_to_private(b);

    #if defined(EASYSIMD_X86_SSE4_2_NATIVE)
      r_.m128i = _mm_sub_epi32(_mm_max_epu32(a_.m128i, b_.m128i), _mm_min_epu32(a_.m128i, b_.m128i));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        int64_t tmp = HEDLEY_STATIC_CAST(int64_t, a_.values[i]) - HEDLEY_STATIC_CAST(int64_t, b_.values[i]);
        r_.values[i] = HEDLEY_STATIC_CAST(uint32_t, tmp < 0 ? -tmp : tmp);
      }
    #endif

    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vabdq_u32
  #define vabdq_u32(a, b) easysimd_vabdq_u32((a), (b))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_ABD_H) */
