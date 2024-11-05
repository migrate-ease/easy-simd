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

#if !defined(EASYSIMD_ARM_NEON_QABS_H)
#define EASYSIMD_ARM_NEON_QABS_H

#include "types.h"

#include "abs.h"
#include "add.h"
#include "bsl.h"
#include "dup_n.h"
#include "mvn.h"
#include "reinterpret.h"
#include "shr_n.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_vqabsb_s8(int8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqabsb_s8(a);
  #else
    return a == INT8_MIN ? INT8_MAX : (a < 0 ? -a : a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqabsb_s8
  #define vqabsb_s8(a) easysimd_vqabsb_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vqabsh_s16(int16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqabsh_s16(a);
  #else
    return a == INT16_MIN ? INT16_MAX : (a < 0 ? -a : a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqabsh_s16
  #define vqabsh_s16(a) easysimd_vqabsh_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vqabss_s32(int32_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqabss_s32(a);
  #else
    return a == INT32_MIN ? INT32_MAX : (a < 0 ? -a : a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqabss_s32
  #define vqabss_s32(a) easysimd_vqabss_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vqabsd_s64(int64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqabsd_s64(a);
  #else
    return a == INT64_MIN ? INT64_MAX : (a < 0 ? -a : a);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqabsd_s64
  #define vqabsd_s64(a) easysimd_vqabsd_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vqabs_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqabs_s8(a);
  #else
    easysimd_int8x8_t tmp = easysimd_vabs_s8(a);
    return easysimd_vadd_s8(tmp, easysimd_vshr_n_s8(tmp, 7));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqabs_s8
  #define vqabs_s8(a) easysimd_vqabs_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vqabs_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqabs_s16(a);
  #else
    easysimd_int16x4_t tmp = easysimd_vabs_s16(a);
    return easysimd_vadd_s16(tmp, easysimd_vshr_n_s16(tmp, 15));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqabs_s16
  #define vqabs_s16(a) easysimd_vqabs_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vqabs_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqabs_s32(a);
  #else
    easysimd_int32x2_t tmp = easysimd_vabs_s32(a);
    return easysimd_vadd_s32(tmp, easysimd_vshr_n_s32(tmp, 31));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqabs_s32
  #define vqabs_s32(a) easysimd_vqabs_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vqabs_s64(easysimd_int64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqabs_s64(a);
  #else
    easysimd_int64x1_t tmp = easysimd_vabs_s64(a);
    return easysimd_vadd_s64(tmp, easysimd_vshr_n_s64(tmp, 63));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqabs_s64
  #define vqabs_s64(a) easysimd_vqabs_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vqabsq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqabsq_s8(a);
  #elif defined(EASYSIMD_X86_SSE4_1_NATIVE)
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(easysimd_vabsq_s8(a));

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.m128i = _mm_min_epu8(a_.m128i, _mm_set1_epi8(INT8_MAX));
    #else
      r_.m128i =
        _mm_add_epi8(
          a_.m128i,
          _mm_cmpgt_epi8(_mm_setzero_si128(), a_.m128i)
        );
    #endif

    return easysimd_int8x16_from_private(r_);
  #else
    easysimd_int8x16_t tmp = easysimd_vabsq_s8(a);
    return
      easysimd_vbslq_s8(
        easysimd_vreinterpretq_u8_s8(easysimd_vshrq_n_s8(tmp, 7)),
        easysimd_vmvnq_s8(tmp),
        tmp
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqabsq_s8
  #define vqabsq_s8(a) easysimd_vqabsq_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vqabsq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqabsq_s16(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(easysimd_vabsq_s16(a));

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.m128i = _mm_min_epu16(a_.m128i, _mm_set1_epi16(INT16_MAX));
    #else
      r_.m128i =
        _mm_add_epi16(
          a_.m128i,
          _mm_srai_epi16(a_.m128i, 15)
        );
    #endif

    return easysimd_int16x8_from_private(r_);
  #else
    easysimd_int16x8_t tmp = easysimd_vabsq_s16(a);
    return
      easysimd_vbslq_s16(
        easysimd_vreinterpretq_u16_s16(easysimd_vshrq_n_s16(tmp, 15)),
        easysimd_vmvnq_s16(tmp),
        tmp
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqabsq_s16
  #define vqabsq_s16(a) easysimd_vqabsq_s16(a)
#endif

#if (!defined(__clang__))
EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vqabsq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqabsq_s32(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(easysimd_vabsq_s32(a));

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
      r_.m128i = _mm_min_epu32(a_.m128i, _mm_set1_epi32(INT32_MAX));
    #else
      r_.m128i =
        _mm_add_epi32(
          a_.m128i,
          _mm_srai_epi32(a_.m128i, 31)
        );
    #endif

    return easysimd_int32x4_from_private(r_);
  #else
    easysimd_int32x4_t tmp = easysimd_vabsq_s32(a);
    return
      easysimd_vbslq_s32(
        easysimd_vreinterpretq_u32_s32(easysimd_vshrq_n_s32(tmp, 31)),
        easysimd_vmvnq_s32(tmp),
        tmp
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqabsq_s32
  #define vqabsq_s32(a) easysimd_vqabsq_s32(a)
#endif
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vqabsq_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqabsq_s64(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_int64x2_private
      r_,
      a_ = easysimd_int64x2_to_private(easysimd_vabsq_s64(a));

    #if defined(EASYSIMD_X86_SSE4_2_NATIVE)
      r_.m128i =
        _mm_add_epi64(
          a_.m128i,
          _mm_cmpgt_epi64(_mm_setzero_si128(), a_.m128i)
        );
    #else
      r_.m128i =
        _mm_add_epi64(
          a_.m128i,
          _mm_shuffle_epi32(
            _mm_srai_epi32(a_.m128i, 31),
            _MM_SHUFFLE(3, 3, 1, 1)
          )
        );
    #endif

    return easysimd_int64x2_from_private(r_);
  #else
    easysimd_int64x2_t tmp = easysimd_vabsq_s64(a);
    return
      easysimd_vbslq_s64(
        easysimd_vreinterpretq_u64_s64(easysimd_vshrq_n_s64(tmp, 63)),
        easysimd_vreinterpretq_s64_s32(easysimd_vmvnq_s32(easysimd_vreinterpretq_s32_s64(tmp))),
        tmp
      );
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqabsq_s64
  #define vqabsq_s64(a) easysimd_vqabsq_s64(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QABS_H) */
