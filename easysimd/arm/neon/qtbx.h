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
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_ARM_NEON_QTBX_H)
#define EASYSIMD_ARM_NEON_QTBX_H

#include "reinterpret.h"
#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vqtbx1_u8(easysimd_uint8x8_t a, easysimd_uint8x16_t t, easysimd_uint8x8_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx1_u8(a, t, idx);
  #else
    easysimd_uint8x16_private t_ = easysimd_uint8x16_to_private(t);
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      idx_ = easysimd_uint8x8_to_private(idx);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      __m128i idx128 = _mm_set1_epi64(idx_.m64);
      idx128 = _mm_or_si128(idx128, _mm_cmpgt_epi8(idx128, _mm_set1_epi8(15)));
      __m128i r128 = _mm_shuffle_epi8(t_.m128i, idx128);
      r128 =  _mm_blendv_epi8(r128, _mm_set1_epi64(a_.m64), idx128);
      r_.m64 = _mm_movepi64_pi64(r128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (idx_.values[i] < 16) ? t_.values[idx_.values[i]] : a_.values[i];
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx1_u8
  #define vqtbx1_u8(a, t, idx) easysimd_vqtbx1_u8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vqtbx1_s8(easysimd_int8x8_t a, easysimd_int8x16_t t, easysimd_uint8x8_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx1_s8(a, t, idx);
  #else
    return easysimd_vreinterpret_s8_u8(easysimd_vqtbx1_u8(easysimd_vreinterpret_u8_s8(a), easysimd_vreinterpretq_u8_s8(t), idx));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx1_s8
  #define vqtbx1_s8(a, t, idx) easysimd_vqtbx1_s8((a), (t), (idx))
#endif

#if !defined(EASYSIMD_BUG_INTEL_857088)

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vqtbx2_u8(easysimd_uint8x8_t a, easysimd_uint8x16x2_t t, easysimd_uint8x8_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx2_u8(a, t, idx);
  #else
    easysimd_uint8x16_private t_[2] = { easysimd_uint8x16_to_private(t.val[0]), easysimd_uint8x16_to_private(t.val[1]) };
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      idx_ = easysimd_uint8x8_to_private(idx);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      __m128i idx128 = _mm_set1_epi64(idx_.m64);
      idx128 = _mm_or_si128(idx128, _mm_cmpgt_epi8(idx128, _mm_set1_epi8(31)));
      __m128i r128_0 = _mm_shuffle_epi8(t_[0].m128i, idx128);
      __m128i r128_1 = _mm_shuffle_epi8(t_[1].m128i, idx128);
      __m128i r128 = _mm_blendv_epi8(r128_0, r128_1, _mm_slli_epi32(idx128, 3));
      r128 =  _mm_blendv_epi8(r128, _mm_set1_epi64(a_.m64), idx128);
      r_.m64 = _mm_movepi64_pi64(r128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (idx_.values[i] < 32) ? t_[idx_.values[i] / 16].values[idx_.values[i] & 15] : a_.values[i];
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx2_u8
  #define vqtbx2_u8(a, t, idx) easysimd_vqtbx2_u8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vqtbx2_s8(easysimd_int8x8_t a, easysimd_int8x16x2_t t, easysimd_uint8x8_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx2_s8(a, t, idx);
  #else
    easysimd_uint8x16x2_t t_;
    easysimd_memcpy(&t_, &t, sizeof(t_));
    return easysimd_vreinterpret_s8_u8(easysimd_vqtbx2_u8(easysimd_vreinterpret_u8_s8(a), t_, idx));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx2_s8
  #define vqtbx2_s8(a, t, idx) easysimd_vqtbx2_s8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vqtbx3_u8(easysimd_uint8x8_t a, easysimd_uint8x16x3_t t, easysimd_uint8x8_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx3_u8(a, t, idx);
  #else
    easysimd_uint8x16_private t_[3] = { easysimd_uint8x16_to_private(t.val[0]), easysimd_uint8x16_to_private(t.val[1]), easysimd_uint8x16_to_private(t.val[2]) };
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      idx_ = easysimd_uint8x8_to_private(idx);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      __m128i idx128 = _mm_set1_epi64(idx_.m64);
      idx128 = _mm_or_si128(idx128, _mm_cmpgt_epi8(idx128, _mm_set1_epi8(47)));
      __m128i r128_0 = _mm_shuffle_epi8(t_[0].m128i, idx128);
      __m128i r128_1 = _mm_shuffle_epi8(t_[1].m128i, idx128);
      __m128i r128_01 = _mm_blendv_epi8(r128_0, r128_1, _mm_slli_epi32(idx128, 3));
      __m128i r128_2 = _mm_shuffle_epi8(t_[2].m128i, idx128);
      __m128i r128 = _mm_blendv_epi8(r128_01, r128_2, _mm_slli_epi32(idx128, 2));
      r128 =  _mm_blendv_epi8(r128, _mm_set1_epi64(a_.m64), idx128);
      r_.m64 = _mm_movepi64_pi64(r128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (idx_.values[i] < 48) ? t_[idx_.values[i] / 16].values[idx_.values[i] & 15] : a_.values[i];
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx3_u8
  #define vqtbx3_u8(a, t, idx) easysimd_vqtbx3_u8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vqtbx3_s8(easysimd_int8x8_t a, easysimd_int8x16x3_t t, easysimd_uint8x8_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx3_s8(a, t, idx);
  #else
    easysimd_uint8x16x3_t t_;
    easysimd_memcpy(&t_, &t, sizeof(t_));
    return easysimd_vreinterpret_s8_u8(easysimd_vqtbx3_u8(easysimd_vreinterpret_u8_s8(a), t_, idx));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx3_s8
  #define vqtbx3_s8(a, t, idx) easysimd_vqtbx3_s8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x8_t
easysimd_vqtbx4_u8(easysimd_uint8x8_t a, easysimd_uint8x16x4_t t, easysimd_uint8x8_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx4_u8(a, t, idx);
  #else
    easysimd_uint8x16_private t_[4] = { easysimd_uint8x16_to_private(t.val[0]), easysimd_uint8x16_to_private(t.val[1]), easysimd_uint8x16_to_private(t.val[2]), easysimd_uint8x16_to_private(t.val[3]) };
    easysimd_uint8x8_private
      r_,
      a_ = easysimd_uint8x8_to_private(a),
      idx_ = easysimd_uint8x8_to_private(idx);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      __m128i idx128 = _mm_set1_epi64(idx_.m64);
      idx128 = _mm_or_si128(idx128, _mm_cmpgt_epi8(idx128, _mm_set1_epi8(63)));
      __m128i idx128_shl3 = _mm_slli_epi32(idx128, 3);
      __m128i r128_0 = _mm_shuffle_epi8(t_[0].m128i, idx128);
      __m128i r128_1 = _mm_shuffle_epi8(t_[1].m128i, idx128);
      __m128i r128_01 = _mm_blendv_epi8(r128_0, r128_1, idx128_shl3);
      __m128i r128_2 = _mm_shuffle_epi8(t_[2].m128i, idx128);
      __m128i r128_3 = _mm_shuffle_epi8(t_[3].m128i, idx128);
      __m128i r128_23 = _mm_blendv_epi8(r128_2, r128_3, idx128_shl3);
      __m128i r128 = _mm_blendv_epi8(r128_01, r128_23, _mm_slli_epi32(idx128, 2));
      r128 =  _mm_blendv_epi8(r128, _mm_set1_epi64(a_.m64), idx128);
      r_.m64 = _mm_movepi64_pi64(r128);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (idx_.values[i] < 64) ? t_[idx_.values[i] / 16].values[idx_.values[i] & 15] : a_.values[i];
      }
    #endif

    return easysimd_uint8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx4_u8
  #define vqtbx4_u8(a, t, idx) easysimd_vqtbx4_u8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vqtbx4_s8(easysimd_int8x8_t a, easysimd_int8x16x4_t t, easysimd_uint8x8_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx4_s8(a, t, idx);
  #else
    easysimd_uint8x16x4_t t_;
    easysimd_memcpy(&t_, &t, sizeof(t_));
    return easysimd_vreinterpret_s8_u8(easysimd_vqtbx4_u8(easysimd_vreinterpret_u8_s8(a), t_, idx));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx4_s8
  #define vqtbx4_s8(a, t, idx) easysimd_vqtbx4_s8((a), (t), (idx))
#endif

#endif /* !defined(EASYSIMD_BUG_INTEL_857088) */

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vqtbx1q_u8(easysimd_uint8x16_t a, easysimd_uint8x16_t t, easysimd_uint8x16_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx1q_u8(a, t, idx);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      t_ = easysimd_uint8x16_to_private(t),
      idx_ = easysimd_uint8x16_to_private(idx);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      idx_.m128i = _mm_or_si128(idx_.m128i, _mm_cmpgt_epi8(idx_.m128i, _mm_set1_epi8(15)));
      r_.m128i =  _mm_blendv_epi8(_mm_shuffle_epi8(t_.m128i, idx_.m128i), a_.m128i, idx_.m128i);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (idx_.values[i] < 16) ? t_.values[idx_.values[i]] : a_.values[i];
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx1q_u8
  #define vqtbx1q_u8(a, t, idx) easysimd_vqtbx1q_u8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vqtbx1q_s8(easysimd_int8x16_t a, easysimd_int8x16_t t, easysimd_uint8x16_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx1q_s8(a, t, idx);
  #else
    return easysimd_vreinterpretq_s8_u8(easysimd_vqtbx1q_u8(easysimd_vreinterpretq_u8_s8(a), easysimd_vreinterpretq_u8_s8(t), idx));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx1q_s8
  #define vqtbx1q_s8(a, t, idx) easysimd_vqtbx1q_s8((a), (t), (idx))
#endif

#if !defined(EASYSIMD_BUG_INTEL_857088)

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vqtbx2q_u8(easysimd_uint8x16_t a, easysimd_uint8x16x2_t t, easysimd_uint8x16_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx2q_u8(a, t, idx);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      t_[2] = { easysimd_uint8x16_to_private(t.val[0]), easysimd_uint8x16_to_private(t.val[1]) },
      idx_ = easysimd_uint8x16_to_private(idx);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      idx_.m128i = _mm_or_si128(idx_.m128i, _mm_cmpgt_epi8(idx_.m128i, _mm_set1_epi8(31)));
      __m128i r_0 = _mm_shuffle_epi8(t_[0].m128i, idx_.m128i);
      __m128i r_1 = _mm_shuffle_epi8(t_[1].m128i, idx_.m128i);
      __m128i r =  _mm_blendv_epi8(r_0, r_1, _mm_slli_epi32(idx_.m128i, 3));
      r_.m128i = _mm_blendv_epi8(r, a_.m128i, idx_.m128i);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (idx_.values[i] < 32) ? t_[idx_.values[i] / 16].values[idx_.values[i] & 15] : a_.values[i];
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx2q_u8
  #define vqtbx2q_u8(a, t, idx) easysimd_vqtbx2q_u8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vqtbx2q_s8(easysimd_int8x16_t a, easysimd_int8x16x2_t t, easysimd_uint8x16_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx2q_s8(a, t, idx);
  #else
    easysimd_uint8x16x2_t t_;
    easysimd_memcpy(&t_, &t, sizeof(t_));
    return easysimd_vreinterpretq_s8_u8(easysimd_vqtbx2q_u8(easysimd_vreinterpretq_u8_s8(a), t_, idx));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx2q_s8
  #define vqtbx2q_s8(a, t, idx) easysimd_vqtbx2q_s8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vqtbx3q_u8(easysimd_uint8x16_t a, easysimd_uint8x16x3_t t, easysimd_uint8x16_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx3q_u8(a, t, idx);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      t_[3] = { easysimd_uint8x16_to_private(t.val[0]), easysimd_uint8x16_to_private(t.val[1]), easysimd_uint8x16_to_private(t.val[2]) },
      idx_ = easysimd_uint8x16_to_private(idx);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      idx_.m128i = _mm_or_si128(idx_.m128i, _mm_cmpgt_epi8(idx_.m128i, _mm_set1_epi8(47)));
      __m128i r_0 = _mm_shuffle_epi8(t_[0].m128i, idx_.m128i);
      __m128i r_1 = _mm_shuffle_epi8(t_[1].m128i, idx_.m128i);
      __m128i r_01 = _mm_blendv_epi8(r_0, r_1, _mm_slli_epi32(idx_.m128i, 3));
      __m128i r_2 = _mm_shuffle_epi8(t_[2].m128i, idx_.m128i);
      __m128i r = _mm_blendv_epi8(r_01, r_2, _mm_slli_epi32(idx_.m128i, 2));
      r_.m128i = _mm_blendv_epi8(r, a_.m128i, idx_.m128i);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (idx_.values[i] < 48) ? t_[idx_.values[i] / 16].values[idx_.values[i] & 15] : a_.values[i];
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx3q_u8
  #define vqtbx3q_u8(a, t, idx) easysimd_vqtbx3q_u8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vqtbx3q_s8(easysimd_int8x16_t a, easysimd_int8x16x3_t t, easysimd_uint8x16_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx3q_s8(a, t, idx);
  #else
    easysimd_uint8x16x3_t t_;
    easysimd_memcpy(&t_, &t, sizeof(t_));
    return easysimd_vreinterpretq_s8_u8(easysimd_vqtbx3q_u8(easysimd_vreinterpretq_u8_s8(a), t_, idx));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx3q_s8
  #define vqtbx3q_s8(a, t, idx) easysimd_vqtbx3q_s8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint8x16_t
easysimd_vqtbx4q_u8(easysimd_uint8x16_t a, easysimd_uint8x16x4_t t, easysimd_uint8x16_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx4q_u8(a, t, idx);
  #else
    easysimd_uint8x16_private
      r_,
      a_ = easysimd_uint8x16_to_private(a),
      t_[4] = { easysimd_uint8x16_to_private(t.val[0]), easysimd_uint8x16_to_private(t.val[1]), easysimd_uint8x16_to_private(t.val[2]), easysimd_uint8x16_to_private(t.val[3]) },
      idx_ = easysimd_uint8x16_to_private(idx);

    #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
      idx_.m128i = _mm_or_si128(idx_.m128i, _mm_cmpgt_epi8(idx_.m128i, _mm_set1_epi8(63)));
      __m128i idx_shl3 = _mm_slli_epi32(idx_.m128i, 3);
      __m128i r_0 = _mm_shuffle_epi8(t_[0].m128i, idx_.m128i);
      __m128i r_1 = _mm_shuffle_epi8(t_[1].m128i, idx_.m128i);
      __m128i r_01 = _mm_blendv_epi8(r_0, r_1, idx_shl3);
      __m128i r_2 = _mm_shuffle_epi8(t_[2].m128i, idx_.m128i);
      __m128i r_3 = _mm_shuffle_epi8(t_[3].m128i, idx_.m128i);
      __m128i r_23 = _mm_blendv_epi8(r_2, r_3, idx_shl3);
      __m128i r = _mm_blendv_epi8(r_01, r_23, _mm_slli_epi32(idx_.m128i, 2));
      r_.m128i = _mm_blendv_epi8(r, a_.m128i, idx_.m128i);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = (idx_.values[i] < 64) ? t_[idx_.values[i] / 16].values[idx_.values[i] & 15] : a_.values[i];
      }
    #endif

    return easysimd_uint8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx4q_u8
  #define vqtbx4q_u8(a, t, idx) easysimd_vqtbx4q_u8((a), (t), (idx))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vqtbx4q_s8(easysimd_int8x16_t a, easysimd_int8x16x4_t t, easysimd_uint8x16_t idx) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqtbx4q_s8(a, t, idx);
  #else
    easysimd_uint8x16x4_t t_;
    easysimd_memcpy(&t_, &t, sizeof(t_));
    return easysimd_vreinterpretq_s8_u8(easysimd_vqtbx4q_u8(easysimd_vreinterpretq_u8_s8(a), t_, idx));
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqtbx4q_s8
  #define vqtbx4q_s8(a, t, idx) easysimd_vqtbx4q_s8((a), (t), (idx))
#endif

#endif /* !defined(EASYSIMD_BUG_INTEL_857088) */

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QTBX_H) */
