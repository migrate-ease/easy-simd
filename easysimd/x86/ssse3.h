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
 *   2017-2020 Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_X86_SSSE3_H)
#define EASYSIMD_X86_SSSE3_H
#include "sse2.h"
#include "sse3.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_abs_epi8 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_abs_epi8(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_min_epu8(a, _mm_sub_epi8(_mm_setzero_si128(), a));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svabs_s8_z(svptrue_b8(), a.sve_i8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i8 = vabsq_s8(a_.neon_i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.u8[i] = HEDLEY_STATIC_CAST(uint8_t, (a_.i8[i] < 0) ? (- a_.i8[i]) : a_.i8[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_abs_epi8(a) easysimd_mm_abs_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_abs_epi16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_abs_epi16(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_max_epi16(a, _mm_sub_epi16(_mm_setzero_si128(), a));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svabs_s16_z(svptrue_b16(), a.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i16 = vabsq_s16(a_.neon_i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (a_.i16[i] < 0) ? (- a_.i16[i]) : a_.i16[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_abs_epi16(a) easysimd_mm_abs_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_abs_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_abs_epi32(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    const __m128i m = _mm_cmpgt_epi32(_mm_setzero_si128(), a);
    return _mm_sub_epi32(_mm_xor_si128(a, m), m);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svabs_s32_z(svptrue_b32(), a.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vabsq_s32(a_.neon_i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        #if defined(_MSC_VER)
          HEDLEY_DIAGNOSTIC_PUSH
          #pragma warning(disable:4146)
        #endif
        r_.u32[i] = (a_.i32[i] < 0) ? (- HEDLEY_STATIC_CAST(uint32_t, a_.i32[i])) : HEDLEY_STATIC_CAST(uint32_t, a_.i32[i]);
        #if defined(_MSC_VER)
          HEDLEY_DIAGNOSTIC_POP
        #endif
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_abs_epi32(a) easysimd_mm_abs_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_abs_pi8 (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_abs_pi8(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_i8 = vabs_s8(a.neon_i8);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.u8[i] = HEDLEY_STATIC_CAST(uint8_t, (a_.i8[i] < 0) ? (- a_.i8[i]) : a_.i8[i]);
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_abs_pi8(a) easysimd_mm_abs_pi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_abs_pi16 (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_abs_pi16(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_i16 = vabs_s16(a.neon_i16);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (a_.i16[i] < 0) ? (- a_.i16[i]) : a_.i16[i]);
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_abs_pi16(a) easysimd_mm_abs_pi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_abs_pi32 (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_abs_pi32(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_i32 = vabs_s32(a.neon_i32);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (a_.i32[i] < 0) ? (- a_.i32[i]) : a_.i32[i]);
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_abs_pi32(a) easysimd_mm_abs_pi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_alignr_epi8 (easysimd__m128i a, easysimd__m128i b, int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  if (HEDLEY_UNLIKELY(count > 31))
    return easysimd_mm_setzero_si128();

  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
    const int srcpos = count + HEDLEY_STATIC_CAST(int, i);
    if (srcpos > 31) {
      r_.i8[i] = 0;
    } else if (srcpos > 15) {
      r_.i8[i] = a_.i8[(srcpos) & 15];
    } else {
      r_.i8[i] = b_.i8[srcpos];
    }
  }

  return easysimd__m128i_from_private(r_);
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE)
  #define easysimd_mm_alignr_epi8(a, b, count) _mm_alignr_epi8(a, b, count)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  #define easysimd_mm_alignr_epi8(a, b, count) ({ \
    easysimd__m128i r; \
    r.sve_i8 = ((count) > 31) ? svdup_n_s8(0) \
      : ( \
        ((count) > 15) \
          ? (svext_s8((a).sve_i8, svdup_n_s8(0), (count) & 15)) \
          : (svext_s8((b).sve_i8, (a).sve_i8, ((count) & 15)))); \
    r; \
  })
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm_alignr_epi8(a, b, count) \
    easysimd__m128i_from_neon_i8( \
      ((count) > 31) \
        ? (vdupq_n_s8(0)) \
        : ( \
          ((count) > 15) \
            ? (vextq_s8(a.neon_i8, vdupq_n_s8(0), ((count) & 15))) \
            : (vextq_s8(b.neon_i8, a.neon_i8, ((count) & 15)))))
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_mm_alignr_epi8(a, b, count) \
    ( \
      ((count) > 31) \
        ? easysimd__m128i_from_neon_i8(vdupq_n_s8(0)) \
        : ( \
          ((count) > 15) \
            ? (easysimd__m128i_from_neon_i8(vextq_s8(easysimd__m128i_to_neon_i8(a), vdupq_n_s8(0), (count) & 15))) \
            : (easysimd__m128i_from_neon_i8(vextq_s8(easysimd__m128i_to_neon_i8(b), easysimd__m128i_to_neon_i8(a), ((count) & 15))))))
#endif
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
  #define _mm_alignr_epi8(a, b, count) easysimd_mm_alignr_epi8(a, b, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_alignr_pi8 (easysimd__m64 a, easysimd__m64 b, const int count)
    EASYSIMD_REQUIRE_CONSTANT(count) {
  easysimd__m64_private
    r_,
    a_ = easysimd__m64_to_private(a),
    b_ = easysimd__m64_to_private(b);

  if (HEDLEY_UNLIKELY(count > 15))
    return easysimd_mm_setzero_si64();

  for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
    const int srcpos = count + HEDLEY_STATIC_CAST(int, i);
    if (srcpos > 15) {
      r_.i8[i] = 0;
    } else if (srcpos > 7) {
      r_.i8[i] = a_.i8[(srcpos) & 7];
    } else {
      r_.i8[i] = b_.i8[srcpos];
    }
  }

  return easysimd__m64_from_private(r_);
}
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
#  define easysimd_mm_alignr_pi8(a, b, count) _mm_alignr_pi8(a, b, count)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_mm_alignr_pi8(a, b, count) \
    ( \
      ((count) > 15) \
        ? easysimd__m64_from_neon_i8(vdup_n_s8(0)) \
        : ( \
          ((count) > 7) \
            ? (easysimd__m64_from_neon_i8(vext_s8(easysimd__m64_to_neon_i8(a), vdup_n_s8(0), (count) & 7))) \
            : (easysimd__m64_from_neon_i8(vext_s8(easysimd__m64_to_neon_i8(b), easysimd__m64_to_neon_i8(a), ((count) & 7))))))
#endif
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_alignr_pi8(a, b, count) easysimd_mm_alignr_pi8(a, b, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shuffle_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_shuffle_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svtbl_u8(a.sve_u8,
                        svand_u8_z(svptrue_b8(), b.sve_u8, svdup_n_u8(0x8F)));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_u8 = vqtbl1q_u8(a.neon_u8, vandq_u8(b.neon_u8, vdupq_n_u8(0x8F)));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      /* Mask out the bits we're not interested in.  vtbl will result in 0
       * for any values outside of [0, 15], so if the high bit is set it
       * will return 0, just like in SSSE3. */
      b_.neon_i8 = vandq_s8(b_.neon_i8, vdupq_n_s8(HEDLEY_STATIC_CAST(int8_t, (1 << 7) | 15)));

      /* Convert a from an int8x16_t to an int8x8x2_t */
      int8x8x2_t i;
      i.val[0] = vget_low_s8(a_.neon_i8);
      i.val[1] = vget_high_s8(a_.neon_i8);

      /* Table lookups */
      int8x8_t l = vtbl2_s8(i, vget_low_s8(b_.neon_i8));
      int8x8_t h = vtbl2_s8(i, vget_high_s8(b_.neon_i8));

      r_.neon_i8 = vcombine_s8(l, h);
    #else
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[b_.i8[i] & 15] & (~(b_.i8[i]) >> 7);
      }
    #endif

    return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_shuffle_epi8(a, b) easysimd_mm_shuffle_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_shuffle_pi8 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_shuffle_pi8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    b.neon_i8 = vand_s8(b.neon_i8, vdup_n_s8(HEDLEY_STATIC_CAST(int8_t, (1 << 7) | 7)));
    r.neon_i8 = vtbl1_s8(a.neon_i8, b.neon_i8);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.i8[i] = a_.i8[b_.i8[i] & 7] & (~(b_.i8[i]) >> 7);
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_shuffle_pi8(a, b) easysimd_mm_shuffle_pi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_hadd_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_hadd_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16 = svadd_s16_z(pg, svuzp1_s16(a.sve_i16, b.sve_i16), svuzp2_s16(a.sve_i16, b.sve_i16));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i16 = vpaddq_s16(a.neon_i16, b.neon_i16);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int16x8x2_t t = vuzpq_s16(easysimd__m128i_to_neon_i16(a), easysimd__m128i_to_neon_i16(b));
    return easysimd__m128i_from_neon_i16(vaddq_s16(t.val[0], t.val[1]));
  #else
    return easysimd_mm_add_epi16(easysimd_x_mm_deinterleaveeven_epi16(a, b), easysimd_x_mm_deinterleaveodd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hadd_epi16(a, b) easysimd_mm_hadd_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_hadd_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_hadd_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32 = svadd_s32_z(pg, svuzp1_s32(a.sve_i32, b.sve_i32), svuzp2_s32(a.sve_i32, b.sve_i32));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i32 = vpaddq_s32(a.neon_i32, b.neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x4x2_t t = vuzpq_s32(easysimd__m128i_to_neon_i32(a), easysimd__m128i_to_neon_i32(b));
    return easysimd__m128i_from_neon_i32(vaddq_s32(t.val[0], t.val[1]));
  #else
    return easysimd_mm_add_epi32(easysimd_x_mm_deinterleaveeven_epi32(a, b), easysimd_x_mm_deinterleaveodd_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hadd_epi32(a, b) easysimd_mm_hadd_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_hadd_pi16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_hadd_pi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m64 r;
    r.neon_i16 = vpadd_s16(a.neon_i16, b.neon_i16);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int16x4x2_t t = vuzp_s16(a_.neon_i16, b_.neon_i16);
      r_.neon_i16 = vadd_s16(t.val[0], t.val[1]);
    #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.i16 =
        EASYSIMD_SHUFFLE_VECTOR_(16, 8, a_.i16, b_.i16, 0, 2, 4, 6) +
        EASYSIMD_SHUFFLE_VECTOR_(16, 8, a_.i16, b_.i16, 1, 3, 5, 7);
    #else
      r_.i16[0] = a_.i16[0] + a_.i16[1];
      r_.i16[1] = a_.i16[2] + a_.i16[3];
      r_.i16[2] = b_.i16[0] + b_.i16[1];
      r_.i16[3] = b_.i16[2] + b_.i16[3];
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hadd_pi16(a, b) easysimd_mm_hadd_pi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_hadd_pi32 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_hadd_pi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m64 r;
    r.neon_i32 = vpadd_s32(a.neon_i32, b.neon_i32);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int32x2x2_t t = vuzp_s32(a_.neon_i32, b_.neon_i32);
      r_.neon_i32 = vadd_s32(t.val[0], t.val[1]);
    #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.i32 =
        EASYSIMD_SHUFFLE_VECTOR_(32, 8, a_.i32, b_.i32, 0, 2) +
        EASYSIMD_SHUFFLE_VECTOR_(32, 8, a_.i32, b_.i32, 1, 3);
    #else
      r_.i32[0] = a_.i32[0] + a_.i32[1];
      r_.i32[1] = b_.i32[0] + b_.i32[1];
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hadd_pi32(a, b) easysimd_mm_hadd_pi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_hadds_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_hadds_epi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int16x8x2_t t = vuzpq_s16(easysimd__m128i_to_neon_i16(a), easysimd__m128i_to_neon_i16(b));
    return easysimd__m128i_from_neon_i16(vqaddq_s16(t.val[0], t.val[1]));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svint16_t sv1 = svuzp1_s16(a.sve_i16, b.sve_i16);
    svint16_t sv2 = svuzp2_s16(a.sve_i16, b.sve_i16);
    r.sve_i16 = svqadd_s16(sv1, sv2);
    return r;
  #else
    return easysimd_mm_adds_epi16(easysimd_x_mm_deinterleaveeven_epi16(a, b), easysimd_x_mm_deinterleaveodd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hadds_epi16(a, b) easysimd_mm_hadds_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_hadds_pi16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_hadds_pi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    int16x4x2_t t = vuzp_s16(a.neon_i16, b.neon_i16);
    r.neon_i16 = vqadd_s16(t.val[0], t.val[1]);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    for (size_t i = 0 ; i < ((sizeof(r_.i16) / sizeof(r_.i16[0])) / 2) ; i++) {
      int32_t ta = HEDLEY_STATIC_CAST(int32_t, a_.i16[i * 2]) + HEDLEY_STATIC_CAST(int32_t, a_.i16[(i * 2) + 1]);
      r_.i16[  i  ] = HEDLEY_LIKELY(ta > INT16_MIN) ? (HEDLEY_LIKELY(ta < INT16_MAX) ? HEDLEY_STATIC_CAST(int16_t, ta) : INT16_MAX) : INT16_MIN;
      int32_t tb = HEDLEY_STATIC_CAST(int32_t, b_.i16[i * 2]) + HEDLEY_STATIC_CAST(int32_t, b_.i16[(i * 2) + 1]);
      r_.i16[i + 2] = HEDLEY_LIKELY(tb > INT16_MIN) ? (HEDLEY_LIKELY(tb < INT16_MAX) ? HEDLEY_STATIC_CAST(int16_t, tb) : INT16_MAX) : INT16_MIN;
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hadds_pi16(a, b) easysimd_mm_hadds_pi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_hsub_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_hsub_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svint16_t sv1 = svuzp1_s16(a.sve_i16, b.sve_i16);
    svint16_t sv2 = svuzp2_s16(a.sve_i16, b.sve_i16);
    r.sve_i16 = svsub_s16_x(svptrue_b16(), sv1, sv2);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    int16x8x2_t t = vuzpq_s16(easysimd__m128i_to_neon_i16(a), easysimd__m128i_to_neon_i16(b));
    return easysimd__m128i_from_neon_i16(vsubq_s16(t.val[0], t.val[1]));
  #else
    easysimd__m128i r_;
    easysimd__m128i a_ = easysimd_x_mm_deinterleaveeven_epi16(a, b);
    easysimd__m128i b_ = easysimd_x_mm_deinterleaveodd_epi16(a, b);
    r_ = easysimd_mm_sub_epi16(a_, b_);
    return r_;
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hsub_epi16(a, b) easysimd_mm_hsub_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_hsub_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_hsub_epi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x4x2_t t = vuzpq_s32(easysimd__m128i_to_neon_i32(a), easysimd__m128i_to_neon_i32(b));
    return easysimd__m128i_from_neon_i32(vsubq_s32(t.val[0], t.val[1]));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svint32_t sv1 = svuzp1_s32(a.sve_i32, b.sve_i32);
    svint32_t sv2 = svuzp2_s32(a.sve_i32, b.sve_i32);
    r.sve_i32 = svsub_s32_z(svptrue_b32(), sv1, sv2);
    return r;
  #else
    return easysimd_mm_sub_epi32(easysimd_x_mm_deinterleaveeven_epi32(a, b), easysimd_x_mm_deinterleaveodd_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hsub_epi32(a, b) easysimd_mm_hsub_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_hsub_pi16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_hsub_pi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    int16x4x2_t t = vuzp_s16(a.neon_i16, b.neon_i16);
    r.neon_i16 = vsub_s16(t.val[0], t.val[1]);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    #if (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.i16 =
        EASYSIMD_SHUFFLE_VECTOR_(16, 8, a_.i16, b_.i16, 0, 2, 4, 6) -
        EASYSIMD_SHUFFLE_VECTOR_(16, 8, a_.i16, b_.i16, 1, 3, 5, 7);
    #else
      r_.i16[0] = a_.i16[0] - a_.i16[1];
      r_.i16[1] = a_.i16[2] - a_.i16[3];
      r_.i16[2] = b_.i16[0] - b_.i16[1];
      r_.i16[3] = b_.i16[2] - b_.i16[3];
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hsub_pi16(a, b) easysimd_mm_hsub_pi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_hsub_pi32 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_hsub_pi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    int32x2x2_t t = vuzp_s32(a.neon_i32, b.neon_i32);
    r.neon_i32 = vsub_s32(t.val[0], t.val[1]);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    #if (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.i32 =
        EASYSIMD_SHUFFLE_VECTOR_(32, 8, a_.i32, b_.i32, 0, 2) -
        EASYSIMD_SHUFFLE_VECTOR_(32, 8, a_.i32, b_.i32, 1, 3);
    #else
      r_.i32[0] = a_.i32[0] - a_.i32[1];
      r_.i32[1] = b_.i32[0] - b_.i32[1];
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hsub_pi32(a, b) easysimd_mm_hsub_pi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_hsubs_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_hsubs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int16x8x2_t t = vuzpq_s16(easysimd__m128i_to_neon_i16(a), easysimd__m128i_to_neon_i16(b));
    return easysimd__m128i_from_neon_i16(vqsubq_s16(t.val[0], t.val[1]));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svint16_t sv1 = svuzp1_s16(a.sve_i16, b.sve_i16);
    svint16_t sv2 = svuzp2_s16(a.sve_i16, b.sve_i16);
    r.sve_i16 = svqsub_s16(sv1, sv2);
    return r;
  #else
    return easysimd_mm_subs_epi16(easysimd_x_mm_deinterleaveeven_epi16(a, b), easysimd_x_mm_deinterleaveodd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hsubs_epi16(a, b) easysimd_mm_hsubs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_hsubs_pi16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_hsubs_pi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    int16x4x2_t t = vuzp_s16(a.neon_i16, b.neon_i16);
    r.neon_i16 = vqsub_s16(t.val[0], t.val[1]);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    for (size_t i = 0 ; i < ((sizeof(r_.i16) / sizeof(r_.i16[0])) / 2) ; i++) {
      r_.i16[  i  ] = easysimd_math_subs_i16(a_.i16[i * 2], a_.i16[(i * 2) + 1]);
      r_.i16[i + 2] = easysimd_math_subs_i16(b_.i16[i * 2], b_.i16[(i * 2) + 1]);
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hsubs_pi16(a, b) easysimd_mm_hsubs_pi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maddubs_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_maddubs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b16();
    r.sve_u8 = svuzp1(a.sve_u8, svdup_n_u8(0));
    svint16_t svaeven = svld1ub_s16(pg, &(r.u8[0]));
    r.sve_u8 = svuzp2(a.sve_u8, svdup_n_u8(0));
    svint16_t svaodd = svld1ub_s16(pg, &(r.u8[0]));

    r.sve_i8 = svuzp1(b.sve_i8, svdup_n_s8(0));
    svint16_t svbeven = svld1sb_s16(pg, &(r.i8[0]));
    r.sve_i8 = svuzp2(b.sve_i8, svdup_n_s8(0));
    svint16_t svbodd = svld1sb_s16(pg, &(r.i8[0]));
    r.sve_i16 = svqadd_s16_x(pg, svmul_s16_x(pg, svaeven, svbeven), svmul_s16_x(pg, svaodd, svbodd));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      /* Zero extend a */
      int16x8_t a_odd = vreinterpretq_s16_u16(vshrq_n_u16(a_.neon_u16, 8));
      int16x8_t a_even = vreinterpretq_s16_u16(vbicq_u16(a_.neon_u16, vdupq_n_u16(0xff00)));

      /* Sign extend by shifting left then shifting right. */
      int16x8_t b_even = vshrq_n_s16(vshlq_n_s16(b_.neon_i16, 8), 8);
      int16x8_t b_odd = vshrq_n_s16(b_.neon_i16, 8);

      /* multiply */
      int16x8_t prod1 = vmulq_s16(a_even, b_even);
      int16x8_t prod2 = vmulq_s16(a_odd, b_odd);

      /* saturated add */
      r_.neon_i16 = vqaddq_s16(prod1, prod2);
    #else
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        const int idx = HEDLEY_STATIC_CAST(int, i) << 1;
        int32_t ts =
          (HEDLEY_STATIC_CAST(int16_t, a_.u8[  idx  ]) * HEDLEY_STATIC_CAST(int16_t, b_.i8[  idx  ])) +
          (HEDLEY_STATIC_CAST(int16_t, a_.u8[idx + 1]) * HEDLEY_STATIC_CAST(int16_t, b_.i8[idx + 1]));
        r_.i16[i] = (ts > INT16_MIN) ? ((ts < INT16_MAX) ? HEDLEY_STATIC_CAST(int16_t, ts) : INT16_MAX) : INT16_MIN;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_maddubs_epi16(a, b) easysimd_mm_maddubs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_maddubs_pi16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_maddubs_pi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m64 r;
    int16x8_t ai = vreinterpretq_s16_u16(vmovl_u8(a.neon_u8));
    int16x8_t bi = vmovl_s8(b.neon_i8);
    int16x8_t p = vmulq_s16(ai, bi);
    int16x4_t l = vget_low_s16(p);
    int16x4_t h = vget_high_s16(p);
    r.neon_i16 = vqadd_s16(vuzp1_s16(l, h), vuzp2_s16(l, h));
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      const int idx = HEDLEY_STATIC_CAST(int, i) << 1;
      int32_t ts =
        (HEDLEY_STATIC_CAST(int16_t, a_.u8[  idx  ]) * HEDLEY_STATIC_CAST(int16_t, b_.i8[  idx  ])) +
        (HEDLEY_STATIC_CAST(int16_t, a_.u8[idx + 1]) * HEDLEY_STATIC_CAST(int16_t, b_.i8[idx + 1]));
      r_.i16[i] = (ts > INT16_MIN) ? ((ts < INT16_MAX) ? HEDLEY_STATIC_CAST(int16_t, ts) : INT16_MAX) : INT16_MIN;
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_maddubs_pi16(a, b) easysimd_mm_maddubs_pi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mulhrs_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_mulhrs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    svint32_t r0, r1, inc = svdup_n_s32(0x00004000);

    r0 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)])),
                     svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)])));
    r1 = svmul_s32_z(pg, svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)])),
                     svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)])));

    r0 = svasr_n_s32_z(pg, svadd_s32_z(pg, r0, inc), 15);
    r1 = svasr_n_s32_z(pg, svadd_s32_z(pg, r1, inc), 15);

    svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]), r1);
    svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]), r0);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      /* Multiply */
      int32x4_t mul_lo = vmull_s16(vget_low_s16(a_.neon_i16),
                                  vget_low_s16(b_.neon_i16));
      int32x4_t mul_hi = vmull_s16(vget_high_s16(a_.neon_i16),
                                  vget_high_s16(b_.neon_i16));

      /* Rounding narrowing shift right
       * narrow = (int16_t)((mul + 16384) >> 15); */
      int16x4_t narrow_lo = vrshrn_n_s32(mul_lo, 15);
      int16x4_t narrow_hi = vrshrn_n_s32(mul_hi, 15);

      /* Join together */
      r_.neon_i16 = vcombine_s16(narrow_lo, narrow_hi);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, (((HEDLEY_STATIC_CAST(int32_t, a_.i16[i]) * HEDLEY_STATIC_CAST(int32_t, b_.i16[i])) + 0x4000) >> 15));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_mulhrs_epi16(a, b) easysimd_mm_mulhrs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_mulhrs_pi16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_mulhrs_pi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    /* Multiply */
    int32x4_t mul = vmull_s16(a.neon_i16, b.neon_i16);

    /* Rounding narrowing shift right
     * narrow = (int16_t)((mul + 16384) >> 15); */
    int16x4_t narrow = vrshrn_n_s32(mul, 15);

    /* Join together */
    r.neon_i16 = narrow;
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, (((HEDLEY_STATIC_CAST(int32_t, a_.i16[i]) * HEDLEY_STATIC_CAST(int32_t, b_.i16[i])) + 0x4000) >> 15));
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_mulhrs_pi16(a, b) easysimd_mm_mulhrs_pi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sign_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_sign_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pgm0 = svcmplt_n_s8(svptrue_b8(), b.sve_i8, INT8_C(0));
    a.sve_i8 = svneg_s8_x(pgm0, a.sve_i8);
    svbool_t pgm1 = svcmpne_n_s8(svptrue_b8(), b.sve_i8, INT8_C(0));
    r.sve_i8 = svsel_s8(pgm1, a.sve_i8, svdup_n_s8(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint8x16_t aneg_mask = vreinterpretq_u8_s8(vshrq_n_s8(b_.neon_i8, 7));
      uint8x16_t bnz_mask;
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        bnz_mask = vceqzq_s8(b_.neon_i8);
      #else
        bnz_mask = vceqq_s8(b_.neon_i8, vdupq_n_s8(0));
      #endif
      bnz_mask = vmvnq_u8(bnz_mask);

      r_.neon_i8 = vbslq_s8(aneg_mask, vnegq_s8(a_.neon_i8), vandq_s8(a_.neon_i8, vreinterpretq_s8_u8(bnz_mask)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = (b_.i8[i] < 0) ? (- a_.i8[i]) : ((b_.i8[i] != 0) ? (a_.i8[i]) : INT8_C(0));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_sign_epi8(a, b) easysimd_mm_sign_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sign_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_sign_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pgm0 = svcmplt_n_s16(svptrue_b16(), b.sve_i16, INT16_C(0));
    a.sve_i16 = svneg_s16_x(pgm0, a.sve_i16);
    svbool_t pgm1 = svcmpne_n_s16(svptrue_b16(), b.sve_i16, INT16_C(0));
    r.sve_i16 = svsel_s16(pgm1, a.sve_i16, svdup_n_s16(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint16x8_t aneg_mask = vreinterpretq_u16_s16(vshrq_n_s16(b_.neon_i16, 15));
      uint16x8_t bnz_mask;
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        bnz_mask = vceqzq_s16(b_.neon_i16);
      #else
        bnz_mask = vceqq_s16(b_.neon_i16, vdupq_n_s16(0));
      #endif
      bnz_mask = vmvnq_u16(bnz_mask);

      r_.neon_i16 = vbslq_s16(aneg_mask, vnegq_s16(a_.neon_i16), vandq_s16(a_.neon_i16, vreinterpretq_s16_u16(bnz_mask)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (b_.i16[i] < 0) ? (- a_.i16[i]) : ((b_.i16[i] != 0) ? (a_.i16[i]) : INT16_C(0));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_sign_epi16(a, b) easysimd_mm_sign_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sign_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE)
    return _mm_sign_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pgm0 = svcmplt_n_s32(svptrue_b32(), b.sve_i32, INT32_C(0));
    a.sve_i32 = svneg_s32_x(pgm0, a.sve_i32);
    svbool_t pgm1 = svcmpne_n_s32(svptrue_b32(), b.sve_i32, INT32_C(0));
    r.sve_i32 = svsel_s32(pgm1, a.sve_i32, svdup_n_s32(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t aneg_mask = vreinterpretq_u32_s32(vshrq_n_s32(b_.neon_i32, 31));
      uint32x4_t bnz_mask;
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        bnz_mask = vceqzq_s32(b_.neon_i32);
      #else
        bnz_mask = vceqq_s32(b_.neon_i32, vdupq_n_s32(0));
      #endif
      bnz_mask = vmvnq_u32(bnz_mask);

      r_.neon_i32 = vbslq_s32(aneg_mask, vnegq_s32(a_.neon_i32), vandq_s32(a_.neon_i32, vreinterpretq_s32_u32(bnz_mask)));
    #else
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = (b_.i32[i] < 0) ? (- a_.i32[i]) : ((b_.i32[i] != 0) ? (a_.i32[i]) : INT32_C(0));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_sign_epi32(a, b) easysimd_mm_sign_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_sign_pi8 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_sign_pi8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    uint8x8_t aneg_mask = vreinterpret_u8_s8(vshr_n_s8(b.neon_i8, 7));
    uint8x8_t bnz_mask;
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      bnz_mask = vceqz_s8(b.neon_i8);
    #else
      bnz_mask = vceq_s8(b.neon_i8, vdup_n_s8(0));
    #endif
    bnz_mask = vmvn_u8(bnz_mask);

    r.neon_i8 = vbsl_s8(aneg_mask, vneg_s8(a.neon_i8), vand_s8(a.neon_i8, vreinterpret_s8_u8(bnz_mask)));
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = (b_.i8[i] < 0) ? (- a_.i8[i]) : ((b_.i8[i] != 0) ? (a_.i8[i]) : INT8_C(0));
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_sign_pi8(a, b) easysimd_mm_sign_pi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_sign_pi16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_sign_pi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    uint16x4_t aneg_mask = vreinterpret_u16_s16(vshr_n_s16(b.neon_i16, 15));
    uint16x4_t bnz_mask;
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      bnz_mask = vceqz_s16(b.neon_i16);
    #else
      bnz_mask = vceq_s16(b.neon_i16, vdup_n_s16(0));
    #endif
    bnz_mask = vmvn_u16(bnz_mask);

    r.neon_i16 = vbsl_s16(aneg_mask, vneg_s16(a.neon_i16), vand_s16(a.neon_i16, vreinterpret_s16_u16(bnz_mask)));
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = (b_.i16[i] < 0) ? (- a_.i16[i]) : ((b_.i16[i] > 0) ? (a_.i16[i]) : INT16_C(0));
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_sign_pi16(a, b) easysimd_mm_sign_pi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_sign_pi32 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSSE3_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_sign_pi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    uint32x2_t aneg_mask = vreinterpret_u32_s32(vshr_n_s32(b.neon_i32, 31));
    uint32x2_t bnz_mask;
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      bnz_mask = vceqz_s32(b.neon_i32);
    #else
      bnz_mask = vceq_s32(b.neon_i32, vdup_n_s32(0));
    #endif
    bnz_mask = vmvn_u32(bnz_mask);

    r.neon_i32 = vbsl_s32(aneg_mask, vneg_s32(a.neon_i32), vand_s32(a.neon_i32, vreinterpret_s32_u32(bnz_mask)));
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = (b_.i32[i] < 0) ? (- a_.i32[i]) : ((b_.i32[i] > 0) ? (a_.i32[i]) : INT32_C(0));
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_sign_pi32(a, b) easysimd_mm_sign_pi32(a, b)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_SSE2_H) */
