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

#include "sse.h"
#if !defined(EASYSIMD_X86_SSE4_1_H)
#define EASYSIMD_X86_SSE4_1_H

#include "ssse3.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

#if !defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
#  define EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_blend_epi16 (easysimd__m128i a, easysimd__m128i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsel_s16(EASYSIMD_MASK_TO_B16(imm8, EASYSIMD_SV_INDEX_0), b.sve_i16, a.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = ((imm8 >> i) & 1) ? b_.u16[i] : a_.u16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #define easysimd_mm_blend_epi16(a, b, imm8) _mm_blend_epi16(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_blend_epi16(a, b, imm8) \
    (__extension__ ({ \
      easysimd__m128i_private \
        easysimd_mm_blend_epi16_a_ = easysimd__m128i_to_private(a), \
        easysimd_mm_blend_epi16_b_ = easysimd__m128i_to_private(b), \
        easysimd_mm_blend_epi16_r_; \
      \
      easysimd_mm_blend_epi16_r_.i16 = \
        EASYSIMD_SHUFFLE_VECTOR_( \
          16, 16, \
          easysimd_mm_blend_epi16_a_.i16, \
          easysimd_mm_blend_epi16_b_.i16, \
          ((imm8) & (1 << 0)) ?  8 : 0, \
          ((imm8) & (1 << 1)) ?  9 : 1, \
          ((imm8) & (1 << 2)) ? 10 : 2, \
          ((imm8) & (1 << 3)) ? 11 : 3, \
          ((imm8) & (1 << 4)) ? 12 : 4, \
          ((imm8) & (1 << 5)) ? 13 : 5, \
          ((imm8) & (1 << 6)) ? 14 : 6, \
          ((imm8) & (1 << 7)) ? 15 : 7  \
        ); \
      \
      easysimd__m128i_from_private(easysimd_mm_blend_epi16_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_blend_epi16
  #define _mm_blend_epi16(a, b, imm8) easysimd_mm_blend_epi16(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_blend_pd (easysimd__m128d a, easysimd__m128d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsel_f64(EASYSIMD_MASK_TO_B64(imm8, EASYSIMD_SV_INDEX_0), b.sve_f64, a.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((imm8 >> i) & 1) ? b_.f64[i] : a_.f64[i];
    }
    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #define easysimd_mm_blend_pd(a, b, imm8) _mm_blend_pd(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_blend_pd(a, b, imm8) \
    (__extension__ ({ \
      easysimd__m128d_private \
        easysimd_mm_blend_pd_a_ = easysimd__m128d_to_private(a), \
        easysimd_mm_blend_pd_b_ = easysimd__m128d_to_private(b), \
        easysimd_mm_blend_pd_r_; \
      \
      easysimd_mm_blend_pd_r_.f64 = \
        EASYSIMD_SHUFFLE_VECTOR_( \
          64, 16, \
          easysimd_mm_blend_pd_a_.f64, \
          easysimd_mm_blend_pd_b_.f64, \
          ((imm8) & (1 << 0)) ?  2 : 0, \
          ((imm8) & (1 << 1)) ?  3 : 1  \
        ); \
      \
      easysimd__m128d_from_private(easysimd_mm_blend_pd_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_blend_pd
  #define _mm_blend_pd(a, b, imm8) easysimd_mm_blend_pd(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_blend_ps (easysimd__m128 a, easysimd__m128 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(imm8, EASYSIMD_SV_INDEX_0), b.sve_f32, a.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((imm8 >> i) & 1) ? b_.f32[i] : a_.f32[i];
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
#  define easysimd_mm_blend_ps(a, b, imm8) _mm_blend_ps(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_blend_ps(a, b, imm8) \
    (__extension__ ({ \
      easysimd__m128_private \
        easysimd_mm_blend_ps_a_ = easysimd__m128_to_private(a), \
        easysimd_mm_blend_ps_b_ = easysimd__m128_to_private(b), \
        easysimd_mm_blend_ps_r_; \
      \
      easysimd_mm_blend_ps_r_.f32 = \
        EASYSIMD_SHUFFLE_VECTOR_( \
          32, 16, \
          easysimd_mm_blend_ps_a_.f32, \
          easysimd_mm_blend_ps_b_.f32, \
          ((imm8) & (1 << 0)) ? 4 : 0, \
          ((imm8) & (1 << 1)) ? 5 : 1, \
          ((imm8) & (1 << 2)) ? 6 : 2, \
          ((imm8) & (1 << 3)) ? 7 : 3  \
        ); \
      \
      easysimd__m128_from_private(easysimd_mm_blend_ps_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_blend_ps
  #define _mm_blend_ps(a, b, imm8) easysimd_mm_blend_ps(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_blendv_epi8 (easysimd__m128i a, easysimd__m128i b, easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_blendv_epi8(a, b, mask);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m = _mm_cmpgt_epi8(_mm_setzero_si128(), mask);
    return _mm_xor_si128(_mm_subs_epu8(_mm_xor_si128(a, b), m), b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b8();
    easysimd_svbool_t pgm1 = svcmpeq_n_u8(pg, svlsr_n_u8_z(pg, mask.sve_u8, 7), 1);
    r.sve_i8 = svsel_s8(pgm1, b.sve_i8, a.sve_i8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      mask_ = easysimd__m128i_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      /* Use a signed shift right to create a mask with the sign bit */
      mask_.neon_i8 = vshrq_n_s8(mask_.neon_i8, 7);
      r_.neon_i8 = vbslq_s8(mask_.neon_u8, b_.neon_i8, a_.neon_i8);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      /* https://software.intel.com/en-us/forums/intel-c-compiler/topic/850087 */
      #if defined(HEDLEY_INTEL_VERSION_CHECK)
        __typeof__(mask_.i8) z = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        mask_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(mask_.i8), mask_.i8 < z);
      #else
        mask_.i8 >>= (CHAR_BIT * sizeof(mask_.i8[0])) - 1;
      #endif

      r_.i8 = (mask_.i8 & b_.i8) | (~mask_.i8 & a_.i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        int8_t m = mask_.i8[i] >> 7;
        r_.i8[i] = (m & b_.i8[i]) | (~m & a_.i8[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_blendv_epi8
  #define _mm_blendv_epi8(a, b, mask) easysimd_mm_blendv_epi8(a, b, mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_blendv_epi16 (easysimd__m128i a, easysimd__m128i b, easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    mask = easysimd_mm_srai_epi16(mask, 15);
    return easysimd_mm_or_si128(easysimd_mm_and_si128(mask, b), easysimd_mm_andnot_si128(mask, a));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      mask_ = easysimd__m128i_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      mask_ = easysimd__m128i_to_private(easysimd_mm_cmplt_epi16(mask, easysimd_mm_setzero_si128()));
      r_.neon_i16 = vbslq_s16(mask_.neon_u16, b_.neon_i16, a_.neon_i16);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      #if defined(HEDLEY_INTEL_VERSION_CHECK)
        __typeof__(mask_.i16) z = { 0, 0, 0, 0, 0, 0, 0, 0 };
        mask_.i16 = mask_.i16 < z;
      #else
        mask_.i16 >>= (CHAR_BIT * sizeof(mask_.i16[0])) - 1;
      #endif

      r_.i16 = (mask_.i16 & b_.i16) | (~mask_.i16 & a_.i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        int16_t m = mask_.i16[i] >> 15;
        r_.i16[i] = (m & b_.i16[i]) | (~m & a_.i16[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_blendv_epi32 (easysimd__m128i a, easysimd__m128i b, easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), _mm_castsi128_ps(mask)));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      mask_ = easysimd__m128i_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      mask_ = easysimd__m128i_to_private(easysimd_mm_cmplt_epi32(mask, easysimd_mm_setzero_si128()));
      r_.neon_i32 = vbslq_s32(mask_.neon_u32, b_.neon_i32, a_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      #if defined(HEDLEY_INTEL_VERSION_CHECK)
        __typeof__(mask_.i32) z = { 0, 0, 0, 0 };
        mask_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(mask_.i32), mask_.i32 < z);
      #else
        mask_.i32 >>= (CHAR_BIT * sizeof(mask_.i32[0])) - 1;
      #endif

      r_.i32 = (mask_.i32 & b_.i32) | (~mask_.i32 & a_.i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        int32_t m = mask_.i32[i] >> 31;
        r_.i32[i] = (m & b_.i32[i]) | (~m & a_.i32[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_blendv_epi64 (easysimd__m128i a, easysimd__m128i b, easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_castpd_si128(_mm_blendv_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b), _mm_castsi128_pd(mask)));
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      mask_ = easysimd__m128i_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      mask_.neon_u64 = vcltq_s64(mask_.neon_i64, vdupq_n_s64(UINT64_C(0)));
      r_.neon_i64 = vbslq_s64(mask_.neon_u64, b_.neon_i64, a_.neon_i64);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      #if defined(HEDLEY_INTEL_VERSION_CHECK)
        __typeof__(mask_.i64) z = { 0, 0 };
        mask_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(mask_.i64), mask_.i64 < z);
      #else
        mask_.i64 >>= (CHAR_BIT * sizeof(mask_.i64[0])) - 1;
      #endif

    r_.i64 = (mask_.i64 & b_.i64) | (~mask_.i64 & a_.i64);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      int64_t m = mask_.i64[i] >> 63;
      r_.i64[i] = (m & b_.i64[i]) | (~m & a_.i64[i]);
    }
  #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_blendv_pd (easysimd__m128d a, easysimd__m128d b, easysimd__m128d mask) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_blendv_pd(a, b, mask);
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    easysimd_svbool_t pg = svcmplt_n_s64(svptrue_b64(), mask.sve_i64, INT64_C(0));
    r.sve_f64 = svsel_f64(pg, b.sve_f64, a.sve_f64);
    return r;
  #else
    return easysimd_mm_castsi128_pd(easysimd_x_mm_blendv_epi64(easysimd_mm_castpd_si128(a), easysimd_mm_castpd_si128(b), easysimd_mm_castpd_si128(mask)));
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_blendv_pd
  #define _mm_blendv_pd(a, b, mask) easysimd_mm_blendv_pd(a, b, mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_blendv_ps (easysimd__m128 a, easysimd__m128 b, easysimd__m128 mask) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_blendv_ps(a, b, mask);
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    easysimd_svbool_t pg = svcmplt_n_s32(svptrue_b32(), mask.sve_i32, INT32_C(0));
    r.sve_f32 = svsel_f32(pg, b.sve_f32, a.sve_f32);
    return r;
  #else
    return easysimd_mm_castsi128_ps(easysimd_x_mm_blendv_epi32(easysimd_mm_castps_si128(a), easysimd_mm_castps_si128(b), easysimd_mm_castps_si128(mask)));
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_blendv_ps
  #define _mm_blendv_ps(a, b, mask) easysimd_mm_blendv_ps(a, b, mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_round_pd (easysimd__m128d a, int rounding)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(rounding, 0, 15) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(svptrue_b64(), a.sve_f64);
    return r;
  #else
  easysimd__m128d_private
    r_,
    a_ = easysimd__m128d_to_private(a);

  switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC) {
    case EASYSIMD_MM_FROUND_CUR_DIRECTION:
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        r_.neon_f64 = vrndiq_f64(a_.neon_f64);
      #elif defined(easysimd_math_nearbyint)
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_nearbyint(a_.f64[i]);
        }
      #else
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
      #endif
      break;

    case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        r_.neon_f64 = vrndaq_f64(a_.neon_f64);
      #elif defined(easysimd_math_roundeven)
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_roundeven(a_.f64[i]);
        }
      #else
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
      #endif
      break;

    case EASYSIMD_MM_FROUND_TO_NEG_INF:
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        r_.neon_f64 = vrndmq_f64(a_.neon_f64);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_floor(a_.f64[i]);
        }
      #endif
      break;

    case EASYSIMD_MM_FROUND_TO_POS_INF:
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        r_.neon_f64 = vrndpq_f64(a_.neon_f64);
      #elif defined(easysimd_math_ceil)
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_ceil(a_.f64[i]);
        }
      #else
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
      #endif
      break;

    case EASYSIMD_MM_FROUND_TO_ZERO:
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        r_.neon_f64 = vrndq_f64(a_.neon_f64);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_trunc(a_.f64[i]);
        }
      #endif
      break;

    default:
      HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
  }

  return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #define easysimd_mm_round_pd(a, rounding) _mm_round_pd(a, rounding)
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_round_pd
  #define _mm_round_pd(a, rounding) easysimd_mm_round_pd(a, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_ceil_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svrintp_f64_z(svptrue_b64(), a.sve_f64);
    return r;
  #else
    return easysimd_mm_round_pd(a, EASYSIMD_MM_FROUND_TO_POS_INF);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_ceil_pd
  #define _mm_ceil_pd(a) easysimd_mm_ceil_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_ceil_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svrintp_f32_z(svptrue_b32(), a.sve_f32);
    return r;
  #else
    return easysimd_mm_round_ps(a, EASYSIMD_MM_FROUND_TO_POS_INF);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_ceil_ps
  #define _mm_ceil_ps(a) easysimd_mm_ceil_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_ceil_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_ceil_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    b.sve_f64 = svrintp_f64_z(svptrue_b64(), b.sve_f64);
    r.sve_f64 = svdupq_n_f64(b.f64[0], a.f64[1]);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(easysimd_math_ceilf)
      r_ = easysimd__m128d_to_private(easysimd_mm_set_pd(a_.f64[1], easysimd_math_ceil(b_.f64[0])));
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_ceil_sd
  #define _mm_ceil_sd(a, b) easysimd_mm_ceil_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_ceil_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_ceil_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    b.sve_f32 = svrintp_f32_z(svptrue_b32(), b.sve_f32);
    a.f32[0] = b.f32[0];
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_ceil_ps(b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_ceil_ps(easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(easysimd_math_ceilf)
      r_ = easysimd__m128_to_private(easysimd_mm_set_ps(a_.f32[3], a_.f32[2], a_.f32[1], easysimd_math_ceilf(b_.f32[0])));
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_ceil_ss
  #define _mm_ceil_ss(a, b) easysimd_mm_ceil_ss(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpeq_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cmpeq_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b64();
    r.sve_u64 = svdup_n_u64_z(svcmpeq_s64(pg, a.sve_i64, b.sve_i64), ~UINT64_C(0));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_u64 = vceqq_s64(a.neon_i64, b.neon_i64);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      /* (a == b) -> (a_lo == b_lo) && (a_hi == b_hi) */
      uint32x4_t cmp = vceqq_u32(a_.neon_u32, b_.neon_u32);
      uint32x4_t swapped = vrev64q_u32(cmp);
      r_.neon_u32 = vandq_u32(cmp, swapped);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 == b_.i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = (a_.u64[i] == b_.u64[i]) ? ~UINT64_C(0) : UINT64_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpeq_epi64
  #define _mm_cmpeq_epi64(a, b) easysimd_mm_cmpeq_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepi8_epi16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepi8_epi16(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_srai_epi16(_mm_unpacklo_epi8(a, a), 8);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svld1sb_s16(svptrue_b16(), &(a.i8[0]));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int8x16_t s8x16 = a_.neon_i8;                   /* xxxx xxxx xxxx DCBA */
      int16x8_t s16x8 = vmovl_s8(vget_low_s8(s8x16)); /* 0x0x 0x0x 0D0C 0B0A */
      r_.neon_i16 = s16x8;
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_VECTOR_SCALAR) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.i8, a_.i8,
          -1,  0, -1,  1, -1,  2,  -1,  3,
          -1,  4, -1,  5, -1,  6,  -1,  7));
      r_.i16 >>= 8;
    #elif defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i16, a_.m64_private[0].i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepi8_epi16
  #define _mm_cvtepi8_epi16(a) easysimd_mm_cvtepi8_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepi8_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepi8_epi32(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i tmp = _mm_unpacklo_epi8(a, a);
    tmp = _mm_unpacklo_epi16(tmp, tmp);
    return _mm_srai_epi32(tmp, 24);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1sb_s32(svptrue_b32(), &(a.i8[0]));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int8x16_t s8x16 = a_.neon_i8;                     /* xxxx xxxx xxxx DCBA */
      int16x8_t s16x8 = vmovl_s8(vget_low_s8(s8x16));   /* 0x0x 0x0x 0D0C 0B0A */
      int32x4_t s32x4 = vmovl_s16(vget_low_s16(s16x8)); /* 000D 000C 000B 000A */
      r_.neon_i32 = s32x4;
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_VECTOR_SCALAR) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.i8, a_.i8,
          -1, -1, -1,  0, -1, -1,  -1,  1,
          -1, -1, -1,  2, -1, -1,  -1,  3));
      r_.i32 >>= 24;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepi8_epi32
  #define _mm_cvtepi8_epi32(a) easysimd_mm_cvtepi8_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepi8_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepi8_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1sb_s64(svptrue_b64(), &(a.i8[0]));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int8x16_t s8x16 = a_.neon_i8;                     /* xxxx xxxx xxxx xxBA */
      int16x8_t s16x8 = vmovl_s8(vget_low_s8(s8x16));   /* 0x0x 0x0x 0x0x 0B0A */
      int32x4_t s32x4 = vmovl_s16(vget_low_s16(s16x8)); /* 000x 000x 000B 000A */
      int64x2_t s64x2 = vmovl_s32(vget_low_s32(s32x4)); /* 0000 000B 0000 000A */
      r_.neon_i64 = s64x2;
    #elif (!defined(EASYSIMD_ARCH_X86) && !defined(EASYSIMD_ARCH_AMD64)) && defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_VECTOR_SCALAR) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      /* Disabled on x86 due to lack of 64-bit arithmetic shift until
       * until AVX-512 (at which point we would be using the native
       * _mm_cvtepi_epi64 anyways). */
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.i8, a_.i8,
          -1, -1, -1, -1, -1, -1,  -1,  0,
          -1, -1, -1, -1, -1, -1,  -1,  1));
      r_.i64 >>= 56;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepi8_epi64
  #define _mm_cvtepi8_epi64(a) easysimd_mm_cvtepi8_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepu8_epi16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepu8_epi16(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpacklo_epi8(a, _mm_setzero_si128());
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16 = svld1ub_s16(pg, &(a.u8[0]));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint8x16_t u8x16 = a_.neon_u8;                   /* xxxx xxxx xxxx DCBA */
      uint16x8_t u16x8 = vmovl_u8(vget_low_u8(u8x16)); /* 0x0x 0x0x 0D0C 0B0A */
      r_.neon_u16 = u16x8;
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      __typeof__(r_.i8) z = { 0, };
      r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.i8, z,
          0, 16, 1, 17, 2, 18, 3, 19,
          4, 20, 5, 21, 6, 22, 7, 23));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.u8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepu8_epi16
  #define _mm_cvtepu8_epi16(a) easysimd_mm_cvtepu8_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepu8_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepu8_epi32(a);
  #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
    __m128i s = _mm_set_epi8(
        HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x03),
        HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x02),
        HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x01),
        HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x00));
    return _mm_shuffle_epi8(a, s);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi16(_mm_unpacklo_epi8(a, z), z);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1ub_s32(svptrue_b32(), &(a.u8[0]));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint8x16_t u8x16 = a_.neon_u8;                     /* xxxx xxxx xxxx DCBA */
      uint16x8_t u16x8 = vmovl_u8(vget_low_u8(u8x16));   /* 0x0x 0x0x 0D0C 0B0A */
      uint32x4_t u32x4 = vmovl_u16(vget_low_u16(u16x8)); /* 000D 000C 000B 000A */
      r_.neon_u32 = u32x4;
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      __typeof__(r_.i8) z = { 0, };
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.i8, z,
          0, 17, 18, 19, 1, 21, 22, 23,
          2, 25, 26, 27, 3, 29, 30, 31));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.u8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepu8_epi32
  #define _mm_cvtepu8_epi32(a) easysimd_mm_cvtepu8_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepu8_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepu8_epi64(a);
  #elif defined(EASYSIMD_X86_SSSE3_NATIVE)
    __m128i s = _mm_set_epi8(
        HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80),
        HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x01),
        HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80),
        HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x80), HEDLEY_STATIC_CAST(char, 0x00));
    return _mm_shuffle_epi8(a, s);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi32(_mm_unpacklo_epi16(_mm_unpacklo_epi8(a, z), z), z);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1ub_s64(svptrue_b64(), &(a.u8[0]));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint8x16_t u8x16 = a_.neon_u8;                     /* xxxx xxxx xxxx xxBA */
      uint16x8_t u16x8 = vmovl_u8(vget_low_u8(u8x16));   /* 0x0x 0x0x 0x0x 0B0A */
      uint32x4_t u32x4 = vmovl_u16(vget_low_u16(u16x8)); /* 000x 000x 000B 000A */
      uint64x2_t u64x2 = vmovl_u32(vget_low_u32(u32x4)); /* 0000 000B 0000 000A */
      r_.neon_u64 = u64x2;
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      __typeof__(r_.i8) z = { 0, };
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), EASYSIMD_SHUFFLE_VECTOR_(8, 16, a_.i8, z,
          0, 17, 18, 19, 20, 21, 22, 23,
          1, 25, 26, 27, 28, 29, 30, 31));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.u8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepu8_epi64
  #define _mm_cvtepu8_epi64(a) easysimd_mm_cvtepu8_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepi16_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepi16_epi32(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_srai_epi32(_mm_unpacklo_epi16(a, a), 16);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32 = svld1sh_s32(pg, &a.i16[0]);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vmovl_s16(vget_low_s16(a_.neon_i16));
    #elif !defined(EASYSIMD_ARCH_X86) && defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_VECTOR_SCALAR) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.i16, a_.i16, 8, 0, 10, 1, 12, 2, 14, 3));
      r_.i32 >>= 16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepi16_epi32
  #define _mm_cvtepi16_epi32(a) easysimd_mm_cvtepi16_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepu16_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepu16_epi32(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpacklo_epi16(a, _mm_setzero_si128());
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32 = svld1uh_s32(pg, &(a.u16[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vmovl_u16(vget_low_u16(a_.neon_u16));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.u16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepu16_epi32
  #define _mm_cvtepu16_epi32(a) easysimd_mm_cvtepu16_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepu16_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepu16_epi64(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi32(_mm_unpacklo_epi16(a, z), z);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1uh_s64(svptrue_b64(), &(a.u16[0]));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint16x8_t u16x8 = a_.neon_u16;                    /* xxxx xxxx xxxx 0B0A */
      uint32x4_t u32x4 = vmovl_u16(vget_low_u16(u16x8)); /* 000x 000x 000B 000A */
      uint64x2_t u64x2 = vmovl_u32(vget_low_u32(u32x4)); /* 0000 000B 0000 000A */
      r_.neon_u64 = u64x2;
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      __typeof__(r_.u16) z = { 0, };
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.u16, z,
          0,  9, 10, 11,
          1, 13, 14, 15));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.u16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepu16_epi64
  #define _mm_cvtepu16_epi64(a) easysimd_mm_cvtepu16_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepi16_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepi16_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1sh_s64(svptrue_b64(), &(a.i16[0]));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int16x8_t s16x8 = a_.neon_i16;                    /* xxxx xxxx xxxx 0B0A */
      int32x4_t s32x4 = vmovl_s16(vget_low_s16(s16x8)); /* 000x 000x 000B 000A */
      int64x2_t s64x2 = vmovl_s32(vget_low_s32(s32x4)); /* 0000 000B 0000 000A */
      r_.neon_i64 = s64x2;
    #elif (!defined(EASYSIMD_ARCH_X86) && !defined(EASYSIMD_ARCH_AMD64)) && defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_VECTOR_SCALAR) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.i16, a_.i16,
           8,  9, 10, 0,
          12, 13, 14, 1));
      r_.i64 >>= 48;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepi16_epi64
  #define _mm_cvtepi16_epi64(a) easysimd_mm_cvtepi16_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepi32_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepi32_epi64(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i tmp = _mm_shuffle_epi32(a, 0x50);
    tmp = _mm_srai_epi32(tmp, 31);
    tmp = _mm_shuffle_epi32(tmp, 0xed);
    return _mm_unpacklo_epi32(a, tmp);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1sw_gather_s64offset_s64(svptrue_b64(), (const int *)&(a.i32[0]), svdupq_n_s64(0, 4));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vmovl_s32(vget_low_s32(a_.neon_i32));
    #elif !defined(EASYSIMD_ARCH_X86) && defined(EASYSIMD_SHUFFLE_VECTOR_) && defined(EASYSIMD_VECTOR_SCALAR) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.i32, a_.i32, -1, 0, -1, 1));
      r_.i64 >>= 32;
    #elif defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i64, a_.m64_private[0].i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i32[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepi32_epi64
  #define _mm_cvtepi32_epi64(a) easysimd_mm_cvtepi32_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtepu32_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_cvtepu32_epi64(a);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpacklo_epi32(a, _mm_setzero_si128());
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1uw_gather_s64offset_s64(svptrue_b64(), (const uint32_t *)&(a.u32[0]), svdupq_n_s64(0, 4));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u64 = vmovl_u32(vget_low_u32(a_.neon_u32));
    #elif defined(EASYSIMD_VECTOR_SCALAR) && defined(EASYSIMD_SHUFFLE_VECTOR_) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
      __typeof__(r_.u32) z = { 0, };
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.u32, z, 0, 4, 1, 6));
    #elif defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i64, a_.m64_private[0].u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.u32[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepu32_epi64
  #define _mm_cvtepu32_epi64(a) easysimd_mm_cvtepu32_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cvtepu32_pd (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_cvtepu32_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svcvt_f64_u32_z(svptrue_b64(), svtbl_u32(a.sve_u32, svdupq_n_u32(0, 0, 1, 0)));
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = HEDLEY_STATIC_CAST(easysimd_float64, a_.u32[i]);
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_cvtepu32_pd
  #define _mm_cvtepu32_pd(a) easysimd_mm_cvtepu32_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_dp_pd (easysimd__m128d a, easysimd__m128d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  easysimd__m128d_private
    r_,
    a_ = easysimd__m128d_to_private(a),
    b_ = easysimd__m128d_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_f64 = vmulq_f64(a_.neon_f64, b_.neon_f64);

    switch (imm8) {
      case 0xff:
        r_.neon_f64 = vaddq_f64(r_.neon_f64, vextq_f64(r_.neon_f64, r_.neon_f64, 1));
        break;
      case 0x13:
        r_.neon_f64 = vdupq_lane_f64(vget_low_f64(r_.neon_f64), 0);
        break;
      default:
        { /* imm8 is a compile-time constant, so this all becomes just a load */
          uint64_t mask_data[] = {
            (imm8 & (1 << 4)) ? ~UINT64_C(0) : UINT64_C(0),
            (imm8 & (1 << 5)) ? ~UINT64_C(0) : UINT64_C(0),
          };
          r_.neon_f64 = vreinterpretq_f64_u64(vandq_u64(vld1q_u64(mask_data), vreinterpretq_u64_f64(r_.neon_f64)));
        }

        r_.neon_f64 = vdupq_n_f64(vaddvq_f64(r_.neon_f64));

        {
          uint64_t mask_data[] = {
            (imm8 & 1) ? ~UINT64_C(0) : UINT64_C(0),
            (imm8 & 2) ? ~UINT64_C(0) : UINT64_C(0)
          };
          r_.neon_f64 = vreinterpretq_f64_u64(vandq_u64(vld1q_u64(mask_data), vreinterpretq_u64_f64(r_.neon_f64)));
        }
        break;
    }
  #else
    easysimd_float64 sum = EASYSIMD_FLOAT64_C(0.0);

    EASYSIMD_VECTORIZE_REDUCTION(+:sum)
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      sum += ((imm8 >> (i + 4)) & 1) ? (a_.f64[i] * b_.f64[i]) : 0.0;
    }

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = ((imm8 >> i) & 1) ? sum : 0.0;
    }
  #endif

  return easysimd__m128d_from_private(r_);
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
#  define easysimd_mm_dp_pd(a, b, imm8) _mm_dp_pd(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_dp_pd
  #define _mm_dp_pd(a, b, imm8) easysimd_mm_dp_pd(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_dp_ps (easysimd__m128 a, easysimd__m128 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  easysimd__m128_private
    r_,
    a_ = easysimd__m128_to_private(a),
    b_ = easysimd__m128_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_f32 = vmulq_f32(a_.neon_f32, b_.neon_f32);

    switch (imm8) {
      case 0xff:
        r_.neon_f32 = vdupq_n_f32(vaddvq_f32(r_.neon_f32));
        break;
      case 0x7f:
        r_.neon_f32 = vsetq_lane_f32(0, r_.neon_f32, 3);
        r_.neon_f32 = vdupq_n_f32(vaddvq_f32(r_.neon_f32));
        break;
      default:
        {
          {
            uint32_t mask_data[] = {
              (imm8 & (1 << 4)) ? ~UINT32_C(0) : UINT32_C(0),
              (imm8 & (1 << 5)) ? ~UINT32_C(0) : UINT32_C(0),
              (imm8 & (1 << 6)) ? ~UINT32_C(0) : UINT32_C(0),
              (imm8 & (1 << 7)) ? ~UINT32_C(0) : UINT32_C(0)
            };
            r_.neon_f32 = vreinterpretq_f32_u32(vandq_u32(vld1q_u32(mask_data), vreinterpretq_u32_f32(r_.neon_f32)));
          }

          r_.neon_f32 = vdupq_n_f32(vaddvq_f32(r_.neon_f32));

          {
            uint32_t mask_data[] = {
              (imm8 & 1) ? ~UINT32_C(0) : UINT32_C(0),
              (imm8 & 2) ? ~UINT32_C(0) : UINT32_C(0),
              (imm8 & 4) ? ~UINT32_C(0) : UINT32_C(0),
              (imm8 & 8) ? ~UINT32_C(0) : UINT32_C(0)
            };
            r_.neon_f32 = vreinterpretq_f32_u32(vandq_u32(vld1q_u32(mask_data), vreinterpretq_u32_f32(r_.neon_f32)));
          }
        }
        break;
    }
  #else
    easysimd_float32 sum = EASYSIMD_FLOAT32_C(0.0);

    EASYSIMD_VECTORIZE_REDUCTION(+:sum)
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      sum += ((imm8 >> (i + 4)) & 1) ? (a_.f32[i] * b_.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
    }

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((imm8 >> i) & 1) ? sum : EASYSIMD_FLOAT32_C(0.0);
    }
  #endif

  return easysimd__m128_from_private(r_);
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #if defined(HEDLEY_MCST_LCC_VERSION)
    #define easysimd_mm_dp_ps(a, b, imm8) (__extension__ ({ \
      EASYSIMD_LCC_DISABLE_DEPRECATED_WARNINGS \
      _mm_dp_ps((a), (b), (imm8)); \
      EASYSIMD_LCC_REVERT_DEPRECATED_WARNINGS \
    }))
  #else
    #define easysimd_mm_dp_ps(a, b, imm8) _mm_dp_ps(a, b, imm8)
  #endif
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_dp_ps
  #define _mm_dp_ps(a, b, imm8) easysimd_mm_dp_ps(a, b, imm8)
#endif

#if defined(easysimd_mm_extract_epi8)
#  undef easysimd_mm_extract_epi8
#endif
EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_mm_extract_epi8 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.i8[imm8 & 15];
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a);

      return a_.i8[imm8 & 15];
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_MM_EXTRACT_EPI8)
#  define easysimd_mm_extract_epi8(a, imm8) HEDLEY_STATIC_CAST(int8_t, _mm_extract_epi8(a, imm8))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
#  define easysimd_mm_extract_epi8(a, imm8) vgetq_lane_s8(a.neon_i8, imm8);
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_extract_epi8
  #define _mm_extract_epi8(a, imm8) HEDLEY_STATIC_CAST(int, easysimd_mm_extract_epi8(a, imm8))
#endif

#if defined(easysimd_mm_extract_epi32)
#  undef easysimd_mm_extract_epi32
#endif
EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_extract_epi32 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  return a.i32[imm8 & 3];
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a);
    
  #if defined(EASYSIMD_POWER_ALTIVEC_P6_NATIVE)
    #if defined(EASYSIMD_BUG_GCC_95227)
      (void) a_;
      (void) imm8;
    #endif
    return vec_extract(a_.altivec_i32, imm8);
  #else
    return a_.i32[imm8 & 3];
  #endif
#endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
#  define easysimd_mm_extract_epi32(a, imm8) _mm_extract_epi32(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
#  define easysimd_mm_extract_epi32(a, imm8) vgetq_lane_s32(a.neon_i32, imm8)
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_extract_epi32
  #define _mm_extract_epi32(a, imm8) easysimd_mm_extract_epi32(a, imm8)
#endif

#if defined(easysimd_mm_extract_epi64)
#  undef easysimd_mm_extract_epi64
#endif
EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm_extract_epi64 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.i64[imm8 & 1];
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    return a_.i64[imm8 & 1];
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
#  define easysimd_mm_extract_epi64(a, imm8) _mm_extract_epi64(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
#  define easysimd_mm_extract_epi64(a, imm8) vgetq_lane_s64(a.neon_i64, imm8);
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
  #undef _mm_extract_epi64
  #define _mm_extract_epi64(a, imm8) easysimd_mm_extract_epi64(a, imm8)
#endif

#if defined(easysimd_mm_extract_ps)
#  undef easysimd_mm_extract_ps
#endif
EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_extract_ps (easysimd__m128 a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.i32[imm8 & 3];
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);
    return a_.i32[imm8 & 3];
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #define easysimd_mm_extract_ps(a, imm8) _mm_extract_ps(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_mm_extract_ps(a, imm8) vgetq_lane_s32(a.neon_i32, imm8);
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_extract_ps
  #define _mm_extract_ps(a, imm8) easysimd_mm_extract_ps(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_floor_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svrintm_f64_z(svptrue_b64(), a.sve_f64);
    return r;
  #else
    return easysimd_mm_round_pd(a, EASYSIMD_MM_FROUND_TO_NEG_INF);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_floor_pd
  #define _mm_floor_pd(a) easysimd_mm_floor_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_floor_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svrintm_f32_z(svptrue_b32(), a.sve_f32);
    return r;
  #else
    return easysimd_mm_round_ps(a, EASYSIMD_MM_FROUND_TO_NEG_INF);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_floor_ps
  #define _mm_floor_ps(a) easysimd_mm_floor_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_floor_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_floor_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    b.sve_f64 = svrintm_f64_z(svptrue_b64(), b.sve_f64);
    r.sve_f64 = svdupq_n_f64(b.f64[0], a.f64[1]);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(easysimd_math_floor)
      r_.f64[0] = easysimd_math_floor(b_.f64[0]);
      r_.f64[1] = a_.f64[1];
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_floor_sd
  #define _mm_floor_sd(a, b) easysimd_mm_floor_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_floor_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_floor_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    b.sve_f32 = svrintm_f32_z(svptrue_b32(), b.sve_f32);
    a.f32[0] = b.f32[0];
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
      return easysimd_mm_move_ss(a, easysimd_mm_floor_ps(b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_floor_ps(easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(easysimd_math_floorf)
      r_.f32[0] = easysimd_math_floorf(b_.f32[0]);
      for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i];
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_floor_ss
  #define _mm_floor_ss(a, b) easysimd_mm_floor_ss(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_insert_epi8 (easysimd__m128i a, int i, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i8[imm8 & 15] = (int8_t)i;
    return a;
  #else
    easysimd__m128i_private
      r_ = easysimd__m128i_to_private(a);

    r_.i8[imm8] = HEDLEY_STATIC_CAST(int8_t, i);

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
  /* clang-3.8 returns an incompatible type, so we need the cast.  MSVC
   * can't handle the cast ("error C2440: 'type cast': cannot convert
   * from '__m128i' to '__m128i'").  */
  #if defined(__clang__)
    #define easysimd_mm_insert_epi8(a, i, imm8) HEDLEY_REINTERPRET_CAST(__m128i, _mm_insert_epi8(a, i, imm8))
  #else
    #define easysimd_mm_insert_epi8(a, i, imm8) _mm_insert_epi8(a, i, imm8)
  #endif
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    #define easysimd_mm_insert_epi8(a, i, imm8) \
    ({ \
        easysimd__m128i b_; \
        b_.neon_i8 = vsetq_lane_s8(i, a.neon_i8, imm8); \
        b_; \
    })
#endif

#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_insert_epi8
  #define _mm_insert_epi8(a, i, imm8) easysimd_mm_insert_epi8(a, i, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_insert_epi32 (easysimd__m128i a, int i, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i32[imm8 & 3] = i;
    return a;
  #else
    easysimd__m128i_private
      r_ = easysimd__m128i_to_private(a);

    r_.i32[imm8 & 3] = HEDLEY_STATIC_CAST(int32_t, i);

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #if defined(__clang__)
    #define easysimd_mm_insert_epi32(a, i, imm8) HEDLEY_REINTERPRET_CAST(__m128i, _mm_insert_epi32(a, i, imm8))
  #else
    #define easysimd_mm_insert_epi32(a, i, imm8) _mm_insert_epi32(a, i, imm8)
  #endif
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm_insert_epi32(a, i, imm8) \
    ({ \
        easysimd__m128i b_; \
        b_.neon_i32 = vsetq_lane_s32(i, a.neon_i32, imm8); \
        b_; \
    })
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_insert_epi32
  #define _mm_insert_epi32(a, i, imm8) easysimd_mm_insert_epi32(a, i, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_insert_epi64 (easysimd__m128i a, int64_t i, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1)  {
  #if defined(EASYSIMD_BUG_GCC_94482)
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a);

    switch(imm8) {
      case 0:
        return easysimd_mm_set_epi64x(a_.i64[1], i);
        break;
      case 1:
        return easysimd_mm_set_epi64x(i, a_.i64[0]);
        break;
      default:
        HEDLEY_UNREACHABLE();
        break;
    }
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_i64[imm8 & 1] = i;
    return a;
  #else
    easysimd__m128i_private
      r_ = easysimd__m128i_to_private(a);

    r_.i64[imm8 & 1] = i;
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
#  define easysimd_mm_insert_epi64(a, i, imm8) _mm_insert_epi64(a, i, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
#  define easysimd_mm_insert_epi64(a, i, imm8) \
    ({ \
      easysimd__m128i b_; \
      b_.neon_i64 = vsetq_lane_s64(i, a.neon_i64, imm8); \
      b_; \
    })
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
  #undef _mm_insert_epi64
  #define _mm_insert_epi64(a, i, imm8) easysimd_mm_insert_epi64(a, i, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_insert_ps (easysimd__m128 a, easysimd__m128 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    a.f32[0] = b.f32[(imm8 >> 6) & 3];
    a.f32[(imm8 >> 4) & 3] = a.f32[0];
    r.sve_f32 = svsel_f32(EASYSIMD_MASK_TO_B32(imm8, EASYSIMD_SV_INDEX_0), svdup_n_f32(0.0), a.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    a_.f32[0] = b_.f32[(imm8 >> 6) & 3];
    a_.f32[(imm8 >> 4) & 3] = a_.f32[0];

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = ((imm8 >> i) & 1) ? EASYSIMD_FLOAT32_C(0.0) : a_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
#  define easysimd_mm_insert_ps(a, b, imm8) _mm_insert_ps(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_insert_ps
  #define _mm_insert_ps(a, b, imm8) easysimd_mm_insert_ps(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_max_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && !defined(__PGI)
    return _mm_max_epi8(a, b);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m = _mm_cmpgt_epi8(a, b);
    return _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8 = svmax_s8_x(pg, a.sve_i8, b.sve_i8);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_i8 = vmaxq_s8(a.neon_i8, b.neon_i8);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = a_.i8[i] > b_.i8[i] ? a_.i8[i] : b_.i8[i];
    }
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_max_epi8
  #define _mm_max_epi8(a, b) easysimd_mm_max_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_max_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && !defined(__PGI)
    return _mm_max_epi32(a, b);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m = _mm_cmpgt_epi32(a, b);
    return _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32 = svmax_s32_x(pg, a.sve_i32, b.sve_i32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_i32 = vmaxq_s32(a.neon_i32, b.neon_i32);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[i] > b_.i32[i] ? a_.i32[i] : b_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_max_epi32
  #define _mm_max_epi32(a, b) easysimd_mm_max_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_max_epu16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_max_epu16(a, b);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    /* https://github.com/simd-everywhere/simde/issues/855#issuecomment-881656284 */
    return _mm_add_epi16(b, _mm_subs_epu16(a, b));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16 = svmax_u16_z(pg, a.sve_u16, b.sve_u16);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_u16 = vmaxq_u16(a.neon_u16, b.neon_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u16 = vmaxq_u16(a_.neon_u16, b_.neon_u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = a_.u16[i] > b_.u16[i] ? a_.u16[i] : b_.u16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_max_epu16
  #define _mm_max_epu16(a, b) easysimd_mm_max_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_max_epu32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_max_epu32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u32 = svmax_u32_x(svptrue_b32(), a.sve_u32, b.sve_u32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_u32 = vmaxq_u32(a.neon_u32, b.neon_u32);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] > b_.u32[i] ? a_.u32[i] : b_.u32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_max_epu32
  #define _mm_max_epu32(a, b) easysimd_mm_max_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_min_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && !defined(__PGI)
    return _mm_min_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svmin_s8_x(svptrue_b8(), a.sve_i8, b.sve_i8);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_i8 = vminq_s8(a.neon_i8, b.neon_i8);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[i] < b_.i8[i] ? a_.i8[i] : b_.i8[i];
      }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_min_epi8
  #define _mm_min_epi8(a, b) easysimd_mm_min_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_min_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE) && !defined(__PGI)
    return _mm_min_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svmin_s32_x(svptrue_b32(), a.sve_i32, b.sve_i32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_i32 = vminq_s32(a.neon_i32, b.neon_i32);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
       r_.i32[i] = a_.i32[i] < b_.i32[i] ? a_.i32[i] : b_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_min_epi32
  #define _mm_min_epi32(a, b) easysimd_mm_min_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_min_epu16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_min_epu16(a, b);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    /* https://github.com/simd-everywhere/simde/issues/855#issuecomment-881656284 */
    return _mm_sub_epi16(a, _mm_subs_epu16(a, b));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16 = svmin_u16_z(pg, a.sve_u16, b.sve_u16);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_u16 = vminq_u16(a.neon_u16, b.neon_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u16 = vminq_u16(a_.neon_u16, b_.neon_u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = a_.u16[i] < b_.u16[i] ? a_.u16[i] : b_.u16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_min_epu16
  #define _mm_min_epu16(a, b) easysimd_mm_min_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_min_epu32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_min_epu32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u32 = svmin_u32_x(svptrue_b32(), a.sve_u32, b.sve_u32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_u32 = vminq_u32(a.neon_u32, b.neon_u32);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] < b_.u32[i] ? a_.u32[i] : b_.u32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_min_epu32
  #define _mm_min_epu32(a, b) easysimd_mm_min_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_minpos_epu16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_minpos_epu16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svdup_n_u16(0);
    svbool_t gp = svptrue_b16();
    uint16_t min = svminv_u16(gp, a.sve_u16);
    svbool_t pg = svcmpeq_n_u16(gp, a.sve_u16, min);
    r.i16[0] = min;
    r.u16[1] = svcntp_b16(gp, svbrkb_b_z(gp, pg));
    return r;
  #else
    easysimd__m128i_private
      r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128()),
      a_ = easysimd__m128i_to_private(a);

    r_.u16[0] = UINT16_MAX;
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      if (a_.u16[i] < r_.u16[0]) {
        r_.u16[0] = a_.u16[i];
        r_.u16[1] = HEDLEY_STATIC_CAST(uint16_t, i);
      }
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_minpos_epu16
  #define _mm_minpos_epu16(a) easysimd_mm_minpos_epu16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mpsadbw_epu8 (easysimd__m128i a, easysimd__m128i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  uint8_t a_offset = imm8 & 4;
  const int b_offset = (imm8 & 3) << 2;
  svuint8_t sva, svb;
  sveuint8_t svarr[4];
  svuint8_t svaindex = svdupq_n_u8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
            svaindex = svadd_u8_x(svptrue_b8(), svaindex, svdup_n_u8(a_offset));
  svuint8_t svbindex = svdup_n_u8(b_offset);
  for(int i = 0; i < 4; i++){
    sva = svtbl_u8(a.sve_u8, svaindex);
    svb = svtbl_u8(b.sve_u8, svbindex);
    svarr[i] = svabd_u8_x(svptrue_b8(), sva, svb);
    svaindex = svadd_u8_x(svptrue_b8(), svaindex, svdup_n_u8(1));
    svbindex = svadd_u8_x(svptrue_b8(), svbindex, svdup_n_u8(1));
  }
  r.sve_u16 = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[0], svarr[1]), svaddlb_u16(svarr[2], svarr[3]));
  return r;
#else
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  const int a_offset = imm8 & 4;
  const int b_offset = (imm8 & 3) << 2;

#if defined(easysimd_math_abs)
  for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, (sizeof(r_.u16) / sizeof(r_.u16[0]))) ; i++) {
    r_.u16[i] =
      HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[a_offset + i + 0] - b_.u8[b_offset + 0]))) +
      HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[a_offset + i + 1] - b_.u8[b_offset + 1]))) +
      HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[a_offset + i + 2] - b_.u8[b_offset + 2]))) +
      HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[a_offset + i + 3] - b_.u8[b_offset + 3])));
  }
#else
  HEDLEY_UNREACHABLE();
#endif

  return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE) && !defined(EASYSIMD_BUG_PGI_30107)
#  define easysimd_mm_mpsadbw_epu8(a, b, imm8) _mm_mpsadbw_epu8(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_mpsadbw_epu8
  #define _mm_mpsadbw_epu8(a, b, imm8) easysimd_mm_mpsadbw_epu8(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mul_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_mul_epi32(a, b);
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i64 = svmullb_s64(a.sve_i32, b.sve_i32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      // vmull_s32 upcasts instead of masking, so we downcast.
      int32x2_t a_lo = vmovn_s64(a.neon_i64);
      int32x2_t b_lo = vmovn_s64(b.neon_i64);
      r_.neon_i64 = vmull_s32(a_lo, b_lo);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] =
          HEDLEY_STATIC_CAST(int64_t, a_.i32[i * 2]) *
          HEDLEY_STATIC_CAST(int64_t, b_.i32[i * 2]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_mul_epi32
  #define _mm_mul_epi32(a, b) easysimd_mm_mul_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mullo_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_mullo_epi32(a, b);
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vmulq_s32(a.neon_i32, b.neon_i32);
    #elif defined(EASYSIMD_ARM_SVE_NATIVE)
      svbool_t pg = svptrue_b32();
      r_.sve_i32 = svmul_s32_z(pg, a.sve_i32, b.sve_i32);
    #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (HEDLEY_STATIC_CAST(uint64_t, (HEDLEY_STATIC_CAST(int64_t, a_.i32[i]) * HEDLEY_STATIC_CAST(int64_t, b_.i32[i]))) & 0xffffffff));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_mullo_epi32
  #define _mm_mullo_epi32(a, b) easysimd_mm_mullo_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_mullo_epu32 (easysimd__m128i a, easysimd__m128i b) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vmulq_u32(a_.neon_u32, b_.neon_u32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u32 = a_.u32 * b_.u32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i] * b_.u32[i];
      }
    #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_packus_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_packus_epi32(a, b);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    const __m128i max = _mm_set1_epi32(UINT16_MAX);
    const __m128i tmpa = _mm_andnot_si128(_mm_srai_epi32(a, 31), a);
    const __m128i tmpb = _mm_andnot_si128(_mm_srai_epi32(b, 31), b);
    return
      _mm_packs_epi32(
        _mm_srai_epi32(_mm_slli_epi32(_mm_or_si128(tmpa, _mm_cmpgt_epi32(tmpa, max)), 16), 16),
        _mm_srai_epi32(_mm_slli_epi32(_mm_or_si128(tmpb, _mm_cmpgt_epi32(tmpb, max)), 16), 16)
      );
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svuzp1_u16(svqxtunb_s32(a.sve_i32), svqxtunb_s32(b.sve_i32));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      #if defined(EASYSIMD_BUG_CLANG_46840)
        r_.neon_u16 = vqmovun_high_s32(vreinterpret_s16_u16(vqmovun_s32(a_.neon_i32)), b_.neon_i32);
      #else
        r_.neon_u16 = vqmovun_high_s32(vqmovun_s32(a_.neon_i32), b_.neon_i32);
      #endif
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u16 =
        vcombine_u16(
          vqmovun_s32(a_.neon_i32),
          vqmovun_s32(b_.neon_i32)
        );
    #elif defined(EASYSIMD_CONVERT_VECTOR_) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      int32_t v EASYSIMD_VECTOR(32) = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.i32, b_.i32, 0, 1, 2, 3, 4, 5, 6, 7);

      v &= ~(v >> 31);
      v |= HEDLEY_REINTERPRET_CAST(__typeof__(v), v > UINT16_MAX);

      EASYSIMD_CONVERT_VECTOR_(r_.i16, v);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        int32_t v = (i < (sizeof(a_.i32) / sizeof(a_.i32[0]))) ? a_.i32[i] : b_.i32[i & 3];
        r_.u16[i] = (v < 0) ? UINT16_C(0) : ((v > UINT16_MAX) ? UINT16_MAX : HEDLEY_STATIC_CAST(uint16_t, v));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_packus_epi32
  #define _mm_packus_epi32(a, b) easysimd_mm_packus_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_round_sd (easysimd__m128d a, easysimd__m128d b, int rounding)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(rounding, 0, 15) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    b.sve_f64 = easysimdfunlistroundf64[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f64(svptrue_b64(), b.sve_f64);
    r.sve_f64 = svdupq_n_f64(b.f64[0], a.f64[1]);
    return r;
  #else
    easysimd__m128d_private
      r_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(easysimd_math_nearbyint)
        case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        case EASYSIMD_MM_FROUND_CUR_DIRECTION:
          r_.f64[0] = easysimd_math_nearbyint(b_.f64[0]);
          break;
      #endif

      #if defined(easysimd_math_floor)
        case EASYSIMD_MM_FROUND_TO_NEG_INF:
          r_.f64[0] = easysimd_math_floor(b_.f64[0]);
          break;
      #endif

      #if defined(easysimd_math_ceil)
        case EASYSIMD_MM_FROUND_TO_POS_INF:
          r_.f64[0] = easysimd_math_ceil(b_.f64[0]);
          break;
      #endif

      #if defined(easysimd_math_trunc)
        case EASYSIMD_MM_FROUND_TO_ZERO:
          r_.f64[0] = easysimd_math_trunc(b_.f64[0]);
          break;
      #endif

      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
#  define easysimd_mm_round_sd(a, b, rounding) _mm_round_sd(a, b, rounding)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && defined(EASYSIMD_FAST_EXCEPTIONS)
#  define easysimd_mm_round_sd(a, b, rounding) easysimd_mm_move_sd(a, easysimd_mm_round_pd(b, rounding))
#elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
  #define easysimd_mm_round_sd(a, b, rounding) easysimd_mm_move_sd(a, easysimd_mm_round_pd(easysimd_x_mm_broadcastlow_pd(b), rounding))
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_round_sd
  #define _mm_round_sd(a, b, rounding) easysimd_mm_round_sd(a, b, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_round_ss (easysimd__m128 a, easysimd__m128 b, int rounding)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(rounding, 0, 15) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    b.sve_f32 = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(svptrue_b32(), b.sve_f32);
    a.f32[0] = b.f32[0];
    return a;
  #else
    easysimd__m128_private
      r_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC) {
      #if defined(easysimd_math_nearbyintf)
        case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        case EASYSIMD_MM_FROUND_CUR_DIRECTION:
          r_.f32[0] = easysimd_math_nearbyintf(b_.f32[0]);
          break;
      #endif

      #if defined(easysimd_math_floorf)
        case EASYSIMD_MM_FROUND_TO_NEG_INF:
          r_.f32[0] = easysimd_math_floorf(b_.f32[0]);
          break;
      #endif

      #if defined(easysimd_math_ceilf)
        case EASYSIMD_MM_FROUND_TO_POS_INF:
          r_.f32[0] = easysimd_math_ceilf(b_.f32[0]);
          break;
      #endif

      #if defined(easysimd_math_truncf)
        case EASYSIMD_MM_FROUND_TO_ZERO:
          r_.f32[0] = easysimd_math_truncf(b_.f32[0]);
          break;
      #endif

      default:
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #define easysimd_mm_round_ss(a, b, rounding) _mm_round_ss(a, b, rounding)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0 && defined(EASYSIMD_FAST_EXCEPTIONS)
  #define easysimd_mm_round_ss(a, b, rounding) easysimd_mm_move_ss((a), easysimd_mm_round_ps((b), (rounding)))
#elif EASYSIMD_NATURAL_VECTOR_SIZE > 0
  #define easysimd_mm_round_ss(a, b, rounding) easysimd_mm_move_ss((a), easysimd_mm_round_ps(easysimd_x_mm_broadcastlow_ps(b), (rounding)))
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_round_ss
  #define _mm_round_ss(a, b, rounding) easysimd_mm_round_ss(a, b, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_stream_load_si128 (const easysimd__m128i* mem_addr) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_stream_load_si128(HEDLEY_CONST_CAST(easysimd__m128i*, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1_s32(svptrue_b32(), (HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr)));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return *mem_addr;
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128i r;
    r.neon_i64 = vreinterpretq_s64_s32(vld1q_s32(HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr)));
    return r;
  #else
    return *mem_addr;
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_stream_load_si128
  #define _mm_stream_load_si128(mem_addr) easysimd_mm_stream_load_si128(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_test_all_ones (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_test_all_ones(a);
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      return r = ((vgetq_lane_s64(a_.neon_i64, 0) & vgetq_lane_s64(a_.neon_i64, 1)) == ~HEDLEY_STATIC_CAST(int64_t, 0));
    #else
      int_fast32_t r_ = ~HEDLEY_STATIC_CAST(int_fast32_t, 0);

      EASYSIMD_VECTORIZE_REDUCTION(&:r_)
      for (size_t i = 0 ; i < (sizeof(a_.i32f) / sizeof(a_.i32f[0])) ; i++) {
        r_ &= a_.i32f[i];
      }

      r = (r_ == ~HEDLEY_STATIC_CAST(int_fast32_t, 0));
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_test_all_ones
  #define _mm_test_all_ones(a) easysimd_mm_test_all_ones(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_test_all_zeros (easysimd__m128i a, easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_test_all_zeros(a, mask);
  #else
    easysimd__m128i_private tmp_ = easysimd__m128i_to_private(easysimd_mm_and_si128(a, mask));
    int r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      return !(vgetq_lane_s64(tmp_.neon_i64, 0) | vgetq_lane_s64(tmp_.neon_i64, 1));
    #else
      int_fast32_t r_ = HEDLEY_STATIC_CAST(int_fast32_t, 0);

      EASYSIMD_VECTORIZE_REDUCTION(|:r_)
      for (size_t i = 0 ; i < (sizeof(tmp_.i32f) / sizeof(tmp_.i32f[0])) ; i++) {
        r_ |= tmp_.i32f[i];
      }

      r = !r_;
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_test_all_zeros
  #define _mm_test_all_zeros(a, mask) easysimd_mm_test_all_zeros(a, mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_test_mix_ones_zeros (easysimd__m128i a, easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_test_mix_ones_zeros(a, mask);
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      mask_ = easysimd__m128i_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int64x2_t s640 = vandq_s64(a_.neon_i64, mask_.neon_i64);
      int64x2_t s641 = vandq_s64(vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(a_.neon_i64))), mask_.neon_i64);
      return (((vgetq_lane_s64(s640, 0) | vgetq_lane_s64(s640, 1)) & (vgetq_lane_s64(s641, 0) | vgetq_lane_s64(s641, 1)))!=0);
    #else
      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++)
        if (((a_.u64[i] & mask_.u64[i]) != 0) && ((~a_.u64[i] & mask_.u64[i]) != 0))
          return 1;

      return 0;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_test_mix_ones_zeros
  #define _mm_test_mix_ones_zeros(a, mask) easysimd_mm_test_mix_ones_zeros(a, mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_testc_si128 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_testc_si128(a, b);
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int64x2_t s64 = vbicq_s64(b_.neon_i64, a_.neon_i64);
      return !(vgetq_lane_s64(s64, 0) & vgetq_lane_s64(s64, 1));
    #else
      int_fast32_t r = 0;

      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i32f) / sizeof(a_.i32f[0])) ; i++) {
        r |= ~a_.i32f[i] & b_.i32f[i];
      }

      return HEDLEY_STATIC_CAST(int, !r);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_testc_si128
  #define _mm_testc_si128(a, b) easysimd_mm_testc_si128(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_testnzc_si128 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_testnzc_si128(a, b);
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int64x2_t s640 = vandq_s64(b_.neon_i64, a_.neon_i64);
      int64x2_t s641 = vbicq_s64(b_.neon_i64, a_.neon_i64);
      return (((vgetq_lane_s64(s640, 0) | vgetq_lane_s64(s640, 1)) & (vgetq_lane_s64(s641, 0) | vgetq_lane_s64(s641, 1)))!=0);
    #else
      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
        if (((a_.u64[i] & b_.u64[i]) != 0) && ((~a_.u64[i] & b_.u64[i]) != 0))
          return 1;
      }

      return 0;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_testnzc_si128
  #define _mm_testnzc_si128(a, b) easysimd_mm_testnzc_si128(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_testz_si128 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_testz_si128(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    uint64_t ret = 0;
    ret += svaddv_s64(pg, svand_s64_x(pg, a.sve_i64, b.sve_i64));
    return ret == 0 ? 1 : 0;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    int64x2_t s64 = vandq_s64(a.neon_i64, b.neon_i64);
    return !(vgetq_lane_s64(s64, 0) | vgetq_lane_s64(s64, 1));
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
        if ((a_.u64[i] & b_.u64[i]) == 0)
          return 1;
      }

    return 0;
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #undef _mm_testz_si128
  #define _mm_testz_si128(a, b) easysimd_mm_testz_si128(a, b)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_SSE4_1_H) */
