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
 *   2015-2017 John W. Ratcliff <jratcliffscarab@gmail.com>
 *   2015      Brandon Rowlett <browlett@nvidia.com>
 *   2015      Ken Fast <kfast@gdeb.com>
 *   2017      Hasindu Gamaarachchi <hasindu@unsw.edu.au>
 *   2018      Jeff Daily <jeff.daily@amd.com>
 */

#if !defined(EASYSIMD_X86_SSE2_H)
#define EASYSIMD_X86_SSE2_H

#include "sse.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

typedef easysimd__m128_private  easysimd__m128i_private;
typedef easysimd__m128_private  easysimd__m128d_private;

#if defined(EASYSIMD_X86_SSE2_NATIVE)
  typedef __m128i easysimd__m128i;
  typedef __m128d easysimd__m128d;
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(EASYSIMD_CONVERT_TO_PRIVATE)
    typedef int64x2_t easysimd__m128i;
  #else
    typedef easysimd__m128i_private easysimd__m128i;
  #endif
#  if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    #if defined(EASYSIMD_CONVERT_TO_PRIVATE)
      typedef float64x2_t easysimd__m128d;
    #else
      typedef easysimd__m128d_private easysimd__m128d;
    #endif
#  elif defined(EASYSIMD_VECTOR_SUBSCRIPT)
     typedef easysimd_float64 easysimd__m128d EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
#  else
     typedef easysimd__m128d_private easysimd__m128d;
#  endif

#elif defined(EASYSIMD_VECTOR_SUBSCRIPT)
  #if defined(EASYSIMD_CONVERT_TO_PRIVATE)
    typedef int64_t easysimd__m128i EASYSIMD_ALIGN_TO_16 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    typedef easysimd_float64 easysimd__m128d EASYSIMD_ALIGN_TO_16 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
  #else
    typedef easysimd__m128i_private easysimd__m128i;
    typedef easysimd__m128d_private easysimd__m128d;
  #endif
#else
  typedef easysimd__m128i_private easysimd__m128i;
  typedef easysimd__m128d_private easysimd__m128d;
#endif

#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  typedef easysimd__m128i __m128i;
  typedef easysimd__m128d __m128d;
#endif

#if defined(EASYSIMD_ARM_SVE_NATIVE)
static easysimd__m128i mask8l __attribute__((unused)) = {
    .u8 = {1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0,  0,  0,  0,   0}
};

static easysimd__m128i mask8h __attribute__((unused)) = {
    .u8 = {0, 0, 0, 0,  0,  0,  0,   0, 1, 2, 4, 8, 16, 32, 64, 128}
};

static easysimd__m128i mask16 __attribute__((unused)) = {
    .u16 = {1, 2, 4, 8, 16, 32, 64, 128}
};

static easysimd__m128i mask32 __attribute__((unused)) = {
    .u32 = {1, 2, 4, 8}
};

static easysimd__m128i mask64 __attribute__((unused)) = {
    .u64 = {1, 2}
};
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_sse2_cmp_f64_e(easysimd_float64 a, easysimd_float64 b) {
    return fabs(a - b) < 1e-9;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_sse2_cmp_f64_le(easysimd_float64 a, easysimd_float64 b) {
    return a < b || fabs(a - b) < 1e-9;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_sse2_cmp_f64_ge(easysimd_float64 a, easysimd_float64 b) {
    return a > b || fabs(a - b) < 1e-9;
}

HEDLEY_STATIC_ASSERT(16 == sizeof(easysimd__m128i), "easysimd__m128i size incorrect");
HEDLEY_STATIC_ASSERT(16 == sizeof(easysimd__m128i_private), "easysimd__m128i_private size incorrect");
HEDLEY_STATIC_ASSERT(16 == sizeof(easysimd__m128d), "easysimd__m128d size incorrect");
HEDLEY_STATIC_ASSERT(16 == sizeof(easysimd__m128d_private), "easysimd__m128d_private size incorrect");
#if defined(EASYSIMD_CHECK_ALIGNMENT) && defined(EASYSIMD_ALIGN_OF)
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m128i) == 16, "easysimd__m128i is not 16-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m128i_private) == 16, "easysimd__m128i_private is not 16-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m128d) == 16, "easysimd__m128d is not 16-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m128d_private) == 16, "easysimd__m128d_private is not 16-byte aligned");
#endif

#if defined(EASYSIMD_CONVERT_TO_PRIVATE) || defined(EASYSIMD_X86_SSE2_NATIVE)
EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd__m128i_from_private(easysimd__m128i_private v) {
  easysimd__m128i r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i_private
easysimd__m128i_to_private(easysimd__m128i v) {
  easysimd__m128i_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd__m128d_from_private(easysimd__m128d_private v) {
  easysimd__m128d r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d_private
easysimd__m128d_to_private(easysimd__m128d v) {
  easysimd__m128d_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

#else
#define easysimd__m128i_from_private(v) v
#define easysimd__m128i_to_private(v) v
#define easysimd__m128d_from_private(v) v
#define easysimd__m128d_to_private(v) v
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, int8x16_t, neon, i8)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, int16x8_t, neon, i16)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, int32x4_t, neon, i32)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, int64x2_t, neon, i64)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, uint8x16_t, neon, u8)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, uint16x8_t, neon, u16)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, uint32x4_t, neon, u32)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, uint64x2_t, neon, u64)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, float32x4_t, neon, f32)
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128i, float64x2_t, neon, f64)
  #endif
#endif /* defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) */

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, int8x16_t, neon, i8)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, int16x8_t, neon, i16)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, int32x4_t, neon, i32)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, int64x2_t, neon, i64)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, uint8x16_t, neon, u8)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, uint16x8_t, neon, u16)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, uint32x4_t, neon, u32)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, uint64x2_t, neon, u64)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, float32x4_t, neon, f32)
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128d, float64x2_t, neon, f64)
  #endif
#endif /* defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) */

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_set_pd (easysimd_float64 e1, easysimd_float64 e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set_pd(e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdupq_n_f64(e0, e1);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    EASYSIMD_ALIGN_TO_16 easysimd_float64 data[2] = { e0, e1 };
    r.neon_f64 = vld1q_f64(data);
    return r;
  #else
    easysimd__m128d_private r_;

    r_.f64[0] = e0;
    r_.f64[1] = e1;

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set_pd(e1, e0) easysimd_mm_set_pd(e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_set1_pd (easysimd_float64 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set1_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdup_n_f64(a);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128d r;
    r.neon_f64 = vdupq_n_f64(a);
    return r;
  #else
    easysimd__m128d_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.f64[i] = a;
    }
    return easysimd__m128d_from_private(r_);
  #endif
}
#define easysimd_mm_set_pd1(a) easysimd_mm_set1_pd(a)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set1_pd(a) easysimd_mm_set1_pd(a)
  #define _mm_set_pd1(a) easysimd_mm_set1_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_abs_pd(easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    easysimd_float64 mask_;
    uint64_t u64_ = UINT64_C(0x7FFFFFFFFFFFFFFF);
    easysimd_memcpy(&mask_, &u64_, sizeof(u64_));
    return _mm_and_pd(_mm_set1_pd(mask_), a);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vabsq_f64(a_.neon_f64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_fabs(a_.f64[i]);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_not_pd(easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    __m128i ai = _mm_castpd_si128(a);
    return _mm_castsi128_pd(_mm_ternarylogic_epi64(ai, ai, ai, 0x55));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vmvnq_s32(a_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = ~a_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = ~(a_.i32f[i]);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_select_pd(easysimd__m128d a, easysimd__m128d b, easysimd__m128d mask) {
  /* This function is for when you want to blend two elements together
   * according to a mask.  It is similar to _mm_blendv_pd, except that
   * it is undefined whether the blend is based on the highest bit in
   * each lane (like blendv) or just bitwise operations.  This allows
   * us to implement the function efficiently everywhere.
   *
   * Basically, you promise that all the lanes in mask are either 0 or
   * ~0. */
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_blendv_pd(a, b, mask);
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b),
      mask_ = easysimd__m128d_to_private(mask);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = a_.i64 ^ ((a_.i64 ^ b_.i64) & mask_.i64);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vbslq_s64(mask_.neon_u64, b_.neon_i64, a_.neon_i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] ^ ((a_.i64[i] ^ b_.i64[i]) & mask_.i64[i]);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_add_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_add_epi8(a, b);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i8 = svadd_s8_z(svptrue_b8(), a.sve_i8, b.sve_i8);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i8 = vaddq_s8(a.neon_i8, b.neon_i8);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      r_.i8 = a_.i8 + b_.i8;
    #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[i] + b_.i8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_add_epi8(a, b) easysimd_mm_add_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_add_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_add_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svadd_s16_z(svptrue_b16(), a.sve_i16, b.sve_i16);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i16 = vaddq_s16(a.neon_i16, b.neon_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i16 = vaddq_s16(a_.neon_i16, b_.neon_i16);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = a_.i16 + b_.i16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i16[i] + b_.i16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_add_epi16(a, b) easysimd_mm_add_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_add_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_add_epi32(a, b);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i32 = svadd_s32_z(svptrue_b32(), a.sve_i32, b.sve_i32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vaddq_s32(a.neon_i32, b.neon_i32);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] + b_.i32[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_add_epi32(a, b) easysimd_mm_add_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_add_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_add_epi64(a, b);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      svbool_t pg = svptrue_b64();
      r_.sve_i64 = svadd_s64_z(pg, a.sve_i64, b.sve_i64);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vaddq_s64(a.neon_i64, b.neon_i64);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] + b_.i64[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_add_epi64(a, b) easysimd_mm_add_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_add_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_add_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svadd_f64_z(svptrue_b64(), a.sve_f64, b.sve_f64);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    a.neon_f64 = vaddq_f64(a.neon_f64, b.neon_f64);
    return a;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 + b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[i] + b_.f64[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_add_pd(a, b) easysimd_mm_add_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_move_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_move_sd(a, b);
  #elif (defined(EASYSIMD_ARM_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
    easysimd__m128d r;
    r.neon_f64 = vsetq_lane_f64(vgetq_lane_f64(b.neon_f64, 0), a.neon_f64, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f64[0] = b.f64[0];
    return a;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.f64, b_.f64, 2, 1);
    #else
      r_.f64[0] = b_.f64[0];
      r_.f64[1] = a_.f64[1];
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_move_sd(a, b) easysimd_mm_move_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_broadcastlow_pd(easysimd__m128d a) {
  /* This function broadcasts the first element in the input vector to
   * all lanes.  It is used to avoid generating spurious exceptions in
   * *_sd functions since there may be garbage in the upper lanes. */

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_castsi128_pd(_mm_shuffle_epi32(_mm_castpd_si128(a), 0x44));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vdupq_laneq_f64(a_.neon_f64, 0);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.f64, a_.f64, 0, 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[0];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_add_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_add_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svadd_f64_z(svptrue_b64(), a.sve_f64, svdupq_n_f64(b.f64[0], 0.0));
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_add_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_add_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    r_.f64[0] = a_.f64[0] + b_.f64[0];
    r_.f64[1] = a_.f64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_add_sd(a, b) easysimd_mm_add_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_add_si64 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_add_si64(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_i64 = vadd_s64(a.neon_i64, b.neon_i64);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    r_.i64[0] = a_.i64[0] + b_.i64[0];

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_add_si64(a, b) easysimd_mm_add_si64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_adds_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_adds_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svqadd_s8(a.sve_i8, b.sve_i8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i8 = vqaddq_s8(a_.neon_i8, b_.neon_i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = easysimd_math_adds_i8(a_.i8[i], b_.i8[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_adds_epi8(a, b) easysimd_mm_adds_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_adds_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_adds_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svqadd_s16(a.sve_i16, b.sve_i16);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
    easysimd__m128i r;
    r.neon_i16 = vqaddq_s16(a.neon_i16, b.neon_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = easysimd_math_adds_i16(a_.i16[i], b_.i16[i]);
    }
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_adds_epi16(a, b) easysimd_mm_adds_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_adds_epu8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_adds_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svqadd_u8(a.sve_u8, b.sve_u8);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_A64V8_NATIVE))
    easysimd__m128i r;
    r.neon_u8 = vqaddq_u8(a.neon_u8, b.neon_u8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = easysimd_math_adds_u8(a_.u8[i], b_.u8[i]);
    }
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_adds_epu8(a, b) easysimd_mm_adds_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_adds_epu16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_adds_epu16(a, b);
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svqadd_u16_x(svptrue_b16(), a.sve_u16, b.sve_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u16 = vqaddq_u16(a_.neon_u16, b_.neon_u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = easysimd_math_adds_u16(a_.u16[i], b_.u16[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_adds_epu16(a, b) easysimd_mm_adds_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_and_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_and_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u32 = svand_u32_z(svptrue_b32(), a.sve_u32, b.sve_u32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    a.neon_i32 = vandq_s32(a.neon_i32, b.neon_i32);
    return a;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f & b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = a_.i32f[i] & b_.i32f[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_and_pd(a, b) easysimd_mm_and_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_and_si128 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_and_si128(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svand_s32_x(svptrue_b32(), a.sve_i32, b.sve_i32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i32 = vandq_s32(b.neon_i32, a.neon_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f & b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = a_.i32f[i] & b_.i32f[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_and_si128(a, b) easysimd_mm_and_si128(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_andnot_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_andnot_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svbic_u64_z(svptrue_b64(), b.sve_u64, a.sve_u64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vbicq_s32(b_.neon_i32, a_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = ~a_.i32f & b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = ~a_.u64[i] & b_.u64[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_andnot_pd(a, b) easysimd_mm_andnot_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_andnot_si128 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_andnot_si128(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svbic_s32_x(svptrue_b32(), b.sve_i32, a.sve_i32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_i32 = vbicq_s32(b.neon_i32, a.neon_i32);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = ~a_.i32f & b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = ~(a_.i32f[i]) & b_.i32f[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_andnot_si128(a, b) easysimd_mm_andnot_si128(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_xor_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_xor_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_i64 = sveor_s64_z(svptrue_b64(), a.sve_i64, b.sve_i64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f ^ b_.i32f;
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = veorq_s64(a_.neon_i64, b_.neon_i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = a_.i32f[i] ^ b_.i32f[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_xor_pd(a, b) easysimd_mm_xor_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_avg_epu8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_avg_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svrhadd_u8_z(svptrue_b8(), a.sve_u8, b.sve_u8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u8 = vrhaddq_u8(b_.neon_u8, a_.neon_u8);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && defined(EASYSIMD_CONVERT_VECTOR_)
      uint16_t wa EASYSIMD_VECTOR(32);
      uint16_t wb EASYSIMD_VECTOR(32);
      uint16_t wr EASYSIMD_VECTOR(32);
      EASYSIMD_CONVERT_VECTOR_(wa, a_.u8);
      EASYSIMD_CONVERT_VECTOR_(wb, b_.u8);
      wr = (wa + wb + 1) >> 1;
      EASYSIMD_CONVERT_VECTOR_(r_.u8, wr);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = (a_.u8[i] + b_.u8[i] + 1) >> 1;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_avg_epu8(a, b) easysimd_mm_avg_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_avg_epu16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_avg_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svrhadd_u16_z(svptrue_b16(), a.sve_u16, b.sve_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u16 = vrhaddq_u16(b_.neon_u16, a_.neon_u16);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && defined(EASYSIMD_CONVERT_VECTOR_)
      uint32_t wa EASYSIMD_VECTOR(32);
      uint32_t wb EASYSIMD_VECTOR(32);
      uint32_t wr EASYSIMD_VECTOR(32);
      EASYSIMD_CONVERT_VECTOR_(wa, a_.u16);
      EASYSIMD_CONVERT_VECTOR_(wb, b_.u16);
      wr = (wa + wb + 1) >> 1;
      EASYSIMD_CONVERT_VECTOR_(r_.u16, wr);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = (a_.u16[i] + b_.u16[i] + 1) >> 1;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_avg_epu16(a, b) easysimd_mm_avg_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_setzero_si128 (void) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_setzero_si128();
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svdup_n_s32(0);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i32 = vdupq_n_s32(0);
    return r;
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vdupq_n_s32(0);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT)
      r_.i32 = __extension__ (__typeof__(r_.i32)) { 0, 0, 0, 0 };
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = 0;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_setzero_si128() (easysimd_mm_setzero_si128())
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_bslli_si128 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  r.sve_i8 = svsplice_s8(svwhilelt_b8(0, imm8), svdup_n_s8(0), a.sve_i8);
  return r;
#elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
	easysimd__m128i res;
	if (imm8 > 0 && imm8 <= 15) {
		int8x16_t zero = vdupq_n_s8(0);
		__asm__ __volatile__ (
			"ext %0.16b, %1.16b, %2.16b, #%3"
			: "=w"(res.neon_i8)
			: "w"(zero), "w"(a.neon_i8), "i"(16 - imm8)
			: /*No clobbers */);
	} else if (imm8 == 0) {
		res = a;
	} else {
		res.neon_i8 = vdupq_n_s8(0);
	}
	return res;
#else
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a);

  if (HEDLEY_UNLIKELY((imm8 & ~15))) {
    return easysimd_mm_setzero_si128();
  }

  #if defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
    r_.altivec_i8 = vec_srb(a_.altivec_i8, vec_splats(HEDLEY_STATIC_CAST(unsigned char, (imm8 & 15) << 3)));
  #elif defined(EASYSIMD_HAVE_INT128_) && (EASYSIMD_ENDIAN_ORDER == EASYSIMD_ENDIAN_LITTLE)
    r_.u128[0] = a_.u128[0] << (imm8 * 8);
  #else
    r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128());
    for (int i = imm8 ; i < HEDLEY_STATIC_CAST(int, sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = a_.i8[i - imm8];
    }
  #endif

  return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
  #define easysimd_mm_bslli_si128(a, imm8) _mm_slli_si128(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(__clang__)
  #define easysimd_mm_bslli_si128(a, imm8) \
  easysimd__m128i_from_neon_i8(((imm8) <= 0) ? easysimd__m128i_to_neon_i8(a) : (((imm8) > 15) ? (vdupq_n_s8(0)) : (vextq_s8(vdupq_n_s8(0), easysimd__m128i_to_neon_i8(a), 16 - (imm8)))))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_bslli_si128(a, imm8) (__extension__ ({ \
    const easysimd__m128i_private easysimd__tmp_a_ = easysimd__m128i_to_private(a); \
    const easysimd__m128i_private easysimd__tmp_z_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128()); \
    easysimd__m128i_private easysimd__tmp_r_; \
    if (HEDLEY_UNLIKELY(imm8 > 15)) { \
      easysimd__tmp_r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128()); \
    } else { \
      easysimd__tmp_r_.i8 = \
        EASYSIMD_SHUFFLE_VECTOR_(8, 16, \
          easysimd__tmp_z_.i8, \
          (easysimd__tmp_a_).i8, \
          HEDLEY_STATIC_CAST(int8_t, (16 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (17 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (18 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (19 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (20 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (21 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (22 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (23 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (24 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (25 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (26 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (27 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (28 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (29 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (30 - imm8) & 31), \
          HEDLEY_STATIC_CAST(int8_t, (31 - imm8) & 31)); \
    } \
    easysimd__m128i_from_private(easysimd__tmp_r_); }))
#endif
#define easysimd_mm_slli_si128(a, imm8) easysimd_mm_bslli_si128(a, imm8)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_bslli_si128(a, imm8) easysimd_mm_bslli_si128(a, imm8)
  #define _mm_slli_si128(a, imm8) easysimd_mm_bslli_si128(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_bsrli_si128 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsplice_s8(svwhilele_b8(0, 15 - imm8),
                           svtbl_s8(a.sve_i8, svindex_u8(imm8, 1)),
                           svdup_n_s8(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    if (HEDLEY_UNLIKELY((imm8 & ~15))) {
      return easysimd_mm_setzero_si128();
    }

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      const int e = HEDLEY_STATIC_CAST(int, i) + imm8;
      r_.i8[i] = (e < 16) ? a_.i8[e] : 0;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
  #define easysimd_mm_bsrli_si128(a, imm8) _mm_srli_si128(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm_bsrli_si128(a, imm8) \
    easysimd__m128i_from_neon_i8( \
      ((imm8 < 0) || (imm8 > 15)) ? \
        vdupq_n_s8(0) : \
        vextq_s8(a.neon_i8, vdupq_n_s8(0), (imm8)))
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(__clang__)
  #define easysimd_mm_bsrli_si128(a, imm8) \
  easysimd__m128i_from_neon_i8(((imm8 < 0) || (imm8 > 15)) ? vdupq_n_s8(0) : (vextq_s8(easysimd__m128i_to_private(a).neon_i8, vdupq_n_s8(0), ((imm8 & 15) != 0) ? imm8 : (imm8 & 15))))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_bsrli_si128(a, imm8) (__extension__ ({ \
    const easysimd__m128i_private easysimd__tmp_a_ = easysimd__m128i_to_private(a); \
    const easysimd__m128i_private easysimd__tmp_z_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128()); \
    easysimd__m128i_private easysimd__tmp_r_ = easysimd__m128i_to_private(a); \
    if (HEDLEY_UNLIKELY(imm8 > 15)) { \
      easysimd__tmp_r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128()); \
    } else { \
      easysimd__tmp_r_.i8 = \
      EASYSIMD_SHUFFLE_VECTOR_(8, 16, \
        easysimd__tmp_z_.i8, \
        (easysimd__tmp_a_).i8, \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 16) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 17) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 18) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 19) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 20) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 21) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 22) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 23) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 24) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 25) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 26) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 27) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 28) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 29) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 30) & 31), \
        HEDLEY_STATIC_CAST(int8_t, (imm8 + 31) & 31)); \
    } \
    easysimd__m128i_from_private(easysimd__tmp_r_); }))
#endif
#define easysimd_mm_srli_si128(a, imm8) easysimd_mm_bsrli_si128((a), (imm8))
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_bsrli_si128(a, imm8) easysimd_mm_bsrli_si128((a), (imm8))
  #define _mm_srli_si128(a, imm8) easysimd_mm_bsrli_si128((a), (imm8))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_clflush (void const* p) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_clflush(p);
  #else
    (void) p;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_clflush(a, b) easysimd_mm_clflush()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comieq_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_comieq_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return easysimd_sse2_cmp_f64_e(a.f64[0], b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      return !!vgetq_lane_u64(vceqq_f64(a_.neon_f64, b_.neon_f64), 0);
    #else
    return easysimd_sse2_cmp_f64_e(a_.f64[0], b_.f64[0]);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_comieq_sd(a, b) easysimd_mm_comieq_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comige_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_comige_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return easysimd_sse2_cmp_f64_ge(a.f64[0], b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      return !!vgetq_lane_u64(vcgeq_f64(a_.neon_f64, b_.neon_f64), 0);
    #else
      return easysimd_sse2_cmp_f64_ge(a_.f64[0], b_.f64[0]);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_comige_sd(a, b) easysimd_mm_comige_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comigt_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_comigt_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.f64[0] > b.f64[0];
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      return !!vgetq_lane_u64(vcgtq_f64(a_.neon_f64, b_.neon_f64), 0);
    #else
      return a_.f64[0] > b_.f64[0];
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_comigt_sd(a, b) easysimd_mm_comigt_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comile_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_comile_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return easysimd_sse2_cmp_f64_le(a.f64[0], b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      return !!vgetq_lane_u64(vcleq_f64(a_.neon_f64, b_.neon_f64), 0);
    #else
      return easysimd_sse2_cmp_f64_le(a_.f64[0], b_.f64[0]);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_comile_sd(a, b) easysimd_mm_comile_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comilt_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_comilt_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.f64[0] < b.f64[0];
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      return !!vgetq_lane_u64(vcltq_f64(a_.neon_f64, b_.neon_f64), 0);
    #else
      return a_.f64[0] < b_.f64[0];
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_comilt_sd(a, b) easysimd_mm_comilt_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comineq_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_comineq_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return !easysimd_sse2_cmp_f64_e(a.f64[0], b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      return !vgetq_lane_u64(vceqq_f64(a_.neon_f64, b_.neon_f64), 0);
    #else
    return !easysimd_sse2_cmp_f64_e(a_.f64[0], b_.f64[0]);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_comineq_sd(a, b) easysimd_mm_comineq_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_copysign_pd(easysimd__m128d dest, easysimd__m128d src) {
  easysimd__m128d_private
    r_,
    dest_ = easysimd__m128d_to_private(dest),
    src_ = easysimd__m128d_to_private(src);

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint64x2_t sign_pos = vreinterpretq_u64_f64(vdupq_n_f64(-EASYSIMD_FLOAT64_C(0.0)));
    #else
      easysimd_float64 dbl_nz = -EASYSIMD_FLOAT64_C(0.0);
      uint64_t u64_nz;
      easysimd_memcpy(&u64_nz, &dbl_nz, sizeof(u64_nz));
      uint64x2_t sign_pos = vdupq_n_u64(u64_nz);
    #endif
    r_.neon_u64 = vbslq_u64(sign_pos, src_.neon_u64, dest_.neon_u64);
  #elif defined(easysimd_math_copysign)
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_copysign(dest_.f64[i], src_.f64[i]);
    }
  #else
    easysimd__m128d sgnbit = easysimd_mm_set1_pd(-EASYSIMD_FLOAT64_C(0.0));
    return easysimd_mm_xor_pd(easysimd_mm_and_pd(sgnbit, src), easysimd_mm_andnot_pd(sgnbit, dest));
  #endif

  return easysimd__m128d_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_xorsign_pd(easysimd__m128d dest, easysimd__m128d src) {
  return easysimd_mm_xor_pd(easysimd_mm_and_pd(easysimd_mm_set1_pd(-0.0), src), dest);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_castpd_ps (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_castpd_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svreinterpret_f32_f64(a.sve_f64);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r_;
    r_.neon_f32 = vreinterpretq_f32_f64(a.neon_f64);
    return r_;
  #else
    easysimd__m128 r;
    easysimd_memcpy(&r, &a, sizeof(a));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_castpd_ps(a) easysimd_mm_castpd_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_castpd_si128 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_castpd_si128(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (easysimd__m128d)a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r_;
    r_.neon_i64 = vreinterpretq_s64_f64(a.neon_f64);
    return r_;
  #else
    easysimd__m128i r;
    easysimd_memcpy(&r, &a, sizeof(a));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_castpd_si128(a) easysimd_mm_castpd_si128(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_castps_pd (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_castps_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svreinterpret_f64_f32(a.sve_f32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r_;
    r_.neon_f64 = vreinterpretq_f64_f32(a.neon_f32);
    return r_;
  #else
    easysimd__m128d r;
    easysimd_memcpy(&r, &a, sizeof(a));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_castps_pd(a) easysimd_mm_castps_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_castps_si128 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_castps_si128(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (easysimd__m128i)a;
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return easysimd__m128i_from_neon_i32(easysimd__m128_to_private(a).neon_i32);
  #else
    easysimd__m128i r;
    easysimd_memcpy(&r, &a, sizeof(a));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_castps_si128(a) easysimd_mm_castps_si128(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_castsi128_pd (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_castsi128_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svreinterpret_f64_s64(a.sve_i64);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r_;
    r_.neon_f64 = vreinterpretq_f64_s64(a.neon_i64);
    return r_;
  #else
    easysimd__m128d r;
    easysimd_memcpy(&r, &a, sizeof(a));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_castsi128_pd(a) easysimd_mm_castsi128_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_castsi128_ps (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_castsi128_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svreinterpret_f32_s32(a.sve_i32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128 res;
    res.neon_f32 = vreinterpretq_f32_s32(a.neon_i32);
    return res;
  #else
    easysimd__m128 r;
    easysimd_memcpy(&r, &a, sizeof(a));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_castsi128_ps(a) easysimd_mm_castsi128_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpeq_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpeq_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8 = svdup_n_u8_z(svcmpeq_s8(pg, a.sve_i8, b.sve_i8), 0xFF);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_u8 = vceqq_s8(a.neon_i8, b.neon_i8);
    return res;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), (a_.i8 == b_.i8));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] == b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpeq_epi8(a, b) easysimd_mm_cmpeq_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpeq_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpeq_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svcmpeq_s16(svptrue_b16(), a.sve_i16, b.sve_i16);
    r.sve_i16 = svdup_n_s16_z(pg, 0xFFFF);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_u16 = vceqq_s16(b.neon_i16, a.neon_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = (a_.i16 == b_.i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] == b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpeq_epi16(a, b) easysimd_mm_cmpeq_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpeq_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpeq_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    r.sve_u32 = svdup_n_u32_z(svcmpeq_s32(pg, a.sve_i32, b.sve_i32), ~UINT32_C(0));
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_u32 = vceqq_s32(a.neon_i32, b.neon_i32);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 == b_.i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = (a_.i32[i] == b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpeq_epi32(a, b) easysimd_mm_cmpeq_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpeq_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpeq_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b64();
    r.sve_u64 = svdup_n_u64_z(svcmpeq_f64(pg, a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128 res;
    res.neon_u64 = vceqq_f64(a.neon_f64, b.neon_f64);
    return res;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 == b_.f64));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.u64[i] = easysimd_sse2_cmp_f64_e(a.f64[0], b.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpeq_pd(a, b) easysimd_mm_cmpeq_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpeq_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpeq_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = (a.u64[0] == b.u64[0]) ? ~UINT64_C(0) : 0;
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpeq_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpeq_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    r_.u64[0] = (a_.u64[0] == b_.u64[0]) ? ~UINT64_C(0) : 0;
    r_.u64[1] = a_.u64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpeq_sd(a, b) easysimd_mm_cmpeq_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpneq_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpneq_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b64();
    r.sve_u64 = svdup_n_u64_z(svcmpne_f64(pg, a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_u32 = vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(b_.neon_f64, a_.neon_f64)));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != b_.f64));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.u64[i] = easysimd_sse2_cmp_f64_e(a.f64[0], b.f64[0]) ? UINT64_C(0) : ~UINT64_C(0);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpneq_pd(a, b) easysimd_mm_cmpneq_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpneq_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpneq_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = easysimd_sse2_cmp_f64_e(a.f64[0], b.f64[0]) ? UINT64_C(0) : ~UINT64_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpneq_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpneq_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    r_.u64[0] = easysimd_sse2_cmp_f64_e(a_.f64[0], b_.f64[0]) ? UINT64_C(0) : ~UINT64_C(0);
    r_.u64[1] = a_.u64[1];


    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpneq_sd(a, b) easysimd_mm_cmpneq_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmplt_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmplt_epi8(a, b);
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i8 = svdup_n_s8_z(svcmplt_s8(svptrue_b8(), a.sve_i8, b.sve_i8), ~INT8_C(0));
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u8 = vcltq_s8(a.neon_i8, b.neon_i8);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] < b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmplt_epi8(a, b) easysimd_mm_cmplt_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmplt_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmplt_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svdup_n_s16_z(svcmplt_s16(svptrue_b16(), a.sve_i16, b.sve_i16), ~INT16_C(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u16 = vcltq_s16(a_.neon_i16, b_.neon_i16);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), (a_.i16 < b_.i16));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmplt_epi16(a, b) easysimd_mm_cmplt_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmplt_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmplt_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svdup_n_s32_z(svcmplt_s32(svptrue_b32(), a.sve_i32, b.sve_i32), ~INT32_C(0));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vcltq_s32(a_.neon_i32, b_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.i32 < b_.i32));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = (a_.i32[i] < b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmplt_epi32(a, b) easysimd_mm_cmplt_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmplt_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmplt_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svdup_n_u64_z(svcmplt_f64(svptrue_b64(), a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    a.neon_u64 = vcltq_f64(a.neon_f64, b.neon_f64);
    return a;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 < b_.f64));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.u64[i] = (a_.f64[i] < b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmplt_pd(a, b) easysimd_mm_cmplt_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmplt_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmplt_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = (a.f64[0] < b.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_cmplt_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_cmplt_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    r_.u64[0] = (a_.f64[0] < b_.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    r_.u64[1] = a_.u64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmplt_sd(a, b) easysimd_mm_cmplt_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmple_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmple_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svdup_n_u64_z(svcmple_f64(svptrue_b64(), a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 <= b_.f64));
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_u64 = vcleq_f64(a_.neon_f64, b_.neon_f64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.u64[i] = (a_.f64[i] > b_.f64[i]) ? UINT64_C(0) : ~UINT64_C(0);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmple_pd(a, b) easysimd_mm_cmple_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmple_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmple_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = easysimd_sse2_cmp_f64_le(a.f64[0], b.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_cmple_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_cmple_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    r_.u64[0] = easysimd_sse2_cmp_f64_le(a_.f64[0], b_.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    r_.u64[1] = a_.u64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmple_sd(a, b) easysimd_mm_cmple_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpgt_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpgt_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8 = svdup_n_u8_z(svcmpgt_s8(pg, a.sve_i8, b.sve_i8), 0xFF);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_u8 = vcgtq_s8(a.neon_i8, b.neon_i8);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), (a_.i8 > b_.i8));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] > b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpgt_epi8(a, b) easysimd_mm_cmpgt_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpgt_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpgt_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16 = svdup_n_u16_z(svcmpgt_s16(pg, a.sve_i16, b.sve_i16), 0xFFFF);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_u16 = vcgtq_s16(a.neon_i16, b.neon_i16);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i16), (a_.i16 > b_.i16));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpgt_epi16(a, b) easysimd_mm_cmpgt_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpgt_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpgt_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    r.sve_u32 = svdup_n_u32_z(svcmpgt_s32(pg, a.sve_i32, b.sve_i32), ~UINT32_C(0));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_u32 = vcgtq_s32(a.neon_i32, b.neon_i32);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.i32 > b_.i32));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = (a_.i32[i] > b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpgt_epi32(a, b) easysimd_mm_cmpgt_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpgt_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpgt_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svdup_n_u64_z(svcmpgt_f64(svptrue_b64(), a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128 res;
    res.neon_u64 = vcgtq_f64(a.neon_f64, b.neon_f64);
    return res;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 > b_.f64));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.u64[i] = (a_.f64[i] > b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpgt_pd(a, b) easysimd_mm_cmpgt_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpgt_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
    return _mm_cmpgt_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = (a.f64[0] > b.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpgt_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpgt_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    r_.u64[0] = (a_.f64[0] > b_.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    r_.u64[1] = a_.u64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpgt_sd(a, b) easysimd_mm_cmpgt_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpge_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpge_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svdup_n_u64_z(svcmpge_f64(svptrue_b64(), a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 >= b_.f64));
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_u64 = vcgeq_f64(a_.neon_f64, b_.neon_f64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.u64[i] = easysimd_sse2_cmp_f64_ge(a_.f64[i], b_.f64[i]) ? ~UINT64_C(0) : UINT64_C(0);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpge_pd(a, b) easysimd_mm_cmpge_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpge_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
    return _mm_cmpge_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = easysimd_sse2_cmp_f64_ge(a.f64[0], b.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpge_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpge_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    r_.u64[0] = easysimd_sse2_cmp_f64_ge(a_.f64[0], b_.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    r_.u64[1] = a_.u64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpge_sd(a, b) easysimd_mm_cmpge_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpngt_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpngt_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svdup_n_u64_z(svcmple_f64(svptrue_b64(), a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #else
    return easysimd_mm_cmple_pd(a, b);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpngt_pd(a, b) easysimd_mm_cmpngt_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpngt_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
    return _mm_cmpngt_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = (a.f64[0] > b.f64[0]) ? UINT64_C(0) : ~UINT64_C(0);
    return a;
  #else
    return easysimd_mm_cmple_sd(a, b);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpngt_sd(a, b) easysimd_mm_cmpngt_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpnge_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpnge_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svdup_n_u64_z(svcmplt_f64(svptrue_b64(), a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #else
    return easysimd_mm_cmplt_pd(a, b);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpnge_pd(a, b) easysimd_mm_cmpnge_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpnge_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
    return _mm_cmpnge_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = (a.f64[0] < b.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    return a;
  #else
    return easysimd_mm_cmplt_sd(a, b);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpnge_sd(a, b) easysimd_mm_cmpnge_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpnlt_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpnlt_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svdup_n_u64_z(svcmpge_f64(svptrue_b64(), a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #else
    return easysimd_mm_cmpge_pd(a, b);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpnlt_pd(a, b) easysimd_mm_cmpnlt_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpnlt_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpnlt_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = (a.f64[0] < b.f64[0]) ? UINT64_C(0) : ~UINT64_C(0);
    return a;
  #else
    return easysimd_mm_cmpge_sd(a, b);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpnlt_sd(a, b) easysimd_mm_cmpnlt_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpnle_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpnle_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_u64 = svdup_n_u64_z(svcmpgt_f64(svptrue_b64(), a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #else
    return easysimd_mm_cmpgt_pd(a, b);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpnle_pd(a, b) easysimd_mm_cmpnle_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpnle_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpnle_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = (a.f64[0] > b.f64[0]) ? ~UINT64_C(0) : UINT64_C(0);
    return a;
  #else
    return easysimd_mm_cmpgt_sd(a, b);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpnle_sd(a, b) easysimd_mm_cmpnle_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpord_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpord_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svdup_n_u64_z(svnot_b_z(svptrue_b64(), svcmpuo_f64(svptrue_b64(), a.sve_f64, b.sve_f64)), ~UINT64_C(0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      /* Note: NEON does not have ordered compare builtin
        Need to compare a eq a and b eq b to check for NaN
        Do AND of results to get final */
      uint64x2_t ceqaa = vceqq_f64(a_.neon_f64, a_.neon_f64);
      uint64x2_t ceqbb = vceqq_f64(b_.neon_f64, b_.neon_f64);
      r_.neon_u64 = vandq_u64(ceqaa, ceqbb);
    #elif defined(easysimd_math_isnan)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.u64[i] = (!easysimd_math_isnan(a_.f64[i]) && !easysimd_math_isnan(b_.f64[i])) ? ~UINT64_C(0) : UINT64_C(0);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpord_pd(a, b) easysimd_mm_cmpord_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64
easysimd_mm_cvtsd_f64 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
    return _mm_cvtsd_f64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.f64[0];
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      return HEDLEY_STATIC_CAST(easysimd_float64, vgetq_lane_f64(a_.neon_f64, 0));
    #else
      return a_.f64[0];
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtsd_f64(a) easysimd_mm_cvtsd_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm_cvtsd_i64 (easysimd__m128d a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svcvt_s64_f64_z(svptrue_b64(), a.sve_f64);
    return r.i64[0];
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);
    return HEDLEY_STATIC_CAST(int64_t, a_.f64[0]);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtsd_i64(a) easysimd_mm_cvtsd_i64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpord_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpord_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = (!easysimd_math_isnan(a.f64[0]) && !easysimd_math_isnan(b.f64[0])) ? ~UINT64_C(0) : UINT64_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpord_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpord_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(easysimd_math_isnan)
      r_.u64[0] = (!easysimd_math_isnan(a_.f64[0]) && !easysimd_math_isnan(b_.f64[0])) ? ~UINT64_C(0) : UINT64_C(0);
      r_.u64[1] = a_.u64[1];
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpord_sd(a, b) easysimd_mm_cmpord_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpunord_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpunord_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svdup_n_u64_z(svcmpuo_f64(svptrue_b64(), a.sve_f64, b.sve_f64), ~UINT64_C(0));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint64x2_t ceqaa = vceqq_f64(a_.neon_f64, a_.neon_f64);
      uint64x2_t ceqbb = vceqq_f64(b_.neon_f64, b_.neon_f64);
      r_.neon_u64 = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vandq_u64(ceqaa, ceqbb))));
    #elif defined(easysimd_math_isnan)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.u64[i] = (easysimd_math_isnan(a_.f64[i]) || easysimd_math_isnan(b_.f64[i])) ? ~UINT64_C(0) : UINT64_C(0);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpunord_pd(a, b) easysimd_mm_cmpunord_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmpunord_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cmpunord_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = (easysimd_math_isnan(a.f64[0]) || easysimd_math_isnan(b.f64[0])) ? ~UINT64_C(0) : UINT64_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpunord_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_cmpunord_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(easysimd_math_isnan)
      r_.u64[0] = (easysimd_math_isnan(a_.f64[0]) || easysimd_math_isnan(b_.f64[0])) ? ~UINT64_C(0) : UINT64_C(0);
      r_.u64[1] = a_.u64[1];
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cmpunord_sd(a, b) easysimd_mm_cmpunord_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cvtepi32_pd (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtepi32_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svcvt_f64_s32_z(svptrue_b64(), svtbl_s32(a.sve_i32, svdupq_n_u32(0, 0, 1, 0)));
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.f64, a_.m64_private[0].i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = (easysimd_float64) a_.i32[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtepi32_pd(a) easysimd_mm_cvtepi32_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtepi32_ps (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtepi32_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svcvt_f32_s32_z(svptrue_b32(), a.sve_i32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcvtq_f32_s32(a.neon_i32);
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.f32, a_.i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = (easysimd_float32) a_.i32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtepi32_ps(a) easysimd_mm_cvtepi32_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_cvtpd_pi32 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtpd_pi32(a);
  #else
    easysimd__m64_private r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      easysimd_float64 v = easysimd_math_round(a_.f64[i]);
      #if defined(EASYSIMD_FAST_CONVERSION_RANGE)
        r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, v);
      #else
        r_.i32[i] = ((v > HEDLEY_STATIC_CAST(easysimd_float64, INT32_MIN)) && (v < HEDLEY_STATIC_CAST(easysimd_float64, INT32_MAX))) ?
          EASYSIMD_CONVERT_FTOI(int32_t, v) : INT32_MIN;
      #endif
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtpd_pi32(a) easysimd_mm_cvtpd_pi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtpd_epi32 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(EASYSIMD_BUG_PGI_30107)
    return _mm_cvtpd_epi32(a);
  #else
    easysimd__m128i_private r_;

    r_.m64[0] = easysimd_mm_cvtpd_pi32(a);
    r_.m64[1] = easysimd_mm_setzero_si64();

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtpd_epi32(a) easysimd_mm_cvtpd_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtpd_ps (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtpd_ps(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcombine_f32(vcvt_f32_f64(a.neon_f64), vdup_n_f32(0.0f));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svtbl_f32(svcvt_f32_f64_z(svptrue_b32(), a.sve_f64), svdupq_n_u32(0, 2, 4, 4));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    #if HEDLEY_HAS_BUILTIN(__builtin_shufflevector) && HEDLEY_HAS_BUILTIN(__builtin_convertvector)
      float __attribute__((__vector_size__(8))) z = { 0.0f, 0.0f };
      r_.f32 =
        __builtin_shufflevector(
          __builtin_convertvector(__builtin_shufflevector(a_.f64, a_.f64, 0, 1), __typeof__(z)), z,
          0, 1, 2, 3
        );
    #else
      r_.f32[0] = HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[0]);
      r_.f32[1] = HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[1]);
      r_.f32[2] = EASYSIMD_FLOAT32_C(0.0);
      r_.f32[3] = EASYSIMD_FLOAT32_C(0.0);
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtpd_ps(a) easysimd_mm_cvtpd_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cvtpi32_pd (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtpi32_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdupq_n_f64((float64_t)a.i32[0], (float64_t)a.i32[1]);
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd__m64_private a_ = easysimd__m64_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.f64, a_.i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = (easysimd_float64) a_.i32[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtpi32_pd(a) easysimd_mm_cvtpi32_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtps_epi32 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtps_epi32(a);
  #else
    easysimd__m128i_private r_;
    easysimd__m128_private a_;

    #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_FAST_CONVERSION_RANGE) && defined(EASYSIMD_FAST_ROUND_TIES) && !defined(EASYSIMD_BUG_GCC_95399)
      a_ = easysimd__m128_to_private(a);
      r_.neon_i32 = vcvtnq_s32_f32(a_.neon_f32);
    #else
      a_ = easysimd__m128_to_private(easysimd_x_mm_round_ps(a, EASYSIMD_MM_FROUND_TO_NEAREST_INT, 1));
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        easysimd_float32 v = easysimd_math_roundf(a_.f32[i]);
        #if defined(EASYSIMD_FAST_CONVERSION_RANGE)
          r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, v);
        #else
          r_.i32[i] = ((v > HEDLEY_STATIC_CAST(easysimd_float32, INT32_MIN)) && (v < HEDLEY_STATIC_CAST(easysimd_float32, INT32_MAX))) ?
            EASYSIMD_CONVERT_FTOI(int32_t, v) : INT32_MIN;
        #endif
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtps_epi32(a) easysimd_mm_cvtps_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cvtps_pd (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtps_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svcvt_f64_f32_z(svptrue_b64(), svtbl_f32(a.sve_f32, svdupq_n_u32(0, 0, 1, 0)));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vcvt_f64_f32(vget_low_f32(a.neon_f32));
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.f64, a_.m64_private[0].f32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f32[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtps_pd(a) easysimd_mm_cvtps_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_cvtsd_si32 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtsd_si32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f64[0] = easysimd_math_round(a.f64[0]);
    if ((a.f64[0] > HEDLEY_STATIC_CAST(easysimd_float64, INT32_MIN)) && (a.f64[0] < HEDLEY_STATIC_CAST(easysimd_float64, INT32_MAX))) {
      return EASYSIMD_CONVERT_FTOI(int32_t, a.f64[0]);
    }
    return INT32_MIN;
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    easysimd_float64 v = easysimd_math_round(a_.f64[0]);
    #if defined(EASYSIMD_FAST_CONVERSION_RANGE)
      return EASYSIMD_CONVERT_FTOI(int32_t, v);
    #else
      return ((v > HEDLEY_STATIC_CAST(easysimd_float64, INT32_MIN)) && (v < HEDLEY_STATIC_CAST(easysimd_float64, INT32_MAX))) ?
        EASYSIMD_CONVERT_FTOI(int32_t, v) : INT32_MIN;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtsd_si32(a) easysimd_mm_cvtsd_si32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm_cvtsd_si64 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
    #if defined(__PGI)
      return _mm_cvtsd_si64x(a);
    #else
      return _mm_cvtsd_si64(a);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return EASYSIMD_CONVERT_FTOI(int64_t, easysimd_math_round(a.f64[0]));
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);
    return EASYSIMD_CONVERT_FTOI(int64_t, easysimd_math_round(a_.f64[0]));
  #endif
}
#define easysimd_mm_cvtsd_si64x(a) easysimd_mm_cvtsd_si64(a)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
  #define _mm_cvtsd_si64(a) easysimd_mm_cvtsd_si64(a)
  #define _mm_cvtsd_si64x(a) easysimd_mm_cvtsd_si64x(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtsd_ss (easysimd__m128 a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtsd_ss(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsetq_lane_f32(vcvtxd_f32_f64(vgetq_lane_f64(b.neon_f64, 0)), a.neon_f32, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f32[0] = HEDLEY_STATIC_CAST(easysimd_float32, b.f64[0]);
    return a;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);
    easysimd__m128d_private b_ = easysimd__m128d_to_private(b);

    r_.f32[0] = HEDLEY_STATIC_CAST(easysimd_float32, b_.f64[0]);
    EASYSIMD_VECTORIZE
    for (size_t i = 1 ; i < (sizeof(r_) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtsd_ss(a, b) easysimd_mm_cvtsd_ss(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_x_mm_cvtsi128_si16 (easysimd__m128i a) {
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a);

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vgetq_lane_s16(a_.neon_i16, 0);
  #else
    return a_.i16[0];
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_cvtsi128_si32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtsi128_si32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.sve_i32[0];
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    return vgetq_lane_s32(a.neon_i32, 0);
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a);
      return a_.i32[0];
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtsi128_si32(a) easysimd_mm_cvtsi128_si32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm_cvtsi128_si64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
    #if defined(__PGI)
      return _mm_cvtsi128_si64x(a);
    #else
      return _mm_cvtsi128_si64(a);
    #endif
  #else
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.sve_i64[0];
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vgetq_lane_s64(a.neon_i64, 0);
  #endif
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    return a_.i64[0];
  #endif
}
#define easysimd_mm_cvtsi128_si64x(a) easysimd_mm_cvtsi128_si64(a)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
  #define _mm_cvtsi128_si64(a) easysimd_mm_cvtsi128_si64(a)
  #define _mm_cvtsi128_si64x(a) easysimd_mm_cvtsi128_si64x(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cvtsi32_sd (easysimd__m128d a, int32_t b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtsi32_sd(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vsetq_lane_f64(HEDLEY_STATIC_CAST(float64_t, b), a.neon_f64, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f64[0] = HEDLEY_STATIC_CAST(float64_t, b);
    return a;
  #else
    easysimd__m128d_private r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    r_.f64[0] = HEDLEY_STATIC_CAST(easysimd_float64, b);
    r_.i64[1] = a_.i64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtsi32_sd(a, b) easysimd_mm_cvtsi32_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_cvtsi16_si128 (int16_t a) {
  easysimd__m128i_private r_;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    r_.neon_i16 = vsetq_lane_s16(a, vdupq_n_s16(0), 0);
  #else
    r_.i16[0] = a;
    r_.i16[1] = 0;
    r_.i16[2] = 0;
    r_.i16[3] = 0;
    r_.i16[4] = 0;
    r_.i16[5] = 0;
    r_.i16[6] = 0;
    r_.i16[7] = 0;
  #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtsi32_si128 (int32_t a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtsi32_si128(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svdupq_n_s32(a, 0, 0, 0);
    return r;
  #else
    easysimd__m128i_private r_;
    #if (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
      r_.neon_i32 = vsetq_lane_s32(a, vdupq_n_s32(0), 0);
    #else
      r_.i32[0] = a;
      r_.i32[1] = 0;
      r_.i32[2] = 0;
      r_.i32[3] = 0;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtsi32_si128(a) easysimd_mm_cvtsi32_si128(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cvtsi64_sd (easysimd__m128d a, int64_t b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
    #if !defined(__PGI)
      return _mm_cvtsi64_sd(a, b);
    #else
      return _mm_cvtsi64x_sd(a, b);
    #endif
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vsetq_lane_f64(HEDLEY_STATIC_CAST(float64_t, b), a.neon_f64, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f64[0] = b;
    return a;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    r_.f64[0] = HEDLEY_STATIC_CAST(easysimd_float64, b);
    r_.f64[1] = a_.f64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#define easysimd_mm_cvtsi64x_sd(a, b) easysimd_mm_cvtsi64_sd(a, b)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
  #define _mm_cvtsi64_sd(a, b) easysimd_mm_cvtsi64_sd(a, b)
  #define _mm_cvtsi64x_sd(a, b) easysimd_mm_cvtsi64x_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvtsi64_si128 (int64_t a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
    #if !defined(__PGI)
      return _mm_cvtsi64_si128(a);
    #else
      return _mm_cvtsi64x_si128(a);
    #endif
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i64[0] = a;
      r_.sve_i64[1] = 0;
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vsetq_lane_s64(a, vdupq_n_s64(0), 0);
    #else
      r_.i64[0] = a;
      r_.i64[1] = 0;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#define easysimd_mm_cvtsi64x_si128(a) easysimd_mm_cvtsi64_si128(a)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
  #define _mm_cvtsi64_si128(a) easysimd_mm_cvtsi64_si128(a)
  #define _mm_cvtsi64x_si128(a) easysimd_mm_cvtsi64x_si128(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cvtss_sd (easysimd__m128d a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvtss_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f64[0] = HEDLEY_STATIC_CAST(easysimd_float64, b.f32[0]);
    return a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    float64x2_t temp = vcvt_f64_f32(vset_lane_f32(vgetq_lane_f32(easysimd__m128_to_private(b).neon_f32, 0), vdup_n_f32(0), 0));
    easysimd__m128d r;
    r.neon_f64 = vsetq_lane_f64(vgetq_lane_f64(easysimd__m128d_to_private(a).neon_f64, 1), temp, 1);
    return r;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a);
    easysimd__m128_private b_ = easysimd__m128_to_private(b);

    a_.f64[0] = HEDLEY_STATIC_CAST(easysimd_float64, b_.f32[0]);

    return easysimd__m128d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvtss_sd(a, b) easysimd_mm_cvtss_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_cvttpd_pi32 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvttpd_pi32(a);
  #else
    easysimd__m64_private r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_) && defined(EASYSIMD_FAST_CONVERSION_RANGE)
      EASYSIMD_CONVERT_VECTOR_(r_.i32, a_.f64);
    #else
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        easysimd_float64 v = a_.f64[i];
        #if defined(EASYSIMD_FAST_CONVERSION_RANGE)
          r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, v);
        #else
          r_.i32[i] = ((v > HEDLEY_STATIC_CAST(easysimd_float64, INT32_MIN)) && (v < HEDLEY_STATIC_CAST(easysimd_float64, INT32_MAX))) ?
            EASYSIMD_CONVERT_FTOI(int32_t, v) : INT32_MIN;
        #endif
      }
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvttpd_pi32(a) easysimd_mm_cvttpd_pi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvttpd_epi32 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvttpd_epi32(a);
  #else
    easysimd__m128i_private r_;

    r_.m64[0] = easysimd_mm_cvttpd_pi32(a);
    r_.m64[1] = easysimd_mm_setzero_si64();

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvttpd_epi32(a) easysimd_mm_cvttpd_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cvttps_epi32 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvttps_epi32(a);
  #else
    easysimd__m128i_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vcvtq_s32_f32(a_.neon_f32);

      #if !defined(EASYSIMD_FAST_CONVERSION_RANGE) || !defined(EASYSIMD_FAST_NANS)
        /* Values below INT32_MIN saturate anyways, so we don't need to
         * test for that. */
        #if !defined(EASYSIMD_FAST_CONVERSION_RANGE) && !defined(EASYSIMD_FAST_NANS)
          uint32x4_t valid_input =
            vandq_u32(
              vcltq_f32(a_.neon_f32, vdupq_n_f32(EASYSIMD_FLOAT32_C(2147483648.0))),
              vceqq_f32(a_.neon_f32, a_.neon_f32)
            );
        #elif !defined(EASYSIMD_FAST_CONVERSION_RANGE)
          uint32x4_t valid_input = vcltq_f32(a_.neon_f32, vdupq_n_f32(EASYSIMD_FLOAT32_C(2147483648.0)));
        #elif !defined(EASYSIMD_FAST_NANS)
          uint32x4_t valid_input = vceqq_f32(a_.neon_f32, a_.neon_f32);
        #endif

        r_.neon_i32 = vbslq_s32(valid_input, r_.neon_i32, vdupq_n_s32(INT32_MIN));
      #endif
    #elif defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i32, a_.f32);

      #if !defined(EASYSIMD_FAST_CONVERSION_RANGE) || !defined(EASYSIMD_FAST_NANS)
        #if !defined(EASYSIMD_FAST_CONVERSION_RANGE)
          static const easysimd_float32 EASYSIMD_VECTOR(16) first_too_high = { EASYSIMD_FLOAT32_C(2147483648.0), EASYSIMD_FLOAT32_C(2147483648.0), EASYSIMD_FLOAT32_C(2147483648.0), EASYSIMD_FLOAT32_C(2147483648.0) };

          __typeof__(r_.i32) valid_input =
            HEDLEY_REINTERPRET_CAST(
              __typeof__(r_.i32),
              (a_.f32 < first_too_high) & (a_.f32 >= -first_too_high)
            );
        #elif !defined(EASYSIMD_FAST_NANS)
          __typeof__(r_.i32) valid_input = HEDLEY_REINTERPRET_CAST( __typeof__(valid_input), a_.f32 == a_.f32);
        #endif

        __typeof__(r_.i32) invalid_output = { INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN };
        r_.i32 = (r_.i32 & valid_input) | (invalid_output & ~valid_input);
      #endif
    #else
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        easysimd_float32 v = a_.f32[i];
        #if defined(EASYSIMD_FAST_CONVERSION_RANGE) && defined(EASYSIMD_FAST_NANS)
          r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, v);
        #else
          r_.i32[i] = ((v > HEDLEY_STATIC_CAST(easysimd_float32, INT32_MIN)) && (v < HEDLEY_STATIC_CAST(easysimd_float32, INT32_MAX))) ?
            EASYSIMD_CONVERT_FTOI(int32_t, v) : INT32_MIN;
        #endif
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvttps_epi32(a) easysimd_mm_cvttps_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_cvttsd_si32 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_cvttsd_si32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    if ((a.f64[0] > HEDLEY_STATIC_CAST(easysimd_float64, INT32_MIN)) && (a.f64[0] < HEDLEY_STATIC_CAST(easysimd_float64, INT32_MAX))) {
      return EASYSIMD_CONVERT_FTOI(int32_t, a.f64[0]);
    }
    return INT32_MIN;
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);
    easysimd_float64 v = a_.f64[0];
    #if defined(EASYSIMD_FAST_CONVERSION_RANGE)
      return EASYSIMD_CONVERT_FTOI(int32_t, v);
    #else
      return ((v > HEDLEY_STATIC_CAST(easysimd_float64, INT32_MIN)) && (v < HEDLEY_STATIC_CAST(easysimd_float64, INT32_MAX))) ?
        EASYSIMD_CONVERT_FTOI(int32_t, v) : INT32_MIN;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_cvttsd_si32(a) easysimd_mm_cvttsd_si32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm_cvttsd_si64 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
    #if !defined(__PGI)
      return _mm_cvttsd_si64(a);
    #else
      return _mm_cvttsd_si64x(a);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svcvt_s64_f64_z(svptrue_b64(), a.sve_f64);
    return r.i64[0];
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);
    return EASYSIMD_CONVERT_FTOI(int64_t, a_.f64[0]);
  #endif
}
#define easysimd_mm_cvttsd_si64x(a) easysimd_mm_cvttsd_si64(a)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
  #define _mm_cvttsd_si64(a) easysimd_mm_cvttsd_si64(a)
  #define _mm_cvttsd_si64x(a) easysimd_mm_cvttsd_si64x(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_div_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_div_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdiv_f64_z(svptrue_b64(), a.sve_f64, b.sve_f64);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vdivq_f64(a.neon_f64, b.neon_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 / b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[i] / b_.f64[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_div_pd(a, b) easysimd_mm_div_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_div_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_div_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdiv_f64_z(svptrue_b64(), a.sve_f64, svdupq_n_f64(b.f64[0], 1.0));
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_div_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_div_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      float64x2_t temp = vdivq_f64(a_.neon_f64, b_.neon_f64);
      r_.neon_f64 = vsetq_lane_f64(vgetq_lane(a_.neon_f64, 1), temp, 1);
    #else
      r_.f64[0] = a_.f64[0] / b_.f64[0];
      r_.f64[1] = a_.f64[1];
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_div_sd(a, b) easysimd_mm_div_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_extract_epi16 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 7)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  return (int32_t)(a.sve_u16[imm8 & 7]);
#else
  uint16_t r;
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

  r = a_.u16[imm8 & 7];
  return  HEDLEY_STATIC_CAST(int32_t, r);
#endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(4,6,0))
  #define easysimd_mm_extract_epi16(a, imm8) _mm_extract_epi16(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm_extract_epi16(a, imm8) (HEDLEY_STATIC_CAST(int32_t, vgetq_lane_s16(easysimd__m128i_to_private(a).neon_i16, (imm8))) & (INT32_C(0x0000ffff)))
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_extract_epi16(a, imm8) easysimd_mm_extract_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_insert_epi16 (easysimd__m128i a, int16_t i, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 7)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  a.sve_i16[imm8 & 7] = i;
  return a;
#else
  easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
  a_.i16[imm8 & 7] = i;
  return easysimd__m128i_from_private(a_);
#endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
  #define easysimd_mm_insert_epi16(a, i, imm8) _mm_insert_epi16((a), (i), (imm8))
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_mm_insert_epi16(a, i, imm8) easysimd__m128i_from_neon_i16(vsetq_lane_s16((i), easysimd__m128i_to_neon_i16(a), (imm8)))
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_insert_epi16(a, i, imm8) easysimd_mm_insert_epi16(a, i, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_load_pd (easysimd_float64 const mem_addr[HEDLEY_ARRAY_PARAM(2)]) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_load_pd(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svld1_f64(svptrue_b64(), (float64_t const *)mem_addr);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vld1q_f64((float64_t const *)mem_addr);
    return r;
  #else
    easysimd__m128d_private r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vld1q_u32(HEDLEY_REINTERPRET_CAST(uint32_t const*, mem_addr));
    #else
      easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128d), sizeof(r_));
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_load_pd(mem_addr) easysimd_mm_load_pd(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_load1_pd (easysimd_float64 const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_load1_pd(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdup_n_f64(*mem_addr);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return easysimd__m128d_from_neon_f64(vld1q_dup_f64(mem_addr));
  #else
    return easysimd_mm_set1_pd(*mem_addr);
  #endif
}
#define easysimd_mm_load_pd1(mem_addr) easysimd_mm_load1_pd(mem_addr)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_load_pd1(mem_addr) easysimd_mm_load1_pd(mem_addr)
  #define _mm_load1_pd(mem_addr) easysimd_mm_load1_pd(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_load_sd (easysimd_float64 const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_load_sd(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svld1_f64(svdupq_n_b64(1, 0), mem_addr);
    return r;
  #else
    easysimd__m128d_private r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vsetq_lane_f64(*mem_addr, vdupq_n_f64(0), 0);
    #else
      r_.f64[0] = *mem_addr;
      r_.u64[1] = UINT64_C(0);
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_load_sd(mem_addr) easysimd_mm_load_sd(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_load_epi32(void const* mem_addr) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1_s32(svptrue_b32(), (int32_t const *)mem_addr);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i32 = vld1q_s32((int32_t const *)mem_addr);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(easysimd__m128i));
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_load_epi32(mem_addr) easysimd_mm_load_epi32(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_load_epi64(void const* mem_addr) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1_s64(svptrue_b64(), (int64_t const *)mem_addr);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i64 = vld1q_s64((int64_t const *)mem_addr);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(easysimd__m128i));
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #define _mm_load_epi64(mem_addr) easysimd_mm_load_epi64(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_load_si128 (easysimd__m128i const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_load_si128(HEDLEY_REINTERPRET_CAST(__m128i const*, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1_s32(svptrue_b32(), (int32_t const *)mem_addr);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i32 = vld1q_s32((int32_t const *)mem_addr);
    return r;
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_POWER_ALTIVEC_P6_NATIVE)
      r_.altivec_i32 = vec_ld(0, HEDLEY_REINTERPRET_CAST(EASYSIMD_POWER_ALTIVEC_VECTOR(int) const*, mem_addr));
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vld1q_s32(HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr));
    #else
      easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128i), sizeof(easysimd__m128i));
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_load_si128(mem_addr) easysimd_mm_load_si128(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_loadh_pd (easysimd__m128d a, easysimd_float64 const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadh_pd(a, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f64[1] = *(float64_t const *)mem_addr;
    return a;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vcombine_f64(vget_low_f64(a_.neon_f64), vld1_f64(HEDLEY_REINTERPRET_CAST(const float64_t*, mem_addr)));
    #else
      easysimd_float64 t;

      easysimd_memcpy(&t, mem_addr, sizeof(t));
      r_.f64[0] = a_.f64[0];
      r_.f64[1] = t;
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_loadh_pd(a, mem_addr) easysimd_mm_loadh_pd(a, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_loadl_epi64 (easysimd__m128i const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadl_epi64(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdupq_n_s64(*((int64_t *)mem_addr), 0);
    return r;
  #else
    easysimd__m128i_private r_;

    int64_t value;
    easysimd_memcpy(&value, mem_addr, sizeof(value));

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vcombine_s64(vld1_s64(HEDLEY_REINTERPRET_CAST(int64_t const *, mem_addr)), vdup_n_s64(0));
    #else
      r_.i64[0] = value;
      r_.i64[1] = 0;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_loadl_epi64(mem_addr) easysimd_mm_loadl_epi64(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_loadl_pd (easysimd__m128d a, easysimd_float64 const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadl_pd(a, mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f64[0] = *mem_addr;
    return a;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vcombine_f64(vld1_f64(
        HEDLEY_REINTERPRET_CAST(const float64_t*, mem_addr)), vget_high_f64(a_.neon_f64));
    #else
      r_.f64[0] = *mem_addr;
      r_.u64[1] = a_.u64[1];
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_loadl_pd(a, mem_addr) easysimd_mm_loadl_pd(a, mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_loadr_pd (easysimd_float64 const mem_addr[HEDLEY_ARRAY_PARAM(2)]) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadr_pd(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svrev_f64(svld1_f64(svptrue_b64(), (float64_t const *)mem_addr));
    return r;
  #else
    easysimd__m128d_private
      r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vld1q_f64(mem_addr);
      r_.neon_f64 = vextq_f64(r_.neon_f64, r_.neon_f64, 1);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vld1q_s64(HEDLEY_REINTERPRET_CAST(int64_t const *, mem_addr));
      r_.neon_i64 = vextq_s64(r_.neon_i64, r_.neon_i64, 1);
    #else
      r_.f64[0] = mem_addr[1];
      r_.f64[1] = mem_addr[0];
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_loadr_pd(mem_addr) easysimd_mm_loadr_pd(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_loadu_pd (easysimd_float64 const mem_addr[HEDLEY_ARRAY_PARAM(2)]) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadu_pd(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f64 = svld1_f64(svptrue_b64(), &(mem_addr[0]));
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128 r;
    r.neon_f64 = vld1q_f64(HEDLEY_REINTERPRET_CAST(const float64_t*, mem_addr));
    return r;
  #else
    easysimd__m128d_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_loadu_pd(mem_addr) easysimd_mm_loadu_pd(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_loadu_epi8(void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_CLANG_REV_344862)
    return _mm_loadu_epi8(mem_addr);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadu_si128(EASYSIMD_ALIGN_CAST(__m128i const *, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svld1_s8(svptrue_b8(), (const int8_t *)mem_addr);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i8 = vld1q_s8(HEDLEY_REINTERPRET_CAST(int8_t const*, mem_addr));
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    return easysimd__m128i_from_private(r_);
  #endif
}
#define easysimd_x_mm_loadu_epi8(mem_addr) easysimd_mm_loadu_epi8(mem_addr)
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && (defined(EASYSIMD_BUG_GCC_95483) || defined(EASYSIMD_BUG_CLANG_REV_344862)))
  #undef _mm_loadu_epi8
  #define _mm_loadu_epi8(a) easysimd_mm_loadu_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_loadu_epi16(void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_CLANG_REV_344862)
    return _mm_loadu_epi16(mem_addr);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadu_si128(EASYSIMD_ALIGN_CAST(__m128i const *, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svld1_s16(svptrue_b16(), (const int16_t *)mem_addr);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i16 = vld1q_s16(HEDLEY_REINTERPRET_CAST(int16_t const*, mem_addr));
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    return easysimd__m128i_from_private(r_);
  #endif
}
#define easysimd_x_mm_loadu_epi16(mem_addr) easysimd_mm_loadu_epi16(mem_addr)
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && (defined(EASYSIMD_BUG_GCC_95483) || defined(EASYSIMD_BUG_CLANG_REV_344862)))
  #undef _mm_loadu_epi16
  #define _mm_loadu_epi16(a) easysimd_mm_loadu_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_loadu_epi32(void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_CLANG_REV_344862)
    return _mm_loadu_epi32(mem_addr);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadu_si128(EASYSIMD_ALIGN_CAST(__m128i const *, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1_s32(svptrue_b32(), (const int32_t *)mem_addr);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i32 = vld1q_s32(HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr));
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    return easysimd__m128i_from_private(r_);
  #endif
}
#define easysimd_x_mm_loadu_epi32(mem_addr) easysimd_mm_loadu_epi32(mem_addr)
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && (defined(EASYSIMD_BUG_GCC_95483) || defined(EASYSIMD_BUG_CLANG_REV_344862)))
  #undef _mm_loadu_epi32
  #define _mm_loadu_epi32(a) easysimd_mm_loadu_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_loadu_epi64(void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_CLANG_REV_344862)
    return _mm_loadu_epi64(mem_addr);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadu_si128(EASYSIMD_ALIGN_CAST(__m128i const *, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svld1_s64(svptrue_b64(), (const int64_t *)mem_addr);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i64 = vld1q_s64(HEDLEY_REINTERPRET_CAST(int64_t const*, mem_addr));
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    return easysimd__m128i_from_private(r_);
  #endif
}
#define easysimd_x_mm_loadu_epi64(mem_addr) easysimd_mm_loadu_epi64(mem_addr)
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && (defined(EASYSIMD_BUG_GCC_95483) || defined(EASYSIMD_BUG_CLANG_REV_344862)))
  #undef _mm_loadu_epi64
  #define _mm_loadu_epi64(a) easysimd_mm_loadu_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_loadu_si128 (void const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_loadu_si128(HEDLEY_STATIC_CAST(__m128i const*, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1_s32(svptrue_b32(), (const int32_t *)mem_addr);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i32 = vld1q_s32(HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr));
    return r;
  #else
    easysimd__m128i_private r_;

    #if HEDLEY_GNUC_HAS_ATTRIBUTE(may_alias,3,3,0)
      HEDLEY_DIAGNOSTIC_PUSH
      EASYSIMD_DIAGNOSTIC_DISABLE_PACKED_
      struct easysimd_mm_loadu_si128_s {
        __typeof__(r_) v;
      } __attribute__((__packed__, __may_alias__));
      r_ = HEDLEY_REINTERPRET_CAST(const struct easysimd_mm_loadu_si128_s *, mem_addr)->v;
      HEDLEY_DIAGNOSTIC_POP
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i8 = vld1q_s8(HEDLEY_REINTERPRET_CAST(int8_t const*, mem_addr));
    #else
      easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_loadu_si128(mem_addr) easysimd_mm_loadu_si128(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_madd_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_madd_epi16(a, b);
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i32 = svadd_s32_z(svptrue_b32(), svmullb_s32(a.sve_i16, b.sve_i16), svmullt_s32(a.sve_i16, b.sve_i16));
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      int32x4_t pl = vmull_s16(vget_low_s16(a.neon_i16),  vget_low_s16(b.neon_i16));
      int32x4_t ph = vmull_high_s16(a.neon_i16, b.neon_i16);
      r_.neon_i32 = vpaddq_s32(pl, ph);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int32x4_t pl = vmull_s16(vget_low_s16(a_.neon_i16),  vget_low_s16(b_.neon_i16));
      int32x4_t ph = vmull_s16(vget_high_s16(a_.neon_i16), vget_high_s16(b_.neon_i16));
      int32x2_t rl = vpadd_s32(vget_low_s32(pl), vget_high_s32(pl));
      int32x2_t rh = vpadd_s32(vget_low_s32(ph), vget_high_s32(ph));
      r_.neon_i32 = vcombine_s32(rl, rh);
    #elif defined(EASYSIMD_ZARCH_ZVECTOR_13_NATIVE)
      r_.altivec_i32 = vec_mule(a_.altivec_i16, b_.altivec_i16) + vec_mulo(a_.altivec_i16, b_.altivec_i16);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && defined(EASYSIMD_CONVERT_VECTOR_) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      int32_t EASYSIMD_VECTOR(32) a32, b32, p32;
      EASYSIMD_CONVERT_VECTOR_(a32, a_.i16);
      EASYSIMD_CONVERT_VECTOR_(b32, b_.i16);
      p32 = a32 * b32;
      r_.i32 =
        __builtin_shufflevector(p32, p32, 0, 2, 4, 6) +
        __builtin_shufflevector(p32, p32, 1, 3, 5, 7);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.i16[0])) ; i += 2) {
        r_.i32[i / 2] = (a_.i16[i] * b_.i16[i]) + (a_.i16[i + 1] * b_.i16[i + 1]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_madd_epi16(a, b) easysimd_mm_madd_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_maskmoveu_si128 (easysimd__m128i a, easysimd__m128i mask, int8_t mem_addr[HEDLEY_ARRAY_PARAM(16)]) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_maskmoveu_si128(a, mask, HEDLEY_REINTERPRET_CAST(char*, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b8();
    easysimd_svbool_t pgm = svcmpgt_n_u8(pg, svand_n_u8_z(pg, mask.sve_u8, 0x80), UINT8_C(0));
    svst1_s8(pgm, mem_addr, a.sve_i8);
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      mask_ = easysimd__m128i_to_private(mask);

    for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++) {
      if (mask_.u8[i] & 0x80) {
        mem_addr[i] = a_.i8[i];
      }
    }
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_maskmoveu_si128(a, mask, mem_addr) easysimd_mm_maskmoveu_si128((a), (mask), EASYSIMD_CHECKED_REINTERPRET_CAST(int8_t*,fgggg char*, (mem_addr)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_movemask_epi8 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__INTEL_COMPILER)
    /* ICC has trouble with _mm_movemask_epi8 at -O2 and above: */
    return _mm_movemask_epi8(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int32_t ret = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(ret, svcmplt_n_s8(pg, a.sve_i8, 0), EASYSIMD_SV_INDEX_0);
    return ret;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      int32_t r;
      __asm__ __volatile__ (
        "ushr %[a0].16b, %[a0].16b, #7          \n\t"
        "usra %[a0].8h, %[a0].8h, #7            \n\t"
        "usra %[a0].4s, %[a0].4s, #14           \n\t"
        "usra %[a0].2d, %[a0].2d, #28           \n\t"
        "ins %[a0].b[1], %[a0].b[8]             \n\t"
        "umov %w[r], %[a0].h[0]"
        :[r]"=r"(r), [a0]"+w"(a.neon_u8)
        :
        :
      );
      return r;
    #else
        int32_t r = 0;
        easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
        EASYSIMD_VECTORIZE_REDUCTION(|:r)
        for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
          r |= (a_.u8[15 - i] >> 7) << (15 - i);
        }
        return r;
    #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_movemask_epi8(a) easysimd_mm_movemask_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_movemask_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_movemask_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64, svdup_n_s64(0)), EASYSIMD_SV_INDEX_0);
    return (int32_t)k;
  #else
    int32_t r = 0;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      HEDLEY_DIAGNOSTIC_PUSH
      EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_
      uint64x2_t shifted = vshrq_n_u64(a_.neon_u64, 63);
      r =
        HEDLEY_STATIC_CAST(int32_t, vgetq_lane_u64(shifted, 0)) +
        (HEDLEY_STATIC_CAST(int32_t, vgetq_lane_u64(shifted, 1)) << 1);
      HEDLEY_DIAGNOSTIC_POP
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
        r |= (a_.u64[i] >> 63) << i;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_movemask_pd(a) easysimd_mm_movemask_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_movepi64_pi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_movepi64_pi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m64 r;
    r.i64[0] = a.i64[0];
    return r;
  #else
    easysimd__m64_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i64 = vget_low_s64(a_.neon_i64);
    #else
      r_.i64[0] = a_.i64[0];
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_movepi64_pi64(a) easysimd_mm_movepi64_pi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_movpi64_epi64 (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_movpi64_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdupq_n_s64(a.i64[0], 0);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd__m64_private a_ = easysimd__m64_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vcombine_s64(a_.neon_i64, vdup_n_s64(0));
    #else
      r_.i64[0] = a_.i64[0];
      r_.i64[1] = 0;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_movpi64_epi64(a) easysimd_mm_movpi64_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_min_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_min_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svmin_s16_x(svptrue_b16(), a.sve_i16, b.sve_i16);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_i16 = vminq_s16(a.neon_i16, b.neon_i16);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? a_.i16[i] : b_.i16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_min_epi16(a, b) easysimd_mm_min_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_min_epu8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_min_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svmin_u8_x(svptrue_b8(), a.sve_u8, b.sve_u8);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_u8 = vminq_u8(a.neon_u8, b.neon_u8);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = (a_.u8[i] < b_.u8[i]) ? a_.u8[i] : b_.u8[i];
    }
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_min_epu8(a, b) easysimd_mm_min_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_min_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_min_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svminnm_f64_x(svptrue_b64(), a.sve_f64, b.sve_f64);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128d res;
    res.neon_f64 = vminq_f64(a.neon_f64, b.neon_f64);
    return res;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = (a_.f64[i] < b_.f64[i]) ? a_.f64[i] : b_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_min_pd(a, b) easysimd_mm_min_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_min_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_min_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svmin_f64_z(svptrue_b64(), a.sve_f64, b.sve_f64);
    r.sve_f64 = svdupq_n_f64(r.f64[0], a.f64[1]);
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_min_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_min_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      float64x2_t temp = vminq_f64(a_.neon_f64, b_.neon_f64);
      r_.neon_f64 = vsetq_lane_f64(vgetq_lane(a_.neon_f64, 1), temp, 1);
    #else
      r_.f64[0] = (a_.f64[0] < b_.f64[0]) ? a_.f64[0] : b_.f64[0];
      r_.f64[1] = a_.f64[1];
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_min_sd(a, b) easysimd_mm_min_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_max_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_max_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16 = svmax_s16_x(pg, a.sve_i16, b.sve_i16);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_i16 = vmaxq_s16(a.neon_i16, b.neon_i16);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? a_.i16[i] : b_.i16[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_max_epi16(a, b) easysimd_mm_max_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_max_epu8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_max_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svmax_u8_x(svptrue_b8(), a.sve_u8, b.sve_u8);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_u8 = vmaxq_u8(a.neon_u8, b.neon_u8);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = (a_.u8[i] > b_.u8[i]) ? a_.u8[i] : b_.u8[i];
    }
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_max_epu8(a, b) easysimd_mm_max_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_max_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_max_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b64();
    r.sve_f64 = svmax_f64_x(pg, a.sve_f64, b.sve_f64);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_f64 = vmaxq_f64(a.neon_f64, b.neon_f64);
    return res;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = (a_.f64[i] > b_.f64[i]) ? a_.f64[i] : b_.f64[i];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_max_pd(a, b) easysimd_mm_max_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_max_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_max_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svmax_f64_z(svptrue_b64(), a.sve_f64, b.sve_f64);
    r.sve_f64 = svdupq_n_f64(r.f64[0], a.f64[1]);
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_max_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_max_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      float64x2_t temp = vmaxq_f64(a_.neon_f64, b_.neon_f64);
      r_.neon_f64 = vsetq_lane_f64(vgetq_lane(a_.neon_f64, 1), temp, 1);
    #else
      r_.f64[0] = (a_.f64[0] > b_.f64[0]) ? a_.f64[0] : b_.f64[0];
      r_.f64[1] = a_.f64[1];
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_max_sd(a, b) easysimd_mm_max_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_move_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_move_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.i64[1] = 0;
    return a;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vsetq_lane_s64(0, a_.neon_i64, 1);
    #else
      r_.i64[0] = a_.i64[0];
      r_.i64[1] = 0;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_move_epi64(a) easysimd_mm_move_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mul_epu32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_mul_epu32(a, b);
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_u64 = svmullb_u64(a.sve_u32, b.sve_u32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x2_t a_lo = vmovn_u64(a.neon_u64);
      uint32x2_t b_lo = vmovn_u64(b.neon_u64);
      r_.neon_u64 = vmull_u32(a_lo, b_lo);
    #else
      easysimd__m128i_private 
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = HEDLEY_STATIC_CAST(uint64_t, a_.u32[i * 2]) * HEDLEY_STATIC_CAST(uint64_t, b_.u32[i * 2]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_mul_epu32(a, b) easysimd_mm_mul_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_mul_epi64 (easysimd__m128i a, easysimd__m128i b) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    r_.i64 = a_.i64 * b_.i64;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] * b_.i64[i];
    }
  #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_mod_epi64 (easysimd__m128i a, easysimd__m128i b) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_PGI_30104)
    r_.i64 = a_.i64 % b_.i64;
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] % b_.i64[i];
    }
  #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mul_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_mul_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svmul_f64_z(svptrue_b64(), a.sve_f64, b.sve_f64);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vmulq_f64(a.neon_f64, b.neon_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 * b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[i] * b_.f64[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_mul_pd(a, b) easysimd_mm_mul_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mul_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_mul_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svmul_f64_z(svptrue_b64(), a.sve_f64, svdupq_n_f64(b.f64[0], 1.0));
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_mul_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_mul_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      float64x2_t temp = vmulq_f64(a_.neon_f64, b_.neon_f64);
      r_.neon_f64 = vsetq_lane_f64(vgetq_lane(a_.neon_f64, 1), temp, 1);
    #else
      r_.f64[0] = a_.f64[0] * b_.f64[0];
      r_.f64[1] = a_.f64[1];
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_mul_sd(a, b) easysimd_mm_mul_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_mul_su32 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE) && !defined(__PGI)
    return _mm_mul_su32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m64 r;
    r.u64[0] = HEDLEY_STATIC_CAST(uint64_t, a.u32[0]) * HEDLEY_STATIC_CAST(uint64_t, b.u32[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.u64[0] = vget_lane_u64(vget_low_u64(vmull_u32(vreinterpret_u32_s64(a.neon_i64), vreinterpret_u32_s64(b.neon_i64))), 0);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    r_.u64[0] = HEDLEY_STATIC_CAST(uint64_t, a_.u32[0]) * HEDLEY_STATIC_CAST(uint64_t, b_.u32[0]);

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_mul_su32(a, b) easysimd_mm_mul_su32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mulhi_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_mulhi_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16 = svmulh_s16_z(pg, a.sve_i16, b.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int16x4_t a3210 = vget_low_s16(a_.neon_i16);
      int16x4_t b3210 = vget_low_s16(b_.neon_i16);
      int32x4_t ab3210 = vmull_s16(a3210, b3210); /* 3333222211110000 */
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        int32x4_t ab7654 = vmull_high_s16(a_.neon_i16, b_.neon_i16);
        r_.neon_i16 = vuzp2q_s16(vreinterpretq_s16_s32(ab3210), vreinterpretq_s16_s32(ab7654));
      #else
        int16x4_t a7654 = vget_high_s16(a_.neon_i16);
        int16x4_t b7654 = vget_high_s16(b_.neon_i16);
        int32x4_t ab7654 = vmull_s16(a7654, b7654); /* 7777666655554444 */
        uint16x8x2_t rv = vuzpq_u16(vreinterpretq_u16_s32(ab3210), vreinterpretq_u16_s32(ab7654));
        r_.neon_u16 = rv.val[1];
      #endif
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (HEDLEY_STATIC_CAST(uint32_t, HEDLEY_STATIC_CAST(int32_t, a_.i16[i]) * HEDLEY_STATIC_CAST(int32_t, b_.i16[i])) >> 16));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_mulhi_epi16(a, b) easysimd_mm_mulhi_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mulhi_epu16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
    return _mm_mulhi_epu16(a, b);
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_u16 = svmulh_u16_z(svptrue_b16(), a.sve_u16, b.sve_u16);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint16x4_t a3210 = vget_low_u16(a.neon_u16);
      uint16x4_t b3210 = vget_low_u16(b.neon_u16);
      uint32x4_t ab3210 = vmull_u16(a3210, b3210); /* 3333222211110000 */
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        uint32x4_t ab7654 = vmull_high_u16(a.neon_u16, b.neon_u16);
        r_.neon_u16 = vuzp2q_u16(vreinterpretq_u16_u32(ab3210), vreinterpretq_u16_u32(ab7654));
      #else
        uint16x4_t a7654 = vget_high_u16(a.neon_u16);
        uint16x4_t b7654 = vget_high_u16(b.neon_u16);
        uint32x4_t ab7654 = vmull_u16(a7654, b7654); /* 7777666655554444 */
        uint16x8x2_t neon_r = vuzpq_u16(vreinterpretq_u16_u32(ab3210), vreinterpretq_u16_u32(ab7654));
        r_.neon_u16 = neon_r.val[1];
      #endif
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) * HEDLEY_STATIC_CAST(uint32_t, b_.u16[i]) >> 16);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_mulhi_epu16(a, b) easysimd_mm_mulhi_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mullo_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_mullo_epi16(a, b);
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i16 = svmul_s16_z(svptrue_b16(), a.sve_i16, b.sve_i16);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i16 = vmulq_s16(a.neon_i16, b.neon_i16);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) * HEDLEY_STATIC_CAST(uint32_t, b_.u16[i]));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_mullo_epi16(a, b) easysimd_mm_mullo_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_or_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_or_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_u64 = svorr_u64_z(svptrue_b64(), a.sve_u64, b.sve_u64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f | b_.i32f;
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vorrq_s64(a_.neon_i64, b_.neon_i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = a_.i32f[i] | b_.i32f[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_or_pd(a, b) easysimd_mm_or_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_or_si128 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_or_si128(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svorr_s32_z(svptrue_b32(), a.sve_i32, b.sve_i32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i32 = vorrq_s32(a.neon_i32, b.neon_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_POWER_ALTIVEC_P6_NATIVE)
      r_.altivec_i32 = vec_or(a_.altivec_i32, b_.altivec_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f | b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = a_.i32f[i] | b_.i32f[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_or_si128(a, b) easysimd_mm_or_si128(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_packs_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_packs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svuzp1_s8(svqxtnb_s16(a.sve_i16), svqxtnb_s16(b.sve_i16));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i8 = vqmovn_high_s16(vqmovn_s16(a.neon_i16), b.neon_i16);
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i8 = vcombine_s8(vqmovn_s16(a_.neon_i16), vqmovn_s16(b_.neon_i16));
    #elif defined(EASYSIMD_CONVERT_VECTOR_) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      int16_t EASYSIMD_VECTOR(32) v = EASYSIMD_SHUFFLE_VECTOR_(16, 32, a_.i16, b_.i16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
      const int16_t EASYSIMD_VECTOR(32) min = { INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN, INT8_MIN };
      const int16_t EASYSIMD_VECTOR(32) max = { INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MAX };

      int16_t m EASYSIMD_VECTOR(32);
      m = HEDLEY_REINTERPRET_CAST(__typeof__(m), v < min);
      v = (v & ~m) | (min & m);

      m = v > max;
      v = (v & ~m) | (max & m);

      EASYSIMD_CONVERT_VECTOR_(r_.i8, v);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        int16_t v = (i < (sizeof(a_.i16) / sizeof(a_.i16[0]))) ? a_.i16[i] : b_.i16[i & 7];
        r_.i8[i] = (v < INT8_MIN) ? INT8_MIN : ((v > INT8_MAX) ? INT8_MAX : HEDLEY_STATIC_CAST(int8_t, v));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_packs_epi16(a, b) easysimd_mm_packs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_packs_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_packs_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svuzp1_s16(svqxtnb_s32(a.sve_i32), svqxtnb_s32(b.sve_i32));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i16 = vqmovn_high_s32(vqmovn_s32(a.neon_i32), b.neon_i32);
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i16 = vcombine_s16(vqmovn_s32(a_.neon_i32), vqmovn_s32(b_.neon_i32));
    #elif defined(EASYSIMD_X86_SSE2_NATIVE)
      r_.sse_m128i = _mm_packs_epi32(a_.sse_m128i, b_.sse_m128i);
    #elif defined(EASYSIMD_CONVERT_VECTOR_) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      int32_t EASYSIMD_VECTOR(32) v = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.i32, b_.i32, 0, 1, 2, 3, 4, 5, 6, 7);
      const int32_t EASYSIMD_VECTOR(32) min = { INT16_MIN, INT16_MIN, INT16_MIN, INT16_MIN, INT16_MIN, INT16_MIN, INT16_MIN, INT16_MIN };
      const int32_t EASYSIMD_VECTOR(32) max = { INT16_MAX, INT16_MAX, INT16_MAX, INT16_MAX, INT16_MAX, INT16_MAX, INT16_MAX, INT16_MAX };

      int32_t m EASYSIMD_VECTOR(32);
      m = HEDLEY_REINTERPRET_CAST(__typeof__(m), v < min);
      v = (v & ~m) | (min & m);

      m = HEDLEY_REINTERPRET_CAST(__typeof__(m), v > max);
      v = (v & ~m) | (max & m);

      EASYSIMD_CONVERT_VECTOR_(r_.i16, v);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        int32_t v = (i < (sizeof(a_.i32) / sizeof(a_.i32[0]))) ? a_.i32[i] : b_.i32[i & 3];
        r_.i16[i] = (v < INT16_MIN) ? INT16_MIN : ((v > INT16_MAX) ? INT16_MAX : HEDLEY_STATIC_CAST(int16_t, v));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_packs_epi32(a, b) easysimd_mm_packs_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_packus_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_packus_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svuzp1_u8(svqxtunb_s16(a.sve_i16), svqxtunb_s16(b.sve_i16));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      #if defined(EASYSIMD_BUG_CLANG_46840)
        r_.neon_u8 = vqmovun_high_s16(vreinterpret_s8_u8(vqmovun_s16(a_.neon_i16)), b_.neon_i16);
      #else
        r_.neon_u8 = vqmovun_high_s16(vqmovun_s16(a_.neon_i16), b_.neon_i16);
      #endif
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u8 =
        vcombine_u8(
          vqmovun_s16(a_.neon_i16),
          vqmovun_s16(b_.neon_i16)
        );
    #elif defined(EASYSIMD_CONVERT_VECTOR_) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      int16_t v EASYSIMD_VECTOR(32) = EASYSIMD_SHUFFLE_VECTOR_(16, 32, a_.i16, b_.i16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

      v &= ~(v >> 15);
      v |= HEDLEY_REINTERPRET_CAST(__typeof__(v), v > UINT8_MAX);

      EASYSIMD_CONVERT_VECTOR_(r_.i8, v);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        int16_t v = (i < (sizeof(a_.i16) / sizeof(a_.i16[0]))) ? a_.i16[i] : b_.i16[i & 7];
        r_.u8[i] = (v < 0) ? UINT8_C(0) : ((v > UINT8_MAX) ? UINT8_MAX : HEDLEY_STATIC_CAST(uint8_t, v));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_packus_epi16(a, b) easysimd_mm_packus_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_pause (void) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_pause();
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_pause() (easysimd_mm_pause())
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sad_epu8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sad_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u8 = svabd_u8_x(svptrue_b8(), a.sve_u8, b.sve_u8);
    r.u64[0] = svaddv_u8(svwhilele_b8(1, 8), r.sve_u8);
    r.u64[1] = svaddv_u8(svwhilege_b8(15, 8), r.sve_u8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      const uint16x8_t t = vpaddlq_u8(vabdq_u8(a_.neon_u8, b_.neon_u8));
      r_.neon_u64 = vcombine_u64(
        vpaddl_u32(vpaddl_u16(vget_low_u16(t))),
        vpaddl_u32(vpaddl_u16(vget_high_u16(t))));
    #else
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        uint16_t tmp = 0;
        EASYSIMD_VECTORIZE_REDUCTION(+:tmp)
        for (size_t j = 0 ; j < ((sizeof(r_.u8) / sizeof(r_.u8[0])) / 2) ; j++) {
          const size_t e = j + (i * 8);
          tmp += (a_.u8[e] > b_.u8[e]) ? (a_.u8[e] - b_.u8[e]) : (b_.u8[e] - a_.u8[e]);
        }
        r_.i64[i] = tmp;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sad_epu8(a, b) easysimd_mm_sad_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set_epi8 (int8_t e15, int8_t e14, int8_t e13, int8_t e12,
       int8_t e11, int8_t e10, int8_t  e9, int8_t  e8,
       int8_t  e7, int8_t  e6, int8_t  e5, int8_t  e4,
       int8_t  e3, int8_t  e2, int8_t  e1, int8_t  e0) {

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set_epi8(
      e15, e14, e13, e12, e11, e10,  e9,  e8,
       e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i8 = svdupq_n_s8(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    #else
      r_.i8[ 0] =  e0;
      r_.i8[ 1] =  e1;
      r_.i8[ 2] =  e2;
      r_.i8[ 3] =  e3;
      r_.i8[ 4] =  e4;
      r_.i8[ 5] =  e5;
      r_.i8[ 6] =  e6;
      r_.i8[ 7] =  e7;
      r_.i8[ 8] =  e8;
      r_.i8[ 9] =  e9;
      r_.i8[10] = e10;
      r_.i8[11] = e11;
      r_.i8[12] = e12;
      r_.i8[13] = e13;
      r_.i8[14] = e14;
      r_.i8[15] = e15;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set_epi8(e15, e14, e13, e12, e11, e10,  e9,  e8,  e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0) easysimd_mm_set_epi8(e15, e14, e13, e12, e11, e10,  e9,  e8,  e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set_epi16 (int16_t e7, int16_t e6, int16_t e5, int16_t e4,
        int16_t e3, int16_t e2, int16_t e1, int16_t e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set_epi16(e7, e6, e5, e4, e3, e2, e1, e0);
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i16 = svdupq_n_s16(e0, e1, e2, e3, e4, e5, e6, e7);
    #else
      r_.i16[0] = e0;
      r_.i16[1] = e1;
      r_.i16[2] = e2;
      r_.i16[3] = e3;
      r_.i16[4] = e4;
      r_.i16[5] = e5;
      r_.i16[6] = e6;
      r_.i16[7] = e7;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set_epi16(e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0) easysimd_mm_set_epi16(e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_loadu_si16 (void const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && ( \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0) || \
      HEDLEY_INTEL_VERSION_CHECK(20,21,1))
    return _mm_loadu_si16(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svdup_n_s16(0);
    r.sve_i16 = svinsr_n_s16(r.sve_i16, *(int16_t const* )mem_addr);
    return r;
  #else
    int16_t val;
    easysimd_memcpy(&val, mem_addr, sizeof(val));
    return easysimd_x_mm_cvtsi16_si128(val);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_loadu_si16(mem_addr) easysimd_mm_loadu_si16(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set_epi32 (int32_t e3, int32_t e2, int32_t e1, int32_t e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set_epi32(e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svdupq_n_s32(e0, e1, e2, e3);
    return r;
  #else
    easysimd__m128i_private r_;
    r_.i32[0] = e0;
    r_.i32[1] = e1;
    r_.i32[2] = e2;
    r_.i32[3] = e3;
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set_epi32(e3,  e2,  e1,  e0) easysimd_mm_set_epi32(e3,  e2,  e1,  e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_loadu_si32 (void const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && ( \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0) || \
      HEDLEY_INTEL_VERSION_CHECK(20,21,1))
    return _mm_loadu_si32(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svdup_n_s32(0);
    r.sve_i32 = svinsr_n_s32(r.sve_i32, *(int32_t const* )mem_addr);
    return r;
  #else
    int32_t val;
    easysimd_memcpy(&val, mem_addr, sizeof(val));
    return easysimd_mm_cvtsi32_si128(val);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_loadu_si32(mem_addr) easysimd_mm_loadu_si32(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set_epi64 (easysimd__m64 e1, easysimd__m64 e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_set_epi64(e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdupq_n_s64(e0.i64[0], e1.i64[0]);
    return r;
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vcombine_s64(easysimd__m64_to_neon_i64(e0), easysimd__m64_to_neon_i64(e1));
    #else
      r_.m64[0] = e0;
      r_.m64[1] = e1;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set_epi64(e1, e0) (easysimd_mm_set_epi64((e1), (e0)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set_epi64x (int64_t e1, int64_t e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && (!defined(HEDLEY_MSVC_VERSION) || HEDLEY_MSVC_VERSION_CHECK(19,0,0))
    return _mm_set_epi64x(e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdupq_n_s64(e0, e1);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
//    SET64x2(r.neon_i64, e0, e1);
    EASYSIMD_ALIGN_LIKE_16(int64x2_t) int64_t data[2] = {e0, e1};
    r.neon_i64 = vld1q_s64(data);
    return r;
  #else
    easysimd__m128i_private r_;

    r_.i64[0] = e0;
    r_.i64[1] = e1;

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set_epi64x(e1, e0) easysimd_mm_set_epi64x(e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_loadu_si64 (void const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && ( \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0) || \
      HEDLEY_GCC_VERSION_CHECK(11,0,0) || \
      HEDLEY_INTEL_VERSION_CHECK(20,21,1))
    return _mm_loadu_si64(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdup_n_s64(0);
    r.sve_i64 = svinsr_n_s64(r.sve_i64, *(int64_t const* )mem_addr);
    return r;
  #else
  int64_t val;
    easysimd_memcpy(&val, mem_addr, sizeof(val));
    return easysimd_mm_cvtsi64_si128(val);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_loadu_si64(mem_addr) easysimd_mm_loadu_si64(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_set_epu8 (uint8_t e15, uint8_t e14, uint8_t e13, uint8_t e12,
         uint8_t e11, uint8_t e10, uint8_t  e9, uint8_t  e8,
         uint8_t  e7, uint8_t  e6, uint8_t  e5, uint8_t  e4,
         uint8_t  e3, uint8_t  e2, uint8_t  e1, uint8_t  e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set_epi8(
      HEDLEY_STATIC_CAST(char, e15), HEDLEY_STATIC_CAST(char, e14), HEDLEY_STATIC_CAST(char, e13), HEDLEY_STATIC_CAST(char, e12),
      HEDLEY_STATIC_CAST(char, e11), HEDLEY_STATIC_CAST(char, e10), HEDLEY_STATIC_CAST(char,  e9), HEDLEY_STATIC_CAST(char,  e8),
      HEDLEY_STATIC_CAST(char,  e7), HEDLEY_STATIC_CAST(char,  e6), HEDLEY_STATIC_CAST(char,  e5), HEDLEY_STATIC_CAST(char,  e4),
      HEDLEY_STATIC_CAST(char,  e3), HEDLEY_STATIC_CAST(char,  e2), HEDLEY_STATIC_CAST(char,  e1), HEDLEY_STATIC_CAST(char,  e0));
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      EASYSIMD_ALIGN_LIKE_16(uint8x16_t) uint8_t data[16] = {
        e0,  e1,  e2,  e3,
        e4,  e5,  e6,  e7,
        e8,  e9,  e10, e11,
        e12, e13, e14, e15};
      r_.neon_u8 = vld1q_u8(data);
    #else
      r_.u8[ 0] =  e0; r_.u8[ 1] =  e1; r_.u8[ 2] =  e2; r_.u8[ 3] =  e3;
      r_.u8[ 4] =  e4; r_.u8[ 5] =  e5; r_.u8[ 6] =  e6; r_.u8[ 7] =  e7;
      r_.u8[ 8] =  e8; r_.u8[ 9] =  e9; r_.u8[10] = e10; r_.u8[11] = e11;
      r_.u8[12] = e12; r_.u8[13] = e13; r_.u8[14] = e14; r_.u8[15] = e15;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_set_epu16 (uint16_t e7, uint16_t e6, uint16_t e5, uint16_t e4,
          uint16_t e3, uint16_t e2, uint16_t e1, uint16_t e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set_epi16(
      HEDLEY_STATIC_CAST(short,  e7), HEDLEY_STATIC_CAST(short,  e6), HEDLEY_STATIC_CAST(short,  e5), HEDLEY_STATIC_CAST(short,  e4),
      HEDLEY_STATIC_CAST(short,  e3), HEDLEY_STATIC_CAST(short,  e2), HEDLEY_STATIC_CAST(short,  e1), HEDLEY_STATIC_CAST(short,  e0));
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      EASYSIMD_ALIGN_LIKE_16(uint16x8_t) uint16_t data[8] = { e0, e1, e2, e3, e4, e5, e6, e7 };
      r_.neon_u16 = vld1q_u16(data);
    #else
      r_.u16[0] = e0; r_.u16[1] = e1; r_.u16[2] = e2; r_.u16[3] = e3;
      r_.u16[4] = e4; r_.u16[5] = e5; r_.u16[6] = e6; r_.u16[7] = e7;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_set_epu32 (uint32_t e3, uint32_t e2, uint32_t e1, uint32_t e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set_epi32(
      HEDLEY_STATIC_CAST(int,  e3), HEDLEY_STATIC_CAST(int,  e2), HEDLEY_STATIC_CAST(int,  e1), HEDLEY_STATIC_CAST(int,  e0));
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      EASYSIMD_ALIGN_LIKE_16(uint32x4_t) uint32_t data[4] = { e0, e1, e2, e3 };
      r_.neon_u32 = vld1q_u32(data);
    #else
      r_.u32[0] = e0;
      r_.u32[1] = e1;
      r_.u32[2] = e2;
      r_.u32[3] = e3;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_set_epu64x (uint64_t e1, uint64_t e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && (!defined(HEDLEY_MSVC_VERSION) || HEDLEY_MSVC_VERSION_CHECK(19,0,0))
    return _mm_set_epi64x(HEDLEY_STATIC_CAST(int64_t,  e1), HEDLEY_STATIC_CAST(int64_t,  e0));
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      EASYSIMD_ALIGN_LIKE_16(uint64x2_t) uint64_t data[2] = {e0, e1};
      r_.neon_u64 = vld1q_u64(data);
    #else
      r_.u64[0] = e0;
      r_.u64[1] = e1;
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_set_sd (easysimd_float64 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set_sd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdupq_n_f64(a, 0.0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 =  vsetq_lane_f64(a, vdupq_n_f64(EASYSIMD_FLOAT64_C(0.0)), 0);
    return r;
  #else
    return easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(0.0), a);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set_sd(a) easysimd_mm_set_sd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set1_epi8 (int8_t a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set1_epi8(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svdup_n_s8(a);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i8 = vdupq_n_s8(a);
    return r;
  #else
    easysimd__m128i_private r_;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = a;
    }
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set1_epi8(a) easysimd_mm_set1_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set1_epi16 (int16_t a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set1_epi16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svdup_n_s16(a);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i16 = vdupq_n_s16(a);
    return r;
  #else
    easysimd__m128i_private r_;

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a;
      }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set1_epi16(a) easysimd_mm_set1_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set1_epi32 (int32_t a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_set1_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svdup_n_s32(a);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i32 = vdupq_n_s32(a);
    return r;
  #else
    easysimd__m128i_private r_;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a;
    }
    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set1_epi32(a) easysimd_mm_set1_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set1_epi64x (int64_t a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && (!defined(HEDLEY_MSVC_VERSION) || HEDLEY_MSVC_VERSION_CHECK(19,0,0))
    return _mm_set1_epi64x(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdup_n_s64(a);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i64 = vdupq_n_s64(a);
    return r;
  #else
    easysimd__m128i_private r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set1_epi64x(a) easysimd_mm_set1_epi64x(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_set1_epi64 (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_set1_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdup_n_s64(a.i64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i64 = vdupq_n_s64(a.i64[0]);
    return r;
  #else
    easysimd__m64_private a_ = easysimd__m64_to_private(a);
    return easysimd_mm_set1_epi64x(a_.i64[0]);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_set1_epi64(a) easysimd_mm_set1_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_set1_epu8 (uint8_t value) {
    return easysimd_mm_set1_epi8(HEDLEY_STATIC_CAST(int8_t, value));
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_set1_epu16 (uint16_t value) {
    return easysimd_mm_set1_epi16(HEDLEY_STATIC_CAST(int16_t, value));
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_set1_epu32 (uint32_t value) {
    return easysimd_mm_set1_epi32(HEDLEY_STATIC_CAST(int32_t, value));
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_set1_epu64 (uint64_t value) {
    return easysimd_mm_set1_epi64x(HEDLEY_STATIC_CAST(int64_t, value));
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_setr_epi8 (int8_t e15, int8_t e14, int8_t e13, int8_t e12,
        int8_t e11, int8_t e10, int8_t  e9, int8_t  e8,
        int8_t  e7, int8_t  e6, int8_t  e5, int8_t  e4,
        int8_t  e3, int8_t  e2, int8_t  e1, int8_t  e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_setr_epi8(
      e15, e14, e13, e12, e11, e10,  e9,    e8,
      e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svdupq_n_s8(e15, e14, e13, e12, e11, e10,  e9, e8,
                            e7,  e6,  e5,  e4,  e3,  e2,  e1, e0);
    return r;
  #else
    return easysimd_mm_set_epi8(
      e0, e1, e2, e3, e4, e5, e6, e7,
      e8, e9, e10, e11, e12, e13, e14, e15);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_setr_epi8(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) easysimd_mm_setr_epi8(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_setr_epi16 (int16_t e7, int16_t e6, int16_t e5, int16_t e4,
         int16_t e3, int16_t e2, int16_t e1, int16_t e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_setr_epi16(e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svdupq_n_s16(e7, e6, e5, e4, e3, e2, e1, e0);
    return r;
  #else
    return easysimd_mm_set_epi16(e0, e1, e2, e3, e4, e5, e6, e7);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_setr_epi16(e7, e6, e5, e4, e3, e2, e1, e0) easysimd_mm_setr_epi16(e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_setr_epi32 (int32_t e3, int32_t e2, int32_t e1, int32_t e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_setr_epi32(e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svdupq_n_s32(e3, e2, e1, e0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    int32_t data[4] = {e3, e2, e1, e0};
    r.neon_i32 = vld1q_s32(data);
    return r;
  #else
    return easysimd_mm_set_epi32(e0, e1, e2, e3);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_setr_epi32(e3, e2, e1, e0) easysimd_mm_setr_epi32(e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_setr_epi64 (easysimd__m64 e1, easysimd__m64 e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_setr_epi64(e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdupq_n_s64(e1.i64[0], e0.i64[0]);
    return r;
  #else
    return easysimd_mm_set_epi64(e0, e1);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_setr_epi64(e1, e0) (easysimd_mm_setr_epi64((e1), (e0)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_setr_pd (easysimd_float64 e1, easysimd_float64 e0) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_setr_pd(e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdupq_n_f64(e1, e0);
    return r;
  #else
    return easysimd_mm_set_pd(e0, e1);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_setr_pd(e1, e0) easysimd_mm_setr_pd(e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_setzero_pd (void) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_setzero_pd();
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdup_n_f64(0.0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vdupq_n_f64(0.0);
    return r;
  #else
    return easysimd_mm_castsi128_pd(easysimd_mm_setzero_si128());
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_setzero_pd() easysimd_mm_setzero_pd()
#endif

#if defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_undefined_pd (void) {
  easysimd__m128d_private r_;

  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD__HAVE_UNDEFINED128)
    r_.n = _mm_undefined_pd();
  #elif !defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
    r_ = easysimd__m128d_to_private(easysimd_mm_setzero_pd());
  #endif

  return easysimd__m128d_from_private(r_);
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_undefined_pd() easysimd_mm_undefined_pd()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_undefined_si128 (void) {
  easysimd__m128i_private r_;

  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD__HAVE_UNDEFINED128)
    r_.n = _mm_undefined_si128();
  #elif !defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
    r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128());
  #endif

  return easysimd__m128i_from_private(r_);
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_undefined_si128() (easysimd_mm_undefined_si128())
#endif

#if defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
HEDLEY_DIAGNOSTIC_POP
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_setone_pd (void) {
  return easysimd_mm_castps_pd(easysimd_x_mm_setone_ps());
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_setone_si128 (void) {
  return easysimd_mm_castps_si128(easysimd_x_mm_setone_ps());
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shuffle_epi32 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  svuint32_t index = svdupq_n_u32(imm8 & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
  r.sve_i32 = svtbl_s32(a.sve_i32, index);
  return r;
#else
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a);

  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
    r_.i32[i] = a_.i32[(imm8 >> (i * 2)) & 3];
  }

  return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_shuffle_epi32(a, imm8) _mm_shuffle_epi32((a), (imm8))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm_shuffle_epi32(a, imm8) ({easysimd__m128i r; \
    r.neon_i32 = vmovq_n_s32(vgetq_lane_s32(a.neon_i32, (imm8) & (0x3))); \
    r.neon_i32 = vsetq_lane_s32(vgetq_lane_s32(a.neon_i32, ((imm8) >> 2) & 0x3), r.neon_i32, 1); \
    r.neon_i32 = vsetq_lane_s32(vgetq_lane_s32(a.neon_i32, ((imm8) >> 4) & 0x3), r.neon_i32, 2); \
    r.neon_i32 = vsetq_lane_s32(vgetq_lane_s32(a.neon_i32, ((imm8) >> 6) & 0x3), r.neon_i32, 3); \
    r; \
  })
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_shuffle_epi32(a, imm8) (__extension__ ({ \
      const easysimd__m128i_private easysimd__tmp_a_ = easysimd__m128i_to_private(a); \
      easysimd__m128i_from_private((easysimd__m128i_private) { .i32 = \
        EASYSIMD_SHUFFLE_VECTOR_(32, 16, \
          (easysimd__tmp_a_).i32, \
          (easysimd__tmp_a_).i32, \
          ((imm8)     ) & 3, \
          ((imm8) >> 2) & 3, \
          ((imm8) >> 4) & 3, \
          ((imm8) >> 6) & 3) }); }))
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm_shuffle_epi32(a, imm8) \
    (__extension__ ({ \
      const int32x4_t easysimd_mm_shuffle_epi32_a_ = easysimd__m128i_to_neon_i32(a); \
      int32x4_t easysimd_mm_shuffle_epi32_r_; \
      easysimd_mm_shuffle_epi32_r_ = vmovq_n_s32(vgetq_lane_s32(easysimd_mm_shuffle_epi32_a_, (imm8) & (0x3))); \
      easysimd_mm_shuffle_epi32_r_ = vsetq_lane_s32(vgetq_lane_s32(easysimd_mm_shuffle_epi32_a_, ((imm8) >> 2) & 0x3), easysimd_mm_shuffle_epi32_r_, 1); \
      easysimd_mm_shuffle_epi32_r_ = vsetq_lane_s32(vgetq_lane_s32(easysimd_mm_shuffle_epi32_a_, ((imm8) >> 4) & 0x3), easysimd_mm_shuffle_epi32_r_, 2); \
      easysimd_mm_shuffle_epi32_r_ = vsetq_lane_s32(vgetq_lane_s32(easysimd_mm_shuffle_epi32_a_, ((imm8) >> 6) & 0x3), easysimd_mm_shuffle_epi32_r_, 3); \
      vreinterpretq_s64_s32(easysimd_mm_shuffle_epi32_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_shuffle_epi32(a, imm8) easysimd_mm_shuffle_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_shuffle_pd (easysimd__m128d a, easysimd__m128d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdupq_n_f64(a.f64[imm8 & 0x1], b.f64[(imm8 >> 1) & 0x1]);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    r_.f64[0] = ((imm8 & 1) == 0) ? a_.f64[0] : a_.f64[1];
    r_.f64[1] = ((imm8 & 2) == 0) ? b_.f64[0] : b_.f64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(__PGI)
  #define easysimd_mm_shuffle_pd(a, b, imm8) _mm_shuffle_pd((a), (b), (imm8))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm_shuffle_pd(a, b, imm8) \
  ({ easysimd__m128 r; \
      r.neon_f64 = vmovq_n_f64(vgetq_lane_f64(a.neon_f64, (imm8) & (0x1)));  \
      r.neon_f64 = vsetq_lane_f64(vgetq_lane_f64(b.neon_f64, ((imm8) >> 1) & 0x1), r.neon_f64, 1); \
      r; \
  })
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_shuffle_pd(a, b, imm8) (__extension__ ({ \
      easysimd__m128d_from_private((easysimd__m128d_private) { .f64 = \
        EASYSIMD_SHUFFLE_VECTOR_(64, 16, \
          easysimd__m128d_to_private(a).f64, \
          easysimd__m128d_to_private(b).f64, \
          (((imm8)     ) & 1), \
          (((imm8) >> 1) & 1) + 2) }); }))
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_shuffle_pd(a, b, imm8) easysimd_mm_shuffle_pd(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shufflehi_epi16 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  svuint16_t svindex = svdupq_n_u16(0, 1, 2, 3, (imm8 >> 0) & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
             svindex = svadd_n_u16_m(svwhilege_b16(8, 5), svindex, 4);
  a.sve_u16 = svtbl_u16(a.sve_u16, svindex);
  return a;
#else
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < ((sizeof(a_.i16) / sizeof(a_.i16[0])) / 2) ; i++) {
    r_.i16[i] = a_.i16[i];
  }
  for (size_t i = ((sizeof(a_.i16) / sizeof(a_.i16[0])) / 2) ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
    r_.i16[i] = a_.i16[((imm8 >> ((i - 4) * 2)) & 3) + 4];
  }

  return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_shufflehi_epi16(a, imm8) _mm_shufflehi_epi16((a), (imm8))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_shufflehi_epi16(a, imm8) (__extension__ ({ \
      const easysimd__m128i_private easysimd__tmp_a_ = easysimd__m128i_to_private(a); \
      easysimd__m128i_from_private((easysimd__m128i_private) { .i16 = \
        EASYSIMD_SHUFFLE_VECTOR_(16, 16, \
          (easysimd__tmp_a_).i16, \
          (easysimd__tmp_a_).i16, \
          0, 1, 2, 3, \
          (((imm8)     ) & 3) + 4, \
          (((imm8) >> 2) & 3) + 4, \
          (((imm8) >> 4) & 3) + 4, \
          (((imm8) >> 6) & 3) + 4) }); }))
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm_shufflehi_epi16(a, imm8) \
    (__extension__ ({ \
      int16x8_t easysimd_mm_shufflehi_epi16_a_ = easysimd__m128i_to_neon_i16(a); \
      int16x8_t easysimd_mm_shufflehi_epi16_r_ = easysimd_mm_shufflehi_epi16_a_; \
      easysimd_mm_shufflehi_epi16_r_ = vsetq_lane_s16(vgetq_lane_s16(easysimd_mm_shufflehi_epi16_a_, (((imm8)     ) & 0x3) + 4), easysimd_mm_shufflehi_epi16_r_, 4); \
      easysimd_mm_shufflehi_epi16_r_ = vsetq_lane_s16(vgetq_lane_s16(easysimd_mm_shufflehi_epi16_a_, (((imm8) >> 2) & 0x3) + 4), easysimd_mm_shufflehi_epi16_r_, 5); \
      easysimd_mm_shufflehi_epi16_r_ = vsetq_lane_s16(vgetq_lane_s16(easysimd_mm_shufflehi_epi16_a_, (((imm8) >> 4) & 0x3) + 4), easysimd_mm_shufflehi_epi16_r_, 6); \
      easysimd_mm_shufflehi_epi16_r_ = vsetq_lane_s16(vgetq_lane_s16(easysimd_mm_shufflehi_epi16_a_, (((imm8) >> 6) & 0x3) + 4), easysimd_mm_shufflehi_epi16_r_, 7); \
      easysimd__m128i_from_neon_i16(easysimd_mm_shufflehi_epi16_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_shufflehi_epi16(a, imm8) easysimd_mm_shufflehi_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_shufflelo_epi16 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  svuint16_t index = svdupq_n_u16((imm8 >> 0) & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03, 4, 5, 6, 7);
  r.sve_i16 = svtbl_s16(a.sve_i16, index);
  return r;
#else
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a);

  for (size_t i = 0 ; i < ((sizeof(r_.i16) / sizeof(r_.i16[0])) / 2) ; i++) {
    r_.i16[i] = a_.i16[((imm8 >> (i * 2)) & 3)];
  }
  EASYSIMD_VECTORIZE
  for (size_t i = ((sizeof(a_.i16) / sizeof(a_.i16[0])) / 2) ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
    r_.i16[i] = a_.i16[i];
  }

  return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_shufflelo_epi16(a, imm8) _mm_shufflelo_epi16((a), (imm8))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_shufflelo_epi16(a, imm8) (__extension__ ({ \
      const easysimd__m128i_private easysimd__tmp_a_ = easysimd__m128i_to_private(a); \
      easysimd__m128i_from_private((easysimd__m128i_private) { .i16 = \
        EASYSIMD_SHUFFLE_VECTOR_(16, 16, \
          (easysimd__tmp_a_).i16, \
          (easysimd__tmp_a_).i16, \
          (((imm8)     ) & 3), \
          (((imm8) >> 2) & 3), \
          (((imm8) >> 4) & 3), \
          (((imm8) >> 6) & 3), \
          4, 5, 6, 7) }); }))
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm_shufflelo_epi16(a, imm8) \
    (__extension__({ \
      int16x8_t easysimd_mm_shufflelo_epi16_a_ = easysimd__m128i_to_neon_i16(a); \
      int16x8_t easysimd_mm_shufflelo_epi16_r_ = easysimd_mm_shufflelo_epi16_a_; \
      easysimd_mm_shufflelo_epi16_r_ = vsetq_lane_s16(vgetq_lane_s16(easysimd_mm_shufflelo_epi16_a_, (((imm8)     ) & 0x3)), easysimd_mm_shufflelo_epi16_r_, 0); \
      easysimd_mm_shufflelo_epi16_r_ = vsetq_lane_s16(vgetq_lane_s16(easysimd_mm_shufflelo_epi16_a_, (((imm8) >> 2) & 0x3)), easysimd_mm_shufflelo_epi16_r_, 1); \
      easysimd_mm_shufflelo_epi16_r_ = vsetq_lane_s16(vgetq_lane_s16(easysimd_mm_shufflelo_epi16_a_, (((imm8) >> 4) & 0x3)), easysimd_mm_shufflelo_epi16_r_, 2); \
      easysimd_mm_shufflelo_epi16_r_ = vsetq_lane_s16(vgetq_lane_s16(easysimd_mm_shufflelo_epi16_a_, (((imm8) >> 6) & 0x3)), easysimd_mm_shufflelo_epi16_r_, 3); \
      easysimd__m128i_from_neon_i16(easysimd_mm_shufflelo_epi16_r_); \
    }))
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_shufflelo_epi16(a, imm8) easysimd_mm_shufflelo_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sll_epi16 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sll_epi16(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svlsl_n_s16_x(svptrue_b16(), a.sve_i16, count.u64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    if (HEDLEY_LIKELY((count.neon_i64[0]) >= 0 && (count.neon_i64[0]) < 16)) {
      r.neon_i16 = vshlq_s16(a.neon_i16, vdupq_n_s16(HEDLEY_STATIC_CAST(int16_t, count.neon_i64[0])));
    } else {
      r.neon_i16 = vdupq_n_s16(0);
    }
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    if (count_.u64[0] > 15)
      return easysimd_mm_setzero_si128();

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u16 = (a_.u16 << count_.u64[0]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (a_.u16[i] << count_.u64[0]));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sll_epi16(a, count) easysimd_mm_sll_epi16((a), (count))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sll_epi32 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sll_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svlsl_n_s32_x(svptrue_b32(), a.sve_i32, count.u64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      easysimd__m128i r;
      if (HEDLEY_LIKELY((count.neon_i64[0]) >= 0 && (count.neon_i64[0]) < 32)) {
          r.neon_i32 = vshlq_s32(a.neon_i32, vdupq_n_s32(HEDLEY_STATIC_CAST(int32_t, count.neon_i64[0])));
      } else {
          r.neon_i32 = vdupq_n_s32(0);
      }
      return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    if (count_.u64[0] > 31)
      return easysimd_mm_setzero_si128();

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u32 = (a_.u32 << count_.u64[0]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (a_.u32[i] << count_.u64[0]));
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sll_epi32(a, count) (easysimd_mm_sll_epi32(a, (count)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sll_epi64 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sll_epi64(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svlsl_n_s64_x(svptrue_b64(), a.sve_i64, count.u64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_u64 = vshlq_u64(a.neon_u64, vdupq_n_s64(HEDLEY_STATIC_CAST(int64_t, count.u64[0])));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    if (count_.u64[0] > 63)
      return easysimd_mm_setzero_si128();

    const int_fast16_t s = HEDLEY_STATIC_CAST(int_fast16_t, count_.u64[0]);

      #if !defined(EASYSIMD_BUG_GCC_94488)
        EASYSIMD_VECTORIZE
      #endif
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = a_.u64[i] << s;
      }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sll_epi64(a, count) (easysimd_mm_sll_epi64(a, (count)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_sqrt_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sqrt_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsqrt_f64_z(svptrue_b64(), a.sve_f64);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vsqrtq_f64(a.neon_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    #if defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_sqrt(a_.f64[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sqrt_pd(a) easysimd_mm_sqrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_sqrt_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sqrt_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d tmp, r;
    tmp.sve_f64 = svsqrt_f64_z(svptrue_b64(), b.sve_f64);
    r.sve_f64 = svdupq_n_f64(tmp.f64[0], a.f64[1]);
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_sqrt_pd(b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_sqrt_pd(easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(easysimd_math_sqrt)
      r_.f64[0] = easysimd_math_sqrt(b_.f64[0]);
      r_.f64[1] = a_.f64[1];
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sqrt_sd(a, b) easysimd_mm_sqrt_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srl_epi16 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_srl_epi16(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    const int cnt = HEDLEY_STATIC_CAST(int, (count.i64[0] > 16 ? 16 : count.i64[0]));
    easysimd__m128i r;
    r.sve_u16 = svlsr_n_u16_z(svptrue_b16(), a.sve_u16, cnt);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    const int cnt = HEDLEY_STATIC_CAST(int, (count_.i64[0] > 16 ? 16 : count_.i64[0]));

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u16 = vshlq_u16(a_.neon_u16, vdupq_n_s16(HEDLEY_STATIC_CAST(int16_t, -cnt)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = a_.u16[i] >> cnt;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_srl_epi16(a, count) (easysimd_mm_srl_epi16(a, (count)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srl_epi32 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_srl_epi32(a, count);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      const uint32_t cnt = HEDLEY_STATIC_CAST(uint32_t, (count.i64[0] > 32 ? 32 : count.i64[0]));
      r_.sve_u32 = svlsr_n_u32_z(svptrue_b32(), a.sve_u32, cnt);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      const int cnt = HEDLEY_STATIC_CAST(int, (count.i64[0] > 32 ? 32 : count.i64[0]));
      r_.neon_u32 = vshlq_u32(a.neon_u32, vdupq_n_s32(HEDLEY_STATIC_CAST(int32_t, -cnt)));
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);
      const int cnt = HEDLEY_STATIC_CAST(int, (count_.i64[0] > 32 ? 32 : count_.i64[0]));

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i] >> cnt;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_srl_epi32(a, count) (easysimd_mm_srl_epi32(a, (count)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srl_epi64 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_srl_epi64(a, count);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      const uint64_t cnt = HEDLEY_STATIC_CAST(uint64_t, (count.i64[0] > 64 ? 64 : count.i64[0]));
      r_.sve_u64 = svlsr_n_u64_z(svptrue_b64(), a.sve_u64, cnt);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      const int cnt = HEDLEY_STATIC_CAST(int, (count.i64[0] > 64 ? 64 : count.i64[0]));
      r_.neon_u64 = vshlq_u64(a.neon_u64, vdupq_n_s64(HEDLEY_STATIC_CAST(int64_t, -cnt)));
    #else
      easysimd__m128i_private
        a_ = easysimd__m128i_to_private(a),
        count_ = easysimd__m128i_to_private(count);
        const int cnt = HEDLEY_STATIC_CAST(int, (count_.i64[0] > 64 ? 64 : count_.i64[0]));
      #if !defined(EASYSIMD_BUG_GCC_94488)
        EASYSIMD_VECTORIZE
      #endif
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = a_.u64[i] >> cnt;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_srl_epi64(a, count) (easysimd_mm_srl_epi64(a, (count)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srai_epi16 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  r.sve_i16 = svasr_n_s16_z(svptrue_b16(), a.sve_i16, imm8 &0xFF);
  return r;
#else
  /* MSVC requires a range of (0, 255). */
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a);

  const int cnt = (imm8 & ~15) ? 15 : imm8;

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    r_.neon_i16 = vshlq_s16(a_.neon_i16, vdupq_n_s16(HEDLEY_STATIC_CAST(int16_t, -cnt)));
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = a_.i16[i] >> cnt;
    }
  #endif

  return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_srai_epi16(a, imm8) _mm_srai_epi16((a), (imm8))
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_srai_epi16(a, imm8) easysimd_mm_srai_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srai_epi32 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  /* MSVC requires a range of (0, 255). */
  const int cnt = (imm8 & ~31) ? 31 : imm8;
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32 = svasr_n_s32_z(pg, a.sve_i32, cnt);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vshlq_s32(a_.neon_i32, vdupq_n_s32(-cnt));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] >> cnt;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_srai_epi32(a, imm8) _mm_srai_epi32((a), (imm8))
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_srai_epi32(a, imm8) easysimd_mm_srai_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srai_epi64 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  /* MSVC requires a range of (0, 255). */
  const int cnt = (imm8 & ~63) ? 63 : imm8;
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64 = svasr_n_s64_z(pg, a.sve_i64, cnt);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] >> cnt;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #define _mm_srai_epi64(a, imm8) easysimd_mm_srai_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sra_epi16 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sra_epi16(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    const int shift = count.u64[0] > 15 ? 15 : HEDLEY_STATIC_CAST(int, count.u64[0]);
    r.sve_i16 = svasr_n_s16_z(svptrue_b16(), a.sve_i16, shift);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    const int cnt = HEDLEY_STATIC_CAST(int, (count_.i64[0] > 15 ? 15 : count_.i64[0]));

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i16 = vshlq_s16(a_.neon_i16, vdupq_n_s16(HEDLEY_STATIC_CAST(int16_t, -cnt)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i16[i] >> cnt;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sra_epi16(a, count) (easysimd_mm_sra_epi16(a, count))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sra_epi32 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(EASYSIMD_BUG_GCC_BAD_MM_SRA_EPI32)
    return _mm_sra_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    const int shift = count.u64[0] > 31 ? 31 : HEDLEY_STATIC_CAST(int, count.u64[0]);
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32 = svasr_n_s32_z(pg, a.sve_i32, shift);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    const int cnt = count_.u64[0] > 31 ? 31 : HEDLEY_STATIC_CAST(int, count_.u64[0]);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vshlq_s32(a_.neon_i32, vdupq_n_s32(HEDLEY_STATIC_CAST(int32_t, -cnt)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] >> cnt;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sra_epi32(a, count) (easysimd_mm_sra_epi32(a, (count)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sra_epi64 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    const int shift = count.u64[0] > 63 ? 63 : HEDLEY_STATIC_CAST(int, count.u64[0]);
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64 = svasr_n_s64_z(pg, a.sve_i64, shift);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);
    const int cnt = count_.u64[0] > 63 ? 63 : HEDLEY_STATIC_CAST(int, count_.u64[0]);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[i] >> cnt;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #define _mm_sra_epi64(a, count) (easysimd_mm_sra_epi64(a, (count)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_slli_epi16 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  if (HEDLEY_UNLIKELY((imm8 > 15))) {
    return easysimd_mm_setzero_si128();
  }

  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svlsl_n_s16_x(svptrue_b16(), a.sve_i16, imm8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i16 = a_.i16 << EASYSIMD_CAST_VECTOR_SHIFT_COUNT(8, imm8 & 0xff);
    #else
      const int s = (imm8 > HEDLEY_STATIC_CAST(int, sizeof(r_.i16[0]) * CHAR_BIT) - 1) ? 0 : imm8;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << s);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_slli_epi16(a, imm8) _mm_slli_epi16(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE) 
#elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
  #define easysimd_mm_slli_epi16(a, imm8) ({ \
    easysimd__m128i r; \
    r.neon_i16 = (imm8 <= 0) ? \
      (a.neon_i16) : \
      (imm8 > 15) ? \
          (vdupq_n_s16(0)) : \
          (vshlq_n_s16(a.neon_i16, imm8)); \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_slli_epi16(a, imm8) easysimd_mm_slli_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_slli_epi32 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  if (HEDLEY_UNLIKELY((imm8 > 31))) {
    return easysimd_mm_setzero_si128();
  }

  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svlsl_n_s32_x(svptrue_b32(), a.sve_i32, imm8);
    return r;
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
      r_.i32 = a_.i32 << imm8;
    #else
      easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] << (imm8 & 0xff);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_slli_epi32(a, imm8) _mm_slli_epi32(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE) 
#elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
    #define easysimd_mm_slli_epi32(a, imm8) ({ \
      easysimd__m128i r; \
      if (HEDLEY_UNLIKELY((imm8 > 31))) { \
        r.neon_i32 = vdupq_n_s32(0); \
      } else {\
        r.neon_i32 = vshlq_n_s32(a.neon_i32, imm8); \
      } \
      r; \
    })
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_slli_epi32(a, imm8) easysimd_mm_slli_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_slli_epi64 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  if (HEDLEY_UNLIKELY((imm8 > 63))) {
    return easysimd_mm_setzero_si128();
  }

  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svlsl_n_s64_x(svptrue_b64(), a.sve_i64, imm8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i64 = a_.i64 << imm8;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] << (imm8 & 0xff);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_slli_epi64(a, imm8) _mm_slli_epi64(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
  #define easysimd_mm_slli_epi64(a, imm8) ({ \
    easysimd__m128i r; \
    r.neon_i64 = (imm8 <= 0) ? \
      (a.neon_i64) : \
      (imm8 > 63) ? \
        (vdupq_n_s64(0)) : \
        (vshlq_n_s64(a.neon_i64, imm8)); \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_slli_epi64(a, imm8) easysimd_mm_slli_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srli_epi16 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  if (HEDLEY_UNLIKELY((imm8 > 15))) {
    return easysimd_mm_setzero_si128();
  }

  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u16 = svlsr_n_u16_z(svptrue_b16(), a.sve_u16, imm8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u16 = a_.u16 >> EASYSIMD_CAST_VECTOR_SHIFT_COUNT(8, imm8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.u16[i] = a_.u16[i] >> (imm8 & 0xff);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_srli_epi16(a, imm8) _mm_srli_epi16(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
  #define easysimd_mm_srli_epi16(a, imm8) ({ \
    easysimd__m128i r; \
    r.neon_u16 = (imm8 <= 0) ? \
      (a.neon_u16) : \
      (imm8 > 15) ? \
        (vdupq_n_u16(0)) : \
        (vshrq_n_u16(a.neon_u16, imm8)); \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_srli_epi16(a, imm8) easysimd_mm_srli_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srli_epi32 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  if (HEDLEY_UNLIKELY((imm8 > 31))) {
    return easysimd_mm_setzero_si128();
  }
  easysimd__m128i_private r_;

  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    r_.sve_u32 = svlsr_n_u32_z(svptrue_b32(), a.sve_u32, imm8);
  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    r_.u32 = a_.u32 >> EASYSIMD_CAST_VECTOR_SHIFT_COUNT(8, imm8 & 0xff);
  #else
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] >> (imm8 & 0xff);
    }
  #endif

  return easysimd__m128i_from_private(r_);
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_srli_epi32(a, imm8) _mm_srli_epi32(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE) 
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm_srli_epi32(a, imm8) ({ \
    easysimd__m128i r; \
    r.neon_u32 = (imm8 <= 0) ? \
      (a.neon_u32) : \
      (imm8 > 31) ? \
        (vdupq_n_u32(0)) : \
        (vshrq_n_u32(a.neon_u32, imm8)); \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_srli_epi32(a, imm8) easysimd_mm_srli_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srli_epi64 (easysimd__m128i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  if (HEDLEY_UNLIKELY((imm8 > 63))) {
    return easysimd_mm_setzero_si128();
  }

  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_u64 = svlsr_n_u64_x(svptrue_b64(), a.sve_u64, imm8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && !defined(EASYSIMD_BUG_GCC_94488)
      r_.u64 = a_.u64 >> EASYSIMD_CAST_VECTOR_SHIFT_COUNT(8, imm8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.u64[i] = a_.u64[i] >> imm8;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  #define easysimd_mm_srli_epi64(a, imm8) _mm_srli_epi64(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
  #define easysimd_mm_srli_epi64(a, imm8) ({\
    easysimd__m128i r; \
    r.neon_u64 = (imm8 <= 0) ? \
      (a.neon_u64) : \
      (imm8 > 63) ? \
          (vdupq_n_u64(0)) : \
          (vshrq_n_u64(a.neon_u64, imm8)); \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_srli_epi64(a, imm8) easysimd_mm_srli_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_store_pd (easysimd_float64 mem_addr[HEDLEY_ARRAY_PARAM(2)], easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_store_pd(mem_addr, a);
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    svst1_f64(svptrue_b64(), &(mem_addr[0]), a.sve_f64);
    return;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_f64(mem_addr, easysimd__m128d_to_private(a).neon_f64);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_s64(HEDLEY_REINTERPRET_CAST(int64_t*, mem_addr), easysimd__m128d_to_private(a).neon_i64);
  #else
    easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128d), &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_store_pd(mem_addr, a) easysimd_mm_store_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_store1_pd (easysimd_float64 mem_addr[HEDLEY_ARRAY_PARAM(2)], easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_store1_pd(mem_addr, a);
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    svst1_f64(svptrue_b64(), &(mem_addr[0]), svdup_n_f64(a.f64[0]));
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      vst1q_f64(mem_addr, vdupq_laneq_f64(a_.neon_f64, 0));
    #else
      mem_addr[0] = a_.f64[0];
      mem_addr[1] = a_.f64[0];
    #endif
  #endif
}
#define easysimd_mm_store_pd1(mem_addr, a) easysimd_mm_store1_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_store1_pd(mem_addr, a) easysimd_mm_store1_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
  #define _mm_store_pd1(mem_addr, a) easysimd_mm_store_pd1(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_store_sd (easysimd_float64* mem_addr, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_store_sd(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    *mem_addr = a.f64[0];
    return;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_lane_f64(mem_addr, a.neon_f64, 0);
    return;
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      const int64_t v = vgetq_lane_s64(a_.neon_i64, 0);
      easysimd_memcpy(HEDLEY_REINTERPRET_CAST(int64_t*, mem_addr), &v, sizeof(v));
    #else
      easysimd_float64 v = a_.f64[0];
      easysimd_memcpy(mem_addr, &v, sizeof(easysimd_float64));
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_store_sd(mem_addr, a) easysimd_mm_store_sd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_store_si128 (easysimd__m128i* mem_addr, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_store_si128(HEDLEY_STATIC_CAST(__m128i*, mem_addr), a);
  #else

    #if defined (EASYSIMD_ARM_SVE_NATIVE)
      svst1_s32(svptrue_b32(), (int32_t *)mem_addr, a.sve_i32);
      return;
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      vst1q_s32(HEDLEY_REINTERPRET_CAST(int32_t*, mem_addr), a.neon_i32);
    #else
      easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
      easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128i), &a_, sizeof(a_));
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_store_si128(mem_addr, a) easysimd_mm_store_si128(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
  easysimd_mm_storeh_pd (easysimd_float64* mem_addr, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_storeh_pd(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    *mem_addr = a.sve_f64[1];
    return;
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      *mem_addr = vgetq_lane_f64(a_.neon_f64, 1);
    #else
      *mem_addr = a_.f64[1];
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_storeh_pd(mem_addr, a) easysimd_mm_storeh_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storel_epi64 (easysimd__m128i* mem_addr, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_storel_epi64(HEDLEY_STATIC_CAST(__m128i*, mem_addr), a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    (*mem_addr).i64[0] = a.i64[0];
    return;
  #else
    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int64_t tmp = vgetq_lane_s64(a.neon_i64, 0);
      easysimd_memcpy(mem_addr, &tmp, sizeof(tmp));
    #else
      easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
      int64_t tmp = a_.i64[0];
      easysimd_memcpy(mem_addr, &tmp, sizeof(tmp));
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_storel_epi64(mem_addr, a) easysimd_mm_storel_epi64(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storel_pd (easysimd_float64* mem_addr, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_storel_pd(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    *mem_addr = a.f64[0];
    return;
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    easysimd_float64 tmp;
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      tmp = vgetq_lane_f64(a_.neon_f64, 0);
    #else
      tmp = a_.f64[0];
    #endif
    easysimd_memcpy(mem_addr, &tmp, sizeof(tmp));
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_storel_pd(mem_addr, a) easysimd_mm_storel_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storer_pd (easysimd_float64 mem_addr[2], easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_storer_pd(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f64(svptrue_b64(), &(mem_addr[0]), svrev_f64(a.sve_f64));
  #else
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      vst1q_s64(HEDLEY_REINTERPRET_CAST(int64_t*, mem_addr), vextq_s64(a_.neon_i64, a_.neon_i64, 1));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      a_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.f64, a_.f64, 1, 0);
      easysimd_mm_store_pd(mem_addr, easysimd__m128d_from_private(a_));
    #else
      mem_addr[0] = a_.f64[1];
      mem_addr[1] = a_.f64[0];
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_storer_pd(mem_addr, a) easysimd_mm_storer_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storeu_pd (easysimd_float64* mem_addr, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_storeu_pd(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f64(svptrue_b64(), mem_addr, a.sve_f64);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_f64(mem_addr, easysimd__m128d_to_private(a).neon_f64);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_storeu_pd(mem_addr, a) easysimd_mm_storeu_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storeu_si128 (void* mem_addr, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_storeu_si128(HEDLEY_STATIC_CAST(__m128i*, mem_addr), a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_s32(svptrue_b32(), (int32_t *)mem_addr, a.sve_i32);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_s32((int32_t *)mem_addr, a.neon_i32);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_storeu_si128(mem_addr, a) easysimd_mm_storeu_si128(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storeu_si16 (void* mem_addr, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && ( \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0) || \
      HEDLEY_GCC_VERSION_CHECK(11,0,0) || \
      HEDLEY_INTEL_VERSION_CHECK(20,21,1))
    _mm_storeu_si16(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_memcpy(mem_addr, &(a.i16[0]), sizeof(a.i16[0]));
  #else
    int16_t val = easysimd_x_mm_cvtsi128_si16(a);
    easysimd_memcpy(mem_addr, &val, sizeof(val));
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_storeu_si16(mem_addr, a) easysimd_mm_storeu_si16(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storeu_si32 (void* mem_addr, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && ( \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0) || \
      HEDLEY_GCC_VERSION_CHECK(11,0,0) || \
      HEDLEY_INTEL_VERSION_CHECK(20,21,1))
    _mm_storeu_si32(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_memcpy(mem_addr, &(a.i32[0]), sizeof(a.i32[0]));
  #else
    int32_t val = easysimd_mm_cvtsi128_si32(a);
    easysimd_memcpy(mem_addr, &val, sizeof(val));
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_storeu_si32(mem_addr, a) easysimd_mm_storeu_si32(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storeu_si64 (void* mem_addr, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && ( \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0) || \
      HEDLEY_GCC_VERSION_CHECK(11,0,0) || \
      HEDLEY_INTEL_VERSION_CHECK(20,21,1))
    _mm_storeu_si64(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_memcpy(mem_addr, &(a.i64[0]), sizeof(a.i64[0]));
  #else
    int64_t val = easysimd_mm_cvtsi128_si64(a);
    easysimd_memcpy(mem_addr, &val, sizeof(val));
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_storeu_si64(mem_addr, a) easysimd_mm_storeu_si64(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_stream_pd (easysimd_float64 mem_addr[HEDLEY_ARRAY_PARAM(2)], easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_stream_pd(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f64(svptrue_b64(), mem_addr, a.sve_f64);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_stream_pd(mem_addr, a) easysimd_mm_stream_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_stream_si128 (easysimd__m128i* mem_addr, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
    _mm_stream_si128(HEDLEY_STATIC_CAST(__m128i*, mem_addr), a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_s64(svptrue_b64(), (int64_t*)mem_addr, a.sve_i64);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_s64((int64_t*)mem_addr, a.neon_i64);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_stream_si128(mem_addr, a) easysimd_mm_stream_si128(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_stream_si32 (int32_t* mem_addr, int32_t a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_stream_si32(mem_addr, a);
  #else
    *mem_addr = a;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_stream_si32(mem_addr, a) easysimd_mm_stream_si32(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_stream_si64 (int64_t* mem_addr, int64_t a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_ARCH_AMD64) && !defined(HEDLEY_MSVC_VERSION)
    _mm_stream_si64(EASYSIMD_CHECKED_REINTERPRET_CAST(long long int*, int64_t*, mem_addr), a);
  #else
    *mem_addr = a;
  #endif
}
#define easysimd_mm_stream_si64x(mem_addr, a) easysimd_mm_stream_si64(mem_addr, a)
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
  #define _mm_stream_si64(mem_addr, a) easysimd_mm_stream_si64(EASYSIMD_CHECKED_REINTERPRET_CAST(int64_t*, __int64*, mem_addr), a)
  #define _mm_stream_si64x(mem_addr, a) easysimd_mm_stream_si64(EASYSIMD_CHECKED_REINTERPRET_CAST(int64_t*, __int64*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sub_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sub_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svsub_s8_x(svptrue_b8(), a.sve_i8, b.sve_i8);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_i8 = vsubq_s8(a.neon_i8, b.neon_i8);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = a_.i8 - b_.i8;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[i] - b_.i8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sub_epi8(a, b) easysimd_mm_sub_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sub_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sub_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svsub_s16_x(svptrue_b16(), a.sve_i16, b.sve_i16);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_i16 = vsubq_s16(a.neon_i16, b.neon_i16);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = a_.i16 - b_.i16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i16[i] - b_.i16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sub_epi16(a, b) easysimd_mm_sub_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sub_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sub_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsub_s32_x(svptrue_b32(), a.sve_i32, b.sve_i32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_i32 = vsubq_s32(a.neon_i32, b.neon_i32);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 - b_.i32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] - b_.i32[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sub_epi32(a, b) easysimd_mm_sub_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sub_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sub_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svsub_s64_x(svptrue_b64(), a.sve_i64, b.sve_i64);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i res;
    res.neon_i64 = vsubq_s64(a.neon_i64, b.neon_i64);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = a_.i64 - b_.i64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] - b_.i64[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sub_epi64(a, b) easysimd_mm_sub_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_sub_epu32 (easysimd__m128i a, easysimd__m128i b) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    r_.u32 = a_.u32 - b_.u32;
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    r_.neon_u32 = vsubq_u32(a_.neon_u32, b_.neon_u32);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] - b_.u32[i];
    }
  #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_sub_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sub_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsub_f64_z(svptrue_b64(), a.sve_f64, b.sve_f64);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vsubq_f64(a.neon_f64, b.neon_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 - b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[i] - b_.f64[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sub_pd(a, b) easysimd_mm_sub_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_sub_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_sub_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsub_f64_z(svptrue_b64(), a.sve_f64, svdupq_n_f64(b.f64[0], 0.0));
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_sd(a, easysimd_mm_sub_pd(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_sd(a, easysimd_mm_sub_pd(easysimd_x_mm_broadcastlow_pd(a), easysimd_x_mm_broadcastlow_pd(b)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    r_.f64[0] = a_.f64[0] - b_.f64[0];
    r_.f64[1] = a_.f64[1];

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sub_sd(a, b) easysimd_mm_sub_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_sub_si64 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_sub_si64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m64 r;
    r.i64[0] = a.i64[0] - b.i64[0];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m64 r;
    r.neon_i64 = vsub_s64(a.neon_i64, b.neon_i64);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = a_.i64 - b_.i64;
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vsub_s64(a_.neon_i64, b_.neon_i64);
    #else
      r_.i64[0] = a_.i64[0] - b_.i64[0];
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_sub_si64(a, b) easysimd_mm_sub_si64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_subs_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_subs_epi8(a, b);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i8 = svqsub_s8(a.sve_i8, b.sve_i8);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i8 = vqsubq_s8(a.neon_i8, b.neon_i8);
    #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = easysimd_math_subs_i8(a_.i8[i], b_.i8[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_subs_epi8(a, b) easysimd_mm_subs_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_subs_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_subs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svqsub_s16(a.sve_i16, b.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i16 = vqsubq_s16(a_.neon_i16, b_.neon_i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = easysimd_math_subs_i16(a_.i16[i], b_.i16[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_subs_epi16(a, b) easysimd_mm_subs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_subs_epu8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_subs_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u8 = svqsub_u8_x(svptrue_b8(), a.sve_u8, b.sve_u8);
    return a;
  #elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
    easysimd__m128i r;
    r.neon_u8 = vqsubq_u8(a.neon_u8, b.neon_u8);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = easysimd_math_subs_u8(a_.u8[i], b_.u8[i]);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_subs_epu8(a, b) easysimd_mm_subs_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_subs_epu16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_subs_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u16 = svqsub_u16_x(svptrue_b16(), a.sve_u16, b.sve_u16);
    return a;
  #elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
    easysimd__m128i r;
    r.neon_u16 = vqsubq_u16(a.neon_u16, b.neon_u16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = easysimd_math_subs_u16(a_.u16[i], b_.u16[i]);
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_subs_epu16(a, b) easysimd_mm_subs_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomieq_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_ucomieq_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return easysimd_sse2_cmp_f64_e(a.f64[0], b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint64x2_t a_not_nan = vceqq_f64(a_.neon_f64, a_.neon_f64);
      uint64x2_t b_not_nan = vceqq_f64(b_.neon_f64, b_.neon_f64);
      uint64x2_t a_or_b_nan = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vandq_u64(a_not_nan, b_not_nan))));
      uint64x2_t a_eq_b = vceqq_f64(a_.neon_f64, b_.neon_f64);
      r = !!(vgetq_lane_u64(vorrq_u64(a_or_b_nan, a_eq_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r =  easysimd_sse2_cmp_f64_e(a_.f64[0], b_.f64[0]);
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = easysimd_sse2_cmp_f64_e(a_.f64[0], b_.f64[0]);
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_ucomieq_sd(a, b) easysimd_mm_ucomieq_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomige_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_ucomige_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return easysimd_sse2_cmp_f64_ge(a.f64[0], b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint64x2_t a_not_nan = vceqq_f64(a_.neon_f64, a_.neon_f64);
      uint64x2_t b_not_nan = vceqq_f64(b_.neon_f64, b_.neon_f64);
      uint64x2_t a_and_b_not_nan = vandq_u64(a_not_nan, b_not_nan);
      uint64x2_t a_ge_b = vcgeq_f64(a_.neon_f64, b_.neon_f64);
      r = !!(vgetq_lane_u64(vandq_u64(a_and_b_not_nan, a_ge_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = easysimd_sse2_cmp_f64_ge(a_.f64[0], b_.f64[0]);
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = easysimd_sse2_cmp_f64_ge(a_.f64[0], b_.f64[0]);
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_ucomige_sd(a, b) easysimd_mm_ucomige_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomigt_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_ucomigt_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (a.f64[0] > b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint64x2_t a_not_nan = vceqq_f64(a_.neon_f64, a_.neon_f64);
      uint64x2_t b_not_nan = vceqq_f64(b_.neon_f64, b_.neon_f64);
      uint64x2_t a_and_b_not_nan = vandq_u64(a_not_nan, b_not_nan);
      uint64x2_t a_gt_b = vcgtq_f64(a_.neon_f64, b_.neon_f64);
      r = !!(vgetq_lane_u64(vandq_u64(a_and_b_not_nan, a_gt_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = a_.f64[0] > b_.f64[0];
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = a_.f64[0] > b_.f64[0];
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_ucomigt_sd(a, b) easysimd_mm_ucomigt_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomile_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_ucomile_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return easysimd_sse2_cmp_f64_le(a.f64[0], b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint64x2_t a_not_nan = vceqq_f64(a_.neon_f64, a_.neon_f64);
      uint64x2_t b_not_nan = vceqq_f64(b_.neon_f64, b_.neon_f64);
      uint64x2_t a_or_b_nan = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vandq_u64(a_not_nan, b_not_nan))));
      uint64x2_t a_le_b = vcleq_f64(a_.neon_f64, b_.neon_f64);
      r = !!(vgetq_lane_u64(vorrq_u64(a_or_b_nan, a_le_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = easysimd_sse2_cmp_f64_le(a_.f64[0], b_.f64[0]);
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = easysimd_sse2_cmp_f64_le(a_.f64[0], b_.f64[0]);
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_ucomile_sd(a, b) easysimd_mm_ucomile_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomilt_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_ucomilt_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (a.f64[0] < b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint64x2_t a_not_nan = vceqq_f64(a_.neon_f64, a_.neon_f64);
      uint64x2_t b_not_nan = vceqq_f64(b_.neon_f64, b_.neon_f64);
      uint64x2_t a_or_b_nan = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vandq_u64(a_not_nan, b_not_nan))));
      uint64x2_t a_lt_b = vcltq_f64(a_.neon_f64, b_.neon_f64);
      r = !!(vgetq_lane_u64(vorrq_u64(a_or_b_nan, a_lt_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = a_.f64[0] < b_.f64[0];
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = a_.f64[0] < b_.f64[0];
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_ucomilt_sd(a, b) easysimd_mm_ucomilt_sd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomineq_sd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_ucomineq_sd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return !easysimd_sse2_cmp_f64_e(a.f64[0], b.f64[0]);
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      uint64x2_t a_not_nan = vceqq_f64(a_.neon_f64, a_.neon_f64);
      uint64x2_t b_not_nan = vceqq_f64(b_.neon_f64, b_.neon_f64);
      uint64x2_t a_and_b_not_nan = vandq_u64(a_not_nan, b_not_nan);
      uint64x2_t a_neq_b = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(a_.neon_f64, b_.neon_f64))));
      r = !!(vgetq_lane_u64(vandq_u64(a_and_b_not_nan, a_neq_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = !easysimd_sse2_cmp_f64_e(a_.f64[0], b_.f64[0]);
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = !easysimd_sse2_cmp_f64_e(a_.f64[0], b_.f64[0]);
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_ucomineq_sd(a, b) easysimd_mm_ucomineq_sd(a, b)
#endif

#if defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
  HEDLEY_DIAGNOSTIC_PUSH
  EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_
#endif

#if defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
  HEDLEY_DIAGNOSTIC_POP
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_lfence (void) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_lfence();
  #else
    easysimd_mm_sfence();
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_lfence() easysimd_mm_lfence()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_mfence (void) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    _mm_mfence();
  #else
    easysimd_mm_sfence();
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_mfence() easysimd_mm_mfence()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_unpackhi_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpackhi_epi8(a, b);
  #else
      easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i8 = svzip2_s8(a.sve_i8, b.sve_i8);
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i8 = vzip2q_s8(a.neon_i8, b.neon_i8);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int8x8_t a1 = vreinterpret_s8_s16(vget_high_s16(a.neon_i16));
      int8x8_t b1 = vreinterpret_s8_s16(vget_high_s16(b.neon_i16));
      int8x8x2_t result = vzip_s8(a1, b1);
      r_.neon_i8 = vcombine_s8(result.val[0], result.val[1]);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.i8[0])) / 2) ; i++) {
        r_.i8[(i * 2)]     = a_.i8[i + ((sizeof(r_) / sizeof(r_.i8[0])) / 2)];
        r_.i8[(i * 2) + 1] = b_.i8[i + ((sizeof(r_) / sizeof(r_.i8[0])) / 2)];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpackhi_epi8(a, b) easysimd_mm_unpackhi_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_unpackhi_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpackhi_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svzip2_s16(a.sve_i16, b.sve_i16);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i16 = vzip2q_s16(a_.neon_i16, b_.neon_i16);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int16x4_t a1 = vget_high_s16(a_.neon_i16);
      int16x4_t b1 = vget_high_s16(b_.neon_i16);
      int16x4x2_t result = vzip_s16(a1, b1);
      r_.neon_i16 = vcombine_s16(result.val[0], result.val[1]);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.i16 = EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.i16, b_.i16, 4, 12, 5, 13, 6, 14, 7, 15);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.i16[0])) / 2) ; i++) {
        r_.i16[(i * 2)]     = a_.i16[i + ((sizeof(r_) / sizeof(r_.i16[0])) / 2)];
        r_.i16[(i * 2) + 1] = b_.i16[i + ((sizeof(r_) / sizeof(r_.i16[0])) / 2)];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpackhi_epi16(a, b) easysimd_mm_unpackhi_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_unpackhi_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpackhi_epi32(a, b);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i32 = svzip2_s32(a.sve_i32, b.sve_i32);
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i32 = vzip2q_s32(a.neon_i32, b.neon_i32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int32x2_t a1 = vget_high_s32(a.neon_i32);
      int32x2_t b1 = vget_high_s32(b.neon_i32);
      int32x2x2_t result = vzip_s32(a1, b1);
      r_.neon_i32 = vcombine_s32(result.val[0], result.val[1]);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.i32[0])) / 2) ; i++) {
        r_.i32[(i * 2)]     = a_.i32[i + ((sizeof(r_) / sizeof(r_.i32[0])) / 2)];
        r_.i32[(i * 2) + 1] = b_.i32[i + ((sizeof(r_) / sizeof(r_.i32[0])) / 2)];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpackhi_epi32(a, b) easysimd_mm_unpackhi_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_unpackhi_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpackhi_epi64(a, b);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i64 = svzip2_s64(a.sve_i64, b.sve_i64);
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i64 = vzip2q_s64(a.neon_i64, b.neon_i64);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int64x1_t a_h = vget_high_s64(a.neon_i64);
      int64x1_t b_h = vget_high_s64(b.neon_i64);
      r_.neon_i64 = vcombine_s64(a_h, b_h);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.i64[0])) / 2) ; i++) {
        r_.i64[(i * 2)]     = a_.i64[i + ((sizeof(r_) / sizeof(r_.i64[0])) / 2)];
        r_.i64[(i * 2) + 1] = b_.i64[i + ((sizeof(r_) / sizeof(r_.i64[0])) / 2)];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpackhi_epi64(a, b) easysimd_mm_unpackhi_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_unpackhi_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpackhi_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svzip2_f64(a.sve_f64, b.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vzip2q_f64(a_.neon_f64, b_.neon_f64);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.f64, b_.f64, 1, 3);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.f64[0])) / 2) ; i++) {
        r_.f64[(i * 2)]     = a_.f64[i + ((sizeof(r_) / sizeof(r_.f64[0])) / 2)];
        r_.f64[(i * 2) + 1] = b_.f64[i + ((sizeof(r_) / sizeof(r_.f64[0])) / 2)];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpackhi_pd(a, b) easysimd_mm_unpackhi_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_unpacklo_epi8 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpacklo_epi8(a, b);
  #else
    easysimd__m128i_private r_;

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i8 = svzip1_s8(a.sve_i8, b.sve_i8);
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i8 = vzip1q_s8(a.neon_i8, b.neon_i8);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int8x8_t a1 = vreinterpret_s8_s16(vget_low_s16(a.neon_i16));
      int8x8_t b1 = vreinterpret_s8_s16(vget_low_s16(b.neon_i16));
      int8x8x2_t result = vzip_s8(a1, b1);
      r_.neon_i8 = vcombine_s8(result.val[0], result.val[1]);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.i8[0])) / 2) ; i++) {
        r_.i8[(i * 2)]     = a_.i8[i];
        r_.i8[(i * 2) + 1] = b_.i8[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpacklo_epi8(a, b) easysimd_mm_unpacklo_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_unpacklo_epi16 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpacklo_epi16(a, b);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i16 = svzip1_s16(a.sve_i16, b.sve_i16);
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i16 = vzip1q_s16(a.neon_i16, b.neon_i16);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.i16[0])) / 2) ; i++) {
        r_.i16[(i * 2)]     = a_.i16[i];
        r_.i16[(i * 2) + 1] = b_.i16[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpacklo_epi16(a, b) easysimd_mm_unpacklo_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_unpacklo_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpacklo_epi32(a, b);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i32 = svzip1_s32(a.sve_i32, b.sve_i32);
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i32 = vzip1q_s32(a.neon_i32, b.neon_i32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int32x2_t a1 = vget_low_s32(a.neon_i32);
      int32x2_t b1 = vget_low_s32(b.neon_i32);
      int32x2x2_t result = vzip_s32(a1, b1);
      r_.neon_i32 = vcombine_s32(result.val[0], result.val[1]);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.i32[0])) / 2) ; i++) {
        r_.i32[(i * 2)]     = a_.i32[i];
        r_.i32[(i * 2) + 1] = b_.i32[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpacklo_epi32(a, b) easysimd_mm_unpacklo_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_unpacklo_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpacklo_epi64(a, b);
  #else
    easysimd__m128i_private r_;
    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      r_.sve_i64 = svzip1_s64(a.sve_i64, b.sve_i64);
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_i64 = vzip1q_s64(a.neon_i64, b.neon_i64);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int64x1_t a_l = vget_low_s64(a.neon_i64);
      int64x1_t b_l = vget_low_s64(b.neon_i64);
      r_.neon_i64 = vcombine_s64(a_l, b_l);
    #else
      easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.i64[0])) / 2) ; i++) {
        r_.i64[(i * 2)]     = a_.i64[i];
        r_.i64[(i * 2) + 1] = b_.i64[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpacklo_epi64(a, b) easysimd_mm_unpacklo_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_unpacklo_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_unpacklo_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svzip1_f64(a.sve_f64, b.sve_f64);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vzip1q_f64(a_.neon_f64, b_.neon_f64);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.f64, b_.f64, 0, 2);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_) / sizeof(r_.f64[0])) / 2) ; i++) {
        r_.f64[(i * 2)]     = a_.f64[i];
        r_.f64[(i * 2) + 1] = b_.f64[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_unpacklo_pd(a, b) easysimd_mm_unpacklo_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_negate_pd(easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return easysimd_mm_xor_pd(a, _mm_set1_pd(EASYSIMD_FLOAT64_C(-0.0)));
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vnegq_f64(a_.neon_f64);
    #elif defined(EASYSIMD_VECTOR_NEGATE)
      r_.f64 = -a_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = -a_.f64[i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_xor_si128 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    return _mm_xor_si128(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = sveor_s32_x(svptrue_b32(), a.sve_i32, b.sve_i32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i32 = veorq_s32(a.neon_i32, b.neon_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f ^ b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = a_.i32f[i] ^ b_.i32f[i];
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _mm_xor_si128(a, b) easysimd_mm_xor_si128(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_not_si128 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_ternarylogic_epi32(a, a, a, 0x55);
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vmvnq_s32(a_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = ~a_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = ~(a_.i32f[i]);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}

#define EASYSIMD_MM_SHUFFLE2(x, y) (((x) << 1) | (y))
#if defined(EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES)
  #define _MM_SHUFFLE2(x, y) EASYSIMD_MM_SHUFFLE2(x, y)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_pext_u32 (uint32_t a, uint32_t mask){
  #if defined(EASYSIMD_X86_AVX512VBMI2_NATIVE)
    return (uint32_t)(_pext_u32(a, mask));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE) && defined (__ARM_FEATURE_SVE2_BITPERM)
    easysimd__m128i r;
    r.sve_u32 = svbext_u32(svdup_n_u32(a), svdup_n_u32(mask));
    return r.u32[0];
  #else
    uint32_t r = 0;
    for (uint32_t bp = 1; mask != 0; bp += bp) {
      if (a & mask & -mask) {
        r |= bp;
      }
      mask &= (mask - 1);
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI2_ENABLE_NATIVE_ALIASES)
  #define _pext_u32(a, mask) easysimd_pext_u32(a, mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_pext_u64 (uint64_t a, uint64_t mask){
  #if defined(EASYSIMD_X86_AVX512VBMI2_NATIVE)
    return (uint64_t)(_pext_u64(a, mask));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE) && defined (__ARM_FEATURE_SVE2_BITPERM)
    easysimd__m128i r;
    r.sve_u64 = svbext_u64(svdup_n_u64(a), svdup_n_u64(mask));
    return r.u64[0];
  #else
    uint64_t r = 0;
    for (uint64_t bp = 1; mask != 0; bp += bp) {
      if (a & mask & -mask) {
        r |= bp;
      }
      mask &= (mask - 1);
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI2_ENABLE_NATIVE_ALIASES)
  #define _pext_u64(a, mask) easysimd_pext_u64(a, mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_pdep_u64 (uint64_t a, uint64_t mask){
  #if defined(EASYSIMD_X86_AVX512VBMI2_NATIVE)
    return (uint64_t)(_pdep_u64(a, mask));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE) && defined (__ARM_FEATURE_SVE2_BITPERM)
    easysimd__m128i r;
    r.sve_u64 = svbdep_u64(svdup_n_u64(a), svdup_n_u64(mask));
    return r.u64[0];
  #else
    uint64_t r = 0;
    uint64_t i, j = 0;
    for (i = 0; i < 64; i++) {
      if ((mask >> i) & 1) {
        r |= ((a >> j) & 1) << i;
        j++;
      }
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX512VBMI2_ENABLE_NATIVE_ALIASES)
  #define _pdep_u64(a, mask) easysimd_pdep_u64(a, mask)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_SSE2_H) */
