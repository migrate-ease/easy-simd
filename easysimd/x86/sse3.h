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

#if !defined(EASYSIMD_X86_SSE3_H)
#define EASYSIMD_X86_SSE3_H

#include "sse2.h"
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_deinterleaveeven_epi16 (easysimd__m128i a, easysimd__m128i b) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_i16 = vuzp1q_s16(a_.neon_i16, b_.neon_i16);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int16x8x2_t t = vuzpq_s16(a_.neon_i16, b_.neon_i16);
    r_.neon_i16 = t.val[0];
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.i16 = EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.i16, b_.i16, 0, 2, 4, 6, 8, 10, 12, 14);
  #else
    const size_t halfway_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 2;
    for(size_t i = 0 ; i < halfway_point ; i++) {
      r_.i16[i] = a_.i16[2 * i];
      r_.i16[i + halfway_point] = b_.i16[2 * i];
    }
  #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_deinterleaveodd_epi16 (easysimd__m128i a, easysimd__m128i b) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_i16 = vuzp2q_s16(a_.neon_i16, b_.neon_i16);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int16x8x2_t t = vuzpq_s16(a_.neon_i16, b_.neon_i16);
    r_.neon_i16 = t.val[1];
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.i16 = EASYSIMD_SHUFFLE_VECTOR_(16, 16, a_.i16, b_.i16, 1, 3, 5, 7, 9, 11, 13, 15);
  #else
    const size_t halfway_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 2;
    for(size_t i = 0 ; i < halfway_point ; i++) {
      r_.i16[i] = a_.i16[2 * i + 1];
      r_.i16[i + halfway_point] = b_.i16[2 * i + 1];
    }
  #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_deinterleaveeven_epi32 (easysimd__m128i a, easysimd__m128i b) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_i32 = vuzp1q_s32(a_.neon_i32, b_.neon_i32);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x4x2_t t = vuzpq_s32(a_.neon_i32, b_.neon_i32);
    r_.neon_i32 = t.val[0];
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.i32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.i32, b_.i32, 0, 2, 4, 6);
  #else
    const size_t halfway_point = (sizeof(r_.i32) / sizeof(r_.i32[0])) / 2;
    for(size_t i = 0 ; i < halfway_point ; i++) {
      r_.i32[i] = a_.i32[2 * i];
      r_.i32[i + halfway_point] = b_.i32[2 * i];
    }
  #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_x_mm_deinterleaveodd_epi32 (easysimd__m128i a, easysimd__m128i b) {
  easysimd__m128i_private
    r_,
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_i32 = vuzp2q_s32(a_.neon_i32, b_.neon_i32);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x4x2_t t = vuzpq_s32(a_.neon_i32, b_.neon_i32);
    r_.neon_i32 = t.val[1];
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.i32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.i32, b_.i32, 1, 3, 5, 7);
  #else
    const size_t halfway_point = (sizeof(r_.i32) / sizeof(r_.i32[0])) / 2;
    for(size_t i = 0 ; i < halfway_point ; i++) {
      r_.i32[i] = a_.i32[2 * i + 1];
      r_.i32[i + halfway_point] = b_.i32[2 * i + 1];
    }
  #endif

  return easysimd__m128i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_deinterleaveeven_ps (easysimd__m128 a, easysimd__m128 b) {
  easysimd__m128_private
    r_,
    a_ = easysimd__m128_to_private(a),
    b_ = easysimd__m128_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_f32 = vuzp1q_f32(a_.neon_f32, b_.neon_f32);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    float32x4x2_t t = vuzpq_f32(a_.neon_f32, b_.neon_f32);
    r_.neon_f32 = t.val[0];
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, b_.f32, 0, 2, 4, 6);
  #else
    const size_t halfway_point = (sizeof(r_.f32) / sizeof(r_.f32[0])) / 2;
    for(size_t i = 0 ; i < halfway_point ; i++) {
      r_.f32[i] = a_.f32[2 * i];
      r_.f32[i + halfway_point] = b_.f32[2 * i];
    }
  #endif

  return easysimd__m128_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_deinterleaveodd_ps (easysimd__m128 a, easysimd__m128 b) {
  easysimd__m128_private
    r_,
    a_ = easysimd__m128_to_private(a),
    b_ = easysimd__m128_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_f32 = vuzp2q_f32(a_.neon_f32, b_.neon_f32);
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    float32x4x2_t t = vuzpq_f32(a_.neon_f32, b_.neon_f32);
    r_.neon_f32 = t.val[1];
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, b_.f32, 1, 3, 5, 7);
  #else
    const size_t halfway_point = (sizeof(r_.f32) / sizeof(r_.f32[0])) / 2;
    for(size_t i = 0 ; i < halfway_point ; i++) {
      r_.f32[i] = a_.f32[2 * i + 1];
      r_.f32[i + halfway_point] = b_.f32[2 * i + 1];
    }
  #endif

  return easysimd__m128_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_deinterleaveeven_pd (easysimd__m128d a, easysimd__m128d b) {
  easysimd__m128d_private
    r_,
    a_ = easysimd__m128d_to_private(a),
    b_ = easysimd__m128d_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_f64 = vuzp1q_f64(a_.neon_f64, b_.neon_f64);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.f64, b_.f64, 0, 2);
  #else
    const size_t halfway_point = (sizeof(r_.f64) / sizeof(r_.f64[0])) / 2;
    for(size_t i = 0 ; i < halfway_point ; i++) {
      r_.f64[i] = a_.f64[2 * i];
      r_.f64[i + halfway_point] = b_.f64[2 * i];
    }
  #endif

  return easysimd__m128d_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_x_mm_deinterleaveodd_pd (easysimd__m128d a, easysimd__m128d b) {
  easysimd__m128d_private
    r_,
    a_ = easysimd__m128d_to_private(a),
    b_ = easysimd__m128d_to_private(b);

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    r_.neon_f64 = vuzp2q_f64(a_.neon_f64, b_.neon_f64);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.f64, b_.f64, 1, 3);
  #else
    const size_t halfway_point = (sizeof(r_.f64) / sizeof(r_.f64[0])) / 2;
    for(size_t i = 0 ; i < halfway_point ; i++) {
      r_.f64[i] = a_.f64[2 * i + 1];
      r_.f64[i + halfway_point] = b_.f64[2 * i + 1];
    }
  #endif

  return easysimd__m128d_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_addsub_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_addsub_pd(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    float64x2_t rs = vsubq_f64(a.neon_f64, b.neon_f64);
    float64x2_t ra = vaddq_f64(a.neon_f64, b.neon_f64);
    r.neon_f64 = vcombine_f64(vget_low_f64(rs), vget_high_f64(ra));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svadd_f64_z(svptrue_b64(), svsub_f64_z(svdupq_n_b64(1, 0), a.sve_f64, b.sve_f64),
                            svadd_f64_z(svdupq_n_b64(0, 1), a.sve_f64, b.sve_f64));
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

    #if (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.f64 - b_.f64, a_.f64 + b_.f64, 0, 3);
    #else
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
        r_.f64[  i  ] = a_.f64[  i  ] - b_.f64[  i  ];
        r_.f64[1 + i] = a_.f64[1 + i] + b_.f64[1 + i];
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_addsub_pd(a, b) easysimd_mm_addsub_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_addsub_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_addsub_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    float32x4_t rs = vsubq_f32(a.neon_f32, b.neon_f32);
    float32x4_t ra = vaddq_f32(a.neon_f32, b.neon_f32);
    r.neon_f32 = vtrn2q_f32(vreinterpretq_f32_s32(vrev64q_s32(vreinterpretq_s32_f32(rs))), ra);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svadd_f32_z(svptrue_b32(), svsub_f32_z(svdupq_n_b32(1, 0, 1, 0), a.sve_f32, b.sve_f32),
                            svadd_f32_z(svdupq_n_b32(0, 1, 0, 1), a.sve_f32, b.sve_f32));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32 - b_.f32, a_.f32 + b_.f32, 0, 5, 2, 7);
    #else
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 2) {
        r_.f32[  i  ] = a_.f32[  i  ] - b_.f32[  i  ];
        r_.f32[1 + i] = a_.f32[1 + i] + b_.f32[1 + i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_addsub_ps(a, b) easysimd_mm_addsub_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_hadd_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_hadd_pd(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vpaddq_f64(a.neon_f64, b.neon_f64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svadd_f64_z(svptrue_b64(), svuzp1_f64(a.sve_f64, b.sve_f64), svuzp2_f64(a.sve_f64, b.sve_f64));
    return r;
  #else
    return easysimd_mm_add_pd(easysimd_x_mm_deinterleaveeven_pd(a, b), easysimd_x_mm_deinterleaveodd_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hadd_pd(a, b) easysimd_mm_hadd_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_hadd_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_hadd_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vpaddq_f32(a.neon_f32, b.neon_f32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svadd_f32_z(svptrue_b32(), svuzp1_f32(a.sve_f32, b.sve_f32), svuzp2_f32(a.sve_f32, b.sve_f32));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    float32x4x2_t t = vuzpq_f32(easysimd__m128_to_neon_f32(a), easysimd__m128_to_neon_f32(b));
    return easysimd__m128_from_neon_f32(vaddq_f32(t.val[0], t.val[1]));
  #else
    return easysimd_mm_add_ps(easysimd_x_mm_deinterleaveeven_ps(a, b), easysimd_x_mm_deinterleaveodd_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hadd_ps(a, b) easysimd_mm_hadd_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_hsub_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_hsub_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svsub_f64_z(svptrue_b64(), svuzp1_f64(a.sve_f64, b.sve_f64), svuzp2_f64(a.sve_f64, b.sve_f64));
    return r;
  #else
    return easysimd_mm_sub_pd(easysimd_x_mm_deinterleaveeven_pd(a, b), easysimd_x_mm_deinterleaveodd_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hsub_pd(a, b) easysimd_mm_hsub_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_hsub_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_hsub_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsub_f32_z(svptrue_b32(), svuzp1_f32(a.sve_f32, b.sve_f32), svuzp2_f32(a.sve_f32, b.sve_f32));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    float32x4x2_t t = vuzpq_f32(easysimd__m128_to_neon_f32(a), easysimd__m128_to_neon_f32(b));
    return easysimd__m128_from_neon_f32(vaddq_f32(t.val[0], vnegq_f32(t.val[1])));
  #else
    return easysimd_mm_sub_ps(easysimd_x_mm_deinterleaveeven_ps(a, b), easysimd_x_mm_deinterleaveodd_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_hsub_ps(a, b) easysimd_mm_hsub_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_lddqu_si128 (easysimd__m128i const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_lddqu_si128(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svld1_s32(svptrue_b32(), (int32_t const *)mem_addr);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128i r;
    r.neon_i32 = vld1q_s32((int32_t const *)mem_addr);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_lddqu_si128(mem_addr) easysimd_mm_lddqu_si128(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_loaddup_pd (easysimd_float64 const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_loaddup_pd(mem_addr);
  #else
    easysimd__m128d_private r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vdupq_n_f64(*mem_addr);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vdupq_n_s64(*HEDLEY_REINTERPRET_CAST(int64_t const*, mem_addr));
    #else
      r_.f64[0] = *mem_addr;
      r_.f64[1] = *mem_addr;
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_loaddup_pd(mem_addr) easysimd_mm_loaddup_pd(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_movedup_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_movedup_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdup_n_f64(a.f64[0]);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f64 = vdupq_laneq_f64(a_.neon_f64, 0);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 16, a_.f64, a_.f64, 0, 0);
    #else
      r_.f64[0] = a_.f64[0];
      r_.f64[1] = a_.f64[0];
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_movedup_pd(a) easysimd_mm_movedup_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_movehdup_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE3_NATIVE)
    return _mm_movehdup_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svtrn2_f32(a.sve_f32, a.sve_f32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vtrn2q_f32(a.neon_f32, a.neon_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, a_.f32, 1, 1, 3, 3);
    #else
      r_.f32[0] = a_.f32[1];
      r_.f32[1] = a_.f32[1];
      r_.f32[2] = a_.f32[3];
      r_.f32[3] = a_.f32[3];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_movehdup_ps(a) easysimd_mm_movehdup_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_moveldup_ps (easysimd__m128 a) {
  #if defined(EASYSIMD__SSE3_NATIVE)
    return _mm_moveldup_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svtrn1_f32(a.sve_f32, a.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f32 = vtrn1q_f32(a_.neon_f32, a_.neon_f32);
    #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, a_.f32, 0, 0, 2, 2);
    #else
      r_.f32[0] = a_.f32[0];
      r_.f32[1] = a_.f32[0];
      r_.f32[2] = a_.f32[2];
      r_.f32[3] = a_.f32[2];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES)
#  define _mm_moveldup_ps(a) easysimd_mm_moveldup_ps(a)
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_SSE3_H) */
