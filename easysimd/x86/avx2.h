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
 *   2018-2020 Evan Nemerson <evan@nemerson.com>
 *   2019-2020 Michael R. Crusoe <crusoe@debian.org>
 *   2020      Himanshi Mathur <himanshi18037@iiitd.ac.in>
 *   2020      Hidayat Khan <huk2209@gmail.com>
 */

#if !defined(EASYSIMD_X86_AVX2_H)
#define EASYSIMD_X86_AVX2_H

#include "avx.h"
#if defined(EASYSIMD_ARM_SVE_NATIVE)
#include "../arm/sve.h"
#endif
#include <easysimd/x86/avx512/setzero.h>
HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_abs_epi8 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_abs_epi8(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i8 = vabsq_s8(a.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vabsq_s8(a.m128i[1].neon_i8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svabs_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svabs_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_abs_epi8(a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_abs_epi8(a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] < INT32_C(0)) ? -a_.i8[i] : a_.i8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_abs_epi8
  #define _mm256_abs_epi8(a) easysimd_mm256_abs_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_abs_epi16 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_abs_epi16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svabs_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svabs_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i16 = vabsq_s16(a.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vabsq_s16(a.m128i[1].neon_i16);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_abs_epi16(a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_abs_epi16(a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] < INT32_C(0)) ? -a_.i16[i] : a_.i16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_abs_epi16
  #define _mm256_abs_epi16(a) easysimd_mm256_abs_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_abs_epi32(easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_abs_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svabs_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svabs_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vabsq_s32(a.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vabsq_s32(a.m128i[1].neon_i32);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_abs_epi32(a_.m128i[0]);
      r_.m128i[1] = easysimd_mm_abs_epi32(a_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0; i < (sizeof(r_.i32) / sizeof(r_.i32[0])); i++) {
        r_.i32[i] = (a_.i32[i] < INT32_C(0)) ? -a_.i32[i] : a_.i32[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_abs_epi32
  #define _mm256_abs_epi32(a) easysimd_mm256_abs_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_add_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_add_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i8 = vaddq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vaddq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_add_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_add_epi8(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = a_.i8 + b_.i8;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[i] + b_.i8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_add_epi8
  #define _mm256_add_epi8(a, b) easysimd_mm256_add_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_add_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_add_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i16 = vaddq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vaddq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
      
    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_add_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_add_epi16(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = a_.i16 + b_.i16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i16[i] + b_.i16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_add_epi16
  #define _mm256_add_epi16(a, b) easysimd_mm256_add_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_hadd_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_hadd_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svadd_s16_z(pg, svuzp1_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), svuzp2_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svadd_s16_z(pg, svuzp1_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), svuzp2_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    return easysimd_mm256_add_epi16(easysimd_x_mm256_deinterleaveeven_epi16(a, b), easysimd_x_mm256_deinterleaveodd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hadd_epi16
  #define _mm256_hadd_epi16(a, b) easysimd_mm256_hadd_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_add_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_add_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svadd_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svadd_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vaddq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vaddq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_add_epi32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_add_epi32(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 + b_.i32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] + b_.i32[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_add_epi32
  #define _mm256_add_epi32(a, b) easysimd_mm256_add_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_hadd_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_hadd_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svadd_s32_z(pg, svuzp1_s32(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), svuzp2_s32(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svadd_s32_z(pg, svuzp1_s32(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), svuzp2_s32(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    return easysimd_mm256_add_epi32(easysimd_x_mm256_deinterleaveeven_epi32(a, b), easysimd_x_mm256_deinterleaveodd_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hadd_epi32
  #define _mm256_hadd_epi32(a, b) easysimd_mm256_hadd_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_add_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_add_epi64(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svadd_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svadd_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i64 = vaddq_s64(a.m128i[0].neon_i64, b.m128i[0].neon_i64);
    r.m128i[1].neon_i64 = vaddq_s64(a.m128i[1].neon_i64, b.m128i[1].neon_i64);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_add_epi64(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_add_epi64(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && !defined(EASYSIMD_BUG_CLANG_BAD_VI64_OPS)
      r_.i64 = a_.i64 + b_.i64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] + b_.i64[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_add_epi64
  #define _mm256_add_epi64(a, b) easysimd_mm256_add_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_alignr_epi8 (easysimd__m256i a, easysimd__m256i b, int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  if (HEDLEY_UNLIKELY(count > 31))
    return easysimd_mm256_setzero_si256();

  for (size_t h = 0 ; h < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; h++) {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.m128i_private[h].i8) / sizeof(r_.m128i_private[h].i8[0])) ; i++) {
      const int srcpos = count + HEDLEY_STATIC_CAST(int, i);
      if (srcpos > 31) {
        r_.m128i_private[h].i8[i] = 0;
      } else if (srcpos > 15) {
        r_.m128i_private[h].i8[i] = a_.m128i_private[h].i8[(srcpos) & 15];
      } else {
        r_.m128i_private[h].i8[i] = b_.m128i_private[h].i8[srcpos];
      }
    }
  }

  return easysimd__m256i_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX2_NATIVE) && !defined(EASYSIMD_BUG_PGI_30106)
#  define easysimd_mm256_alignr_epi8(a, b, count) _mm256_alignr_epi8(a, b, count)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  EASYSIMD_FUNCTION_ATTRIBUTES
  easysimd__m256i
  easysimd__m256i_from_sve_i8(svint8_t svlo, svint8_t svhi) {
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svlo;
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svhi;
    return r;
  }

  #define easysimd_mm256_alignr_epi8(a, b, count) ({  \
    easysimd__m256i_from_sve_i8( \
      ((count) > 31) \
        ? svdup_n_s8(0) \
        : ( \
          ((count) > 15) \
            ? (svext_s8((a).sve_i8[EASYSIMD_SV_INDEX_0], svdup_n_s8(0), (count) & 15)) \
            : (svext_s8((b).sve_i8[EASYSIMD_SV_INDEX_0], (a).sve_i8[EASYSIMD_SV_INDEX_0], ((count) & 15)))),  \
      ((count) > 31) \
        ? svdup_n_s8(0) \
        : ( \
          ((count) > 15) \
            ? (svext_s8((a).sve_i8[EASYSIMD_SV_INDEX_1], svdup_n_s8(0), (count) & 15)) \
            : (svext_s8((b).sve_i8[EASYSIMD_SV_INDEX_1], (a).sve_i8[EASYSIMD_SV_INDEX_1], ((count) & 15)))) \
    );  \
  })
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_alignr_epi8(a, b, count) \
      easysimd_mm256_set_m128i( \
          easysimd_mm_alignr_epi8(easysimd_mm256_extracti128_si256(a, 1), easysimd_mm256_extracti128_si256(b, 1), (count)), \
          easysimd_mm_alignr_epi8(easysimd_mm256_extracti128_si256(a, 0), easysimd_mm256_extracti128_si256(b, 0), (count)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_alignr_epi8
  #define _mm256_alignr_epi8(a, b, count) easysimd_mm256_alignr_epi8(a, b, (count))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m512i
easysimd_mm512_alignr_epi8 (easysimd__m512i a, easysimd__m512i b, int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
#if defined(EASYSIMD_X86_AVX512BW_NATIVE)
  return _mm512_alignr_epi8(a, b, count);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m512i r;
  sveint8_t temp[3];
  int offset = count > 32 ? 32 : count;
  temp[0] = b.sve_i8[EASYSIMD_SV_INDEX_0];
  temp[1] = a.sve_i8[EASYSIMD_SV_INDEX_0];
  temp[2] = svdup_n_s8(0);
  r.sve_i8[EASYSIMD_SV_INDEX_0] = svld1_s8(svptrue_b8(), (int8_t *)temp + offset);

  temp[0] = b.sve_i8[EASYSIMD_SV_INDEX_1];
  temp[1] = a.sve_i8[EASYSIMD_SV_INDEX_1];
  temp[2] = svdup_n_s8(0);
  r.sve_i8[EASYSIMD_SV_INDEX_1] = svld1_s8(svptrue_b8(), (int8_t *)temp + offset);

  temp[0] = b.sve_i8[EASYSIMD_SV_INDEX_2];
  temp[1] = a.sve_i8[EASYSIMD_SV_INDEX_2];
  temp[2] = svdup_n_s8(0);
  r.sve_i8[EASYSIMD_SV_INDEX_2] = svld1_s8(svptrue_b8(), (int8_t *)temp + offset);

  temp[0] = b.sve_i8[EASYSIMD_SV_INDEX_3];
  temp[1] = a.sve_i8[EASYSIMD_SV_INDEX_3];
  temp[2] = svdup_n_s8(0);
  r.sve_i8[EASYSIMD_SV_INDEX_3] = svld1_s8(svptrue_b8(), (int8_t *)temp + offset);
  
  return r;
#else
  easysimd__m512i_private
    r_,
    a_ = easysimd__m512i_to_private(a),
    b_ = easysimd__m512i_to_private(b);

  if (HEDLEY_UNLIKELY(count > 31))
    return easysimd_mm512_setzero_si512();

  for (size_t h = 0 ; h < (sizeof(r_.m128i) / sizeof(r_.m128i[0])) ; h++) {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.m128i_private[h].i8) / sizeof(r_.m128i_private[h].i8[0])) ; i++) {
      const int srcpos = count + HEDLEY_STATIC_CAST(int, i);
      if (srcpos > 31) {
        r_.m128i_private[h].i8[i] = 0;
      } else if (srcpos > 15) {
        r_.m128i_private[h].i8[i] = a_.m128i_private[h].i8[(srcpos) & 15];
      } else {
        r_.m128i_private[h].i8[i] = b_.m128i_private[h].i8[srcpos];
      }
    }
  }

  return easysimd__m512i_from_private(r_);
#endif
}


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_alignr_epi32 (easysimd__m256i a, easysimd__m256i b, const int imm8) {
#if defined(EASYSIMD_X86_AVX512F_NATIVE)
  return _mm256_alignr_epi32(a, b, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  int32_t array[16] = {0};
  svbool_t pg = svptrue_b32();
  svst1_s32(pg, &(array[EASYSIMD_SV_INDEX_0     ]), b.sve_i32[EASYSIMD_SV_INDEX_0]);
  svst1_s32(pg, &(array[EASYSIMD_SV_INDEX_1 << 2]), b.sve_i32[EASYSIMD_SV_INDEX_1]);
  svst1_s32(pg, &(array[EASYSIMD_SV_INDEX_0     ]) + 8, a.sve_i32[EASYSIMD_SV_INDEX_0]);
  svst1_s32(pg, &(array[EASYSIMD_SV_INDEX_1 << 2]) + 8, a.sve_i32[EASYSIMD_SV_INDEX_1]);

  r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(pg, &(array[(EASYSIMD_SV_INDEX_0 << 2) + (imm8 & 0x07)]));
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(pg, &(array[(EASYSIMD_SV_INDEX_1 << 2) + (imm8 & 0x07)]));

  return r;
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  easysimd__m256i r;
  int32_t array[16] = {0};
  vst1q_s32(&(array[ 0]), b.m128i[0].neon_i32);
  vst1q_s32(&(array[ 4]), b.m128i[1].neon_i32);
  vst1q_s32(&(array[ 8]), a.m128i[0].neon_i32);
  vst1q_s32(&(array[12]), a.m128i[1].neon_i32);

  r.m128i[0].neon_i32 = vld1q_s32(&(array[0 + (imm8 & 0x07)]));
  r.m128i[1].neon_i32 = vld1q_s32(&(array[4 + (imm8 & 0x07)]));

  return r;
#else
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  int32_t array[16] = {0};
  easysimd_memcpy(&(array[0]), &(b_.i32), sizeof(b_));
  easysimd_memcpy(&(array[8]), &(a_.i32), sizeof(a_));
  for(int i = 0; i < 8; i++){
    r_.i32[i] = array[i + imm8];
  }

  return easysimd__m256i_from_private(r_);
#endif

}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_alignr_epi32
  #define _mm256_alignr_epi32(a, b, imm8) easysimd_mm256_alignr_epi32(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_and_si256 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_and_si256(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svand_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svand_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vandq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vandq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_and_si128(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_and_si128(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f & b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] & b_.i64[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_and_si256
  #define _mm256_and_si256(a, b) easysimd_mm256_and_si256(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_andnot_si256 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_andnot_si256(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svbic_s32_z(pg, b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vbicq_s32(b.m128i[0].neon_i32, a.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vbicq_s32(b.m128i[1].neon_i32, a.m128i[1].neon_i32);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_andnot_si128(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_andnot_si128(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = ~(a_.i32f[i]) & b_.i32f[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_andnot_si256
  #define _mm256_andnot_si256(a, b) easysimd_mm256_andnot_si256(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_adds_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_adds_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svqadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svqadd_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i8 = vqaddq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vqaddq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_adds_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_adds_epi8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = easysimd_math_adds_i8(a_.i8[i], b_.i8[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_adds_epi8
  #define _mm256_adds_epi8(a, b) easysimd_mm256_adds_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_adds_epi16(easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_adds_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svqadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svqadd_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i16 = vqaddq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vqaddq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_adds_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_adds_epi16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = easysimd_math_adds_i16(a_.i16[i], b_.i16[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_adds_epi16
  #define _mm256_adds_epi16(a, b) easysimd_mm256_adds_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_hadds_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_hadds_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svqadd_s16_z(pg, svuzp1_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), svuzp2_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svqadd_s16_z(pg, svuzp1_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), svuzp2_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    return easysimd_mm256_adds_epi16(easysimd_x_mm256_deinterleaveeven_epi16(a, b), easysimd_x_mm256_deinterleaveodd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hadds_epi16
  #define _mm256_hadds_epi16(a, b) easysimd_mm256_hadds_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_adds_epu8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_adds_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svqadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svqadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_u8 = vqaddq_u8(a.m128i[0].neon_u8, b.m128i[0].neon_u8);
    r.m128i[1].neon_u8 = vqaddq_u8(a.m128i[1].neon_u8, b.m128i[1].neon_u8);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_adds_epu8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_adds_epu8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = easysimd_math_adds_u8(a_.u8[i], b_.u8[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_adds_epu8
  #define _mm256_adds_epu8(a, b) easysimd_mm256_adds_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_adds_epu16(easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_adds_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svqadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svqadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_u16 = vqaddq_u16(a.m128i[0].neon_u16, b.m128i[0].neon_u16);
    r.m128i[1].neon_u16 = vqaddq_u16(a.m128i[1].neon_u16, b.m128i[1].neon_u16);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_adds_epu16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_adds_epu16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = easysimd_math_adds_u16(a_.u16[i], b_.u16[i]);
    }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_adds_epu16
  #define _mm256_adds_epu16(a, b) easysimd_mm256_adds_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_avg_epu8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_avg_epu8(a, b);

  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_u8 = vrhaddq_u8(a.m128i[0].neon_u8, b.m128i[0].neon_u8);
    r.m128i[1].neon_u8 = vrhaddq_u8(a.m128i[1].neon_u8, b.m128i[1].neon_u8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svrhadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svrhadd_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = (a_.u8[i] + b_.u8[i] + 1) >> 1;
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_avg_epu8
  #define _mm256_avg_epu8(a, b) easysimd_mm256_avg_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_avg_epu16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_avg_epu16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_u16 = vrhaddq_u16(a.m128i[0].neon_u16, b.m128i[0].neon_u16);
    r.m128i[1].neon_u16 = vrhaddq_u16(a.m128i[1].neon_u16, b.m128i[1].neon_u16);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svrhadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svrhadd_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = (a_.u16[i] + b_.u16[i] + 1) >> 1;
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_avg_epu16
  #define _mm256_avg_epu16(a, b) easysimd_mm256_avg_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_blend_epi32(easysimd__m128i a, easysimd__m128i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svsel_s32(EASYSIMD_MASK_TO_B32(imm8, EASYSIMD_SV_INDEX_0), b.sve_i32, a.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((imm8 >> i) & 1) ? b_.i32[i] : a_.i32[i];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm_blend_epi32(a, b, imm8) _mm_blend_epi32(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE_LE(128)
#  define easysimd_mm_blend_epi32(a, b, imm8) \
  easysimd_mm_castps_si128(easysimd_mm_blend_ps(easysimd_mm_castsi128_ps(a), easysimd_mm_castsi128_ps(b), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_blend_epi32
  #define _mm_blend_epi32(a, b, imm8) easysimd_mm_blend_epi32(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_blend_epi16(easysimd__m256i a, easysimd__m256i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(EASYSIMD_MASK_TO_B16(imm8, EASYSIMD_SV_INDEX_0), b.sve_i16[EASYSIMD_SV_INDEX_0], a.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(EASYSIMD_MASK_TO_B16(imm8, EASYSIMD_SV_INDEX_0), b.sve_i16[EASYSIMD_SV_INDEX_1], a.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = ((imm8 >> i%8) & 1) ? b_.i16[i] : a_.i16[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE) && defined(EASYSIMD_BUG_CLANG_REV_234560)
#  define easysimd_mm256_blend_epi16(a, b, imm8) _mm256_castpd_si256(_mm256_blend_epi16(a, b, imm8))
#elif defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_blend_epi16(a, b, imm8) _mm256_blend_epi16(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_blend_epi16(a, b, imm8) \
      easysimd_mm256_set_m128i( \
          easysimd_mm_blend_epi16(easysimd_mm256_extracti128_si256(a, 1), easysimd_mm256_extracti128_si256(b, 1), (imm8)), \
          easysimd_mm_blend_epi16(easysimd_mm256_extracti128_si256(a, 0), easysimd_mm256_extracti128_si256(b, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_blend_epi16
  #define _mm256_blend_epi16(a, b, imm8) easysimd_mm256_blend_epi16(a, b, imm8)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_blend_epi32(easysimd__m256i a, easysimd__m256i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(EASYSIMD_MASK_TO_B32(imm8, EASYSIMD_SV_INDEX_0), b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(EASYSIMD_MASK_TO_B32(imm8, EASYSIMD_SV_INDEX_1), b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    uint32_t g_mask_epi32[4] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08};
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    uint32x4_t vect_imm = vdupq_n_u32(imm8);
    uint32x4_t flag[2];
    flag[0] = vtstq_u32(vect_imm, vect_mask);
    flag[1] = vtstq_u32(vshrq_n_u32(vect_imm, 4), vect_mask);
    r.m128i[0].neon_i32 = vbslq_s32(flag[0], b.m128i[0].neon_i32, a.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vbslq_s32(flag[1], b.m128i[1].neon_i32, a.m128i[1].neon_i32);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ((imm8 >> i) & 1) ? b_.i32[i] : a_.i32[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_blend_epi32(a, b, imm8) _mm256_blend_epi32(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_blend_epi32(a, b, imm8) \
      easysimd_mm256_set_m128i( \
          easysimd_mm_blend_epi32(easysimd_mm256_extracti128_si256(a, 1), easysimd_mm256_extracti128_si256(b, 1), (imm8) >> 4), \
          easysimd_mm_blend_epi32(easysimd_mm256_extracti128_si256(a, 0), easysimd_mm256_extracti128_si256(b, 0), (imm8) & 0x0F))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_blend_epi32
  #define _mm256_blend_epi32(a, b, imm8) easysimd_mm256_blend_epi32(a, b, imm8)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_blendv_epi8(easysimd__m256i a, easysimd__m256i b, easysimd__m256i mask) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_blendv_epi8(a, b, mask);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    uint8x16_t vect_flag[2];
    vect_flag[0] = vcgeq_s8(mask.m128i[0].neon_i8, vdupq_n_s8(0));
    vect_flag[1] = vcgeq_s8(mask.m128i[1].neon_i8, vdupq_n_s8(0));
    r.m128i[0].neon_i8 = vbslq_s8(vect_flag[0], a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vbslq_s8(vect_flag[1], a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b8();
    easysimd_svbool_t pgm1 = svcmpeq_n_u8(pg, svlsr_n_u8_z(pg, mask.sve_u8[EASYSIMD_SV_INDEX_0], 7), 1);
    easysimd_svbool_t pgm2 = svcmpeq_n_u8(pg, svlsr_n_u8_z(pg, mask.sve_u8[EASYSIMD_SV_INDEX_1], 7), 1);

    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(pgm1, b.sve_i8[EASYSIMD_SV_INDEX_0], a.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(pgm2, b.sve_i8[EASYSIMD_SV_INDEX_1], a.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b),
      mask_ = easysimd__m256i_to_private(mask);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_blendv_epi8(a_.m128i[0], b_.m128i[0], mask_.m128i[0]);
      r_.m128i[1] = easysimd_mm_blendv_epi8(a_.m128i[1], b_.m128i[1], mask_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      __typeof__(mask_.i8) tmp = mask_.i8 >> 7;
      r_.i8 = (tmp & b_.i8) | (~tmp & a_.i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        int8_t tmp = mask_.i8[i] >> 7;
        r_.i8[i] = (tmp & b_.i8[i]) | (~tmp & a_.i8[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_blendv_epi8(a, b, imm8)  _mm256_blendv_epi8(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_blendv_epi8
  #define _mm256_blendv_epi8(a, b, mask) easysimd_mm256_blendv_epi8(a, b, mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_broadcastb_epi8 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm_broadcastb_epi8(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i8 = svdup_n_s8(a.i8[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i8 = vdupq_n_s8(a.i8[0]);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = a_.i8[0];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcastb_epi8
  #define _mm_broadcastb_epi8(a) easysimd_mm_broadcastb_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_broadcastb_epi8 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_broadcastb_epi8(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svdup_n_s8(a.i8[0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svdup_n_s8(a.i8[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i8 = vdupq_n_s8(a.i8[0]);
    r.m128i[1].neon_i8 = vdupq_n_s8(a.i8[0]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = a_.i8[0];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcastb_epi8
  #define _mm256_broadcastb_epi8(a) easysimd_mm256_broadcastb_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_broadcastw_epi16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm_broadcastw_epi16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i16 = svdup_n_s16(a.i16[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i16 = vdupq_n_s16(a.i16[0]);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = a_.i16[0];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcastw_epi16
  #define _mm_broadcastw_epi16(a) easysimd_mm_broadcastw_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_broadcastw_epi16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_broadcastw_epi16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svdup_n_s16(a.i16[0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svdup_n_s16(a.i16[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i16 = vdupq_n_s16(a.i16[0]);
    r.m128i[1].neon_i16 = vdupq_n_s16(a.i16[0]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = a_.i16[0];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcastw_epi16
  #define _mm256_broadcastw_epi16(a) easysimd_mm256_broadcastw_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_broadcastd_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm_broadcastd_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svdup_n_s32(a.i32[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i32 = vdupq_n_s32(a.i32[0]);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[0];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcastd_epi32
  #define _mm_broadcastd_epi32(a) easysimd_mm_broadcastd_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_broadcastd_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_broadcastd_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdup_n_s32(a.i32[0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32(a.i32[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vdupq_n_s32(a.i32[0]);
    r.m128i[1].neon_i32 = vdupq_n_s32(a.i32[0]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[0];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcastd_epi32
  #define _mm256_broadcastd_epi32(a) easysimd_mm256_broadcastd_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_broadcastq_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm_broadcastq_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i64 = svdup_n_s64(a.i64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_i64 = vdupq_n_s64(a.i64[0]);
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[0];
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcastq_epi64
  #define _mm_broadcastq_epi64(a) easysimd_mm_broadcastq_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_broadcastq_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_broadcastq_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = r.sve_i64[EASYSIMD_SV_INDEX_1] = svdup_n_s64(a.i64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    __asm__ __volatile__ (
        "dup %[r0].2d, %[a].d[0]           \n\t"
        "dup %[r1].2d, %[a].d[0]           \n\t"
        :[r0]"=w"(res.m128i[0].neon_i64), [r1]"=w"(res.m128i[1].neon_i64)
        :[a]"w"(a.neon_i64)
        :
    );
    return res;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_= easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i64[0];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcastq_epi64
  #define _mm256_broadcastq_epi64(a) easysimd_mm256_broadcastq_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_broadcastss_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm_broadcastss_ps(a);
  #elif defined(EASYSIMD_X86_SSE_NATIVE)
    return easysimd_mm_shuffle_ps(a, a, 0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdup_n_f32(a.f32[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vdupq_n_f32(a.f32[0]);
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m128_private a_= easysimd__m128_to_private(a);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, a_.f32, 0, 0, 0, 0);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[0];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcastss_ps
  #define _mm_broadcastss_ps(a) easysimd_mm_broadcastss_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_broadcastss_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_broadcastss_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svdup_n_f32(a.f32[0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svdup_n_f32(a.f32[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0].neon_f32 = vdupq_n_f32(a.f32[0]);
    r.m128[1].neon_f32 = vdupq_n_f32(a.f32[0]);
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m128_private a_= easysimd__m128_to_private(a);

    #if defined(EASYSIMD_X86_AVX_NATIVE)
      __m128 tmp = _mm_permute_ps(a_.n, 0);
      r_.n = _mm256_insertf128_ps(_mm256_castps128_ps256(tmp), tmp, 1);
    #elif HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      r_.f32 = __builtin_shufflevector(a_.f32, a_.f32, 0, 0, 0, 0, 0, 0, 0, 0);
    #elif EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE_LE(128)
      r_.m128[0] = r_.m128[1] = easysimd_mm_broadcastss_ps(easysimd__m128_from_private(a_));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[0];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcastss_ps
  #define _mm256_broadcastss_ps(a) easysimd_mm256_broadcastss_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_broadcastsd_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = svdup_n_f64(a.f64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128d r;
    r.neon_f64 = vdupq_n_f64(a.f64[0]);
    return r;
  #else
    return easysimd_mm_movedup_pd(a);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcastsd_pd
  #define _mm_broadcastsd_pd(a) easysimd_mm_broadcastsd_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_broadcastsd_pd (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_broadcastsd_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svdup_n_f64(a.f64[0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svdup_n_f64(a.f64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    r.m128d[0].neon_f64 = vdupq_n_f64(a.f64[0]);
    r.m128d[1].neon_f64 = vdupq_n_f64(a.f64[0]);
    return r;
  #else
    easysimd__m256d_private r_;
    easysimd__m128d_private a_= easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = a_.f64[0];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcastsd_pd
  #define _mm256_broadcastsd_pd(a) easysimd_mm256_broadcastsd_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_broadcastsi128_si256 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE) && \
      (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(4,8,0))
    return _mm256_broadcastsi128_si256(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_1] = r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32;
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0] = a;
    res.m128i[1] = a;
    return res;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i_private[0] = a_;
      r_.m128i_private[1] = a_;
    #else
      r_.i64[0] = a_.i64[0];
      r_.i64[1] = a_.i64[1];
      r_.i64[2] = a_.i64[0];
      r_.i64[3] = a_.i64[1];
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#define easysimd_mm_broadcastsi128_si256(a) easysimd_mm256_broadcastsi128_si256(a)
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcastsi128_si256
  #define _mm256_broadcastsi128_si256(a) easysimd_mm256_broadcastsi128_si256(a)
  #undef _mm_broadcastsi128_si256
  #define _mm_broadcastsi128_si256(a) easysimd_mm256_broadcastsi128_si256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_bslli_epi128 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    sveuint8_t svid = svindex_u8(imm8, 1);
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svrev_s8(svtbl_s8(svrev_s8(a.sve_i8[EASYSIMD_SV_INDEX_0]), svid));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svrev_s8(svtbl_s8(svrev_s8(a.sve_i8[EASYSIMD_SV_INDEX_1]), svid));
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    const int ssize = HEDLEY_STATIC_CAST(int, (sizeof(r_.i8) / sizeof(r_.i8[0])));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < ssize ; i++) {
      const int e = i - imm8;
      if(i >= (ssize/2)) {
        if(e >= (ssize/2) && e < ssize)
          r_.i8[i] = a_.i8[e];
        else
          r_.i8[i] = 0;
      }
      else{
        if(e >= 0 && e < (ssize/2))
          r_.i8[i] = a_.i8[e];
        else
          r_.i8[i] = 0;
      }
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE) && \
    (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(4,8,0)) && \
    EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,7,0)
  #define easysimd_mm256_bslli_epi128(a, imm8) _mm256_bslli_epi128(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_bslli_epi128
  #define _mm256_bslli_epi128(a, imm8) easysimd_mm256_bslli_epi128(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_bsrli_epi128 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    sveuint8_t svid = svindex_u8(imm8, 1);
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svtbl_s8(a.sve_i8[EASYSIMD_SV_INDEX_0], svid);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svtbl_s8(a.sve_i8[EASYSIMD_SV_INDEX_1], svid);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    const int ssize = HEDLEY_STATIC_CAST(int, (sizeof(r_.i8) / sizeof(r_.i8[0])));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < ssize ; i++) {
      const int e = i + imm8;
      if(i < (ssize/2)) {
        if(e >= 0 && e < (ssize/2))
          r_.i8[i] = a_.i8[e];
        else
          r_.i8[i] = 0;
      }
      else{
        if(e >= (ssize/2) && e < ssize)
          r_.i8[i] = a_.i8[e];
        else
          r_.i8[i] = 0;
      }
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE) && \
    (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(4,8,0)) && \
    EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,7,0)
  #define easysimd_mm256_bsrli_epi128(a, imm8) _mm256_bsrli_epi128(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_bsrli_epi128
  #define _mm256_bsrli_epi128(a, imm8) easysimd_mm256_bsrli_epi128(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cmpeq_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cmpeq_epi8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_u8 = vceqq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    res.m128i[1].neon_u8 = vceqq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svdup_n_s8_z(svcmpeq_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), 0xFF);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svdup_n_s8_z(svcmpeq_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), 0xFF);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_cmpeq_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_cmpeq_epi8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] == b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpeq_epi8
  #define _mm256_cmpeq_epi8(a, b) easysimd_mm256_cmpeq_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cmpeq_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cmpeq_epi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_u16 = vceqq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    res.m128i[1].neon_u16 = vceqq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svdup_n_s16_z(svcmpeq_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), 0xFFFF);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svdup_n_s16_z(svcmpeq_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), 0xFFFF);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_cmpeq_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_cmpeq_epi16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] == b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpeq_epi16
  #define _mm256_cmpeq_epi16(a, b) easysimd_mm256_cmpeq_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cmpeq_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cmpeq_epi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_u32 = vceqq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_u32 = vceqq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdup_n_s32_z(svcmpeq_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), 0xFFFFFFFF);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32_z(svcmpeq_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), 0xFFFFFFFF);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_cmpeq_epi32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_cmpeq_epi32(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = (a_.i32[i] == b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpeq_epi32
  #define _mm256_cmpeq_epi32(a, b) easysimd_mm256_cmpeq_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cmpeq_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cmpeq_epi64(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_u64 = vceqq_s64(a.m128i[0].neon_i64, b.m128i[0].neon_i64);
    res.m128i[1].neon_u64 = vceqq_s64(a.m128i[1].neon_i64, b.m128i[1].neon_i64);
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svdup_n_s64_z(svcmpeq_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), 0xFFFFFFFFFFFFFFFF);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svdup_n_s64_z(svcmpeq_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), 0xFFFFFFFFFFFFFFFF);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_cmpeq_epi64(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_cmpeq_epi64(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = (a_.i64[i] == b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpeq_epi64
  #define _mm256_cmpeq_epi64(a, b) easysimd_mm256_cmpeq_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cmpgt_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cmpgt_epi8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_u8 = vcgtq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    res.m128i[1].neon_u8 = vcgtq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svdup_n_s8_z(svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]), 0xFF);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svdup_n_s8_z(svcmpgt_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]), 0xFF);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_cmpgt_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_cmpgt_epi8(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i8), a_.i8 > b_.i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = (a_.i8[i] > b_.i8[i]) ? ~INT8_C(0) : INT8_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epi8
  #define _mm256_cmpgt_epi8(a, b) easysimd_mm256_cmpgt_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cmpgt_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cmpgt_epi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_u16 = vcgtq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    res.m128i[1].neon_u16 = vcgtq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svdup_n_s16_z(svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), 0xFFFF);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svdup_n_s16_z(svcmpgt_s16(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), 0xFFFF);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_cmpgt_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_cmpgt_epi16(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = a_.i16 > b_.i16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? ~INT16_C(0) : INT16_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epi16
  #define _mm256_cmpgt_epi16(a, b) easysimd_mm256_cmpgt_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cmpgt_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cmpgt_epi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_u32 = vcgtq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_u32 = vcgtq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdup_n_s32_z(svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), 0xFFFFFFFF);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32_z(svcmpgt_s32(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), 0xFFFFFFFF);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_cmpgt_epi32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_cmpgt_epi32(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.i32 > b_.i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = (a_.i32[i] > b_.i32[i]) ? ~INT32_C(0) : INT32_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epi32
  #define _mm256_cmpgt_epi32(a, b) easysimd_mm256_cmpgt_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cmpgt_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cmpgt_epi64(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_u64 = vcgtq_s64(a.m128i[0].neon_i64, b.m128i[0].neon_i64);
    res.m128i[1].neon_u64 = vcgtq_s64(a.m128i[1].neon_i64, b.m128i[1].neon_i64);
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svdup_n_s64_z(svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]), 0xFFFFFFFFFFFFFFFF);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svdup_n_s64_z(svcmpgt_s64(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]), 0xFFFFFFFFFFFFFFFF);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_cmpgt_epi64(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_cmpgt_epi64(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 > b_.i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = (a_.i64[i] > b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmpgt_epi64
  #define _mm256_cmpgt_epi64(a, b) easysimd_mm256_cmpgt_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepi8_epi16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepi8_epi16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svld1sb_s16(pg, &(a.i8[EASYSIMD_SV_INDEX_0 * 8]));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svld1sb_s16(pg, &(a.i8[EASYSIMD_SV_INDEX_1 * 8]));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i16, a_.i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepi8_epi16
  #define _mm256_cvtepi8_epi16(a) easysimd_mm256_cvtepi8_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepi8_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepi8_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1sb_s32(pg, &a.i8[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1sb_s32(pg, &a.i8[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i32, a_.m64_private[0].i8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepi8_epi32
  #define _mm256_cvtepi8_epi32(a) easysimd_mm256_cvtepi8_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepi8_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepi8_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1sb_s64(pg, &a.i8[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1sb_s64(pg, &a.i8[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i8[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepi8_epi64
  #define _mm256_cvtepi8_epi64(a) easysimd_mm256_cvtepi8_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepi16_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepi16_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1sh_s32(pg, &a.i16[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1sh_s32(pg, &a.i16[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vmovl_s16(vget_low_s16(a.neon_i16));
    r.m128i[1].neon_i32 = vmovl_s16(vget_high_s16(a.neon_i16));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i32, a_.i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepi16_epi32
  #define _mm256_cvtepi16_epi32(a) easysimd_mm256_cvtepi16_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepi16_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepi16_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1sh_s64(pg, &a.i16[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1sh_s64(pg, &a.i16[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i64, a_.m64_private[0].i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepi16_epi64
  #define _mm256_cvtepi16_epi64(a) easysimd_mm256_cvtepi16_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepi32_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepi32_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1sw_s64(pg, &(a.i32[EASYSIMD_SV_INDEX_0]));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1sw_s64(pg, &(a.i32[EASYSIMD_SV_INDEX_1 * 2]));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.i32[i];
    }
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepi32_epi64
  #define _mm256_cvtepi32_epi64(a) easysimd_mm256_cvtepi32_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_cvtepi64_ps (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cvtepi64_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp1, tmp2, r;
    svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32 = svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32);
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.i64[i]);
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepi64_ps
  #define _mm256_cvtepi64_ps(a) easysimd_mm256_cvtepi64_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_cvtepu64_ps (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm256_cvtepu64_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp1, tmp2, r;
    svbool_t pg = svptrue_b32();
    tmp1.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0]);
    tmp2.sve_f32 = svcvt_f32_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1]);
    r.sve_f32 = svuzp1_f32(tmp1.sve_f32, tmp2.sve_f32);
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.u64[i]);
    }
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepu64_ps
  #define _mm256_cvtepu64_ps(a) easysimd_mm256_cvtepu64_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepu8_epi16 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepu8_epi16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svld1ub_s16(pg, &(a.u8[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 4)]));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svld1ub_s16(pg, &(a.u8[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 4)]));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i16, a_.u8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.u8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepu8_epi16
  #define _mm256_cvtepu8_epi16(a) easysimd_mm256_cvtepu8_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepu8_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepu8_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1ub_s32(pg, &(a.u8[EASYSIMD_SV_INDEX_0 << 2]));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1ub_s32(pg, &(a.u8[EASYSIMD_SV_INDEX_1 << 2]));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i32, a_.m64_private[0].u8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.u8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepu8_epi32
  #define _mm256_cvtepu8_epi32(a) easysimd_mm256_cvtepu8_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepu8_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepu8_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1ub_s64(pg, &(a.u8[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)]));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1ub_s64(pg, &(a.u8[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)]));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.u8[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepu8_epi64
  #define _mm256_cvtepu8_epi64(a) easysimd_mm256_cvtepu8_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepu16_epi32 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepu16_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1uh_s32(pg, &(a.u16[EASYSIMD_SV_INDEX_0 * 4]));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1uh_s32(pg, &(a.u16[EASYSIMD_SV_INDEX_1 * 4]));
    return r;
  #else
      easysimd__m256i_private r_;
      easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.u16[i];
      }
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepu16_epi32
  #define _mm256_cvtepu16_epi32(a) easysimd_mm256_cvtepu16_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepu16_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepu16_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1uh_s64(pg, &(a.u16[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)]));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1uh_s64(pg, &(a.u16[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)]));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.i64, a_.m64_private[0].u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.u16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepu16_epi64
  #define _mm256_cvtepu16_epi64(a) easysimd_mm256_cvtepu16_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtepu32_epi64 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_cvtepu32_epi64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1uw_s64(pg, &(a.u32[EASYSIMD_SV_INDEX_0 * 2]));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1uw_s64(pg, &(a.u32[EASYSIMD_SV_INDEX_1 * 2]));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = a_.u32[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepu32_epi64
  #define _mm256_cvtepu32_epi64(a) easysimd_mm256_cvtepu32_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_extract_epi8 (easysimd__m256i a, const int index)
    EASYSIMD_REQUIRE_RANGE(index, 0, 31){
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.i8[index];
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    return a_.i8[index];
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_extract_epi8(a, index) _mm256_extract_epi8(a, index)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extract_epi8
  #define _mm256_extract_epi8(a, index) easysimd_mm256_extract_epi8(a, index)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_extract_epi16 (easysimd__m256i a, const int index)
    EASYSIMD_REQUIRE_RANGE(index, 0, 15)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.i16[index];
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    return a_.i16[index];
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_extract_epi16(a, index) _mm256_extract_epi16(a, index)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extract_epi16
  #define _mm256_extract_epi16(a, index) easysimd_mm256_extract_epi16(a, index)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm256_extracti128_si256 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.m128i[imm8 & 1];
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128i[imm8 & 1];
  #else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
  return a_.m128i[imm8 & 1];
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_extracti128_si256(a, imm8) _mm256_extracti128_si256(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extracti128_si256
  #define _mm256_extracti128_si256(a, imm8) easysimd_mm256_extracti128_si256(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_i32gather_epi32(const int32_t* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    svint32_t index = svmul_n_s32_z(pg, vindex.sve_i32, scale);
    r.sve_i32 = svld1_gather_s32offset_s32(pg, base_addr, index);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex),
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int32_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i32[i] = dst;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_i32gather_epi32(base_addr, vindex, scale) _mm_i32gather_epi32(EASYSIMD_CHECKED_REINTERPRET_CAST(int const*, int32_t const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_i32gather_epi32
  #define _mm_i32gather_epi32(base_addr, vindex, scale) easysimd_mm_i32gather_epi32(EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, int const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_i32gather_epi32(easysimd__m128i src, const int32_t* base_addr, easysimd__m128i vindex, easysimd__m128i mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    svint32_t index = svmul_n_s32_z(pg, vindex.sve_i32, scale);
    easysimd_svbool_t pgm= svcmplt_n_s32(pg, mask.sve_i32, INT32_C(0));
    r.sve_i32 = svsel_s32(pgm, svld1_gather_s32offset_s32(pg, base_addr, index), src.sve_i32);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex),
      src_ = easysimd__m128i_to_private(src),
      mask_ = easysimd__m128i_to_private(mask),
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      if ((mask_.i32[i] >> 31) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        int32_t dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.i32[i] = dst;
      }
      else {
        r_.i32[i] = src_.i32[i];
      }
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_mask_i32gather_epi32(src, base_addr, vindex, mask, scale) _mm_mask_i32gather_epi32(src, EASYSIMD_CHECKED_REINTERPRET_CAST(int const*, int32_t const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_i32gather_epi32
  #define _mm_mask_i32gather_epi32(src, base_addr, vindex, mask, scale) easysimd_mm_mask_i32gather_epi32(src, EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, int const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_i32gather_epi32(const int32_t* base_addr, easysimd__m256i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    svint32_t index0 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_0], scale);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_gather_s32offset_s32(pg, base_addr, index0);

    svint32_t index1 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_1], scale);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_gather_s32offset_s32(pg, base_addr, index1);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.i32[ 0] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 0] * scale));
    r.i32[ 1] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 1] * scale));
    r.i32[ 2] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 2] * scale));
    r.i32[ 3] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 3] * scale));
    r.i32[ 4] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 4] * scale));
    r.i32[ 5] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 5] * scale));
    r.i32[ 6] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 6] * scale));
    r.i32[ 7] = *((int32_t *)((uint8_t *)base_addr + vindex.i32[ 7] * scale));
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex),
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int32_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i32[i] = dst;
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_i32gather_epi32(base_addr, vindex, scale) _mm256_i32gather_epi32(EASYSIMD_CHECKED_REINTERPRET_CAST(int const*, int32_t const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_i32gather_epi32
  #define _mm256_i32gather_epi32(base_addr, vindex, scale) easysimd_mm256_i32gather_epi32(EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, int const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_i32gather_epi32(easysimd__m256i src, const int32_t* base_addr, easysimd__m256i vindex, easysimd__m256i mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    svint32_t index0 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_0], scale);
    easysimd_svbool_t pgm= svcmplt_n_s32(pg, mask.sve_i32[EASYSIMD_SV_INDEX_0], INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(pgm, svld1_gather_s32offset_s32(pg, base_addr, index0), src.sve_i32[EASYSIMD_SV_INDEX_0]);

    svint32_t index1 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_1], scale);
    pgm= svcmplt_n_s32(pg, mask.sve_i32[EASYSIMD_SV_INDEX_1], INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(pgm, svld1_gather_s32offset_s32(pg, base_addr, index1), src.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex),
      src_ = easysimd__m256i_to_private(src),
      mask_ = easysimd__m256i_to_private(mask),
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      if ((mask_.i32[i] >> 31) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        int32_t dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.i32[i] = dst;
      }
      else {
        r_.i32[i] = src_.i32[i];
      }
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_mask_i32gather_epi32(src, base_addr, vindex, mask, scale) _mm256_mask_i32gather_epi32(src, EASYSIMD_CHECKED_REINTERPRET_CAST(int const*, int32_t const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_i32gather_epi32
  #define _mm256_mask_i32gather_epi32(src, base_addr, vindex, mask, scale) easysimd_mm256_mask_i32gather_epi32(src, EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, int const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_i64gather_epi32(const int32_t* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    sveint32_t index = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)), 0, 0);
    r.sve_i32 = svmul_s32_z(pg, svld1_gather_s32offset_s32(pg, base_addr, index), svdupq_n_s32(1, 1, 0, 0));
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex),
      r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int32_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i32[i] = dst;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_i64gather_epi32(base_addr, vindex, scale) _mm_i64gather_epi32(EASYSIMD_CHECKED_REINTERPRET_CAST(int const*, int32_t const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_i64gather_epi32
  #define _mm_i64gather_epi32(base_addr, vindex, scale) easysimd_mm_i64gather_epi32(EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, int const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_i64gather_epi32(easysimd__m128i src, const int32_t* base_addr, easysimd__m128i vindex, easysimd__m128i mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    sveint32_t index = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)), 0, 0);
    easysimd_svbool_t pgm= svcmplt_n_s32(pg, mask.sve_i32, INT32_C(0));
    r.sve_i32 = svsel_s32(pgm, svld1_gather_s32offset_s32(pg, base_addr, index), src.sve_i32);
    r.sve_i32 = svmul_s32_z(pg, r.sve_i32, svdupq_n_s32(1, 1, 0, 0));
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex),
      src_ = easysimd__m128i_to_private(src),
      mask_ = easysimd__m128i_to_private(mask),
      r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((mask_.i32[i] >> 31) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        int32_t dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.i32[i] = dst;
      }
      else {
        r_.i32[i] = src_.i32[i];
      }
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_mask_i64gather_epi32(src, base_addr, vindex, mask, scale) _mm_mask_i64gather_epi32(src, EASYSIMD_CHECKED_REINTERPRET_CAST(int const*, int32_t const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_i64gather_epi32
  #define _mm_mask_i64gather_epi32(src, base_addr, vindex, mask, scale) easysimd_mm_mask_i64gather_epi32(src, EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, int const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm256_i64gather_epi32(const int32_t* base_addr, easysimd__m256i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    sveint32_t index = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[2])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[3])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_i32 = svld1_gather_s32offset_s32(pg, base_addr, index);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex);
    easysimd__m128i_private
      r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int32_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i32[i] = dst;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_i64gather_epi32(base_addr, vindex, scale) _mm256_i64gather_epi32(EASYSIMD_CHECKED_REINTERPRET_CAST(int const*, int32_t const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_i64gather_epi32
  #define _mm256_i64gather_epi32(base_addr, vindex, scale) easysimd_mm256_i64gather_epi32(EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, int const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm256_mask_i64gather_epi32(easysimd__m128i src, const int32_t* base_addr, easysimd__m256i vindex, easysimd__m128i mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b32();
    sveint32_t index = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[2])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[3])* HEDLEY_STATIC_CAST(size_t, scale)));
    easysimd_svbool_t pgm= svcmplt_n_s32(pg, mask.sve_i32, INT32_C(0));
    r.sve_i32 = svsel_s32(pgm, svld1_gather_s32offset_s32(pg, base_addr, index), src.sve_i32);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex);
    easysimd__m128i_private
      src_ = easysimd__m128i_to_private(src),
      mask_ = easysimd__m128i_to_private(mask),
      r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((mask_.i32[i] >> 31) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        int32_t dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.i32[i] = dst;
      }
      else {
        r_.i32[i] = src_.i32[i];
      }
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_mask_i64gather_epi32(src, base_addr, vindex, mask, scale) _mm256_mask_i64gather_epi32(src, EASYSIMD_CHECKED_REINTERPRET_CAST(int const*, int32_t const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_i64gather_epi32
  #define _mm256_mask_i64gather_epi32(src, base_addr, vindex, mask, scale) easysimd_mm256_mask_i64gather_epi32(src, EASYSIMD_CHECKED_REINTERPRET_CAST(int32_t const*, int const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_i32gather_epi64(const int64_t* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128i r;
  svbool_t pg = svptrue_b64();
  svint64_t index = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_0])), HEDLEY_STATIC_CAST(int64_t, scale));
  r.sve_i64 = svld1_gather_s64offset_s64(pg, base_addr, index);
  return r;
#else
  easysimd__m128i_private
    vindex_ = easysimd__m128i_to_private(vindex),
    r_;
  const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);
  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
    const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
    int64_t dst;
    easysimd_memcpy(&dst, src, sizeof(dst));
    r_.i64[i] = dst;
  }

  return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #if EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm_i32gather_epi64(base_addr, vindex, scale) _mm_i32gather_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, scale)
  #else
    #define easysimd_mm_i32gather_epi64(base_addr, vindex, scale) _mm_i32gather_epi64(HEDLEY_REINTERPRET_CAST(long long const*, base_addr), vindex, scale)
  #endif
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_i32gather_epi64
  #define _mm_i32gather_epi64(base_addr, vindex, scale) easysimd_mm_i32gather_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_i32gather_epi64(easysimd__m128i src, const int64_t* base_addr, easysimd__m128i vindex, easysimd__m128i mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b64();
    svint64_t index = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_0])), HEDLEY_STATIC_CAST(int64_t, scale));
    easysimd_svbool_t pgm= svcmplt_n_s64(pg, mask.sve_i64, INT64_C(0));
    r.sve_i64 = svsel_s64(pgm, svld1_gather_s64offset_s64(pg, base_addr, index), src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex),
      src_ = easysimd__m128i_to_private(src),
      mask_ = easysimd__m128i_to_private(mask),
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      if ((mask_.i64[i] >> 63) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        int64_t dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.i64[i] = dst;
      }
      else {
        r_.i64[i] = src_.i64[i];
      }
   }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #if EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm_mask_i32gather_epi64(src, base_addr, vindex, mask, scale) _mm_mask_i32gather_epi64(src, HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, mask, scale)
  #else
    #define easysimd_mm_mask_i32gather_epi64(src, base_addr, vindex, mask, scale) _mm_mask_i32gather_epi64(src, HEDLEY_REINTERPRET_CAST(long long const*, base_addr), vindex, mask, scale)
  #endif
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_i32gather_epi64
  #define _mm_mask_i32gather_epi64(src, base_addr, vindex, mask, scale) easysimd_mm_mask_i32gather_epi64(src, HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_i32gather_epi64(const int64_t* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  svbool_t pg = svptrue_b64();
  svint64_t index0 = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_0     ])), scale);
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1_gather_s64offset_s64(pg, base_addr, index0);

  svint64_t index1 = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_1 << 1])), scale);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1_gather_s64offset_s64(pg, base_addr, index1);
  return r;
#else
  easysimd__m128i_private
    vindex_ = easysimd__m128i_to_private(vindex);
  easysimd__m256i_private
    r_;
  const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int64_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i64[i] = dst;
    }

  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #if EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm256_i32gather_epi64(base_addr, vindex, scale) _mm256_i32gather_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, scale)
  #else
    #define easysimd_mm256_i32gather_epi64(base_addr, vindex, scale) _mm256_i32gather_epi64(HEDLEY_REINTERPRET_CAST(long long const*, base_addr), vindex, scale)
  #endif
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_i32gather_epi64
  #define _mm256_i32gather_epi64(base_addr, vindex, scale) easysimd_mm256_i32gather_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_i32gather_epi64(easysimd__m256i src, const int64_t* base_addr, easysimd__m128i vindex, easysimd__m256i mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    svint64_t index0 = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)])), HEDLEY_STATIC_CAST(int64_t, scale));
    easysimd_svbool_t pgm= svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_0], INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(pgm, svld1_gather_s64offset_s64(pg, base_addr, index0), src.sve_i64[EASYSIMD_SV_INDEX_0]);

    svint64_t index1 = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)])), HEDLEY_STATIC_CAST(int64_t, scale));
    pgm= svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_1], INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(pgm, svld1_gather_s64offset_s64(pg, base_addr, index1), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      src_ = easysimd__m256i_to_private(src),
      mask_ = easysimd__m256i_to_private(mask),
      r_;
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      if ((mask_.i64[i] >> 63) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        int64_t dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.i64[i] = dst;
      }
      else {
        r_.i64[i] = src_.i64[i];
      }
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #if EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm256_mask_i32gather_epi64(src, base_addr, vindex, mask, scale) _mm256_mask_i32gather_epi64(src, HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, mask, scale)
  #else
    #define easysimd_mm256_mask_i32gather_epi64(src, base_addr, vindex, mask, scale) _mm256_mask_i32gather_epi64(src, HEDLEY_REINTERPRET_CAST(long long const*, base_addr), vindex, mask, scale)
  #endif
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_i32gather_epi64
  #define _mm256_mask_i32gather_epi64(src, base_addr, vindex, mask, scale) easysimd_mm256_mask_i32gather_epi64(src, HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_i64gather_epi64(const int64_t* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b64();
    svint64_t index = svmul_n_s64_z(pg, vindex.sve_i64, HEDLEY_STATIC_CAST(int64_t, scale));
    r.sve_i64 = svld1_gather_s64offset_s64(pg, base_addr, index);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex),
      r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int64_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i64[i] = dst;
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #if EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm_i64gather_epi64(base_addr, vindex, scale) _mm_i64gather_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, scale)
  #else
    #define easysimd_mm_i64gather_epi64(base_addr, vindex, scale) _mm_i64gather_epi64(HEDLEY_REINTERPRET_CAST(long long const*, base_addr), vindex, scale)
  #endif
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_i64gather_epi64
  #define _mm_i64gather_epi64(base_addr, vindex, scale) easysimd_mm_i64gather_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_mask_i64gather_epi64(easysimd__m128i src, const int64_t* base_addr, easysimd__m128i vindex, easysimd__m128i mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b64();
    svint64_t index = svmul_n_s64_z(pg, vindex.sve_i64, HEDLEY_STATIC_CAST(int64_t, scale));
    easysimd_svbool_t pgm= svcmplt_n_s64(pg, mask.sve_i64, INT64_C(0));
    r.sve_i64 = svsel_s64(pgm, svld1_gather_s64offset_s64(pg, base_addr, index), src.sve_i64);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex),
      src_ = easysimd__m128i_to_private(src),
      mask_ = easysimd__m128i_to_private(mask),
      r_ = easysimd__m128i_to_private(easysimd_mm_setzero_si128());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((mask_.i64[i] >> 63) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        int64_t dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.i64[i] = dst;
      }
      else {
        r_.i64[i] = src_.i64[i];
      }
    }

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #if EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm_mask_i64gather_epi64(src, base_addr, vindex, mask, scale) _mm_mask_i64gather_epi64(src, HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, mask, scale)
  #else
    #define easysimd_mm_mask_i64gather_epi64(src, base_addr, vindex, mask, scale) _mm_mask_i64gather_epi64(src, HEDLEY_REINTERPRET_CAST(long long const*, base_addr), vindex, mask, scale)
  #endif
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_i64gather_epi64
  #define _mm_mask_i64gather_epi64(src, base_addr, vindex, mask, scale) easysimd_mm_mask_i64gather_epi64(src, HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_i64gather_epi64(const int64_t* base_addr, easysimd__m256i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    svint64_t index0 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_0], scale);
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1_gather_s64offset_s64(pg, base_addr, index0);

    svint64_t index1 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_1], scale);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1_gather_s64offset_s64(pg, base_addr, index1);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex),
      r_ = easysimd__m256i_to_private(easysimd_mm256_setzero_si256());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      int64_t dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.i64[i] = dst;
    }
    return easysimd__m256i_from_private(r_);

  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #if EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm256_i64gather_epi64(base_addr, vindex, scale) _mm256_i64gather_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, scale)
  #else
    #define easysimd_mm256_i64gather_epi64(base_addr, vindex, scale) _mm256_i64gather_epi64(HEDLEY_REINTERPRET_CAST(long long const*, base_addr), vindex, scale)
  #endif
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_i64gather_epi64
  #define _mm256_i64gather_epi64(base_addr, vindex, scale) easysimd_mm256_i64gather_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mask_i64gather_epi64(easysimd__m256i src, const int64_t* base_addr, easysimd__m256i vindex, easysimd__m256i mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    svint64_t index0 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_0], HEDLEY_STATIC_CAST(int64_t, scale));
    easysimd_svbool_t pgm= svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_0], INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsel_s64(pgm, svld1_gather_s64offset_s64(pg, base_addr, index0), src.sve_i64[EASYSIMD_SV_INDEX_0]);

    svint64_t index1 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_1], HEDLEY_STATIC_CAST(int64_t, scale));
    pgm= svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_1], INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsel_s64(pgm, svld1_gather_s64offset_s64(pg, base_addr, index1), src.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex),
      src_ = easysimd__m256i_to_private(src),
      mask_ = easysimd__m256i_to_private(mask),
      r_ = easysimd__m256i_to_private(easysimd_mm256_setzero_si256());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((mask_.i64[i] >> 63) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        int64_t dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.i64[i] = dst;
      }
      else {
        r_.i64[i] = src_.i64[i];
      }
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #if EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm256_mask_i64gather_epi64(src, base_addr, vindex, mask, scale) _mm256_mask_i64gather_epi64(src, HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, mask, scale)
  #else
    #define easysimd_mm256_mask_i64gather_epi64(src, base_addr, vindex, mask, scale) _mm256_mask_i64gather_epi64(src, HEDLEY_REINTERPRET_CAST(long long const*, base_addr), vindex, mask, scale)
  #endif
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_i64gather_epi64
  #define _mm256_mask_i64gather_epi64(src, base_addr, vindex, mask, scale) easysimd_mm256_mask_i64gather_epi64(src, HEDLEY_REINTERPRET_CAST(int64_t const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_i32gather_ps(const easysimd_float32* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b32();
    svint32_t index = svmul_n_s32_z(pg, vindex.sve_i32, scale);
    r.sve_f32 = svld1_gather_s32offset_f32(pg, base_addr, index);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    easysimd__m128_private
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      easysimd_float32 dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f32[i] = dst;
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_i32gather_ps(base_addr, vindex, scale) _mm_i32gather_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float const*, easysimd_float32 const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_i32gather_ps
  #define _mm_i32gather_ps(base_addr, vindex, scale) easysimd_mm_i32gather_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, float const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_i32gather_ps(easysimd__m128 src, const easysimd_float32* base_addr, easysimd__m128i vindex, easysimd__m128 mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b32();
    sveint32_t index = svmul_n_s32_z(pg, vindex.sve_i32, scale);
    easysimd_svbool_t pgm= svcmplt_n_s32(pg, mask.sve_i32, INT32_C(0));
    r.sve_f32 = svsel_f32(pgm, svld1_gather_s32offset_f32(pg, base_addr, index), src.sve_f32);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      mask_ = easysimd__m128_to_private(mask),
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      if ((mask_.i32[i] >> 31) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        easysimd_float32 dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.f32[i] = dst;
      }
      else {
        r_.f32[i] = src_.f32[i];
      }
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_mask_i32gather_ps(src, base_addr, vindex, mask, scale) _mm_mask_i32gather_ps(src, EASYSIMD_CHECKED_REINTERPRET_CAST(float const*, easysimd_float32 const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_i32gather_ps
  #define _mm_mask_i32gather_ps(src, base_addr, vindex, mask, scale) easysimd_mm_mask_i32gather_ps(src, EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, float const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_i32gather_ps(const easysimd_float32* base_addr, easysimd__m256i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    sveint32_t index0 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_0], scale);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_gather_s32offset_f32(pg, base_addr, index0);

    sveint32_t index1 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_1], scale);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_gather_s32offset_f32(pg, base_addr, index1);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.f32[ 0] = *((float *)((uint8_t *)base_addr + vindex.i32[ 0] * scale));
    r.f32[ 1] = *((float *)((uint8_t *)base_addr + vindex.i32[ 1] * scale));
    r.f32[ 2] = *((float *)((uint8_t *)base_addr + vindex.i32[ 2] * scale));
    r.f32[ 3] = *((float *)((uint8_t *)base_addr + vindex.i32[ 3] * scale));
    r.f32[ 4] = *((float *)((uint8_t *)base_addr + vindex.i32[ 4] * scale));
    r.f32[ 5] = *((float *)((uint8_t *)base_addr + vindex.i32[ 5] * scale));
    r.f32[ 6] = *((float *)((uint8_t *)base_addr + vindex.i32[ 6] * scale));
    r.f32[ 7] = *((float *)((uint8_t *)base_addr + vindex.i32[ 7] * scale));
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex);
    easysimd__m256_private
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      easysimd_float32 dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f32[i] = dst;
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_i32gather_ps(base_addr, vindex, scale) _mm256_i32gather_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float const*, easysimd_float32 const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_i32gather_ps
  #define _mm256_i32gather_ps(base_addr, vindex, scale) easysimd_mm256_i32gather_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, float const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mask_i32gather_ps(easysimd__m256 src, const easysimd_float32* base_addr, easysimd__m256i vindex, easysimd__m256 mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    sveint32_t index0 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_0], scale);
    easysimd_svbool_t pgm= svcmplt_n_s32(pg, mask.sve_i32[EASYSIMD_SV_INDEX_0], INT32_C(0));
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(pgm, svld1_gather_s32offset_f32(pg, base_addr, index0), src.sve_f32[EASYSIMD_SV_INDEX_0]);

    sveint32_t index1 = svmul_n_s32_z(pg, vindex.sve_i32[EASYSIMD_SV_INDEX_1], scale);
    pgm= svcmplt_n_s32(pg, mask.sve_i32[EASYSIMD_SV_INDEX_1], INT32_C(0));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(pgm, svld1_gather_s32offset_f32(pg, base_addr, index1), src.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex);
    easysimd__m256_private
      src_ = easysimd__m256_to_private(src),
      mask_ = easysimd__m256_to_private(mask),
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      if ((mask_.i32[i] >> 31) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        easysimd_float32 dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
       r_.f32[i] = dst;
      }
      else {
        r_.f32[i] = src_.f32[i];
      }
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale) _mm256_mask_i32gather_ps(src, EASYSIMD_CHECKED_REINTERPRET_CAST(float const*, easysimd_float32 const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_i32gather_ps
  #define _mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale) easysimd_mm256_mask_i32gather_ps(src, EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, float const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_i64gather_ps(const easysimd_float32* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b32();
    sveint32_t index = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)), 0, 0);
    r.sve_f32 = svmul_f32_z(pg, svld1_gather_s32offset_f32(pg, base_addr, index), svdupq_n_f32(1.0, 1.0, 0.0, 0.0));
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    easysimd__m128_private
      r_ = easysimd__m128_to_private(easysimd_mm_setzero_ps());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      easysimd_float32 dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f32[i] = dst;
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_i64gather_ps(base_addr, vindex, scale) _mm_i64gather_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float const*, easysimd_float32 const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_i64gather_ps
  #define _mm_i64gather_ps(base_addr, vindex, scale) easysimd_mm_i64gather_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, float const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mask_i64gather_ps(easysimd__m128 src, const easysimd_float32* base_addr, easysimd__m128i vindex, easysimd__m128 mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b32();
    sveint32_t index = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)), 0, 0);
    easysimd_svbool_t pgm= svcmplt_n_s32(pg, mask.sve_i32, INT32_C(0));
    r.sve_f32 = svsel_f32(pgm, svld1_gather_s32offset_f32(pg, base_addr, index), src.sve_f32);
    r.sve_f32 = svmul_f32_z(pg, r.sve_f32, svdupq_n_f32(1.0, 1.0, 0.0, 0.0));
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      mask_ = easysimd__m128_to_private(mask),
      r_ = easysimd__m128_to_private(easysimd_mm_setzero_ps());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((mask_.i32[i] >> 31) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
       easysimd_float32 dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.f32[i] = dst;
      }
      else {
        r_.f32[i] = src_.f32[i];
      }
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_mask_i64gather_ps(src, base_addr, vindex, mask, scale) _mm_mask_i64gather_ps(src, EASYSIMD_CHECKED_REINTERPRET_CAST(float const*, float32_t const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_i64gather_ps
  #define _mm_mask_i64gather_ps(src, base_addr, vindex, mask, scale) easysimd_mm_mask_i64gather_ps(src, EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, float const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_i64gather_ps(const easysimd_float32* base_addr, easysimd__m256i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b32();
    sveint32_t index = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[2])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[3])* HEDLEY_STATIC_CAST(size_t, scale)));
    r.sve_f32 = svld1_gather_s32offset_f32(pg, base_addr, index);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex);
    easysimd__m128_private
      r_ = easysimd__m128_to_private(easysimd_mm_setzero_ps());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      easysimd_float32 dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f32[i] = dst;
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_i64gather_ps(base_addr, vindex, scale) _mm256_i64gather_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float const*, easysimd_float32 const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_i64gather_ps
  #define _mm256_i64gather_ps(base_addr, vindex, scale) easysimd_mm256_i64gather_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, float const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_mask_i64gather_ps(easysimd__m128 src, const easysimd_float32* base_addr, easysimd__m256i vindex, easysimd__m128 mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b32();
    sveint32_t index = svdupq_n_s32((HEDLEY_STATIC_CAST(size_t, vindex.i64[0])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[1])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[2])* HEDLEY_STATIC_CAST(size_t, scale)),
                                    (HEDLEY_STATIC_CAST(size_t, vindex.i64[3])* HEDLEY_STATIC_CAST(size_t, scale)));
    easysimd_svbool_t pgm= svcmplt_n_s32(pg, mask.sve_i32, INT32_C(0));
    r.sve_f32 = svsel_f32(pgm, svld1_gather_s32offset_f32(pg, base_addr, index), src.sve_f32);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex);
    easysimd__m128_private
      src_ = easysimd__m128_to_private(src),
      mask_ = easysimd__m128_to_private(mask),
      r_ = easysimd__m128_to_private(easysimd_mm_setzero_ps());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((mask_.i32[i] >> 31) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        easysimd_float32 dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.f32[i] = dst;
      }
      else {
        r_.f32[i] = src_.f32[i];
      }
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_mask_i64gather_ps(src, base_addr, vindex, mask, scale) _mm256_mask_i64gather_ps(src, EASYSIMD_CHECKED_REINTERPRET_CAST(float const*, easysimd_float32 const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_i64gather_ps
  #define _mm256_mask_i64gather_ps(src, base_addr, vindex, mask, scale) easysimd_mm256_mask_i64gather_ps(src, EASYSIMD_CHECKED_REINTERPRET_CAST(easysimd_float32 const*, float const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_i32gather_pd(const easysimd_float64* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    svbool_t pg = svptrue_b64();
    svint64_t index = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_0])), HEDLEY_STATIC_CAST(int64_t, scale));
    r.sve_f64 = svld1_gather_s64offset_f64(pg, base_addr, index);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    easysimd__m128d_private
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      easysimd_float64 dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f64[i] = dst;
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_i32gather_pd(base_addr, vindex, scale) _mm_i32gather_pd(HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_i32gather_pd
  #define _mm_i32gather_pd(base_addr, vindex, scale) easysimd_mm_i32gather_pd(HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_i32gather_pd(easysimd__m128d src, const easysimd_float64* base_addr, easysimd__m128i vindex, easysimd__m128d mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    svbool_t pg = svptrue_b64();
    svint64_t index = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_0])), HEDLEY_STATIC_CAST(int64_t, scale));
    easysimd_svbool_t pgm= svcmplt_n_s64(pg, mask.sve_i64, INT64_C(0));
    r.sve_f64 = svsel_f64(pgm, svld1_gather_s64offset_f64(pg, base_addr, index), src.sve_f64);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      mask_ = easysimd__m128d_to_private(mask),
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      if ((mask_.i64[i] >> 63) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        easysimd_float64 dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.f64[i] = dst;
      }
      else {
        r_.f64[i] = src_.f64[i];
      }
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_mask_i32gather_pd(src, base_addr, vindex, mask, scale) _mm_mask_i32gather_pd(src, HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_i32gather_pd
  #define _mm_mask_i32gather_pd(src, base_addr, vindex, mask, scale) easysimd_mm_mask_i32gather_pd(src, HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_i32gather_pd(const easysimd_float64* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    svint64_t index0 = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)])), HEDLEY_STATIC_CAST(int64_t, scale));
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_gather_s64offset_f64(pg, base_addr, index0);

    svint64_t index1 = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)])), HEDLEY_STATIC_CAST(int64_t, scale));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_gather_s64offset_f64(pg, base_addr, index1);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    easysimd__m256d_private
      r_;
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      easysimd_float64 dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f64[i] = dst;
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_i32gather_pd(base_addr, vindex, scale) _mm256_i32gather_pd(HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_i32gather_pd
  #define _mm256_i32gather_pd(base_addr, vindex, scale) easysimd_mm256_i32gather_pd(HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_i32gather_pd(easysimd__m256d src, const easysimd_float64* base_addr, easysimd__m128i vindex, easysimd__m256d mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    svint64_t index0 = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)])), HEDLEY_STATIC_CAST(int64_t, scale));
    easysimd_svbool_t pgm= svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_0], INT64_C(0));
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(pgm, svld1_gather_s64offset_f64(pg, base_addr, index0), src.sve_f64[EASYSIMD_SV_INDEX_0]);

    svint64_t index1 = svmul_n_s64_z(pg, svld1sw_s64(pg, &(vindex.i32[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)])), HEDLEY_STATIC_CAST(int64_t, scale));
    pgm= svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_1], INT64_C(0));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(pgm, svld1_gather_s64offset_f64(pg, base_addr, index1), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      src_ = easysimd__m256d_to_private(src),
      mask_ = easysimd__m256d_to_private(mask),
      r_;
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i32) / sizeof(vindex_.i32[0])) ; i++) {
      if ((mask_.i64[i] >> 63) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i32[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        easysimd_float64 dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.f64[i] = dst;
      }
      else {
        r_.f64[i] = src_.f64[i];
      }
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_mask_i32gather_pd(src, base_addr, vindex, mask, scale) _mm256_mask_i32gather_pd(src, HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_i32gather_pd
  #define _mm256_mask_i32gather_pd(src, base_addr, vindex, mask, scale) easysimd_mm256_mask_i32gather_pd(src, HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_i64gather_pd(const easysimd_float64* base_addr, easysimd__m128i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    svbool_t pg = svptrue_b64();
    svint64_t index = svmul_n_s64_z(pg, vindex.sve_i64, HEDLEY_STATIC_CAST(int64_t, scale));
    r.sve_f64 = svld1_gather_s64offset_f64(pg, base_addr, index);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    easysimd__m128d_private
      r_ = easysimd__m128d_to_private(easysimd_mm_setzero_pd());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      easysimd_float64 dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f64[i] = dst;
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_i64gather_pd(base_addr, vindex, scale) _mm_i64gather_pd(HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_i64gather_pd
  #define _mm_i64gather_pd(base_addr, vindex, scale) easysimd_mm_i64gather_pd(HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_mask_i64gather_pd(easysimd__m128d src, const easysimd_float64* base_addr, easysimd__m128i vindex, easysimd__m128d mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    svbool_t pg = svptrue_b64();
    svint64_t index = svmul_n_s64_z(pg, vindex.sve_i64, HEDLEY_STATIC_CAST(int64_t, scale));
    easysimd_svbool_t pgm= svcmplt_n_s64(pg, mask.sve_i64, INT64_C(0));
    r.sve_f64 = svsel_f64(pgm, svld1_gather_s64offset_f64(pg, base_addr, index), src.sve_f64);
    return r;
  #else
    easysimd__m128i_private
      vindex_ = easysimd__m128i_to_private(vindex);
    easysimd__m128d_private
      src_ = easysimd__m128d_to_private(src),
      mask_ = easysimd__m128d_to_private(mask),
      r_ = easysimd__m128d_to_private(easysimd_mm_setzero_pd());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((mask_.i64[i] >> 63) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        easysimd_float64 dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.f64[i] = dst;
      }
      else {
        r_.f64[i] = src_.f64[i];
      }
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_mask_i64gather_pd(src, base_addr, vindex, mask, scale) _mm_mask_i64gather_pd(src, HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_mask_i64gather_pd
  #define _mm_mask_i64gather_pd(src, base_addr, vindex, mask, scale) easysimd_mm_mask_i64gather_pd(src, HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_i64gather_pd(const easysimd_float64* base_addr, easysimd__m256i vindex, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    svint64_t index0 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_0], HEDLEY_STATIC_CAST(int64_t, scale));
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_gather_s64offset_f64(pg, base_addr, index0);

    svint64_t index1 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_1], HEDLEY_STATIC_CAST(int64_t, scale));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_gather_s64offset_f64(pg, base_addr, index1);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex);
    easysimd__m256d_private
      r_ = easysimd__m256d_to_private(easysimd_mm256_setzero_pd());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      const uint8_t* src = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
      easysimd_float64 dst;
      easysimd_memcpy(&dst, src, sizeof(dst));
      r_.f64[i] = dst;
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_i64gather_pd(base_addr, vindex, scale) _mm256_i64gather_pd(HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_i64gather_pd
  #define _mm256_i64gather_pd(base_addr, vindex, scale) easysimd_mm256_i64gather_pd(HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mask_i64gather_pd(easysimd__m256d src, const easysimd_float64* base_addr, easysimd__m256i vindex, easysimd__m256d mask, const int32_t scale)
    EASYSIMD_REQUIRE_CONSTANT(scale)
    HEDLEY_REQUIRE_MSG((scale && scale <= 8 && !(scale & (scale - 1))), "`scale' must be a power of two less than or equal to 8") {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    svint64_t index0 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_0], HEDLEY_STATIC_CAST(int64_t, scale));
    easysimd_svbool_t pgm= svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_0], INT64_C(0));
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(pgm, svld1_gather_s64offset_f64(pg, base_addr, index0), src.sve_f64[EASYSIMD_SV_INDEX_0]);

    svint64_t index1 = svmul_n_s64_z(pg, vindex.sve_i64[EASYSIMD_SV_INDEX_1], HEDLEY_STATIC_CAST(int64_t, scale));
    pgm= svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_1], INT64_C(0));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(pgm, svld1_gather_s64offset_f64(pg, base_addr, index1), src.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      vindex_ = easysimd__m256i_to_private(vindex);
    easysimd__m256d_private
      src_ = easysimd__m256d_to_private(src),
      mask_ = easysimd__m256d_to_private(mask),
      r_ = easysimd__m256d_to_private(easysimd_mm256_setzero_pd());
    const uint8_t* addr = HEDLEY_REINTERPRET_CAST(const uint8_t*, base_addr);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(vindex_.i64) / sizeof(vindex_.i64[0])) ; i++) {
      if ((mask_.i64[i] >> 63) & 1) {
        const uint8_t* src1 = addr + (HEDLEY_STATIC_CAST(size_t, vindex_.i64[i]) * HEDLEY_STATIC_CAST(size_t, scale));
        easysimd_float64 dst;
        easysimd_memcpy(&dst, src1, sizeof(dst));
        r_.f64[i] = dst;
      }
      else {
        r_.f64[i] = src_.f64[i];
      }
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_mask_i64gather_pd(src, base_addr, vindex, mask, scale) _mm256_mask_i64gather_pd(src, HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, mask, scale)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mask_i64gather_pd
  #define _mm256_mask_i64gather_pd(src, base_addr, vindex, mask, scale) easysimd_mm256_mask_i64gather_pd(src, HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, base_addr), vindex, mask, scale)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_inserti128_si256(easysimd__m256i a, easysimd__m128i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  easysimd__m256i res;
  uint32x4_t vmask = vceqq_s32(vdupq_n_s32(imm8 & 1), vdupq_n_s32(0));
  res.m128i[0].neon_i32 = vbslq_s32(vmask, b.neon_i32, a.m128i[0].neon_i32);
  res.m128i[1].neon_i32 = vbslq_s32(vmask, a.m128i[1].neon_i32, b.neon_i32);
  return res;
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  a.sve_i32[imm8 & 1] = b.sve_i32;
  return a;
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
  easysimd__m128i_private b_ = easysimd__m128i_to_private(b);
  a_.m128i_private[ imm8 & 1 ] = b_;
  return easysimd__m256i_from_private(a_);
#endif

}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_inserti128_si256(a, b, imm8) _mm256_inserti128_si256(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_inserti128_si256
  #define _mm256_inserti128_si256(a, b, imm8) easysimd_mm256_inserti128_si256(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_madd_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_madd_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svadd_s32_z(pg, svmullb_s32(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), svmullt_s32(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_madd_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_madd_epi16(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && defined(EASYSIMD_CONVERT_VECTOR_) && HEDLEY_HAS_BUILTIN(__builtin_shufflevector)
      EASYSIMD_ALIGN_TO_32 int32_t product EASYSIMD_VECTOR(64);
      EASYSIMD_ALIGN_TO_32 int32_t a32x16 EASYSIMD_VECTOR(64);
      EASYSIMD_ALIGN_TO_32 int32_t b32x16 EASYSIMD_VECTOR(64);
      EASYSIMD_ALIGN_TO_32 int32_t even EASYSIMD_VECTOR(32);
      EASYSIMD_ALIGN_TO_32 int32_t odd EASYSIMD_VECTOR(32);

      EASYSIMD_CONVERT_VECTOR_(a32x16, a_.i16);
      EASYSIMD_CONVERT_VECTOR_(b32x16, b_.i16);
      product = a32x16 * b32x16;

      even = __builtin_shufflevector(product, product, 0, 2, 4, 6, 8, 10, 12, 14);
      odd  = __builtin_shufflevector(product, product, 1, 3, 5, 7, 9, 11, 13, 15);

      r_.i32 = even + odd;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_) / sizeof(r_.i16[0])) ; i += 2) {
        r_.i32[i / 2] = (a_.i16[i] * b_.i16[i]) + (a_.i16[i + 1] * b_.i16[i + 1]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_madd_epi16
  #define _mm256_madd_epi16(a, b) easysimd_mm256_madd_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maddubs_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_maddubs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd__m128i tmp;
    svbool_t pg = svptrue_b16();
    tmp.sve_u8 = svuzp1_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], svdup_n_u8(0));
    svint16_t sva1_even = svld1ub_s16(pg, &(tmp.u8[0]));
    tmp.sve_u8 = svuzp1_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], svdup_n_u8(0));
    svint16_t sva2_even = svld1ub_s16(pg, &(tmp.u8[0]));

    tmp.sve_u8 = svuzp2_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], svdup_n_u8(0));
    svint16_t sva1_odd = svld1ub_s16(pg, &(tmp.u8[0]));
    tmp.sve_u8 = svuzp2_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], svdup_n_u8(0));
    svint16_t sva2_odd = svld1ub_s16(pg, &(tmp.u8[0]));

    tmp.sve_i8 = svuzp1_s8(b.sve_i8[EASYSIMD_SV_INDEX_0], svdup_n_s8(0));
    svint16_t svb1_even = svld1sb_s16(pg, &(tmp.i8[0]));
    tmp.sve_i8 = svuzp1_s8(b.sve_i8[EASYSIMD_SV_INDEX_1], svdup_n_s8(0));
    svint16_t svb2_even = svld1sb_s16(pg, &(tmp.i8[0]));

    tmp.sve_i8 = svuzp2_s8(b.sve_i8[EASYSIMD_SV_INDEX_0], svdup_n_s8(0));
    svint16_t svb1_odd = svld1sb_s16(pg, &(tmp.i8[0]));
    tmp.sve_i8 = svuzp2_s8(b.sve_i8[EASYSIMD_SV_INDEX_1], svdup_n_s8(0));
    svint16_t svb2_odd = svld1sb_s16(pg, &(tmp.i8[0]));

    r.sve_i16[EASYSIMD_SV_INDEX_0] = svqadd_s16_z(pg, svmul_s16_z(pg, sva1_even, svb1_even), svmul_s16_z(pg, sva1_odd, svb1_odd));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svqadd_s16_z(pg, svmul_s16_z(pg, sva2_even, svb2_even), svmul_s16_z(pg, sva2_odd, svb2_odd));

    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_maddubs_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_maddubs_epi16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        const int idx = HEDLEY_STATIC_CAST(int, i) << 1;
        int32_t ts =
          (HEDLEY_STATIC_CAST(int16_t, a_.u8[  idx  ]) * HEDLEY_STATIC_CAST(int16_t, b_.i8[  idx  ])) +
          (HEDLEY_STATIC_CAST(int16_t, a_.u8[idx + 1]) * HEDLEY_STATIC_CAST(int16_t, b_.i8[idx + 1]));
        r_.i16[i] = (ts > INT16_MIN) ? ((ts < INT16_MAX) ? HEDLEY_STATIC_CAST(int16_t, ts) : INT16_MAX) : INT16_MIN;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maddubs_epi16
  #define _mm256_maddubs_epi16(a, b) easysimd_mm256_maddubs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskload_epi32 (const int32_t mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm_maskload_epi32(mem_addr, mask);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b32();
    svbool_t pgm = svcmplt_n_s32(pg, mask.sve_i32, INT32_C(0));
    r.sve_i32 = svld1_s32(pgm, mem_addr);
    return r;
  #else
    easysimd__m128i_private
      mem_ = easysimd__m128i_to_private(easysimd_x_mm_loadu_epi32(mem_addr)),
      r_,
      mask_ = easysimd__m128i_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vandq_s32(mem_.neon_i32, vshrq_n_s32(mask_.neon_i32, 31));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = mem_.i32[i] & (mask_.i32[i] >> 31);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskload_epi32
  #define _mm_maskload_epi32(mem_addr, mask) easysimd_mm_maskload_epi32(HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr), mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskload_epi32 (const int32_t mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m256i mask) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_maskload_epi32(mem_addr, mask);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svint32_t zero = svdup_n_s32(0);
    svbool_t pg = svptrue_b32();
    svbool_t pgm = svcmplt_s32(pg, mask.sve_i32[EASYSIMD_SV_INDEX_0], zero);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(pgm, &(mem_addr[0]) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS / 32));

    pgm = svcmplt_s32(pg, mask.sve_i32[EASYSIMD_SV_INDEX_1], zero);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(pgm, &(mem_addr[0]) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS / 32));

    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i ret;
    int32x4_t vecZero = vdupq_n_s32(0);
    easysimd__m128i flag;

    flag.neon_u32 = vcltq_s32(mask.m128i[0].neon_i32, vecZero);
    ret.m128i[0].neon_i32 = vandq_s32(flag.neon_i32, vld1q_s32(mem_addr));

    flag.neon_u32 = vcltq_s32(mask.m128i[1].neon_i32, vecZero);
    ret.m128i[1].neon_i32 = vandq_s32(flag.neon_i32, vld1q_s32(mem_addr + 4));

    return ret;
  #else
    easysimd__m256i_private
      mask_ = easysimd__m256i_to_private(mask),
      r_ = easysimd__m256i_to_private(easysimd_x_mm256_loadu_epi32(mem_addr));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] &= mask_.i32[i] >> 31;
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskload_epi32
  #define _mm256_maskload_epi32(mem_addr, mask) easysimd_mm256_maskload_epi32(HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr), mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_maskload_epi64 (const int64_t mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm_maskload_epi64(HEDLEY_REINTERPRET_CAST(const long long *, mem_addr), mask);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b64();
    svbool_t pgm = svcmplt_n_s64(pg, mask.sve_i64, INT64_C(0));
    r.sve_i64 = svld1_s64(pgm, mem_addr);
    return r;
  #else
    easysimd__m128i_private
      mem_ = easysimd__m128i_to_private(easysimd_x_mm_loadu_epi64((mem_addr))),
      r_,
      mask_ = easysimd__m128i_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vandq_s64(mem_.neon_i64, vshrq_n_s64(mask_.neon_i64, 63));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = mem_.i64[i] & (mask_.i64[i] >> 63);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskload_epi64
  #define _mm_maskload_epi64(mem_addr, mask) easysimd_mm_maskload_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, mem_addr), mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_maskload_epi64 (const int64_t mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m256i mask) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_maskload_epi64(HEDLEY_REINTERPRET_CAST(const long long *, mem_addr), mask);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    easysimd_svbool_t pgm = svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_0], INT64_C(0));
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1_s64(pgm, &(mem_addr[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)]));
    pgm = svcmplt_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_1], svdup_n_s64(0));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1_s64(pgm, &(mem_addr[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)]));
    return r;
  #else
    easysimd__m256i_private
      mask_ = easysimd__m256i_to_private(mask),
      r_ = easysimd__m256i_to_private(easysimd_x_mm256_loadu_epi64((mem_addr)));

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] &= mask_.i64[i] >> 63;
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskload_epi64
  #define _mm256_maskload_epi64(mem_addr, mask) easysimd_mm256_maskload_epi64(HEDLEY_REINTERPRET_CAST(int64_t const*, mem_addr), mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_maskstore_epi32 (int32_t mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m128i mask, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    _mm_maskstore_epi32(mem_addr, mask, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b32();
    easysimd_svbool_t pgm = svcmplt_n_s32(pg, mask.sve_i32, INT32_C(0));
    svst1_s32(pgm, mem_addr, a.sve_i32);
  #else
    easysimd__m128i_private mask_ = easysimd__m128i_to_private(mask);
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      if (mask_.u32[i] & (UINT32_C(1) << 31))
        mem_addr[i] = a_.i32[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskstore_epi32
  #define _mm_maskstore_epi32(mem_addr, mask, a) easysimd_mm_maskstore_epi32(HEDLEY_REINTERPRET_CAST(int32_t *, mem_addr), mask, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_maskstore_epi32 (int32_t mem_addr[HEDLEY_ARRAY_PARAM(8)], easysimd__m256i mask, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    _mm256_maskstore_epi32(mem_addr, mask, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b32();
    easysimd_svbool_t pgm = svcmplt_n_s32(pg, mask.sve_i32[EASYSIMD_SV_INDEX_0], INT32_C(0));
    svst1_s32(pgm, &(mem_addr[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5)]), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    pgm = svcmplt_n_s32(pg, mask.sve_i32[EASYSIMD_SV_INDEX_1], INT32_C(0));
    svst1_s32(pgm, &(mem_addr[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5)]), a.sve_i32[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd__m256i_private mask_ = easysimd__m256i_to_private(mask);
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i32) / sizeof(a_.i32[0])) ; i++) {
      if (mask_.u32[i] & (UINT32_C(1) << 31))
        mem_addr[i] = a_.i32[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskstore_epi32
  #define _mm256_maskstore_epi32(mem_addr, mask, a) easysimd_mm256_maskstore_epi32(HEDLEY_REINTERPRET_CAST(int32_t *, mem_addr), mask, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_maskstore_epi64 (int64_t mem_addr[HEDLEY_ARRAY_PARAM(2)], easysimd__m128i mask, easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    _mm_maskstore_epi64(HEDLEY_REINTERPRET_CAST(long long *, mem_addr), mask, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b64();
    easysimd_svbool_t pgm = svcmplt_n_s64(pg, mask.sve_i64, INT64_C(0));
    svst1_s64(pgm, mem_addr, a.sve_i64);
  #else
    easysimd__m128i_private mask_ = easysimd__m128i_to_private(mask);
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      if (mask_.u64[i] >> 63)
        mem_addr[i] = a_.i64[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskstore_epi64
  #define _mm_maskstore_epi64(mem_addr, mask, a) easysimd_mm_maskstore_epi64(HEDLEY_REINTERPRET_CAST(int64_t *, mem_addr), mask, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_maskstore_epi64 (int64_t mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m256i mask, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    _mm256_maskstore_epi64(HEDLEY_REINTERPRET_CAST(long long *, mem_addr), mask, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b64();
    easysimd_svbool_t pgm = svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_0], INT64_C(0));
    svst1_s64(pgm, &(mem_addr[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6)]), a.sve_i64[EASYSIMD_SV_INDEX_0]);
    pgm = svcmplt_n_s64(pg, mask.sve_i64[EASYSIMD_SV_INDEX_1], INT64_C(0));
    svst1_s64(pgm, &(mem_addr[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6)]), a.sve_i64[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd__m256i_private mask_ = easysimd__m256i_to_private(mask);
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i64) / sizeof(a_.i64[0])) ; i++) {
      if (mask_.u64[i] & (UINT64_C(1) << 63))
        mem_addr[i] = a_.i64[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskstore_epi64
  #define _mm256_maskstore_epi64(mem_addr, mask, a) easysimd_mm256_maskstore_epi64(HEDLEY_REINTERPRET_CAST(int64_t *, mem_addr), mask, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_max_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE) && !defined(__PGI)
    return _mm256_max_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svmax_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svmax_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_max_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_max_epi8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[i] > b_.i8[i] ? a_.i8[i] : b_.i8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_max_epi8
  #define _mm256_max_epi8(a, b) easysimd_mm256_max_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_max_epu8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_max_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svmax_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svmax_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_max_epu8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_max_epu8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = (a_.u8[i] > b_.u8[i]) ? a_.u8[i] : b_.u8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_max_epu8
  #define _mm256_max_epu8(a, b) easysimd_mm256_max_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_max_epu16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_max_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svmax_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svmax_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_max_epu16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_max_epu16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = (a_.u16[i] > b_.u16[i]) ? a_.u16[i] : b_.u16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_max_epu16
  #define _mm256_max_epu16(a, b) easysimd_mm256_max_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_max_epu32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_max_epu32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svmax_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svmax_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_max_epu32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_max_epu32(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = (a_.u32[i] > b_.u32[i]) ? a_.u32[i] : b_.u32[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_max_epu32
  #define _mm256_max_epu32(a, b) easysimd_mm256_max_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_max_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_max_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svmax_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svmax_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_max_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_max_epi16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? a_.i16[i] : b_.i16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_max_epi16
  #define _mm256_max_epi16(a, b) easysimd_mm256_max_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_max_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_max_epi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vmaxq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vmaxq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svmax_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svmax_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_max_epi32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_max_epi32(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] > b_.i32[i] ? a_.i32[i] : b_.i32[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_max_epi32
  #define _mm256_max_epi32(a, b) easysimd_mm256_max_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_min_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE) && !defined(__PGI)
    return _mm256_min_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svmin_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svmin_s8_x(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_min_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_min_epi8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[i] < b_.i8[i] ? a_.i8[i] : b_.i8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_epi8
  #define _mm256_min_epi8(a, b) easysimd_mm256_min_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_min_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_min_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svmin_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svmin_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_min_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_min_epi16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? a_.i16[i] : b_.i16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_epi16
  #define _mm256_min_epi16(a, b) easysimd_mm256_min_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_min_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_min_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svmin_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svmin_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_min_epi32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_min_epi32(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] < b_.i32[i] ? a_.i32[i] : b_.i32[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_epi32
  #define _mm256_min_epi32(a, b) easysimd_mm256_min_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_min_epu8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_min_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svmin_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svmin_u8_x(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_min_epu8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_min_epu8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = (a_.u8[i] < b_.u8[i]) ? a_.u8[i] : b_.u8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_epu8
  #define _mm256_min_epu8(a, b) easysimd_mm256_min_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_min_epu16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_min_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svmin_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svmin_u16_x(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_min_epu16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_min_epu16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = (a_.u16[i] < b_.u16[i]) ? a_.u16[i] : b_.u16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_epu16
  #define _mm256_min_epu16(a, b) easysimd_mm256_min_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_min_epu32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_min_epu32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svmin_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svmin_u32_x(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_min_epu32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_min_epu32(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = (a_.u32[i] < b_.u32[i]) ? a_.u32[i] : b_.u32[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_epu32
  #define _mm256_min_epu32(a, b) easysimd_mm256_min_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm256_movemask_epi8 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_movemask_epi8(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    int res;
    __asm__ __volatile__ (
        "ushr %[a0].16b, %[a0].16b, #7          \n\t"
        "ushr %[a1].16b, %[a1].16b, #7          \n\t"
        "usra %[a0].8h, %[a0].8h, #7            \n\t"
        "usra %[a1].8h, %[a1].8h, #7            \n\t"
        "usra %[a0].4s, %[a0].4s, #14           \n\t"
        "usra %[a1].4s, %[a1].4s, #14           \n\t"
        "usra %[a0].2d, %[a0].2d, #28           \n\t"
        "usra %[a1].2d, %[a1].2d, #28           \n\t"
        "ins %[a0].b[1], %[a0].b[8]             \n\t"
        "ins %[a0].b[2], %[a1].b[0]             \n\t"
        "ins %[a0].b[3], %[a1].b[8]             \n\t"
        "umov %w[r], %[a0].s[0]"
        :[r]"=r"(res), [a0]"+w"(a.m128i[0].neon_u8), [a1]"+w"(a.m128i[1].neon_u8)
        :
        :
    );
    return res;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int32_t ret = 0;
    svbool_t pg = svptrue_b8();
    EASYSIMD_B8_TO_MASK(ret, svcmplt_n_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], 0), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B8_TO_MASK(ret, svcmplt_n_s8(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], 0), EASYSIMD_SV_INDEX_1);
    return ret;
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    uint32_t r = 0;

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      for (size_t i = 0 ; i < (sizeof(a_.m128i) / sizeof(a_.m128i[0])) ; i++) {
        r |= HEDLEY_STATIC_CAST(uint32_t,easysimd_mm_movemask_epi8(a_.m128i[i])) << (16 * i);
      }
    #else
      r = 0;
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.u8) / sizeof(a_.u8[0])) ; i++) {
        r |= HEDLEY_STATIC_CAST(uint32_t, (a_.u8[31 - i] >> 7)) << (31 - i);
      }
    #endif

    return HEDLEY_STATIC_CAST(int32_t, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_movemask_epi8
  #define _mm256_movemask_epi8(a) easysimd_mm256_movemask_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mpsadbw_epu8 (easysimd__m256i a, easysimd__m256i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  svuint8_t sva, svb;
  sveuint8_t svarr[8];
  svuint8_t svaindex = svdupq_n_u8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
  svuint8_t svaindex0 = svadd_u8_x(svptrue_b8(), svaindex, svdup_n_u8(imm8 & 4));
  svuint8_t svaindex1 = svadd_u8_x(svptrue_b8(), svaindex, svdup_n_u8((imm8 >> 3) & 4));
  svuint8_t svbindex0 = svdup_n_u8((imm8 & 3) << 2);
  svuint8_t svbindex1 = svdup_n_u8(((imm8 >> 3) & 3) << 2);
  for(int i = 0; i < 4; i++){
    sva = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], svaindex0);
    svb = svtbl_u8(b.sve_u8[EASYSIMD_SV_INDEX_0], svbindex0);
    svarr[i] = svabd_u8_x(svptrue_b8(), sva, svb);
    sva = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], svaindex1);
    svb = svtbl_u8(b.sve_u8[EASYSIMD_SV_INDEX_1], svbindex1);
    svarr[i + 4] = svabd_u8_x(svptrue_b8(), sva, svb);
    svaindex0 = svadd_u8_x(svptrue_b8(), svaindex0, svdup_n_u8(1));
    svbindex0 = svadd_u8_x(svptrue_b8(), svbindex0, svdup_n_u8(1));
    svaindex1 = svadd_u8_x(svptrue_b8(), svaindex1, svdup_n_u8(1));
    svbindex1 = svadd_u8_x(svptrue_b8(), svbindex1, svdup_n_u8(1));
  }
  r.sve_u16[EASYSIMD_SV_INDEX_0] = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[0], svarr[1]), svaddlb_u16(svarr[2], svarr[3]));
  r.sve_u16[EASYSIMD_SV_INDEX_1] = svadd_u16_x(svptrue_b16(), svaddlb_u16(svarr[4], svarr[5]), svaddlb_u16(svarr[6], svarr[7]));
  return r;
#else
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  const int a_offset1 = imm8 & 4;
  const int b_offset1 = (imm8 & 3) << 2;
  const int a_offset2 = (imm8 >> 3) & 4;
  const int b_offset2 = ((imm8 >> 3) & 3) << 2;

  #if defined(easysimd_math_abs)
    const int halfway_point = HEDLEY_STATIC_CAST(int, (sizeof(r_.u16) / sizeof(r_.u16[0])) ) / 2;
    for (int i = 0 ; i < halfway_point ; i++) {
      r_.u16[i] =
        HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[a_offset1 + i + 0] - b_.u8[b_offset1 + 0]))) +
        HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[a_offset1 + i + 1] - b_.u8[b_offset1 + 1]))) +
        HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[a_offset1 + i + 2] - b_.u8[b_offset1 + 2]))) +
        HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[a_offset1 + i + 3] - b_.u8[b_offset1 + 3])));
      r_.u16[halfway_point + i] =
        HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[2 * halfway_point + a_offset2 + i + 0] - b_.u8[2 * halfway_point + b_offset2 + 0]))) +
        HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[2 * halfway_point + a_offset2 + i + 1] - b_.u8[2 * halfway_point + b_offset2 + 1]))) +
        HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[2 * halfway_point + a_offset2 + i + 2] - b_.u8[2 * halfway_point + b_offset2 + 2]))) +
        HEDLEY_STATIC_CAST(uint16_t, easysimd_math_abs(HEDLEY_STATIC_CAST(int, a_.u8[2 * halfway_point + a_offset2 + i + 3] - b_.u8[2 * halfway_point + b_offset2 + 3])));
    }
  #else
    HEDLEY_UNREACHABLE();
  #endif

  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE) && EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,9,0)
  #define easysimd_mm256_mpsadbw_epu8(a, b, imm8) _mm256_mpsadbw_epu8(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
  #define easysimd_mm256_mpsadbw_epu8(a, b, imm8) \
     easysimd_mm256_set_m128i( \
       easysimd_mm_mpsadbw_epu8(easysimd_mm256_extracti128_si256(a, 1), easysimd_mm256_extracti128_si256(b, 1), (imm8 >> 3)), \
       easysimd_mm_mpsadbw_epu8(easysimd_mm256_extracti128_si256(a, 0), easysimd_mm256_extracti128_si256(b, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mpsadbw_epu8
  #define _mm256_mpsadbw_epu8(a, b, imm8) easysimd_mm256_mpsadbw_epu8(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mul_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_mul_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svmullb_s64(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    __asm__ __volatile__ (
        "ins %[a0].s[1], %[a0].s[2]             \n\t"
        "ins %[a1].s[1], %[a1].s[2]             \n\t"
        "ins %[b0].s[1], %[b0].s[2]             \n\t"
        "ins %[b1].s[1], %[b1].s[2]             \n\t"
        "smull %[a0].2d, %[a0].2s, %[b0].2s     \n\t"
        "smull %[a1].2d, %[a1].2s, %[b1].2s     \n\t"
        :[a0]"+w"(a.m128i[0].neon_i32), [a1]"+w"(a.m128i[1].neon_i32), [b0]"+w"(b.m128i[0].neon_i32), [b1]"+w"(b.m128i[1].neon_i32)
        :
        :
    );
    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_mul_epi32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_mul_epi32(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] =
          HEDLEY_STATIC_CAST(int64_t, a_.i32[i * 2]) *
          HEDLEY_STATIC_CAST(int64_t, b_.i32[i * 2]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
#  define _mm256_mul_epi32(a, b) easysimd_mm256_mul_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mul_epu32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_mul_epu32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svmullb_u64(a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    __asm__ __volatile__ (
        "ins %[a0].s[1], %[a0].s[2]             \n\t"
        "ins %[a1].s[1], %[a1].s[2]             \n\t"
        "ins %[b0].s[1], %[b0].s[2]             \n\t"
        "ins %[b1].s[1], %[b1].s[2]             \n\t"
        "umull %[a0].2d, %[a0].2s, %[b0].2s     \n\t"
        "umull %[a1].2d, %[a1].2s, %[b1].2s     \n\t"
        :[a0]"+w"(a.m128i[0].neon_u32), [a1]"+w"(a.m128i[1].neon_u32), [b0]"+w"(b.m128i[0].neon_u32), [b1]"+w"(b.m128i[1].neon_u32)
        :
        :
    );
    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_mul_epu32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_mul_epu32(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = HEDLEY_STATIC_CAST(uint64_t, a_.u32[i * 2]) * HEDLEY_STATIC_CAST(uint64_t, b_.u32[i * 2]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
#  define _mm256_mul_epu32(a, b) easysimd_mm256_mul_epu32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mulhi_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_mulhi_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svmulh_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svmulh_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    __asm__ __volatile__ (
        "mov v4.d[0], %[a0].d[1]                    \n\t"
        "mov v5.d[0], %[a1].d[1]                    \n\t"
        "smull %[a0].4s, %[a0].4h, %[b0].4h         \n\t"
        "mov v6.d[0], %[b0].d[1]                    \n\t"
        "mov v7.d[0], %[b1].d[1]                    \n\t"
        "smull %[b0].4s, v4.4h, v6.4h               \n\t"
        "uzp2  %[a0].8h, %[a0].8h, %[b0].8h         \n\t"
        "smull %[a1].4s, %[a1].4h, %[b1].4h         \n\t"
        "smull %[b1].4s, v5.4h, v7.4h               \n\t"
        "uzp2  %[a1].8h, %[a1].8h, %[b1].8h         \n\t"
        :[a0]"+w"(a.m128i[0].neon_i16), [a1]"+w"(a.m128i[1].neon_i16), [b0]"+w"(b.m128i[0].neon_i16), [b1]"+w"(b.m128i[1].neon_i16)
        :
        :"v4", "v5", "v6", "v7"
    );
    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, (HEDLEY_STATIC_CAST(uint32_t, HEDLEY_STATIC_CAST(int32_t, a_.i16[i]) * HEDLEY_STATIC_CAST(int32_t, b_.i16[i])) >> 16));
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
#  define _mm256_mulhi_epi16(a, b) easysimd_mm256_mulhi_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mulhi_epu16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_mulhi_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svmulh_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svmulh_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    __asm__ __volatile__ (
        "mov v4.d[0], %[a0].d[1]                    \n\t"
        "mov v5.d[0], %[a1].d[1]                    \n\t"
        "umull %[a0].4s, %[a0].4h, %[b0].4h         \n\t"
        "mov v6.d[0], %[b0].d[1]                    \n\t"
        "mov v7.d[0], %[b1].d[1]                    \n\t"
        "umull %[b0].4s, v4.4h, v6.4h               \n\t"
        "uzp2  %[a0].8h, %[a0].8h, %[b0].8h         \n\t"
        "umull %[a1].4s, %[a1].4h, %[b1].4h         \n\t"
        "umull %[b1].4s, v5.4h, v7.4h               \n\t"
        "uzp2  %[a1].8h, %[a1].8h, %[b1].8h         \n\t"
        :[a0]"+w"(a.m128i[0].neon_u16), [a1]"+w"(a.m128i[1].neon_u16), [b0]"+w"(b.m128i[0].neon_u16), [b1]"+w"(b.m128i[1].neon_u16)
        :
        :"v4", "v5", "v6", "v7"
    );
    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) * HEDLEY_STATIC_CAST(uint32_t, b_.u16[i]) >> 16);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
#  define _mm256_mulhi_epu16(a, b) easysimd_mm256_mulhi_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mulhi_epi32 (easysimd__m256i a, easysimd__m256i b) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  r.sve_i32[EASYSIMD_SV_INDEX_0] = svmulh_s32_z(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svmulh_s32_z(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
  return r;
#elif defined(EASYSIMD_ARM_NEON_NATIVE)
    __asm__ __volatile__ (
        "mov v4.d[0], %[a0].d[1]                    \n\t"
        "mov v5.d[0], %[a1].d[1]                    \n\t"
        "smull %[a0].2d, %[a0].2s, %[b0].2s         \n\t"
        "mov v6.d[0], %[b0].d[1]                    \n\t"
        "mov v7.d[0], %[b1].d[1]                    \n\t"
        "smull %[b0].2d, v4.2s, v6.2s               \n\t"
        "uzp2  %[a0].4s, %[a0].4s, %[b0].4s         \n\t"
        "smull %[a1].2d, %[a1].2s, %[b1].2s         \n\t"
        "smull %[b1].2d, v5.2s, v7.2s               \n\t"
        "uzp2  %[a1].4s, %[a1].4s, %[b1].4s         \n\t"
        :[a0]"+w"(a.m128i[0].neon_i32), [a1]"+w"(a.m128i[1].neon_i32), [b0]"+w"(b.m128i[0].neon_i32), [b1]"+w"(b.m128i[1].neon_i32)
        :
        :"v4", "v5", "v6", "v7"
    );
    return a;
#else
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
    r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (HEDLEY_STATIC_CAST(uint64_t, HEDLEY_STATIC_CAST(int64_t, a_.i32[i]) * HEDLEY_STATIC_CAST(int64_t, b_.i32[i])) >> 32));
  }

  return easysimd__m256i_from_private(r_);
#endif
}


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mulhi_epu32 (easysimd__m256i a, easysimd__m256i b) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  r.sve_u32[EASYSIMD_SV_INDEX_0] = svmulh_u32_z(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
  r.sve_u32[EASYSIMD_SV_INDEX_1] = svmulh_u32_z(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
  return r;
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    __asm__ __volatile__ (
        "mov v4.d[0], %[a0].d[1]                    \n\t"
        "mov v5.d[0], %[a1].d[1]                    \n\t"
        "umull %[a0].2d, %[a0].2s, %[b0].2s         \n\t"
        "mov v6.d[0], %[b0].d[1]                    \n\t"
        "mov v7.d[0], %[b1].d[1]                    \n\t"
        "umull %[b0].2d, v4.2s, v6.2s               \n\t"
        "uzp2  %[a0].4s, %[a0].4s, %[b0].4s         \n\t"
        "umull %[a1].2d, %[a1].2s, %[b1].2s         \n\t"
        "umull %[b1].2d, v5.2s, v7.2s               \n\t"
        "uzp2  %[a1].4s, %[a1].4s, %[b1].4s         \n\t"
        :[a0]"+w"(a.m128i[0].neon_u32), [a1]"+w"(a.m128i[1].neon_u32), [b0]"+w"(b.m128i[0].neon_u32), [b1]"+w"(b.m128i[1].neon_u32)
        :
        :"v4", "v5", "v6", "v7"
    );
    return a;
#else
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
    r_.u32[i] = HEDLEY_STATIC_CAST(uint32_t, (HEDLEY_STATIC_CAST(uint64_t, HEDLEY_STATIC_CAST(uint64_t, a_.u32[i]) * HEDLEY_STATIC_CAST(uint64_t, b_.u32[i])) >> 32));
  }

  return easysimd__m256i_from_private(r_);
#endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mulhrs_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_mulhrs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  svbool_t pg = svptrue_b32();
  svint32_t r0, r1, r2, r3, inc = svdup_n_s32(0x00004000);

  r0 = svmul_s32_z(pg,
                   svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)])),
                   svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)])));
  r1 = svmul_s32_z(pg,
                   svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)])),
                   svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)])));
  r2 = svmul_s32_z(pg,
                   svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(1, 0)])),
                   svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(1, 0)])));
  r3 = svmul_s32_z(pg,
                   svld1sh_s32(pg, &(a.i16[EASYSIMD_SV_B16_B32_INDEX(1, 1)])),
                   svld1sh_s32(pg, &(b.i16[EASYSIMD_SV_B16_B32_INDEX(1, 1)])));

  r0 = svasr_n_s32_z(pg, svadd_s32_z(pg, r0, inc), 15);
  r1 = svasr_n_s32_z(pg, svadd_s32_z(pg, r1, inc), 15);
  r2 = svasr_n_s32_z(pg, svadd_s32_z(pg, r2, inc), 15);
  r3 = svasr_n_s32_z(pg, svadd_s32_z(pg, r3, inc), 15);

  svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(1, 1)]), r3);
  svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(1, 0)]), r2);
  svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(0, 1)]), r1);
  svst1h_s32(pg, &(r.i16[EASYSIMD_SV_B16_B32_INDEX(0, 0)]), r0);
  return r;

  #elif defined(EASYSIMD_AMR_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    int32x4_t r_0 = vmull_s16(vget_low_s16(a.m128i[0].neon_i16), vget_low_s16(b.m128i[0].neon_i16));
    int32x4_t r_1 = vmull_s16(vget_low_s16(a.m128i[1].neon_i16), vget_low_s16(b.m128i[1].neon_i16));
    int32x4_t r_2 = vmull_s16(vget_high_s16(a.m128i[0].neon_i16), vget_high_s16(b.m128i[0].neon_i16));
    int32x4_t r_3 = vmull_s16(vget_high_s16(a.m128i[1].neon_i16), vget_high_s16(b.m128i[1].neon_i16));
    
    int32x4_t inc = vdupq_n_s32(0x00004000);
    r_0 = vshrq_n_s32(vaddq_s32(r_0, inc), 15);
    r_1 = vshrq_n_s32(vaddq_s32(r_1, inc), 15);
    r_2 = vshrq_n_s32(vaddq_s32(r_2, inc), 15);
    r_3 = vshrq_n_s32(vaddq_s32(r_3, inc), 15);
    res.m128i[0].neon_i16 = vuzp1q_s16(vreinterpretq_s16_s32(r_0), vreinterpretq_s16_s32(r_2));
    res.m128i[1].neon_i16 = vuzp1q_s16(vreinterpretq_s16_s32(r_1), vreinterpretq_s16_s32(r_3));
    return res;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, (((HEDLEY_STATIC_CAST(int32_t, a_.i16[i]) * HEDLEY_STATIC_CAST(int32_t, b_.i16[i])) + 0x4000) >> 15));
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
#  define _mm256_mulhrs_epi16(a, b) easysimd_mm256_mulhrs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mullo_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_mullo_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svmul_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svmul_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_i16 = vmulq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    res.m128i[1].neon_i16 = vmulq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    return res;
  #else
    easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b),
    r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] * b_.i16[i]);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mullo_epi16
  #define _mm256_mullo_epi16(a, b) easysimd_mm256_mullo_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mullo_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_mullo_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svmul_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svmul_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_i32 = vmulq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    res.m128i[1].neon_i32 = vmulq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return res;
  #else
    easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b),
    r_;

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, a_.i32[i] * b_.i32[i]);
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mullo_epi32
  #define _mm256_mullo_epi32(a, b) easysimd_mm256_mullo_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_mullo_epi64 (easysimd__m256i a, easysimd__m256i b) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svmul_s64_z(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svmul_s64_z(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
  return r;
#elif defined(EASYSIMD_ARM_NEON_NATIVE)
    easysimd__m256i res = easysimd_mm256_setzero_si256();
    int64_t ptr_a[4], ptr_b[4], ptr_r[4];
    easysimd_mm256_convert_to_int64(ptr_a, a);
    easysimd_mm256_convert_to_int64(ptr_b, b);
    ptr_r[0] = ptr_a[0] * ptr_b[0];
    ptr_r[1] = ptr_a[1] * ptr_b[1];
    ptr_r[2] = ptr_a[2] * ptr_b[2];
    ptr_r[3] = ptr_a[3] * ptr_b[3];
    res.m128i[0].neon_i64 = vsetq_lane_s64(ptr_r[0], res.m128i[0].neon_i64, 0);
    res.m128i[0].neon_i64 = vsetq_lane_s64(ptr_r[1], res.m128i[0].neon_i64, 1);
    res.m128i[1].neon_i64 = vsetq_lane_s64(ptr_r[2], res.m128i[1].neon_i64, 0);
    res.m128i[1].neon_i64 = vsetq_lane_s64(ptr_r[3], res.m128i[1].neon_i64, 1);
    return res;
#else
  easysimd__m256i_private
  a_ = easysimd__m256i_to_private(a),
  b_ = easysimd__m256i_to_private(b),
  r_;

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
    r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.i64[i] * b_.i64[i]);
  }

  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_mullo_epi64(a, b) _mm256_mullo_epi64(a, b)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mullo_epi64
  #define _mm256_mullo_epi64(a, b) easysimd_mm256_mullo_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_mullo_epu32 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.u32 = a_.u32 * b_.u32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i] * b_.u32[i];
      }
    #endif

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_or_si256 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_or_si256(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svorr_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svorr_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vorrq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vorrq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_or_si128(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_or_si128(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f | b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = a_.i32f[i] | b_.i32f[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_or_si256
  #define _mm256_or_si256(a, b) easysimd_mm256_or_si256(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_packs_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_packs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svuzp1_s8(svqxtnb_s16(a.sve_i16[EASYSIMD_SV_INDEX_0]), svqxtnb_s16(b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svuzp1_s8(svqxtnb_s16(a.sve_i16[EASYSIMD_SV_INDEX_1]), svqxtnb_s16(b.sve_i16[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_packs_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_packs_epi16(a_.m128i[1], b_.m128i[1]);
    #else
      const size_t halfway_point = (sizeof(r_.i8) / sizeof(r_.i8[0]))/2;
      const size_t quarter_point = (sizeof(r_.i8) / sizeof(r_.i8[0]))/4;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < quarter_point ; i++) {
        r_.i8[i]     = (a_.i16[i] > INT8_MAX) ? INT8_MAX : ((a_.i16[i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, a_.i16[i]));
        r_.i8[i + quarter_point] = (b_.i16[i] > INT8_MAX) ? INT8_MAX : ((b_.i16[i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, b_.i16[i]));
        r_.i8[halfway_point + i]     = (a_.i16[quarter_point + i] > INT8_MAX) ? INT8_MAX : ((a_.i16[quarter_point + i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, a_.i16[quarter_point + i]));
        r_.i8[halfway_point + i + quarter_point] = (b_.i16[quarter_point + i] > INT8_MAX) ? INT8_MAX : ((b_.i16[quarter_point + i] < INT8_MIN) ? INT8_MIN : HEDLEY_STATIC_CAST(int8_t, b_.i16[quarter_point + i]));
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_packs_epi16
  #define _mm256_packs_epi16(a, b) easysimd_mm256_packs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_packs_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_packs_epi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i16 = vcombine_s16(vqmovn_s32(a.m128i[0].neon_i32), vqmovn_s32(b.m128i[0].neon_i32));
    r.m128i[1].neon_i16 = vcombine_s16(vqmovn_s32(a.m128i[1].neon_i32), vqmovn_s32(b.m128i[1].neon_i32));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svuzp1_s16(svqxtnb_s32(a.sve_i32[EASYSIMD_SV_INDEX_0]), svqxtnb_s32(b.sve_i32[EASYSIMD_SV_INDEX_0]));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svuzp1_s16(svqxtnb_s32(a.sve_i32[EASYSIMD_SV_INDEX_1]), svqxtnb_s32(b.sve_i32[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256i_private
      r_,
      v_[] = {
        easysimd__m256i_to_private(a),
        easysimd__m256i_to_private(b)
      };

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_packs_epi32(v_[0].m128i[0], v_[1].m128i[0]);
      r_.m128i[1] = easysimd_mm_packs_epi32(v_[0].m128i[1], v_[1].m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        const int32_t v = v_[(i >> 2) & 1].i32[(i & 11) - ((i & 8) >> 1)];
        r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, (v > INT16_MAX) ? INT16_MAX : ((v < INT16_MIN) ? INT16_MIN : v));
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_packs_epi32
  #define _mm256_packs_epi32(a, b) easysimd_mm256_packs_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_packus_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_packus_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svuzp1_u8(svqxtunb_s16(a.sve_i16[EASYSIMD_SV_INDEX_0]), svqxtunb_s16(b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svuzp1_u8(svqxtunb_s16(a.sve_i16[EASYSIMD_SV_INDEX_1]), svqxtunb_s16(b.sve_i16[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_packus_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_packus_epi16(a_.m128i[1], b_.m128i[1]);
    #else
      const size_t halfway_point = (sizeof(r_.i8) / sizeof(r_.i8[0])) / 2;
      const size_t quarter_point = (sizeof(r_.i8) / sizeof(r_.i8[0])) / 4;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < quarter_point ; i++) {
        r_.u8[i] = (a_.i16[i] > UINT8_MAX) ? UINT8_MAX : ((a_.i16[i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, a_.i16[i]));
        r_.u8[i + quarter_point] = (b_.i16[i] > UINT8_MAX) ? UINT8_MAX : ((b_.i16[i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, b_.i16[i]));
        r_.u8[halfway_point + i] = (a_.i16[quarter_point + i] > UINT8_MAX) ? UINT8_MAX : ((a_.i16[quarter_point + i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, a_.i16[quarter_point + i]));
        r_.u8[halfway_point + i + quarter_point] = (b_.i16[quarter_point + i] > UINT8_MAX) ? UINT8_MAX : ((b_.i16[quarter_point + i] < 0) ? UINT8_C(0) : HEDLEY_STATIC_CAST(uint8_t, b_.i16[quarter_point + i]));
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_packus_epi16
  #define _mm256_packus_epi16(a, b) easysimd_mm256_packus_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_packus_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_packus_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svuzp1_u16(svqxtunb_s32(a.sve_i32[EASYSIMD_SV_INDEX_0]), svqxtunb_s32(b.sve_i32[EASYSIMD_SV_INDEX_0]));
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svuzp1_u16(svqxtunb_s32(a.sve_i32[EASYSIMD_SV_INDEX_1]), svqxtunb_s32(b.sve_i32[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_packus_epi32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_packus_epi32(a_.m128i[1], b_.m128i[1]);
    #else
      const size_t halfway_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 2;
      const size_t quarter_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 4;
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < quarter_point ; i++) {
        r_.u16[i] = (a_.i32[i] > UINT16_MAX) ? UINT16_MAX : ((a_.i32[i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, a_.i32[i]));
        r_.u16[i + quarter_point] = (b_.i32[i] > UINT16_MAX) ? UINT16_MAX : ((b_.i32[i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, b_.i32[i]));
        r_.u16[halfway_point + i]     = (a_.i32[quarter_point + i] > UINT16_MAX) ? UINT16_MAX : ((a_.i32[quarter_point + i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, a_.i32[quarter_point + i]));
        r_.u16[halfway_point + i + quarter_point] = (b_.i32[quarter_point + i] > UINT16_MAX) ? UINT16_MAX : ((b_.i32[quarter_point + i] < 0) ? UINT16_C(0) : HEDLEY_STATIC_CAST(uint16_t, b_.i32[quarter_point + i]));
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_packus_epi32
  #define _mm256_packus_epi32(a, b) easysimd_mm256_packus_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permute2x128_si256 (easysimd__m256i a, easysimd__m256i b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    int bit_0 = imm8 & 0x1, bit_1 = imm8 & 0x2, bit_3 = imm8 & 0x8;
    if (bit_1 == 0) {
        r.m128i[0].neon_i32 = a.m128i[bit_0].neon_i32;
    } else {
        r.m128i[0].neon_i32 = b.m128i[bit_0].neon_i32;
    }
    if (bit_3) {
        r.m128i[0].neon_i32 = vdupq_n_s32(0);
    }
    bit_0 = (imm8 & 0x10) >> 4, bit_1 = imm8 & 0x20, bit_3 = imm8 & 0x80;
    if (bit_1 == 0) {
        r.m128i[1].neon_i32 = a.m128i[bit_0].neon_i32;
    } else {
        r.m128i[1].neon_i32 = b.m128i[bit_0].neon_i32;
    }
    if (bit_3) {
        r.m128i[1].neon_i32 = vdupq_n_s32(0);
    }
    return r;
   #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    int bit_0 = (imm8 & 0x1) ? EASYSIMD_SV_INDEX_1 : EASYSIMD_SV_INDEX_0;
    int bit_1 = imm8 & 0x2, bit_3 = imm8 & 0x8;
    if (bit_1 == 0) {
      r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32[bit_0];
    } else {
      r.sve_i32[EASYSIMD_SV_INDEX_0] = b.sve_i32[bit_0];
    }
    if (bit_3) {
      r.sve_i32[EASYSIMD_SV_INDEX_0] = svdup_n_s32(0);
    }
    bit_0 = ((imm8 & 0x10) >> 4) ? EASYSIMD_SV_INDEX_1 : EASYSIMD_SV_INDEX_0;
    bit_1 = imm8 & 0x20, bit_3 = imm8 & 0x80;
    if (bit_1 == 0) {
      r.sve_i32[EASYSIMD_SV_INDEX_1] = a.sve_i32[bit_0];
    } else {
      r.sve_i32[EASYSIMD_SV_INDEX_1] = b.sve_i32[bit_0];
    }
    if (bit_3) {
      r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32(0);
    }
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    r_.m128i_private[0] = (imm8 & 0x08) ? easysimd__m128i_to_private(easysimd_mm_setzero_si128()) : ((imm8 & 0x02) ? b_.m128i_private[(imm8     ) & 1] : a_.m128i_private[(imm8     ) & 1]);
    r_.m128i_private[1] = (imm8 & 0x80) ? easysimd__m128i_to_private(easysimd_mm_setzero_si128()) : ((imm8 & 0x20) ? b_.m128i_private[(imm8 >> 4) & 1] : a_.m128i_private[(imm8 >> 4) & 1]);

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_permute2x128_si256(a, b, imm8) _mm256_permute2x128_si256(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permute2x128_si256
  #define _mm256_permute2x128_si256(a, b, imm8) easysimd_mm256_permute2x128_si256(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permute4x64_epi64 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  svint64x2_t svtemp = svcreate2_s64(a.sve_i64[0], a.sve_i64[1]);
  svuint64_t svoffset0 = svdupq_n_u64((imm8 >> 0) & 0x03, (imm8 >> 2) & 0x03);
  svuint64_t svoffset1 = svdupq_n_u64((imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svtbl2_s64(svtemp, svoffset0);
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svtbl2_s64(svtemp, svoffset1);
  return r;
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  easysimd__m256i res = easysimd_mm256_setzero_si256();
  int64_t ptr_a[4];
  vst1q_s64(ptr_a    , a.m128i[0].neon_i64);
  vst1q_s64(ptr_a + 2, a.m128i[1].neon_i64);
  const int id0 = imm8 & 0x03;
  const int id1 = (imm8 >> 2) & 0x03;
  const int id2 = (imm8 >> 4) & 0x03;
  const int id3 = (imm8 >> 6) & 0x03;
  res.m128i[0].neon_i64 = vsetq_lane_s64(ptr_a[id0], res.m128i[0].neon_i64, 0);
  res.m128i[0].neon_i64 = vsetq_lane_s64(ptr_a[id1], res.m128i[0].neon_i64, 1);
  res.m128i[1].neon_i64 = vsetq_lane_s64(ptr_a[id2], res.m128i[1].neon_i64, 0);
  res.m128i[1].neon_i64 = vsetq_lane_s64(ptr_a[id3], res.m128i[1].neon_i64, 1);
  return res;
#else
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a);

  r_.i64[0] = (imm8 & 0x02) ? a_.i64[((imm8       ) & 1)+2] : a_.i64[(imm8       ) & 1];
  r_.i64[1] = (imm8 & 0x08) ? a_.i64[((imm8 >> 2  ) & 1)+2] : a_.i64[(imm8 >> 2  ) & 1];
  r_.i64[2] = (imm8 & 0x20) ? a_.i64[((imm8 >> 4  ) & 1)+2] : a_.i64[(imm8 >> 4  ) & 1];
  r_.i64[3] = (imm8 & 0x80) ? a_.i64[((imm8 >> 6  ) & 1)+2] : a_.i64[(imm8 >> 6  ) & 1];

  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_permute4x64_epi64(a, imm8) _mm256_permute4x64_epi64(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permute4x64_epi64
  #define _mm256_permute4x64_epi64(a, imm8) easysimd_mm256_permute4x64_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_permute4x64_pd (easysimd__m256d a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svfloat64x2_t svtemp = svcreate2(a.sve_f64[0], a.sve_f64[1]);
    svuint64_t svoffset0 = svdupq_n_u64((imm8 >> 0) & 0x03, (imm8 >> 2) & 0x03);
    svuint64_t svoffset1 = svdupq_n_u64((imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svtbl2_f64(svtemp, svoffset0);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svtbl2_f64(svtemp, svoffset1);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    r_.f64[0] = (imm8 & 0x02) ? a_.f64[((imm8       ) & 1)+2] : a_.f64[(imm8       ) & 1];
    r_.f64[1] = (imm8 & 0x08) ? a_.f64[((imm8 >> 2  ) & 1)+2] : a_.f64[(imm8 >> 2  ) & 1];
    r_.f64[2] = (imm8 & 0x20) ? a_.f64[((imm8 >> 4  ) & 1)+2] : a_.f64[(imm8 >> 4  ) & 1];
    r_.f64[3] = (imm8 & 0x80) ? a_.f64[((imm8 >> 6  ) & 1)+2] : a_.f64[(imm8 >> 6  ) & 1];

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_permute4x64_pd(a, imm8) _mm256_permute4x64_pd(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permute4x64_pd
  #define _mm256_permute4x64_pd(a, imm8) easysimd_mm256_permute4x64_pd(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permutevar8x32_epi32 (easysimd__m256i a, easysimd__m256i idx) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_permutevar8x32_epi32(a, idx);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    svuint32_t index0 = svand_n_u32_z(pg, idx.sve_u32[EASYSIMD_SV_INDEX_0], 0x07);
    svuint32_t index1 = svand_n_u32_z(pg, idx.sve_u32[EASYSIMD_SV_INDEX_1], 0x07);
    svint32x2_t svtemp = svcreate2_s32(a.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_1]);
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svtbl2_s32(svtemp, index0);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svtbl2_s32(svtemp, index1);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      idx_ = easysimd__m256i_to_private(idx);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = a_.i32[idx_.i32[i] & 7];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutevar8x32_epi32
  #define _mm256_permutevar8x32_epi32(a, idx) easysimd_mm256_permutevar8x32_epi32(a, idx)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_permutevar8x32_ps (easysimd__m256 a, easysimd__m256i idx) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
      return _mm256_permutevar8x32_ps(a, HEDLEY_REINTERPRET_CAST(easysimd__m256, idx));
    #else
      return _mm256_permutevar8x32_ps(a, idx);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    svuint32_t index0 = svand_n_u32_z(pg, idx.sve_u32[EASYSIMD_SV_INDEX_0], 0x07);
    svuint32_t index1 = svand_n_u32_z(pg, idx.sve_u32[EASYSIMD_SV_INDEX_1], 0x07);
    svfloat32x2_t svtemp = svcreate2_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svtbl2_f32(svtemp, index0);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svtbl2_f32(svtemp, index1);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r = easysimd_mm256_setzero_ps();
    int32_t index[8]  = {0};
    easysimd__m128i ind;
    ind.neon_i32  = vandq_s32(idx.m128i[0].neon_i32, vdupq_n_s32(0x07));
    vst1q_s32(index    , ind.neon_i32);
    ind.neon_i32  = vandq_s32(idx.m128i[1].neon_i32, vdupq_n_s32(0x07));
    vst1q_s32(index + 4, ind.neon_i32);

    r.m128[0].neon_f32 = vsetq_lane_f32(a.f32[index[0]], r.m128[0].neon_f32, 0);
    r.m128[0].neon_f32 = vsetq_lane_f32(a.f32[index[1]], r.m128[0].neon_f32, 1);
    r.m128[0].neon_f32 = vsetq_lane_f32(a.f32[index[2]], r.m128[0].neon_f32, 2);
    r.m128[0].neon_f32 = vsetq_lane_f32(a.f32[index[3]], r.m128[0].neon_f32, 3);
    r.m128[1].neon_f32 = vsetq_lane_f32(a.f32[index[4]], r.m128[1].neon_f32, 0);
    r.m128[1].neon_f32 = vsetq_lane_f32(a.f32[index[5]], r.m128[1].neon_f32, 1);
    r.m128[1].neon_f32 = vsetq_lane_f32(a.f32[index[6]], r.m128[1].neon_f32, 2);
    r.m128[1].neon_f32 = vsetq_lane_f32(a.f32[index[7]], r.m128[1].neon_f32, 3);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);
    easysimd__m256i_private
      idx_ = easysimd__m256i_to_private(idx);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = a_.f32[idx_.i32[i] & 7];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutevar8x32_ps
  #define _mm256_permutevar8x32_ps(a, idx) easysimd_mm256_permutevar8x32_ps(a, idx)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sad_epu8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sad_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();

    easysimd__m256i r0, r1, r2, r3;
    r0.sve_u16[EASYSIMD_SV_INDEX_0] = svld1ub_u16(pg, &(a.u8[0]));
    r0.sve_u16[EASYSIMD_SV_INDEX_1] = svld1ub_u16(pg, &(a.u8[8]));
    r1.sve_u16[EASYSIMD_SV_INDEX_0] = svld1ub_u16(pg, &(a.u8[16]));
    r1.sve_u16[EASYSIMD_SV_INDEX_1] = svld1ub_u16(pg, &(a.u8[24]));

    r2.sve_u16[EASYSIMD_SV_INDEX_0] = svld1ub_u16(pg, &(b.u8[0]));
    r2.sve_u16[EASYSIMD_SV_INDEX_1] = svld1ub_u16(pg, &(b.u8[8]));
    r3.sve_u16[EASYSIMD_SV_INDEX_0] = svld1ub_u16(pg, &(b.u8[16]));
    r3.sve_u16[EASYSIMD_SV_INDEX_1] = svld1ub_u16(pg, &(b.u8[24]));

    r.u64[0] = svaddv_u16(pg, svabd_u16_z(pg, r0.sve_u16[EASYSIMD_SV_INDEX_0], r2.sve_u16[EASYSIMD_SV_INDEX_0]));
    r.u64[1] = svaddv_u16(pg, svabd_u16_z(pg, r0.sve_u16[EASYSIMD_SV_INDEX_1], r2.sve_u16[EASYSIMD_SV_INDEX_1]));
    r.u64[2] = svaddv_u16(pg, svabd_u16_z(pg, r1.sve_u16[EASYSIMD_SV_INDEX_0], r3.sve_u16[EASYSIMD_SV_INDEX_0]));
    r.u64[3] = svaddv_u16(pg, svabd_u16_z(pg, r1.sve_u16[EASYSIMD_SV_INDEX_1], r3.sve_u16[EASYSIMD_SV_INDEX_1]));

    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    uint16x8_t v0, v1;
    v0 = vmovl_u8(vget_low_u8(a.m128i[0].neon_u8));
    v1 = vmovl_u8(vget_low_u8(b.m128i[0].neon_u8));
    v0 = vabdq_u16(v0, v1);
    a.u64[0] = vaddvq_u16(v0);

    v0 = vmovl_u8(vget_high_u8(a.m128i[0].neon_u8));
    v1 = vmovl_u8(vget_high_u8(b.m128i[0].neon_u8));
    v0 = vabdq_u16(v0, v1);
    a.u64[1] = vaddvq_u16(v0);

    v0 = vmovl_u8(vget_low_u8(a.m128i[1].neon_u8));
    v1 = vmovl_u8(vget_low_u8(b.m128i[1].neon_u8));
    v0 = vabdq_u16(v0, v1);
    a.u64[2] = vaddvq_u16(v0);

    v0 = vmovl_u8(vget_high_u8(a.m128i[1].neon_u8));
    v1 = vmovl_u8(vget_high_u8(b.m128i[1].neon_u8));
    v0 = vabdq_u16(v0, v1);
    a.u64[3] = vaddvq_u16(v0);
    return a;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sad_epu8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_sad_epu8(a_.m128i[1], b_.m128i[1]);
    #else
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        uint16_t tmp = 0;
        EASYSIMD_VECTORIZE_REDUCTION(+:tmp)
        for (size_t j = 0 ; j < ((sizeof(r_.u8) / sizeof(r_.u8[0])) / 4) ; j++) {
          const size_t e = j + (i * 8);
          tmp += (a_.u8[e] > b_.u8[e]) ? (a_.u8[e] - b_.u8[e]) : (b_.u8[e] - a_.u8[e]);
        }
        r_.i64[i] = tmp;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sad_epu8
  #define _mm256_sad_epu8(a, b) easysimd_mm256_sad_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shuffle_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_shuffle_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    svuint8_t svi, svmask = svdup_n_u8(0x8F);
    svi = svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_0], svmask);
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], svi);

    svi = svand_u8_z(pg, b.sve_u8[EASYSIMD_SV_INDEX_1], svmask);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], svi);

    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    uint8x16_t mask_and = vdupq_n_u8(0x8f);
    r.m128i[0].neon_u8 = vqtbl1q_u8(a.m128i[0].neon_u8, vandq_u8(b.m128i[0].neon_u8, mask_and));
    r.m128i[1].neon_u8 = vqtbl1q_u8(a.m128i[1].neon_u8, vandq_u8(b.m128i[1].neon_u8, mask_and));

    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_shuffle_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_shuffle_epi8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < ((sizeof(r_.u8) / sizeof(r_.u8[0])) / 2) ; i++) {
        r_.u8[  i   ] = (b_.u8[  i   ] & 0x80) ? 0 : a_.u8[(b_.u8[  i   ] & 0x0f)     ];
        r_.u8[i + 16] = (b_.u8[i + 16] & 0x80) ? 0 : a_.u8[(b_.u8[i + 16] & 0x0f) + 16];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shuffle_epi8
  #define _mm256_shuffle_epi8(a, b) easysimd_mm256_shuffle_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shuffle_epi32 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    sveuint32_t imm = svdupq_n_u32((imm8 & 3), ((imm8 >> 2) & 3), ((imm8 >> 4) & 3), ((imm8 >> 6) & 3));
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svtbl_s32(a.sve_i32[EASYSIMD_SV_INDEX_0], imm);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svtbl_s32(a.sve_i32[EASYSIMD_SV_INDEX_1], imm);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r = easysimd_mm256_setzero_si256();
    int32_t p_a[8];
    vst1q_s32(p_a, a.m128i[0].neon_i32);
    vst1q_s32(p_a + 4, a.m128i[1].neon_i32);
    r.m128i[0].neon_i32 = vsetq_lane_s32(p_a[imm8 & 3], r.m128i[0].neon_i32, 0);
    r.m128i[0].neon_i32 = vsetq_lane_s32(p_a[(imm8 >> 2) & 3], r.m128i[0].neon_i32, 1);
    r.m128i[0].neon_i32 = vsetq_lane_s32(p_a[(imm8 >> 4) & 3], r.m128i[0].neon_i32, 2);
    r.m128i[0].neon_i32 = vsetq_lane_s32(p_a[(imm8 >> 6) & 3], r.m128i[0].neon_i32, 3);
    r.m128i[1].neon_i32 = vsetq_lane_s32(p_a[(imm8 & 3) + 4], r.m128i[1].neon_i32, 0);
    r.m128i[1].neon_i32 = vsetq_lane_s32(p_a[((imm8 >> 2) & 3) + 4], r.m128i[1].neon_i32, 1);
    r.m128i[1].neon_i32 = vsetq_lane_s32(p_a[((imm8 >> 4) & 3) + 4], r.m128i[1].neon_i32, 2);
    r.m128i[1].neon_i32 = vsetq_lane_s32(p_a[((imm8 >> 6) & 3) + 4], r.m128i[1].neon_i32, 3);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    for (size_t i = 0 ; i < ((sizeof(r_.i32) / sizeof(r_.i32[0])) / 2) ; i++) {
      r_.i32[i] = a_.i32[(imm8 >> (i * 2)) & 3];
    }
    for (size_t i = 0 ; i < ((sizeof(r_.i32) / sizeof(r_.i32[0])) / 2) ; i++) {
      r_.i32[i + 4] = a_.i32[((imm8 >> (i * 2)) & 3) + 4];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_shuffle_epi32(a, imm8) _mm256_shuffle_epi32(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128) && !defined(__PGI)
#  define easysimd_mm256_shuffle_epi32(a, imm8) \
     easysimd_mm256_set_m128i( \
       easysimd_mm_shuffle_epi32(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
       easysimd_mm_shuffle_epi32(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
#  define easysimd_mm256_shuffle_epi32(a, imm8) (__extension__ ({ \
      const easysimd__m256i_private easysimd__tmp_a_ = easysimd__m256i_to_private(a); \
      easysimd__m256i_from_private((easysimd__m256i_private) { .i32 = \
          EASYSIMD_SHUFFLE_VECTOR_(32, 32, \
                                (easysimd__tmp_a_).i32, \
                                (easysimd__tmp_a_).i32, \
                                ((imm8)     ) & 3, \
                                ((imm8) >> 2) & 3, \
                                ((imm8) >> 4) & 3, \
                                ((imm8) >> 6) & 3, \
                                (((imm8)     ) & 3) + 4, \
                                (((imm8) >> 2) & 3) + 4, \
                                (((imm8) >> 4) & 3) + 4, \
                                (((imm8) >> 6) & 3) + 4) }); }))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shuffle_epi32
  #define _mm256_shuffle_epi32(a, imm8) easysimd_mm256_shuffle_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shufflehi_epi16 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    sveuint16_t imm = svdupq_n_u16(0, 1, 2, 3, (imm8 & 3) + 4, ((imm8 >> 2) & 3) + 4, ((imm8 >> 4) & 3) + 4, ((imm8 >> 6) & 3) + 4);
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svtbl_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], imm);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svtbl_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], imm);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    for (size_t i = 0 ; i < ((sizeof(r_.i32) / sizeof(r_.i32[0])) / 2) ; i++) {
      r_.i32[i] = a_.i32[i];
      r_.i32[i + 4] = a_.i32[((imm8 >> (i * 2)) & 3) + 4];
    }
    for (size_t i = 0 ; i < ((sizeof(r_.i32) / sizeof(r_.i32[0])) / 2) ; i++) {
      r_.i32[i + 8] = a_.i32[i + 8];
      r_.i32[i + 12] = a_.i32[((imm8 >> (i * 2)) & 3) + 12];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}

#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_shufflehi_epi16(a, imm8) _mm256_shufflehi_epi16(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_shufflehi_epi16(a, imm8) \
     easysimd_mm256_set_m128i( \
       easysimd_mm_shufflehi_epi16(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
       easysimd_mm_shufflehi_epi16(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
#  define easysimd_mm256_shufflehi_epi16(a, imm8) (__extension__ ({ \
      const easysimd__m256i_private easysimd__tmp_a_ = easysimd__m256i_to_private(a); \
      easysimd__m256i_from_private((easysimd__m256i_private) { .i16 = \
        EASYSIMD_SHUFFLE_VECTOR_(16, 32, \
          (easysimd__tmp_a_).i16, \
          (easysimd__tmp_a_).i16, \
          0, 1, 2, 3, \
          (((imm8)     ) & 3) + 4, \
          (((imm8) >> 2) & 3) + 4, \
          (((imm8) >> 4) & 3) + 4, \
          (((imm8) >> 6) & 3) + 4, \
          8, 9, 10, 11, \
          ((((imm8)     ) & 3) + 8 + 4), \
          ((((imm8) >> 2) & 3) + 8 + 4), \
          ((((imm8) >> 4) & 3) + 8 + 4), \
          ((((imm8) >> 6) & 3) + 8 + 4) \
          ) }); }))
#else
#  define easysimd_mm256_shufflehi_epi16(a, imm8) \
     easysimd_mm256_set_m128i( \
       easysimd_mm_shufflehi_epi16(easysimd_mm256_extracti128_si256(a, 1), imm8), \
       easysimd_mm_shufflehi_epi16(easysimd_mm256_extracti128_si256(a, 0), imm8))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shufflehi_epi16
  #define _mm256_shufflehi_epi16(a, imm8) easysimd_mm256_shufflehi_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_shufflelo_epi16 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    sveuint16_t imm = svdupq_n_u16((imm8 & 3), ((imm8 >> 2) & 3), ((imm8 >> 4) & 3), ((imm8 >> 6) & 3), 4, 5, 6, 7);
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svtbl_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], imm);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svtbl_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], imm);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    for (size_t i = 0 ; i < ((sizeof(r_.i32) / sizeof(r_.i32[0])) / 2) ; i++) {
      r_.i32[i] = a_.i32[(imm8 >> (i * 2)) & 3];
      r_.i32[i + 4] = a_.i32[i + 4];
    }
    for (size_t i = 0 ; i < ((sizeof(r_.i32) / sizeof(r_.i32[0])) / 2) ; i++) {
      r_.i32[i + 8] = a_.i32[((imm8 >> (i * 2)) & 3) + 8];
      r_.i32[i + 12] = a_.i32[i + 12];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}

#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_shufflelo_epi16(a, imm8) _mm256_shufflelo_epi16(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_shufflelo_epi16(a, imm8) \
     easysimd_mm256_set_m128i( \
       easysimd_mm_shufflelo_epi16(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
       easysimd_mm_shufflelo_epi16(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
#  define easysimd_mm256_shufflelo_epi16(a, imm8) (__extension__ ({ \
      const easysimd__m256i_private easysimd__tmp_a_ = easysimd__m256i_to_private(a); \
      easysimd__m256i_from_private((easysimd__m256i_private) { .i16 = \
        EASYSIMD_SHUFFLE_VECTOR_(16, 32, \
          (easysimd__tmp_a_).i16, \
          (easysimd__tmp_a_).i16, \
          (((imm8)     ) & 3), \
          (((imm8) >> 2) & 3), \
          (((imm8) >> 4) & 3), \
          (((imm8) >> 6) & 3), \
          4, 5, 6, 7, \
          ((((imm8)     ) & 3) + 8), \
          ((((imm8) >> 2) & 3) + 8), \
          ((((imm8) >> 4) & 3) + 8), \
          ((((imm8) >> 6) & 3) + 8), \
          12, 13, 14, 15) }); }))
#else
#  define easysimd_mm256_shufflelo_epi16(a, imm8) \
     easysimd_mm256_set_m128i( \
       easysimd_mm_shufflelo_epi16(easysimd_mm256_extracti128_si256(a, 1), imm8), \
       easysimd_mm_shufflelo_epi16(easysimd_mm256_extracti128_si256(a, 0), imm8))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shufflelo_epi16
  #define _mm256_shufflelo_epi16(a, imm8) easysimd_mm256_shufflelo_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sign_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sign_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b8();
    easysimd_svbool_t pgm0 = svcmplt_n_s8(pg, b.sve_i8[EASYSIMD_SV_INDEX_0], INT8_C(0));
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsel_s8(pgm0, svneg_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0]), a.sve_i8[EASYSIMD_SV_INDEX_0]);

    easysimd_svbool_t pgm1 = svcmplt_n_s8(pg, b.sve_i8[EASYSIMD_SV_INDEX_1], INT8_C(0));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsel_s8(pgm1, svneg_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1]), a.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
      r_.i8[i] = (b_.i8[i] < INT32_C(0)) ? -a_.i8[i] : a_.i8[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sign_epi8
  #define _mm256_sign_epi8(a, b) easysimd_mm256_sign_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sign_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sign_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    easysimd_svbool_t pgm0 = svcmplt_n_s16(pg, b.sve_i16[EASYSIMD_SV_INDEX_0], INT16_C(0));
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsel_s16(pgm0, svneg_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0]), a.sve_i16[EASYSIMD_SV_INDEX_0]);

    easysimd_svbool_t pgm1 = svcmplt_n_s16(pg, b.sve_i16[EASYSIMD_SV_INDEX_1], INT16_C(0));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsel_s16(pgm1, svneg_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1]), a.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = (b_.i16[i] < INT32_C(0)) ? -a_.i16[i] : a_.i16[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sign_epi16
  #define _mm256_sign_epi16(a, b) easysimd_mm256_sign_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sign_epi32(easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sign_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    easysimd_svbool_t pgm0 = svcmplt_n_s32(pg, b.sve_i32[EASYSIMD_SV_INDEX_0], INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsel_s32(pgm0, svneg_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]), a.sve_i32[EASYSIMD_SV_INDEX_0]);

    easysimd_svbool_t pgm1 = svcmplt_n_s32(pg, b.sve_i32[EASYSIMD_SV_INDEX_1], INT32_C(0));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsel_s32(pgm1, svneg_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]), a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0; i < (sizeof(r_.i32) / sizeof(r_.i32[0])); i++) {
      r_.i32[i] = (b_.i32[i] < INT32_C(0)) ? -a_.i32[i] : a_.i32[i];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sign_epi32
  #define _mm256_sign_epi32(a, b) easysimd_mm256_sign_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sll_epi16 (easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sll_epi16(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], count.u64[0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svlsl_n_s16_x(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], count.u64[0]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sll_epi16(a_.m128i[0], count);
      r_.m128i[1] = easysimd_mm_sll_epi16(a_.m128i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
      if (shift > 15)
        return easysimd_mm256_setzero_si256();

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i16 = a_.i16 << HEDLEY_STATIC_CAST(int16_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (shift));
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sll_epi16
  #define _mm256_sll_epi16(a, count) easysimd_mm256_sll_epi16(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sll_epi32 (easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sll_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], count.u64[0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], count.u64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    if (HEDLEY_LIKELY((count.neon_i64[0]) >= 0 && (count.neon_i64[0]) < 32)) {
      r.m128i[0].neon_i32 = vshlq_s32(a.m128i[0].neon_i32, vdupq_n_s32(HEDLEY_STATIC_CAST(int32_t, count.neon_i64[0])));
      r.m128i[1].neon_i32 = vshlq_s32(a.m128i[1].neon_i32, vdupq_n_s32(HEDLEY_STATIC_CAST(int32_t, count.neon_i64[0])));
    } else {
      r.m128i[0].neon_i32 = vdupq_n_s32(0);
      r.m128i[1].neon_i32 = vdupq_n_s32(0);
    } 
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sll_epi32(a_.m128i[0], count);
      r_.m128i[1] = easysimd_mm_sll_epi32(a_.m128i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
      if (shift > 31)
        return easysimd_mm256_setzero_si256();

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i32 = a_.i32 << HEDLEY_STATIC_CAST(int32_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, a_.i32[i] << (shift));
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sll_epi32
  #define _mm256_sll_epi32(a, count) easysimd_mm256_sll_epi32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sll_epi64 (easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sll_epi64(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], count.u64[0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], count.u64[0]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    if (HEDLEY_LIKELY((count.neon_i64[0]) >= 0 && (count.neon_i64[0]) < 64)) {
      r.m128i[0].neon_i64 = vshlq_s64(a.m128i[0].neon_i64, vdupq_n_s64(count.neon_i64[0]));
      r.m128i[1].neon_i64 = vshlq_s64(a.m128i[1].neon_i64, vdupq_n_s64(count.neon_i64[0]));
    } else {
      r.m128i[0].neon_i64 = vdupq_n_s64(0);
      r.m128i[1].neon_i64 = vdupq_n_s64(0);
    } 
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sll_epi64(a_.m128i[0], count);
      r_.m128i[1] = easysimd_mm_sll_epi64(a_.m128i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);
      if (shift > 63)
        return easysimd_mm256_setzero_si256();

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i64 = a_.i64 << HEDLEY_STATIC_CAST(int64_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
          r_.i64[i] = HEDLEY_STATIC_CAST(int64_t, a_.i64[i] << (shift));
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sll_epi64
  #define _mm256_sll_epi64(a, count) easysimd_mm256_sll_epi64(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_slli_epi16 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  /* Note: There is no consistency in how compilers handle values outside of
     the expected range, hence the discrepancy between what we allow and what
     Intel specifies.  Some compilers will return 0, others seem to just mask
     off everything outside of the range. */
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a);
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
      if (imm8 > 15)
        return easysimd_mm256_setzero_si256();
      svbool_t pg = svptrue_b16();
      r_.sve_i16[EASYSIMD_SV_INDEX_0] = svlsl_n_s16_z(pg, a_.sve_i16[EASYSIMD_SV_INDEX_0], imm8);
      r_.sve_i16[EASYSIMD_SV_INDEX_1] = svlsl_n_s16_z(pg, a_.sve_i16[EASYSIMD_SV_INDEX_1], imm8);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = HEDLEY_STATIC_CAST(int16_t, a_.i16[i] << (imm8 & 0xff));
    }
  #endif

  return easysimd__m256i_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_slli_epi16(a, imm8) _mm256_slli_epi16(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_slli_epi16
  #define _mm256_slli_epi16(a, imm8) easysimd_mm256_slli_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_slli_epi32 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    if (imm8 > 31) {
      return easysimd_mm256_setzero_si256();
    } else {
      easysimd_svbool_t pg = svptrue_b32();
      r.sve_i32[EASYSIMD_SV_INDEX_0] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], imm8);
      r.sve_i32[EASYSIMD_SV_INDEX_1] = svlsl_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], imm8);
    }
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i32 = a_.i32 << HEDLEY_STATIC_CAST(int32_t, imm8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] << (imm8 & 0xff);
      }
    #endif
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_slli_epi32(a, imm8) _mm256_slli_epi32(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm256_slli_epi32(a, imm8) ({ \
    easysimd__m256i r; \
    if (HEDLEY_LIKELY(imm8 >= 0 && imm8 < 32)) { \
      r.m128i[0].neon_i32 = vshlq_n_s32(a.m128i[0].neon_i32, imm8); \
      r.m128i[1].neon_i32 = vshlq_n_s32(a.m128i[1].neon_i32, imm8); \
    } else { \
      r.m128i[0].neon_i32 = vdupq_n_s32(0); \
      r.m128i[1].neon_i32 = vdupq_n_s32(0); \
    } \
    r; \
  })
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_slli_epi32(a, imm8) \
     easysimd_mm256_set_m128i( \
         easysimd_mm_slli_epi32(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
         easysimd_mm_slli_epi32(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_slli_epi32
  #define _mm256_slli_epi32(a, imm8) easysimd_mm256_slli_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_slli_epi64 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    if (imm8 > 63) {
      return easysimd_mm256_setzero_si256();
    } else {
      easysimd_svbool_t pg = svptrue_b64();
      r.sve_i64[EASYSIMD_SV_INDEX_0] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], imm8);
      r.sve_i64[EASYSIMD_SV_INDEX_1] = svlsl_n_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], imm8);
    }
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i64 = a_.i64 << HEDLEY_STATIC_CAST(int64_t, imm8);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] << (imm8 & 0xff);
      }
    #endif
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_slli_epi64(a, imm8) _mm256_slli_epi64(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm256_slli_epi64(a, imm8) ({ \
    easysimd__m256i r; \
    if (HEDLEY_LIKELY(imm8 >= 0 && imm8 < 64)) { \
      r.m128i[0].neon_i64 = vshlq_n_s64(a.m128i[0].neon_i64, imm8); \
      r.m128i[1].neon_i64 = vshlq_n_s64(a.m128i[1].neon_i64, imm8); \
    } else { \
      r.m128i[0].neon_i64 = vdupq_n_s64(0); \
      r.m128i[1].neon_i64 = vdupq_n_s64(0); \
    } \
    r; \
  })
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_slli_epi64(a, imm8) \
     easysimd_mm256_set_m128i( \
         easysimd_mm_slli_epi64(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
         easysimd_mm_slli_epi64(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_slli_epi64
  #define _mm256_slli_epi64(a, imm8) easysimd_mm256_slli_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_slli_si256 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pgm = svwhilele_b8(1, imm8);
    easysimd_svint8_t  svz = svdup_n_s8(0);

    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsplice_s8(pgm, svz, a.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsplice_s8(pgm, svz, a.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif 0 //defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    if (HEDLEY_LIKELY(imm8 > 0 && imm8 <= 15)) {
        r.m128i[0].neon_i8 = vextq_s8(vdupq_n_s8(0), a.m128i[0].neon_i8, 16 - imm8);
        r.m128i[1].neon_i8 = vextq_s8(vdupq_n_s8(0), a.m128i[1].neon_i8, 16 - imm8);
    } else if (imm8 == 0) {
        r = a;
    } else {
        r.m128i[0].neon_i8 = vdupq_n_s8(0);
        r.m128i[1].neon_i8 = vdupq_n_s8(0);
    }
    return r;
  #else
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a);
  for (size_t h = 0 ; h < (sizeof(r_.m128i_private) / sizeof(r_.m128i_private[0])) ; h++) {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.m128i_private[h].i8) / sizeof(r_.m128i_private[h].i8[0])) ; i++) {
      const int e = HEDLEY_STATIC_CAST(int, i) - imm8;
      r_.m128i_private[h].i8[i] = (e >= 0) ? a_.m128i_private[h].i8[e] : 0;
    }
  }
  return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_slli_si256(a, imm8) _mm256_slli_si256(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128) && !defined(__PGI)
#  define easysimd_mm256_slli_si256(a, imm8) \
     easysimd_mm256_set_m128i( \
         easysimd_mm_slli_si128(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
         easysimd_mm_slli_si128(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
#  define easysimd_mm256_slli_si256(a, imm8) \
     easysimd_mm256_set_m128i( \
       easysimd_mm_bslli_si128(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
       easysimd_mm_bslli_si128(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_slli_si256
  #define _mm256_slli_si256(a, imm8) easysimd_mm256_slli_si256(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sllv_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_u32 = vshlq_u32(a.neon_u32, vreinterpretq_s32_u32(b.neon_u32));
    r.neon_u32 = vandq_u32(r.neon_u32, vcltq_u32(b.neon_u32, vdupq_n_u32(32)));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svcmplt_n_u32(svptrue_b32(), b.sve_u32, 32);
    r.sve_u32 = svsel_u32(pg, svlsl_u32_z(svptrue_b32(), a.sve_u32, b.sve_u32), svdup_n_u32(0));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vshlq_u32(a_.neon_u32, vreinterpretq_s32_u32(b_.neon_u32));
      r_.neon_u32 = vandq_u32(r_.neon_u32, vcltq_u32(b_.neon_u32, vdupq_n_u32(32)));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u32), (b_.u32 < UINT32_C(32))) & (a_.u32 << b_.u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = (b_.u32[i] < 32) ? (a_.u32[i] << b_.u32[i]) : 0;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_sllv_epi32(a, b) _mm_sllv_epi32(a, b)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_sllv_epi32
  #define _mm_sllv_epi32(a, b) easysimd_mm_sllv_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sllv_epi32 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private r_;
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    sveuint32_t sva = svlsl_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r_.sve_u32[EASYSIMD_SV_INDEX_0] = svand_n_u32_z(svcmplt_n_u32(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], 32), sva, 0xFFFFFFFF);

    sveuint32_t svb = svlsl_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    r_.sve_u32[EASYSIMD_SV_INDEX_1] = svand_n_u32_z(svcmplt_n_u32(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], 32), svb, 0xFFFFFFFF);
  #else
    easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = (b_.u32[i] < 32) ? (a_.u32[i] << b_.u32[i]) : 0;
    }
  #endif

  return easysimd__m256i_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_sllv_epi32(a, b) _mm256_sllv_epi32(a, b)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sllv_epi32
  #define _mm256_sllv_epi32(a, b) easysimd_mm256_sllv_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sllv_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i r;
    r.neon_u64 = vshlq_u64(a.neon_u64, vreinterpretq_s64_u64(b.neon_u64));
    r.neon_u64 = vandq_u64(r.neon_u64, vcltq_u64(b.neon_u64, vdupq_n_u64(64)));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svcmplt_n_u64(svptrue_b64(), b.sve_u64, 64);
    r.sve_u64 = svsel_u64(pg, svlsl_u64_z(svptrue_b64(), a.sve_u64, b.sve_u64), svdup_n_u64(0));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u64), (b_.u64 < 64)) & (a_.u64 << b_.u64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = (b_.u64[i] < 64) ? (a_.u64[i] << b_.u64[i]) : 0;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_sllv_epi64(a, b) _mm_sllv_epi64(a, b)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_sllv_epi64
  #define _mm_sllv_epi64(a, b) easysimd_mm_sllv_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sllv_epi64 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private r_;
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    sveuint64_t sva = svlsl_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r_.sve_u64[EASYSIMD_SV_INDEX_0] = svand_n_u64_z(svcmplt_n_u64(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], 64), sva, 0xFFFFFFFFFFFFFFFF);

    sveuint64_t svb = svlsl_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    r_.sve_u64[EASYSIMD_SV_INDEX_1] = svand_n_u64_z(svcmplt_n_u64(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], 64), svb, 0xFFFFFFFFFFFFFFFF);
  #else
    easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = (b_.u64[i] < 64) ? (a_.u64[i] << b_.u64[i]) : 0;
    }
  #endif

  return easysimd__m256i_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_sllv_epi64(a, b) _mm256_sllv_epi64(a, b)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sllv_epi64
  #define _mm256_sllv_epi64(a, b) easysimd_mm256_sllv_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sra_epi16 (easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sra_epi16(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    if (shift > 15) {
      shift = 15;
    }
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svasr_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svasr_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sra_epi16(a_.m128i[0], count);
      r_.m128i[1] = easysimd_mm_sra_epi16(a_.m128i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);

      if (shift > 15) shift = 15;

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i16 = a_.i16 >> HEDLEY_STATIC_CAST(int16_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.i16[i] = a_.i16[i] >> shift;
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sra_epi16
  #define _mm256_sra_epi16(a, count) easysimd_mm256_sra_epi16(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sra_epi32 (easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sra_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count.i64[0]);
    if (shift > 31) {
      shift = 31;
    }
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svasr_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svasr_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sra_epi32(a_.m128i[0], count);
      r_.m128i[1] = easysimd_mm_sra_epi32(a_.m128i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);
      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, count_.i64[0]);

      if (shift > 31) shift = 31;

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.i32 = a_.i32 >> HEDLEY_STATIC_CAST(int16_t, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.i32[i] = a_.i32[i] >> shift;
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sra_epi32
  #define _mm256_sra_epi32(a, count) easysimd_mm256_sra_epi32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srai_epi16 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    unsigned int shift = HEDLEY_STATIC_CAST(unsigned int, imm8);
    if (shift > 15) {
      shift = 15;
    }
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svasr_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svasr_n_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    unsigned int shift = HEDLEY_STATIC_CAST(unsigned int, imm8);

    if (shift > 15) shift = 15;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i16 = a_.i16 >> HEDLEY_STATIC_CAST(int16_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i16[i] >> shift;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_srai_epi16(a, imm8) _mm256_srai_epi16(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_srai_epi16(a, imm8) \
     easysimd_mm256_set_m128i( \
         easysimd_mm_srai_epi16(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
         easysimd_mm_srai_epi16(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srai_epi16
  #define _mm256_srai_epi16(a, imm8) easysimd_mm256_srai_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srai_epi32 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    unsigned int shift = HEDLEY_STATIC_CAST(unsigned int, imm8);
    if (shift > 31) {
      shift = 31;
    }
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svasr_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], shift);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svasr_n_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);
    unsigned int shift = HEDLEY_STATIC_CAST(unsigned int, imm8);

    if (shift > 31) shift = 31;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.i32 = a_.i32 >> HEDLEY_STATIC_CAST(int16_t, shift);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] >> shift;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_srai_epi32(a, imm8) _mm256_srai_epi32(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_srai_epi32(a, imm8) \
     easysimd_mm256_set_m128i( \
         easysimd_mm_srai_epi32(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
         easysimd_mm_srai_epi32(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srai_epi32
  #define _mm256_srai_epi32(a, imm8) easysimd_mm256_srai_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srav_epi32 (easysimd__m128i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm_srav_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32 = svasr_s32_z(pg, a.sve_i32, svmin_u32_z(pg, count.sve_u32, svdup_n_u32(31)));
    return r;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      count_ = easysimd__m128i_to_private(count);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      int32x4_t cnt = vreinterpretq_s32_u32(vminq_u32(count_.neon_u32, vdupq_n_u32(31)));
      r_.neon_i32 = vshlq_s32(a_.neon_i32, vnegq_s32(cnt));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        uint32_t shift = HEDLEY_STATIC_CAST(uint32_t, count_.i32[i]);
        r_.i32[i] = a_.i32[i] >> HEDLEY_STATIC_CAST(int, shift > 31 ? 31 : shift);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_srav_epi32
  #define _mm_srav_epi32(a, count) easysimd_mm_srav_epi32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srav_epi32 (easysimd__m256i a, easysimd__m256i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_srav_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    sveuint32_t min;
    min = svmin_u32_z(pg, count.sve_u32[EASYSIMD_SV_INDEX_0], svdup_n_u32(31));
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svasr_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], min);
    min = svmin_u32_z(pg, count.sve_u32[EASYSIMD_SV_INDEX_1], svdup_n_u32(31));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svasr_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], min);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      count_ = easysimd__m256i_to_private(count);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_srav_epi32(a_.m128i[0], count_.m128i[0]);
      r_.m128i[1] = easysimd_mm_srav_epi32(a_.m128i[1], count_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        uint32_t shift = HEDLEY_STATIC_CAST(uint32_t, count_.i32[i]);
        if (shift > 31) shift = 31;
        r_.i32[i] = a_.i32[i] >> shift;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srav_epi32
  #define _mm256_srav_epi32(a, count) easysimd_mm256_srav_epi32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srl_epi16 (easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_srl_epi16(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    uint16_t shift = HEDLEY_STATIC_CAST(uint16_t , (count.i64[0] > 16 ? 16 : count.i64[0]));
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_srl_epi16(a_.m128i[0], count);
      r_.m128i[1] = easysimd_mm_srl_epi16(a_.m128i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t , (count_.i64[0] > 16 ? 16 : count_.i64[0]));

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.u16 = a_.u16 >> EASYSIMD_CAST_VECTOR_SHIFT_COUNT(16, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
          r_.u16[i] = a_.u16[i] >> (shift);
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srl_epi16
  #define _mm256_srl_epi16(a, count) easysimd_mm256_srl_epi16(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srl_epi32 (easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_srl_epi32(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    uint32_t shift = HEDLEY_STATIC_CAST(uint32_t , (count.i64[0] > 32 ? 32 : count.i64[0]));
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], shift);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svlsr_n_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_srl_epi32(a_.m128i[0], count);
      r_.m128i[1] = easysimd_mm_srl_epi32(a_.m128i[1], count);
    #else
      easysimd__m128i_private
        count_ = easysimd__m128i_to_private(count);

      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t , (count_.i64[0] > 32 ? 32 : count_.i64[0]));

      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.u32 = a_.u32 >> EASYSIMD_CAST_VECTOR_SHIFT_COUNT(32, shift);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
          r_.u32[i] = a_.u32[i] >> (shift);
        }
      #endif
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srl_epi32
  #define _mm256_srl_epi32(a, count) easysimd_mm256_srl_epi32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srl_epi64 (easysimd__m256i a, easysimd__m128i count) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_srl_epi64(a, count);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b64();
    uint64_t cnt = HEDLEY_STATIC_CAST(uint64_t, (count.i64[0] > 64 ? 64 : count.i64[0]));
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], cnt);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svlsr_n_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], cnt);
    return r;
  #else
    easysimd__m256i_private r_;
    
    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_srl_epi64(a.m128i[0], count);
      r_.m128i[1] = easysimd_mm_srl_epi64(a.m128i[1], count);
    #else
      easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
      easysimd__m128i_private count_ = easysimd__m128i_to_private(count);
      uint64_t shift = HEDLEY_STATIC_CAST(uint64_t, (count_.i64[0] > 64 ? 64 : count_.i64[0]));
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.u64[i] = a_.u64[i] >> (shift);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srl_epi64
  #define _mm256_srl_epi64(a, count) easysimd_mm256_srl_epi64(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srli_epi16 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    if (imm8 > 15) {
      return easysimd_mm256_setzero_si256();
    }
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    uint8_t shift = HEDLEY_STATIC_CAST(uint8_t, imm8);
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], shift);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svlsr_n_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], shift);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a);

    if (imm8 > 15) {
      return easysimd_mm256_setzero_si256();
    }

    if (HEDLEY_STATIC_CAST(unsigned int, imm8) > 15) {
      easysimd_memset(&r_, 0, sizeof(r_));
    } else {
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
        r_.u16 = a_.u16 >> EASYSIMD_CAST_VECTOR_SHIFT_COUNT(16, imm8);
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
          r_.u16[i] = a_.u16[i] >> imm8;
        }
      #endif
    }
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_srli_epi16(a, imm8) _mm256_srli_epi16(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm256_srli_epi16(a, imm8) ({ \
    easysimd__m256i r; \
    if (imm8 > 15) { \
      r = easysimd_mm256_setzero_si256(); \
    } else {\
      r.m128i[0].neon_u16 = vshrq_n_u16(a.m128i[0].neon_u16, imm8); \
      r.m128i[1].neon_u16 = vshrq_n_u16(a.m128i[1].neon_u16, imm8); \
    } \
    r; \
  })
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_srli_epi16(a, imm8) \
     easysimd_mm256_set_m128i( \
         easysimd_mm_srli_epi16(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
         easysimd_mm_srli_epi16(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srli_epi16
  #define _mm256_srli_epi16(a, imm8) easysimd_mm256_srli_epi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srli_epi32 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svlsr_n_u32_z(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_0], imm8);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svlsr_n_u32_z(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_1], imm8);
    return r;
  #else
    easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] >> imm8;
    }
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_srli_epi32(a, imm8) _mm256_srli_epi32(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm256_srli_epi32(a, imm8) ({ \
    easysimd__m256i r; \
    if (imm8 > 31) { \
      r = easysimd_mm256_setzero_si256(); \
    } else {\
      r.m128i[0].neon_u32 = vshrq_n_u32(a.m128i[0].neon_u32, imm8); \
      r.m128i[1].neon_u32 = vshrq_n_u32(a.m128i[1].neon_u32, imm8); \
    } \
    r; \
  })
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srli_epi32
  #define _mm256_srli_epi32(a, imm8) easysimd_mm256_srli_epi32(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srli_epi64 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  r.sve_u64[EASYSIMD_SV_INDEX_0] = svlsr_n_u64_z(svptrue_b64(), a.sve_u64[EASYSIMD_SV_INDEX_0], imm8);
  r.sve_u64[EASYSIMD_SV_INDEX_1] = svlsr_n_u64_z(svptrue_b64(), a.sve_u64[EASYSIMD_SV_INDEX_1], imm8);
  return r;
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  easysimd__m256i res;
  if (HEDLEY_LIKELY(imm8 >= 0 && imm8 < 64)) {
    int64x2_t vect_imm = vdupq_n_s64(-imm8);
    res.m128i[0].neon_u64 = vshlq_u64(a.m128i[0].neon_u64, vect_imm);
    res.m128i[1].neon_u64 = vshlq_u64(a.m128i[1].neon_u64, vect_imm);
  } else {
    res.m128i[0].neon_u64 = vdupq_n_u64(0);
    res.m128i[1].neon_u64 = vdupq_n_u64(0);
  } 
  return res;
#else
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
    r_.u64 = a_.u64 >> EASYSIMD_CAST_VECTOR_SHIFT_COUNT(32, imm8);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = a_.u64[i] >> imm8;
    }
  #endif
  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_srli_epi64(a, imm8) _mm256_srli_epi64(a, imm8)
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_srli_epi64(a, imm8) \
     easysimd_mm256_set_m128i( \
         easysimd_mm_srli_epi64(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
         easysimd_mm_srli_epi64(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srli_epi64
  #define _mm256_srli_epi64(a, imm8) easysimd_mm256_srli_epi64(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srli_si256 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_0], svindex_u8(imm8, 1));
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svtbl_u8(a.sve_u8[EASYSIMD_SV_INDEX_1], svindex_u8(imm8, 1));
    return r;
  #else
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a);

  for (size_t h = 0 ; h < (sizeof(r_.m128i_private) / sizeof(r_.m128i_private[0])) ; h++) {
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.m128i_private[h].i8) / sizeof(r_.m128i_private[h].i8[0])) ; i++) {
      const int e = imm8 + HEDLEY_STATIC_CAST(int, i);
      r_.m128i_private[h].i8[i] = (e < 16) ? a_.m128i_private[h].i8[e] : 0;
    }
  }

  return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
#  define easysimd_mm256_srli_si256(a, imm8) _mm256_srli_si256(a, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm256_srli_si256(a, imm8) ({ \
    easysimd__m256i r; \
    if (HEDLEY_LIKELY(imm8 > 0 && imm8 <= 15)) { \
        r.m128i[0].neon_i8 = vextq_s8(a.m128i[0].neon_i8, vdupq_n_s8(0), imm8); \
        r.m128i[1].neon_i8 = vextq_s8(a.m128i[1].neon_i8, vdupq_n_s8(0), imm8); \
    } else if (imm8 == 0) { \
        r = a; \
    } else { \
        r.m128i[0].neon_i8 = vdupq_n_s8(0); \
        r.m128i[1].neon_i8 = vdupq_n_s8(0); \
    } \
    r; \
  })
#elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128) && !defined(__PGI)
#  define easysimd_mm256_srli_si256(a, imm8) \
     easysimd_mm256_set_m128i( \
         easysimd_mm_srli_si128(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
         easysimd_mm_srli_si128(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
#  define easysimd_mm256_srli_si256(a, imm8) \
     easysimd_mm256_set_m128i( \
       easysimd_mm_bsrli_si128(easysimd_mm256_extracti128_si256(a, 1), (imm8)), \
       easysimd_mm_bsrli_si128(easysimd_mm256_extracti128_si256(a, 0), (imm8)))
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srli_si256
  #define _mm256_srli_si256(a, imm8) easysimd_mm256_srli_si256(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srlv_epi32 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svcmplt_n_u32(svptrue_b32(), b.sve_u32, 32);
    r.sve_u32 = svsel_u32(pg, svlsr_u32_z(svptrue_b32(), a.sve_u32, b.sve_u32), svdup_n_u32(0));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u32), (b_.u32 < 32)) & (a_.u32 >> b_.u32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = (b_.u32[i] < 32) ? (a_.u32[i] >> b_.u32[i]) : 0;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_srlv_epi32(a, b) _mm_srlv_epi32(a, b)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_srlv_epi32
  #define _mm_srlv_epi32(a, b) easysimd_mm_srlv_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srlv_epi32 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private r_;
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    sveuint32_t sva = svlsr_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r_.sve_u32[EASYSIMD_SV_INDEX_0] = svand_n_u32_z(svcmplt_n_u32(pg, b.sve_u32[EASYSIMD_SV_INDEX_0], 32), sva, 0xFFFFFFFF);

    sveuint32_t svb = svlsr_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    r_.sve_u32[EASYSIMD_SV_INDEX_1] = svand_n_u32_z(svcmplt_n_u32(pg, b.sve_u32[EASYSIMD_SV_INDEX_1], 32), svb, 0xFFFFFFFF);

  #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) 
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    r_.u32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u32), (b_.u32 < 32)) & (a_.u32 >> b_.u32);
  #else
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = (b_.u32[i] < 32) ? (a_.u32[i] >> b_.u32[i]) : 0;
    }
  #endif

  return easysimd__m256i_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_srlv_epi32(a, b) _mm256_srlv_epi32(a, b)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srlv_epi32
  #define _mm256_srlv_epi32(a, b) easysimd_mm256_srlv_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_srlv_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    easysimd_svbool_t pg = svcmplt_n_u64(svptrue_b64(), b.sve_u64, 64);
    r.sve_u64 = svsel_u64(pg, svlsr_u64_z(svptrue_b64(), a.sve_u64, b.sve_u64), svdup_n_u64(0));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b),
      r_;

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.u64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.u64), (b_.u64 < 64)) & (a_.u64 >> b_.u64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = (b_.u64[i] < 64) ? (a_.u64[i] >> b_.u64[i]) : 0;
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm_srlv_epi64(a, b) _mm_srlv_epi64(a, b)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm_srlv_epi64
  #define _mm_srlv_epi64(a, b) easysimd_mm_srlv_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_srlv_epi64 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private r_;
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    sveuint64_t sva = svlsr_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r_.sve_u64[EASYSIMD_SV_INDEX_0] = svand_n_u64_z(svcmplt_n_u64(pg, b.sve_u64[EASYSIMD_SV_INDEX_0], 64), sva, 0xFFFFFFFFFFFFFFFF);

    sveuint64_t svb = svlsr_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    r_.sve_u64[EASYSIMD_SV_INDEX_1] = svand_n_u64_z(svcmplt_n_u64(pg, b.sve_u64[EASYSIMD_SV_INDEX_1], 64), svb, 0xFFFFFFFFFFFFFFFF);
  #else
    easysimd__m256i_private
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
      r_.u64[i] = (b_.u64[i] < 64) ? (a_.u64[i] >> b_.u64[i]) : 0;
    }
  #endif

  return easysimd__m256i_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX2_NATIVE)
  #define easysimd_mm256_srlv_epi64(a, b) _mm256_srlv_epi64(a, b)
#endif
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_srlv_epi64
  #define _mm256_srlv_epi64(a, b) easysimd_mm256_srlv_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_stream_load_si256 (const easysimd__m256i* mem_addr) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_stream_load_si256(HEDLEY_CONST_CAST(easysimd__m256i*, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r_;
    r_.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(svptrue_b32(), (HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr)) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS / 32));
    r_.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(svptrue_b32(), (HEDLEY_REINTERPRET_CAST(int32_t const*, mem_addr)) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS / 32));
    return r_;
  #else
    easysimd__m256i r;
    easysimd_memcpy(&r, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256i), sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
#  define _mm256_stream_load_si256(mem_addr) easysimd_mm256_stream_load_si256(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sub_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sub_epi8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i8 = vsubq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vsubq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svsub_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svsub_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sub_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_sub_epi8(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i8 = a_.i8 - b_.i8;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a_.i8[i] - b_.i8[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sub_epi8
  #define _mm256_sub_epi8(a, b) easysimd_mm256_sub_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sub_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sub_epi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i16 = vsubq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vsubq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsub_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsub_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sub_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_sub_epi16(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i16 = a_.i16 - b_.i16;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a_.i16[i] - b_.i16[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sub_epi16
  #define _mm256_sub_epi16(a, b) easysimd_mm256_sub_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_hsub_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_hsub_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svsub_s16_z(pg, svuzp1_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), svuzp2_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svsub_s16_z(pg, svuzp1_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), svuzp2_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    return easysimd_mm256_sub_epi16(easysimd_x_mm256_deinterleaveeven_epi16(a, b), easysimd_x_mm256_deinterleaveodd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hsub_epi16
  #define _mm256_hsub_epi16(a, b) easysimd_mm256_hsub_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sub_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sub_epi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vsubq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vsubq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsub_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsub_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sub_epi32(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_sub_epi32(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 - b_.i32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] - b_.i32[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sub_epi32
  #define _mm256_sub_epi32(a, b) easysimd_mm256_sub_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_hsub_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_hsub_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svsub_s32_z(pg, svuzp1_s32(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]), svuzp2_s32(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svsub_s32_z(pg, svuzp1_s32(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]), svuzp2_s32(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    return easysimd_mm256_sub_epi32(easysimd_x_mm256_deinterleaveeven_epi32(a, b), easysimd_x_mm256_deinterleaveodd_epi32(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hsub_epi32
  #define _mm256_hsub_epi32(a, b) easysimd_mm256_hsub_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_sub_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_sub_epi64(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i64 = vsubq_s64(a.m128i[0].neon_i64, b.m128i[0].neon_i64);
    r.m128i[1].neon_i64 = vsubq_s64(a.m128i[1].neon_i64, b.m128i[1].neon_i64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svsub_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svsub_s64_z(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_sub_epi64(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_sub_epi64(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = a_.i64 - b_.i64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] - b_.i64[i];
      }
    #endif

  return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sub_epi64
  #define _mm256_sub_epi64(a, b) easysimd_mm256_sub_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_sub_epu32 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    r_.u32 = a_.u32 - b_.u32;
  #elif EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
    r_.m128i[0] = easysimd_x_mm_sub_epu32(a_.m128i[0], b_.m128i[0]);
    r_.m128i[1] = easysimd_x_mm_sub_epu32(a_.m128i[1], b_.m128i[1]);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
      r_.u32[i] = a_.u32[i] - b_.u32[i];
    }
  #endif

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_subs_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_subs_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svqsub_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svqsub_s8_z(pg, a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i8 = vqsubq_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vqsubq_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_subs_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_subs_epi8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = easysimd_math_subs_i8(a_.i8[i], b_.i8[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_subs_epi8
  #define _mm256_subs_epi8(a, b) easysimd_mm256_subs_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_subs_epi16(easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_subs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svqsub_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svqsub_s16_z(pg, a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i16 = vqsubq_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vqsubq_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_subs_epi16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_subs_epi16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = easysimd_math_subs_i16(a_.i16[i], b_.i16[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_subs_epi16
  #define _mm256_subs_epi16(a, b) easysimd_mm256_subs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_hsubs_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_hsubs_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    easysimd_svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svqsub_s16_z(pg, svuzp1_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]), svuzp2_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svqsub_s16_z(pg, svuzp1_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]), svuzp2_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    return easysimd_mm256_subs_epi16(easysimd_x_mm256_deinterleaveeven_epi16(a, b), easysimd_x_mm256_deinterleaveodd_epi16(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hsubs_epi16
  #define _mm256_hsubs_epi16(a, b) easysimd_mm256_hsubs_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_subs_epu8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_subs_epu8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_u8[EASYSIMD_SV_INDEX_0] = svqsub_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_0], b.sve_u8[EASYSIMD_SV_INDEX_0]);
    r.sve_u8[EASYSIMD_SV_INDEX_1] = svqsub_u8_z(pg, a.sve_u8[EASYSIMD_SV_INDEX_1], b.sve_u8[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_u8 = vqsubq_u8(a.m128i[0].neon_u8, b.m128i[0].neon_u8);
    r.m128i[1].neon_u8 = vqsubq_u8(a.m128i[1].neon_u8, b.m128i[1].neon_u8);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_subs_epu8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_subs_epu8(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
        r_.u8[i] = easysimd_math_subs_u8(a_.u8[i], b_.u8[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_subs_epu8
  #define _mm256_subs_epu8(a, b) easysimd_mm256_subs_epu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_subs_epu16(easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_subs_epu16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_u16[EASYSIMD_SV_INDEX_0] = svqsub_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_0], b.sve_u16[EASYSIMD_SV_INDEX_0]);
    r.sve_u16[EASYSIMD_SV_INDEX_1] = svqsub_u16_z(pg, a.sve_u16[EASYSIMD_SV_INDEX_1], b.sve_u16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_u16 = vqsubq_u16(a.m128i[0].neon_u16, b.m128i[0].neon_u16);
    r.m128i[1].neon_u16 = vqsubq_u16(a.m128i[1].neon_u16, b.m128i[1].neon_u16);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_subs_epu16(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_subs_epu16(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
        r_.u16[i] = easysimd_math_subs_u16(a_.u16[i], b_.u16[i]);
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_subs_epu16
  #define _mm256_subs_epu16(a, b) easysimd_mm256_subs_epu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_x_mm256_test_all_ones (easysimd__m256i a) {
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
  int r;
  int_fast32_t r_ = ~HEDLEY_STATIC_CAST(int_fast32_t, 0);

  EASYSIMD_VECTORIZE_REDUCTION(&:r_)
  for (size_t i = 0 ; i < (sizeof(a_.i32f) / sizeof(a_.i32f[0])) ; i++) {
    r_ &= a_.i32f[i];
  }

  r = (r_ == ~HEDLEY_STATIC_CAST(int_fast32_t, 0));

  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_unpacklo_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_unpacklo_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svzip1_s8(a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svzip1_s8(a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i8 = vzip1q_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vzip1q_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_unpacklo_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_unpacklo_epi8(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.i8 = EASYSIMD_SHUFFLE_VECTOR_(8, 32, a_.i8, b_.i8,
           0, 32,  1, 33,  2, 34,  3, 35,
           4, 36,  5, 37,  6, 38,  7, 39,
          16, 48, 17, 49, 18, 50, 19, 51,
          20, 52, 21, 53, 22, 54, 23, 55);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0]) / 2) ; i++) {
        r_.i8[2 * i] = a_.i8[i + ~(~i | 7)];
        r_.i8[2 * i + 1] = b_.i8[i + ~(~i | 7)];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpacklo_epi8
  #define _mm256_unpacklo_epi8(a, b) easysimd_mm256_unpacklo_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_unpacklo_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_unpacklo_epi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i16 = vzip1q_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vzip1q_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svzip1_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svzip1_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
      easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0]) / 2) ; i++) {
        r_.i16[2 * i] = a_.i16[i + ~(~i | 3)];
        r_.i16[2 * i + 1] = b_.i16[i + ~(~i | 3)];
      }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpacklo_epi16
  #define _mm256_unpacklo_epi16(a, b) easysimd_mm256_unpacklo_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_unpacklo_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_unpacklo_epi32(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vzip1q_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vzip1q_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svzip1_s32(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svzip1_s32(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
      easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0]) / 2) ; i++) {
        r_.i32[2 * i] = a_.i32[i + ~(~i | 1)];
        r_.i32[2 * i + 1] = b_.i32[i + ~(~i | 1)];
      }
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpacklo_epi32
  #define _mm256_unpacklo_epi32(a, b) easysimd_mm256_unpacklo_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_unpacklo_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_unpacklo_epi64(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i64 = vzip1q_s64(a.m128i[0].neon_i64, b.m128i[0].neon_i64);
    r.m128i[1].neon_i64 = vzip1q_s64(a.m128i[1].neon_i64, b.m128i[1].neon_i64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svzip1_s64(a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svzip1_s64(a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_unpacklo_epi64(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_unpacklo_epi64(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.i64 = EASYSIMD_SHUFFLE_VECTOR_(64, 32, a_.i64, b_.i64, 0, 4, 2, 6);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0]) / 2) ; i++) {
        r_.i64[2 * i] = a_.i64[2 * i];
        r_.i64[2 * i + 1] = b_.i64[2 * i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpacklo_epi64
  #define _mm256_unpacklo_epi64(a, b) easysimd_mm256_unpacklo_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_unpackhi_epi8 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_unpackhi_epi8(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svzip2_s8(a.sve_i8[EASYSIMD_SV_INDEX_0], b.sve_i8[EASYSIMD_SV_INDEX_0]);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svzip2_s8(a.sve_i8[EASYSIMD_SV_INDEX_1], b.sve_i8[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i8 = vzip2q_s8(a.m128i[0].neon_i8, b.m128i[0].neon_i8);
    r.m128i[1].neon_i8 = vzip2q_s8(a.m128i[1].neon_i8, b.m128i[1].neon_i8);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_unpackhi_epi8(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_unpackhi_epi8(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.i8 = EASYSIMD_SHUFFLE_VECTOR_(8, 32, a_.i8, b_.i8,
           8, 40,  9, 41, 10, 42, 11, 43,
          12, 44, 13, 45, 14, 46, 15, 47,
          24, 56, 25, 57, 26, 58, 27, 59,
          28, 60, 29, 61, 30, 62, 31, 63);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0]) / 2) ; i++) {
        r_.i8[2 * i] = a_.i8[i + 8 + ~(~i | 7)];
        r_.i8[2 * i + 1] = b_.i8[i + 8 + ~(~i | 7)];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpackhi_epi8
  #define _mm256_unpackhi_epi8(a, b) easysimd_mm256_unpackhi_epi8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_unpackhi_epi16 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_unpackhi_epi16(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svzip2_s16(a.sve_i16[EASYSIMD_SV_INDEX_0], b.sve_i16[EASYSIMD_SV_INDEX_0]);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svzip2_s16(a.sve_i16[EASYSIMD_SV_INDEX_1], b.sve_i16[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i16 = vzip2q_s16(a.m128i[0].neon_i16, b.m128i[0].neon_i16);
    r.m128i[1].neon_i16 = vzip2q_s16(a.m128i[1].neon_i16, b.m128i[1].neon_i16);
    return r;
  #else
    easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0]) / 2) ; i++) {
      r_.i16[2 * i] = a_.i16[i + 4 + ~(~i | 3)];
      r_.i16[2 * i + 1] = b_.i16[i + 4 + ~(~i | 3)];
    }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpackhi_epi16
  #define _mm256_unpackhi_epi16(a, b) easysimd_mm256_unpackhi_epi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_unpackhi_epi32 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_unpackhi_epi32(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svzip2_s32(a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svzip2_s32(a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vzip2q_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = vzip2q_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0]) / 2) ; i++) {
        r_.i32[2 * i] = a_.i32[i + 2 + ~(~i | 1)];
        r_.i32[2 * i + 1] = b_.i32[i + 2 + ~(~i | 1)];
      }

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpackhi_epi32
  #define _mm256_unpackhi_epi32(a, b) easysimd_mm256_unpackhi_epi32(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_unpackhi_epi64 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_unpackhi_epi64(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i64 = vzip2q_s64(a.m128i[0].neon_i64, b.m128i[0].neon_i64);
    r.m128i[1].neon_i64 = vzip2q_s64(a.m128i[1].neon_i64, b.m128i[1].neon_i64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svzip2_s64(a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svzip2_s64(a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_unpackhi_epi64(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_unpackhi_epi64(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.i64 = EASYSIMD_SHUFFLE_VECTOR_(64, 32, a_.i64, b_.i64, 1, 5, 3, 7);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0]) / 2) ; i++) {
        r_.i64[2 * i] = a_.i64[2 * i + 1];
        r_.i64[2 * i + 1] = b_.i64[2 * i + 1];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpackhi_epi64
  #define _mm256_unpackhi_epi64(a, b) easysimd_mm256_unpackhi_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_xor_si256 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return _mm256_xor_si256(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = sveor_s32_z(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = sveor_s32_z(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = veorq_s32(a.m128i[0].neon_i32, b.m128i[0].neon_i32);
    r.m128i[1].neon_i32 = veorq_s32(a.m128i[1].neon_i32, b.m128i[1].neon_i32);
    return r;
  #else
    easysimd__m256i_private
      r_,
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_xor_si128(a_.m128i[0], b_.m128i[0]);
      r_.m128i[1] = easysimd_mm_xor_si128(a_.m128i[1], b_.m128i[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f ^ b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] ^ b_.i64[i];
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES)
  #undef _mm256_xor_si256
  #define _mm256_xor_si256(a, b) easysimd_mm256_xor_si256(a, b)
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX2_H) */
