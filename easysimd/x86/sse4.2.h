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
 *   2017      Evan Nemerson <evan@nemerson.com>
 *   2020      Hidayat Khan <huk2209@gmail.com>
 */

#if !defined(EASYSIMD_X86_SSE4_2_H)
#define EASYSIMD_X86_SSE4_2_H

#include "sse4.1.h"

#if defined(__ARM_ACLE) || (defined(__GNUC__) && defined(__ARM_FEATURE_CRC32))
  #include <arm_acle.h>
#endif
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if defined(EASYSIMD_X86_SSE4_2_NATIVE)
  #define EASYSIMD_SIDD_UBYTE_OPS _SIDD_UBYTE_OPS
  #define EASYSIMD_SIDD_UWORD_OPS _SIDD_UWORD_OPS
  #define EASYSIMD_SIDD_SBYTE_OPS _SIDD_SBYTE_OPS
  #define EASYSIMD_SIDD_SWORD_OPS _SIDD_SWORD_OPS
  #define EASYSIMD_SIDD_CMP_EQUAL_ANY _SIDD_CMP_EQUAL_ANY
  #define EASYSIMD_SIDD_CMP_RANGES _SIDD_CMP_RANGES
  #define EASYSIMD_SIDD_CMP_EQUAL_EACH _SIDD_CMP_EQUAL_EACH
  #define EASYSIMD_SIDD_CMP_EQUAL_ORDERED _SIDD_CMP_EQUAL_ORDERED
  #define EASYSIMD_SIDD_POSITIVE_POLARITY _SIDD_POSITIVE_POLARITY
  #define EASYSIMD_SIDD_NEGATIVE_POLARITY _SIDD_NEGATIVE_POLARITY
  #define EASYSIMD_SIDD_MASKED_POSITIVE_POLARITY _SIDD_MASKED_POSITIVE_POLARITY
  #define EASYSIMD_SIDD_MASKED_NEGATIVE_POLARITY _SIDD_MASKED_NEGATIVE_POLARITY
  #define EASYSIMD_SIDD_LEAST_SIGNIFICANT _SIDD_LEAST_SIGNIFICANT
  #define EASYSIMD_SIDD_MOST_SIGNIFICANT _SIDD_MOST_SIGNIFICANT
  #define EASYSIMD_SIDD_BIT_MASK _SIDD_BIT_MASK
  #define EASYSIMD_SIDD_UNIT_MASK _SIDD_UNIT_MASK
#else
  #define EASYSIMD_SIDD_UBYTE_OPS 0x00
  #define EASYSIMD_SIDD_UWORD_OPS 0x01
  #define EASYSIMD_SIDD_SBYTE_OPS 0x02
  #define EASYSIMD_SIDD_SWORD_OPS 0x03
  #define EASYSIMD_SIDD_CMP_EQUAL_ANY 0x00
  #define EASYSIMD_SIDD_CMP_RANGES 0x04
  #define EASYSIMD_SIDD_CMP_EQUAL_EACH 0x08
  #define EASYSIMD_SIDD_CMP_EQUAL_ORDERED 0x0c
  #define EASYSIMD_SIDD_POSITIVE_POLARITY 0x00
  #define EASYSIMD_SIDD_NEGATIVE_POLARITY 0x10
  #define EASYSIMD_SIDD_MASKED_POSITIVE_POLARITY 0x20
  #define EASYSIMD_SIDD_MASKED_NEGATIVE_POLARITY 0x30
  #define EASYSIMD_SIDD_LEAST_SIGNIFICANT 0x00
  #define EASYSIMD_SIDD_MOST_SIGNIFICANT 0x40
  #define EASYSIMD_SIDD_BIT_MASK 0x00
  #define EASYSIMD_SIDD_UNIT_MASK 0x40
#endif

#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES) && !defined(_SIDD_UBYTE_OPS)
  #define _SIDD_UBYTE_OPS EASYSIMD_SIDD_UBYTE_OPS
  #define _SIDD_UWORD_OPS EASYSIMD_SIDD_UWORD_OPS
  #define _SIDD_SBYTE_OPS EASYSIMD_SIDD_SBYTE_OPS
  #define _SIDD_SWORD_OPS EASYSIMD_SIDD_SWORD_OPS
  #define _SIDD_CMP_EQUAL_ANY EASYSIMD_SIDD_CMP_EQUAL_ANY
  #define _SIDD_CMP_RANGES EASYSIMD_SIDD_CMP_RANGES
  #define _SIDD_CMP_EQUAL_EACH EASYSIMD_SIDD_CMP_EQUAL_EACH
  #define _SIDD_CMP_EQUAL_ORDERED EASYSIMD_SIDD_CMP_EQUAL_ORDERED
  #define _SIDD_POSITIVE_POLARITY EASYSIMD_SIDD_POSITIVE_POLARITY
  #define _SIDD_NEGATIVE_POLARITY EASYSIMD_SIDD_NEGATIVE_POLARITY
  #define _SIDD_MASKED_POSITIVE_POLARITY EASYSIMD_SIDD_MASKED_POSITIVE_POLARITY
  #define _SIDD_MASKED_NEGATIVE_POLARITY EASYSIMD_SIDD_MASKED_NEGATIVE_POLARITY
  #define _SIDD_LEAST_SIGNIFICANT EASYSIMD_SIDD_LEAST_SIGNIFICANT
  #define _SIDD_MOST_SIGNIFICANT EASYSIMD_SIDD_MOST_SIGNIFICANT
  #define _SIDD_BIT_MASK EASYSIMD_SIDD_BIT_MASK
  #define _SIDD_UNIT_MASK EASYSIMD_SIDD_UNIT_MASK
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_mm_cmpestrs (easysimd__m128i a, int la, easysimd__m128i b, int lb, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if !defined(HEDLEY_PGI_VERSION)
    /* https://www.pgroup.com/userforum/viewtopic.php?f=4&p=27590&sid=cf89f8bf30be801831fe4a2ff0a2fa6c */
    (void) a;
    (void) b;
  #endif
  (void) la;
  (void) lb;
  return la <= ((128 / ((imm8 & EASYSIMD_SIDD_UWORD_OPS) ? 16 : 8)) - 1);
}
#if defined(EASYSIMD_X86_SSE4_2_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm_cmpestrs(a, la, b, lb, imm8) \
      _mm_cmpestrs( \
        HEDLEY_REINTERPRET_CAST(__v16qi, a), la, \
        HEDLEY_REINTERPRET_CAST(__v16qi, b), lb, \
        imm8)
  #else
    #define easysimd_mm_cmpestrs(a, la, b, lb, imm8) _mm_cmpestrs(a, la, b, lb, imm8)
  #endif
#endif
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpestrs
  #define _mm_cmpestrs(a, la, b, lb, imm8) easysimd_mm_cmpestrs(a, la, b, lb, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_mm_cmpestrz (easysimd__m128i a, int la, easysimd__m128i b, int lb, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if !defined(HEDLEY_PGI_VERSION)
    /* https://www.pgroup.com/userforum/viewtopic.php?f=4&p=27590&sid=cf89f8bf30be801831fe4a2ff0a2fa6c */
    (void) a;
    (void) b;
  #endif
  (void) la;
  (void) lb;
  return lb <= ((128 / ((imm8 & EASYSIMD_SIDD_UWORD_OPS) ? 16 : 8)) - 1);
}
#if defined(EASYSIMD_X86_SSE4_2_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm_cmpestrz(a, la, b, lb, imm8) \
      _mm_cmpestrz( \
        HEDLEY_REINTERPRET_CAST(__v16qi, a), la, \
        HEDLEY_REINTERPRET_CAST(__v16qi, b), lb, \
        imm8)
  #else
    #define easysimd_mm_cmpestrz(a, la, b, lb, imm8) _mm_cmpestrz(a, la, b, lb, imm8)
  #endif
#endif
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpestrz
  #define _mm_cmpestrz(a, la, b, lb, imm8) easysimd_mm_cmpestrz(a, la, b, lb, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpgt_epi64 (easysimd__m128i a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_SSE4_2_NATIVE)
    return _mm_cmpgt_epi64(a, b);
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    /* https://stackoverflow.com/a/65175746/501126 */
    __m128i r = _mm_and_si128(_mm_cmpeq_epi32(a, b), _mm_sub_epi64(b, a));
    r = _mm_or_si128(r, _mm_cmpgt_epi32(a, b));
    return _mm_shuffle_epi32(r, _MM_SHUFFLE(3, 3, 1, 1));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b64();
    r.sve_u64 = svdup_n_u64_z(svcmpgt_s64(pg, a.sve_i64, b.sve_i64), ~UINT64_C(0));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_u64 = vcgtq_s64(a.neon_i64, b.neon_i64);
    return res;
  #else
    easysimd__m128i_private
      r_,
      a_ = easysimd__m128i_to_private(a),
      b_ = easysimd__m128i_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      /* https://stackoverflow.com/a/65223269/501126 */
      r_.neon_i64 = vshrq_n_s64(vqsubq_s64(b_.neon_i64, a_.neon_i64), 63);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), a_.i64 > b_.i64);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = (a_.i64[i] > b_.i64[i]) ? ~INT64_C(0) : INT64_C(0);
      }
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpgt_epi64
  #define _mm_cmpgt_epi64(a, b) easysimd_mm_cmpgt_epi64(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_cmpistrs_8_(easysimd__m128i a) {
  easysimd__m128i_private a_= easysimd__m128i_to_private(a);
  const int upper_bound = (128 / 8) - 1;
  int a_invalid = 0;
  EASYSIMD_VECTORIZE
  for (int i = 0 ; i <= upper_bound ; i++) {
    if(!a_.i8[i])
      a_invalid = 1;
  }
  return a_invalid;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_cmpistrs_16_(easysimd__m128i a) {
  easysimd__m128i_private a_= easysimd__m128i_to_private(a);
  const int upper_bound = (128 / 16) - 1;
  int a_invalid = 0;
  EASYSIMD_VECTORIZE
  for (int i = 0 ; i <= upper_bound ; i++) {
    if(!a_.i16[i])
      a_invalid = 1;
  }
  return a_invalid;
}

#if defined(EASYSIMD_X86_SSE4_2_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm_cmpistrs(a, b, imm8) \
      _mm_cmpistrs( \
        HEDLEY_REINTERPRET_CAST(__v16qi, a), \
        HEDLEY_REINTERPRET_CAST(__v16qi, b), \
        imm8)
  #else
    #define easysimd_mm_cmpistrs(a, b, imm8) _mm_cmpistrs(a, b, imm8)
  #endif
#else
  #define easysimd_mm_cmpistrs(a, b, imm8) \
     (((imm8) & EASYSIMD_SIDD_UWORD_OPS) \
       ? easysimd_mm_cmpistrs_16_((a)) \
       : easysimd_mm_cmpistrs_8_((a)))
#endif
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpistrs
  #define _mm_cmpistrs(a, b, imm8) easysimd_mm_cmpistrs(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_cmpistrz_8_(easysimd__m128i b) {
  easysimd__m128i_private b_= easysimd__m128i_to_private(b);
  const int upper_bound = (128 / 8) - 1;
  int b_invalid = 0;
  EASYSIMD_VECTORIZE
  for (int i = 0 ; i <= upper_bound ; i++) {
    if(!b_.i8[i])
      b_invalid = 1;
  }
  return b_invalid;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_cmpistrz_16_(easysimd__m128i b) {
  easysimd__m128i_private b_= easysimd__m128i_to_private(b);
  const int upper_bound = (128 / 16) - 1;
  int b_invalid = 0;
  EASYSIMD_VECTORIZE
  for (int i = 0 ; i <= upper_bound ; i++) {
    if(!b_.i16[i])
      b_invalid = 1;
  }
  return b_invalid;
}

#if defined(EASYSIMD_X86_SSE4_2_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
    #define easysimd_mm_cmpistrz(a, b, imm8) \
      _mm_cmpistrz( \
        HEDLEY_REINTERPRET_CAST(__v16qi, a), \
        HEDLEY_REINTERPRET_CAST(__v16qi, b), \
        imm8)
  #else
    #define easysimd_mm_cmpistrz(a, b, imm8) _mm_cmpistrz(a, b, imm8)
  #endif
#else
  #define easysimd_mm_cmpistrz(a, b, imm8) \
     (((imm8) & EASYSIMD_SIDD_UWORD_OPS) \
       ? easysimd_mm_cmpistrz_16_((b)) \
       : easysimd_mm_cmpistrz_8_((b)))
#endif
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpistrz
  #define _mm_cmpistrz(a, b, imm8) easysimd_mm_cmpistrz(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_mm_crc32_u8(uint32_t prevcrc, uint8_t v) {
  #if defined(EASYSIMD_X86_SSE4_2_NATIVE)
    return _mm_crc32_u8(prevcrc, v);
  #else
    #if defined(EASYSIMD_ARM_SVE_NATIVE) && defined(__ARM_FEATURE_CRC32)
      __asm__ __volatile__(
        "crc32cb %w[c], %w[c], %w[v]\n\t" 
        : [c] "+r"(prevcrc) 
        : [v] "r"(v)
        );

      return prevcrc;
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_CRC32)
      return __crc32cb(prevcrc, v);
    #else
      uint32_t crc = prevcrc;
      crc ^= v;
      for(int bit = 0 ; bit < 8 ; bit++) {
        if (crc & 1)
          crc = (crc >> 1) ^ UINT32_C(0x82f63b78);
        else
          crc = (crc >> 1);
      }
      return crc;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #define _mm_crc32_u8(prevcrc, v) easysimd_mm_crc32_u8(prevcrc, v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_mm_crc32_u16(uint32_t prevcrc, uint16_t v) {
  #if defined(EASYSIMD_X86_SSE4_2_NATIVE)
    return _mm_crc32_u16(prevcrc, v);
  #else
    #if defined(EASYSIMD_ARM_SVE_NATIVE) && defined(__ARM_FEATURE_CRC32)
      __asm__ __volatile__(
        "crc32ch %w[c], %w[c], %w[v]\n\t" 
        : [c] "+r"(prevcrc) 
        : [v] "r"(v)
        );

      return prevcrc;
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_CRC32)
      return __crc32ch(prevcrc, v);
    #else
      uint32_t crc = prevcrc;
      crc = easysimd_mm_crc32_u8(crc, v & 0xff);
      crc = easysimd_mm_crc32_u8(crc, (v >> 8) & 0xff);
      return crc;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #define _mm_crc32_u16(prevcrc, v) easysimd_mm_crc32_u16(prevcrc, v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_mm_crc32_u32(uint32_t prevcrc, uint32_t v) {
  #if defined(EASYSIMD_X86_SSE4_2_NATIVE)
    return _mm_crc32_u32(prevcrc, v);
  #else
    #if defined(EASYSIMD_ARM_SVE_NATIVE) && defined(__ARM_FEATURE_CRC32)
      __asm__ __volatile__(
        "crc32cw %w[c], %w[c], %w[v]\n\t" 
        : [c] "+r"(prevcrc) 
        : [v] "r"(v)
        );

      return prevcrc;
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_CRC32)
      return __crc32cw(prevcrc, v);
    #else
      uint32_t crc = prevcrc;
      crc = easysimd_mm_crc32_u16(crc, v & 0xffff);
      crc = easysimd_mm_crc32_u16(crc, (v >> 16) & 0xffff);
      return crc;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #define _mm_crc32_u32(prevcrc, v) easysimd_mm_crc32_u32(prevcrc, v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_mm_crc32_u64(uint64_t prevcrc, uint64_t v) {
  #if defined(EASYSIMD_X86_SSE4_2_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
    return _mm_crc32_u64(prevcrc, v);
  #else
    #if defined(EASYSIMD_ARM_SVE_NATIVE) && defined(__ARM_FEATURE_CRC32)
      __asm__ __volatile__(
        "crc32cx %w[c], %w[c], %x[v]\n\t" 
        : [c] "+r"(prevcrc) 
        : [v] "r"(v)
        );

      return prevcrc;
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && defined(__ARM_FEATURE_CRC32)
      return __crc32cd(HEDLEY_STATIC_CAST(uint32_t, prevcrc), v);
    #else
      uint64_t crc = prevcrc;
      crc = easysimd_mm_crc32_u32(HEDLEY_STATIC_CAST(uint32_t, crc), v & 0xffffffff);
      crc = easysimd_mm_crc32_u32(HEDLEY_STATIC_CAST(uint32_t, crc), (v >> 32) & 0xffffffff);
      return crc;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
  #define _mm_crc32_u64(prevcrc, v) easysimd_mm_crc32_u64(prevcrc, v)
#endif

#if (!defined(__clang__))

#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)

static uint16_t mask_epi16[8] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
static uint8_t mask_epi8[16] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 
                                                             0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

#define _SIDD_NEON_UBYTE_OPS 0x00  // unsigned 8-bit characters
#define _SIDD_NEON_UWORD_OPS 0x01  // unsigned 16-bit characters
#define _SIDD_NEON_SBYTE_OPS 0x02  // signed 8-bit characters
#define _SIDD_NEON_SWORD_OPS 0x03  // signed 16-bit characters

#define _SIDD_NEON_CMP_EQUAL_ANY 0x00      // compare equal any
#define _SIDD_NEON_CMP_RANGES 0x04         // compare ranges
#define _SIDD_NEON_CMP_EQUAL_EACH 0x08     // compare equal each
#define _SIDD_NEON_CMP_EQUAL_ORDERED 0x0C  // compare equal ordered

#define _SIDD_NEON_POSITIVE_POLARITY 0x00
#define _SIDD_NEON_MASKED_POSITIVE_POLARITY 0x20
#define _SIDD_NEON_NEGATIVE_POLARITY 0x10         // negate results
#define _SIDD_NEON_MASKED_NEGATIVE_POLARITY 0x30  // negate results only before end of string

#define _SIDD_NEON_LEAST_SIGNIFICANT 0x00  // index only: return last significant bit
#define _SIDD_NEON_MOST_SIGNIFICANT 0x40   // index only: return most significant bit

#define _SIDD_NEON_BIT_MASK 0x00   // mask only: return bit mask
#define _SIDD_NEON_UNIT_MASK 0x40  // mask only: return byte/word mask

#define EASYSIMD_PCMPSTR_EQ_16x8(a, b, mtx)                                                                          \
    {                                                                                                       \
        mtx[0].neon_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 0)), a.neon_u16);                \
        mtx[1].neon_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 1)), a.neon_u16);                \
        mtx[2].neon_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 2)), a.neon_u16);                \
        mtx[3].neon_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 3)), a.neon_u16);                \
        mtx[4].neon_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 4)), a.neon_u16);                \
        mtx[5].neon_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 5)), a.neon_u16);                \
        mtx[6].neon_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 6)), a.neon_u16);                \
        mtx[7].neon_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 7)), a.neon_u16);                \
    }

#define EASYSIMD_PCMPSTR_EQ_8x16(a, b, mtx)                                                                          \
    {                                                                                                       \
        mtx[ 0].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  0)), a.neon_u8);                      \
        mtx[ 1].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  1)), a.neon_u8);                      \
        mtx[ 2].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  2)), a.neon_u8);                      \
        mtx[ 3].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  3)), a.neon_u8);                      \
        mtx[ 4].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  4)), a.neon_u8);                      \
        mtx[ 5].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  5)), a.neon_u8);                      \
        mtx[ 6].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  6)), a.neon_u8);                      \
        mtx[ 7].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  7)), a.neon_u8);                      \
        mtx[ 8].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  8)), a.neon_u8);                      \
        mtx[ 9].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  9)), a.neon_u8);                      \
        mtx[10].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 10)), a.neon_u8);                    \
        mtx[11].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 11)), a.neon_u8);                    \
        mtx[12].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 12)), a.neon_u8);                    \
        mtx[13].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 13)), a.neon_u8);                    \
        mtx[14].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 14)), a.neon_u8);                    \
        mtx[15].neon_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 15)), a.neon_u8);                    \
    }

#define EASYSIMD_PCMPSTR_RNG_U16x8(a, b, mtx)                                                                                \
    {                                                                                                               \
        uint16x8_t vect_b[8];                                                                                       \
        easysimd__m128i mask;                                                                                               \
        mask.neon_u32 = vdupq_n_u32(0xffff);                                                                        \
        vect_b[0] = vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 0));                                                     \
        vect_b[1] = vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 1));                                                     \
        vect_b[2] = vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 2));                                                     \
        vect_b[3] = vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 3));                                                     \
        vect_b[4] = vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 4));                                                     \
        vect_b[5] = vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 5));                                                     \
        vect_b[6] = vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 6));                                                     \
        vect_b[7] = vdupq_n_u16(vgetq_lane_u16(b.neon_u16, 7));                                                     \
        int i;                                                                                                      \
        for (i = 0; i < 8; i++) {                                                                                   \
            mtx[i].neon_u16 = vbslq_u16(mask.neon_u16, vcgeq_u16(vect_b[i], a.neon_u16),                            \
            vcleq_u16(vect_b[i], a.neon_u16));                                                                      \
        }                                                                                                           \
    }
#define EASYSIMD_PCMPSTR_RNG_S16x8(a, b, mtx)                                                                                \
    {                                                                                                               \
        int16x8_t vect_b[8];                                                                                        \
        easysimd__m128i mask;                                                                                               \
        mask.neon_u32 = vdupq_n_u32(0xffff);                                                                        \
        vect_b[0] = vdupq_n_s16(vgetq_lane_s16(b.neon_i16, 0));                                                     \
        vect_b[1] = vdupq_n_s16(vgetq_lane_s16(b.neon_i16, 1));                                                     \
        vect_b[2] = vdupq_n_s16(vgetq_lane_s16(b.neon_i16, 2));                                                     \
        vect_b[3] = vdupq_n_s16(vgetq_lane_s16(b.neon_i16, 3));                                                     \
        vect_b[4] = vdupq_n_s16(vgetq_lane_s16(b.neon_i16, 4));                                                     \
        vect_b[5] = vdupq_n_s16(vgetq_lane_s16(b.neon_i16, 5));                                                     \
        vect_b[6] = vdupq_n_s16(vgetq_lane_s16(b.neon_i16, 6));                                                     \
        vect_b[7] = vdupq_n_s16(vgetq_lane_s16(b.neon_i16, 7));                                                     \
        int i;                                                                                                      \
        for (i = 0; i < 8; i++) {                                                                                   \
            mtx[i].neon_u16 = vbslq_u16(mask.neon_u16, vcgeq_s16(vect_b[i], a.neon_i16),                            \
            vcleq_s16(vect_b[i], a.neon_i16));                                                                      \
        }                                                                                                           \
    }

#define EASYSIMD_PCMPSTR_RNG_U8x16(a, b, mtx)                                                                                \
    {                                                                                                               \
        uint8x16_t vect_b[16];                                                                                      \
        easysimd__m128i mask;                                                                                               \
        mask.neon_u16 = vdupq_n_u16(0xff);                                                                          \
        vect_b[ 0] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  0));                                                        \
        vect_b[ 1] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  1));                                                        \
        vect_b[ 2] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  2));                                                        \
        vect_b[ 3] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  3));                                                        \
        vect_b[ 4] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  4));                                                        \
        vect_b[ 5] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  5));                                                        \
        vect_b[ 6] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  6));                                                        \
        vect_b[ 7] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  7));                                                        \
        vect_b[ 8] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  8));                                                        \
        vect_b[ 9] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8,  9));                                                      \
        vect_b[10] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 10));                                                      \
        vect_b[11] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 11));                                                      \
        vect_b[12] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 12));                                                      \
        vect_b[13] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 13));                                                      \
        vect_b[14] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 14));                                                      \
        vect_b[15] = vdupq_n_u8(vgetq_lane_u8(b.neon_u8, 15));                                                      \
        int i;                                                                                                      \
        for (i = 0; i < 16; i++) {                                                                                  \
            mtx[i].neon_u8 = vbslq_u8(mask.neon_u8, vcgeq_u8(vect_b[i], a.neon_u8), vcleq_u8(vect_b[i], a.neon_u8));\
        }                                                                                                           \
    }

#define EASYSIMD_PCMPSTR_RNG_S8x16(a, b, mtx)                                                                                \
    {                                                                                                               \
        int8x16_t vect_b[16];                                                                                       \
        easysimd__m128i mask;                                                                                               \
        mask.neon_u16 = vdupq_n_u16(0xff);                                                                          \
        vect_b[0] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 0));                                                        \
        vect_b[1] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 1));                                                        \
        vect_b[2] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 2));                                                        \
        vect_b[3] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 3));                                                        \
        vect_b[4] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 4));                                                        \
        vect_b[5] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 5));                                                        \
        vect_b[6] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 6));                                                        \
        vect_b[7] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 7));                                                        \
        vect_b[8] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 8));                                                        \
        vect_b[9] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 9));                                                        \
        vect_b[10] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 10));                                                      \
        vect_b[11] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 11));                                                      \
        vect_b[12] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 12));                                                      \
        vect_b[13] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 13));                                                      \
        vect_b[14] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 14));                                                      \
        vect_b[15] = vdupq_n_s8(vgetq_lane_s8(b.neon_i8, 15));                                                      \
        int i;                                                                                                      \
        for (i = 0; i < 16; i++) {                                                                                  \
            mtx[i].neon_u8 = vbslq_u8(mask.neon_u8, vcgeq_s8(vect_b[i], a.neon_i8), vcleq_s8(vect_b[i], a.neon_i8));\
        }                                                                                                           \
    }


EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_agg_equal_any_8x16(int la, int lb, easysimd__m128i mtx[16])
{
    int res = 0;
    int j;
    int m = (1 << la) - 1;
    uint8x8_t vect_mask = vld1_u8(mask_epi8);
    uint8x8_t t_lo = vtst_u8(vdup_n_u8(m & 0xff), vect_mask);
    uint8x8_t t_hi = vtst_u8(vdup_n_u8(m >> 8), vect_mask);
    uint8x16_t vect = vcombine_u8(t_lo, t_hi);
    for (j = 0; j < lb; j++) {
        mtx[j].neon_u8 = vandq_u8(vect, mtx[j].neon_u8);
        mtx[j].neon_u8 = vshrq_n_u8(mtx[j].neon_u8, 7);
        int tmp = vaddvq_u8(mtx[j].neon_u8) ? 1 : 0;
        res |= ( tmp << j);
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_agg_equal_any_16x8(int la, int lb, easysimd__m128i mtx[16])
{
    int res = 0;
    int j;
    int m = (1 << la) - 1;
    uint16x8_t vect = vtstq_u16(vdupq_n_u16(m), vld1q_u16(mask_epi16));
    for (j = 0; j < lb; j++) {
        mtx[j].neon_u16 = vandq_u16(vect, mtx[j].neon_u16);
        mtx[j].neon_u16 = vshrq_n_u16(mtx[j].neon_u16, 15);
        int tmp = vaddvq_u16(mtx[j].neon_u16) ? 1 : 0;
        res |= (tmp << j);
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_byte_equal_any(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    easysimd__m128i mtx[16];
    EASYSIMD_PCMPSTR_EQ_8x16(a, b, mtx);
    return easysimd_agg_equal_any_8x16(la, lb, mtx);
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_word_equal_any(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    easysimd__m128i mtx[16];
    EASYSIMD_PCMPSTR_EQ_16x8(a, b, mtx);
    return easysimd_agg_equal_any_16x8(la, lb, mtx);
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_agg_ranges_16x8(int la, int lb, easysimd__m128i mtx[16])
{
    int res = 0;
    int j;
    int m = (1 << la) - 1;
    uint16x8_t vect = vtstq_u16(vdupq_n_u16(m), vld1q_u16(mask_epi16));
    for (j = 0; j < lb; j++) {
        mtx[j].neon_u16 = vandq_u16(vect, mtx[j].neon_u16);
        mtx[j].neon_u16 = vshrq_n_u16(mtx[j].neon_u16, 15);
        easysimd__m128i tmp;
        tmp.neon_u32 = vshrq_n_u32(mtx[j].neon_u32, 16);
        uint32x4_t vect_res = vandq_u32(mtx[j].neon_u32, tmp.neon_u32);
        int t = vaddvq_u32(vect_res) ? 1 : 0;
        res |= (t << j);
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_agg_ranges_8x16(int la, int lb, easysimd__m128i mtx[16])
{
    int res = 0;
    int j;
    int m = (1 << la) - 1;
    uint8x8_t vect_mask = vld1_u8(mask_epi8);
    uint8x8_t t_lo = vtst_u8(vdup_n_u8(m & 0xff), vect_mask);
    uint8x8_t t_hi = vtst_u8(vdup_n_u8(m >> 8), vect_mask);
    uint8x16_t vect = vcombine_u8(t_lo, t_hi);
    for (j = 0; j < lb; j++) {
        mtx[j].neon_u8 = vandq_u8(vect, mtx[j].neon_u8);
        mtx[j].neon_u8 = vshrq_n_u8(mtx[j].neon_u8, 7);
        easysimd__m128i tmp;
        tmp.neon_u16 = vshrq_n_u16(mtx[j].neon_u16, 8);
        uint16x8_t vect_res = vandq_u16(mtx[j].neon_u16, tmp.neon_u16);
        int t = vaddvq_u16(vect_res) ? 1 : 0;
        res |= (t << j);
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_ubyte_ranges(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    easysimd__m128i mtx[16];
    EASYSIMD_PCMPSTR_RNG_U8x16(a, b, mtx);
    return easysimd_agg_ranges_8x16(la, lb, mtx);
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_sbyte_ranges(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    easysimd__m128i mtx[16];
    EASYSIMD_PCMPSTR_RNG_S8x16(a, b, mtx);
    return easysimd_agg_ranges_8x16(la, lb, mtx);
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_uword_ranges(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    easysimd__m128i mtx[16];
    EASYSIMD_PCMPSTR_RNG_U16x8(a, b, mtx);
    return easysimd_agg_ranges_16x8(la, lb, mtx);
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_sword_ranges(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    easysimd__m128i mtx[16];
    EASYSIMD_PCMPSTR_RNG_S16x8(a, b, mtx);
    return easysimd_agg_ranges_16x8(la, lb, mtx);
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_byte_equal_each(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    uint8x16_t mtx = vceqq_u8(a.neon_u8, b.neon_u8);
    int m0 = (la < lb) ? 0 : ((1 << la) - (1 << lb));
    int m1 = 0x10000 - (1 << la);
    int tb = 0x10000 - (1 << lb);
    uint8x8_t vect_mask, vect0_lo, vect0_hi, vect1_lo, vect1_hi;
    uint8x8_t tmp_lo, tmp_hi, res_lo, res_hi;
    vect_mask = vld1_u8(mask_epi8);
    vect0_lo = vtst_u8(vdup_n_u8(m0), vect_mask);
    vect0_hi = vtst_u8(vdup_n_u8(m0 >> 8), vect_mask);
    vect1_lo = vtst_u8(vdup_n_u8(m1), vect_mask);
    vect1_hi = vtst_u8(vdup_n_u8(m1 >> 8), vect_mask);
    tmp_lo = vtst_u8(vdup_n_u8(tb), vect_mask);
    tmp_hi = vtst_u8(vdup_n_u8(tb >> 8), vect_mask);

    res_lo = vbsl_u8(vect0_lo, vdup_n_u8(0), vget_low_u8(mtx));
    res_hi = vbsl_u8(vect0_hi, vdup_n_u8(0), vget_high_u8(mtx));
    res_lo = vbsl_u8(vect1_lo, tmp_lo, res_lo);
    res_hi = vbsl_u8(vect1_hi, tmp_hi, res_hi);
    res_lo = vand_u8(res_lo, vect_mask);
    res_hi = vand_u8(res_hi, vect_mask);

    int res = vaddv_u8(res_lo) + (vaddv_u8(res_hi) << 8);
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_word_equal_each(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    uint16x8_t mtx = vceqq_u16(a.neon_u16, b.neon_u16);
    int m0 = (la < lb) ? 0 : ((1 << la) - (1 << lb));
    int m1 = 0x100 - (1 << la);
    int tb = 0x100 - (1 << lb);
    uint16x8_t vect_mask = vld1q_u16(mask_epi16);
    uint16x8_t vect0 = vtstq_u16(vdupq_n_u16(m0), vect_mask);
    uint16x8_t vect1 = vtstq_u16(vdupq_n_u16(m1), vect_mask);
    uint16x8_t tmp = vtstq_u16(vdupq_n_u16(tb), vect_mask);
    mtx = vbslq_u16(vect0, vdupq_n_u16(0), mtx);
    mtx = vbslq_u16(vect1, tmp, mtx);
    mtx = vandq_u16(mtx, vect_mask);
    return vaddvq_u16(mtx);
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_agg_equal_ordered_8x16(int bound, int la, int lb, easysimd__m128i mtx[16])
{
    int res = 0;
    int i, j, k;
    int m1 = 0x10000 - (1 << la);
    uint8x8_t vect_mask = vld1_u8(mask_epi8);
    uint8x16_t vect1 = vcombine_u8(vtst_u8(vdup_n_u8(m1), vect_mask), vtst_u8(vdup_n_u8(m1 >> 8), vect_mask));
    uint8x16_t vect_minusone = vdupq_n_u8(-1);
    uint8x16_t vect_zero = vdupq_n_u8(0);
    for (j = 0; j < lb; j++) {
        mtx[j].neon_u8 = vbslq_u8(vect1, vect_minusone, mtx[j].neon_u8);
    }
    for (j = lb; j < bound; j++) {
        mtx[j].neon_u8 = vbslq_u8(vect1, vect_minusone, vect_zero);
    }
    unsigned char *ptr = (unsigned char*)mtx;
    for (i = 0; i < bound; i++) {
        int val = 1;
        for (j = 0, k = i; j < bound - i && k < bound; j++, k++) {
            val &= ptr[k * bound + j];
        }
        res = (val << i) + res;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_agg_equal_ordered_16x8(int bound, int la, int lb, easysimd__m128i mtx[16])
{
    int res = 0;
    int i, j, k;
    int m1 = 0x100 - (1 << la);
    uint16x8_t vect_mask = vld1q_u16(mask_epi16);
    uint16x8_t vect1 = vtstq_u16(vdupq_n_u16(m1), vect_mask);
    uint16x8_t vect_minusone = vdupq_n_u16(-1);
    uint16x8_t vect_zero = vdupq_n_u16(0);
    for (j = 0; j < lb; j++) {
        mtx[j].neon_u16 = vbslq_u16(vect1, vect_minusone, mtx[j].neon_u16);
    }
    for (j = lb; j < bound; j++) {
        mtx[j].neon_u16 = vbslq_u16(vect1, vect_minusone, vect_zero);
    }
    unsigned short *ptr = (unsigned short*)mtx;
    for (i = 0; i < bound; i++) {
        int val = 1;
        for (j = 0, k = i; j < bound - i && k < bound; j++, k++) {
            val &= ptr[k * bound + j];
        }
        res = (val << i) + res;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_byte_equal_ordered(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    easysimd__m128i mtx[16];
    EASYSIMD_PCMPSTR_EQ_8x16(a, b, mtx);
    return easysimd_agg_equal_ordered_8x16(16, la, lb, mtx);
}

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_cal_res_word_equal_ordered(easysimd__m128i a, int la, easysimd__m128i b, int lb)
{
    easysimd__m128i mtx[16];
    EASYSIMD_PCMPSTR_EQ_16x8(a, b, mtx);
    return easysimd_agg_equal_ordered_16x8(8, la, lb, mtx);
}

typedef enum {
    EASYSIMD_CMP_UBYTE_EQUAL_ANY,
    EASYSIMD_CMP_UWORD_EQUAL_ANY,
    EASYSIMD_CMP_SBYTE_EQUAL_ANY,
    EASYSIMD_CMP_SWORD_EQUAL_ANY,
    EASYSIMD_CMP_UBYTE_RANGES,
    EASYSIMD_CMP_UWORD_RANGES,
    EASYSIMD_CMP_SBYTE_RANGES,
    EASYSIMD_CMP_SWORD_RANGES,
    EASYSIMD_CMP_UBYTE_EQUAL_EACH,
    EASYSIMD_CMP_UWORD_EQUAL_EACH,
    EASYSIMD_CMP_SBYTE_EQUAL_EACH,
    EASYSIMD_CMP_SWORD_EQUAL_EACH,
    EASYSIMD_CMP_UBYTE_EQUAL_ORDERED,
    EASYSIMD_CMP_UWORD_EQUAL_ORDERED,
    EASYSIMD_CMP_SBYTE_EQUAL_ORDERED,
    EASYSIMD_CMP_SWORD_EQUAL_ORDERED
} EASYSIMD_MM_CMPESTR_ENUM;

typedef struct {
    EASYSIMD_MM_CMPESTR_ENUM cmpintEnum;
    int (*cmpFun)(easysimd__m128i, int, easysimd__m128i, int);
} easysimd_CmpestrFuncList;

static easysimd_CmpestrFuncList cmpstrfunlist[] = {{EASYSIMD_CMP_UBYTE_EQUAL_ANY, easysimd_cal_res_byte_equal_any},
    {EASYSIMD_CMP_UWORD_EQUAL_ANY, easysimd_cal_res_word_equal_any},
    {EASYSIMD_CMP_SBYTE_EQUAL_ANY, easysimd_cal_res_byte_equal_any},
    {EASYSIMD_CMP_SWORD_EQUAL_ANY, easysimd_cal_res_word_equal_any},
    {EASYSIMD_CMP_UBYTE_RANGES, easysimd_cal_res_ubyte_ranges},
    {EASYSIMD_CMP_UWORD_RANGES, easysimd_cal_res_uword_ranges},
    {EASYSIMD_CMP_SBYTE_RANGES, easysimd_cal_res_sbyte_ranges},
    {EASYSIMD_CMP_SWORD_RANGES, easysimd_cal_res_sword_ranges},
    {EASYSIMD_CMP_UBYTE_EQUAL_EACH, easysimd_cal_res_byte_equal_each},
    {EASYSIMD_CMP_UWORD_EQUAL_EACH, easysimd_cal_res_word_equal_each},
    {EASYSIMD_CMP_SBYTE_EQUAL_EACH, easysimd_cal_res_byte_equal_each},
    {EASYSIMD_CMP_SWORD_EQUAL_EACH, easysimd_cal_res_word_equal_each},
    {EASYSIMD_CMP_UBYTE_EQUAL_ORDERED, easysimd_cal_res_byte_equal_ordered},
    {EASYSIMD_CMP_UWORD_EQUAL_ORDERED, easysimd_cal_res_word_equal_ordered},
    {EASYSIMD_CMP_SBYTE_EQUAL_ORDERED, easysimd_cal_res_byte_equal_ordered},
    {EASYSIMD_CMP_SWORD_EQUAL_ORDERED, easysimd_cal_res_word_equal_ordered}};

EASYSIMD_FUNCTION_ATTRIBUTES
int easysimd_neg_fun(int res, int lb, int imm8, int bound)
{
    int m;
    switch (imm8 & 0x30) {
        case _SIDD_NEON_NEGATIVE_POLARITY:
            res ^= 0xffffffff;
            break;
        case _SIDD_NEON_MASKED_NEGATIVE_POLARITY:
            m = (1 << lb) - 1;
            res ^= m;
            break;
        default:
            break;
    }

    return res & ((bound == 8) ? 0xFF : 0xFFFF);
}

#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpestrm(easysimd__m128i a, int la, easysimd__m128i b, int lb, const int imm8)
{
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_cmpestrm(a, la, b, lb, imm8);
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_SEV_NATIVE))
    easysimd__m128i res;
    int bound = (imm8 & 0x01) ? 8 : 16;
    __asm__ __volatile__ (
        "eor w0, %w[a], %w[a], asr31          \n\t"
        "sub %w[a], w0, %w[a], asr31          \n\t"
        "eor w1, %w[b], %w[b], asr31          \n\t"
        "sub %w[b], w1, %w[b], asr31          \n\t"
        "cmp %w[a], %w[bd]                  \n\t"
        "csel %w[a], %w[bd], %w[a], gt      \n\t"
        "cmp %w[b], %w[bd]                  \n\t"
        "csel %w[b], %w[bd], %w[b], gt      \n\t"
        :[a]"+r"(la), [b]"+r"(lb)
        :[bd]"r"(bound)
        :"w0", "w1"
    );

    int r2 = cmpstrfunlist[imm8 & 0x0f].cmpFun(a, la, b, lb);
    r2 = easysimd_neg_fun(r2, lb, imm8, bound);

    res.neon_u8 = vdupq_n_u8(0);
    if (imm8 & 0x40) {
        if (bound == 8) {
            uint16x8_t tmp = vtstq_u16(vdupq_n_u16(r2), vld1q_u16(mask_epi16));
            res.neon_u16 = vbslq_u16(tmp, vdupq_n_u16(-1), res.neon_u16);
        } else {
            uint8x16_t vect_r2 = vcombine_u8(vdup_n_u8(r2), vdup_n_u8(r2 >> 8));
            uint8x16_t tmp = vtstq_u8(vect_r2, vld1q_u8(mask_epi8));
            res.neon_u8 = vbslq_u8(tmp, vdupq_n_u8(-1), res.neon_u8);
        }
    } else {
        if (bound == 16) {
            res.neon_u16 = vsetq_lane_u16(r2 & 0xffff, res.neon_u16, 0);
        } else {
            res.neon_u8 = vsetq_lane_u8(r2 & 0xff, res.neon_u8, 0);
        }
    }

    return res;
  #else
    return easysimd_mm_setzero_si128();
  #endif
}
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpestrm
  #define _mm_cmpestrm(a,  la,  b,  lb, imm8) easysimd_mm_cmpestrm(a,  la,  b,  lb, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_cmpistrm(easysimd__m128i a, easysimd__m128i b, const int imm8)
{
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_cmpistrm(a, b, imm8);
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_SEV_NATIVE))
    easysimd__m128i res;
    int bound = (imm8 & 0x01) ? 8 : 16, la = bound, lb = bound;
    __asm__ __volatile__ (
        "eor w0, %w[a], %w[a], asr31          \n\t"
        "sub %w[a], w0, %w[a], asr31          \n\t"
        "eor w1, %w[b], %w[b], asr31          \n\t"
        "sub %w[b], w1, %w[b], asr31          \n\t"
        "cmp %w[a], %w[bd]                  \n\t"
        "csel %w[a], %w[bd], %w[a], gt      \n\t"
        "cmp %w[b], %w[bd]                  \n\t"
        "csel %w[b], %w[bd], %w[b], gt      \n\t"
        :[a]"+r"(la), [b]"+r"(lb)
        :[bd]"r"(bound)
        :"w0", "w1"
    );

    int r2 = cmpstrfunlist[imm8 & 0x0f].cmpFun(a, bound, b, bound);
    r2 = easysimd_neg_fun(r2, bound, imm8, bound);

    res.neon_u8 = vdupq_n_u8(0);
    if (imm8 & 0x40) {
        if (bound == 8) {
            uint16x8_t tmp = vtstq_u16(vdupq_n_u16(r2), vld1q_u16(mask_epi16));
            res.neon_u16 = vbslq_u16(tmp, vdupq_n_u16(-1), res.neon_u16);
        } else {
            uint8x16_t vect_r2 = vcombine_u8(vdup_n_u8(r2), vdup_n_u8(r2 >> 8));
            uint8x16_t tmp = vtstq_u8(vect_r2, vld1q_u8(mask_epi8));
            res.neon_u8 = vbslq_u8(tmp, vdupq_n_u8(-1), res.neon_u8);
        }
    } else {
        if (bound == 16) {
            res.neon_u16 = vsetq_lane_u16(r2 & 0xffff, res.neon_u16, 0);
        } else {
            res.neon_u8 = vsetq_lane_u8(r2 & 0xff, res.neon_u8, 0);
        }
    }
    return res;
  #else
    return easysimd_mm_setzero_si128();
  #endif
}

#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpistrm
  #define _mm_cmpistrm(a, b, imm8) easysimd_mm_cmpistrm(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_cmpestri(easysimd__m128i a, int la, easysimd__m128i b, int lb, const int imm8){
#if (defined(EASYSIMD_X86_AVX_NATIVE) && defined(EASYSIMD_X86_SSE4_2_NATIVE))
  return _mm_cmpestri(a, la, b, lb, imm8);
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_SVE_NATIVE)
    int bound = (imm8 & 0x01) ? 8 : 16;
    __asm__ __volatile__ (
        "eor w0, %w[a], %w[a], asr31          \n\t"
        "sub %w[a], w0, %w[a], asr31          \n\t"
        "eor w1, %w[b], %w[b], asr31          \n\t"
        "sub %w[b], w1, %w[b], asr31          \n\t"
        "cmp %w[a], %w[bd]                  \n\t"
        "csel %w[a], %w[bd], %w[a], gt      \n\t"
        "cmp %w[b], %w[bd]                  \n\t"
        "csel %w[b], %w[bd], %w[b], gt      \n\t"
        :[a]"+r"(la), [b]"+r"(lb)
        :[bd]"r"(bound)
        :"w0", "w1"
    );

    int r2 = cmpstrfunlist[imm8 & 0x0f].cmpFun(a, la, b, lb);
    r2 = easysimd_neg_fun(r2, lb, imm8, bound);
    return (r2 == 0) ? bound : ((imm8 & 0x40) ? (31 - __builtin_clz(r2)) : __builtin_ctz(r2));
#else
  return 0;
#endif
}
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpestri
  #define _mm_cmpestri(a,  la,  b,  lb, imm8) easysimd_mm_cmpestri(a,  la,  b,  lb, imm8)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_cmpistrc(easysimd__m128i a, easysimd__m128i b, const int imm8)
{
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_cmpistrc(a, b, imm8);
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_SEV_NATIVE))
    int bound = (imm8 & 0x01) ? 8 : 16, la = bound, lb = bound;
    __asm__ __volatile__ (
        "eor w0, %w[a], %w[a], asr31          \n\t"
        "sub %w[a], w0, %w[a], asr31          \n\t"
        "eor w1, %w[b], %w[b], asr31          \n\t"
        "sub %w[b], w1, %w[b], asr31          \n\t"
        "cmp %w[a], %w[bd]                  \n\t"
        "csel %w[a], %w[bd], %w[a], gt      \n\t"
        "cmp %w[b], %w[bd]                  \n\t"
        "csel %w[b], %w[bd], %w[b], gt      \n\t"
        :[a]"+r"(la), [b]"+r"(lb)
        :[bd]"r"(bound)
        :"w0", "w1"
    );

    int r2 = cmpstrfunlist[imm8 & 0x0f].cmpFun(a, bound, b, bound);
    r2 = easysimd_neg_fun(r2, bound, imm8, bound);

    if (r2 == 0) {
      return 0;
    }
    return 1;
  #else
    return 0;
  #endif
}
#if defined(EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmpistrc
  #define _mm_cmpistrc(a, b, imm8) easysimd_mm_cmpistrc(a, b, imm8)
#endif

#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

#endif /* !defined(EASYSIMD_X86_SSE4_2_H) */
