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
 *        2020 Michael R. Crusoe <crusoe@debian.org>
 */

#include "sse.h"
#if !defined(EASYSIMD_X86_AVX_H)
#define EASYSIMD_X86_AVX_H

#include "sse4.2.h"

#if defined(EASYSIMD_ARM_SVE_NATIVE) 
#include "../arm/sve.h"
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

typedef union {
 #if defined(EASYSIMD_VECTOR_SUBSCRIPT)
    EASYSIMD_ALIGN_TO_32 int8_t          i8 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 int16_t        i16 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 int32_t        i32 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 int64_t        i64 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 uint8_t         u8 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 uint16_t       u16 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 uint32_t       u32 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 uint64_t       u64 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    #if defined(EASYSIMD_HAVE_INT128_)
    EASYSIMD_ALIGN_TO_32 easysimd_int128  i128 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 easysimd_uint128 u128 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    #endif
    EASYSIMD_ALIGN_TO_32 easysimd_float32  f32 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 easysimd_float64  f64 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 int_fast32_t  i32f EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_32 uint_fast32_t u32f EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
  #else
    EASYSIMD_ALIGN_TO_32 int8_t          i8[32];
    EASYSIMD_ALIGN_TO_32 int16_t        i16[16];
    EASYSIMD_ALIGN_TO_32 int32_t        i32[8];
    EASYSIMD_ALIGN_TO_32 int64_t        i64[4];
    EASYSIMD_ALIGN_TO_32 uint8_t         u8[32];
    EASYSIMD_ALIGN_TO_32 uint16_t       u16[16];
    EASYSIMD_ALIGN_TO_32 uint32_t       u32[8];
    EASYSIMD_ALIGN_TO_32 uint64_t       u64[4];
    #if defined(EASYSIMD_HAVE_INT128_)
    EASYSIMD_ALIGN_TO_32 easysimd_int128  i128[2];
    EASYSIMD_ALIGN_TO_32 easysimd_uint128 u128[2];
    #endif
    EASYSIMD_ALIGN_TO_32 easysimd_float32  f32[8];
    EASYSIMD_ALIGN_TO_32 easysimd_float64  f64[4];
    EASYSIMD_ALIGN_TO_32 int_fast32_t  i32f[32 / sizeof(int_fast32_t)];
    EASYSIMD_ALIGN_TO_32 uint_fast32_t u32f[32 / sizeof(uint_fast32_t)];
  #endif

  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint8_t    sve_i8[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint16_t   sve_i16[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint32_t   sve_i32[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint64_t   sve_i64[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint8_t   sve_u8[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint16_t  sve_u16[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint32_t  sve_u32[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint64_t  sve_u64[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat8_t  sve_f8[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat16_t sve_f16[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat32_t sve_f32[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat64_t sve_f64[EASYSIMD_256_BITS_SV_ARRAY_SIZE];
  #endif

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_ALIGN_TO_32 int8x16x2_t  neon_i8x2;
    EASYSIMD_ALIGN_TO_32 uint8x16x2_t neon_u8x2;
  #endif

    EASYSIMD_ALIGN_TO_32 easysimd__m128d_private m128d_private[2];
    EASYSIMD_ALIGN_TO_32 easysimd__m128d         m128d[2];
    EASYSIMD_ALIGN_TO_32 easysimd__m128i_private m128i_private[2];
    EASYSIMD_ALIGN_TO_32 easysimd__m128i         m128i[2];
    EASYSIMD_ALIGN_TO_32 easysimd__m128_private m128_private[2];
    EASYSIMD_ALIGN_TO_32 easysimd__m128         m128[2];

  #if defined(EASYSIMD_X86_BF16_NATIVE)
    EASYSIMD_ALIGN_TO_32 __m256bh       nbh;
  #endif 
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    EASYSIMD_ALIGN_TO_32 __m256d        nd;
    EASYSIMD_ALIGN_TO_32 __m256i        ni;
    EASYSIMD_ALIGN_TO_32 __m256         n;
  #endif
} easysimd__m256_private;
typedef easysimd__m256_private  easysimd__m256i_private;
typedef easysimd__m256_private  easysimd__m256d_private;

#if defined(EASYSIMD_X86_AVX_NATIVE)
  typedef __m256 easysimd__m256;
  typedef __m256i easysimd__m256i;
  typedef __m256d easysimd__m256d;
#elif defined(EASYSIMD_VECTOR_SUBSCRIPT)
  #if defined(EASYSIMD_CONVERT_TO_PRIVATE)
  typedef easysimd_float32 easysimd__m256  EASYSIMD_ALIGN_TO_32 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
  typedef int_fast32_t  easysimd__m256i EASYSIMD_ALIGN_TO_32 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
  typedef easysimd_float64 easysimd__m256d EASYSIMD_ALIGN_TO_32 EASYSIMD_VECTOR(32) EASYSIMD_MAY_ALIAS;
  #else 
  typedef easysimd__m256_private  easysimd__m256;
  typedef easysimd__m256i_private easysimd__m256i;
  typedef easysimd__m256d_private easysimd__m256d;
  #endif
#else
  typedef easysimd__m256_private  easysimd__m256;
  typedef easysimd__m256i_private easysimd__m256i;
  typedef easysimd__m256d_private easysimd__m256d;
#endif

#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #if !defined(HEDLEY_INTEL_VERSION) && !defined(_AVXINTRIN_H_INCLUDED) && !defined(__AVXINTRIN_H) && !defined(_CMP_EQ_OQ)
    typedef easysimd__m256 __m256;
    typedef easysimd__m256i __m256i;
    typedef easysimd__m256d __m256d;
  #else
    #undef __m256
    #define __m256 easysimd__m256
    #undef __m256i
    #define __m256i easysimd__m256i
    #undef __m256d
    #define __m256d easysimd__m256d
  #endif
#endif

HEDLEY_STATIC_ASSERT(32 == sizeof(easysimd__m256), "easysimd__m256 size incorrect");
HEDLEY_STATIC_ASSERT(32 == sizeof(easysimd__m256_private), "easysimd__m256_private size incorrect");
HEDLEY_STATIC_ASSERT(32 == sizeof(easysimd__m256i), "easysimd__m256i size incorrect");
HEDLEY_STATIC_ASSERT(32 == sizeof(easysimd__m256i_private), "easysimd__m256i_private size incorrect");
HEDLEY_STATIC_ASSERT(32 == sizeof(easysimd__m256d), "easysimd__m256d size incorrect");
HEDLEY_STATIC_ASSERT(32 == sizeof(easysimd__m256d_private), "easysimd__m256d_private size incorrect");
#if defined(EASYSIMD_CHECK_ALIGNMENT) && defined(EASYSIMD_ALIGN_OF)
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m256) == 32, "easysimd__m256 is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m256_private) == 32, "easysimd__m256_private is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m256i) == 32, "easysimd__m256i is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m256i_private) == 32, "easysimd__m256i_private is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m256d) == 32, "easysimd__m256d is not 32-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m256d_private) == 32, "easysimd__m256d_private is not 32-byte aligned");
#endif

#if defined(EASYSIMD_CONVERT_TO_PRIVATE) || defined(EASYSIMD_X86_AVX_NATIVE)
EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd__m256_from_private(easysimd__m256_private v) {
  easysimd__m256 r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256_private
easysimd__m256_to_private(easysimd__m256 v) {
  easysimd__m256_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd__m256i_from_private(easysimd__m256i_private v) {
  easysimd__m256i r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i_private
easysimd__m256i_to_private(easysimd__m256i v) {
  easysimd__m256i_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd__m256d_from_private(easysimd__m256d_private v) {
  easysimd__m256d r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d_private
easysimd__m256d_to_private(easysimd__m256d v) {
  easysimd__m256d_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}
#else 

#define easysimd__m256_from_private(v) v
#define easysimd__m256_private(v) v
#define easysimd__m256_to_private(v) v
#define easysimd__m256i_from_private(v) v
#define easysimd__m256i_private(v) v
#define easysimd__m256i_to_private(v) v
#define easysimd__m256d_from_private(v) v
#define easysimd__m256d_to_private(v) v

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256_private
easysimd__m256i_to__m256_private(easysimd__m256i v) {
  easysimd__m256_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef __m256i_to__m256
  #define __m256i_to__m256(v) easysimd__m256i_to__m256_private(v)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i_private
easysimd__m256_to__m256i_private(easysimd__m256 v) {
  easysimd__m256i_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef __m256_to__m256i
  #define __m256_to__m256i(v) easysimd__m256_to__m256i_private(v)
#endif

#endif

#define EASYSIMD_CMP_EQ_OQ     0
#define EASYSIMD_CMP_LT_OS     1
#define EASYSIMD_CMP_LE_OS     2
#define EASYSIMD_CMP_UNORD_Q   3
#define EASYSIMD_CMP_NEQ_UQ    4
#define EASYSIMD_CMP_NLT_US    5
#define EASYSIMD_CMP_NLE_US    6
#define EASYSIMD_CMP_ORD_Q     7
#define EASYSIMD_CMP_EQ_UQ     8
#define EASYSIMD_CMP_NGE_US    9
#define EASYSIMD_CMP_NGT_US   10
#define EASYSIMD_CMP_FALSE_OQ 11
#define EASYSIMD_CMP_NEQ_OQ   12
#define EASYSIMD_CMP_GE_OS    13
#define EASYSIMD_CMP_GT_OS    14
#define EASYSIMD_CMP_TRUE_UQ  15
#define EASYSIMD_CMP_EQ_OS    16
#define EASYSIMD_CMP_LT_OQ    17
#define EASYSIMD_CMP_LE_OQ    18
#define EASYSIMD_CMP_UNORD_S  19
#define EASYSIMD_CMP_NEQ_US   20
#define EASYSIMD_CMP_NLT_UQ   21
#define EASYSIMD_CMP_NLE_UQ   22
#define EASYSIMD_CMP_ORD_S    23
#define EASYSIMD_CMP_EQ_US    24
#define EASYSIMD_CMP_NGE_UQ   25
#define EASYSIMD_CMP_NGT_UQ   26
#define EASYSIMD_CMP_FALSE_OS 27
#define EASYSIMD_CMP_NEQ_OS   28
#define EASYSIMD_CMP_GE_OQ    29
#define EASYSIMD_CMP_GT_OQ    30
#define EASYSIMD_CMP_TRUE_US  31


#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
#define SET8x16(res, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15)          \
    __asm__ __volatile__ (                                      \
        "mov %[r].b[0],  %w[a0]         \n\t"                   \
        "mov %[r].b[1],  %w[a1]         \n\t"                   \
        "mov %[r].b[2],  %w[a2]         \n\t"                   \
        "mov %[r].b[3],  %w[a3]         \n\t"                   \
        "mov %[r].b[4],  %w[a4]         \n\t"                   \
        "mov %[r].b[5],  %w[a5]         \n\t"                   \
        "mov %[r].b[6],  %w[a6]         \n\t"                   \
        "mov %[r].b[7],  %w[a7]         \n\t"                   \
        "mov %[r].b[8],  %w[a8]         \n\t"                   \
        "mov %[r].b[9],  %w[a9]         \n\t"                   \
        "mov %[r].b[10], %w[a10]        \n\t"                   \
        "mov %[r].b[11], %w[a11]        \n\t"                   \
        "mov %[r].b[12], %w[a12]        \n\t"                   \
        "mov %[r].b[13], %w[a13]        \n\t"                   \
        "mov %[r].b[14], %w[a14]        \n\t"                   \
        "mov %[r].b[15], %w[a15]        \n\t"                   \
        :[r]"=w"(res)                                                   \
        :[a0]"r"(e0),   [a1]"r"(e1),   [a2]"r"(e2),   [a3]"r"(e3),      \
         [a4]"r"(e4),   [a5]"r"(e5),   [a6]"r"(e6),   [a7]"r"(e7),      \
         [a8]"r"(e8),   [a9]"r"(e9),   [a10]"r"(e10), [a11]"r"(e11),    \
         [a12]"r"(e12), [a13]"r"(e13), [a14]"r"(e14), [a15]"r"(e15)     \
        :                                                               \
    );

#define SET16x8(res, e0, e1, e2, e3, e4, e5, e6, e7)          \
    __asm__ __volatile__ (                                    \
        "mov %[r].h[0], %w[a0]        \n\t"                   \
        "mov %[r].h[1], %w[a1]        \n\t"                   \
        "mov %[r].h[2], %w[a2]        \n\t"                   \
        "mov %[r].h[3], %w[a3]        \n\t"                   \
        "mov %[r].h[4], %w[a4]        \n\t"                   \
        "mov %[r].h[5], %w[a5]        \n\t"                   \
        "mov %[r].h[6], %w[a6]        \n\t"                   \
        "mov %[r].h[7], %w[a7]        \n\t"                   \
        :[r]"=w"(res)                                         \
        :[a0]"r"(e0), [a1]"r"(e1), [a2]"r"(e2), [a3]"r"(e3),  \
         [a4]"r"(e4), [a5]"r"(e5), [a6]"r"(e6), [a7]"r"(e7)   \
        :                                                     \
    );

#define SET32x4(res, e0, e1, e2, e3)                     \
    __asm__ __volatile__ (                                  \
        "mov %[r].s[0], %w[x]        \n\t"                  \
        "mov %[r].s[1], %w[y]        \n\t"                  \
        "mov %[r].s[2], %w[z]        \n\t"                  \
        "mov %[r].s[3], %w[k]        \n\t"                  \
        :[r]"=w"(res)                                       \
        :[x]"r"(e0), [y]"r"(e1), [z]"r"(e2), [k]"r"(e3)     \
    );
#define  SET64x2(res, e0, e1)                            \
    __asm__ __volatile__ (                                  \
        "mov %[r].d[0], %[x]         \n\t"                  \
        "mov %[r].d[1], %[y]         \n\t"                  \
        :[r]"=w"(res)                                       \
        :[x]"r"(e0), [y]"r"(e1)                             \
    );

typedef struct {
    int OpDef;
    uint64x2_t (*neoncmpfun_pd)(easysimd__m128d a, easysimd__m128d b);
} NeonFunListCmpPd;

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_eq_oq(easysimd__m128d a, easysimd__m128d b)
{ /* Equal (ordered, non-signaling) */
    return vceqq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_lt_os(easysimd__m128d a, easysimd__m128d b)
{ /* Less-than (ordered, signaling)  */
    return vcltq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_le_os(easysimd__m128d a, easysimd__m128d b)
{ /* Less-than-or-equal (ordered, signaling)  */
    return vcleq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_unord_q(easysimd__m128d a, easysimd__m128d b)
{ /* Unordered (non-signaling)  */

    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    ptr_r[0] = easysimd_math_isunordered(ptr_a[0], ptr_b[0]) ? -1 : 0;
    ptr_r[1] = easysimd_math_isunordered(ptr_a[1], ptr_b[1]) ? -1 : 0;
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_neq_uq(easysimd__m128d a, easysimd__m128d b)
{ /* Not-equal (unordered, non-signaling)  */
    easysimd__m128i res;
    res.neon_u64 = vceqq_f64(a.neon_f64, b.neon_f64);
    res.neon_u32 = vmvnq_u32(res.neon_u32);
    return res.neon_u64;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_nlt_us(easysimd__m128d a, easysimd__m128d b)
{ /* Not-less-than (unordered, signaling) */

    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vcgeq_f64(a.neon_f64, b.neon_f64);
    
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_nle_us(easysimd__m128d a, easysimd__m128d b)
{ /* Not-less-than-or-equal (unordered, signaling)  */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vcgtq_f64(a.neon_f64, b.neon_f64);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_ord_q(easysimd__m128d a, easysimd__m128d b)
{ /* Ordered (nonsignaling) */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    ptr_r[0] = !easysimd_math_isunordered(ptr_a[0], ptr_b[0]) ? -1 : 0;
    ptr_r[1] = !easysimd_math_isunordered(ptr_a[1], ptr_b[1]) ? -1 : 0;
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_eq_uq(easysimd__m128d a, easysimd__m128d b)
{ /* Equal (unordered, non-signaling) */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vceqq_f64(a.neon_f64, b.neon_f64);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_nge_us(easysimd__m128d a, easysimd__m128d b)
{ /* Not-greater-than-or-equal (unordered, signaling) */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vcltq_f64(a.neon_f64, b.neon_f64);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_ngt_us(easysimd__m128d a, easysimd__m128d b)
{ /* Not-greater-than (unordered, signaling)  */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vcleq_f64(a.neon_f64, b.neon_f64);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_false_oq(easysimd__m128d a, easysimd__m128d b)
{ /* False (ordered, non-signaling)  */
    (void)a;
    (void)b;

    return vdupq_n_u64(0);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_neq_oq(easysimd__m128d a, easysimd__m128d b)
{ /* Not-equal (ordered, non-signaling)  */
    easysimd__m128i res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i.neon_u64 = vceqq_f64(a.neon_f64, b.neon_f64);
    res_m128i.neon_u32 = vmvnq_u32(res_m128i.neon_u32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = 0;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = 0;
    }
    return res_m128i.neon_u64;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_ge_os(easysimd__m128d a, easysimd__m128d b)
{ /* Greater-than-or-equal (ordered, signaling)  */
    return vcgeq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_gt_os(easysimd__m128d a, easysimd__m128d b)
{ /* Greater-than (ordered, signaling)  */
    return vcgtq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_true_uq(easysimd__m128d a, easysimd__m128d b)
{ /* True (unordered, non-signaling) */
    (void)a;
    (void)b;

    return vdupq_n_u64(-1);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_eq_os(easysimd__m128d a, easysimd__m128d b)
{ /* Equal (ordered, signaling)  */
    return vceqq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_lt_oq(easysimd__m128d a, easysimd__m128d b)
{ /* Less-than (ordered, non-signaling)  */
    return vcltq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_le_oq(easysimd__m128d a, easysimd__m128d b)
{ /* Less-than-or-equal (ordered, non-signaling)  */
    return vcleq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_unord_s(easysimd__m128d a, easysimd__m128d b)
{ /* Unordered (signaling)  */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    ptr_r[0] = easysimd_math_isunordered(ptr_a[0], ptr_b[0]) ? -1 : 0;
    ptr_r[1] = easysimd_math_isunordered(ptr_a[1], ptr_b[1]) ? -1 : 0;
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_neq_us(easysimd__m128d a, easysimd__m128d b)
{ /* Not-equal (unordered, signaling) */
    easysimd__m128i res_m128i;
    res_m128i.neon_u64 = vceqq_f64(a.neon_f64, b.neon_f64);
    res_m128i.neon_u32 = vmvnq_u32(res_m128i.neon_u32);
    return res_m128i.neon_u64;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_nlt_uq(easysimd__m128d a, easysimd__m128d b)
{ /* Not-less-than (unordered, non-signaling)*/
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vcgeq_f64(a.neon_f64, b.neon_f64);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_nle_uq(easysimd__m128d a, easysimd__m128d b)
{ /* Not-less-than-or-equal (unordered, non-signaling) */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vcgtq_f64(a.neon_f64, b.neon_f64);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_ord_s(easysimd__m128d a, easysimd__m128d b)
{ /* Ordered (signaling)  */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vceqq_f64(a.neon_f64, b.neon_f64);
    ptr_r[0] = !easysimd_math_isunordered(ptr_a[0], ptr_b[0]) ? -1 : 0;
    ptr_r[1] = !easysimd_math_isunordered(ptr_a[1], ptr_b[1]) ? -1 : 0;
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_eq_us(easysimd__m128d a, easysimd__m128d b)
{ /* Equal (unordered, signaling)  */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vceqq_f64(a.neon_f64, b.neon_f64);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_nge_uq(easysimd__m128d a, easysimd__m128d b)
{ /* Not-greater-than-or-equal (unordered, non-signaling)  */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vcltq_f64(a.neon_f64, b.neon_f64);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_ngt_uq(easysimd__m128d a, easysimd__m128d b)
{ /* Not-greater-than (unordered, non-signaling) */
    uint64x2_t res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i = vcleq_f64(a.neon_f64, b.neon_f64);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    return res_m128i;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_false_os(easysimd__m128d a, easysimd__m128d b)
{ /* False (ordered, signaling)  */
    (void)a;
    (void)b;

    return vdupq_n_u64(0);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_neq_os(easysimd__m128d a, easysimd__m128d b)
{ /* Not-equal (ordered, signaling)  */
    easysimd__m128i res_m128i;
    float64_t *ptr_a = (float64_t *)&(a.neon_f64);
    float64_t *ptr_b = (float64_t *)&(b.neon_f64);
    uint64_t *ptr_r = (uint64_t *)&res_m128i;

    res_m128i.neon_u64 = vceqq_f64(a.neon_f64, b.neon_f64);
    res_m128i.neon_u32 = vmvnq_u32(res_m128i.neon_u32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = 0;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = 0;
    }
    return res_m128i.neon_u64;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_ge_oq(easysimd__m128d a, easysimd__m128d b)
{ /* Greater-than-or-equal (ordered, non-signaling)  */
    return vcgeq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_gt_oq(easysimd__m128d a, easysimd__m128d b)
{ /* Greater-than (ordered, non-signaling)  */
    return vcgtq_f64(a.neon_f64, b.neon_f64);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint64x2_t neon_cmp_true_us(easysimd__m128d a, easysimd__m128d b)
{ /* True (unordered, signaling)  */
    (void)a;
    (void)b;

    return vdupq_n_u64(-1);
}

__attribute__((unused)) static NeonFunListCmpPd neonfunlistcmppd[] = {
    {EASYSIMD_CMP_EQ_OQ, neon_cmp_eq_oq},   {EASYSIMD_CMP_LT_OS, neon_cmp_lt_os},   {EASYSIMD_CMP_LE_OS, neon_cmp_le_os},   {EASYSIMD_CMP_UNORD_Q, neon_cmp_unord_q},
    {EASYSIMD_CMP_NEQ_UQ, neon_cmp_neq_uq}, {EASYSIMD_CMP_NLT_US, neon_cmp_nlt_us}, {EASYSIMD_CMP_NLE_US, neon_cmp_nle_us}, {EASYSIMD_CMP_ORD_Q, neon_cmp_ord_q},
    {EASYSIMD_CMP_EQ_UQ, neon_cmp_eq_uq},   {EASYSIMD_CMP_NGE_US, neon_cmp_nge_us}, {EASYSIMD_CMP_NGT_US, neon_cmp_ngt_us}, {EASYSIMD_CMP_FALSE_OQ, neon_cmp_false_oq},
    {EASYSIMD_CMP_NEQ_OQ, neon_cmp_neq_oq}, {EASYSIMD_CMP_GE_OS, neon_cmp_ge_os},   {EASYSIMD_CMP_GT_OS, neon_cmp_gt_os},   {EASYSIMD_CMP_TRUE_UQ, neon_cmp_true_uq},
    {EASYSIMD_CMP_EQ_OS, neon_cmp_eq_os},   {EASYSIMD_CMP_LT_OQ, neon_cmp_lt_oq},   {EASYSIMD_CMP_LE_OQ, neon_cmp_le_oq},   {EASYSIMD_CMP_UNORD_S, neon_cmp_unord_s},
    {EASYSIMD_CMP_NEQ_US, neon_cmp_neq_us}, {EASYSIMD_CMP_NLT_UQ, neon_cmp_nlt_uq}, {EASYSIMD_CMP_NLE_UQ, neon_cmp_nle_uq}, {EASYSIMD_CMP_ORD_S, neon_cmp_ord_s},
    {EASYSIMD_CMP_EQ_US, neon_cmp_eq_us},   {EASYSIMD_CMP_NGE_UQ, neon_cmp_nge_uq}, {EASYSIMD_CMP_NGT_UQ, neon_cmp_ngt_uq}, {EASYSIMD_CMP_FALSE_OS, neon_cmp_false_os},
    {EASYSIMD_CMP_NEQ_OS, neon_cmp_neq_os}, {EASYSIMD_CMP_GE_OQ, neon_cmp_ge_oq},   {EASYSIMD_CMP_GT_OQ, neon_cmp_gt_oq},   {EASYSIMD_CMP_TRUE_US, neon_cmp_true_us}};

typedef struct {
    int OpDef;
    uint32x4_t (*neoncmpfun_ps)(easysimd__m128 a, easysimd__m128 b);
} NeonFunListCmpPs;

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_eq_oq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Equal (ordered, non-signaling) */
    return vceqq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_lt_os_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Less-than (ordered, signaling)  */
    return vcltq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_le_os_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Less-than-or-equal (ordered, signaling)  */
    return vcleq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_unord_q_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Unordered (non-signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    ptr_r[0] = easysimd_math_isunordered(ptr_a[0], ptr_b[0]) ? -1 : 0;
    ptr_r[1] = easysimd_math_isunordered(ptr_a[1], ptr_b[1]) ? -1 : 0;
    ptr_r[2] = easysimd_math_isunordered(ptr_a[2], ptr_b[2]) ? -1 : 0;
    ptr_r[3] = easysimd_math_isunordered(ptr_a[3], ptr_b[3]) ? -1 : 0;
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_neq_uq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-equal (unordered, non-signaling)  */
    uint32x4_t res = vceqq_f32(a.neon_f32, b.neon_f32);
    return vmvnq_u32(res);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_nlt_us_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-less-than (unordered, signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vcgeq_f32(a.neon_f32, b.neon_f32);
    
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_nle_us_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-less-than-or-equal (unordered, signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vcgtq_f32(a.neon_f32, b.neon_f32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_ord_q_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Ordered (nonsignaling) */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    ptr_r[0] = !easysimd_math_isunordered(ptr_a[0], ptr_b[0]) ? -1 : 0;
    ptr_r[1] = !easysimd_math_isunordered(ptr_a[1], ptr_b[1]) ? -1 : 0;
    ptr_r[2] = !easysimd_math_isunordered(ptr_a[2], ptr_b[2]) ? -1 : 0;
    ptr_r[3] = !easysimd_math_isunordered(ptr_a[3], ptr_b[3]) ? -1 : 0;
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_eq_uq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Equal (unordered, non-signaling) */ 
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vceqq_f32(a.neon_f32, b.neon_f32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_nge_us_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-greater-than-or-equal (unordered, signaling) */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vcltq_f32(a.neon_f32, b.neon_f32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_ngt_us_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-greater-than (unordered, signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vcleq_f32(a.neon_f32, b.neon_f32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_false_oq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* False (ordered, non-signaling)  */
    (void)a;
    (void)b;
    return vdupq_n_u32(0);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_neq_oq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-equal (ordered, non-signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vceqq_f32(a.neon_f32, b.neon_f32);
    res = vmvnq_u32(res);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = 0;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = 0;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = 0;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = 0;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_ge_os_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Greater-than-or-equal (ordered, signaling)  */
    return vcgeq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_gt_os_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Greater-than (ordered, signaling)  */
    return vcgtq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_true_uq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* True (unordered, non-signaling) */
    (void)a;
    (void)b;
    return vdupq_n_u32(-1);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_eq_os_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Equal (ordered, signaling)  */
    return vceqq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_lt_oq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Less-than (ordered, non-signaling)  */
    return vcltq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_le_oq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Less-than-or-equal (ordered, non-signaling)  */
    return vcleq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_unord_s_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Unordered (signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    ptr_r[0] = easysimd_math_isunordered(ptr_a[0], ptr_b[0]) ? -1 : 0;
    ptr_r[1] = easysimd_math_isunordered(ptr_a[1], ptr_b[1]) ? -1 : 0;
    ptr_r[2] = easysimd_math_isunordered(ptr_a[2], ptr_b[2]) ? -1 : 0;
    ptr_r[3] = easysimd_math_isunordered(ptr_a[3], ptr_b[3]) ? -1 : 0;
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_neq_us_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-equal (unordered, signaling) */
    uint32x4_t res = vceqq_f32(a.neon_f32, b.neon_f32);
    return vmvnq_u32(res);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_nlt_uq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-less-than (unordered, non-signaling)*/  
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vcgeq_f32(a.neon_f32, b.neon_f32);
    
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_nle_uq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-less-than-or-equal (unordered, non-signaling) */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vcgtq_f32(a.neon_f32, b.neon_f32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_ord_s_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Ordered (signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    ptr_r[0] = !easysimd_math_isunordered(ptr_a[0], ptr_b[0]) ? -1 : 0;
    ptr_r[1] = !easysimd_math_isunordered(ptr_a[1], ptr_b[1]) ? -1 : 0;
    ptr_r[2] = !easysimd_math_isunordered(ptr_a[2], ptr_b[2]) ? -1 : 0;
    ptr_r[3] = !easysimd_math_isunordered(ptr_a[3], ptr_b[3]) ? -1 : 0;
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_eq_us_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Equal (unordered, signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vceqq_f32(a.neon_f32, b.neon_f32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_nge_uq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-greater-than-or-equal (unordered, non-signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vcltq_f32(a.neon_f32, b.neon_f32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_ngt_uq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-greater-than (unordered, non-signaling) */ 
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vcleq_f32(a.neon_f32, b.neon_f32);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = -1;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = -1;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_false_os_ps(easysimd__m128 a, easysimd__m128 b)
{ /* False (ordered, signaling)  */
    (void)a;
    (void)b;
    return vdupq_n_u32(0);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_neq_os_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Not-equal (ordered, signaling)  */
    uint32x4_t res;
    float32_t *ptr_a = (float32_t *)&(a.neon_f32);
    float32_t *ptr_b = (float32_t *)&(b.neon_f32);
    uint32_t *ptr_r = (uint32_t *)&res;

    res = vceqq_f32(a.neon_f32, b.neon_f32);
    res = vmvnq_u32(res);
    if (easysimd_math_isunordered(ptr_a[0], ptr_b[0])) {
        ptr_r[0] = 0;
    }
    if (easysimd_math_isunordered(ptr_a[1], ptr_b[1])) {
        ptr_r[1] = 0;
    }
    if (easysimd_math_isunordered(ptr_a[2], ptr_b[2])) {
        ptr_r[2] = 0;
    }
    if (easysimd_math_isunordered(ptr_a[3], ptr_b[3])) {
        ptr_r[3] = 0;
    }
    return res;
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_ge_oq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Greater-than-or-equal (ordered, non-signaling)  */
    return vcgeq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_gt_oq_ps(easysimd__m128 a, easysimd__m128 b)
{ /* Greater-than (ordered, non-signaling)  */
    return vcgtq_f32(a.neon_f32, b.neon_f32);
}

EASYSIMD_FUNCTION_ATTRIBUTES uint32x4_t neon_cmp_true_us_ps(easysimd__m128 a, easysimd__m128 b)
{ /* True (unordered, signaling)  */
    (void)a;
    (void)b;
    return vdupq_n_u32(-1);
}

__attribute__((unused)) static NeonFunListCmpPs neonfunlistcmpps[] = {
    {EASYSIMD_CMP_EQ_OQ, neon_cmp_eq_oq_ps},       {EASYSIMD_CMP_LT_OS, neon_cmp_lt_os_ps},     {EASYSIMD_CMP_LE_OS, neon_cmp_le_os_ps},
    {EASYSIMD_CMP_UNORD_Q, neon_cmp_unord_q_ps},   {EASYSIMD_CMP_NEQ_UQ, neon_cmp_neq_uq_ps},   {EASYSIMD_CMP_NLT_US, neon_cmp_nlt_us_ps},
    {EASYSIMD_CMP_NLE_US, neon_cmp_nle_us_ps},     {EASYSIMD_CMP_ORD_Q, neon_cmp_ord_q_ps},     {EASYSIMD_CMP_EQ_UQ, neon_cmp_eq_uq_ps},
    {EASYSIMD_CMP_NGE_US, neon_cmp_nge_us_ps},     {EASYSIMD_CMP_NGT_US, neon_cmp_ngt_us_ps},   {EASYSIMD_CMP_FALSE_OQ, neon_cmp_false_oq_ps},
    {EASYSIMD_CMP_NEQ_OQ, neon_cmp_neq_oq_ps},     {EASYSIMD_CMP_GE_OS, neon_cmp_ge_os_ps},     {EASYSIMD_CMP_GT_OS, neon_cmp_gt_os_ps},
    {EASYSIMD_CMP_TRUE_UQ, neon_cmp_true_uq_ps},   {EASYSIMD_CMP_EQ_OS, neon_cmp_eq_os_ps},     {EASYSIMD_CMP_LT_OQ, neon_cmp_lt_oq_ps},
    {EASYSIMD_CMP_LE_OQ, neon_cmp_le_oq_ps},       {EASYSIMD_CMP_UNORD_S, neon_cmp_unord_s_ps}, {EASYSIMD_CMP_NEQ_US, neon_cmp_neq_us_ps},
    {EASYSIMD_CMP_NLT_UQ, neon_cmp_nlt_uq_ps},     {EASYSIMD_CMP_NLE_UQ, neon_cmp_nle_uq_ps},   {EASYSIMD_CMP_ORD_S, neon_cmp_ord_s_ps},
    {EASYSIMD_CMP_EQ_US, neon_cmp_eq_us_ps},       {EASYSIMD_CMP_NGE_UQ, neon_cmp_nge_uq_ps},   {EASYSIMD_CMP_NGT_UQ, neon_cmp_ngt_uq_ps},
    {EASYSIMD_CMP_FALSE_OS, neon_cmp_false_os_ps}, {EASYSIMD_CMP_NEQ_OS, neon_cmp_neq_os_ps},   {EASYSIMD_CMP_GE_OQ, neon_cmp_ge_oq_ps},
    {EASYSIMD_CMP_GT_OQ, neon_cmp_gt_oq_ps},       {EASYSIMD_CMP_TRUE_US, neon_cmp_true_us_ps}};

#endif

#if defined (EASYSIMD_ARM_SVE_NATIVE)
/*ps:*/
EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_eq_oq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmpeq_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_eq_os_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmpeq_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_eq_uq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svorr_b_z(pg, svcmpuo_f32(pg, sva, svb), svcmpeq_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_eq_us_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svorr_b_z(pg, svcmpuo_f32(pg, sva, svb), svcmpeq_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_lt_os_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmplt_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_lt_oq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmplt_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_le_os_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmple_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_le_oq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmple_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_unord_q_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgord = svcmpuo_f32(pg, sva, svb);
  return svdup_n_s32_z(pgord, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_unord_s_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgord = svcmpuo_f32(pg, sva, svb);
  return svdup_n_s32_z(pgord, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_ord_q_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgord = svcmpuo_f32(pg, sva, svb);
                 pgord = svnot_b_z(pg, pgord);
  return svdup_n_s32_z(pgord, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_ord_s_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgord = svcmpuo_f32(pg, sva, svb);
                 pgord = svnot_b_z(pg, pgord);
  return svdup_n_s32_z(pgord, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_neq_oq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svbic_b_z(pg, svcmpne_f32(pg, sva, svb), svcmpuo_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_neq_os_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svbic_b_z(pg, svcmpne_f32(pg, sva, svb), svcmpuo_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_neq_uq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmpne_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_neq_us_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmpne_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_nlt_uq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmplt_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_nlt_us_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmplt_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_nle_uq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmple_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_nle_us_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmple_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_nge_uq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmpge_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_nge_us_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmpge_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_ngt_uq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmpgt_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_ngt_us_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmpgt_f32(pg, sva, svb));
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_false_oq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  HEDLEY_UNUSED(sva);
  HEDLEY_UNUSED(svb);
  sveint32_t svr = svdup_n_s32_z(pg, 0);
  return svr;
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_false_os_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  HEDLEY_UNUSED(sva);
  HEDLEY_UNUSED(svb);
  sveint32_t svr = svdup_n_s32_z(pg, 0);
  return svr;
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_ge_os_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmpge_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_ge_oq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmpge_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_gt_os_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmpgt_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_gt_oq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  easysimd_svbool_t pgcmp = svcmpgt_f32(pg, sva, svb);
  return svdup_n_s32_z(pgcmp, 0xFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_true_uq_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  HEDLEY_UNUSED(sva);
  HEDLEY_UNUSED(svb);
  sveint32_t svr = svdup_n_s32_z(pg, 0xFFFFFFFF);
  return svr;
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint32_t _cmp_true_us_ps(easysimd_svbool_t pg,  svefloat32_t sva, svefloat32_t svb){
  HEDLEY_UNUSED(sva);
  HEDLEY_UNUSED(svb);
  sveint32_t svr = svdup_n_s32_z(pg, 0xFFFFFFFF);
  return svr;
}

typedef struct {
  int OP;
  sveint32_t (*cmpfun_ps)(easysimd_svbool_t,  svefloat32_t, svefloat32_t);
} funListCmpPs;

__attribute__((unused)) static funListCmpPs funlistcmpps[] = {
    {EASYSIMD_CMP_EQ_OQ,   _cmp_eq_oq_ps}, {EASYSIMD_CMP_LT_OS,   _cmp_lt_os_ps}, {EASYSIMD_CMP_LE_OS,   _cmp_le_os_ps}, {EASYSIMD_CMP_UNORD_Q,   _cmp_unord_q_ps},
    {EASYSIMD_CMP_NEQ_UQ, _cmp_neq_uq_ps}, {EASYSIMD_CMP_NLT_US, _cmp_nlt_us_ps}, {EASYSIMD_CMP_NLE_US, _cmp_nle_us_ps}, {EASYSIMD_CMP_ORD_Q,       _cmp_ord_q_ps},
    {EASYSIMD_CMP_EQ_UQ,   _cmp_eq_uq_ps}, {EASYSIMD_CMP_NGE_US, _cmp_nge_us_ps}, {EASYSIMD_CMP_NGT_US, _cmp_ngt_us_ps}, {EASYSIMD_CMP_FALSE_OQ, _cmp_false_oq_ps},
    {EASYSIMD_CMP_NEQ_OQ, _cmp_neq_oq_ps}, {EASYSIMD_CMP_GE_OS,   _cmp_ge_os_ps}, {EASYSIMD_CMP_GT_OS,   _cmp_gt_os_ps}, {EASYSIMD_CMP_TRUE_UQ,   _cmp_true_uq_ps},
    {EASYSIMD_CMP_EQ_OS,   _cmp_eq_os_ps}, {EASYSIMD_CMP_LT_OQ,   _cmp_lt_oq_ps}, {EASYSIMD_CMP_LE_OQ,   _cmp_le_oq_ps}, {EASYSIMD_CMP_UNORD_S,   _cmp_unord_s_ps},
    {EASYSIMD_CMP_NEQ_US, _cmp_neq_us_ps}, {EASYSIMD_CMP_NLT_UQ, _cmp_nlt_uq_ps}, {EASYSIMD_CMP_NLE_UQ, _cmp_nle_uq_ps}, {EASYSIMD_CMP_ORD_S,       _cmp_ord_s_ps},
    {EASYSIMD_CMP_EQ_US,   _cmp_eq_us_ps}, {EASYSIMD_CMP_NGE_UQ, _cmp_nge_uq_ps}, {EASYSIMD_CMP_NGT_UQ, _cmp_ngt_uq_ps}, {EASYSIMD_CMP_FALSE_OS, _cmp_false_os_ps},
    {EASYSIMD_CMP_NEQ_OS, _cmp_neq_os_ps}, {EASYSIMD_CMP_GE_OQ,   _cmp_ge_oq_ps}, {EASYSIMD_CMP_GT_OQ,   _cmp_gt_oq_ps}, {EASYSIMD_CMP_TRUE_US,   _cmp_true_us_ps}};

/*pd:*/
EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_eq_oq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmpeq_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_eq_os_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmpeq_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_eq_uq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svorr_b_z(pg, svcmpuo_f64(pg, sva, svb), svcmpeq_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_eq_us_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svorr_b_z(pg, svcmpuo_f64(pg, sva, svb), svcmpeq_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_lt_os_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmplt_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_lt_oq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmplt_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_le_os_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmple_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_le_oq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmple_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_unord_q_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgord = svcmpuo_f64(pg, sva, svb);
  return svdup_n_s64_z(pgord, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_unord_s_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgord = svcmpuo_f64(pg, sva, svb);
  return svdup_n_s64_z(pgord, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_ord_q_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgord = svcmpuo_f64(pg, sva, svb);
                 pgord = svnot_b_z(pg, pgord);
  return svdup_n_s64_z(pgord, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_ord_s_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgord = svcmpuo_f64(pg, sva, svb);
                 pgord = svnot_b_z(pg, pgord);
  return svdup_n_s64_z(pgord, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_neq_oq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svbic_b_z(pg, svcmpne_f64(pg, sva, svb), svcmpuo_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_neq_os_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svbic_b_z(pg, svcmpne_f64(pg, sva, svb), svcmpuo_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_neq_uq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmpne_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_neq_us_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmpne_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_nlt_uq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmplt_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_nlt_us_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmplt_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_nle_uq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmple_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_nle_us_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmple_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_nge_uq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmpge_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_nge_us_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmpge_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_ngt_uq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmpgt_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_ngt_us_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svnot_b_z(pg, svcmpgt_f64(pg, sva, svb));
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_false_oq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  HEDLEY_UNUSED(sva);
  HEDLEY_UNUSED(svb);
  sveint64_t svr = svdup_n_s64_z(pg, 0);
  return svr;
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_false_os_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  HEDLEY_UNUSED(sva);
  HEDLEY_UNUSED(svb);
  sveint64_t svr = svdup_n_s64_z(pg, 0);
  return svr;
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_ge_os_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmpge_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_ge_oq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmpge_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_gt_os_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmpgt_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_gt_oq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  easysimd_svbool_t pgcmp = svcmpgt_f64(pg, sva, svb);
  return svdup_n_s64_z(pgcmp, 0xFFFFFFFFFFFFFFFF);
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_true_uq_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  HEDLEY_UNUSED(sva);
  HEDLEY_UNUSED(svb);
  sveint64_t svr = svdup_n_s64_z(pg, 0xFFFFFFFFFFFFFFFF);
  return svr;
}

EASYSIMD_FUNCTION_ATTRIBUTES sveint64_t _cmp_true_us_pd(easysimd_svbool_t pg,  svefloat64_t sva, svefloat64_t svb){
  HEDLEY_UNUSED(sva);
  HEDLEY_UNUSED(svb);
  sveint64_t svr = svdup_n_s64_z(pg, 0xFFFFFFFFFFFFFFFF);
  return svr;
}

typedef struct {
  int OP;
  sveint64_t (*cmpfun_pd)(easysimd_svbool_t,  svefloat64_t, svefloat64_t);
} funListCmpPd;

__attribute__((unused)) static funListCmpPd funlistcmppd[] = {
    {EASYSIMD_CMP_EQ_OQ,   _cmp_eq_oq_pd}, {EASYSIMD_CMP_LT_OS,   _cmp_lt_os_pd}, {EASYSIMD_CMP_LE_OS,   _cmp_le_os_pd}, {EASYSIMD_CMP_UNORD_Q,   _cmp_unord_q_pd},
    {EASYSIMD_CMP_NEQ_UQ, _cmp_neq_uq_pd}, {EASYSIMD_CMP_NLT_US, _cmp_nlt_us_pd}, {EASYSIMD_CMP_NLE_US, _cmp_nle_us_pd}, {EASYSIMD_CMP_ORD_Q,       _cmp_ord_q_pd},
    {EASYSIMD_CMP_EQ_UQ,   _cmp_eq_uq_pd}, {EASYSIMD_CMP_NGE_US, _cmp_nge_us_pd}, {EASYSIMD_CMP_NGT_US, _cmp_ngt_us_pd}, {EASYSIMD_CMP_FALSE_OQ, _cmp_false_oq_pd},
    {EASYSIMD_CMP_NEQ_OQ, _cmp_neq_oq_pd}, {EASYSIMD_CMP_GE_OS,   _cmp_ge_os_pd}, {EASYSIMD_CMP_GT_OS,   _cmp_gt_os_pd}, {EASYSIMD_CMP_TRUE_UQ,   _cmp_true_uq_pd},
    {EASYSIMD_CMP_EQ_OS,   _cmp_eq_os_pd}, {EASYSIMD_CMP_LT_OQ,   _cmp_lt_oq_pd}, {EASYSIMD_CMP_LE_OQ,   _cmp_le_oq_pd}, {EASYSIMD_CMP_UNORD_S,   _cmp_unord_s_pd},
    {EASYSIMD_CMP_NEQ_US, _cmp_neq_us_pd}, {EASYSIMD_CMP_NLT_UQ, _cmp_nlt_uq_pd}, {EASYSIMD_CMP_NLE_UQ, _cmp_nle_uq_pd}, {EASYSIMD_CMP_ORD_S,       _cmp_ord_s_pd},
    {EASYSIMD_CMP_EQ_US,   _cmp_eq_us_pd}, {EASYSIMD_CMP_NGE_UQ, _cmp_nge_uq_pd}, {EASYSIMD_CMP_NGT_UQ, _cmp_ngt_uq_pd}, {EASYSIMD_CMP_FALSE_OS, _cmp_false_os_pd},
    {EASYSIMD_CMP_NEQ_OS, _cmp_neq_os_pd}, {EASYSIMD_CMP_GE_OQ,   _cmp_ge_oq_pd}, {EASYSIMD_CMP_GT_OQ,   _cmp_gt_oq_pd}, {EASYSIMD_CMP_TRUE_US,   _cmp_true_us_pd}};

#endif

#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES) && !defined(_CMP_EQ_OQ)
#define _CMP_EQ_OQ EASYSIMD_CMP_EQ_OQ
#define _CMP_LT_OS EASYSIMD_CMP_LT_OS
#define _CMP_LE_OS EASYSIMD_CMP_LE_OS
#define _CMP_UNORD_Q EASYSIMD_CMP_UNORD_Q
#define _CMP_NEQ_UQ EASYSIMD_CMP_NEQ_UQ
#define _CMP_NLT_US EASYSIMD_CMP_NLT_US
#define _CMP_NLE_US EASYSIMD_CMP_NLE_US
#define _CMP_ORD_Q EASYSIMD_CMP_ORD_Q
#define _CMP_EQ_UQ EASYSIMD_CMP_EQ_UQ
#define _CMP_NGE_US EASYSIMD_CMP_NGE_US
#define _CMP_NGT_US EASYSIMD_CMP_NGT_US
#define _CMP_FALSE_OQ EASYSIMD_CMP_FALSE_OQ
#define _CMP_NEQ_OQ EASYSIMD_CMP_NEQ_OQ
#define _CMP_GE_OS EASYSIMD_CMP_GE_OS
#define _CMP_GT_OS EASYSIMD_CMP_GT_OS
#define _CMP_TRUE_UQ EASYSIMD_CMP_TRUE_UQ
#define _CMP_EQ_OS EASYSIMD_CMP_EQ_OS
#define _CMP_LT_OQ EASYSIMD_CMP_LT_OQ
#define _CMP_LE_OQ EASYSIMD_CMP_LE_OQ
#define _CMP_UNORD_S EASYSIMD_CMP_UNORD_S
#define _CMP_NEQ_US EASYSIMD_CMP_NEQ_US
#define _CMP_NLT_UQ EASYSIMD_CMP_NLT_UQ
#define _CMP_NLE_UQ EASYSIMD_CMP_NLE_UQ
#define _CMP_ORD_S EASYSIMD_CMP_ORD_S
#define _CMP_EQ_US EASYSIMD_CMP_EQ_US
#define _CMP_NGE_UQ EASYSIMD_CMP_NGE_UQ
#define _CMP_NGT_UQ EASYSIMD_CMP_NGT_UQ
#define _CMP_FALSE_OS EASYSIMD_CMP_FALSE_OS
#define _CMP_NEQ_OS EASYSIMD_CMP_NEQ_OS
#define _CMP_GE_OQ EASYSIMD_CMP_GE_OQ
#define _CMP_GT_OQ EASYSIMD_CMP_GT_OQ
#define _CMP_TRUE_US EASYSIMD_CMP_TRUE_US
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_castps_pd (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castps_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32[EASYSIMD_SV_INDEX_0];
    r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32[EASYSIMD_SV_INDEX_1];
    return r;
  #else
    /*Cast vector of type __m256 to type __m256d */
    return *HEDLEY_REINTERPRET_CAST(easysimd__m256d*, &a);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castps_pd
  #define _mm256_castps_pd(a) easysimd_mm256_castps_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_castps_si256 (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castps_si256(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = a.sve_u32[EASYSIMD_SV_INDEX_0];
    r.sve_u32[EASYSIMD_SV_INDEX_1] = a.sve_u32[EASYSIMD_SV_INDEX_1];
    return r;
  #else
    return *HEDLEY_REINTERPRET_CAST(easysimd__m256i*, &a);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castps_si256
  #define _mm256_castps_si256(a) easysimd_mm256_castps_si256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_castsi256_pd (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castsi256_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = a.sve_u32[EASYSIMD_SV_INDEX_0];
    r.sve_u32[EASYSIMD_SV_INDEX_1] = a.sve_u32[EASYSIMD_SV_INDEX_1];
    return r;
  #else
    return *HEDLEY_REINTERPRET_CAST(easysimd__m256d*, &a);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castsi256_pd
  #define _mm256_castsi256_pd(a) easysimd_mm256_castsi256_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_castsi256_ps (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castsi256_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svreinterpret_f32_s32(a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svreinterpret_f32_s32(a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 res;
    res.m128[0].neon_f32 = vreinterpretq_f32_s32(a.m128[0].neon_i32);
    res.m128[1].neon_f32 = vreinterpretq_f32_s32(a.m128[1].neon_i32);
    return res;
  #else
    return *HEDLEY_REINTERPRET_CAST(easysimd__m256*, &a);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castsi256_ps
  #define _mm256_castsi256_ps(a) easysimd_mm256_castsi256_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_castpd_ps (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castpd_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32[EASYSIMD_SV_INDEX_0];
  r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32[EASYSIMD_SV_INDEX_1];
  return r;
  #else
    return *HEDLEY_REINTERPRET_CAST(easysimd__m256*, &a);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castpd_ps
  #define _mm256_castpd_ps(a) easysimd_mm256_castpd_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_castpd_si256 (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castpd_si256(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = a.sve_u64[EASYSIMD_SV_INDEX_0];
    r.sve_u64[EASYSIMD_SV_INDEX_1] = a.sve_u64[EASYSIMD_SV_INDEX_1];
    return r;
  #else
    return *HEDLEY_REINTERPRET_CAST(easysimd__m256i*, &a);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castpd_si256
  #define _mm256_castpd_si256(a) easysimd_mm256_castpd_si256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_setzero_si256 (void) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_setzero_si256();
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32(0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = r.m128i[1].neon_i32 = vdupq_n_s32(0);
    return r;
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_setzero_si128();
      r_.m128i[1] = easysimd_mm_setzero_si128();
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = 0;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setzero_si256
  #define _mm256_setzero_si256() easysimd_mm256_setzero_si256()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_setzero_ps (void) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_setzero_ps();
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = r.sve_f32[EASYSIMD_SV_INDEX_1] = svdup_n_f32(0.0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0].neon_f32 = r.m128[1].neon_f32 = vdupq_n_f32(0.0f);
    return r;
  #else 
    return easysimd_mm256_castsi256_ps(easysimd_mm256_setzero_si256());
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setzero_ps
  #define _mm256_setzero_ps() easysimd_mm256_setzero_ps()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_setzero_pd (void) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_setzero_pd();
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = r.sve_f64[EASYSIMD_SV_INDEX_1] = svdup_n_f64(0.0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    r.m128d[0].neon_f64 = r.m128d[1].neon_f64 = vdupq_n_f64(0.0);
    return r;
  #else
    return easysimd_mm256_castsi256_pd(easysimd_mm256_setzero_si256());
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setzero_pd
  #define _mm256_setzero_pd() easysimd_mm256_setzero_pd()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_not_ps(easysimd__m256 a) {
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    r_.i32 = ~a_.i32;
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    r_.m128[0] = easysimd_x_mm_not_ps(a_.m128[0]);
    r_.m128[1] = easysimd_x_mm_not_ps(a_.m128[1]);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = ~(a_.i32[i]);
    }
  #endif

  return easysimd__m256_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_select_ps(easysimd__m256 a, easysimd__m256 b, easysimd__m256 mask) {
  /* This function is for when you want to blend two elements together
   * according to a mask.  It is similar to _mm256_blendv_ps, except that
   * it is undefined whether the blend is based on the highest bit in
   * each lane (like blendv) or just bitwise operations.  This allows
   * us to implement the function efficiently everywhere.
   *
   * Basically, you promise that all the lanes in mask are either 0 or
   * ~0. */
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_blendv_ps(a, b, mask);
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b),
      mask_ = easysimd__m256_to_private(mask);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 ^ ((a_.i32 ^ b_.i32) & mask_.i32);
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
      r_.m128[0] = easysimd_x_mm_select_ps(a_.m128[0], b_.m128[0], mask_.m128[0]);
      r_.m128[1] = easysimd_x_mm_select_ps(a_.m128[1], b_.m128[1], mask_.m128[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] ^ ((a_.i32[i] ^ b_.i32[i]) & mask_.i32[i]);
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_not_pd(easysimd__m256d a) {
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a);

  #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    r_.i64 = ~a_.i64;
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    r_.m128d[0] = easysimd_x_mm_not_pd(a_.m128d[0]);
    r_.m128d[1] = easysimd_x_mm_not_pd(a_.m128d[1]);
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
      r_.i64[i] = ~(a_.i64[i]);
    }
  #endif

  return easysimd__m256d_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_select_pd(easysimd__m256d a, easysimd__m256d b, easysimd__m256d mask) {
  /* This function is for when you want to blend two elements together
   * according to a mask.  It is similar to _mm256_blendv_pd, except that
   * it is undefined whether the blend is based on the highest bit in
   * each lane (like blendv) or just bitwise operations.  This allows
   * us to implement the function efficiently everywhere.
   *
   * Basically, you promise that all the lanes in mask are either 0 or
   * ~0. */
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_blendv_pd(a, b, mask);
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b),
      mask_ = easysimd__m256d_to_private(mask);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i64 = a_.i64 ^ ((a_.i64 ^ b_.i64) & mask_.i64);
    #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
      r_.m128d[0] = easysimd_x_mm_select_pd(a_.m128d[0], b_.m128d[0], mask_.m128d[0]);
      r_.m128d[1] = easysimd_x_mm_select_pd(a_.m128d[1], b_.m128d[1], mask_.m128d[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a_.i64[i] ^ ((a_.i64[i] ^ b_.i64[i]) & mask_.i64[i]);
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_setone_si256 (void) {
  easysimd__m256i_private r_;

#if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
  __typeof__(r_.i32f) rv = { 0, };
  r_.i32f = ~rv;
#elif defined(EASYSIMD_X86_AVX2_NATIVE)
  __m256i t = _mm256_setzero_si256();
  r_.n = _mm256_cmpeq_epi32(t, t);
#else
  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
    r_.i32f[i] = ~HEDLEY_STATIC_CAST(int_fast32_t, 0);
  }
#endif

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_setone_ps (void) {
  return easysimd_mm256_castsi256_ps(easysimd_x_mm256_setone_si256());
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_setone_pd (void) {
  return easysimd_mm256_castsi256_pd(easysimd_x_mm256_setone_si256());
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_set_epi8 (int8_t e31, int8_t e30, int8_t e29, int8_t e28,
                      int8_t e27, int8_t e26, int8_t e25, int8_t e24,
                      int8_t e23, int8_t e22, int8_t e21, int8_t e20,
                      int8_t e19, int8_t e18, int8_t e17, int8_t e16,
                      int8_t e15, int8_t e14, int8_t e13, int8_t e12,
                      int8_t e11, int8_t e10, int8_t  e9, int8_t  e8,
                      int8_t  e7, int8_t  e6, int8_t  e5, int8_t  e4,
                      int8_t  e3, int8_t  e2, int8_t  e1, int8_t  e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set_epi8(e31, e30, e29, e28, e27, e26, e25, e24,
                           e23, e22, e21, e20, e19, e18, e17, e16,
                           e15, e14, e13, e12, e11, e10,  e9,  e8,
                            e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svdupq_n_s8( e0,  e1,  e2,  e3,  e4,  e5,  e6,  e7,
                                              e8,  e9, e10, e11, e12, e13, e14, e15);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svdupq_n_s8(e16, e17, e18, e19, e20, e21, e22, e23,
                                             e24, e25, e26, e27, e28, e29, e30, e31);
    return r;
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_set_epi8(
        e15, e14, e13, e12, e11, e10,  e9,  e8,
        e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
      r_.m128i[1] = easysimd_mm_set_epi8(
        e31, e30, e29, e28, e27, e26, e25, e24,
        e23, e22, e21, e20, e19, e18, e17, e16);
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
      r_.i8[16] = e16;
      r_.i8[17] = e17;
      r_.i8[18] = e18;
      r_.i8[19] = e19;
      r_.i8[20] = e20;
      r_.i8[21] = e21;
      r_.i8[22] = e22;
      r_.i8[23] = e23;
      r_.i8[24] = e24;
      r_.i8[25] = e25;
      r_.i8[26] = e26;
      r_.i8[27] = e27;
      r_.i8[28] = e28;
      r_.i8[29] = e29;
      r_.i8[30] = e30;
      r_.i8[31] = e31;
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set_epi8
  #define _mm256_set_epi8(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) \
  easysimd_mm256_set_epi8(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_set_epi16 (int16_t e15, int16_t e14, int16_t e13, int16_t e12,
                       int16_t e11, int16_t e10, int16_t  e9, int16_t  e8,
                       int16_t  e7, int16_t  e6, int16_t  e5, int16_t  e4,
                       int16_t  e3, int16_t  e2, int16_t  e1, int16_t  e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set_epi16(e15, e14, e13, e12, e11, e10,  e9,  e8,
                            e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svdupq_n_s16( e0,  e1,  e2,  e3,  e4,  e5,  e6,  e7);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svdupq_n_s16( e8,  e9, e10, e11, e12, e13, e14, e15);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    SET16x8(r.m128i[0], e0,  e1,  e2,  e3,  e4,  e5,  e6,  e7)
    SET16x8(r.m128i[1], e8,  e9, e10, e11, e12, e13, e14, e15)
    return r;
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_set_epi16( e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
      r_.m128i[1] = easysimd_mm_set_epi16(e15, e14, e13, e12, e11, e10,  e9,  e8);
    #else
      r_.i16[ 0] =  e0;
      r_.i16[ 1] =  e1;
      r_.i16[ 2] =  e2;
      r_.i16[ 3] =  e3;
      r_.i16[ 4] =  e4;
      r_.i16[ 5] =  e5;
      r_.i16[ 6] =  e6;
      r_.i16[ 7] =  e7;
      r_.i16[ 8] =  e8;
      r_.i16[ 9] =  e9;
      r_.i16[10] = e10;
      r_.i16[11] = e11;
      r_.i16[12] = e12;
      r_.i16[13] = e13;
      r_.i16[14] = e14;
      r_.i16[15] = e15;
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set_epi16
  #define _mm256_set_epi16(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) \
  easysimd_mm256_set_epi16(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_set_epi32 (int32_t e7, int32_t e6, int32_t e5, int32_t e4,
                       int32_t e3, int32_t e2, int32_t e1, int32_t e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdupq_n_s32(e0, e1, e2, e3);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdupq_n_s32(e4, e5, e6, e7);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    SET32x4(res.m128i[0].neon_i32, e0, e1, e2, e3);
    SET32x4(res.m128i[1].neon_i32, e4, e5, e6, e7);
    return res;
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_set_epi32(e3, e2, e1, e0);
      r_.m128i[1] = easysimd_mm_set_epi32(e7, e6, e5, e4);
    #else
      r_.i32[ 0] =  e0;
      r_.i32[ 1] =  e1;
      r_.i32[ 2] =  e2;
      r_.i32[ 3] =  e3;
      r_.i32[ 4] =  e4;
      r_.i32[ 5] =  e5;
      r_.i32[ 6] =  e6;
      r_.i32[ 7] =  e7;
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set_epi32
  #define _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0) \
  easysimd_mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_set_epi64x (int64_t  e3, int64_t  e2, int64_t  e1, int64_t  e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set_epi64x(e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svdupq_n_s64(e0, e1);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svdupq_n_s64(e2, e3);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    SET64x2(res.m128i[0].neon_i64, e0, e1);
    SET64x2(res.m128i[1].neon_i64, e2, e3);
    return res;
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_set_epi64x(e1, e0);
      r_.m128i[1] = easysimd_mm_set_epi64x(e3, e2);
    #else
      r_.i64[0] = e0;
      r_.i64[1] = e1;
      r_.i64[2] = e2;
      r_.i64[3] = e3;
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set_epi64x
  #define _mm256_set_epi64x(e3, e2, e1, e0) easysimd_mm256_set_epi64x(e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_set_epu8 (uint8_t e31, uint8_t e30, uint8_t e29, uint8_t e28,
                        uint8_t e27, uint8_t e26, uint8_t e25, uint8_t e24,
                        uint8_t e23, uint8_t e22, uint8_t e21, uint8_t e20,
                        uint8_t e19, uint8_t e18, uint8_t e17, uint8_t e16,
                        uint8_t e15, uint8_t e14, uint8_t e13, uint8_t e12,
                        uint8_t e11, uint8_t e10, uint8_t  e9, uint8_t  e8,
                        uint8_t  e7, uint8_t  e6, uint8_t  e5, uint8_t  e4,
                        uint8_t  e3, uint8_t  e2, uint8_t  e1, uint8_t  e0) {
  easysimd__m256i_private r_;

  r_.u8[ 0] =  e0;
  r_.u8[ 1] =  e1;
  r_.u8[ 2] =  e2;
  r_.u8[ 3] =  e3;
  r_.u8[ 4] =  e4;
  r_.u8[ 5] =  e5;
  r_.u8[ 6] =  e6;
  r_.u8[ 7] =  e7;
  r_.u8[ 8] =  e8;
  r_.u8[ 9] =  e9;
  r_.u8[10] = e10;
  r_.u8[11] = e11;
  r_.u8[12] = e12;
  r_.u8[13] = e13;
  r_.u8[14] = e14;
  r_.u8[15] = e15;
  r_.u8[16] = e16;
  r_.u8[17] = e17;
  r_.u8[18] = e18;
  r_.u8[19] = e19;
  r_.u8[20] = e20;
  r_.u8[20] = e20;
  r_.u8[21] = e21;
  r_.u8[22] = e22;
  r_.u8[23] = e23;
  r_.u8[24] = e24;
  r_.u8[25] = e25;
  r_.u8[26] = e26;
  r_.u8[27] = e27;
  r_.u8[28] = e28;
  r_.u8[29] = e29;
  r_.u8[30] = e30;
  r_.u8[31] = e31;

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_set_epu16 (uint16_t e15, uint16_t e14, uint16_t e13, uint16_t e12,
                       uint16_t e11, uint16_t e10, uint16_t  e9, uint16_t  e8,
                       uint16_t  e7, uint16_t  e6, uint16_t  e5, uint16_t  e4,
                       uint16_t  e3, uint16_t  e2, uint16_t  e1, uint16_t  e0) {
  easysimd__m256i_private r_;

  r_.u16[ 0] =  e0;
  r_.u16[ 1] =  e1;
  r_.u16[ 2] =  e2;
  r_.u16[ 3] =  e3;
  r_.u16[ 4] =  e4;
  r_.u16[ 5] =  e5;
  r_.u16[ 6] =  e6;
  r_.u16[ 7] =  e7;
  r_.u16[ 8] =  e8;
  r_.u16[ 9] =  e9;
  r_.u16[10] = e10;
  r_.u16[11] = e11;
  r_.u16[12] = e12;
  r_.u16[13] = e13;
  r_.u16[14] = e14;
  r_.u16[15] = e15;

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_set_epu32 (uint32_t e7, uint32_t e6, uint32_t e5, uint32_t e4,
                         uint32_t e3, uint32_t e2, uint32_t e1, uint32_t e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set_epi32(HEDLEY_STATIC_CAST(int32_t, e7), HEDLEY_STATIC_CAST(int32_t, e6), HEDLEY_STATIC_CAST(int32_t, e5), HEDLEY_STATIC_CAST(int32_t, e4),
                            HEDLEY_STATIC_CAST(int32_t, e3), HEDLEY_STATIC_CAST(int32_t, e2), HEDLEY_STATIC_CAST(int32_t, e1), HEDLEY_STATIC_CAST(int32_t, e0));
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_set_epi32(HEDLEY_STATIC_CAST(int32_t, e3), HEDLEY_STATIC_CAST(int32_t, e2), HEDLEY_STATIC_CAST(int32_t, e1), HEDLEY_STATIC_CAST(int32_t, e0));
      r_.m128i[1] = easysimd_mm_set_epi32(HEDLEY_STATIC_CAST(int32_t, e7), HEDLEY_STATIC_CAST(int32_t, e6), HEDLEY_STATIC_CAST(int32_t, e5), HEDLEY_STATIC_CAST(int32_t, e4));
    #else
      r_.u32[ 0] =  e0;
      r_.u32[ 1] =  e1;
      r_.u32[ 2] =  e2;
      r_.u32[ 3] =  e3;
      r_.u32[ 4] =  e4;
      r_.u32[ 5] =  e5;
      r_.u32[ 6] =  e6;
      r_.u32[ 7] =  e7;
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_set_epu64x (uint64_t  e3, uint64_t  e2, uint64_t  e1, uint64_t  e0) {
  easysimd__m256i_private r_;

  r_.u64[0] = e0;
  r_.u64[1] = e1;
  r_.u64[2] = e2;
  r_.u64[3] = e3;

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_set_ps (easysimd_float32 e7, easysimd_float32 e6, easysimd_float32 e5, easysimd_float32 e4,
                    easysimd_float32 e3, easysimd_float32 e2, easysimd_float32 e1, easysimd_float32 e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svdupq_n_f32(e0, e1, e2, e3);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svdupq_n_f32(e4, e5, e6, e7);
    return r; 
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    SET32x4(r.m128[0].neon_f32, e0, e1, e2, e3);
    SET32x4(r.m128[1].neon_f32, e4, e5, e6, e7);
    return r;  
  #else
    easysimd__m256_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_set_ps(e3, e2, e1, e0);
      r_.m128[1] = easysimd_mm_set_ps(e7, e6, e5, e4);
    #else
      r_.f32[0] = e0;
      r_.f32[1] = e1;
      r_.f32[2] = e2;
      r_.f32[3] = e3;
      r_.f32[4] = e4;
      r_.f32[5] = e5;
      r_.f32[6] = e6;
      r_.f32[7] = e7;
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set_ps
  #define _mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0) \
  easysimd_mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_set_pd (easysimd_float64 e3, easysimd_float64 e2, easysimd_float64 e1, easysimd_float64 e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set_pd(e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svdupq_n_f64(e0, e1);  //must be 128 bit 
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svdupq_n_f64(e2, e3);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    SET64x2(r.m128d[0].neon_f64, e0, e1);
    SET64x2(r.m128d[1].neon_f64, e2, e3);
    return r;
  #else
    easysimd__m256d_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_set_pd(e1, e0);
      r_.m128d[1] = easysimd_mm_set_pd(e3, e2);
    #else
      r_.f64[0] = e0;
      r_.f64[1] = e1;
      r_.f64[2] = e2;
      r_.f64[3] = e3;
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set_pd
  #define _mm256_set_pd(e3, e2, e1, e0) \
  easysimd_mm256_set_pd(e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_set_m128 (easysimd__m128 e1, easysimd__m128 e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_insertf128_ps(_mm256_castps128_ps256(e0), e1, 1);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = e0.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_1] = e1.sve_f32;
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m128_private
      e1_ = easysimd__m128_to_private(e1),
      e0_ = easysimd__m128_to_private(e0);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128_private[0] = e0_;
      r_.m128_private[1] = e1_;
    #elif defined(EASYSIMD_HAVE_INT128_)
      r_.i128[0] = e0_.i128[0];
      r_.i128[1] = e1_.i128[0];
    #else
      r_.i64[0] = e0_.i64[0];
      r_.i64[1] = e0_.i64[1];
      r_.i64[2] = e1_.i64[0];
      r_.i64[3] = e1_.i64[1];
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set_m128
  #define _mm256_set_m128(e1, e0) easysimd_mm256_set_m128(e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_set_m128d (easysimd__m128d e1, easysimd__m128d e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(e0), e1, 1);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = e0.sve_f64;
    r.sve_f64[EASYSIMD_SV_INDEX_1] = e1.sve_f64;
    return r;
  #else
    easysimd__m256d_private r_;
    easysimd__m128d_private
      e1_ = easysimd__m128d_to_private(e1),
      e0_ = easysimd__m128d_to_private(e0);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d_private[0] = e0_;
      r_.m128d_private[1] = e1_;
    #else
      r_.i64[0] = e0_.i64[0];
      r_.i64[1] = e0_.i64[1];
      r_.i64[2] = e1_.i64[0];
      r_.i64[3] = e1_.i64[1];
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set_m128d
  #define _mm256_set_m128d(e1, e0) easysimd_mm256_set_m128d(e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_set_m128i (easysimd__m128i e1, easysimd__m128i e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_insertf128_si256(_mm256_castsi128_si256(e0), e1, 1);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = e0.sve_i64;
    r.sve_i64[EASYSIMD_SV_INDEX_1] = e1.sve_i64;
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0] = e0;
    res.m128i[1] = e1;
    return res;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private
      e1_ = easysimd__m128i_to_private(e1),
      e0_ = easysimd__m128i_to_private(e0);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i_private[0] = e0_;
      r_.m128i_private[1] = e1_;
    #else
      r_.i64[0] = e0_.i64[0];
      r_.i64[1] = e0_.i64[1];
      r_.i64[2] = e1_.i64[0];
      r_.i64[3] = e1_.i64[1];
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set_m128i
  #define _mm256_set_m128i(e1, e0) easysimd_mm256_set_m128i(e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_set1_epi8 (int8_t a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set1_epi8(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = r.sve_i8[EASYSIMD_SV_INDEX_1] = svdup_n_s8(a);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_i8 = res.m128i[1].neon_i8 = vdupq_n_s8(a);
    return res;
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_set1_epi8(a);
      r_.m128i[1] = easysimd_mm_set1_epi8(a);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i8) / sizeof(r_.i8[0])) ; i++) {
        r_.i8[i] = a;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set1_epi8
  #define _mm256_set1_epi8(a) easysimd_mm256_set1_epi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_set1_epi16 (int16_t a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set1_epi16(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = r.sve_i16[EASYSIMD_SV_INDEX_1] = svdup_n_s16(a);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_i16 = res.m128i[1].neon_i16 = vdupq_n_s16(a);
    return res;
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_set1_epi16(a);
      r_.m128i[1] = easysimd_mm_set1_epi16(a);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = a;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set1_epi16
  #define _mm256_set1_epi16(a) easysimd_mm256_set1_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_set1_epi32 (int32_t a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set1_epi32(a);
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32(a);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = r.m128i[1].neon_i32 = vdupq_n_s32(a);
    return r;
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_set1_epi32(a);
      r_.m128i[1] = easysimd_mm_set1_epi32(a);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set1_epi32
  #define _mm256_set1_epi32(a) easysimd_mm256_set1_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_set1_epi64x (int64_t a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set1_epi64x(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = r.sve_i64[EASYSIMD_SV_INDEX_1] = svdup_n_s64(a);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_i64 = res.m128i[1].neon_i64 = vdupq_n_s64(a);
    return res;
  #else
    easysimd__m256i_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128i[0] = easysimd_mm_set1_epi64x(a);
      r_.m128i[1] = easysimd_mm_set1_epi64x(a);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i64) / sizeof(r_.i64[0])) ; i++) {
        r_.i64[i] = a;
      }
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set1_epi64x
  #define _mm256_set1_epi64x(a) easysimd_mm256_set1_epi64x(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_set1_ps (easysimd_float32 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set1_ps(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0].neon_f32 = r.m128[1].neon_f32 = vdupq_n_f32(a);
    return r;
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = r.sve_f32[EASYSIMD_SV_INDEX_1] = svdup_n_f32(a);
    return r;
  #else
    easysimd__m256_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_set1_ps(a);
      r_.m128[1] = easysimd_mm_set1_ps(a);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a;
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set1_ps
  #define _mm256_set1_ps(a) easysimd_mm256_set1_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_set1_pd (easysimd_float64 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_set1_pd(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    r.m128d[0].neon_f64 = r.m128d[1].neon_f64 = vdupq_n_f64(a);
    return r;
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = r.sve_f64[EASYSIMD_SV_INDEX_1] = svdup_n_f64(a);
    return r;
  #else
    easysimd__m256d_private r_;

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_set1_pd(a);
      r_.m128d[1] = easysimd_mm_set1_pd(a);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a;
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_set1_pd
  #define _mm256_set1_pd(a) easysimd_mm256_set1_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_deinterleaveeven_epi16 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    r_.m128i[0] = easysimd_x_mm_deinterleaveeven_epi16(a_.m128i[0], b_.m128i[0]);
    r_.m128i[1] = easysimd_x_mm_deinterleaveeven_epi16(a_.m128i[1], b_.m128i[1]);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.i16 = EASYSIMD_SHUFFLE_VECTOR_(16, 32, a_.i16, b_.i16, 0, 2, 4, 6, 16, 18, 20, 22, 8, 10, 12, 14, 24, 26, 28, 30);
  #else
    const size_t halfway_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 2;
    const size_t quarter_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 4;
    for (size_t i = 0 ; i < quarter_point ; i++) {
      r_.i16[i] = a_.i16[2 * i];
      r_.i16[i + quarter_point] = b_.i16[2 * i];
      r_.i16[halfway_point + i] = a_.i16[halfway_point + 2 * i];
      r_.i16[halfway_point + i + quarter_point] = b_.i16[halfway_point + 2 * i];
    }
  #endif

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_deinterleaveodd_epi16 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    r_.m128i[0] = easysimd_x_mm_deinterleaveodd_epi16(a_.m128i[0], b_.m128i[0]);
    r_.m128i[1] = easysimd_x_mm_deinterleaveodd_epi16(a_.m128i[1], b_.m128i[1]);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.i16 = EASYSIMD_SHUFFLE_VECTOR_(16, 32, a_.i16, b_.i16, 1, 3, 5, 7, 17, 19, 21, 23, 9, 11, 13, 15, 25, 27, 29, 31);
  #else
    const size_t halfway_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 2;
    const size_t quarter_point = (sizeof(r_.i16) / sizeof(r_.i16[0])) / 4;
    for (size_t i = 0 ; i < quarter_point ; i++) {
      r_.i16[i] = a_.i16[2 * i + 1];
      r_.i16[i + quarter_point] = b_.i16[2 * i + 1];
      r_.i16[halfway_point + i] = a_.i16[halfway_point + 2 * i + 1];
      r_.i16[halfway_point + i + quarter_point] = b_.i16[halfway_point + 2 * i + 1];
    }
  #endif

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_deinterleaveeven_epi32 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    r_.m128i[0] = easysimd_x_mm_deinterleaveeven_epi32(a_.m128i[0], b_.m128i[0]);
    r_.m128i[1] = easysimd_x_mm_deinterleaveeven_epi32(a_.m128i[1], b_.m128i[1]);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.i32 = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.i32, b_.i32, 0, 2, 8, 10, 4, 6, 12, 14);
  #else
    const size_t halfway_point = (sizeof(r_.i32) / sizeof(r_.i32[0])) / 2;
    const size_t quarter_point = (sizeof(r_.i32) / sizeof(r_.i32[0])) / 4;
    for (size_t i = 0 ; i < quarter_point ; i++) {
      r_.i32[i] = a_.i32[2 * i];
      r_.i32[i + quarter_point] = b_.i32[2 * i];
      r_.i32[halfway_point + i] = a_.i32[halfway_point + 2 * i];
      r_.i32[halfway_point + i + quarter_point] = b_.i32[halfway_point + 2 * i];
    }
  #endif

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_x_mm256_deinterleaveodd_epi32 (easysimd__m256i a, easysimd__m256i b) {
  easysimd__m256i_private
    r_,
    a_ = easysimd__m256i_to_private(a),
    b_ = easysimd__m256i_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    r_.m128i[0] = easysimd_x_mm_deinterleaveodd_epi32(a_.m128i[0], b_.m128i[0]);
    r_.m128i[1] = easysimd_x_mm_deinterleaveodd_epi32(a_.m128i[1], b_.m128i[1]);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.i32 = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.i32, b_.i32, 1, 3, 9, 11, 5, 7, 13, 15);
  #else
    const size_t halfway_point = (sizeof(r_.i32) / sizeof(r_.i32[0])) / 2;
    const size_t quarter_point = (sizeof(r_.i32) / sizeof(r_.i32[0])) / 4;
    for (size_t i = 0 ; i < quarter_point ; i++) {
      r_.i32[i] = a_.i32[2 * i + 1];
      r_.i32[i + quarter_point] = b_.i32[2 * i + 1];
      r_.i32[halfway_point + i] = a_.i32[halfway_point + 2 * i + 1];
      r_.i32[halfway_point + i + quarter_point] = b_.i32[halfway_point + 2 * i + 1];
    }
  #endif

  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_deinterleaveeven_ps (easysimd__m256 a, easysimd__m256 b) {
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a),
    b_ = easysimd__m256_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    r_.m128[0] = easysimd_x_mm_deinterleaveeven_ps(a_.m128[0], b_.m128[0]);
    r_.m128[1] = easysimd_x_mm_deinterleaveeven_ps(a_.m128[1], b_.m128[1]);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.f32, b_.f32, 0, 2, 8, 10, 4, 6, 12, 14);
  #else
    const size_t halfway_point = (sizeof(r_.f32) / sizeof(r_.f32[0])) / 2;
    const size_t quarter_point = (sizeof(r_.f32) / sizeof(r_.f32[0])) / 4;
    for (size_t i = 0 ; i < quarter_point ; i++) {
      r_.f32[i] = a_.f32[2 * i];
      r_.f32[i + quarter_point] = b_.f32[2 * i];
      r_.f32[halfway_point + i] = a_.f32[halfway_point + 2 * i];
      r_.f32[halfway_point + i + quarter_point] = b_.f32[halfway_point + 2 * i];
    }
  #endif

  return easysimd__m256_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_deinterleaveodd_ps (easysimd__m256 a, easysimd__m256 b) {
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a),
    b_ = easysimd__m256_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    r_.m128[0] = easysimd_x_mm_deinterleaveodd_ps(a_.m128[0], b_.m128[0]);
    r_.m128[1] = easysimd_x_mm_deinterleaveodd_ps(a_.m128[1], b_.m128[1]);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.f32, b_.f32, 1, 3, 9, 11, 5, 7, 13, 15);
  #else
    const size_t halfway_point = (sizeof(r_.f32) / sizeof(r_.f32[0])) / 2;
    const size_t quarter_point = (sizeof(r_.f32) / sizeof(r_.f32[0])) / 4;
    for (size_t i = 0 ; i < quarter_point ; i++) {
      r_.f32[i] = a_.f32[2 * i + 1];
      r_.f32[i + quarter_point] = b_.f32[2 * i + 1];
      r_.f32[halfway_point + i] = a_.f32[halfway_point + 2 * i + 1];
      r_.f32[halfway_point + i + quarter_point] = b_.f32[halfway_point + 2 * i + 1];
    }
  #endif

  return easysimd__m256_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_deinterleaveeven_pd (easysimd__m256d a, easysimd__m256d b) {
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a),
    b_ = easysimd__m256d_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    r_.m128d[0] = easysimd_x_mm_deinterleaveeven_pd(a_.m128d[0], b_.m128d[0]);
    r_.m128d[1] = easysimd_x_mm_deinterleaveeven_pd(a_.m128d[1], b_.m128d[1]);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 32, a_.f64, b_.f64, 0, 4, 2, 6);
  #else
    const size_t halfway_point = (sizeof(r_.f64) / sizeof(r_.f64[0])) / 2;
    const size_t quarter_point = (sizeof(r_.f64) / sizeof(r_.f64[0])) / 4;
    for (size_t i = 0 ; i < quarter_point ; i++) {
      r_.f64[i] = a_.f64[2 * i];
      r_.f64[i + quarter_point] = b_.f64[2 * i];
      r_.f64[halfway_point + i] = a_.f64[halfway_point + 2 * i];
      r_.f64[halfway_point + i + quarter_point] = b_.f64[halfway_point + 2 * i];
    }
  #endif

  return easysimd__m256d_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_deinterleaveodd_pd (easysimd__m256d a, easysimd__m256d b) {
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a),
    b_ = easysimd__m256d_to_private(b);

  #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
    r_.m128d[0] = easysimd_x_mm_deinterleaveodd_pd(a_.m128d[0], b_.m128d[0]);
    r_.m128d[1] = easysimd_x_mm_deinterleaveodd_pd(a_.m128d[1], b_.m128d[1]);
  #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
    r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 32, a_.f64, b_.f64, 1, 5, 3, 7);
  #else
    const size_t halfway_point = (sizeof(r_.f64) / sizeof(r_.f64[0])) / 2;
    const size_t quarter_point = (sizeof(r_.f64) / sizeof(r_.f64[0])) / 4;
    for (size_t i = 0 ; i < quarter_point ; i++) {
      r_.f64[i] = a_.f64[2 * i + 1];
      r_.f64[i + quarter_point] = b_.f64[2 * i + 1];
      r_.f64[halfway_point + i] = a_.f64[halfway_point + 2 * i + 1];
      r_.f64[halfway_point + i + quarter_point] = b_.f64[halfway_point + 2 * i + 1];
    }
  #endif

  return easysimd__m256d_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_abs_ps(easysimd__m256 a) {
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_fabsf(a_.f32[i]);
      }
    return easysimd__m256_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_abs_pd(easysimd__m256d a) {
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_fabs(a_.f64[i]);
      }
    return easysimd__m256d_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_add_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_add_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0].neon_f32 = vaddq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vaddq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);
      
    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_add_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_add_ps(a_.m128[1], b_.m128[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 + b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i] + b_.f32[i];
      }
    #endif
    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_add_ps
  #define _mm256_add_ps(a, b) easysimd_mm256_add_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_hadd_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_hadd_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svfloat32_t sv1, sv2;
    sv1 = svuzp1_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    sv2 = svuzp2_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_x(svptrue_b32(), sv1, sv2);
    sv1 = svuzp1_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    sv2 = svuzp2_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_x(svptrue_b32(), sv1, sv2);
    return r;
  #else
    return easysimd_mm256_add_ps(easysimd_x_mm256_deinterleaveeven_ps(a, b), easysimd_x_mm256_deinterleaveodd_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hadd_ps
  #define _mm256_hadd_ps(a, b) easysimd_mm256_hadd_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_add_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_add_pd(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    r.m128d[0].neon_f64 = vaddq_f64(a.m128d[0].neon_f64, b.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vaddq_f64(a.m128d[1].neon_f64, b.m128d[1].neon_f64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svadd_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_add_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_add_pd(a_.m128d[1], b_.m128d[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 + b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[i] + b_.f64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_add_pd
  #define _mm256_add_pd(a, b) easysimd_mm256_add_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_hadd_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_hadd_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svfloat64_t sv1, sv2;
    sv1 = svuzp1_f64(a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    sv2 = svuzp2_f64(a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svadd_f64_x(svptrue_b64(), sv1, sv2);
    sv1 = svuzp1_f64(a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    sv2 = svuzp2_f64(a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svadd_f64_x(svptrue_b64(), sv1, sv2);
    return r;
  #else
      return easysimd_mm256_add_pd(easysimd_x_mm256_deinterleaveeven_pd(a, b), easysimd_x_mm256_deinterleaveodd_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hadd_pd
  #define _mm256_hadd_pd(a, b) easysimd_mm256_hadd_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_addsub_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_addsub_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 c = easysimd_mm256_setzero_ps();
    __asm__ __volatile__ (
        "fsub %2.4s, %0.4s, %4.4s        \n\t"
        "fsub %3.4s, %1.4s, %5.4s        \n\t"
        "fadd %0.4s, %0.4s, %4.4s        \n\t"
        "fadd %1.4s, %1.4s, %5.4s        \n\t"
        "mov %2.s[1], %0.s[1]            \n\t"
        "mov %3.s[1], %1.s[1]            \n\t"
        "mov %2.s[3], %0.s[3]            \n\t"
        "mov %3.s[3], %1.s[3]            \n\t"
        :"+w"(a.m128[0].neon_f32), "+w"(a.m128[1].neon_f32), "+w"(c.m128[0].neon_f32), "+w"(c.m128[1].neon_f32)
        :"w"(b.m128[0].neon_f32), "w"(b.m128[1].neon_f32)
    );
    return c;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pgadd = svdupq_n_b32(0, 1, 0, 1),
             pgsub = svdupq_n_b32(1, 0, 1, 0),
             pg    = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svadd_f32_z(pg, 
                                              svadd_f32_z(pgadd, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]),
                                              svsub_f32_z(pgsub, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svadd_f32_z(pg, 
                                              svadd_f32_z(pgadd, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]),
                                              svsub_f32_z(pgsub, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);
    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_addsub_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_addsub_ps(a_.m128[1], b_.m128[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 2) {
        r_.f32[  i  ] = a_.f32[  i  ] - b_.f32[  i  ];
        r_.f32[i + 1] = a_.f32[i + 1] + b_.f32[i + 1];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_addsub_ps
  #define _mm256_addsub_ps(a, b) easysimd_mm256_addsub_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_addsub_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_addsub_pd(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d c = easysimd_mm256_setzero_pd();
    __asm__ __volatile__ (
        "fsub %2.2d, %0.2d, %4.2d        \n\t"
        "fsub %3.2d, %1.2d, %5.2d        \n\t"
        "fadd %0.2d, %0.2d, %4.2d        \n\t"
        "fadd %1.2d, %1.2d, %5.2d        \n\t"
        "mov %2.d[1], %0.d[1]            \n\t"
        "mov %3.d[1], %1.d[1]            \n\t"
        :"+w"(a.m128d[0].neon_f64), "+w"(a.m128d[1].neon_f64), "+w"(c.m128d[0].neon_f64), "+w"(c.m128d[1].neon_f64)
        :"w"(b.m128d[0].neon_f64), "w"(b.m128d[1].neon_f64)
    );
    return c;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pgadd = svdupq_n_b64(0, 1),
             pgsub = svdupq_n_b64(1, 0),
             pg    = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svadd_f64_z(pg, 
                                              svadd_f64_z(pgadd, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]),
                                              svsub_f64_z(pgsub, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svadd_f64_z(pg, 
                                              svadd_f64_z(pgadd, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]),
                                              svsub_f64_z(pgsub, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);
    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_addsub_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_addsub_pd(a_.m128d[1], b_.m128d[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
        r_.f64[  i  ] = a_.f64[  i  ] - b_.f64[  i  ];
        r_.f64[i + 1] = a_.f64[i + 1] + b_.f64[i + 1];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_addsub_pd
  #define _mm256_addsub_pd(a, b) easysimd_mm256_addsub_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_and_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_and_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svand_u32_x(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svand_u32_x(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_and_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_and_ps(a_.m128[1], b_.m128[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f & b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = a_.i32f[i] & b_.i32f[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_and_ps
  #define _mm256_and_ps(a, b) easysimd_mm256_and_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_and_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_and_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svand_u32_x(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svand_u32_x(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_and_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_and_pd(a_.m128d[1], b_.m128d[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f & b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = a_.i32f[i] & b_.i32f[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_and_pd
  #define _mm256_and_pd(a, b) easysimd_mm256_and_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_andnot_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_andnot_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_u32[EASYSIMD_SV_INDEX_0] = svbic_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_0], a.sve_u32[EASYSIMD_SV_INDEX_0]);
    a.sve_u32[EASYSIMD_SV_INDEX_1] = svbic_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_1], a.sve_u32[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_andnot_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_andnot_ps(a_.m128[1], b_.m128[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = ~a_.i32f & b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = ~a_.i32f[i] & b_.i32f[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_andnot_ps
  #define _mm256_andnot_ps(a, b) easysimd_mm256_andnot_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_andnot_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_andnot_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svbic_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_0], a.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svbic_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_1], a.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_andnot_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_andnot_pd(a_.m128d[1], b_.m128d[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = ~a_.i32f & b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32f) / sizeof(r_.i32f[0])) ; i++) {
        r_.i32f[i] = ~a_.i32f[i] & b_.i32f[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_andnot_pd
  #define _mm256_andnot_pd(a, b) easysimd_mm256_andnot_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_blend_ps (easysimd__m256 a, easysimd__m256 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(EASYSIMD_MASK_TO_B32(imm8, EASYSIMD_SV_INDEX_0), b.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(EASYSIMD_MASK_TO_B32(imm8, EASYSIMD_SV_INDEX_1), b.sve_f32[EASYSIMD_SV_INDEX_1], a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    uint32_t g_mask_epi32[4] __attribute__((aligned(16))) = {0x01, 0x02, 0x04, 0x08};
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    uint32x4_t vect_imm = vdupq_n_u32(imm8);
    uint32x4_t flag[2];
    flag[0] = vtstq_u32(vect_imm, vect_mask);
    flag[1] = vtstq_u32(vshrq_n_u32(vect_imm, 4), vect_mask);
    r.m128[0].neon_f32 = vbslq_f32(flag[0], b.m128[0].neon_f32, a.m128[0].neon_f32);
    r.m128[1].neon_f32 = vbslq_f32(flag[1], b.m128[1].neon_f32, a.m128[1].neon_f32);
    return r;
  #else
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a),
    b_ = easysimd__m256_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = ((imm8 >> i) & 1) ? b_.f32[i] : a_.f32[i];
  }
  return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_blend_ps(a, b, imm8) _mm256_blend_ps(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_blend_ps(a, b, imm8) \
      easysimd_mm256_set_m128( \
          easysimd_mm_blend_ps(easysimd_mm256_extractf128_ps(a, 1), easysimd_mm256_extractf128_ps(b, 1), (imm8) >> 4), \
          easysimd_mm_blend_ps(easysimd_mm256_extractf128_ps(a, 0), easysimd_mm256_extractf128_ps(b, 0), (imm8) & 0x0F))
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_blend_ps
  #define _mm256_blend_ps(a, b, imm8) easysimd_mm256_blend_ps(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_blend_pd (easysimd__m256d a, easysimd__m256d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    uint64_t g_mask_epi64[2] __attribute__((aligned(16))) = {0x01, 0x02};
    uint64x2_t vect_mask = vld1q_u64(g_mask_epi64);
    uint64x2_t vect_imm = vdupq_n_u64(imm8);
    uint64x2_t flag[2];
    flag[0] = vtstq_u64(vect_imm, vect_mask);
    flag[1] = vtstq_u64(vshrq_n_u64(vect_imm, 2), vect_mask);
    r.m128d[0].neon_f64 = vbslq_f64(flag[0], b.m128d[0].neon_f64, a.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vbslq_f64(flag[1], b.m128d[1].neon_f64, a.m128d[1].neon_f64);
    return r;
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(EASYSIMD_MASK_TO_B64(imm8, EASYSIMD_SV_INDEX_0), b.sve_f64[EASYSIMD_SV_INDEX_0], a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(EASYSIMD_MASK_TO_B64(imm8, EASYSIMD_SV_INDEX_1), b.sve_f64[EASYSIMD_SV_INDEX_1], a.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a),
    b_ = easysimd__m256d_to_private(b);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    r_.f64[i] = ((imm8 >> i) & 1) ? b_.f64[i] : a_.f64[i];
  }
  return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_blend_pd(a, b, imm8) _mm256_blend_pd(a, b, imm8)
#elif defined (EASYSIMD_ARM_SVE_NATIVE) || defined (EASYSIMD_ARM_NEON_A64V8_NATIVE)
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
#  define easysimd_mm256_blend_pd(a, b, imm8) \
      easysimd_mm256_set_m128d( \
          easysimd_mm_blend_pd(easysimd_mm256_extractf128_pd(a, 1), easysimd_mm256_extractf128_pd(b, 1), (imm8) >> 2), \
          easysimd_mm_blend_pd(easysimd_mm256_extractf128_pd(a, 0), easysimd_mm256_extractf128_pd(b, 0), (imm8) & 3))
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_blend_pd
  #define _mm256_blend_pd(a, b, imm8) easysimd_mm256_blend_pd(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_blendv_ps (easysimd__m256 a, easysimd__m256 b, easysimd__m256 mask) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_blendv_ps(a, b, mask);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    uint32x4_t vect_flag[2];
    vect_flag[0] = vcgeq_s32((int32x4_t)mask.m128[0].neon_f32, vdupq_n_s32(0));
    vect_flag[1] = vcgeq_s32((int32x4_t)mask.m128[1].neon_f32, vdupq_n_s32(0));
    r.m128[0].neon_f32 = vbslq_f32(vect_flag[0], a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vbslq_f32(vect_flag[1], a.m128[1].neon_f32, b.m128[1].neon_f32);
    return r;
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    easysimd_svbool_t pgm1 = svcmpeq_n_u32(pg, svlsr_n_u32_z(pg, mask.sve_u32[EASYSIMD_SV_INDEX_0], 31), 1);
    easysimd_svbool_t pgm2 = svcmpeq_n_u32(pg, svlsr_n_u32_z(pg, mask.sve_u32[EASYSIMD_SV_INDEX_1], 31), 1);

    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsel_f32(pgm1, b.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsel_f32(pgm2, b.sve_f32[EASYSIMD_SV_INDEX_1], a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b),
      mask_ = easysimd__m256_to_private(mask);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_blendv_ps(a_.m128[0], b_.m128[0], mask_.m128[0]);
      r_.m128[1] = easysimd_mm_blendv_ps(a_.m128[1], b_.m128[1], mask_.m128[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.f32[i] = (mask_.u32[i] & (UINT32_C(1) << 31)) ? b_.f32[i] : a_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_blendv_ps
  #define _mm256_blendv_ps(a, b, imm8) easysimd_mm256_blendv_ps(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_blendv_pd (easysimd__m256d a, easysimd__m256d b, easysimd__m256d mask) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_blendv_pd(a, b, mask);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    uint64x2_t vect_flag[2];
    vect_flag[0] = vcgeq_s64((int64x2_t)mask.m128d[0].neon_f64, vdupq_n_s64(0));
    vect_flag[1] = vcgeq_s64((int64x2_t)mask.m128d[1].neon_f64, vdupq_n_s64(0));
    r.m128d[0].neon_f64 = vbslq_f64(vect_flag[0], a.m128d[0].neon_f64, b.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vbslq_f64(vect_flag[1], a.m128d[1].neon_f64, b.m128d[1].neon_f64);
    return r;
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();

    easysimd_svbool_t pgm1 = svcmpeq_n_u64(pg, svlsr_n_u64_z(pg, mask.sve_u64[EASYSIMD_SV_INDEX_0], 63), 1);
    easysimd_svbool_t pgm2 = svcmpeq_n_u64(pg, svlsr_n_u64_z(pg, mask.sve_u64[EASYSIMD_SV_INDEX_1], 63), 1);

    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsel_f64(pgm1, b.sve_f64[EASYSIMD_SV_INDEX_0], a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsel_f64(pgm2, b.sve_f64[EASYSIMD_SV_INDEX_1], a.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b),
      mask_ = easysimd__m256d_to_private(mask);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_blendv_pd(a_.m128d[0], b_.m128d[0], mask_.m128d[0]);
      r_.m128d[1] = easysimd_mm_blendv_pd(a_.m128d[1], b_.m128d[1], mask_.m128d[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.f64[i] = (mask_.u64[i] & (UINT64_C(1) << 63)) ? b_.f64[i] : a_.f64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_blendv_pd
  #define _mm256_blendv_pd(a, b, imm8) easysimd_mm256_blendv_pd(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_broadcast_pd (easysimd__m128d const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_broadcast_pd(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_f64(svptrue_b64(), (float64_t const *)mem_addr);
    return r;
  #else
    easysimd__m256d_private r_;

    easysimd__m128d tmp = easysimd_mm_loadu_pd(HEDLEY_REINTERPRET_CAST(easysimd_float64 const*, mem_addr));
    r_.m128d[0] = tmp;
    r_.m128d[1] = tmp;

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_pd
  #define _mm256_broadcast_pd(mem_addr) easysimd_mm256_broadcast_pd(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_broadcast_ps (easysimd__m128 const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_broadcast_ps(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(svptrue_b32(), (float32_t const *)mem_addr);
    return r;
  #else
    easysimd__m256_private r_;

    easysimd__m128 tmp = easysimd_mm_loadu_ps(HEDLEY_REINTERPRET_CAST(easysimd_float32 const*, mem_addr));
    r_.m128[0] = tmp;
    r_.m128[1] = tmp;

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_ps
  #define _mm256_broadcast_ps(mem_addr) easysimd_mm256_broadcast_ps(HEDLEY_REINTERPRET_CAST(easysimd__m128 const*, mem_addr))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_broadcast_sd (easysimd_float64 const * a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_broadcast_sd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = r.sve_f64[EASYSIMD_SV_INDEX_1] = svdup_n_f64(*a);
    return r;
  #else
    return easysimd_mm256_set1_pd(*a);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_sd
  #define _mm256_broadcast_sd(mem_addr) easysimd_mm256_broadcast_sd(HEDLEY_REINTERPRET_CAST(double const*, mem_addr))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_broadcast_ss (easysimd_float32 const * a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_broadcast_ss(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdup_n_f32(*a);
    return r;
  #else
    return easysimd_mm_set1_ps(*a);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcast_ss
  #define _mm_broadcast_ss(a) easysimd_mm_broadcast_ss(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_broadcast_i32x2 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    return _mm_broadcast_i32x2(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = svtbl_s32(a.sve_i32, svdupq_n_u32(0, 1, 0, 1));
    return r;
  #else
    easysimd__m128i_private
      a_ = easysimd__m128i_to_private(a),
      r_;

    r_.i32[0] = a_.i32[0];
    r_.i32[1] = a_.i32[1];
    r_.i32[2] = a_.i32[0];
    r_.i32[3] = a_.i32[1];

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm_broadcast_i32x2
  #define _mm_broadcast_i32x2(a) easysimd_mm_broadcast_i32x2(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_broadcast_ss (easysimd_float32 const * a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_broadcast_ss(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = r.sve_f32[EASYSIMD_SV_INDEX_1] = svdup_n_f32(*a);
    return r;
  #else
    return easysimd_mm256_set1_ps(*a);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_broadcast_ss
  #define _mm256_broadcast_ss(mem_addr) easysimd_mm256_broadcast_ss(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_castpd128_pd256 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castpd128_pd256(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64;
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    r.m128d[0] = a;
    return r;
  #else
    easysimd__m256d_private r_;
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    r_.m128d_private[0] = a_;

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castpd128_pd256
  #define _mm256_castpd128_pd256(a) easysimd_mm256_castpd128_pd256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm256_castpd256_pd128 (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castpd256_pd128(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.m128d[0];
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128d[0];
  #else
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);
    return a_.m128d[0];
    #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castpd256_pd128
  #define _mm256_castpd256_pd128(a) easysimd_mm256_castpd256_pd128(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_castps128_ps256 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castps128_ps256(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32;
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0] = a;
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

      r_.m128_private[0] = a_;

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castps128_ps256
  #define _mm256_castps128_ps256(a) easysimd_mm256_castps128_ps256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_castps256_ps128 (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castps256_ps128(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.m128[0];
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128[0];
  #else
    easysimd__m256_private a_ = easysimd__m256_to_private(a);
    return a_.m128[0];
    #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castps256_ps128
  #define _mm256_castps256_ps128(a) easysimd_mm256_castps256_ps128(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_castsi128_si256 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castsi128_si256(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.m128i[0] = a;
    return r;
  #elif defined(EASYSIMD_ARM_NEON_NATIVE)
    easysimd__m256i res;
    res.m128i[0] = a;
    return res;
  #else
    easysimd__m256i_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);

    r_.m128i_private[0] = a_;
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castsi128_si256
  #define _mm256_castsi128_si256(a) easysimd_mm256_castsi128_si256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm256_castsi256_si128 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_castsi256_si128(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    r.sve_i32 = a.sve_i32[EASYSIMD_SV_INDEX_0];
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128i res;
    res.neon_i32 = a.m128i[0].neon_i32;
    return res;
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    return a_.m128i[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_castsi256_si128
  #define _mm256_castsi256_si128(a) easysimd_mm256_castsi256_si128(a)
#endif

#if defined(EASYSIMD_ARM_SVE_NATIVE)
/*ps*/
EASYSIMD_FUNCTION_ATTRIBUTES
svfloat32_t _round_ps_nearest(svfloat32_t svinput){
  return svrinta_f32_x(svptrue_b32(), svinput);
}

EASYSIMD_FUNCTION_ATTRIBUTES
svfloat32_t _round_ps_neg_inf(svfloat32_t svinput){
  return svrintm_f32_x(svptrue_b32(), svinput);
}

EASYSIMD_FUNCTION_ATTRIBUTES
svfloat32_t _round_ps_pos_inf(svfloat32_t svinput){
  return svrintp_f32_x(svptrue_b32(), svinput);
}

EASYSIMD_FUNCTION_ATTRIBUTES
svfloat32_t _round_ps_zero(svfloat32_t svinput){
  return svrintz_f32_x(svptrue_b32(), svinput);
}

EASYSIMD_FUNCTION_ATTRIBUTES
svfloat32_t _round_ps_current(svfloat32_t svinput){
  return svrintx_f32_x(svptrue_b32(), svinput);
}

typedef struct {
  int OP;
  svfloat32_t (*roundfun)(svfloat32_t);
} roundFunPs;

static roundFunPs roundfunlistps[] = {
  {EASYSIMD_MM_FROUND_TO_NEAREST_INT, _round_ps_nearest},
  {EASYSIMD_MM_FROUND_TO_NEG_INF,     _round_ps_neg_inf},
  {EASYSIMD_MM_FROUND_TO_POS_INF,     _round_ps_pos_inf},
  {EASYSIMD_MM_FROUND_TO_ZERO,        _round_ps_zero   },
  {EASYSIMD_MM_FROUND_CUR_DIRECTION,  _round_ps_current}
};

/*pd*/
EASYSIMD_FUNCTION_ATTRIBUTES
svfloat64_t _round_pd_nearest(svfloat64_t svinput){
  return svrinta_f64_x(svptrue_b64(), svinput);
}

EASYSIMD_FUNCTION_ATTRIBUTES
svfloat64_t _round_pd_neg_inf(svfloat64_t svinput){
  return svrintm_f64_x(svptrue_b64(), svinput);
}

EASYSIMD_FUNCTION_ATTRIBUTES
svfloat64_t _round_pd_pos_inf(svfloat64_t svinput){
  return svrintp_f64_x(svptrue_b64(), svinput);
}

EASYSIMD_FUNCTION_ATTRIBUTES
svfloat64_t _round_pd_zero(svfloat64_t svinput){
  return svrintz_f64_x(svptrue_b64(), svinput);
}

EASYSIMD_FUNCTION_ATTRIBUTES
svfloat64_t _round_pd_current(svfloat64_t svinput){
  return svrintx_f64_x(svptrue_b64(), svinput);
}

typedef struct {
  int OP;
  svfloat64_t (*roundfun)(svfloat64_t);
} roundFunPd;

static roundFunPd roundfunlistpd[] = {
  {EASYSIMD_MM_FROUND_TO_NEAREST_INT, _round_pd_nearest},
  {EASYSIMD_MM_FROUND_TO_NEG_INF,     _round_pd_neg_inf},
  {EASYSIMD_MM_FROUND_TO_POS_INF,     _round_pd_pos_inf},
  {EASYSIMD_MM_FROUND_TO_ZERO,        _round_pd_zero   },
  {EASYSIMD_MM_FROUND_CUR_DIRECTION,  _round_pd_current}
};
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_round_ps (easysimd__m256 a, const int rounding) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  r.sve_f32[EASYSIMD_SV_INDEX_0] = roundfunlistps[rounding & ~EASYSIMD_MM_FROUND_NO_EXC].roundfun(a.sve_f32[EASYSIMD_SV_INDEX_0]);
  r.sve_f32[EASYSIMD_SV_INDEX_1] = roundfunlistps[rounding & ~EASYSIMD_MM_FROUND_NO_EXC].roundfun(a.sve_f32[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a);

  switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC) {
    #if defined(easysimd_math_nearbyintf)
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_nearbyintf(a_.f32[i]);
        }
        break;
    #endif

    #if defined(easysimd_math_roundf)
      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_roundf(a_.f32[i]);
        }
        break;
    #endif

    #if defined(easysimd_math_floorf)
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_floorf(a_.f32[i]);
        }
        break;
    #endif

    #if defined(easysimd_math_ceilf)
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_ceilf(a_.f32[i]);
        }
        break;
    #endif

    #if defined(easysimd_math_truncf)
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_truncf(a_.f32[i]);
        }
        break;
    #endif

    default:
      HEDLEY_UNREACHABLE_RETURN(easysimd_mm256_undefined_ps());
  }

  return easysimd__m256_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_round_ps(a, rounding) _mm256_round_ps(a, rounding)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm256_round_ps(a, rounding) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m256_private \
      easysimd_mm256_round_ps_r_, \
      easysimd_mm256_round_ps_a_ = easysimd__m256_to_private(a); \
    \
    for (size_t easysimd_mm256_round_ps_i = 0 ; easysimd_mm256_round_ps_i < (sizeof(easysimd_mm256_round_ps_r_.m128) / sizeof(easysimd_mm256_round_ps_r_.m128[0])) ; easysimd_mm256_round_ps_i++) { \
      easysimd_mm256_round_ps_r_.m128[easysimd_mm256_round_ps_i] = easysimd_mm_round_ps(easysimd_mm256_round_ps_a_.m128[easysimd_mm256_round_ps_i], rounding); \
    } \
    \
    easysimd__m256_from_private(easysimd_mm256_round_ps_r_); \
  }))
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_round_ps
  #define _mm256_round_ps(a, rounding) easysimd_mm256_round_ps(a, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_round_pd (easysimd__m256d a, const int rounding) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256d r;
  r.sve_f64[EASYSIMD_SV_INDEX_0] = roundfunlistpd[rounding & ~EASYSIMD_MM_FROUND_NO_EXC].roundfun(a.sve_f64[EASYSIMD_SV_INDEX_0]);
  r.sve_f64[EASYSIMD_SV_INDEX_1] = roundfunlistpd[rounding & ~EASYSIMD_MM_FROUND_NO_EXC].roundfun(a.sve_f64[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a);

  switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC) {
    #if defined(easysimd_math_nearbyint)
      case EASYSIMD_MM_FROUND_CUR_DIRECTION:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_nearbyint(a_.f64[i]);
        }
        break;
    #endif

    #if defined(easysimd_math_round)
      case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_round(a_.f64[i]);
        }
        break;
    #endif

    #if defined(easysimd_math_floor)
      case EASYSIMD_MM_FROUND_TO_NEG_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_floor(a_.f64[i]);
        }
        break;
    #endif

    #if defined(easysimd_math_ceil)
      case EASYSIMD_MM_FROUND_TO_POS_INF:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_ceil(a_.f64[i]);
        }
        break;
    #endif

    #if defined(easysimd_math_trunc)
      case EASYSIMD_MM_FROUND_TO_ZERO:
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.f64[i] = easysimd_math_trunc(a_.f64[i]);
        }
        break;
    #endif

    default:
      HEDLEY_UNREACHABLE_RETURN(easysimd_mm256_undefined_pd());
  }

  return easysimd__m256d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_round_pd(a, rounding) _mm256_round_pd(a, rounding)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm256_round_pd(a, rounding) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m256d_private \
      easysimd_mm256_round_pd_r_, \
      easysimd_mm256_round_pd_a_ = easysimd__m256d_to_private(a); \
    \
    for (size_t easysimd_mm256_round_pd_i = 0 ; easysimd_mm256_round_pd_i < (sizeof(easysimd_mm256_round_pd_r_.m128d) / sizeof(easysimd_mm256_round_pd_r_.m128d[0])) ; easysimd_mm256_round_pd_i++) { \
      easysimd_mm256_round_pd_r_.m128d[easysimd_mm256_round_pd_i] = easysimd_mm_round_pd(easysimd_mm256_round_pd_a_.m128d[easysimd_mm256_round_pd_i], rounding); \
    } \
    \
    easysimd__m256d_from_private(easysimd_mm256_round_pd_r_); \
  }))
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_round_pd
  #define _mm256_round_pd(a, rounding) easysimd_mm256_round_pd(a, rounding)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_ceil_pd (easysimd__m256d a) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256d r;
  svbool_t pg = svptrue_b64();
  r.sve_f64[EASYSIMD_SV_INDEX_0] = svrintp_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
  r.sve_f64[EASYSIMD_SV_INDEX_1] = svrintp_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  return easysimd_mm256_round_pd(a, EASYSIMD_MM_FROUND_TO_POS_INF);
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_ceil_pd
  #define _mm256_ceil_pd(a) easysimd_mm256_ceil_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_ceil_ps (easysimd__m256 a) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  svbool_t pg = svptrue_b32();
  r.sve_f32[EASYSIMD_SV_INDEX_0] = svrintp_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]);
  r.sve_f32[EASYSIMD_SV_INDEX_1] = svrintp_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  return easysimd_mm256_round_ps(a, EASYSIMD_MM_FROUND_TO_POS_INF);
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_ceil_ps
  #define _mm256_ceil_ps(a) easysimd_mm256_ceil_ps(a)
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DIAGNOSTIC_DISABLE_FLOAT_EQUAL

/* This implementation does not support signaling NaNs (yet?) */
EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmp_pd (easysimd__m128d a, easysimd__m128d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128d r;
  svbool_t pg = svptrue_b64();
  r.sve_i64 = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64, b.sve_f64);
  return r;
#else
  switch (imm8) {
    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      return easysimd_mm_or_pd(easysimd_mm_cmpunord_pd(a, b), easysimd_mm_cmpeq_pd(a, b));
      break;
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      return easysimd_mm_cmpeq_pd(a, b);
      break;
    case EASYSIMD_CMP_NGE_US:
    case EASYSIMD_CMP_NGE_UQ:
      return easysimd_x_mm_not_pd(easysimd_mm_cmpge_pd(a, b));
      break;
    case EASYSIMD_CMP_LT_OS:
    case EASYSIMD_CMP_LT_OQ:
      return easysimd_mm_cmplt_pd(a, b);
      break;
    case EASYSIMD_CMP_NGT_US:
    case EASYSIMD_CMP_NGT_UQ:
      return easysimd_x_mm_not_pd(easysimd_mm_cmpgt_pd(a, b));
      break;
    case EASYSIMD_CMP_LE_OS:
    case EASYSIMD_CMP_LE_OQ:
      return easysimd_mm_cmple_pd(a, b);
      break;
    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      return easysimd_mm_cmpneq_pd(a, b);
      break;
    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      return easysimd_mm_and_pd(easysimd_mm_cmpord_pd(a, b), easysimd_mm_cmpneq_pd(a, b));
      break;
    case EASYSIMD_CMP_NLT_US:
    case EASYSIMD_CMP_NLT_UQ:
      return easysimd_x_mm_not_pd(easysimd_mm_cmplt_pd(a, b));
      break;
    case EASYSIMD_CMP_GE_OS:
    case EASYSIMD_CMP_GE_OQ:
      return easysimd_mm_cmpge_pd(a, b);
      break;
    case EASYSIMD_CMP_NLE_US:
    case EASYSIMD_CMP_NLE_UQ:
      return easysimd_x_mm_not_pd(easysimd_mm_cmple_pd(a, b));
      break;
    case EASYSIMD_CMP_GT_OS:
    case EASYSIMD_CMP_GT_OQ:
      return easysimd_mm_cmpgt_pd(a, b);
      break;
    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      return easysimd_mm_setzero_pd();
      break;
    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      return easysimd_x_mm_setone_pd();
      break;
    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      return easysimd_mm_cmpunord_pd(a, b);
      break;
    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      return easysimd_mm_cmpord_pd(a, b);
      break;
  }

  HEDLEY_UNREACHABLE_RETURN(easysimd_mm_setzero_pd());
#endif
}
#if defined(__clang__) && defined(__AVX512DQ__)
  #define easysimd_mm_cmp_pd(a, b, imm8) (__extension__ ({ \
    easysimd__m128d easysimd_mm_cmp_pd_r; \
    switch (imm8) { \
      case EASYSIMD_CMP_FALSE_OQ: \
      case EASYSIMD_CMP_FALSE_OS: \
        easysimd_mm_cmp_pd_r = easysimd_mm_setzero_pd(); \
        break; \
      case EASYSIMD_CMP_TRUE_UQ: \
      case EASYSIMD_CMP_TRUE_US: \
        easysimd_mm_cmp_pd_r = easysimd_x_mm_setone_pd(); \
        break; \
      default: \
        easysimd_mm_cmp_pd_r = easysimd_mm_cmp_pd(a, b, imm8); \
        break; \
    } \
    easysimd_mm_cmp_pd_r; \
  }))
#elif defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm_cmp_pd(a, b, imm8) _mm_cmp_pd(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_pd
  #define _mm_cmp_pd(a, b, imm8) easysimd_mm_cmp_pd(a, b, imm8)
#endif

EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmp_ps (easysimd__m128 a, easysimd__m128 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;
  svbool_t pg = svptrue_b32();
  r.sve_i32 = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32, b.sve_f32);
  return r;
#else
  switch (imm8) {
    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      return easysimd_mm_or_ps(easysimd_mm_cmpunord_ps(a, b), easysimd_mm_cmpeq_ps(a, b));
      break;
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      return easysimd_mm_cmpeq_ps(a, b);
      break;
    case EASYSIMD_CMP_NGE_US:
    case EASYSIMD_CMP_NGE_UQ:
      return easysimd_x_mm_not_ps(easysimd_mm_cmpge_ps(a, b));
      break;
    case EASYSIMD_CMP_LT_OS:
    case EASYSIMD_CMP_LT_OQ:
      return easysimd_mm_cmplt_ps(a, b);
      break;
    case EASYSIMD_CMP_NGT_US:
    case EASYSIMD_CMP_NGT_UQ:
      return easysimd_x_mm_not_ps(easysimd_mm_cmpgt_ps(a, b));
      break;
    case EASYSIMD_CMP_LE_OS:
    case EASYSIMD_CMP_LE_OQ:
      return easysimd_mm_cmple_ps(a, b);
      break;
    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      return easysimd_mm_cmpneq_ps(a, b);
      break;
    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      return easysimd_mm_and_ps(easysimd_mm_cmpord_ps(a, b), easysimd_mm_cmpneq_ps(a, b));
      break;
    case EASYSIMD_CMP_NLT_US:
    case EASYSIMD_CMP_NLT_UQ:
      return easysimd_x_mm_not_ps(easysimd_mm_cmplt_ps(a, b));
      break;
    case EASYSIMD_CMP_GE_OS:
    case EASYSIMD_CMP_GE_OQ:
      return easysimd_mm_cmpge_ps(a, b);
      break;
    case EASYSIMD_CMP_NLE_US:
    case EASYSIMD_CMP_NLE_UQ:
      return easysimd_x_mm_not_ps(easysimd_mm_cmple_ps(a, b));
      break;
    case EASYSIMD_CMP_GT_OS:
    case EASYSIMD_CMP_GT_OQ:
      return easysimd_mm_cmpgt_ps(a, b);
      break;
    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      return easysimd_mm_setzero_ps();
      break;
    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      return easysimd_x_mm_setone_ps();
      break;
    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      return easysimd_mm_cmpunord_ps(a, b);
      break;
    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      return easysimd_mm_cmpord_ps(a, b);
      break;
  }
#endif

  HEDLEY_UNREACHABLE_RETURN(easysimd_mm_setzero_ps());
}
/* Prior to 9.0 clang has problems with _mm{,256}_cmp_{ps,pd} for all four of the true/false
 * comparisons, but only when AVX-512 is enabled. */
#if defined(__clang__) && defined(__AVX512DQ__)
  #define easysimd_mm_cmp_ps(a, b, imm8) (__extension__ ({ \
    easysimd__m128 easysimd_mm_cmp_ps_r; \
    switch (imm8) { \
      case EASYSIMD_CMP_FALSE_OQ: \
      case EASYSIMD_CMP_FALSE_OS: \
        easysimd_mm_cmp_ps_r = easysimd_mm_setzero_ps(); \
        break; \
      case EASYSIMD_CMP_TRUE_UQ: \
      case EASYSIMD_CMP_TRUE_US: \
        easysimd_mm_cmp_ps_r = easysimd_x_mm_setone_ps(); \
        break; \
      default: \
        easysimd_mm_cmp_ps_r = easysimd_mm_cmp_ps(a, b, imm8); \
        break; \
    } \
    easysimd_mm_cmp_ps_r; \
  }))
#elif defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm_cmp_ps(a, b, imm8) _mm_cmp_ps(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_ps
  #define _mm_cmp_ps(a, b, imm8) easysimd_mm_cmp_ps(a, b, imm8)
#endif

EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_cmp_sd (easysimd__m128d a, easysimd__m128d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128d r;
  svbool_t pg = svdupq_n_b64(1, 0);
  r.sve_i64 = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64, b.sve_f64);
  r.sve_i64 = svsel_s64(svdupq_n_b64(1, 0), r.sve_i64, a.sve_i64);
  return r;
#else
  easysimd__m128d_private
    a_ = easysimd__m128d_to_private(a),
    b_ = easysimd__m128d_to_private(b);

  switch (imm8) {
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      a_.i64[0] = (a_.f64[0] == b_.f64[0]) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_LT_OQ:
    case EASYSIMD_CMP_LT_OS:
      a_.i64[0] = (a_.f64[0] < b_.f64[0]) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_LE_OQ:
    case EASYSIMD_CMP_LE_OS:
      a_.i64[0] = (a_.f64[0] <= b_.f64[0]) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      a_.i64[0] = ((a_.f64[0] != a_.f64[0]) || (b_.f64[0] != b_.f64[0])) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      a_.i64[0] = ((a_.f64[0] == a_.f64[0]) & (b_.f64[0] == b_.f64[0]) & (a_.f64[0] != b_.f64[0])) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      a_.i64[0] = ((a_.f64[0] == a_.f64[0]) & (b_.f64[0] == b_.f64[0]) & (a_.f64[0] != b_.f64[0])) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_NLT_UQ:
    case EASYSIMD_CMP_NLT_US:
      a_.i64[0] = !(a_.f64[0] < b_.f64[0]) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_NLE_UQ:
    case EASYSIMD_CMP_NLE_US:
      a_.i64[0] = !(a_.f64[0] <= b_.f64[0]) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      a_.i64[0] = ((a_.f64[0] == a_.f64[0]) & (b_.f64[0] == b_.f64[0])) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      a_.i64[0] = ((a_.f64[0] != a_.f64[0]) | (b_.f64[0] != b_.f64[0]) | (a_.f64[0] == b_.f64[0])) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_NGE_UQ:
    case EASYSIMD_CMP_NGE_US:
      a_.i64[0] = !(a_.f64[0] >= b_.f64[0]) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_NGT_UQ:
    case EASYSIMD_CMP_NGT_US:
      a_.i64[0] = !(a_.f64[0] > b_.f64[0]) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      a_.i64[0] = INT64_C(0);
      break;

    case EASYSIMD_CMP_GE_OQ:
    case EASYSIMD_CMP_GE_OS:
      a_.i64[0] = (a_.f64[0] >= b_.f64[0]) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_GT_OQ:
    case EASYSIMD_CMP_GT_OS:
      a_.i64[0] = (a_.f64[0] > b_.f64[0]) ? ~INT64_C(0) : INT64_C(0);
      break;

    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      a_.i64[0] = ~INT64_C(0);
      break;

    default:
      HEDLEY_UNREACHABLE();
  }

  return easysimd__m128d_from_private(a_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm_cmp_sd(a, b, imm8) _mm_cmp_sd(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_sd
  #define _mm_cmp_sd(a, b, imm8) easysimd_mm_cmp_sd(a, b, imm8)
#endif

EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmp_ss (easysimd__m128 a, easysimd__m128 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;
  svbool_t pg = svdupq_n_b32(1, 0, 0, 0);
  r.sve_i32 = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32, b.sve_f32);
  r.sve_i32 = svsel_s32(svdupq_n_b32(1, 0, 0, 0), r.sve_i32, a.sve_i32);
  return r;
#else
  easysimd__m128_private
    a_ = easysimd__m128_to_private(a),
    b_ = easysimd__m128_to_private(b);

  switch (imm8) {
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      a_.i32[0] = (a_.f32[0] == b_.f32[0]) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_LT_OQ:
    case EASYSIMD_CMP_LT_OS:
      a_.i32[0] = (a_.f32[0] < b_.f32[0]) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_LE_OQ:
    case EASYSIMD_CMP_LE_OS:
      a_.i32[0] = (a_.f32[0] <= b_.f32[0]) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      a_.i32[0] = ((a_.f32[0] != a_.f32[0]) || (b_.f32[0] != b_.f32[0])) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      a_.i32[0] = ((a_.f32[0] == a_.f32[0]) & (b_.f32[0] == b_.f32[0]) & (a_.f32[0] != b_.f32[0])) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      a_.i32[0] = ((a_.f32[0] == a_.f32[0]) & (b_.f32[0] == b_.f32[0]) & (a_.f32[0] != b_.f32[0])) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_NLT_UQ:
    case EASYSIMD_CMP_NLT_US:
      a_.i32[0] = !(a_.f32[0] < b_.f32[0]) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_NLE_UQ:
    case EASYSIMD_CMP_NLE_US:
      a_.i32[0] = !(a_.f32[0] <= b_.f32[0]) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      a_.i32[0] = ((a_.f32[0] == a_.f32[0]) & (b_.f32[0] == b_.f32[0])) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      a_.i32[0] = ((a_.f32[0] != a_.f32[0]) | (b_.f32[0] != b_.f32[0]) | (a_.f32[0] == b_.f32[0])) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_NGE_UQ:
    case EASYSIMD_CMP_NGE_US:
      a_.i32[0] = !(a_.f32[0] >= b_.f32[0]) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_NGT_UQ:
    case EASYSIMD_CMP_NGT_US:
      a_.i32[0] = !(a_.f32[0] > b_.f32[0]) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      a_.i32[0] = INT32_C(0);
      break;

    case EASYSIMD_CMP_GE_OQ:
    case EASYSIMD_CMP_GE_OS:
      a_.i32[0] = (a_.f32[0] >= b_.f32[0]) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_GT_OQ:
    case EASYSIMD_CMP_GT_OS:
      a_.i32[0] = (a_.f32[0] > b_.f32[0]) ? ~INT32_C(0) : INT32_C(0);
      break;

    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      a_.i32[0] = ~INT32_C(0);
      break;

    default:
      HEDLEY_UNREACHABLE();
  }

  return easysimd__m128_from_private(a_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm_cmp_ss(a, b, imm8) _mm_cmp_ss(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_cmp_ss
  #define _mm_cmp_ss(a, b, imm8) easysimd_mm_cmp_ss(a, b, imm8)
#endif

EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__m256d
#if defined(__clang__) && defined(__AVX512DQ__)
easysimd_mm256_cmp_pd_internal_
#else
easysimd_mm256_cmp_pd
#endif
(easysimd__m256d a, easysimd__m256d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = funlistcmppd[imm8].cmpfun_pd(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    r.m128d[0].neon_f64 = (float64x2_t)neonfunlistcmppd[imm8].neoncmpfun_pd(a.m128d[0], b.m128d[0]);
    r.m128d[1].neon_f64 = (float64x2_t)neonfunlistcmppd[imm8].neoncmpfun_pd(a.m128d[1], b.m128d[1]);
    return r;
  #else
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a),
    b_ = easysimd__m256d_to_private(b);
  switch (imm8) {
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 == b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] == b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LT_OQ:
    case EASYSIMD_CMP_LT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 < b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] < b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LE_OQ:
    case EASYSIMD_CMP_LE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 <= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] <= b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != a_.f64) | (b_.f64 != b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] != a_.f64[i]) || (b_.f64[i] != b_.f64[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] != b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 == a_.f64) & (b_.f64 == b_.f64) & (a_.f64 != b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] == a_.f64[i]) & (b_.f64[i] == b_.f64[i]) & (a_.f64[i] != b_.f64[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLT_UQ:
    case EASYSIMD_CMP_NLT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 < b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] < b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLE_UQ:
    case EASYSIMD_CMP_NLE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 <= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] <= b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ((a_.f64 == a_.f64) & (b_.f64 == b_.f64)));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] == a_.f64[i]) & (b_.f64[i] == b_.f64[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 != a_.f64) | (b_.f64 != b_.f64) | (a_.f64 == b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = ((a_.f64[i] != a_.f64[i]) | (b_.f64[i] != b_.f64[i]) | (a_.f64[i] == b_.f64[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGE_UQ:
    case EASYSIMD_CMP_NGE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 >= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] >= b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGT_UQ:
    case EASYSIMD_CMP_NGT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), ~(a_.f64 > b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = !(a_.f64[i] > b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      r_ = easysimd__m256d_to_private(easysimd_mm256_setzero_pd());
      break;

    case EASYSIMD_CMP_GE_OQ:
    case EASYSIMD_CMP_GE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 >= b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] >= b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_GT_OQ:
    case EASYSIMD_CMP_GT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i64 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i64), (a_.f64 > b_.f64));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
          r_.i64[i] = (a_.f64[i] > b_.f64[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      r_ = easysimd__m256d_to_private(easysimd_x_mm256_setone_pd());
      break;

    default:
      HEDLEY_UNREACHABLE();
  }
  return easysimd__m256d_from_private(r_);
  #endif

}
#if defined(__clang__) && defined(__AVX512DQ__)
  #define easysimd_mm256_cmp_pd(a, b, imm8) (__extension__ ({ \
    easysimd__m256d easysimd_mm256_cmp_pd_r; \
    switch (imm8) { \
      case EASYSIMD_CMP_FALSE_OQ: \
      case EASYSIMD_CMP_FALSE_OS: \
        easysimd_mm256_cmp_pd_r = easysimd_mm256_setzero_pd(); \
        break; \
      case EASYSIMD_CMP_TRUE_UQ: \
      case EASYSIMD_CMP_TRUE_US: \
        easysimd_mm256_cmp_pd_r = easysimd_x_mm256_setone_pd(); \
        break; \
      default: \
        easysimd_mm256_cmp_pd_r = easysimd_mm256_cmp_pd_internal_(a, b, imm8); \
        break; \
    } \
    easysimd_mm256_cmp_pd_r; \
  }))
#elif defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_cmp_pd(a, b, imm8) _mm256_cmp_pd(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_pd
  #define _mm256_cmp_pd(a, b, imm8) easysimd_mm256_cmp_pd(a, b, imm8)
#endif

EASYSIMD_HUGE_FUNCTION_ATTRIBUTES
easysimd__m256
#if defined(__clang__) && defined(__AVX512DQ__)
easysimd_mm256_cmp_ps_internal_
#else
easysimd_mm256_cmp_ps
#endif
(easysimd__m256 a, easysimd__m256 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 31) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    easysimd_svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = funlistcmpps[imm8].cmpfun_ps(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0].neon_f32 = vreinterpretq_f32_u32(neonfunlistcmpps[imm8].neoncmpfun_ps(a.m128[0], b.m128[0]));
    r.m128[1].neon_f32 = vreinterpretq_f32_u32(neonfunlistcmpps[imm8].neoncmpfun_ps(a.m128[1], b.m128[1]));
    return r;
  #else
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a),
    b_ = easysimd__m256_to_private(b);
  switch (imm8) {
    case EASYSIMD_CMP_EQ_OQ:
    case EASYSIMD_CMP_EQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 == b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] == b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LT_OQ:
    case EASYSIMD_CMP_LT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 < b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] < b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_LE_OQ:
    case EASYSIMD_CMP_LE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 <= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] <= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_UNORD_Q:
    case EASYSIMD_CMP_UNORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != a_.f32) | (b_.f32 != b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] != a_.f32[i]) || (b_.f32[i] != b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_UQ:
    case EASYSIMD_CMP_NEQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] != b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NEQ_OQ:
    case EASYSIMD_CMP_NEQ_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 == a_.f32) & (b_.f32 == b_.f32) & (a_.f32 != b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] == a_.f32[i]) & (b_.f32[i] == b_.f32[i]) & (a_.f32[i] != b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLT_UQ:
    case EASYSIMD_CMP_NLT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 < b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] < b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NLE_UQ:
    case EASYSIMD_CMP_NLE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 <= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] <= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_ORD_Q:
    case EASYSIMD_CMP_ORD_S:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ((a_.f32 == a_.f32) & (b_.f32 == b_.f32)));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] == a_.f32[i]) & (b_.f32[i] == b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_EQ_UQ:
    case EASYSIMD_CMP_EQ_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != a_.f32) | (b_.f32 != b_.f32) | (a_.f32 == b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = ((a_.f32[i] != a_.f32[i]) | (b_.f32[i] != b_.f32[i]) | (a_.f32[i] == b_.f32[i])) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGE_UQ:
    case EASYSIMD_CMP_NGE_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 >= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] >= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_NGT_UQ:
    case EASYSIMD_CMP_NGT_US:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), ~(a_.f32 > b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = !(a_.f32[i] > b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_FALSE_OQ:
    case EASYSIMD_CMP_FALSE_OS:
      r_ = easysimd__m256_to_private(easysimd_mm256_setzero_ps());
      break;

    case EASYSIMD_CMP_GE_OQ:
    case EASYSIMD_CMP_GE_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 >= b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] >= b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_GT_OQ:
    case EASYSIMD_CMP_GT_OS:
      #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
        r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 > b_.f32));
      #else
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.i32[i] = (a_.f32[i] > b_.f32[i]) ? ~INT32_C(0) : INT32_C(0);
        }
      #endif
      break;

    case EASYSIMD_CMP_TRUE_UQ:
    case EASYSIMD_CMP_TRUE_US:
      r_ = easysimd__m256_to_private(easysimd_x_mm256_setone_ps());
      break;

    default:
      HEDLEY_UNREACHABLE();
  }
  return easysimd__m256_from_private(r_);
  #endif

}
#if defined(__clang__) && defined(__AVX512DQ__)
  #define easysimd_mm256_cmp_ps(a, b, imm8) (__extension__ ({ \
    easysimd__m256 easysimd_mm256_cmp_ps_r; \
    switch (imm8) { \
      case EASYSIMD_CMP_FALSE_OQ: \
      case EASYSIMD_CMP_FALSE_OS: \
        easysimd_mm256_cmp_ps_r = easysimd_mm256_setzero_ps(); \
        break; \
      case EASYSIMD_CMP_TRUE_UQ: \
      case EASYSIMD_CMP_TRUE_US: \
        easysimd_mm256_cmp_ps_r = easysimd_x_mm256_setone_ps(); \
        break; \
      default: \
        easysimd_mm256_cmp_ps_r = easysimd_mm256_cmp_ps_internal_(a, b, imm8); \
        break; \
    } \
    easysimd_mm256_cmp_ps_r; \
  }))
#elif defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_cmp_ps(a, b, imm8) _mm256_cmp_ps(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_STATEMENT_EXPR_) && EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
  #define easysimd_mm256_cmp_ps(a, b, imm8) EASYSIMD_STATEMENT_EXPR_(({ \
    easysimd__m256_private \
      easysimd_mm256_cmp_ps_r_, \
      easysimd_mm256_cmp_ps_a_ = easysimd__m256_to_private((a)), \
      easysimd_mm256_cmp_ps_b_ = easysimd__m256_to_private((b)); \
    \
    for (size_t i = 0 ; i < (sizeof(easysimd_mm256_cmp_ps_r_.m128) / sizeof(easysimd_mm256_cmp_ps_r_.m128[0])) ; i++) { \
      easysimd_mm256_cmp_ps_r_.m128[i] = easysimd_mm_cmp_ps(easysimd_mm256_cmp_ps_a_.m128[i], easysimd_mm256_cmp_ps_b_.m128[i], (imm8)); \
    } \
    \
    easysimd__m256_from_private(easysimd_mm256_cmp_ps_r_); \
  }))
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cmp_ps
  #define _mm256_cmp_ps(a, b, imm8) easysimd_mm256_cmp_ps(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_copysign_ps(easysimd__m256 dest, easysimd__m256 src) {
  easysimd__m256_private
    r_,
    dest_ = easysimd__m256_to_private(dest),
    src_ = easysimd__m256_to_private(src);

  #if defined(easysimd_math_copysignf)
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_copysignf(dest_.f32[i], src_.f32[i]);
    }
  #else
    easysimd__m256 sgnbit = easysimd_mm256_xor_ps(easysimd_mm256_set1_ps(EASYSIMD_FLOAT32_C(0.0)), easysimd_mm256_set1_ps(-EASYSIMD_FLOAT32_C(0.0)));
    return easysimd_mm256_xor_ps(easysimd_mm256_and_ps(sgnbit, src), easysimd_mm256_andnot_ps(sgnbit, dest));
  #endif

  return easysimd__m256_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_copysign_pd(easysimd__m256d dest, easysimd__m256d src) {
  easysimd__m256d_private
    r_,
    dest_ = easysimd__m256d_to_private(dest),
    src_ = easysimd__m256d_to_private(src);

  #if defined(easysimd_math_copysign)
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = easysimd_math_copysign(dest_.f64[i], src_.f64[i]);
    }
  #else
    easysimd__m256d sgnbit = easysimd_mm256_xor_pd(easysimd_mm256_set1_pd(EASYSIMD_FLOAT64_C(0.0)), easysimd_mm256_set1_pd(-EASYSIMD_FLOAT64_C(0.0)));
    return easysimd_mm256_xor_pd(easysimd_mm256_and_pd(sgnbit, src), easysimd_mm256_andnot_pd(sgnbit, dest));
  #endif

  return easysimd__m256d_from_private(r_);
}

HEDLEY_DIAGNOSTIC_POP /* -Wfloat-equal */

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_cvtepi32_pd (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cvtepi32_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    easysimd_svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svcvt_f64_s64_z(pg, svld1sw_s64(pg, (const int32_t *)(&(a.i32[EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6 )]))));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svcvt_f64_s64_z(pg, svld1sw_s64(pg, (const int32_t *)(&(a.i32[EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6 )]))));
    return r;
  #elif 0 //defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    __asm__ __volatile__ (
        "scvtf v0.4s, %[a].4s           \n\t"
        "fcvtl %[r0].2d, v0.2s          \n\t"
        "mov v1.d[0], v0.d[1]           \n\t"
        "fcvtl %[r1].2d, v1.2s          \n\t"
        :[r0]"=w"(r.m128d[0].neon_f64), [r1]"=w"(r.m128d[1].neon_f64)
        :[a]"w"(a.neon_i32)
        :"v0", "v1"
    );
    return r;
  #else
    easysimd__m256d_private r_;
    easysimd__m128i_private a_ = easysimd__m128i_to_private(a);
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = HEDLEY_STATIC_CAST(easysimd_float64, a_.i32[i]);
    }
    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepi32_pd
  #define _mm256_cvtepi32_pd(a) easysimd_mm256_cvtepi32_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_cvtepi32_ps (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cvtepi32_ps(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 ret;
    ret.m128[0].neon_f32 = vcvtq_f32_s32(a.m128[0].neon_i32);
    ret.m128[1].neon_f32 = vcvtq_f32_s32(a.m128[1].neon_i32);
    return ret;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svcvt_f32_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svcvt_f32_s32_z(pg, a.sve_i32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256 r;
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r.f32) / sizeof(r.f32[0])) ; i++) {
      r.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a.i32[i]);
    }
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtepi32_ps
  #define _mm256_cvtepi32_ps(a) easysimd_mm256_cvtepi32_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm256_cvtpd_epi32 (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cvtpd_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b64();
    r.sve_i32 = svuzp1_s32(svcvt_s32_f64_x(pg, svrinta_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0])),
                           svcvt_s32_f64_x(pg, svrinta_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1])));
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    #if defined(easysimd_math_nearbyint)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
        r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, easysimd_math_nearbyint(a_.f64[i]));
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtpd_epi32
  #define _mm256_cvtpd_epi32(a) easysimd_mm256_cvtpd_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_cvtpd_ps (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cvtpd_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svuzp1_f32(svcvt_f32_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_0]),
                           svcvt_f32_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.f64[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtpd_ps
  #define _mm256_cvtpd_ps(a) easysimd_mm256_cvtpd_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvtps_epi32 (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cvtps_epi32(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i ret;
    ret.m128i[0].neon_i32 = vcvtnq_s32_f32(a.m128[0].neon_f32);
    ret.m128i[1].neon_i32 = vcvtnq_s32_f32(a.m128[1].neon_f32);
    return ret;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svcvt_s32_f32_z(pg, svrinta_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svcvt_s32_f32_z(pg, svrinta_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m256_private a_ = easysimd__m256_to_private(a);

    #if defined(easysimd_math_nearbyintf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
        r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, easysimd_math_nearbyintf(a_.f32[i]));
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtps_epi32
  #define _mm256_cvtps_epi32(a) easysimd_mm256_cvtps_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_cvtps_pd (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cvtps_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svcvt_f64_f32_x(svptrue_b64(), svtbl_f32(a.sve_f32, svdupq_n_u32(0, 2, 1, 3)));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svcvt_f64_f32_x(svptrue_b64(), svtbl_f32(a.sve_f32, svdupq_n_u32(2, 1, 3, 0)));
    return r;
  #else
    easysimd__m256d_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      r_.f64[i] = HEDLEY_STATIC_CAST(double, a_.f32[i]);
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtps_pd
  #define _mm256_cvtps_pd(a) easysimd_mm256_cvtps_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64
easysimd_mm256_cvtsd_f64 (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE) && ( \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,9,0) || \
      HEDLEY_GCC_VERSION_CHECK(7,0,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      HEDLEY_MSVC_VERSION_CHECK(19,14,0))
    return _mm256_cvtsd_f64(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.sve_f64[EASYSIMD_SV_INDEX_0][0];
  #else
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);
    return a_.f64[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtsd_f64
  #define _mm256_cvtsd_f64(a) easysimd_mm256_cvtsd_f64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm256_cvtsi256_si32 (easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE) && ( \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,9,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      HEDLEY_MSVC_VERSION_CHECK(19,14,0))
    return _mm256_cvtsi256_si32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.sve_i32[EASYSIMD_SV_INDEX_0][0];
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    return a_.i32[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtsi256_si32
  #define _mm256_cvtsi256_si32(a) easysimd_mm256_cvtsi256_si32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32
easysimd_mm256_cvtss_f32 (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE) && ( \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,9,0) || \
      HEDLEY_GCC_VERSION_CHECK(7,0,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      HEDLEY_MSVC_VERSION_CHECK(19,14,0))
    return _mm256_cvtss_f32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.sve_f32[EASYSIMD_SV_INDEX_0][0];
  #else
    easysimd__m256_private a_ = easysimd__m256_to_private(a);
    return a_.f32[0];
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvtss_f32
  #define _mm256_cvtss_f32(a) easysimd_mm256_cvtss_f32(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm256_cvttpd_epi32 (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cvttpd_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128i r;
    svbool_t pg = svptrue_b64();
    r.sve_i32 = svuzp1_s32(svcvt_s32_f64_x(pg, svrintz_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0])),
                           svcvt_s32_f64_x(pg, svrintz_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1])));
    return r;
  #else
    easysimd__m128i_private r_;
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    #if defined(easysimd_math_trunc)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
        r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, easysimd_math_trunc(a_.f64[i]));
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvttpd_epi32
  #define _mm256_cvttpd_epi32(a) easysimd_mm256_cvttpd_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_cvttps_epi32 (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_cvttps_epi32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svcvt_s32_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svcvt_s32_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd__m256_private a_ = easysimd__m256_to_private(a);

    #if defined(easysimd_math_truncf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
        r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, easysimd_math_truncf(a_.f32[i]));
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_cvttps_epi32
  #define _mm256_cvttps_epi32(a) easysimd_mm256_cvttps_epi32(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_load_epi32(void const* mem_addr) {
  easysimd__m256i_private r_;
#if defined (EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  svbool_t pg = svptrue_b32();
  r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
  r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
  return r;
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  easysimd__m256i r;
  r.m128i[0].neon_i32 = vld1q_s32((int32_t *)mem_addr);
  r.m128i[1].neon_i32 = vld1q_s32(((int32_t *)mem_addr) + 4);
  return r;
#else
  for(size_t i = 0; i < sizeof(r_.i32) / sizeof(r_.i32[0]); i++){
    r_.i32[i] = *(((int32_t *)mem_addr) + i);
  }
#endif
  return easysimd__m256i_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_load_epi32
  #define _mm256_load_epi32(a) easysimd_mm256_load_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_load_epu32(void const* mem_addr) {
  easysimd__m256i_private r_;
  for(size_t i = 0; i < sizeof(r_.u32) / sizeof(r_.u32[0]); i++){
    r_.u32[i] = *(((uint32_t *)mem_addr) + i);
  }
  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_load_epu64(void const* mem_addr) {
  easysimd__m256i_private r_;
  for(size_t i = 0; i < sizeof(r_.u64) / sizeof(r_.u64[0]); i++){
    r_.u64[i] = *(((uint64_t *)mem_addr) + i);
  }
  return easysimd__m256i_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_load_epi64(void const* mem_addr) {
  easysimd__m256i_private r_;
#if defined (EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  svbool_t pg = svptrue_b64();
  r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1_s64(pg, HEDLEY_STATIC_CAST(const int64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6));
  r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1_s64(pg, HEDLEY_STATIC_CAST(const int64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6));
  return r;
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  easysimd__m256i r;
  r.m128i[0].neon_i64 = vld1q_s64((int64_t *)mem_addr);
  r.m128i[1].neon_i64 = vld1q_s64(((int64_t *)mem_addr) + 2);
  return r;
#else
  for(size_t i = 0; i < sizeof(r_.i64) / sizeof(r_.i64[0]); i++){
    r_.i64[i] = *(((int64_t *)mem_addr) + i);
  }
#endif
  return easysimd__m256i_from_private(r_);
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_load_epi64
  #define _mm256_load_epi64(a) easysimd_mm256_load_epi64(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_convert_to_int32(int32_t* ptr_a, easysimd__m256i b) {
#if defined (EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  svst1_s32(pg, ptr_a                                , b.sve_i32[EASYSIMD_SV_INDEX_0]);
  svst1_s32(pg, ptr_a + (__ARM_FEATURE_SVE_BITS / 32), b.sve_i32[EASYSIMD_SV_INDEX_1]);
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_s32(ptr_a    , b.m128i[0].neon_i32);
  vst1q_s32(ptr_a + 4, b.m128i[1].neon_i32);
#else
  easysimd__m256i_private b_ = easysimd__m256i_to_private(b);
  for(size_t i = 0; i < sizeof(b_.i32) / sizeof(b_.i32[0]); i++){
    *(ptr_a + i) = b_.i32[i];
  }
#endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_convert_to_uint32(uint32_t* ptr_a, easysimd__m256i b) {
#if defined (EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  svst1_u32(pg, ptr_a                                , b.sve_u32[EASYSIMD_SV_INDEX_0]);
  svst1_u32(pg, ptr_a + (__ARM_FEATURE_SVE_BITS / 32), b.sve_u32[EASYSIMD_SV_INDEX_1]);
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_u32(ptr_a    , b.m128i[0].neon_u32);
  vst1q_u32(ptr_a + 4, b.m128i[1].neon_u32);
#else
  easysimd__m256i_private b_ = easysimd__m256i_to_private(b);
  for(size_t i = 0; i < sizeof(b_.u32) / sizeof(b_.u32[0]); i++){
    *(ptr_a + i) = b_.u32[i];
  }
#endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_convert_to_int64(int64_t* ptr_a, easysimd__m256i b) {
#if defined (EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  svst1_s64(pg, ptr_a                                , b.sve_i64[EASYSIMD_SV_INDEX_0]);
  svst1_s64(pg, ptr_a + (__ARM_FEATURE_SVE_BITS / 64), b.sve_i64[EASYSIMD_SV_INDEX_1]);
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_s64(ptr_a    , b.m128i[0].neon_i64);
  vst1q_s64(ptr_a + 2, b.m128i[1].neon_i64);
#else
  easysimd__m256i_private b_ = easysimd__m256i_to_private(b);
  for(size_t i = 0; i < sizeof(b_.i64) / sizeof(b_.i64[0]); i++){
    *(ptr_a + i) = b_.i64[i];
  }
#endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_convert_to_uint64(uint64_t* ptr_a, easysimd__m256i b) {
#if defined (EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  svst1_u64(pg, ptr_a                                , b.sve_u64[EASYSIMD_SV_INDEX_0]);
  svst1_u64(pg, ptr_a + (__ARM_FEATURE_SVE_BITS / 64), b.sve_u64[EASYSIMD_SV_INDEX_1]);
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_u64(ptr_a    , b.m128i[0].neon_u64);
  vst1q_u64(ptr_a + 2, b.m128i[1].neon_u64);
#else
  easysimd__m256i_private b_ = easysimd__m256i_to_private(b);
  for(size_t i = 0; i < sizeof(b_.u64) / sizeof(b_.u64[0]); i++){
    *(ptr_a + i) = b_.u64[i];
  }
#endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_div_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svdiv_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0].neon_f32 = vdivq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vdivq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_div_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_div_ps(a_.m128[1], b_.m128[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 / b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i] / b_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_ps
  #define _mm256_div_ps(a, b) easysimd_mm256_div_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_div_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_div_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svdiv_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    r.m128d[0].neon_f64 = vdivq_f64(a.m128d[0].neon_f64, b.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vdivq_f64(a.m128d[1].neon_f64, b.m128d[1].neon_f64);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_div_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_div_pd(a_.m128d[1], b_.m128d[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 / b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[i] / b_.f64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_div_pd
  #define _mm256_div_pd(a, b) easysimd_mm256_div_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_extractf32x4_ps (easysimd__m256 a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = a.sve_f32[imm8 & 1];
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m256_private a_ = easysimd__m256_to_private(a);
    const size_t offset = HEDLEY_STATIC_CAST(size_t, imm8 & 1) * (sizeof(r_.f32) / sizeof(r_.f32[0]));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = a_.f32[i + offset];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_extractf32x4_ps(a, imm8) _mm256_extractf32x4_ps(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extractf32x4_ps
  #define _mm256_extractf32x4_ps(a, imm8) easysimd_mm256_extractf32x4_ps(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm256_extractf64x2_pd (easysimd__m256d a, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    r.sve_f64 = a.sve_f64[imm8 & 1];
    return r;
  #else
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);
    return a_.m128d[imm8 & 1];
  #endif
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
  #define easysimd_mm256_extractf64x2_pd(a, imm8) _mm256_extractf64x2_pd(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extractf64x2_pd
  #define _mm256_extractf64x2_pd(a, imm8) easysimd_mm256_extractf64x2_pd(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm256_extractf128_pd (easysimd__m256d a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.m128d[imm8 & 1];
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128d[imm8 & 1];
  #else
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);
  return a_.m128d[imm8 & 1];
  #endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_extractf128_pd(a, imm8) _mm256_extractf128_pd(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extractf128_pd
  #define _mm256_extractf128_pd(a, imm8) easysimd_mm256_extractf128_pd(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm256_extractf128_ps (easysimd__m256 a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.m128[imm8 & 1];
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128[imm8 & 1];
  #else
    easysimd__m256_private a_ = easysimd__m256_to_private(a);
  return a_.m128[imm8 & 1];
  #endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_extractf128_ps(a, imm8) _mm256_extractf128_ps(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extractf128_ps
  #define _mm256_extractf128_ps(a, imm8) easysimd_mm256_extractf128_ps(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm256_extractf128_si256 (easysimd__m256i a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
#if defined(EASYSIMD_ARM_SVE_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  return a.m128i[imm8 & 1];
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
  return a_.m128i[imm8 & 1];
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_extractf128_si256(a, imm8) _mm256_extractf128_si256(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extractf128_si256
  #define _mm256_extractf128_si256(a, imm8) easysimd_mm256_extractf128_si256(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_floor_pd (easysimd__m256d a) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256d r;
  svbool_t pg = svptrue_b64();
  r.sve_f64[EASYSIMD_SV_INDEX_0] = svrintm_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0]);
  r.sve_f64[EASYSIMD_SV_INDEX_1] = svrintm_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  return easysimd_mm256_round_pd(a, EASYSIMD_MM_FROUND_TO_NEG_INF);
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_floor_pd
  #define _mm256_floor_pd(a) easysimd_mm256_floor_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_floor_ps (easysimd__m256 a) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  svbool_t pg = svptrue_b32();
  r.sve_f32[EASYSIMD_SV_INDEX_0] = svrintm_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0]);
  r.sve_f32[EASYSIMD_SV_INDEX_1] = svrintm_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  return easysimd_mm256_round_ps(a, EASYSIMD_MM_FROUND_TO_NEG_INF);
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_floor_ps
  #define _mm256_floor_ps(a) easysimd_mm256_floor_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_insert_epi8 (easysimd__m256i a, int8_t i, const int index)
    EASYSIMD_REQUIRE_RANGE(index, 0, 31) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  a.i8[index] = i;
  return a;
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

  a_.i8[index] = i;

  return easysimd__m256i_from_private(a_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_insert_epi8(a, i, index) _mm256_insert_epi8(a, i, index)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_insert_epi8
  #define _mm256_insert_epi8(a, i, index) easysimd_mm256_insert_epi8(a, i, index)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_insert_epi16 (easysimd__m256i a, int16_t i, const int index)
    EASYSIMD_REQUIRE_RANGE(index, 0, 15)  {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  a.i16[index] = i;
  return a;
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);

  a_.i16[index] = i;

  return easysimd__m256i_from_private(a_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_insert_epi16(a, i, index) _mm256_insert_epi16(a, i, index)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_insert_epi16
  #define _mm256_insert_epi16(a, i, imm8) easysimd_mm256_insert_epi16(a, i, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_insert_epi32 (easysimd__m256i a, int32_t i, const int index)
    EASYSIMD_REQUIRE_RANGE(index, 0, 7)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    a.i32[index] = i;
    return a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    a.i32[index] = i;
    return a;
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    a_.i32[index] = i;
    return easysimd__m256i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_insert_epi32(a, i, index) _mm256_insert_epi32(a, i, index)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_insert_epi32
  #define _mm256_insert_epi32(a, i, index) easysimd_mm256_insert_epi32(a, i, index)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_insert_epi64 (easysimd__m256i a, int64_t i, const int index)
    EASYSIMD_REQUIRE_RANGE(index, 0, 3)  {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    a.i64[index] = i;
    return a;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    a.i64[index] = i;
    return a;
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    a_.i64[index] = i;
    return easysimd__m256i_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE) && defined(EASYSIMD_ARCH_AMD64) && \
    (!defined(HEDLEY_MSVC_VERSION) || HEDLEY_MSVC_VERSION_CHECK(19,20,0)) && \
    EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,7,0)
  #define easysimd_mm256_insert_epi64(a, i, index) _mm256_insert_epi64(a, i, index)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_insert_epi64
  #define _mm256_insert_epi64(a, i, index) easysimd_mm256_insert_epi64(a, i, index)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d easysimd_mm256_insertf128_pd(easysimd__m256d a, easysimd__m128d b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    uint64x2_t vmask = vceqq_s64(vdupq_n_s64(imm8 & 1), vdupq_n_s64(0));
    r.m128d[0].neon_f64 = vbslq_f64(vmask, b.neon_f64, a.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vbslq_f64(vmask, a.m128d[1].neon_f64, b.neon_f64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f64[imm8 & 1] = b.sve_f64;
    return a;
  #else
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);
    easysimd__m128d_private b_ = easysimd__m128d_to_private(b);
    a_.m128d_private[imm8 & 1] = b_;
    return easysimd__m256d_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_insertf128_pd
  #define _mm256_insertf128_pd(a, b, imm8) easysimd_mm256_insertf128_pd(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256 easysimd_mm256_insertf128_ps(easysimd__m256 a, easysimd__m128 b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    uint32x4_t vmask = vceqq_s32(vdupq_n_s32(imm8 & 1), vdupq_n_s32(0));
    r.m128[0].neon_f32 = vbslq_f32(vmask, b.neon_f32, a.m128[0].neon_f32);
    r.m128[1].neon_f32 = vbslq_f32(vmask, a.m128[1].neon_f32, b.neon_f32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f32[imm8 & 1] = b.sve_f32;
    return a;
  #else
    easysimd__m256_private a_ = easysimd__m256_to_private(a);
    easysimd__m128_private b_ = easysimd__m128_to_private(b);
    a_.m128_private[imm8 & 1] = b_;
    return easysimd__m256_from_private(a_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_insertf128_ps
  #define _mm256_insertf128_ps(a, b, imm8) easysimd_mm256_insertf128_ps(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i easysimd_mm256_insertf128_si256(easysimd__m256i a, easysimd__m128i b, int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 1) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  a.sve_i32[imm8 & 1] = b.sve_i32;
  return a;
#else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
  easysimd__m128i_private b_ = easysimd__m128i_to_private(b);

  a_.m128i_private[imm8 & 1] = b_;

  return easysimd__m256i_from_private(a_);
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_insertf128_si256
  #define _mm256_insertf128_si256(a, b, imm8) easysimd_mm256_insertf128_si256(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256 easysimd_mm256_dp_ps(easysimd__m256 a, easysimd__m256 b, const int imm8){
#if defined(EASYSIMD_X86_AVX_NATIVE)
  return _mm256_dp_ps(a, b, imm8);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  svbool_t pgmul = svdupq_n_b32((imm8 >> 4) & 1, (imm8 >> 5) & 1, (imm8 >> 6) & 1, (imm8 >> 7) & 1);
  svbool_t pgstr = svdupq_n_b32((imm8 >> 0) & 1, (imm8 >> 1) & 1, (imm8 >> 2) & 1, (imm8 >> 3) & 1);
  float32_t array[2];
  array[EASYSIMD_SV_INDEX_0] = svaddv_f32(svptrue_b32(), svmul_f32_z(pgmul, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]));
  array[EASYSIMD_SV_INDEX_1] = svaddv_f32(svptrue_b32(), svmul_f32_z(pgmul, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]));
  r.sve_f32[EASYSIMD_SV_INDEX_0] = svdup_n_f32_z(pgstr, array[EASYSIMD_SV_INDEX_0]);
  r.sve_f32[EASYSIMD_SV_INDEX_1] = svdup_n_f32_z(pgstr, array[EASYSIMD_SV_INDEX_1]);
  return r;
#else
  easysimd_float32 sum0 = EASYSIMD_FLOAT32_C(0.0);
  easysimd__m128_private r0;
  easysimd_memset(&r0, 0, sizeof(r0));
  EASYSIMD_VECTORIZE_REDUCTION(+:sum0)
  for (size_t i = 0 ; i < (sizeof(r0.f32) / sizeof(r0.f32[0])) ; i++) {
    sum0 += ((imm8 >> (i + 4)) & 1) ? (a.f32[4 + i] * b.f32[4 + i]) : EASYSIMD_FLOAT32_C(0.0);
  }

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r0.f32) / sizeof(r0.f32[0])) ; i++) {
    r0.f32[i] = ((imm8 >> i) & 1) ? sum0 : EASYSIMD_FLOAT32_C(0.0);
  }

  easysimd_float32 sum1 = EASYSIMD_FLOAT32_C(0.0);
  easysimd__m128_private r1;
  easysimd_memset(&r1, 0, sizeof(r1));
  EASYSIMD_VECTORIZE_REDUCTION(+:sum1)
  for (size_t i = 0 ; i < (sizeof(r1.f32) / sizeof(r1.f32[0])) ; i++) {
    sum1 += ((imm8 >> (i + 4)) & 1) ? (a.f32[i] * b.f32[i]) : EASYSIMD_FLOAT32_C(0.0);
  }

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r1.f32) / sizeof(r1.f32[0])) ; i++) {
    r1.f32[i] = ((imm8 >> i) & 1) ? sum1 : EASYSIMD_FLOAT32_C(0.0);
  }

  return easysimd_mm256_set_m128(easysimd__m128_from_private(r0), easysimd__m128_from_private(r1));
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_dp_ps
  #define _mm256_dp_ps(a, b, imm8) easysimd_mm256_dp_ps(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm256_extract_epi32 (easysimd__m256i a, const int index)
    EASYSIMD_REQUIRE_RANGE(index, 0, 7) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128i[!!(index & 0x04)].i32[index & 0x03];
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.i32[index];
  #else
  easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
    return a_.i32[index];
  #endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_extract_epi32(a, index) _mm256_extract_epi32(a, index)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extract_epi32
  #define _mm256_extract_epi32(a, index) easysimd_mm256_extract_epi32(a, index)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm256_extract_epi64 (easysimd__m256i a, const int index)
    EASYSIMD_REQUIRE_RANGE(index, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.i64[index];
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return a.m128i[!!(index & 0x02)].i64[index & 0x01];
  #else
    easysimd__m256i_private a_ = easysimd__m256i_to_private(a);
  return a_.i64[index];
  #endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
  #if !defined(HEDLEY_MSVC_VERSION) || HEDLEY_MSVC_VERSION_CHECK(19,20,0)
    #define easysimd_mm256_extract_epi64(a, index) _mm256_extract_epi64(a, index)
  #endif
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_extract_epi64
  #define _mm256_extract_epi64(a, index) easysimd_mm256_extract_epi64(a, index)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_lddqu_si256 (easysimd__m256i const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu_si256(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(svptrue_b32(), (int32_t const *)mem_addr);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(svptrue_b32(), (int32_t const *)mem_addr + 4 * EASYSIMD_SV_INDEX_1);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vld1q_s32((int32_t const *)mem_addr);
    r.m128i[1].neon_i32 = vld1q_s32((int32_t const *)mem_addr + 4);
    return r;
  #else
    easysimd__m256i r;
    easysimd_memcpy(&r, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256i), sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_lddqu_si256
  #define _mm256_lddqu_si256(a) easysimd_mm256_lddqu_si256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_load_pd (const double mem_addr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_load_pd(mem_addr);
  #else
    #if defined (EASYSIMD_ARM_SVE_NATIVE)
      easysimd__m256d r;
      svbool_t pg = svptrue_b64();
      r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6));
      r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_f64(pg, HEDLEY_STATIC_CAST(const float64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6));
      return r;
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      easysimd__m256d r;
      r.m128d[0].neon_f64 = vld1q_f64((float64_t *)mem_addr);
      r.m128d[1].neon_f64 = vld1q_f64(((float64_t *)mem_addr) + 2);
      return r;
    #else
      easysimd__m256d r;
      easysimd_memcpy(&r, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256d), sizeof(r));
      return r;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_load_pd
  #define _mm256_load_pd(a) easysimd_mm256_load_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_load_ps (const float mem_addr[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_load_ps(mem_addr);
  #else
    #if defined (EASYSIMD_ARM_SVE_NATIVE)
      easysimd__m256 r;
      svbool_t pg = svptrue_b32();
      r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t *, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
      r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(pg, HEDLEY_STATIC_CAST(const float32_t *, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
      return r;
    #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      easysimd__m256 r;
      r.m128[0].neon_f32 = vld1q_f32((float32_t *)mem_addr);
      r.m128[1].neon_f32 = vld1q_f32(((float32_t *)mem_addr) + 4);
      return r;
    #else
    easysimd__m256 r;
    easysimd_memcpy(&r, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256), sizeof(r));
    return r;
    #endif
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_load_ps
  #define _mm256_load_ps(a) easysimd_mm256_load_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_load_si256 (easysimd__m256i const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_load_si256(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(pg, (int32_t *)mem_addr + ((__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_0));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(pg, (int32_t *)mem_addr + ((__ARM_FEATURE_SVE_BITS / 32) * EASYSIMD_SV_INDEX_1));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i ret;
    ret.m128i[0].neon_i32 = vld1q_s32((int32_t const*)mem_addr);
    ret.m128i[1].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 4);
    return ret;
  #else
    easysimd__m256i r;
    easysimd_memcpy(&r, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256i), sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_load_si256
  #define _mm256_load_si256(a) easysimd_mm256_load_si256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_loadu_pd (const double a[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_f64(svptrue_b64(), &(a[0]));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_f64(svptrue_b64(), &(a[0 + EASYSIMD_SV_INDEX_1 * 2]));
    return r;
  #else
    easysimd__m256d r;
    easysimd_memcpy(&r, a, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_loadu_pd
  #define _mm256_loadu_pd(a) easysimd_mm256_loadu_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_loadu_ps (const float a[HEDLEY_ARRAY_PARAM(8)]) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(pg, &(a[EASYSIMD_SV_INDEX_0 << 2]));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(pg, &(a[EASYSIMD_SV_INDEX_1 << 2]));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0].neon_f32 = vld1q_f32(&(a[0]));
    r.m128[1].neon_f32 = vld1q_f32(&(a[4]));
    return r;
  #else
    easysimd__m256 r;
    easysimd_memcpy(&r, a, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_loadu_ps
  #define _mm256_loadu_ps(a) easysimd_mm256_loadu_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_loadu_epi8(void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_CLANG_REV_344862)
    return _mm256_loadu_epi8(mem_addr);
  #elif defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu_si256(EASYSIMD_ALIGN_CAST(__m256i const *, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b8();
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svld1_s8(pg, HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 3));
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svld1_s8(pg, HEDLEY_STATIC_CAST(const int8_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 3));
    return r;
  #else
    easysimd__m256i r;
    easysimd_memcpy(&r, mem_addr, sizeof(r));
    return r;
  #endif
}
#define easysimd_x_mm256_loadu_epi8(mem_addr) easysimd_mm256_loadu_epi8(mem_addr)
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && (defined(EASYSIMD_BUG_GCC_95483) || defined(EASYSIMD_BUG_CLANG_REV_344862)))
  #undef _mm256_loadu_epi8
  #define _mm256_loadu_epi8(a) easysimd_mm256_loadu_epi8(a)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_loadu_epi16(void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512BW_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_CLANG_REV_344862)
    return _mm256_loadu_epi16(mem_addr);
  #elif defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu_si256(EASYSIMD_ALIGN_CAST(__m256i const *, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b16();
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svld1_s16(pg, HEDLEY_STATIC_CAST(const int16_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 4));
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svld1_s16(pg, HEDLEY_STATIC_CAST(const int16_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 4));
    return r;
  #else
    easysimd__m256i r;
    easysimd_memcpy(&r, mem_addr, sizeof(r));
    return r;
  #endif
}
#define easysimd_x_mm256_loadu_epi16(mem_addr) easysimd_mm256_loadu_epi16(mem_addr)
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && (defined(EASYSIMD_BUG_GCC_95483) || defined(EASYSIMD_BUG_CLANG_REV_344862)))
  #undef _mm256_loadu_epi16
  #define _mm256_loadu_epi16(a) easysimd_mm256_loadu_epi16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_loadu_epi32(void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_CLANG_REV_344862)
    return _mm256_loadu_epi32(mem_addr);
  #elif defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu_si256(EASYSIMD_ALIGN_CAST(__m256i const *, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5));
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(pg, HEDLEY_STATIC_CAST(const int32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5));
    return r;
  #else
    easysimd__m256i r;
    easysimd_memcpy(&r, mem_addr, sizeof(r));
    return r;
  #endif
}
#define easysimd_x_mm256_loadu_epi32(mem_addr) easysimd_mm256_loadu_epi32(mem_addr)
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && (defined(EASYSIMD_BUG_GCC_95483) || defined(EASYSIMD_BUG_CLANG_REV_344862))
  #undef _mm256_loadu_epi32
  #define _mm256_loadu_epi32(a) easysimd_mm256_loadu_epi32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_loadu_epi64(void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE) && defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_BUG_GCC_95483) && !defined(EASYSIMD_BUG_CLANG_REV_344862)
    return _mm256_loadu_epi64(mem_addr);
  #elif defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu_si256(EASYSIMD_ALIGN_CAST(__m256i const *, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svld1_s64(pg, HEDLEY_STATIC_CAST(const int64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6));
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svld1_s64(pg, HEDLEY_STATIC_CAST(const int64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6));
    return r;
  #else
    easysimd__m256i r;
    easysimd_memcpy(&r, mem_addr, sizeof(r));
    return r;
  #endif
}
#define easysimd_x_mm256_loadu_epi64(mem_addr) easysimd_mm256_loadu_epi64(mem_addr)
#if defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && (defined(EASYSIMD_BUG_GCC_95483) || defined(EASYSIMD_BUG_CLANG_REV_344862))
  #undef _mm256_loadu_epi64
  #define _mm256_loadu_epi64(a) easysimd_mm256_loadu_epi64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_loadu_si256 (void const * mem_addr) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu_si256(EASYSIMD_ALIGN_CAST(const __m256i*, mem_addr));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svld1_s32(pg, ((int32_t const*)mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5 ));  // SVE_BITS / 32
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svld1_s32(pg, ((int32_t const*)mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5 ));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    r.m128i[0].neon_i32 = vld1q_s32((int32_t const*)mem_addr);
    r.m128i[1].neon_i32 = vld1q_s32(((int32_t const*)mem_addr) + 4);
    return r;
  #else
    easysimd__m256i_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_loadu_si256
  #define _mm256_loadu_si256(mem_addr) easysimd_mm256_loadu_si256(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_loadu2_m128 (const float hiaddr[HEDLEY_ARRAY_PARAM(4)], const float loaddr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu2_m128(hiaddr, loaddr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(svptrue_b32(), &(loaddr[0]));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(svptrue_b32(), &(hiaddr[0]));
    return r;
  #else
    return
      easysimd_mm256_insertf128_ps(easysimd_mm256_castps128_ps256(easysimd_mm_loadu_ps(loaddr)),
              easysimd_mm_loadu_ps(hiaddr), 1);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_loadu2_m128
  #define _mm256_loadu2_m128(hiaddr, loaddr) easysimd_mm256_loadu2_m128(hiaddr, loaddr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_loadu2_m128d (const double hiaddr[HEDLEY_ARRAY_PARAM(2)], const double loaddr[HEDLEY_ARRAY_PARAM(2)]) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu2_m128d(hiaddr, loaddr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_f64(svptrue_b64(), &(loaddr[0]));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_f64(svptrue_b64(), &(hiaddr[0]));
    return r;
  #else
    return
      easysimd_mm256_insertf128_pd(easysimd_mm256_castpd128_pd256(easysimd_mm_loadu_pd(loaddr)),
              easysimd_mm_loadu_pd(hiaddr), 1);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_loadu2_m128d
  #define _mm256_loadu2_m128d(hiaddr, loaddr) easysimd_mm256_loadu2_m128d(hiaddr, loaddr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_loadu2_m128i (const easysimd__m128i* hiaddr, const easysimd__m128i* loaddr) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_loadu2_m128i(hiaddr, loaddr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = loaddr->sve_i32;
    r.sve_i32[EASYSIMD_SV_INDEX_1] = hiaddr->sve_i32;
    return r;
  #else
    return
      easysimd_mm256_insertf128_si256(easysimd_mm256_castsi128_si256(easysimd_mm_loadu_si128(loaddr)),
          easysimd_mm_loadu_si128(hiaddr), 1);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_loadu2_m128i
  #define _mm256_loadu2_m128i(hiaddr, loaddr) easysimd_mm256_loadu2_m128i(hiaddr, loaddr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_maskload_pd (const easysimd_float64 mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
      return _mm_maskload_pd(mem_addr, HEDLEY_REINTERPRET_CAST(easysimd__m128d, mask));
    #else
      return _mm_maskload_pd(mem_addr, mask);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    svbool_t pgmask = svcmplt_n_s64(svptrue_b64(), mask.sve_i64, 0);
    r.sve_f64 = svld1_f64(pgmask, mem_addr);
    return r;
  #else
    easysimd__m128d_private
      mem_ = easysimd__m128d_to_private(easysimd_mm_loadu_pd(mem_addr)),
      r_;
    easysimd__m128i_private mask_ = easysimd__m128i_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i64 = vandq_s64(mem_.neon_i64, vshrq_n_s64(mask_.neon_i64, 63));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.i64[i] = mem_.i64[i] & (mask_.i64[i] >> 63);
      }
    #endif

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskload_pd
  #define _mm_maskload_pd(mem_addr, mask) easysimd_mm_maskload_pd(HEDLEY_REINTERPRET_CAST(double const*, mem_addr), mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_maskload_pd (const easysimd_float64 mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m256i mask) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
      return _mm256_maskload_pd(mem_addr, HEDLEY_REINTERPRET_CAST(easysimd__m256d, mask));
    #else
      return _mm256_maskload_pd(mem_addr, mask);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svld1_f64(svcmplt(svptrue_b64(), mask.sve_i64[EASYSIMD_SV_INDEX_0], svdup_n_s64(0)), (const float64_t *)&(mem_addr[0]));
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svld1_f64(svcmplt(svptrue_b64(), mask.sve_i64[EASYSIMD_SV_INDEX_1], svdup_n_s64(0)), (const float64_t *)&(mem_addr[0 + 2 * EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256d_private r_;
    easysimd__m256i_private mask_ = easysimd__m256i_to_private(mask);

    r_ = easysimd__m256d_to_private(easysimd_mm256_loadu_pd(mem_addr));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.i64[i] &= mask_.i64[i] >> 63;
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskload_pd
  #define _mm256_maskload_pd(mem_addr, mask) easysimd_mm256_maskload_pd(HEDLEY_REINTERPRET_CAST(double const*, mem_addr), mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_maskload_ps (const easysimd_float32 mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m128i mask) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
      return _mm_maskload_ps(mem_addr, HEDLEY_REINTERPRET_CAST(easysimd__m128, mask));
    #else
      return _mm_maskload_ps(mem_addr, mask);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pgm = svcmplt_s32(svptrue_b32(), mask.sve_i32, svdup_n_s32(0));
    r.sve_i32 = svld1_s32(pgm, (const int *)&(mem_addr[0]));
    return r;
  #else
    easysimd__m128_private
      mem_ = easysimd__m128_to_private(easysimd_mm_loadu_ps(mem_addr)),
      r_;
    easysimd__m128i_private mask_ = easysimd__m128i_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vandq_s32(mem_.neon_i32, vshrq_n_s32(mask_.neon_i32, 31));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = mem_.i32[i] & (mask_.i32[i] >> 31);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskload_ps
  #define _mm_maskload_ps(mem_addr, mask) easysimd_mm_maskload_ps(HEDLEY_REINTERPRET_CAST(float const*, mem_addr), mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_maskload_ps (const easysimd_float32 mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m256i mask) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
      return _mm256_maskload_ps(mem_addr, HEDLEY_REINTERPRET_CAST(easysimd__m256, mask));
    #else
      return _mm256_maskload_ps(mem_addr, mask);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svld1_f32(svcmplt(svptrue_b32(), mask.sve_i32[EASYSIMD_SV_INDEX_0], svdup_n_s32(0)), (const float *)&(mem_addr[0]));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svld1_f32(svcmplt(svptrue_b32(), mask.sve_i32[EASYSIMD_SV_INDEX_1], svdup_n_s32(0)), (const float *)&(mem_addr[0 + 4 * EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256_private r_;
    easysimd__m256i_private mask_ = easysimd__m256i_to_private(mask);

    r_ = easysimd__m256_to_private(easysimd_mm256_loadu_ps(mem_addr));
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.i32[i] &= mask_.i32[i] >> 31;
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskload_ps
  #define _mm256_maskload_ps(mem_addr, mask) easysimd_mm256_maskload_ps(HEDLEY_REINTERPRET_CAST(float const*, mem_addr), mask)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_maskstore_pd (easysimd_float64 mem_addr[HEDLEY_ARRAY_PARAM(2)], easysimd__m128i mask, easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
      _mm_maskstore_pd(mem_addr, HEDLEY_REINTERPRET_CAST(easysimd__m128d, mask), a);
    #else
      _mm_maskstore_pd(mem_addr, mask, a);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f64(svcmplt_n_s64(svptrue_b64(), mask.sve_i64, 0), &(mem_addr[0]), a.sve_f64);
  #else
    easysimd__m128i_private mask_ = easysimd__m128i_to_private(mask);
    easysimd__m128d_private a_ = easysimd__m128d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      if (mask_.u64[i] >> 63)
        mem_addr[i] = a_.f64[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskstore_pd
  #define _mm_maskstore_pd(mem_addr, mask, a) easysimd_mm_maskstore_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), mask, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_maskstore_pd (easysimd_float64 mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m256i mask, easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
      _mm256_maskstore_pd(mem_addr, HEDLEY_REINTERPRET_CAST(easysimd__m256d, mask), a);
    #else
      _mm256_maskstore_pd(mem_addr, mask, a);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f64(svcmplt(svptrue_b64(), mask.sve_i64[EASYSIMD_SV_INDEX_0], svdup_n_s64(0)), (float64_t *)&(mem_addr[0]), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    svst1_f64(svcmplt(svptrue_b64(), mask.sve_i64[EASYSIMD_SV_INDEX_1], svdup_n_s64(0)), (float64_t *)&(mem_addr[0 + 2 * EASYSIMD_SV_INDEX_1]), a.sve_f64[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd__m256i_private mask_ = easysimd__m256i_to_private(mask);
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      if (mask_.u64[i] & (UINT64_C(1) << 63))
        mem_addr[i] = a_.f64[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskstore_pd
  #define _mm256_maskstore_pd(mem_addr, mask, a) easysimd_mm256_maskstore_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), mask, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_maskstore_ps (easysimd_float32 mem_addr[HEDLEY_ARRAY_PARAM(4)], easysimd__m128i mask, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
      _mm_maskstore_ps(mem_addr, HEDLEY_REINTERPRET_CAST(easysimd__m128, mask), a);
    #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f32(svcmplt_n_s32(svptrue_b32(), mask.sve_i32, 0), &(mem_addr[0]), a.sve_f32);
    #else
      _mm_maskstore_ps(mem_addr, mask, a);
    #endif
  #else
    easysimd__m128i_private mask_ = easysimd__m128i_to_private(mask);
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      if (mask_.u32[i] & (UINT32_C(1) << 31))
        mem_addr[i] = a_.f32[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_maskstore_ps
  #define _mm_maskstore_ps(mem_addr, mask, a) easysimd_mm_maskstore_ps(HEDLEY_REINTERPRET_CAST(float*, mem_addr), mask, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_maskstore_ps (easysimd_float32 mem_addr[HEDLEY_ARRAY_PARAM(8)], easysimd__m256i mask, easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,8,0)
      _mm256_maskstore_ps(mem_addr, HEDLEY_REINTERPRET_CAST(easysimd__m256, mask), a);
    #else
      _mm256_maskstore_ps(mem_addr, mask, a);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f32(svcmplt(svptrue_b32(), mask.sve_i32[EASYSIMD_SV_INDEX_0], svdup_n_s32(0)), (float *)&(mem_addr[0]), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    svst1_f32(svcmplt(svptrue_b32(), mask.sve_i32[EASYSIMD_SV_INDEX_1], svdup_n_s32(0)), (float *)&(mem_addr[0 + 4 * EASYSIMD_SV_INDEX_1]), a.sve_f32[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd__m256i_private mask_ = easysimd__m256i_to_private(mask);
    easysimd__m256_private a_ = easysimd__m256_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      if (mask_.u32[i] & (UINT32_C(1) << 31))
        mem_addr[i] = a_.f32[i];
    }
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_maskstore_ps
  #define _mm256_maskstore_ps(mem_addr, mask, a) easysimd_mm256_maskstore_ps(HEDLEY_REINTERPRET_CAST(float*, mem_addr), mask, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_min_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_min_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svmin_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svmin_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_min_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_min_ps(a_.m128[1], b_.m128[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = (a_.f32[i] < b_.f32[i]) ? a_.f32[i] : b_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_ps
  #define _mm256_min_ps(a, b) easysimd_mm256_min_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_min_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_min_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svmin_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svmin_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_min_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_min_pd(a_.m128d[1], b_.m128d[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = (a_.f64[i] < b_.f64[i]) ? a_.f64[i] : b_.f64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_min_pd
  #define _mm256_min_pd(a, b) easysimd_mm256_min_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_max_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_max_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svmax_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svmax_f32_x(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_max_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_max_ps(a_.m128[1], b_.m128[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = (a_.f32[i] > b_.f32[i]) ? a_.f32[i] : b_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_max_ps
  #define _mm256_max_ps(a, b) easysimd_mm256_max_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_max_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_max_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svmax_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svmax_f64_x(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_max_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_max_pd(a_.m128d[1], b_.m128d[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = (a_.f64[i] > b_.f64[i]) ? a_.f64[i] : b_.f64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_max_pd
  #define _mm256_max_pd(a, b) easysimd_mm256_max_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_movedup_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_movedup_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    svuint64_t svindex = svdupq_n_u64(0, 2);
    svfloat64_t svtemp = svld1_gather_u64index_f64(pg, (const float64_t *)&(a.f64[0]), svindex);
    svst1_scatter_u64index_f64(pg, (float64_t *)&(a.f64[1]), svindex, svtemp);
    return a;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 32, a_.f64, a_.f64, 0, 0, 2, 2);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i += 2) {
        r_.f64[i] = r_.f64[i + 1] = a_.f64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_movedup_pd
  #define _mm256_movedup_pd(a) easysimd_mm256_movedup_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_movehdup_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_movehdup_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svtrn2_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_0]);
    a.sve_f32[EASYSIMD_SV_INDEX_1] = svtrn2_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.f32, a_.f32, 1, 1, 3, 3, 5, 5, 7, 7);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 2) {
        r_.f32[i - 1] = r_.f32[i] = a_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_movehdup_ps
  #define _mm256_movehdup_ps(a) easysimd_mm256_movehdup_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_moveldup_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_moveldup_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.sve_f32[EASYSIMD_SV_INDEX_0] = svtrn1_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], a.sve_f32[EASYSIMD_SV_INDEX_0]);
    a.sve_f32[EASYSIMD_SV_INDEX_1] = svtrn1_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return a;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.f32, a_.f32, 0, 0, 2, 2, 4, 4, 6, 6);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i += 2) {
        r_.f32[i] = r_.f32[i + 1] = a_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_moveldup_ps
  #define _mm256_moveldup_ps(a) easysimd_mm256_moveldup_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_movemask_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_movemask_ps(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_u32 = vshrq_n_u32(vreinterpretq_u32_f32(a.m128[0].neon_f32), 31);
    res.m128i[1].neon_u32 = vshrq_n_u32(vreinterpretq_u32_f32(a.m128[1].neon_f32), 31);
    res.m128i[0].neon_u64 = vsraq_n_u64(res.m128i[0].neon_u64, res.m128i[0].neon_u64, 31);
    res.m128i[1].neon_u64 = vsraq_n_u64(res.m128i[1].neon_u64, res.m128i[1].neon_u64, 31);
    return (int)(vgetq_lane_u8(res.m128i[0].neon_u8, 0) | (vgetq_lane_u8(res.m128i[0].neon_u8, 8) << 2) |
                (vgetq_lane_u8(res.m128i[1].neon_u8, 0) << 4) | (vgetq_lane_u8(res.m128i[1].neon_u8, 8) << 6));
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B32_TO_MASK(k, svcmplt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], svdup_n_s32(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B32_TO_MASK(k, svcmplt_s32(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], svdup_n_s32(0)), EASYSIMD_SV_INDEX_1);
    return (int)k;
  #else
    easysimd__m256_private a_ = easysimd__m256_to_private(a);
    int r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
      r |= (a_.u32[i] >> 31) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_movemask_ps
  #define _mm256_movemask_ps(a) easysimd_mm256_movemask_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_movemask_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_movemask_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int64_t k = 0;
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_0], svdup_n_s64(0)), EASYSIMD_SV_INDEX_0);
    EASYSIMD_B64_TO_MASK(k, svcmplt_s64(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_1], svdup_n_s64(0)), EASYSIMD_SV_INDEX_1);
    return (int)k;
  #else
    easysimd__m256d_private a_ = easysimd__m256d_to_private(a);
    int r = 0;

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.f64) / sizeof(a_.f64[0])) ; i++) {
      r |= (a_.u64[i] >> 63) << i;
    }

    return r;
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_movemask_pd
  #define _mm256_movemask_pd(a) easysimd_mm256_movemask_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_mul_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_mul_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0].neon_f32 = vmulq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vmulq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svmul_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_mul_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_mul_ps(a_.m128[1], b_.m128[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 * b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i] * b_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mul_ps
  #define _mm256_mul_ps(a, b) easysimd_mm256_mul_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_mul_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_mul_pd(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    r.m128d[0].neon_f64 = vmulq_f64(a.m128d[0].neon_f64, b.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vmulq_f64(a.m128d[1].neon_f64, b.m128d[1].neon_f64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svmul_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_mul_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_mul_pd(a_.m128d[1], b_.m128d[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 * b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[i] * b_.f64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_mul_pd
  #define _mm256_mul_pd(a, b) easysimd_mm256_mul_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_or_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_or_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    r.sve_u32[EASYSIMD_SV_INDEX_0] = svorr_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = svorr_u32_z(pg, a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    __asm__ __volatile(
        "orr %0.16b, %0.16b, %2.16b     \n\t"
        "orr %1.16b, %1.16b, %3.16b     \n\t"
        :"+w"(a.m128[0].neon_f32), "+w"(a.m128[1].neon_f32)
        :"w"(b.m128[0].neon_f32), "w"(b.m128[1].neon_f32)
    );
    return a;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_or_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_or_ps(a_.m128[1], b_.m128[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f | b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i] | b_.u32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_or_ps
  #define _mm256_or_ps(a, b) easysimd_mm256_or_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_or_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_or_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    r.sve_u64[EASYSIMD_SV_INDEX_0] = svorr_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = svorr_u64_z(pg, a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    __asm__ __volatile(
        "orr %0.16b, %0.16b, %2.16b     \n\t"
        "orr %1.16b, %1.16b, %3.16b     \n\t"
        :"+w"(a.m128d[0].neon_f64), "+w"(a.m128d[1].neon_f64)
        :"w"(b.m128d[0].neon_f64), "w"(b.m128d[1].neon_f64)
    );
    return a;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_or_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_or_pd(a_.m128d[1], b_.m128d[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f | b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = a_.u64[i] | b_.u64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_or_pd
  #define _mm256_or_pd(a, b) easysimd_mm256_or_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_permute_ps (easysimd__m256 a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  svuint32_t index = svdupq_n_u32(imm8 & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
  r.sve_f32[EASYSIMD_SV_INDEX_0] = svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], index);
  r.sve_f32[EASYSIMD_SV_INDEX_1] = svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], index);
  return r;
#else
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = a_.m128_private[i >> 2].f32[(imm8 >> ((i << 1) & 7)) & 3];
  }

  return easysimd__m256_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_permute_ps(a, imm8) _mm256_permute_ps(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permute_ps
  #define _mm256_permute_ps(a, imm8) easysimd_mm256_permute_ps(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_permute_pd (easysimd__m256d a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256d r;
  r.sve_f64[EASYSIMD_SV_INDEX_0] = svtbl_f64(a.sve_f64[EASYSIMD_SV_INDEX_0], svdupq_n_u64((imm8 >> 0) & 0x01, (imm8 >> 1) & 0x01));
  r.sve_f64[EASYSIMD_SV_INDEX_1] = svtbl_f64(a.sve_f64[EASYSIMD_SV_INDEX_1], svdupq_n_u64((imm8 >> 2) & 0x01, (imm8 >> 3) & 0x01));
  return r;
#else
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    r_.f64[i] = a_.f64[((imm8 >> i) & 1) + (i & 2)];
  }

  return easysimd__m256d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_permute_pd(a, imm8) _mm256_permute_pd(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permute_pd
  #define _mm256_permute_pd(a, imm8) easysimd_mm256_permute_pd(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_permute_ps (easysimd__m128 a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;
  svuint32_t svindex = svdupq_n_u32(imm8 & 0x03, (imm8 >> 2) & 0x03, (imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03);
  r.sve_f32 = svtbl_f32(a.sve_f32, svindex);
  return r;
#else
  easysimd__m128_private
    r_,
    a_ = easysimd__m128_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
    r_.f32[i] = a_.f32[(imm8 >> ((i << 1) & 7)) & 3];
  }

  return easysimd__m128_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm_permute_ps(a, imm8) _mm_permute_ps(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_permute_ps
  #define _mm_permute_ps(a, imm8) easysimd_mm_permute_ps(a, imm8)
#endif


EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_permute_pd (easysimd__m128d a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128d r;
  svuint64_t svindex = svdupq_n_u64(imm8 & 1, (imm8 >> 1) & 1);
  r.sve_u64 = svtbl_u64(a.sve_u64, svindex);
  return r;
#else
  easysimd__m128d_private
    r_,
    a_ = easysimd__m128d_to_private(a);

  EASYSIMD_VECTORIZE
  for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
    r_.f64[i] = a_.f64[((imm8 >> i) & 1) + (i & 2)];
  }

  return easysimd__m128d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm_permute_pd(a, imm8) _mm_permute_pd(a, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_permute_pd
  #define _mm_permute_pd(a, imm8) easysimd_mm_permute_pd(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_permutevar_ps (easysimd__m128 a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_permutevar_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svuint32_t svindex = svand_n_u32_x(svptrue_b32(), b.sve_u32, 0x03);
    r.sve_f32 = svtbl_f32(a.sve_f32, svindex);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);
    easysimd__m128i_private b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = a_.f32[b_.i32[i] & 3];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutevar_ps
  #define _mm_permutevar_ps(a, b) easysimd_mm_permutevar_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128d
easysimd_mm_permutevar_pd (easysimd__m128d a, easysimd__m128i b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_permutevar_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128d r;
    svuint64_t svindex = svand_n_u64_x(svptrue_b64(), svlsr_n_u64_x(svptrue_b64(), b.sve_u64, 1), 1);
    r.sve_u64 = svtbl_u64(a.sve_u64, svindex);
    return r;
  #else
    easysimd__m128d_private
      r_,
      a_ = easysimd__m128d_to_private(a);
    easysimd__m128i_private b_ = easysimd__m128i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = a_.f64[(b_.i64[i] & 2) >> 1];
    }

    return easysimd__m128d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_permutevar_pd
  #define _mm_permutevar_pd(a, b) easysimd_mm_permutevar_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_permutevar_ps (easysimd__m256 a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_permutevar_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svuint32_t svindex = svand_n_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_0], 0x03);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], svindex);

    svindex = svand_n_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_1], 0x03);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], svindex);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);
    easysimd__m256i_private b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = a_.f32[(b_.i32[i] & 3) + (i & 4)];
    }

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutevar_ps
  #define _mm256_permutevar_ps(a, b) easysimd_mm256_permutevar_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_permutevar_pd (easysimd__m256d a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_permutevar_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svuint64_t svindex = svand_n_u64_x(svptrue_b64(), svlsr_n_u64_x(svptrue_b64(), b.sve_u64[EASYSIMD_SV_INDEX_0], 1), 0x01);
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svtbl_f64(a.sve_f64[EASYSIMD_SV_INDEX_0], svindex);

    svindex = svand_n_u64_x(svptrue_b64(), svlsr_n_u64_x(svptrue_b64(), b.sve_u64[EASYSIMD_SV_INDEX_1], 1), 0x01);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svtbl_f64(a.sve_f64[EASYSIMD_SV_INDEX_1], svindex);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);
    easysimd__m256i_private b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
      r_.f64[i] = a_.f64[((b_.i64[i] & 2) >> 1) + (i & 2)];
    }

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permutevar_pd
  #define _mm256_permutevar_pd(a, b) easysimd_mm256_permutevar_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_permute2f128_ps (easysimd__m256 a, easysimd__m256 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
    int bit_0 = imm8 & 0x1, bit_1 = imm8 & 0x2, bit_3 = imm8 & 0x8;
    if (bit_1 == 0) {
        r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32[bit_0];
    } else {
        r.sve_f32[EASYSIMD_SV_INDEX_0] = b.sve_f32[bit_0];
    }
    if (bit_3) {
        r.sve_f32[EASYSIMD_SV_INDEX_0] = svdup_n_f32(0);
    }
    bit_0 = (imm8 & 0x10) >> 4, bit_1 = imm8 & 0x20, bit_3 = imm8 & 0x80;
    if (bit_1 == 0) {
        r.sve_f32[EASYSIMD_SV_INDEX_1] = a.sve_f32[bit_0];
    } else {
        r.sve_f32[EASYSIMD_SV_INDEX_1] = b.sve_f32[bit_0];
    }
    if (bit_3) {
        r.sve_f32[EASYSIMD_SV_INDEX_1] = svdup_n_f32(0);
    }
    return r;
#else
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a),
    b_ = easysimd__m256_to_private(b);

  r_.m128_private[0] = (imm8 & 0x08) ? easysimd__m128_to_private(easysimd_mm_setzero_ps()) : ((imm8 & 0x02) ? b_.m128_private[(imm8     ) & 1] : a_.m128_private[(imm8     ) & 1]);
  r_.m128_private[1] = (imm8 & 0x80) ? easysimd__m128_to_private(easysimd_mm_setzero_ps()) : ((imm8 & 0x20) ? b_.m128_private[(imm8 >> 4) & 1] : a_.m128_private[(imm8 >> 4) & 1]);

  return easysimd__m256_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_permute2f128_ps(a, b, imm8) _mm256_permute2f128_ps(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permute2f128_ps
  #define _mm256_permute2f128_ps(a, b, imm8) easysimd_mm256_permute2f128_ps(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_permute2f128_pd (easysimd__m256d a, easysimd__m256d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256d r;
    int bit_0 = imm8 & 0x1, bit_1 = imm8 & 0x2, bit_3 = imm8 & 0x8;
    if (bit_1 == 0) {
        r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64[bit_0];
    } else {
        r.sve_f64[EASYSIMD_SV_INDEX_0] = b.sve_f64[bit_0];
    }
    if (bit_3) {
        r.sve_f64[EASYSIMD_SV_INDEX_0] = svdup_n_f64(0);
    }
    bit_0 = (imm8 & 0x10) >> 4, bit_1 = imm8 & 0x20, bit_3 = imm8 & 0x80;
    if (bit_1 == 0) {
        r.sve_f64[EASYSIMD_SV_INDEX_1] = a.sve_f64[bit_0];
    } else {
        r.sve_f64[EASYSIMD_SV_INDEX_1] = b.sve_f64[bit_0];
    }
    if (bit_3) {
        r.sve_f64[EASYSIMD_SV_INDEX_1] = svdup_n_f64(0);
    }
    return r;
#else
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a),
    b_ = easysimd__m256d_to_private(b);

  r_.m128d_private[0] = (imm8 & 0x08) ? easysimd__m128d_to_private(easysimd_mm_setzero_pd()) : ((imm8 & 0x02) ? b_.m128d_private[(imm8     ) & 1] : a_.m128d_private[(imm8     ) & 1]);
  r_.m128d_private[1] = (imm8 & 0x80) ? easysimd__m128d_to_private(easysimd_mm_setzero_pd()) : ((imm8 & 0x20) ? b_.m128d_private[(imm8 >> 4) & 1] : a_.m128d_private[(imm8 >> 4) & 1]);

  return easysimd__m256d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_permute2f128_pd(a, b, imm8) _mm256_permute2f128_pd(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permute2f128_pd
  #define _mm256_permute2f128_pd(a, b, imm8) easysimd_mm256_permute2f128_pd(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_permute2f128_si256 (easysimd__m256i a, easysimd__m256i b, const int imm8)
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
#if defined(EASYSIMD_X86_AVX_NATIVE)
#  define easysimd_mm256_permute2f128_si128(a, b, imm8) _mm256_permute2f128_si128(a, b, imm8)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_permute2f128_si256
  #define _mm256_permute2f128_si256(a, b, imm8) easysimd_mm256_permute2f128_si256(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_rcp_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rcp_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svrecpe_f32(a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svrecpe_f32(a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_rcp_ps(a_.m128[0]);
      r_.m128[1] = easysimd_mm_rcp_ps(a_.m128[1]);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = EASYSIMD_FLOAT32_C(1.0) / a_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rcp_ps
  #define _mm256_rcp_ps(a) easysimd_mm256_rcp_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_rsqrt_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_rsqrt_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svrecpe_f32(svsqrt_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_0]));
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svrecpe_f32(svsqrt_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_1]));
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if defined(easysimd_math_sqrtf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = 1.0f / easysimd_math_sqrtf(a_.f32[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_rsqrt_ps
  #define _mm256_rsqrt_ps(a) easysimd_mm256_rsqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_setr_epi8 (
    int8_t e31, int8_t e30, int8_t e29, int8_t e28, int8_t e27, int8_t e26, int8_t e25, int8_t e24,
    int8_t e23, int8_t e22, int8_t e21, int8_t e20, int8_t e19, int8_t e18, int8_t e17, int8_t e16,
    int8_t e15, int8_t e14, int8_t e13, int8_t e12, int8_t e11, int8_t e10, int8_t  e9, int8_t  e8,
    int8_t  e7, int8_t  e6, int8_t  e5, int8_t  e4, int8_t  e3, int8_t  e2, int8_t  e1, int8_t  e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_setr_epi8(
        e31, e30, e29, e28, e27, e26, e25, e24,
        e23, e22, e21, e20, e19, e18, e17, e16,
        e15, e14, e13, e12, e11, e10,  e9,  e8,
        e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i8[EASYSIMD_SV_INDEX_0] = svdupq_n_s8(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16);
    r.sve_i8[EASYSIMD_SV_INDEX_1] = svdupq_n_s8(e15, e14, e13, e12, e11, e10,  e9,  e8, e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    SET8x16(r.m128i[0], e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16);
    SET8x16(r.m128i[1], e15, e14, e13, e12, e11, e10,  e9,  e8, e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
    return r;
  #else
    return easysimd_mm256_set_epi8(
        e0,  e1,  e2,  e3,  e4,  e5,  e6,  e7,
        e8,  e9, e10, e11, e12, e13, e14, e15,
        e16, e17, e18, e19, e20, e21, e22, e23,
        e24, e25, e26, e27, e28, e29, e30, e31);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setr_epi8
  #define _mm256_setr_epi8(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) \
    easysimd_mm256_setr_epi8(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_setr_epi16 (
    int16_t e15, int16_t e14, int16_t e13, int16_t e12, int16_t e11, int16_t e10, int16_t  e9, int16_t  e8,
    int16_t  e7, int16_t  e6, int16_t  e5, int16_t  e4, int16_t  e3, int16_t  e2, int16_t  e1, int16_t  e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_setr_epi16(
        e15, e14, e13, e12, e11, e10,  e9,  e8,
        e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i16[EASYSIMD_SV_INDEX_0] = svdupq_n_s16(e15, e14, e13, e12, e11, e10,  e9,  e8);
    r.sve_i16[EASYSIMD_SV_INDEX_1] = svdupq_n_s16( e7,  e6,  e5,  e4,  e3,  e2,  e1,  e0);
    return r;
  #else
    return easysimd_mm256_set_epi16(
        e0,  e1,  e2,  e3,  e4,  e5,  e6,  e7,
        e8,  e9, e10, e11, e12, e13, e14, e15);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setr_epi16
  #define _mm256_setr_epi16(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0) \
    easysimd_mm256_setr_epi16(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_setr_epi32 (
    int32_t  e7, int32_t  e6, int32_t  e5, int32_t  e4, int32_t  e3, int32_t  e2, int32_t  e1, int32_t  e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_setr_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = svdupq_n_s32(e7,  e6,  e5,  e4);
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdupq_n_s32(e3,  e2,  e1,  e0);
    return r;
  #else
    return easysimd_mm256_set_epi32(e0, e1, e2, e3, e4, e5, e6, e7);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setr_epi32
  #define _mm256_setr_epi32(e7, e6, e5, e4, e3, e2, e1, e0) \
    easysimd_mm256_setr_epi32(e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_setr_epi64x (int64_t  e3, int64_t  e2, int64_t  e1, int64_t  e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_setr_epi64x(e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i64[EASYSIMD_SV_INDEX_0] = svdupq_n_s64(e3, e2);
    r.sve_i64[EASYSIMD_SV_INDEX_1] = svdupq_n_s64(e1, e0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i r;
    SET64x2(r.m128i[0].neon_i64, e3, e2);
    SET64x2(r.m128i[1].neon_i64, e1, e0);
    return r;
  #else
    return easysimd_mm256_set_epi64x(e0, e1, e2, e3);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setr_epi64x
  #define _mm256_setr_epi64x(e3, e2, e1, e0) \
    easysimd_mm256_setr_epi64x(e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_setr_ps (
    easysimd_float32  e7, easysimd_float32  e6, easysimd_float32  e5, easysimd_float32  e4,
    easysimd_float32  e3, easysimd_float32  e2, easysimd_float32  e1, easysimd_float32  e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_setr_ps(e7, e6, e5, e4, e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svdupq_n_f32(e7, e6, e5, e4);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svdupq_n_f32(e3, e2, e1, e0);
    return r; 
  #else
    return easysimd_mm256_set_ps(e0, e1, e2, e3, e4, e5, e6, e7);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setr_ps
  #define _mm256_setr_ps(e7, e6, e5, e4, e3, e2, e1, e0) \
    easysimd_mm256_setr_ps(e7, e6, e5, e4, e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_setr_pd (easysimd_float64  e3, easysimd_float64  e2, easysimd_float64  e1, easysimd_float64  e0) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_setr_pd(e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svdupq_n_f64(e3, e2);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svdupq_n_f64(e1, e0);
    return r;
  #else
    return easysimd_mm256_set_pd(e0, e1, e2, e3);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setr_pd
  #define _mm256_setr_pd(e3, e2, e1, e0) \
    easysimd_mm256_setr_pd(e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_setr_m128 (easysimd__m128 lo, easysimd__m128 hi) {
  #if defined(EASYSIMD_X86_AVX_NATIVE) && \
      !defined(EASYSIMD_BUG_GCC_REV_247851) && \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,6,0)
    return _mm256_setr_m128(lo, hi);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = lo.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_1] = hi.sve_f32;
    return r;
  #else
    return easysimd_mm256_set_m128(hi, lo);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setr_m128
  #define _mm256_setr_m128(lo, hi) \
    easysimd_mm256_setr_m128(lo, hi)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_setr_m128d (easysimd__m128d lo, easysimd__m128d hi) {
  #if defined(EASYSIMD_X86_AVX_NATIVE) && \
      !defined(EASYSIMD_BUG_GCC_REV_247851) && \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,6,0)
    return _mm256_setr_m128d(lo, hi);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = lo.sve_f64;
    r.sve_f64[EASYSIMD_SV_INDEX_1] = hi.sve_f64;
    return r;
  #else
    return easysimd_mm256_set_m128d(hi, lo);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setr_m128d
  #define _mm256_setr_m128d(lo, hi) \
    easysimd_mm256_setr_m128d(lo, hi)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_setr_m128i (easysimd__m128i lo, easysimd__m128i hi) {
  #if defined(EASYSIMD_X86_AVX_NATIVE) && \
      !defined(EASYSIMD_BUG_GCC_REV_247851) && \
      EASYSIMD_DETECT_CLANG_VERSION_CHECK(3,6,0)
    return _mm256_setr_m128i(lo, hi);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = lo.sve_i32;
    r.sve_i32[EASYSIMD_SV_INDEX_1] = hi.sve_i32;
    return r;
  #else
    return easysimd_mm256_set_m128i(hi, lo);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_setr_m128i
  #define _mm256_setr_m128i(lo, hi) \
    easysimd_mm256_setr_m128i(lo, hi)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_shuffle_ps (easysimd__m256 a, easysimd__m256 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  svbool_t   pgsel    = svdupq_n_b32(1, 1, 0, 0);
  svuint32_t svindexa = svdupq_n_u32(imm8        & 0x03, (imm8 >> 2) & 0x03, 0, 0);
  svuint32_t svindexb = svdupq_n_u32((imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03, 0, 0);
  r.sve_f32[EASYSIMD_SV_INDEX_0] = svsplice(pgsel, svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], svindexa), svtbl_f32(b.sve_f32[EASYSIMD_SV_INDEX_0], svindexb));
  r.sve_f32[EASYSIMD_SV_INDEX_1] = svsplice(pgsel, svtbl_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], svindexa), svtbl_f32(b.sve_f32[EASYSIMD_SV_INDEX_1], svindexb));
  return r;
#else
  easysimd__m256_private
    r_,
    a_ = easysimd__m256_to_private(a),
    b_ = easysimd__m256_to_private(b);

  r_.f32[0] = a_.m128_private[0].f32[(imm8 >> 0) & 3];
  r_.f32[1] = a_.m128_private[0].f32[(imm8 >> 2) & 3];
  r_.f32[2] = b_.m128_private[0].f32[(imm8 >> 4) & 3];
  r_.f32[3] = b_.m128_private[0].f32[(imm8 >> 6) & 3];
  r_.f32[4] = a_.m128_private[1].f32[(imm8 >> 0) & 3];
  r_.f32[5] = a_.m128_private[1].f32[(imm8 >> 2) & 3];
  r_.f32[6] = b_.m128_private[1].f32[(imm8 >> 4) & 3];
  r_.f32[7] = b_.m128_private[1].f32[(imm8 >> 6) & 3];

  return easysimd__m256_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_shuffle_ps(a, b, imm8) _mm256_shuffle_ps(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
  #define easysimd_mm256_shuffle_ps(a, b, imm8) \
      easysimd_mm256_set_m128( \
          easysimd_mm_shuffle_ps(easysimd_mm256_extractf128_ps(a, 1), easysimd_mm256_extractf128_ps(b, 1), (imm8)), \
          easysimd_mm_shuffle_ps(easysimd_mm256_extractf128_ps(a, 0), easysimd_mm256_extractf128_ps(b, 0), (imm8)))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm256_shuffle_ps(a, b, imm8) \
    EASYSIMD_SHUFFLE_VECTOR_(32, 32, a, b, \
      (((imm8) >> 0) & 3) + 0, \
      (((imm8) >> 2) & 3) + 0, \
      (((imm8) >> 4) & 3) + 8, \
      (((imm8) >> 6) & 3) + 8, \
      (((imm8) >> 0) & 3) + 4, \
      (((imm8) >> 2) & 3) + 4, \
      (((imm8) >> 4) & 3) + 12, \
      (((imm8) >> 6) & 3) + 12)

#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shuffle_ps
  #define _mm256_shuffle_ps(a, b, imm8) easysimd_mm256_shuffle_ps(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_shuffle_pd (easysimd__m256d a, easysimd__m256d b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 15) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256d r;
  r.sve_f64[EASYSIMD_SV_INDEX_0][0] = a.sve_f64[EASYSIMD_SV_INDEX_0][(imm8 >> 0) & 1];
  r.sve_f64[EASYSIMD_SV_INDEX_0][1] = b.sve_f64[EASYSIMD_SV_INDEX_0][(imm8 >> 1) & 1];
  r.sve_f64[EASYSIMD_SV_INDEX_1][0] = a.sve_f64[EASYSIMD_SV_INDEX_1][(imm8 >> 2) & 1];
  r.sve_f64[EASYSIMD_SV_INDEX_1][1] = b.sve_f64[EASYSIMD_SV_INDEX_1][(imm8 >> 3) & 1];
  return r;
#else
  easysimd__m256d_private
    r_,
    a_ = easysimd__m256d_to_private(a),
    b_ = easysimd__m256d_to_private(b);

  r_.f64[0] = a_.f64[((imm8     ) & 1)    ];
  r_.f64[1] = b_.f64[((imm8 >> 1) & 1)    ];
  r_.f64[2] = a_.f64[((imm8 >> 2) & 1) | 2];
  r_.f64[3] = b_.f64[((imm8 >> 3) & 1) | 2];

  return easysimd__m256d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_NATIVE)
  #define easysimd_mm256_shuffle_pd(a, b, imm8) _mm256_shuffle_pd(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
  #define easysimd_mm256_shuffle_pd(a, b, imm8) \
      easysimd_mm256_set_m128d( \
          easysimd_mm_shuffle_pd(easysimd_mm256_extractf128_pd(a, 1), easysimd_mm256_extractf128_pd(b, 1), (imm8 >> 0) & 3), \
          easysimd_mm_shuffle_pd(easysimd_mm256_extractf128_pd(a, 0), easysimd_mm256_extractf128_pd(b, 0), (imm8 >> 2) & 3))
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm256_shuffle_pd(a, b, imm8) \
    EASYSIMD_SHUFFLE_VECTOR_(64, 32, a, b, \
      (((imm8) >> 0) & 1) + 0, \
      (((imm8) >> 1) & 1) + 4, \
      (((imm8) >> 2) & 1) + 2, \
      (((imm8) >> 3) & 1) + 6)
#endif
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_shuffle_pd
  #define _mm256_shuffle_pd(a, b, imm8) easysimd_mm256_shuffle_pd(a, b, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_sqrt_ps (easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sqrt_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsqrt_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsqrt_f32_x(svptrue_b32(), a.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_sqrt_ps(a_.m128[0]);
      r_.m128[1] = easysimd_mm_sqrt_ps(a_.m128[1]);
    #elif defined(easysimd_math_sqrtf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_sqrtf(a_.f32[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sqrt_ps
  #define _mm256_sqrt_ps(a) easysimd_mm256_sqrt_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_sqrt_pd (easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sqrt_pd(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsqrt_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsqrt_f64_x(svptrue_b64(), a.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_sqrt_pd(a_.m128d[0]);
      r_.m128d[1] = easysimd_mm_sqrt_pd(a_.m128d[1]);
    #elif defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = easysimd_math_sqrt(a_.f64[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sqrt_pd
  #define _mm256_sqrt_pd(a) easysimd_mm256_sqrt_pd(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_store_epi32(void* mem_addr, easysimd__m256i a) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t *, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_0]);
  svst1_s32(pg, HEDLEY_STATIC_CAST(int32_t *, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_i32[EASYSIMD_SV_INDEX_1]);
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_s32((int32_t *)mem_addr, a.m128i[0].neon_i32);
  vst1q_s32((int32_t *)mem_addr + 4, a.m128i[1].neon_i32);
#else
  easysimd_memcpy(mem_addr, &a, sizeof(a));
#endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_store_epi32
  #define _mm256_store_epi32(mem_addr, a) easysimd_mm256_store_epi32(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_store_epi64(void* mem_addr, easysimd__m256i a) {
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_s64((int64_t *)mem_addr, a.m128i[0].neon_i64);
  vst1q_s64((int64_t *)mem_addr + 2, a.m128i[1].neon_i64);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  svst1_s64(pg, HEDLEY_STATIC_CAST(int64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6), a.sve_i64[EASYSIMD_SV_INDEX_0]);
  svst1_s64(pg, HEDLEY_STATIC_CAST(int64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6), a.sve_i64[EASYSIMD_SV_INDEX_1]);
#else
  easysimd_memcpy(mem_addr, &a, sizeof(a));
#endif
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
#  define easysimd_mm256_store_epi64(mem_addr, a) _mm256_store_epi64(mem_addr, a)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES) && defined(EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES)
  #undef _mm256_store_epi64
  #define _mm256_store_epi64(mem_addr, a) easysimd_mm256_store_epi64(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_store_ps (easysimd_float32 mem_addr[8], easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    _mm256_store_ps(mem_addr, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b32();
  svst1_f32(pg, HEDLEY_STATIC_CAST(float32_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f32[EASYSIMD_SV_INDEX_0]);
  svst1_f32(pg, HEDLEY_STATIC_CAST(float32_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5), a.sve_f32[EASYSIMD_SV_INDEX_1]);
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_f32((float32_t *)mem_addr, a.m128[0].neon_f32);
  vst1q_f32((float32_t *)mem_addr + 4, a.m128[1].neon_f32);
#else
  easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256), &a, sizeof(a));
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_store_ps
  #define _mm256_store_ps(mem_addr, a) easysimd_mm256_store_ps(HEDLEY_REINTERPRET_CAST(float*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_store_pd (easysimd_float64 mem_addr[4], easysimd__m256d a) {
#if defined(EASYSIMD_X86_AVX_NATIVE)
  _mm256_store_pd(mem_addr, a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  svbool_t pg = svptrue_b64();
  svst1_f64(pg, HEDLEY_STATIC_CAST(float64_t*, mem_addr) + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 6), a.sve_f64[EASYSIMD_SV_INDEX_0]);
  svst1_f64(pg, HEDLEY_STATIC_CAST(float64_t*, mem_addr) + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 6), a.sve_f64[EASYSIMD_SV_INDEX_1]);
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  vst1q_f64((float64_t *)mem_addr, a.m128d[0].neon_f64);
  vst1q_f64((float64_t *)mem_addr + 2, a.m128d[1].neon_f64);
#else
  easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256d), &a, sizeof(a));
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_store_pd
  #define _mm256_store_pd(mem_addr, a) easysimd_mm256_store_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_store_si256 (easysimd__m256i* mem_addr, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    _mm256_store_si256(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b32();
    svst1_s32(pg, ((int32_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5 )), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    svst1_s32(pg, ((int32_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5 )), a.sve_i32[EASYSIMD_SV_INDEX_1]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_s32((int32_t*)mem_addr, a.m128i[0].neon_i32);
    vst1q_s32((int32_t*)mem_addr + 4, a.m128i[1].neon_i32);
  #else
    easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256i), &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_store_si256
  #define _mm256_store_si256(mem_addr, a) easysimd_mm256_store_si256(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_storeu_ps (easysimd_float32 mem_addr[8], easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    _mm256_storeu_ps(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    svst1_f32(pg, &(mem_addr[EASYSIMD_SV_INDEX_0 << 2]), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    svst1_f32(pg, &(mem_addr[EASYSIMD_SV_INDEX_1 << 2]), a.sve_f32[EASYSIMD_SV_INDEX_1]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_f32(&(mem_addr[0]), a.m128[0].neon_f32);
    vst1q_f32(&(mem_addr[4]), a.m128[1].neon_f32);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_storeu_ps
  #define _mm256_storeu_ps(mem_addr, a) easysimd_mm256_storeu_ps(HEDLEY_REINTERPRET_CAST(float*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_storeu_pd (easysimd_float64 mem_addr[4], easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    _mm256_storeu_pd(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b64();
    svst1_f64(pg, &(mem_addr[EASYSIMD_SV_INDEX_0 << 2]), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    svst1_f64(pg, &(mem_addr[EASYSIMD_SV_INDEX_1 << 1]), a.sve_f64[EASYSIMD_SV_INDEX_1]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_f64((float64_t *)mem_addr    , a.m128d[0].neon_f64);
    vst1q_f64((float64_t *)mem_addr + 2, a.m128d[1].neon_f64);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_storeu_pd
  #define _mm256_storeu_pd(mem_addr, a) easysimd_mm256_storeu_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_storeu_si256 (void* mem_addr, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    _mm256_storeu_si256(EASYSIMD_ALIGN_CAST(__m256i*, mem_addr), a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    #if 0
      vst1q_s32((int32_t *)mem_addr    , a.m128i[0].neon_i32);
      vst1q_s32((int32_t *)mem_addr + 4, a.m128i[0].neon_i32);
    #endif
    int32_t *addr_a = (int32_t *)(&a);
    __asm__ __volatile__ (
      "ldp    q0,  q1, [%[pa]]      \n\t"
      "stp    q0,  q1, [%[mem]]     \n\t"
      :[mem] "+r"(mem_addr)
      :[pa] "r"(addr_a)
      :"q0", "q1", "memory"
    );
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svbool_t pg = svptrue_b32();
    svst1_u32(pg, (uint32_t *)mem_addr , a.sve_u32[EASYSIMD_SV_INDEX_0]);
    svst1_u32(pg, (uint32_t *)mem_addr + (__ARM_FEATURE_SVE_BITS / 32), a.sve_u32[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd_memcpy(mem_addr, &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_storeu_si256
  #define _mm256_storeu_si256(mem_addr, a) easysimd_mm256_storeu_si256(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_storeu2_m128 (easysimd_float32 hi_addr[4], easysimd_float32 lo_addr[4], easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE) && !defined(EASYSIMD_BUG_GCC_91341) && !defined(EASYSIMD_BUG_MCST_LCC_MISSING_AVX_LOAD_STORE_M128_FUNCS)
    _mm256_storeu2_m128(hi_addr, lo_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f32(svptrue_b32(), &(lo_addr[0]), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    svst1_f32(svptrue_b32(), &(hi_addr[0]), a.sve_f32[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd_mm_storeu_ps(lo_addr, easysimd_mm256_castps256_ps128(a));
    easysimd_mm_storeu_ps(hi_addr, easysimd_mm256_extractf128_ps(a, 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_storeu2_m128
  #define _mm256_storeu2_m128(hi_addr, lo_addr, a) easysimd_mm256_storeu2_m128(hi_addr, lo_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_storeu2_m128d (easysimd_float64 hi_addr[2], easysimd_float64 lo_addr[2], easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE) && !defined(EASYSIMD_BUG_GCC_91341) && !defined(EASYSIMD_BUG_MCST_LCC_MISSING_AVX_LOAD_STORE_M128_FUNCS)
    _mm256_storeu2_m128d(hi_addr, lo_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f64(svptrue_b64(), &(lo_addr[0]), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    svst1_f64(svptrue_b64(), &(hi_addr[0]), a.sve_f64[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd_mm_storeu_pd(lo_addr, easysimd_mm256_castpd256_pd128(a));
    easysimd_mm_storeu_pd(hi_addr, easysimd_mm256_extractf128_pd(a, 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_storeu2_m128d
  #define _mm256_storeu2_m128d(hi_addr, lo_addr, a) easysimd_mm256_storeu2_m128d(hi_addr, lo_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_storeu2_m128i (easysimd__m128i* hi_addr, easysimd__m128i* lo_addr, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE) && !defined(EASYSIMD_BUG_GCC_91341) && !defined(EASYSIMD_BUG_MCST_LCC_MISSING_AVX_LOAD_STORE_M128_FUNCS)
    _mm256_storeu2_m128i(hi_addr, lo_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_s32(svptrue_b32(), (int32_t *)lo_addr, a.sve_i32[EASYSIMD_SV_INDEX_0]);
    svst1_s32(svptrue_b32(), (int32_t *)hi_addr, a.sve_i32[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd_mm_storeu_si128(lo_addr, easysimd_mm256_castsi256_si128(a));
    easysimd_mm_storeu_si128(hi_addr, easysimd_mm256_extractf128_si256(a, 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_storeu2_m128i
  #define _mm256_storeu2_m128i(hi_addr, lo_addr, a) easysimd_mm256_storeu2_m128i(hi_addr, lo_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_stream_ps (easysimd_float32 mem_addr[8], easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    _mm256_stream_ps(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f32(svptrue_b32(), &(mem_addr[0]), a.sve_f32[EASYSIMD_SV_INDEX_0]);
    svst1_f32(svptrue_b32(), &(mem_addr[4]), a.sve_f32[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256), &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_stream_ps
  #define _mm256_stream_ps(mem_addr, a) easysimd_mm256_stream_ps(HEDLEY_REINTERPRET_CAST(float*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_stream_pd (easysimd_float64 mem_addr[4], easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    _mm256_stream_pd(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f64(svptrue_b64(), &(mem_addr[0]), a.sve_f64[EASYSIMD_SV_INDEX_0]);
    svst1_f64(svptrue_b64(), &(mem_addr[2]), a.sve_f64[EASYSIMD_SV_INDEX_1]);
  #else
    easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256d), &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_stream_pd
  #define _mm256_stream_pd(mem_addr, a) easysimd_mm256_stream_pd(HEDLEY_REINTERPRET_CAST(double*, mem_addr), a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm256_stream_si256 (easysimd__m256i* mem_addr, easysimd__m256i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    _mm256_stream_si256(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_svbool_t pg = svptrue_b32();
    svst1_s32(pg, ((int32_t *)mem_addr + EASYSIMD_SV_INDEX_0 * (__ARM_FEATURE_SVE_BITS >> 5 )), a.sve_i32[EASYSIMD_SV_INDEX_0]);
    svst1_s32(pg, ((int32_t *)mem_addr + EASYSIMD_SV_INDEX_1 * (__ARM_FEATURE_SVE_BITS >> 5 )), a.sve_i32[EASYSIMD_SV_INDEX_1]);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    vst1q_s32((int32_t*)mem_addr, a.m128i[0].neon_i32);
    vst1q_s32((int32_t*)mem_addr + 4, a.m128i[1].neon_i32);
  #else
    easysimd_memcpy(EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m256i), &a, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_stream_si256
  #define _mm256_stream_si256(mem_addr, a) easysimd_mm256_stream_si256(mem_addr, a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_sub_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sub_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 r;
    r.m128[0].neon_f32 = vsubq_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    r.m128[1].neon_f32 = vsubq_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svbool_t pg = svptrue_b32();
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsub_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsub_f32_z(pg, a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_sub_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_sub_ps(a_.m128[1], b_.m128[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 - b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i] - b_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sub_ps
  #define _mm256_sub_ps(a, b) easysimd_mm256_sub_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_hsub_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_hsub_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    svfloat32_t sv1, sv2;
    sv1 = svuzp1_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    sv2 = svuzp2_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svsub_f32_x(svptrue_b32(), sv1, sv2);
    sv1 = svuzp1_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    sv2 = svuzp2_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svsub_f32_x(svptrue_b32(), sv1, sv2);
    return r;
  #else
      return easysimd_mm256_sub_ps(easysimd_x_mm256_deinterleaveeven_ps(a, b), easysimd_x_mm256_deinterleaveodd_ps(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hsub_ps
  #define _mm256_hsub_ps(a, b) easysimd_mm256_hsub_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_sub_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_sub_pd(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256d r;
    r.m128d[0].neon_f64 = vsubq_f64(a.m128d[0].neon_f64, b.m128d[0].neon_f64);
    r.m128d[1].neon_f64 = vsubq_f64(a.m128d[1].neon_f64, b.m128d[1].neon_f64);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svbool_t pg = svptrue_b64();
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsub_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsub_f64_z(pg, a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_sub_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_sub_pd(a_.m128d[1], b_.m128d[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f64 = a_.f64 - b_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = a_.f64[i] - b_.f64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_sub_pd
  #define _mm256_sub_pd(a, b) easysimd_mm256_sub_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_hsub_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_hsub_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    svfloat64_t sv1, sv2;
    sv1 = svuzp1_f64(a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    sv2 = svuzp2_f64(a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svsub_f64_x(svptrue_b64(), sv1, sv2);
    sv1 = svuzp1_f64(a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    sv2 = svuzp2_f64(a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svsub_f64_x(svptrue_b64(), sv1, sv2);
    return r;
  #else
      return easysimd_mm256_sub_pd(easysimd_x_mm256_deinterleaveeven_pd(a, b), easysimd_x_mm256_deinterleaveodd_pd(a, b));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_hsub_pd
  #define _mm256_hsub_pd(a, b) easysimd_mm256_hsub_pd(a, b)
#endif

#if defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
  HEDLEY_DIAGNOSTIC_PUSH
  EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_undefined_ps (void) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256 r;
  return r;
#else
  easysimd__m256_private r_;

  #if \
      defined(EASYSIMD_X86_AVX_NATIVE) && \
      (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(5,0,0)) && \
      (!defined(__has_builtin) || HEDLEY_HAS_BUILTIN(__builtin_ia32_undef256))
    r_.n = _mm256_undefined_ps();
  #elif !defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
    r_ = easysimd__m256_to_private(easysimd_mm256_setzero_ps());
  #endif

    return easysimd__m256_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_undefined_ps
  #define _mm256_undefined_ps() easysimd_mm256_undefined_ps()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_undefined_pd (void) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256d r;
  return r;
#else
  easysimd__m256d_private r_;

  #if \
      defined(EASYSIMD_X86_AVX_NATIVE) && \
      (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(5,0,0)) && \
      (!defined(__has_builtin) || HEDLEY_HAS_BUILTIN(__builtin_ia32_undef256))
    r_.n = _mm256_undefined_pd();
  #elif !defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
    r_ = easysimd__m256d_to_private(easysimd_mm256_setzero_pd());
  #endif

    return easysimd__m256d_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_undefined_pd
  #define _mm256_undefined_pd() easysimd_mm256_undefined_pd()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_undefined_si256 (void) {
#if defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m256i r;
  return r;
#else
    easysimd__m256i_private r_;
  #if \
      defined(EASYSIMD_X86_AVX_NATIVE) && \
      (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(5,0,0)) && \
      (!defined(__has_builtin) || HEDLEY_HAS_BUILTIN(__builtin_ia32_undef256))
    r_.n = _mm256_undefined_si256();
  #elif !defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
    r_ = easysimd__m256i_to_private(easysimd_mm256_setzero_si256());
  #endif

    return easysimd__m256i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_undefined_si256
  #define _mm256_undefined_si256() easysimd_mm256_undefined_si256()
#endif

#if defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
  HEDLEY_DIAGNOSTIC_POP
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_xor_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_xor_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_u32[EASYSIMD_SV_INDEX_0] = sveor_u32_x(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    r.sve_u32[EASYSIMD_SV_INDEX_1] = sveor_u32_x(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128[0] = easysimd_mm_xor_ps(a_.m128[0], b_.m128[0]);
      r_.m128[1] = easysimd_mm_xor_ps(a_.m128[1], b_.m128[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f ^ b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i] ^ b_.u32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_xor_ps
  #define _mm256_xor_ps(a, b) easysimd_mm256_xor_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_xor_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_xor_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_u64[EASYSIMD_SV_INDEX_0] = sveor_u64_x(svptrue_b64(), a.sve_u64[EASYSIMD_SV_INDEX_0], b.sve_u64[EASYSIMD_SV_INDEX_0]);
    r.sve_u64[EASYSIMD_SV_INDEX_1] = sveor_u64_x(svptrue_b64(), a.sve_u64[EASYSIMD_SV_INDEX_1], b.sve_u64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r_.m128d[0] = easysimd_mm_xor_pd(a_.m128d[0], b_.m128d[0]);
      r_.m128d[1] = easysimd_mm_xor_pd(a_.m128d[1], b_.m128d[1]);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f ^ b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u64) / sizeof(r_.u64[0])) ; i++) {
        r_.u64[i] = a_.u64[i] ^ b_.u64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_xor_pd
  #define _mm256_xor_pd(a, b) easysimd_mm256_xor_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_xorsign_ps(easysimd__m256 dest, easysimd__m256 src) {
  return easysimd_mm256_xor_ps(easysimd_mm256_and_ps(easysimd_mm256_set1_ps(-0.0f), src), dest);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_xorsign_pd(easysimd__m256d dest, easysimd__m256d src) {
  return easysimd_mm256_xor_pd(easysimd_mm256_and_pd(easysimd_mm256_set1_pd(-0.0), src), dest);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_x_mm256_negate_ps(easysimd__m256 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return easysimd_mm256_xor_ps(a,_mm256_set1_ps(EASYSIMD_FLOAT32_C(-0.0)));
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a);

    #if defined(EASYSIMD_VECTOR_NEGATE)
      r_.f32 = -a_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = -a_.f32[i];
      }
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_x_mm256_negate_pd(easysimd__m256d a) {
  #if defined(EASYSIMD_X86_AVX2_NATIVE)
    return easysimd_mm256_xor_pd(a, _mm256_set1_pd(EASYSIMD_FLOAT64_C(-0.0)));
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a);

    #if defined(EASYSIMD_VECTOR_NEGATE)
      r_.f64 = -a_.f64;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f64) / sizeof(r_.f64[0])) ; i++) {
        r_.f64[i] = -a_.f64[i];
      }
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_unpackhi_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_unpackhi_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svzip2_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svzip2_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 ret;
    ret.m128[0].neon_f32 = vzip2q_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    ret.m128[1].neon_f32 = vzip2q_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    return ret;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.f32, b_.f32, 2, 10, 3, 11, 6, 14, 7, 15);
    #else
      r_.f32[0] = a_.f32[2];
      r_.f32[1] = b_.f32[2];
      r_.f32[2] = a_.f32[3];
      r_.f32[3] = b_.f32[3];
      r_.f32[4] = a_.f32[6];
      r_.f32[5] = b_.f32[6];
      r_.f32[6] = a_.f32[7];
      r_.f32[7] = b_.f32[7];
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpackhi_ps
  #define _mm256_unpackhi_ps(a, b) easysimd_mm256_unpackhi_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_unpackhi_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_unpackhi_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svzip2_f64(a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svzip2_f64(a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 32, a_.f64, b_.f64, 1, 5, 3, 7);
    #else
      r_.f64[0] = a_.f64[1];
      r_.f64[1] = b_.f64[1];
      r_.f64[2] = a_.f64[3];
      r_.f64[3] = b_.f64[3];
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpackhi_pd
  #define _mm256_unpackhi_pd(a, b) easysimd_mm256_unpackhi_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_unpacklo_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_unpacklo_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = svzip1_f32(a.sve_f32[EASYSIMD_SV_INDEX_0], b.sve_f32[EASYSIMD_SV_INDEX_0]);
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svzip1_f32(a.sve_f32[EASYSIMD_SV_INDEX_1], b.sve_f32[EASYSIMD_SV_INDEX_1]);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256 ret;
    ret.m128[0].neon_f32 = vzip1q_f32(a.m128[0].neon_f32, b.m128[0].neon_f32);
    ret.m128[1].neon_f32 = vzip1q_f32(a.m128[1].neon_f32, b.m128[1].neon_f32);
    return ret;
  #else
    easysimd__m256_private
      r_,
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 32, a_.f32, b_.f32, 0, 8, 1, 9, 4, 12, 5, 13);
    #else
      r_.f32[0] = a_.f32[0];
      r_.f32[1] = b_.f32[0];
      r_.f32[2] = a_.f32[1];
      r_.f32[3] = b_.f32[1];
      r_.f32[4] = a_.f32[4];
      r_.f32[5] = b_.f32[4];
      r_.f32[6] = a_.f32[5];
      r_.f32[7] = b_.f32[5];
    #endif

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpacklo_ps
  #define _mm256_unpacklo_ps(a, b) easysimd_mm256_unpacklo_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_unpacklo_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_unpacklo_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = svzip1_f64(a.sve_f64[EASYSIMD_SV_INDEX_0], b.sve_f64[EASYSIMD_SV_INDEX_0]);
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svzip1_f64(a.sve_f64[EASYSIMD_SV_INDEX_1], b.sve_f64[EASYSIMD_SV_INDEX_1]);
    return r;
  #else
    easysimd__m256d_private
      r_,
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f64 = EASYSIMD_SHUFFLE_VECTOR_(64, 32, a_.f64, b_.f64, 0, 4, 2, 6);
    #else
      r_.f64[0] = a_.f64[0];
      r_.f64[1] = b_.f64[0];
      r_.f64[2] = a_.f64[2];
      r_.f64[3] = b_.f64[2];
    #endif

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_unpacklo_pd
  #define _mm256_unpacklo_pd(a, b) easysimd_mm256_unpacklo_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256
easysimd_mm256_zextps128_ps256 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_insertf128_ps(_mm256_setzero_ps(), a, 0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256 r;
    r.sve_f32[EASYSIMD_SV_INDEX_0] = a.sve_f32;
    r.sve_f32[EASYSIMD_SV_INDEX_1] = svdup_n_f32(0.0f);
    return r;
  #else
    easysimd__m256_private r_;

    r_.m128_private[0] = easysimd__m128_to_private(a);
    r_.m128_private[1] = easysimd__m128_to_private(easysimd_mm_setzero_ps());

    return easysimd__m256_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_zextps128_ps256
  #define _mm256_zextps128_ps256(a) easysimd_mm256_zextps128_ps256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256d
easysimd_mm256_zextpd128_pd256 (easysimd__m128d a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_insertf128_pd(_mm256_setzero_pd(), a, 0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m256d r;
    r.sve_f64[EASYSIMD_SV_INDEX_0] = a.sve_f64;
    r.sve_f64[EASYSIMD_SV_INDEX_1] = svdup_n_f64(0.0);
    return r;
  #else
    easysimd__m256d_private r_;

    r_.m128d_private[0] = easysimd__m128d_to_private(a);
    r_.m128d_private[1] = easysimd__m128d_to_private(easysimd_mm_setzero_pd());

    return easysimd__m256d_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_zextpd128_pd256
  #define _mm256_zextpd128_pd256(a) easysimd_mm256_zextpd128_pd256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m256i
easysimd_mm256_zextsi128_si256 (easysimd__m128i a) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_insertf128_si256(_mm256_setzero_si256(), a, 0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE) 
    easysimd__m256i r;
    r.sve_i32[EASYSIMD_SV_INDEX_0] = a.sve_i32;
    r.sve_i32[EASYSIMD_SV_INDEX_1] = svdup_n_s32(0);
    return r;
  #else
    easysimd__m256i_private r_;

    r_.m128i_private[0] = easysimd__m128i_to_private(a);
    r_.m128i_private[1] = easysimd__m128i_to_private(easysimd_mm_setzero_si128());

    return easysimd__m256i_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_zextsi128_si256
  #define _mm256_zextsi128_si256(a) easysimd_mm256_zextsi128_si256(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_testc_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_testc_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint32_t svtemp = svbic_s32_x(svptrue_b32(), b.sve_i32, a.sve_i32);
    int32_t   cf     = svorv_s32(svptrue_b32(), svtemp);
    return cf < 0 ? 0 : 1;
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

      uint_fast32_t r = 0;
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
        r |= ~a_.u32[i] & b_.u32[i];
      }

      return HEDLEY_STATIC_CAST(int, ((~r >> 31) & 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_testc_ps
  #define _mm_testc_ps(a, b) easysimd_mm_testc_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_testc_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_testc_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint64_t svtemp = svbic_s64_x(svptrue_b64(), b.sve_i64, a.sve_i64);
    int64_t   cf     = svorv_s64(svptrue_b64(), svtemp);
    return cf < 0 ? 0 : 1;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

      uint_fast64_t r = 0;
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
        r |= ~a_.u64[i] & b_.u64[i];
      }

      return HEDLEY_STATIC_CAST(int, ((~r >> 63) & 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_testc_pd
  #define _mm_testc_pd(a, b) easysimd_mm_testc_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_testc_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_testc_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint32_t svtemp;
    int32_t cf0, cf1 = 0;
    svtemp = svbic_s32_x(svptrue_b32(), b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    cf0    = svorv_s32(svptrue_b32(), svtemp);
    #if __ARM_FEATURE_SVE_BITS == 128
      svtemp = svbic_s32_x(svptrue_b32(), b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
      cf1    = svorv_s32(svptrue_b32(), svtemp);
    #endif
    return (cf0 | cf1) < 0 ? 0 : 1;
  #else
    uint_fast32_t r = 0;
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
      r |= ~a_.u32[i] & b_.u32[i];
    }

    return HEDLEY_STATIC_CAST(int, ((~r >> 31) & 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testc_ps
  #define _mm256_testc_ps(a, b) easysimd_mm256_testc_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_testc_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_testc_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint64_t svtemp;
    int64_t cf0, cf1 = 0;
    svtemp = svbic_s64_x(svptrue_b64(), b.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_0]);
    cf0    = svorv_s64(svptrue_b64(), svtemp);
    #if __ARM_FEATURE_SVE_BITS == 128
      svtemp = svbic_s64_x(svptrue_b64(), b.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_1]);
      cf1    = svorv_s64(svptrue_b64(), svtemp);
    #endif
    return (cf0 | cf1) < 0 ? 0 : 1;
  #else
    uint_fast64_t r = 0;
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r |= ~a_.u64[i] & b_.u64[i];
    }

    return HEDLEY_STATIC_CAST(int, ((~r >> 63) & 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testc_pd
  #define _mm256_testc_pd(a, b) easysimd_mm256_testc_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_testc_si256 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_testc_si256(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i tmp;
    easysimd__m128i r;
    tmp.m128i[0].neon_u32 = vbicq_u32(b.m128i[0].neon_u32, a.m128i[0].neon_u32);
    tmp.m128i[1].neon_u32 = vbicq_u32(b.m128i[1].neon_u32, a.m128i[1].neon_u32);
    r.neon_u32 = vorrq_u32(tmp.m128i[0].neon_u32, tmp.m128i[0].neon_u32);
    return vaddvq_u32(r.neon_u32) ? 0 : 1;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svuint32_t svtemp;
    uint32_t cf0, cf1 = 0;
    svtemp = svbic_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_0], a.sve_u32[EASYSIMD_SV_INDEX_0]);
    cf0    = svorv_u32(svptrue_b32(), svtemp);
    #if __ARM_FEATURE_SVE_BITS == 128
      svtemp = svbic_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_1], a.sve_u32[EASYSIMD_SV_INDEX_1]);
      cf1    = svorv_u32(svptrue_b32(), svtemp);
    #endif
    return (cf0 | cf1) == 0 ? 1 : 0;
  #else
    int_fast32_t r = 0;
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.i32f) / sizeof(a_.i32f[0])) ; i++) {
      r |= ~a_.i32f[i] & b_.i32f[i];
    }

    return HEDLEY_STATIC_CAST(int, !r);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testc_si256
  #define _mm256_testc_si256(a, b) easysimd_mm256_testc_si256(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_testz_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_testz_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint32_t svtemp = svand_s32_x(svptrue_b32(), a.sve_i32, b.sve_i32);
    int32_t   zf     = svorv_s32(svptrue_b32(), svtemp);
    return zf < 0 ? 0 : 1;
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

      uint_fast32_t r = 0;
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
        r |= a_.u32[i] & b_.u32[i];
      }

      return HEDLEY_STATIC_CAST(int, ((~r >> 31) & 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_testz_ps
  #define _mm_testz_ps(a, b) easysimd_mm_testz_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_testz_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_testz_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint64_t svtemp = svand_s64_x(svptrue_b64(), a.sve_i64, b.sve_i64);
    int64_t   zf     = svorv_s64(svptrue_b64(), svtemp);
    return zf < 0 ? 0 : 1;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

      uint_fast64_t r = 0;
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
        r |= a_.u64[i] & b_.u64[i];
      }

      return HEDLEY_STATIC_CAST(int, ((~r >> 63) & 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_testz_pd
  #define _mm_testz_pd(a, b) easysimd_mm_testz_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_testz_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_testz_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint32_t svtemp;
    int32_t zf0, zf1 = 0;
    svtemp = svand_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    zf0    = svorv_s32(svptrue_b32(), svtemp);
    #if __ARM_FEATURE_SVE_BITS == 128
      svtemp = svand_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
      zf1    = svorv_s32(svptrue_b32(), svtemp);
    #endif
    return (zf0 | zf1) < 0 ? 0 : 1;
  #else
    uint_fast32_t r = 0;
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
      r |= a_.u32[i] & b_.u32[i];
    }

    return HEDLEY_STATIC_CAST(int, ((~r >> 31) & 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testz_ps
  #define _mm256_testz_ps(a, b) easysimd_mm256_testz_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_testz_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_testz_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint64_t svtemp;
    int64_t zf0, zf1 = 0;
    svtemp = svand_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    zf0    = svorv_s64(svptrue_b64(), svtemp);
    #if __ARM_FEATURE_SVE_BITS == 128
      svtemp = svand_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
      zf1    = svorv_s64(svptrue_b64(), svtemp);
    #endif
    return (zf0 | zf1) < 0 ? 0 : 1;
  #else
    uint_fast64_t r = 0;
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      r |= a_.u64[i] & b_.u64[i];
    }

    return HEDLEY_STATIC_CAST(int, ((~r >> 63) & 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testz_pd
  #define _mm256_testz_pd(a, b) easysimd_mm256_testz_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_testz_si256 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_testz_si256(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    uint64_t res = 0;
    svbool_t pg = svptrue_b64();
    res += svaddv_s64(pg, svand_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]));
    res += svaddv_s64(pg, svand_s64_x(pg, a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]));
    return res == 0 ? 1 : 0;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m256i res;
    res.m128i[0].neon_i64 = vandq_s64(a.m128i[0].neon_i64, b.m128i[0].neon_i64);
    res.m128i[1].neon_i64 = vandq_s64(a.m128i[1].neon_i64, b.m128i[1].neon_i64);
    int64x2_t tmp = vorrq_s64(res.m128i[0].neon_i64, res.m128i[1].neon_i64);
    return !(vgetq_lane_s64(tmp, 0) | vgetq_lane_s64(tmp, 1));
  #else
    int_fast32_t r = 0;
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    #if EASYSIMD_NATURAL_VECTOR_SIZE_LE(128)
      r = easysimd_mm_testz_si128(a_.m128i[0], b_.m128i[0]) && easysimd_mm_testz_si128(a_.m128i[1], b_.m128i[1]);
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < (sizeof(a_.i32f) / sizeof(a_.i32f[0])) ; i++) {
        r |= a_.i32f[i] & b_.i32f[i];
      }

      r = !r;
    #endif

    return HEDLEY_STATIC_CAST(int, r);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testz_si256
  #define _mm256_testz_si256(a, b) easysimd_mm256_testz_si256(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_testnzc_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_testnzc_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint32_t svtemp = svbic_s32_x(svptrue_b32(), b.sve_i32, a.sve_i32);
    int32_t   cf     = svorv_s32(svptrue_b32(), svtemp);
              svtemp = svand_s32_x(svptrue_b32(), a.sve_i32, b.sve_i32);
    int32_t   zf     = svorv_s32(svptrue_b32(), svtemp);
    return (cf & zf) < 0 ? 1 : 0;
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

      uint32_t rz = 0, rc = 0;
      for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
        rc |= ~a_.u32[i] & b_.u32[i];
        rz |=  a_.u32[i] & b_.u32[i];
      }

      return
        (rc >> ((sizeof(rc) * CHAR_BIT) - 1)) &
        (rz >> ((sizeof(rz) * CHAR_BIT) - 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_testnzc_ps
  #define _mm_testnzc_ps(a, b) easysimd_mm_testnzc_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_testnzc_pd (easysimd__m128d a, easysimd__m128d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm_testnzc_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint64_t svtemp = svbic_s64_x(svptrue_b64(), b.sve_i64, a.sve_i64);
    int64_t   cf     = svorv_s64(svptrue_b64(), svtemp);
              svtemp = svand_s64_x(svptrue_b64(), a.sve_i64, b.sve_i64);
    int64_t   zf     = svorv_s64(svptrue_b64(), svtemp);
    return (cf & zf) < 0 ? 1 : 0;
  #else
    easysimd__m128d_private
      a_ = easysimd__m128d_to_private(a),
      b_ = easysimd__m128d_to_private(b);

      uint64_t rc = 0, rz = 0;
      for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
        rc |= ~a_.u64[i] & b_.u64[i];
        rz |=  a_.u64[i] & b_.u64[i];
      }

      return
        (rc >> ((sizeof(rc) * CHAR_BIT) - 1)) &
        (rz >> ((sizeof(rz) * CHAR_BIT) - 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm_testnzc_pd
  #define _mm_testnzc_pd(a, b) easysimd_mm_testnzc_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_testnzc_ps (easysimd__m256 a, easysimd__m256 b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_testnzc_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint32_t svtemp;
    int32_t   cf0, zf0, cf1 = 0, zf1 = 0;
    svtemp  = svbic_s32_x(svptrue_b32(), b.sve_i32[EASYSIMD_SV_INDEX_0], a.sve_i32[EASYSIMD_SV_INDEX_0]);
    cf0     = svorv_s32(svptrue_b32(), svtemp);
    svtemp  = svand_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_0], b.sve_i32[EASYSIMD_SV_INDEX_0]);
    zf0     = svorv_s32(svptrue_b32(), svtemp);
    #if __ARM_FEATURE_SVE_BITS == 128
      svtemp  = svbic_s32_x(svptrue_b32(), b.sve_i32[EASYSIMD_SV_INDEX_1], a.sve_i32[EASYSIMD_SV_INDEX_1]);
      cf1     = svorv_s32(svptrue_b32(), svtemp);
      svtemp  = svand_s32_x(svptrue_b32(), a.sve_i32[EASYSIMD_SV_INDEX_1], b.sve_i32[EASYSIMD_SV_INDEX_1]);
      zf1     = svorv_s32(svptrue_b32(), svtemp);
    #endif
    return ((cf0 | cf1) & (zf0 | zf1)) < 0 ? 1 : 0;
  #else
    uint32_t rc = 0, rz = 0;
    easysimd__m256_private
      a_ = easysimd__m256_to_private(a),
      b_ = easysimd__m256_to_private(b);

    for (size_t i = 0 ; i < (sizeof(a_.u32) / sizeof(a_.u32[0])) ; i++) {
      rc |= ~a_.u32[i] & b_.u32[i];
      rz |=  a_.u32[i] & b_.u32[i];
    }

    return
      (rc >> ((sizeof(rc) * CHAR_BIT) - 1)) &
      (rz >> ((sizeof(rz) * CHAR_BIT) - 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testnzc_ps
  #define _mm256_testnzc_ps(a, b) easysimd_mm256_testnzc_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_testnzc_pd (easysimd__m256d a, easysimd__m256d b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_testnzc_pd(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svint64_t svtemp0, svtemp1;
    int64_t   cf0, zf0, cf1 = 0, zf1 = 0;
    svtemp0  = svbic_s64_x(svptrue_b64(), b.sve_i64[EASYSIMD_SV_INDEX_0], a.sve_i64[EASYSIMD_SV_INDEX_0]);
    svtemp1  = svbic_s64_x(svptrue_b64(), b.sve_i64[EASYSIMD_SV_INDEX_1], a.sve_i64[EASYSIMD_SV_INDEX_1]);
    cf0      = svorv_s64(svptrue_b64(), svtemp0);
    cf1      = svorv_s64(svptrue_b64(), svtemp1);
    svtemp0  = svand_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_0], b.sve_i64[EASYSIMD_SV_INDEX_0]);
    svtemp1  = svand_s64_x(svptrue_b64(), a.sve_i64[EASYSIMD_SV_INDEX_1], b.sve_i64[EASYSIMD_SV_INDEX_1]);
    zf0      = svorv_s64(svptrue_b64(), svtemp0);
    zf1      = svorv_s64(svptrue_b64(), svtemp1);
    return ((cf0 | cf1) & (zf0 | zf1)) < 0 ? 1 : 0;
  #else
    uint64_t rc = 0, rz = 0;
    easysimd__m256d_private
      a_ = easysimd__m256d_to_private(a),
      b_ = easysimd__m256d_to_private(b);

    for (size_t i = 0 ; i < (sizeof(a_.u64) / sizeof(a_.u64[0])) ; i++) {
      rc |= ~a_.u64[i] & b_.u64[i];
      rz |=  a_.u64[i] & b_.u64[i];
    }

    return
      (rc >> ((sizeof(rc) * CHAR_BIT) - 1)) &
      (rz >> ((sizeof(rz) * CHAR_BIT) - 1));
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testnzc_pd
  #define _mm256_testnzc_pd(a, b) easysimd_mm256_testnzc_pd(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm256_testnzc_si256 (easysimd__m256i a, easysimd__m256i b) {
  #if defined(EASYSIMD_X86_AVX_NATIVE)
    return _mm256_testnzc_si256(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svuint32_t svtemp0, svtemp1;
    uint32_t cf0, zf0, cf1 = 0, zf1 = 0;
    svtemp0 = svbic_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_0], a.sve_u32[EASYSIMD_SV_INDEX_0]);
    svtemp1 = svbic_u32_x(svptrue_b32(), b.sve_u32[EASYSIMD_SV_INDEX_1], a.sve_u32[EASYSIMD_SV_INDEX_1]);
    cf0     = svorv_u32(svptrue_b32(), svtemp0);
    cf1     = svorv_u32(svptrue_b32(), svtemp1);
    svtemp0 = svand_u32_x(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_0], b.sve_u32[EASYSIMD_SV_INDEX_0]);
    svtemp1 = svand_u32_x(svptrue_b32(), a.sve_u32[EASYSIMD_SV_INDEX_1], b.sve_u32[EASYSIMD_SV_INDEX_1]);
    zf0     = svorv_u32(svptrue_b32(), svtemp0);
    zf1     = svorv_u32(svptrue_b32(), svtemp1);
    return ((cf0 | cf1) | (zf0 | zf1)) == 0 ? 0 : 1;
  #else
    int32_t rc = 0, rz = 0;
    easysimd__m256i_private
      a_ = easysimd__m256i_to_private(a),
      b_ = easysimd__m256i_to_private(b);

    for (size_t i = 0 ; i < (sizeof(a_.i32f) / sizeof(a_.i32f[0])) ; i++) {
      rc |= ~a_.i32f[i] & b_.i32f[i];
      rz |=  a_.i32f[i] & b_.i32f[i];
    }

    return !!(rc & rz);
  #endif
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_testnzc_si256
  #define _mm256_testnzc_si256(a, b) easysimd_mm256_testnzc_si256(a, b)
#endif

/*

*/
#define rotate_right(x, y) (((((x) & 0xFFFFFFFF) >> ((y) & 31)) | ((x) << (32 - ((y) & 31)))) & 0xFFFFFFFF)
#define shift_right(x, y)  (((x) & 0xFFFFFFFF) >> (y))
#define sigma0(x)          (rotate_right((x),  7) ^ rotate_right((x), 18) ^ shift_right((x),  3))
#define sigma1(x)          (rotate_right((x), 17) ^ rotate_right((x), 19) ^ shift_right((x), 10))

#define Sigma_0(x)         (rotate_right((x),  2) ^ rotate_right((x), 13) ^ rotate_right((x), 22))
#define Sigma_1(x)         (rotate_right((x),  6) ^ rotate_right((x), 11) ^ rotate_right((x), 25))

#define Ch(x, y, z)        (((x) & (y)) ^ ((~(x)) & (z)))
#define Maj(x, y, z)       (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))



EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sha256msg1_epu32(easysimd__m128i a, easysimd__m128i b)
{
#if defined(EASYSIMD_X86_SHA_NATIVE)
  return _mm_sha256msg1_epu32(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE) && defined(__ARM_FEATURE_SVE2_SHA3)
  __asm__ __volatile__(
    "sha256su0 %[dst].4S, %[src].4S  \n\t"
    : [dst] "+w" (a)
    : [src] "w" (b)
  );
  return a;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b),
    r_;

    uint32_t w4 = b_.u32[0];
    uint32_t w3 = a_.u32[3];
    uint32_t w2 = a_.u32[2];
    uint32_t w1 = a_.u32[1];
    uint32_t w0 = a_.u32[0];

    r_.u32[3] = w3 + sigma0(w4);
    r_.u32[2] = w2 + sigma0(w3);
    r_.u32[1] = w1 + sigma0(w2);
    r_.u32[0] = w0 + sigma0(w1);

    return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SHA_ENABLE_NATIVE_ALIASES)
  #undef _mm_sha256msg1_epu32
  #define _mm_sha256msg1_epu32(a) easysimd_mm_sha256msg1_epu32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sha256msg2_epu32(easysimd__m128i a, easysimd__m128i b)
{
#if defined(EASYSIMD_X86_SHA_NATIVE)
  return _mm_sha256msg2_epu32(a, b);
#elif defined(EASYSIMD_ARM_SVE_NATIVE) && defined(__ARM_FEATURE_SVE2_SHA3)
  easysimd__m128i zero = easysimd_mm_setzero_si128();
  b.u32[0] = 0;
  b.u32[1] = 0;
  __asm__ __volatile__(
    "sha256su1 %[dst].4S, %[src1].4S, %[src2].4S  \n\t"
    : [dst] "+w" (a)
    : [src1] "w" (zero), [src2] "w" (b)
  );

  return a;
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b),
    r_;

    uint32_t w14 = b_.u32[2];
    uint32_t w15 = b_.u32[3];

    uint32_t w16 = a_.u32[0] + sigma1(w14);
    uint32_t w17 = a_.u32[1] + sigma1(w15);
    uint32_t w18 = a_.u32[2] + sigma1(w16);
    uint32_t w19 = a_.u32[3] + sigma1(w17);

    r_.u32[3] = w19;
    r_.u32[2] = w18;
    r_.u32[1] = w17;
    r_.u32[0] = w16;

    return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SHA_ENABLE_NATIVE_ALIASES)
  #undef _mm_sha256msg2_epu32
  #define _mm_sha256msg2_epu32(a) easysimd_mm_sha256msg2_epu32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_sha256rnds2_epu32(easysimd__m128i a, easysimd__m128i b, easysimd__m128i k)
{
#if defined(EASYSIMD_X86_SHA_NATIVE)
  return _mm_sha256rnds2_epu32(a, b, k);
#else
  easysimd__m128i_private
    a_ = easysimd__m128i_to_private(a),
    b_ = easysimd__m128i_to_private(b),
    k_ = easysimd__m128i_to_private(k),
    r_;

    uint32_t A[3], B[3], C[3], D[3], E[3], F[3], G[3], H[3], K[3];

    A[0] = b_.u32[3];
    B[0] = b_.u32[2];
    C[0] = a_.u32[3];
    D[0] = a_.u32[2];
    E[0] = b_.u32[1];
    F[0] = b_.u32[0];
    G[0] = a_.u32[1];
    H[0] = a_.u32[0];
    K[0] = k_.u32[0];
    K[1] = k_.u32[1];

    for(int i = 0; i < 2; i++){
      uint32_t T0 = Ch(E[i], F[i], G[i]);
      uint32_t T1 = Sigma_1(E[i]) + K[i] + H[i];
      uint32_t T2 = Maj(A[i], B[i], C[i]);

      A[i + 1] = T0 + T1 + T2 + Sigma_0(A[i]);
      B[i + 1] = A[i];
      C[i + 1] = B[i];
      D[i + 1] = C[i];
      E[i + 1] = T0 + T1 + D[i];
      F[i + 1] = E[i];
      G[i + 1] = F[i];
      H[i + 1] = G[i];
    }

    r_.u32[0] = F[2];
    r_.u32[1] = E[2];
    r_.u32[2] = B[2];
    r_.u32[3] = A[2];

    return easysimd__m128i_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SHA_ENABLE_NATIVE_ALIASES)
  #undef _mm_sha256msg2_epu32
  #define _mm_sha256rnds2_epu32(a) easysimd_mm_sha256rnds2_epu32(a)
#endif

#if defined(EASYSIMD_X86_AES_NATIVE) || (defined(EASYSIMD_ARM_SVE_NATIVE) && defined(__ARM_FEATURE_SVE2_AES))
EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128i
easysimd_mm_aesimc_si128(easysimd__m128i a)
{
#if defined(EASYSIMD_X86_AES_NATIVE)
  return _mm_aesimc_si128(a);
#elif defined(EASYSIMD_ARM_SVE_NATIVE) && defined(__ARM_FEATURE_SVE2_AES)
  easysimd__m128i res;
  res.sve_u8 = svaesimc_u8(a.sve_u8);
  return res;
#else
  return easysimd_mm_setzero_si128();
#endif
}
#if defined(EASYSIMD_X86_AES_ENABLE_NATIVE_ALIASES)
  #undef _mm_aesimc_si128
  #define _mm_aesimc_si128(a) easysimd_mm_aesimc_si128(a)
#endif
#endif

void easysimd_mm256_zeroupper(void)
{
  return;
}
#if defined(EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES)
  #undef _mm256_zeroupper
  #define _mm256_zeroupper() easysimd_mm256_zeroupper()
#endif

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX_H) */
