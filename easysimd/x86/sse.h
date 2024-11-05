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
 */

#if !defined(EASYSIMD_X86_SSE_H)
#define EASYSIMD_X86_SSE_H

#include "mmx.h"

#if defined(_WIN32) && !defined(EASYSIMD_X86_SSE_NATIVE) && defined(_MSC_VER)
  #include <windows.h>
#endif

#if defined(__ARM_ACLE)
  #include <arm_acle.h>
#endif

#if defined(EASYSIMD_ARM_SVE_NATIVE)
#include "../arm/sve.h"
#endif
#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif

#if defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
extern int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

typedef union {
#if defined(EASYSIMD_VECTOR_SUBSCRIPT)
    EASYSIMD_ALIGN_TO_16 int8_t          i8 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 int16_t        i16 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 int32_t        i32 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 int64_t        i64 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 uint8_t         u8 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 uint16_t       u16 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 uint32_t       u32 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 uint64_t       u64 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    #if defined(EASYSIMD_HAVE_INT128_)
    EASYSIMD_ALIGN_TO_16 easysimd_int128  i128 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 easysimd_uint128 u128 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    #endif
    EASYSIMD_ALIGN_TO_16 easysimd_float32  f32 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 easysimd_float64  f64 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;

    EASYSIMD_ALIGN_TO_16 int_fast32_t  i32f EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
    EASYSIMD_ALIGN_TO_16 uint_fast32_t u32f EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
  #else
    EASYSIMD_ALIGN_TO_16 int8_t         i8[16];
    EASYSIMD_ALIGN_TO_16 int16_t        i16[8];
    EASYSIMD_ALIGN_TO_16 int32_t        i32[4];
    EASYSIMD_ALIGN_TO_16 int64_t        i64[2];
    EASYSIMD_ALIGN_TO_16 uint8_t        u8[16];
    EASYSIMD_ALIGN_TO_16 uint16_t       u16[8];
    EASYSIMD_ALIGN_TO_16 uint32_t       u32[4];
    EASYSIMD_ALIGN_TO_16 uint64_t       u64[2];
    #if defined(EASYSIMD_HAVE_INT128_)
    EASYSIMD_ALIGN_TO_16 easysimd_int128  i128[1];
    EASYSIMD_ALIGN_TO_16 easysimd_uint128 u128[1];
    #endif
    EASYSIMD_ALIGN_TO_16 easysimd_float32  f32[4];
    EASYSIMD_ALIGN_TO_16 easysimd_float64  f64[2];

    EASYSIMD_ALIGN_TO_16 int_fast32_t  i32f[16 / sizeof(int_fast32_t)];
    EASYSIMD_ALIGN_TO_16 uint_fast32_t u32f[16 / sizeof(uint_fast32_t)];
  #endif

    EASYSIMD_ALIGN_TO_16 easysimd__m64_private m64_private[2];
    EASYSIMD_ALIGN_TO_16 easysimd__m64         m64[2];

  #if defined(EASYSIMD_X86_AVX512BF16_NATIVE)
    EASYSIMD_ALIGN_TO_16 __m128bh         nbh;
  #endif

  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint8_t    sve_i8;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint16_t   sve_i16;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint32_t   sve_i32;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveint64_t   sve_i64;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint8_t   sve_u8;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint16_t  sve_u16;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint32_t  sve_u32;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) sveuint64_t  sve_u64;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat8_t  sve_f8;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat16_t sve_f16;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat32_t sve_f32;
    EASYSIMD_ALIGN_TO(EASYSIMD_ALIGN_16_) svefloat64_t sve_f64;
  #endif

  #if defined(EASYSIMD_X86_SSE_NATIVE)
    EASYSIMD_ALIGN_TO_16 __m128d        nd;
    EASYSIMD_ALIGN_TO_16 __m128i        ni;
    EASYSIMD_ALIGN_TO_16 __m128         n;
  #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    EASYSIMD_ALIGN_TO_16 int8x16_t      neon_i8;
    EASYSIMD_ALIGN_TO_16 int16x8_t      neon_i16;
    EASYSIMD_ALIGN_TO_16 int32x4_t      neon_i32;
    EASYSIMD_ALIGN_TO_16 int64x2_t      neon_i64;
    EASYSIMD_ALIGN_TO_16 uint8x16_t     neon_u8;
    EASYSIMD_ALIGN_TO_16 uint16x8_t     neon_u16;
    EASYSIMD_ALIGN_TO_16 uint32x4_t     neon_u32;
    EASYSIMD_ALIGN_TO_16 uint64x2_t     neon_u64;
    #if defined(__ARM_FP16_FORMAT_IEEE)
    EASYSIMD_ALIGN_TO_16 float16x8_t    neon_f16;
    #endif
    EASYSIMD_ALIGN_TO_16 float32x4_t    neon_f32;
    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      EASYSIMD_ALIGN_TO_16 float64x2_t    neon_f64;
    #endif
  #endif
} easysimd__m128_private;

#if defined(EASYSIMD_X86_SSE_NATIVE)
  typedef __m128 easysimd__m128;
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #if defined(EASYSIMD_CONVERT_TO_PRIVATE)
    typedef float32x4_t easysimd__m128;
  #else
    typedef easysimd__m128_private easysimd__m128;
  #endif

#elif defined(EASYSIMD_VECTOR_SUBSCRIPT)
#if defined(EASYSIMD_CONVERT_TO_PRIVATE)
  typedef easysimd_float32 easysimd__m128 EASYSIMD_ALIGN_TO_16 EASYSIMD_VECTOR(16) EASYSIMD_MAY_ALIAS;
#else
  typedef easysimd__m128_private easysimd__m128;
#endif
#else
  typedef easysimd__m128_private easysimd__m128;
#endif

#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  typedef easysimd__m128 __m128;
#endif

HEDLEY_STATIC_ASSERT(16 == sizeof(easysimd__m128), "easysimd__m128 size incorrect");
HEDLEY_STATIC_ASSERT(16 == sizeof(easysimd__m128_private), "easysimd__m128_private size incorrect");
#if defined(EASYSIMD_CHECK_ALIGNMENT) && defined(EASYSIMD_ALIGN_OF)
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m128) == 16, "easysimd__m128 is not 16-byte aligned");
HEDLEY_STATIC_ASSERT(EASYSIMD_ALIGN_OF(easysimd__m128_private) == 16, "easysimd__m128_private is not 16-byte aligned");
#endif

#if defined(EASYSIMD_CONVERT_TO_PRIVATE) || defined(EASYSIMD_X86_SSE_NATIVE)
EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd__m128_from_private(easysimd__m128_private v) {
  easysimd__m128 r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128_private
easysimd__m128_to_private(easysimd__m128 v) {
  easysimd__m128_private r;
  easysimd_memcpy(&r, &v, sizeof(r));
  return r;
}

#else 
#define easysimd__m128_from_private(v) v
#define easysimd__m128_to_private(v) v
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, int8x16_t, neon, i8)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, int16x8_t, neon, i16)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, int32x4_t, neon, i32)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, int64x2_t, neon, i64)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, uint8x16_t, neon, u8)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, uint16x8_t, neon, u16)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, uint32x4_t, neon, u32)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, uint64x2_t, neon, u64)
  EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, float32x4_t, neon, f32)
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    EASYSIMD_X86_GENERATE_CONVERSION_FUNCTION(m128, float64x2_t, neon, f64)
  #endif
#endif /* defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) */

enum {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    EASYSIMD_MM_ROUND_NEAREST     = _MM_ROUND_NEAREST,
    EASYSIMD_MM_ROUND_DOWN        = _MM_ROUND_DOWN,
    EASYSIMD_MM_ROUND_UP          = _MM_ROUND_UP,
    EASYSIMD_MM_ROUND_TOWARD_ZERO = _MM_ROUND_TOWARD_ZERO
  #else
    EASYSIMD_MM_ROUND_NEAREST     = 0x0000,
    EASYSIMD_MM_ROUND_DOWN        = 0x2000,
    EASYSIMD_MM_ROUND_UP          = 0x4000,
    EASYSIMD_MM_ROUND_TOWARD_ZERO = 0x6000
  #endif
};

#if defined(_MM_FROUND_TO_NEAREST_INT)
#  define EASYSIMD_MM_FROUND_TO_NEAREST_INT _MM_FROUND_TO_NEAREST_INT
#  define EASYSIMD_MM_FROUND_TO_NEG_INF     _MM_FROUND_TO_NEG_INF
#  define EASYSIMD_MM_FROUND_TO_POS_INF     _MM_FROUND_TO_POS_INF
#  define EASYSIMD_MM_FROUND_TO_ZERO        _MM_FROUND_TO_ZERO
#  define EASYSIMD_MM_FROUND_CUR_DIRECTION  _MM_FROUND_CUR_DIRECTION

#  define EASYSIMD_MM_FROUND_RAISE_EXC      _MM_FROUND_RAISE_EXC
#  define EASYSIMD_MM_FROUND_NO_EXC         _MM_FROUND_NO_EXC
#else
#  define EASYSIMD_MM_FROUND_TO_NEAREST_INT 0x00
#  define EASYSIMD_MM_FROUND_TO_NEG_INF     0x01
#  define EASYSIMD_MM_FROUND_TO_POS_INF     0x02
#  define EASYSIMD_MM_FROUND_TO_ZERO        0x03
#  define EASYSIMD_MM_FROUND_CUR_DIRECTION  0x04

#  define EASYSIMD_MM_FROUND_RAISE_EXC      0x00
#  define EASYSIMD_MM_FROUND_NO_EXC         0x08
#endif

#define EASYSIMD_MM_FROUND_NINT \
  (EASYSIMD_MM_FROUND_TO_NEAREST_INT | EASYSIMD_MM_FROUND_RAISE_EXC)
#define EASYSIMD_MM_FROUND_FLOOR \
  (EASYSIMD_MM_FROUND_TO_NEG_INF | EASYSIMD_MM_FROUND_RAISE_EXC)
#define EASYSIMD_MM_FROUND_CEIL \
  (EASYSIMD_MM_FROUND_TO_POS_INF | EASYSIMD_MM_FROUND_RAISE_EXC)
#define EASYSIMD_MM_FROUND_TRUNC \
  (EASYSIMD_MM_FROUND_TO_ZERO | EASYSIMD_MM_FROUND_RAISE_EXC)
#define EASYSIMD_MM_FROUND_RINT \
  (EASYSIMD_MM_FROUND_CUR_DIRECTION | EASYSIMD_MM_FROUND_RAISE_EXC)
#define EASYSIMD_MM_FROUND_NEARBYINT \
  (EASYSIMD_MM_FROUND_CUR_DIRECTION | EASYSIMD_MM_FROUND_NO_EXC)

#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES) && !defined(_MM_FROUND_TO_NEAREST_INT)
#  define _MM_FROUND_TO_NEAREST_INT EASYSIMD_MM_FROUND_TO_NEAREST_INT
#  define _MM_FROUND_TO_NEG_INF EASYSIMD_MM_FROUND_TO_NEG_INF
#  define _MM_FROUND_TO_POS_INF EASYSIMD_MM_FROUND_TO_POS_INF
#  define _MM_FROUND_TO_ZERO EASYSIMD_MM_FROUND_TO_ZERO
#  define _MM_FROUND_CUR_DIRECTION EASYSIMD_MM_FROUND_CUR_DIRECTION
#  define _MM_FROUND_RAISE_EXC EASYSIMD_MM_FROUND_RAISE_EXC
#  define _MM_FROUND_NINT EASYSIMD_MM_FROUND_NINT
#  define _MM_FROUND_FLOOR EASYSIMD_MM_FROUND_FLOOR
#  define _MM_FROUND_CEIL EASYSIMD_MM_FROUND_CEIL
#  define _MM_FROUND_TRUNC EASYSIMD_MM_FROUND_TRUNC
#  define _MM_FROUND_RINT EASYSIMD_MM_FROUND_RINT
#  define _MM_FROUND_NEARBYINT EASYSIMD_MM_FROUND_NEARBYINT
#endif

#if defined(_MM_EXCEPT_INVALID)
#  define EASYSIMD_MM_EXCEPT_INVALID _MM_EXCEPT_INVALID
#else
#  define EASYSIMD_MM_EXCEPT_INVALID (0x0001)
#endif
#if defined(_MM_EXCEPT_DENORM)
#  define EASYSIMD_MM_EXCEPT_DENORM _MM_EXCEPT_DENORM
#else
#  define EASYSIMD_MM_EXCEPT_DENORM (0x0002)
#endif
#if defined(_MM_EXCEPT_DIV_ZERO)
#  define EASYSIMD_MM_EXCEPT_DIV_ZERO _MM_EXCEPT_DIV_ZERO
#else
#  define EASYSIMD_MM_EXCEPT_DIV_ZERO (0x0004)
#endif
#if defined(_MM_EXCEPT_OVERFLOW)
#  define EASYSIMD_MM_EXCEPT_OVERFLOW _MM_EXCEPT_OVERFLOW
#else
#  define EASYSIMD_MM_EXCEPT_OVERFLOW (0x0008)
#endif
#if defined(_MM_EXCEPT_UNDERFLOW)
#  define EASYSIMD_MM_EXCEPT_UNDERFLOW _MM_EXCEPT_UNDERFLOW
#else
#  define EASYSIMD_MM_EXCEPT_UNDERFLOW (0x0010)
#endif
#if defined(_MM_EXCEPT_INEXACT)
#  define EASYSIMD_MM_EXCEPT_INEXACT _MM_EXCEPT_INEXACT
#else
#  define EASYSIMD_MM_EXCEPT_INEXACT (0x0020)
#endif
#if defined(_MM_EXCEPT_MASK)
#  define EASYSIMD_MM_EXCEPT_MASK _MM_EXCEPT_MASK
#else
#  define EASYSIMD_MM_EXCEPT_MASK \
     (EASYSIMD_MM_EXCEPT_INVALID | EASYSIMD_MM_EXCEPT_DENORM | \
      EASYSIMD_MM_EXCEPT_DIV_ZERO | EASYSIMD_MM_EXCEPT_OVERFLOW | \
      EASYSIMD_MM_EXCEPT_UNDERFLOW | EASYSIMD_MM_EXCEPT_INEXACT)
#endif
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _MM_EXCEPT_INVALID EASYSIMD_MM_EXCEPT_INVALID
  #define _MM_EXCEPT_DENORM EASYSIMD_MM_EXCEPT_DENORM
  #define _MM_EXCEPT_DIV_ZERO EASYSIMD_MM_EXCEPT_DIV_ZERO
  #define _MM_EXCEPT_OVERFLOW EASYSIMD_MM_EXCEPT_OVERFLOW
  #define _MM_EXCEPT_UNDERFLOW EASYSIMD_MM_EXCEPT_UNDERFLOW
  #define _MM_EXCEPT_INEXACT EASYSIMD_MM_EXCEPT_INEXACT
  #define _MM_EXCEPT_MASK EASYSIMD_MM_EXCEPT_MASK
#endif

#if defined(_MM_MASK_INVALID)
#  define EASYSIMD_MM_MASK_INVALID _MM_MASK_INVALID
#else
#  define EASYSIMD_MM_MASK_INVALID (0x0080)
#endif
#if defined(_MM_MASK_DENORM)
#  define EASYSIMD_MM_MASK_DENORM _MM_MASK_DENORM
#else
#  define EASYSIMD_MM_MASK_DENORM (0x0100)
#endif
#if defined(_MM_MASK_DIV_ZERO)
#  define EASYSIMD_MM_MASK_DIV_ZERO _MM_MASK_DIV_ZERO
#else
#  define EASYSIMD_MM_MASK_DIV_ZERO (0x0200)
#endif
#if defined(_MM_MASK_OVERFLOW)
#  define EASYSIMD_MM_MASK_OVERFLOW _MM_MASK_OVERFLOW
#else
#  define EASYSIMD_MM_MASK_OVERFLOW (0x0400)
#endif
#if defined(_MM_MASK_UNDERFLOW)
#  define EASYSIMD_MM_MASK_UNDERFLOW _MM_MASK_UNDERFLOW
#else
#  define EASYSIMD_MM_MASK_UNDERFLOW (0x0800)
#endif
#if defined(_MM_MASK_INEXACT)
#  define EASYSIMD_MM_MASK_INEXACT _MM_MASK_INEXACT
#else
#  define EASYSIMD_MM_MASK_INEXACT (0x1000)
#endif
#if defined(_MM_MASK_MASK)
#  define EASYSIMD_MM_MASK_MASK _MM_MASK_MASK
#else
#  define EASYSIMD_MM_MASK_MASK \
     (EASYSIMD_MM_MASK_INVALID | EASYSIMD_MM_MASK_DENORM | \
      EASYSIMD_MM_MASK_DIV_ZERO | EASYSIMD_MM_MASK_OVERFLOW | \
      EASYSIMD_MM_MASK_UNDERFLOW | EASYSIMD_MM_MASK_INEXACT)
#endif
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _MM_MASK_INVALID EASYSIMD_MM_MASK_INVALID
  #define _MM_MASK_DENORM EASYSIMD_MM_MASK_DENORM
  #define _MM_MASK_DIV_ZERO EASYSIMD_MM_MASK_DIV_ZERO
  #define _MM_MASK_OVERFLOW EASYSIMD_MM_MASK_OVERFLOW
  #define _MM_MASK_UNDERFLOW EASYSIMD_MM_MASK_UNDERFLOW
  #define _MM_MASK_INEXACT EASYSIMD_MM_MASK_INEXACT
  #define _MM_MASK_MASK EASYSIMD_MM_MASK_MASK
#endif

#if defined(_MM_FLUSH_ZERO_MASK)
#  define EASYSIMD_MM_FLUSH_ZERO_MASK _MM_FLUSH_ZERO_MASK
#else
#  define EASYSIMD_MM_FLUSH_ZERO_MASK (0x8000)
#endif
#if defined(_MM_FLUSH_ZERO_ON)
#  define EASYSIMD_MM_FLUSH_ZERO_ON _MM_FLUSH_ZERO_ON
#else
#  define EASYSIMD_MM_FLUSH_ZERO_ON (0x8000)
#endif
#if defined(_MM_FLUSH_ZERO_OFF)
#  define EASYSIMD_MM_FLUSH_ZERO_OFF _MM_FLUSH_ZERO_OFF
#else
#  define EASYSIMD_MM_FLUSH_ZERO_OFF (0x0000)
#endif
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _MM_FLUSH_ZERO_MASK EASYSIMD_MM_FLUSH_ZERO_MASK
  #define _MM_FLUSH_ZERO_ON EASYSIMD_MM_FLUSH_ZERO_ON
  #define _MM_FLUSH_ZERO_OFF EASYSIMD_MM_FLUSH_ZERO_OFF
#endif

#if defined(EASYSIMD_ARM_SVE_NATIVE)
/*f16:*/
EASYSIMD_FUNCTION_ATTRIBUTES
svefloat16_t easysimd_fun_list_round_f16(easysimd_svbool_t pg, svefloat16_t src, int rounding)
{
  svefloat16_t svr;
  switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC)
  {
  case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
    svr = svrintn_f16_z(pg, src);
    break;
  case EASYSIMD_MM_FROUND_TO_NEG_INF:
    svr = svrintm_f16_z(pg, src);
    break;
  case EASYSIMD_MM_FROUND_TO_POS_INF:
    svr = svrintp_f16_z(pg, src);
    break;
  case EASYSIMD_MM_FROUND_TO_ZERO:
    svr = svrintz_f16_z(pg, src);
    break;
  case EASYSIMD_MM_FROUND_CUR_DIRECTION:
    svr = svrinti_f16_z(pg, src);
    break;
  default:
    svr = svdup_n_f16(0.0);
    break;
  }

  return svr;
}

/*f32:*/
EASYSIMD_FUNCTION_ATTRIBUTES svefloat32_t easysimd_nearest_int_f32(easysimd_svbool_t pg, svefloat32_t svr)
{
  return svrintn_f32_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat32_t easysimd_neg_inf_f32(easysimd_svbool_t pg, svefloat32_t svr)
{
  return svrintm_f32_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat32_t easysimd_pos_inf_f32(easysimd_svbool_t pg, svefloat32_t svr)
{
  return svrintp_f32_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat32_t easysimd_zero_f32(easysimd_svbool_t pg, svefloat32_t svr)
{
  return svrintz_f32_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat32_t easysimd_cur_direction_f32(easysimd_svbool_t pg, svefloat32_t svr)
{
  return svrinti_f32_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat32_t easysimd_set_zero_f32(easysimd_svbool_t pg, svefloat32_t svr)
{
  HEDLEY_UNUSED(pg);
  HEDLEY_UNUSED(svr);
  return svdup_n_f32(0.0);
}

typedef struct {
  int OP;
  svefloat32_t (*roundfun_f32)(easysimd_svbool_t, svefloat32_t);
} EasysimdFunListRoundF32;

__attribute__((unused)) static EasysimdFunListRoundF32 easysimdfunlistroundf32[] = {
  {EASYSIMD_MM_FROUND_TO_NEAREST_INT, easysimd_nearest_int_f32},
  {EASYSIMD_MM_FROUND_TO_NEG_INF, easysimd_neg_inf_f32},
  {EASYSIMD_MM_FROUND_TO_POS_INF, easysimd_pos_inf_f32},
  {EASYSIMD_MM_FROUND_TO_ZERO, easysimd_zero_f32},
  {EASYSIMD_MM_FROUND_CUR_DIRECTION, easysimd_cur_direction_f32},
  {5, easysimd_set_zero_f32},
  {6, easysimd_set_zero_f32},
  {7, easysimd_set_zero_f32},
  };

/*f64:*/
EASYSIMD_FUNCTION_ATTRIBUTES svefloat64_t easysimd_nearest_int_f64(easysimd_svbool_t pg, svefloat64_t svr)
{
  return svrintn_f64_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat64_t easysimd_neg_inf_f64(easysimd_svbool_t pg, svefloat64_t svr)
{
  return svrintm_f64_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat64_t easysimd_pos_inf_f64(easysimd_svbool_t pg, svefloat64_t svr)
{
  return svrintp_f64_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat64_t easysimd_zero_f64(easysimd_svbool_t pg, svefloat64_t svr)
{
  return svrintz_f64_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat64_t easysimd_cur_direction_f64(easysimd_svbool_t pg, svefloat64_t svr)
{
  return svrinti_f64_z(pg, svr);
}

EASYSIMD_FUNCTION_ATTRIBUTES svefloat64_t easysimd_set_zero_f64(easysimd_svbool_t pg, svefloat64_t svr)
{
  HEDLEY_UNUSED(pg);
  HEDLEY_UNUSED(svr);
  return svdup_n_f64(0.0);
}

typedef struct {
  int OP;
  svefloat64_t (*roundfun_f64)(easysimd_svbool_t, svefloat64_t);
} EasysimdFunListRoundF64;

__attribute__((unused)) static EasysimdFunListRoundF64 easysimdfunlistroundf64[] = {
  {EASYSIMD_MM_FROUND_TO_NEAREST_INT, easysimd_nearest_int_f64},
  {EASYSIMD_MM_FROUND_TO_NEG_INF, easysimd_neg_inf_f64},
  {EASYSIMD_MM_FROUND_TO_POS_INF, easysimd_pos_inf_f64},
  {EASYSIMD_MM_FROUND_TO_ZERO, easysimd_zero_f64},
  {EASYSIMD_MM_FROUND_CUR_DIRECTION, easysimd_cur_direction_f64},
  {5, easysimd_set_zero_f64},
  {6, easysimd_set_zero_f64},
  {7, easysimd_set_zero_f64},
  };
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
unsigned int
EASYSIMD_MM_GET_ROUNDING_MODE(void) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _MM_GET_ROUNDING_MODE();
  #elif defined(EASYSIMD_HAVE_FENV_H)
    unsigned int vfe_mode;

    switch (fegetround()) {
      #if defined(FE_TONEAREST)
        case FE_TONEAREST:
          vfe_mode = EASYSIMD_MM_ROUND_NEAREST;
          break;
      #endif

      #if defined(FE_TOWARDZERO)
        case FE_TOWARDZERO:
          vfe_mode = EASYSIMD_MM_ROUND_DOWN;
          break;
      #endif

      #if defined(FE_UPWARD)
        case FE_UPWARD:
          vfe_mode = EASYSIMD_MM_ROUND_UP;
          break;
      #endif

      #if defined(FE_DOWNWARD)
        case FE_DOWNWARD:
          vfe_mode = EASYSIMD_MM_ROUND_TOWARD_ZERO;
          break;
      #endif

      default:
        vfe_mode = EASYSIMD_MM_ROUND_NEAREST;
        break;
    }

    return vfe_mode;
  #else
    return EASYSIMD_MM_ROUND_NEAREST;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _MM_GET_ROUNDING_MODE() EASYSIMD_MM_GET_ROUNDING_MODE()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
EASYSIMD_MM_SET_ROUNDING_MODE(unsigned int a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _MM_SET_ROUNDING_MODE(a);
  #elif defined(EASYSIMD_HAVE_FENV_H)
    int fe_mode = FE_TONEAREST;

    switch (a) {
      #if defined(FE_TONEAREST)
        case EASYSIMD_MM_ROUND_NEAREST:
          fe_mode = FE_TONEAREST;
          break;
      #endif

      #if defined(FE_TOWARDZERO)
        case EASYSIMD_MM_ROUND_TOWARD_ZERO:
          fe_mode = FE_TOWARDZERO;
          break;
      #endif

      #if defined(FE_DOWNWARD)
        case EASYSIMD_MM_ROUND_DOWN:
          fe_mode = FE_DOWNWARD;
          break;
      #endif

      #if defined(FE_UPWARD)
        case EASYSIMD_MM_ROUND_UP:
          fe_mode = FE_UPWARD;
          break;
      #endif

      default:
        return;
    }

    fesetround(fe_mode);
  #else
    (void) a;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _MM_SET_ROUNDING_MODE(a) EASYSIMD_MM_SET_ROUNDING_MODE(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
EASYSIMD_MM_GET_FLUSH_ZERO_MODE (void) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_getcsr() & _MM_FLUSH_ZERO_MASK;
  #else
    return EASYSIMD_MM_FLUSH_ZERO_OFF;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _MM_SET_FLUSH_ZERO_MODE(a) EASYSIMD_MM_SET_FLUSH_ZERO_MODE(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
EASYSIMD_MM_SET_FLUSH_ZERO_MODE (uint32_t a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _MM_SET_FLUSH_ZERO_MODE(a);
  #else
    (void) a;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _MM_SET_FLUSH_ZERO_MODE(a) EASYSIMD_MM_SET_FLUSH_ZERO_MODE(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint32_t
easysimd_mm_getcsr (void) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_getcsr();
  #else
    return EASYSIMD_MM_GET_ROUNDING_MODE();
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _mm_getcsr() easysimd_mm_getcsr()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_setcsr (uint32_t a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_setcsr(a);
  #else
    EASYSIMD_MM_SET_ROUNDING_MODE(HEDLEY_STATIC_CAST(unsigned int, a));
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _mm_setcsr(a) easysimd_mm_setcsr(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_round_ps (easysimd__m128 a, int rounding, int lax_rounding)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(rounding, 0, 15)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(lax_rounding, 0, 1) {
  (void) lax_rounding;
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = easysimdfunlistroundf32[(rounding & ~EASYSIMD_MM_FROUND_NO_EXC)].roundfun_f32(svptrue_b32(), a.sve_f32);
    return r;
  #else
  easysimd__m128_private
    r_,
    a_ = easysimd__m128_to_private(a);
  /* For architectures which lack a current direction SIMD instruction.
   *
   * Note that NEON actually has a current rounding mode instruction,
   * but in ARMv8+ the rounding mode is ignored and nearest is always
   * used, so we treat ARMv7 as having a rounding mode but ARMv8 as
   * not. */
  #if defined(EASYSIMD_ARM_NEON_A32V8)
    if ((rounding & 7) == EASYSIMD_MM_FROUND_CUR_DIRECTION)
      rounding = HEDLEY_STATIC_CAST(int, EASYSIMD_MM_GET_ROUNDING_MODE()) << 13;
  #endif

  switch (rounding & ~EASYSIMD_MM_FROUND_NO_EXC) {
    case EASYSIMD_MM_FROUND_CUR_DIRECTION:
      #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && !defined(EASYSIMD_BUG_GCC_95399)
        r_.neon_f32 = vrndiq_f32(a_.neon_f32);
      #elif defined(easysimd_math_nearbyintf)
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_nearbyintf(a_.f32[i]);
        }
      #else
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
      #endif
      break;

    case EASYSIMD_MM_FROUND_TO_NEAREST_INT:
      #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
        r_.neon_f32 = vrndnq_f32(a_.neon_f32);
      #elif defined(easysimd_math_roundevenf)
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_roundevenf(a_.f32[i]);
        }
      #else
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
      #endif
      break;

    case EASYSIMD_MM_FROUND_TO_NEG_INF:
      #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
        r_.neon_f32 = vrndmq_f32(a_.neon_f32);
      #elif defined(easysimd_math_floorf)
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_floorf(a_.f32[i]);
        }
      #else
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
      #endif
      break;

    case EASYSIMD_MM_FROUND_TO_POS_INF:
      #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
        r_.neon_f32 = vrndpq_f32(a_.neon_f32);
      #elif defined(easysimd_math_ceilf)
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_ceilf(a_.f32[i]);
        }
      #else
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
      #endif
      break;

    case EASYSIMD_MM_FROUND_TO_ZERO:
      #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
        r_.neon_f32 = vrndq_f32(a_.neon_f32);
      #elif defined(easysimd_math_truncf)
        EASYSIMD_VECTORIZE
        for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
          r_.f32[i] = easysimd_math_truncf(a_.f32[i]);
        }
      #else
        HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
      #endif
      break;

    default:
      HEDLEY_UNREACHABLE_RETURN(easysimd_mm_undefined_pd());
  }

  return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #define easysimd_mm_round_ps(a, rounding) _mm_round_ps((a), (rounding))
#else
  #define easysimd_mm_round_ps(a, rounding) easysimd_x_mm_round_ps((a), (rounding), 0)
#endif
#if defined(EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES)
  #define _mm_round_ps(a, rounding) easysimd_mm_round_ps((a), (rounding))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_set_ps (easysimd_float32 e3, easysimd_float32 e2, easysimd_float32 e1, easysimd_float32 e0) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_set_ps(e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdupq_n_f32(e0, e1, e2, e3);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128 r;
    EASYSIMD_ALIGN_TO_16 easysimd_float32 data[4] = { e0, e1, e2, e3 };
    r.neon_f32 = vld1q_f32(data);
    return r;
  #else
    easysimd__m128_private r_;

    r_.f32[0] = e0;
    r_.f32[1] = e1;
    r_.f32[2] = e2;
    r_.f32[3] = e3;

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_set_ps(e3, e2, e1, e0) easysimd_mm_set_ps(e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_set_ps1 (easysimd_float32 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_set_ps1(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdup_n_f32(a);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vdupq_n_f32(a);
    return r;
  #else
    return easysimd_mm_set_ps(a, a, a, a);
  #endif
}
#define easysimd_mm_set1_ps(a) easysimd_mm_set_ps1(a)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_set_ps1(a) easysimd_mm_set_ps1(a)
#  define _mm_set1_ps(a) easysimd_mm_set1_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_move_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_move_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f32[0] = b.f32[0];
    return a;
  #elif (defined(EASYSIMD_ARM_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
    easysimd__m128 r;
    r.neon_f32 = vsetq_lane_f32(vgetq_lane_f32(b.neon_f32, 0), a.neon_f32, 0);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, b_.f32, 4, 1, 2, 3);
    #else
      r_.f32[0] = b_.f32[0];
      r_.f32[1] = a_.f32[1];
      r_.f32[2] = a_.f32[2];
      r_.f32[3] = a_.f32[3];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_move_ss(a, b) easysimd_mm_move_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_broadcastlow_ps(easysimd__m128 a) {
  /* This function broadcasts the first element in the inpu vector to
   * all lanes.  It is used to avoid generating spurious exceptions in
   * *_ss functions since there may be garbage in the upper lanes. */

  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_shuffle_ps(a, a, 0);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f32 = vdupq_laneq_f32(a_.neon_f32, 0);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
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

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_add_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_add_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svadd_f32_z(svptrue_b32(), a.sve_f32, b.sve_f32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vaddq_f32(a.neon_f32, b.neon_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 + b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i] + b_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_add_ps(a, b) easysimd_mm_add_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_add_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_add_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svadd_f32_z(svptrue_b32(), a.sve_f32, svdupq_n_f32(b.f32[0], 0.0, 0.0, 0.0));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    float32_t b0 = vgetq_lane_f32(b.neon_f32, 0);
    float32x4_t value = vsetq_lane_f32(b0, vdupq_n_f32(0), 0);
    // the upper values in the result must be the remnants of <a>.
    r.neon_f32 = vaddq_f32(a.neon_f32, value);
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_add_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_add_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.f32[0] = a_.f32[0] + b_.f32[0];
    r_.f32[1] = a_.f32[1];
    r_.f32[2] = a_.f32[2];
    r_.f32[3] = a_.f32[3];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_add_ss(a, b) easysimd_mm_add_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_and_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_and_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_i32 = svand_s32_z(svptrue_b32(), a.sve_i32, b.sve_i32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vandq_s32(a_.neon_i32, b_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 & b_.i32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] & b_.i32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_and_ps(a, b) easysimd_mm_and_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_andnot_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_andnot_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_i32 = svbic_s32_z(svptrue_b32(), b.sve_i32, a.sve_i32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vbicq_s32(b_.neon_i32, a_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = ~a_.i32 & b_.i32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ~(a_.i32[i]) & b_.i32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_andnot_ps(a, b) easysimd_mm_andnot_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_xor_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_xor_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_u32 = sveor_u32_z(svptrue_b32(), a.sve_u32, b.sve_u32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_i32 = veorq_s32(a.neon_i32, b.neon_i32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f ^ b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i] ^ b_.u32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_xor_ps(a, b) easysimd_mm_xor_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_or_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_or_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_u32 = svorr_u32_z(svptrue_b32(), a.sve_u32, b.sve_u32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_i32 = vorrq_s32(a.neon_i32, b.neon_i32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32f = a_.i32f | b_.i32f;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i] | b_.u32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_or_ps(a, b) easysimd_mm_or_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_not_ps(easysimd__m128 a) {
  #if defined(EASYSIMD_X86_AVX512VL_NATIVE)
    __m128i ai = _mm_castps_si128(a);
    return _mm_castsi128_ps(_mm_ternarylogic_epi32(ai, ai, ai, 0x55));
  #elif defined(EASYSIMD_X86_SSE2_NATIVE)
    /* Note: we use ints instead of floats because we don't want cmpeq
     * to return false for (NaN, NaN) */
    __m128i ai = _mm_castps_si128(a);
    return _mm_castsi128_ps(_mm_andnot_si128(ai, _mm_cmpeq_epi32(ai, ai)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vmvnq_s32(a_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = ~a_.i32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = ~(a_.i32[i]);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_select_ps(easysimd__m128 a, easysimd__m128 b, easysimd__m128 mask) {
  /* This function is for when you want to blend two elements together
   * according to a mask.  It is similar to _mm_blendv_ps, except that
   * it is undefined whether the blend is based on the highest bit in
   * each lane (like blendv) or just bitwise operations.  This allows
   * us to implement the function efficiently everywhere.
   *
   * Basically, you promise that all the lanes in mask are either 0 or
   * ~0. */
  #if defined(EASYSIMD_X86_SSE4_1_NATIVE)
    return _mm_blendv_ps(a, b, mask);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b),
      mask_ = easysimd__m128_to_private(mask);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_i32 = vbslq_s32(mask_.neon_u32, b_.neon_i32, a_.neon_i32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = a_.i32 ^ ((a_.i32 ^ b_.i32) & mask_.i32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        r_.i32[i] = a_.i32[i] ^ ((a_.i32[i] ^ b_.i32[i]) & mask_.i32[i]);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_avg_pu16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_avg_pu16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_u16 = vrhadd_u16(b.neon_u16, a.neon_u16);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && defined(EASYSIMD_CONVERT_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100761)
      uint32_t wa EASYSIMD_VECTOR(16);
      uint32_t wb EASYSIMD_VECTOR(16);
      uint32_t wr EASYSIMD_VECTOR(16);
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

    return easysimd__m64_from_private(r_);
  #endif
}
#define easysimd_m_pavgw(a, b) easysimd_mm_avg_pu16(a, b)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_avg_pu16(a, b) easysimd_mm_avg_pu16(a, b)
#  define _m_pavgw(a, b) easysimd_mm_avg_pu16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_avg_pu8 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_avg_pu8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_u8 = vrhadd_u8(b.neon_u8, a.neon_u8);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS) && defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR) && defined(EASYSIMD_CONVERT_VECTOR_) && !defined(EASYSIMD_BUG_GCC_100761)
      uint16_t wa EASYSIMD_VECTOR(16);
      uint16_t wb EASYSIMD_VECTOR(16);
      uint16_t wr EASYSIMD_VECTOR(16);
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

    return easysimd__m64_from_private(r_);
  #endif
}
#define easysimd_m_pavgb(a, b) easysimd_mm_avg_pu8(a, b)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_avg_pu8(a, b) easysimd_mm_avg_pu8(a, b)
#  define _m_pavgb(a, b) easysimd_mm_avg_pu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_abs_ps(easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    easysimd_float32 mask_;
    uint32_t u32_ = UINT32_C(0x7FFFFFFF);
    easysimd_memcpy(&mask_, &u32_, sizeof(u32_));
    return _mm_and_ps(_mm_set1_ps(mask_), a);
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_f32 = vabsq_f32(a_.neon_f32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = easysimd_math_fabsf(a_.f32[i]);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpeq_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmpeq_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b32();
    r.sve_u32 = svdup_n_u32_z(svcmpeq_f32(pg, a.sve_f32, b.sve_f32), ~UINT32_C(0));
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128 res;
    res.neon_u32 = vceqq_f32(a.neon_f32, b.neon_f32);
    return res;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), a_.f32 == b_.f32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.u32[i] = (fabsf(a_.f32[i] - b_.f32[i]) < 1e-9f) ? ~UINT32_C(0) : UINT32_C(0);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpeq_ps(a, b) easysimd_mm_cmpeq_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpeq_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmpeq_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u32[0] = (fabsf(a.f32[0] - b.f32[0]) < 1e-9f) ? ~UINT32_C(0) : UINT32_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpeq_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpeq_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.u32[0] = (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f) ? ~UINT32_C(0) : UINT32_C(0);
    EASYSIMD_VECTORIZE
    for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.u32[i] = a_.u32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpeq_ss(a, b) easysimd_mm_cmpeq_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpge_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmpge_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_u32 = svdup_n_u32_z(svcmpge_f32(svptrue_b32(), a.sve_f32, b.sve_f32), ~UINT32_C(0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vcgeq_f32(a_.neon_f32, b_.neon_f32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 >= b_.f32));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.u32[i] = ((a_.f32[i] > b_.f32[i]) || (fabsf(a_.f32[i] - b_.f32[i]) < 1e-9f)) ? ~UINT32_C(0) : UINT32_C(0);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpge_ps(a, b) easysimd_mm_cmpge_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpge_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && !defined(__PGI)
    return _mm_cmpge_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u32[0] = ((a.f32[0] > b.f32[0]) || (fabsf(a.f32[0] - b.f32[0]) < 1e-9f)) ? ~UINT32_C(0) : UINT32_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpge_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpge_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.u32[0] = ((a_.f32[0] > b_.f32[0]) || (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f)) ? ~UINT32_C(0) : UINT32_C(0);
    EASYSIMD_VECTORIZE
    for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.u32[i] = a_.u32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpge_ss(a, b) easysimd_mm_cmpge_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpgt_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmpgt_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t pg = svptrue_b32();
    r.sve_u32 = svdup_n_u32_z(svcmpgt_f32(pg, a.sve_f32, b.sve_f32), ~UINT32_C(0));
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128 res;
    res.neon_u32 = vcgtq_f32(a.neon_f32, b.neon_f32);
    return res;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 > b_.f32));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.u32[i] = (a_.f32[i] > b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpgt_ps(a, b) easysimd_mm_cmpgt_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpgt_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && !defined(__PGI)
    return _mm_cmpgt_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u32[0] = (a.f32[0] > b.f32[0]) ? ~UINT32_C(0) : UINT32_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpgt_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpgt_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.u32[0] = (a_.f32[0] > b_.f32[0]) ? ~UINT32_C(0) : UINT32_C(0);
    EASYSIMD_VECTORIZE
    for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.u32[i] = a_.u32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpgt_ss(a, b) easysimd_mm_cmpgt_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmple_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmple_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_u32 = svdup_n_u32_z(svcmple_f32(svptrue_b32(), a.sve_f32, b.sve_f32), ~UINT32_C(0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vcleq_f32(a_.neon_f32, b_.neon_f32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 <= b_.f32));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.u32[i] = ((a_.f32[i] < b_.f32[i]) || (fabsf(a_.f32[i] - b_.f32[i]) < 1e-9f)) ? ~UINT32_C(0) : UINT32_C(0);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmple_ps(a, b) easysimd_mm_cmple_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmple_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmple_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u32[0] = ((a.f32[0] < b.f32[0]) || (fabsf(a.f32[0] - b.f32[0]) < 1e-9f)) ? ~UINT32_C(0) : UINT32_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_cmple_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_cmple_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.u32[0] = ((a_.f32[0] < b_.f32[0]) || (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f)) ? ~UINT32_C(0) : UINT32_C(0);
    EASYSIMD_VECTORIZE
    for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.u32[i] = a_.u32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmple_ss(a, b) easysimd_mm_cmple_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmplt_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmplt_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_u32 = svdup_n_u32_z(svcmplt_f32(svptrue_b32(), a.sve_f32, b.sve_f32), ~UINT32_C(0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vcltq_f32(a_.neon_f32, b_.neon_f32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 < b_.f32));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.u32[i] = (a_.f32[i] < b_.f32[i]) ? ~UINT32_C(0) : UINT32_C(0);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmplt_ps(a, b) easysimd_mm_cmplt_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmplt_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmplt_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u32[0] = (a.f32[0] < b.f32[0]) ? ~UINT32_C(0) : UINT32_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_cmplt_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_cmplt_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.u32[0] = (a_.f32[0] < b_.f32[0]) ? ~UINT32_C(0) : UINT32_C(0);
    EASYSIMD_VECTORIZE
    for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.u32[i] = a_.u32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmplt_ss(a, b) easysimd_mm_cmplt_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpneq_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmpneq_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_u32 = svdup_n_u32_z(svcmpne_f32(svptrue_b32(), a.sve_f32, b.sve_f32), ~UINT32_C(0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_u32 = vmvnq_u32(vceqq_f32(a_.neon_f32, b_.neon_f32));
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.i32 = HEDLEY_REINTERPRET_CAST(__typeof__(r_.i32), (a_.f32 != b_.f32));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.u32[i] = (fabsf(a_.f32[i] - b_.f32[i]) > 1e-9f) ? ~UINT32_C(0) : UINT32_C(0);
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpneq_ps(a, b) easysimd_mm_cmpneq_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpneq_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmpneq_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u32[0] = (fabsf(a.f32[0] - b.f32[0]) > 1e-9f) ? ~UINT32_C(0) : UINT32_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpneq_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpneq_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.u32[0] = (fabsf(a_.f32[0] - b_.f32[0]) > 1e-9f) ? ~UINT32_C(0) : UINT32_C(0);
    EASYSIMD_VECTORIZE
    for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.u32[i] = a_.u32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpneq_ss(a, b) easysimd_mm_cmpneq_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpnge_ps (easysimd__m128 a, easysimd__m128 b) {
  return easysimd_mm_cmplt_ps(a, b);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpnge_ps(a, b) easysimd_mm_cmpnge_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpnge_ss (easysimd__m128 a, easysimd__m128 b) {
  return easysimd_mm_cmplt_ss(a, b);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpnge_ss(a, b) easysimd_mm_cmpnge_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpngt_ps (easysimd__m128 a, easysimd__m128 b) {
  return easysimd_mm_cmple_ps(a, b);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpngt_ps(a, b) easysimd_mm_cmpngt_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpngt_ss (easysimd__m128 a, easysimd__m128 b) {
  return easysimd_mm_cmple_ss(a, b);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpngt_ss(a, b) easysimd_mm_cmpngt_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpnle_ps (easysimd__m128 a, easysimd__m128 b) {
  return easysimd_mm_cmpgt_ps(a, b);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpnle_ps(a, b) easysimd_mm_cmpnle_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpnle_ss (easysimd__m128 a, easysimd__m128 b) {
  return easysimd_mm_cmpgt_ss(a, b);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpnle_ss(a, b) easysimd_mm_cmpnle_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpnlt_ps (easysimd__m128 a, easysimd__m128 b) {
  return easysimd_mm_cmpge_ps(a, b);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpnlt_ps(a, b) easysimd_mm_cmpnlt_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpnlt_ss (easysimd__m128 a, easysimd__m128 b) {
  return easysimd_mm_cmpge_ss(a, b);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpnlt_ss(a, b) easysimd_mm_cmpnlt_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpord_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmpord_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_u32 = svdup_n_u32_z(svnot_b_z(svptrue_b32(), svcmpuo_f32(svptrue_b32(), a.sve_f32, b.sve_f32)), ~UINT32_C(0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      /* Note: NEON does not have ordered compare builtin
        Need to compare a eq a and b eq b to check for NaN
        Do AND of results to get final */
      uint32x4_t ceqaa = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t ceqbb = vceqq_f32(b_.neon_f32, b_.neon_f32);
      r_.neon_u32 = vandq_u32(ceqaa, ceqbb);
    #elif defined(easysimd_math_isnanf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.u32[i] = (easysimd_math_isnanf(a_.f32[i]) || easysimd_math_isnanf(b_.f32[i])) ? UINT32_C(0) : ~UINT32_C(0);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpord_ps(a, b) easysimd_mm_cmpord_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpunord_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmpunord_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_u32 = svdup_n_u32_z(svcmpuo_f32(svptrue_b32(), a.sve_f32, b.sve_f32), ~UINT32_C(0));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t ceqaa = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t ceqbb = vceqq_f32(b_.neon_f32, b_.neon_f32);
      r_.neon_u32 = vmvnq_u32(vandq_u32(ceqaa, ceqbb));
    #elif defined(easysimd_math_isnanf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.u32[i] = (easysimd_math_isnanf(a_.f32[i]) || easysimd_math_isnanf(b_.f32[i])) ? ~UINT32_C(0) : UINT32_C(0);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpunord_ps(a, b) easysimd_mm_cmpunord_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpunord_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && !defined(__PGI)
    return _mm_cmpunord_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u32[0] = (easysimd_math_isnanf(a.f32[0]) || easysimd_math_isnanf(b.f32[0])) ? ~UINT32_C(0) : UINT32_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpunord_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpunord_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(easysimd_math_isnanf)
      r_.u32[0] = (easysimd_math_isnanf(a_.f32[0]) || easysimd_math_isnanf(b_.f32[0])) ? ~UINT32_C(0) : UINT32_C(0);
      EASYSIMD_VECTORIZE
      for (size_t i = 1 ; i < (sizeof(r_.u32) / sizeof(r_.u32[0])) ; i++) {
        r_.u32[i] = a_.u32[i];
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpunord_ss(a, b) easysimd_mm_cmpunord_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comieq_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_comieq_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (fabsf(a.f32[0] - b.f32[0]) < 1e-9f);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_or_b_nan = vmvnq_u32(vandq_u32(a_not_nan, b_not_nan));
      uint32x4_t a_eq_b = vceqq_f32(a_.neon_f32, b_.neon_f32);
      return !!(vgetq_lane_u32(vorrq_u32(a_or_b_nan, a_eq_b), 0) != 0);
    #else
      return (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_comieq_ss(a, b) easysimd_mm_comieq_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comige_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_comige_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return ((a.f32[0] > b.f32[0]) || (fabsf(a.f32[0] - b.f32[0]) < 1e-9f));
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_and_b_not_nan = vandq_u32(a_not_nan, b_not_nan);
      uint32x4_t a_ge_b = vcgeq_f32(a_.neon_f32, b_.neon_f32);
      return !!(vgetq_lane_u32(vandq_u32(a_and_b_not_nan, a_ge_b), 0) != 0);
    #else
      return ((a_.f32[0] > b_.f32[0]) || (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f));
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_comige_ss(a, b) easysimd_mm_comige_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comigt_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_comigt_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (a.f32[0] > b.f32[0]);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_and_b_not_nan = vandq_u32(a_not_nan, b_not_nan);
      uint32x4_t a_gt_b = vcgtq_f32(a_.neon_f32, b_.neon_f32);
      return !!(vgetq_lane_u32(vandq_u32(a_and_b_not_nan, a_gt_b), 0) != 0);
    #else
      return a_.f32[0] > b_.f32[0];
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_comigt_ss(a, b) easysimd_mm_comigt_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comile_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_comile_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return ((a.f32[0] < b.f32[0]) || (fabsf(a.f32[0] - b.f32[0]) < 1e-9f));
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_or_b_nan = vmvnq_u32(vandq_u32(a_not_nan, b_not_nan));
      uint32x4_t a_le_b = vcleq_f32(a_.neon_f32, b_.neon_f32);
      return !!(vgetq_lane_u32(vorrq_u32(a_or_b_nan, a_le_b), 0) != 0);
    #else
      return ((a_.f32[0] < b_.f32[0]) || (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f));
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_comile_ss(a, b) easysimd_mm_comile_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comilt_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_comilt_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (a.f32[0] < b.f32[0]);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_or_b_nan = vmvnq_u32(vandq_u32(a_not_nan, b_not_nan));
      uint32x4_t a_lt_b = vcltq_f32(a_.neon_f32, b_.neon_f32);
      return !!(vgetq_lane_u32(vorrq_u32(a_or_b_nan, a_lt_b), 0) != 0);
    #else
      return a_.f32[0] < b_.f32[0];
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_comilt_ss(a, b) easysimd_mm_comilt_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_comineq_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_comineq_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (fabsf(a.f32[0] - b.f32[0]) > 1e-9f);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_and_b_not_nan = vandq_u32(a_not_nan, b_not_nan);
      uint32x4_t a_neq_b = vmvnq_u32(vceqq_f32(a_.neon_f32, b_.neon_f32));
      return !!(vgetq_lane_u32(vandq_u32(a_and_b_not_nan, a_neq_b), 0) != 0);
    #else
      return (fabsf(a_.f32[0] - b_.f32[0]) > 1e-9f);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_comineq_ss(a, b) easysimd_mm_comineq_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_copysign_ps(easysimd__m128 dest, easysimd__m128 src) {
  easysimd__m128_private
    r_,
    dest_ = easysimd__m128_to_private(dest),
    src_ = easysimd__m128_to_private(src);

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    const uint32x4_t sign_pos = vreinterpretq_u32_f32(vdupq_n_f32(-EASYSIMD_FLOAT32_C(0.0)));
    r_.neon_u32 = vbslq_u32(sign_pos, src_.neon_u32, dest_.neon_u32);
  #elif defined(EASYSIMD_IEEE754_STORAGE)
    (void) src_;
    (void) dest_;
    easysimd__m128 sign_pos = easysimd_mm_set1_ps(-0.0f);
    r_ = easysimd__m128_to_private(easysimd_mm_xor_ps(dest, easysimd_mm_and_ps(easysimd_mm_xor_ps(dest, src), sign_pos)));
  #else
    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = easysimd_math_copysignf(dest_.f32[i], src_.f32[i]);
    }
  #endif

  return easysimd__m128_from_private(r_);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_xorsign_ps(easysimd__m128 dest, easysimd__m128 src) {
  return easysimd_mm_xor_ps(easysimd_mm_and_ps(easysimd_mm_set1_ps(-0.0f), src), dest);
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvt_pi2ps (easysimd__m128 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvt_pi2ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcombine_f32(vcvt_f32_s32(b.neon_i32), vget_high_f32(a.neon_f32));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f32[0] = HEDLEY_STATIC_CAST(float32_t, b.i32[0]);
    a.f32[1] = HEDLEY_STATIC_CAST(float32_t, b.i32[1]);
    return a;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);
    easysimd__m64_private b_ = easysimd__m64_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.m64_private[0].f32, b_.i32);
      r_.m64_private[1] = a_.m64_private[1];
    #else
      r_.f32[0] = (easysimd_float32) b_.i32[0];
      r_.f32[1] = (easysimd_float32) b_.i32[1];
      r_.i32[2] = a_.i32[2];
      r_.i32[3] = a_.i32[3];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvt_pi2ps(a, b) easysimd_mm_cvt_pi2ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_cvt_ps2pi (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvt_ps2pi(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    a = easysimd_mm_round_ps(a, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    r.neon_i32 = vcvt_s32_f32(vget_low_f32(a.neon_f32));
    return r;
  #else
    easysimd__m64_private r_;
    easysimd__m128_private a_;

  #if defined(EASYSIMD_CONVERT_VECTOR_) && EASYSIMD_NATURAL_VECTOR_SIZE_GE(128) && !defined(EASYSIMD_BUG_GCC_100761)
    a_ = easysimd__m128_to_private(easysimd_mm_round_ps(a, EASYSIMD_MM_FROUND_CUR_DIRECTION));
    EASYSIMD_CONVERT_VECTOR_(r_.i32, a_.m64_private[0].f32);
  #else
    a_ = easysimd__m128_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
      r_.i32[i] = HEDLEY_STATIC_CAST(int32_t, easysimd_math_nearbyintf(a_.f32[i]));
    }
  #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvt_ps2pi(a) easysimd_mm_cvt_ps2pi((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvt_si2ss (easysimd__m128 a, int32_t b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cvt_si2ss(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsetq_lane_f32(HEDLEY_STATIC_CAST(float, b), a.neon_f32, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f32[0] = HEDLEY_STATIC_CAST(easysimd_float32, b);
    return a;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    r_.f32[0] = HEDLEY_STATIC_CAST(easysimd_float32, b);
    r_.i32[1] = a_.i32[1];
    r_.i32[2] = a_.i32[2];
    r_.i32[3] = a_.i32[3];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvt_si2ss(a, b) easysimd_mm_cvt_si2ss((a), b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_cvt_ss2si (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cvt_ss2si(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a = easysimd_mm_round_ps(a, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    if ((a.f32[0] > HEDLEY_STATIC_CAST(easysimd_float32, INT32_MIN)) && (a.f32[0] < HEDLEY_STATIC_CAST(easysimd_float32, INT32_MAX))) {
      return EASYSIMD_CONVERT_FTOI(int32_t, a.f32[0]);
    }
    return INT32_MIN;
  #elif defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_FAST_CONVERSION_RANGE) && !defined(EASYSIMD_BUG_GCC_95399)
    return vgetq_lane_s32(vcvtnq_s32_f32(easysimd__m128_to_neon_f32(a)), 0);
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(easysimd_mm_round_ps(a, EASYSIMD_MM_FROUND_CUR_DIRECTION));
    #if !defined(EASYSIMD_FAST_CONVERSION_RANGE)
      return ((a_.f32[0] > HEDLEY_STATIC_CAST(easysimd_float32, INT32_MIN)) &&
          (a_.f32[0] < HEDLEY_STATIC_CAST(easysimd_float32, INT32_MAX))) ?
        EASYSIMD_CONVERT_FTOI(int32_t, a_.f32[0]) : INT32_MIN;
    #else
      return EASYSIMD_CONVERT_FTOI(int32_t, a_.f32[0]);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvt_ss2si(a) easysimd_mm_cvt_ss2si((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtpi16_ps (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtpi16_ps(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcvtq_f32_s32(vmovl_s16(a.neon_i16));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m64_private a_ = easysimd__m64_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.f32, a_.i16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        easysimd_float32 v = a_.i16[i];
        r_.f32[i] = v;
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtpi16_ps(a) easysimd_mm_cvtpi16_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtpi32_ps (easysimd__m128 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtpi32_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcombine_f32(vcvt_f32_s32(b.neon_i32), vget_high_f32(a.neon_f32));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f32[0] = HEDLEY_STATIC_CAST(float32_t, b.i32[0]);
    a.f32[1] = HEDLEY_STATIC_CAST(float32_t, b.i32[1]);
    return a;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);
    easysimd__m64_private b_ = easysimd__m64_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.m64_private[0].f32, b_.i32);
      r_.m64_private[1] = a_.m64_private[1];
    #else
      r_.f32[0] = (easysimd_float32) b_.i32[0];
      r_.f32[1] = (easysimd_float32) b_.i32[1];
      r_.i32[2] = a_.i32[2];
      r_.i32[3] = a_.i32[3];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtpi32_ps(a, b) easysimd_mm_cvtpi32_ps((a), b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtpi32x2_ps (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtpi32x2_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcvtq_f32_s32(vcombine_s32(a.neon_i32, b.neon_i32));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp, r;
    tmp.sve_i32 = svdupq_n_s32(a.i32[0], a.i32[1], b.i32[0], b.i32[1]);
    r.sve_f32 = svcvt_f32_s32_z(svptrue_b32(), tmp.sve_i32);
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m64_private
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.m64_private[0].f32, a_.i32);
      EASYSIMD_CONVERT_VECTOR_(r_.m64_private[1].f32, b_.i32);
    #else
      r_.f32[0] = (easysimd_float32) a_.i32[0];
      r_.f32[1] = (easysimd_float32) a_.i32[1];
      r_.f32[2] = (easysimd_float32) b_.i32[0];
      r_.f32[3] = (easysimd_float32) b_.i32[1];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtpi32x2_ps(a, b) easysimd_mm_cvtpi32x2_ps(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtpi8_ps (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtpi8_ps(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(a.neon_i8))));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m64_private a_ = easysimd__m64_to_private(a);

    r_.f32[0] = HEDLEY_STATIC_CAST(easysimd_float32, a_.i8[0]);
    r_.f32[1] = HEDLEY_STATIC_CAST(easysimd_float32, a_.i8[1]);
    r_.f32[2] = HEDLEY_STATIC_CAST(easysimd_float32, a_.i8[2]);
    r_.f32[3] = HEDLEY_STATIC_CAST(easysimd_float32, a_.i8[3]);

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtpi8_ps(a) easysimd_mm_cvtpi8_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_cvtps_pi16 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtps_pi16(a);
  #else
    easysimd__m64_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && !defined(EASYSIMD_BUG_GCC_95399)
      r_.neon_i16 = vmovn_s32(vcvtq_s32_f32(vrndiq_f32(a_.neon_f32)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
        r_.i16[i] = EASYSIMD_CONVERT_FTOI(int16_t, easysimd_math_roundf(a_.f32[i]));
      }
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtps_pi16(a) easysimd_mm_cvtps_pi16((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_cvtps_pi32 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtps_pi32(a);
  #else
    easysimd__m64_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && defined(EASYSIMD_FAST_CONVERSION_RANGE) && !defined(EASYSIMD_BUG_GCC_95399)
      r_.neon_i32 = vcvt_s32_f32(vget_low_f32(vrndiq_f32(a_.neon_f32)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.i32) / sizeof(r_.i32[0])) ; i++) {
        easysimd_float32 v = easysimd_math_roundf(a_.f32[i]);
        #if !defined(EASYSIMD_FAST_CONVERSION_RANGE)
          r_.i32[i] = ((v > HEDLEY_STATIC_CAST(easysimd_float32, INT32_MIN)) && (v < HEDLEY_STATIC_CAST(easysimd_float32, INT32_MAX))) ?
            EASYSIMD_CONVERT_FTOI(int32_t, v) : INT32_MIN;
        #else
          r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, v);
        #endif
      }
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtps_pi32(a) easysimd_mm_cvtps_pi32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_cvtps_pi8 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtps_pi8(a);
  #else
    easysimd__m64_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && !defined(EASYSIMD_BUG_GCC_95471)
      /* Clamp the input to [INT8_MIN, INT8_MAX], round, convert to i32, narrow to
      * i16, combine with an all-zero vector of i16 (which will become the upper
      * half), narrow to i8. */
      float32x4_t max = vdupq_n_f32(HEDLEY_STATIC_CAST(easysimd_float32, INT8_MAX));
      float32x4_t min = vdupq_n_f32(HEDLEY_STATIC_CAST(easysimd_float32, INT8_MIN));
      float32x4_t values = vrndnq_f32(vmaxq_f32(vminq_f32(max, a_.neon_f32), min));
      r_.neon_i8 = vmovn_s16(vcombine_s16(vmovn_s32(vcvtq_s32_f32(values)), vdup_n_s16(0)));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(a_.f32) / sizeof(a_.f32[0])) ; i++) {
        if (a_.f32[i] > HEDLEY_STATIC_CAST(easysimd_float32, INT8_MAX))
          r_.i8[i] = INT8_MAX;
        else if (a_.f32[i] <  HEDLEY_STATIC_CAST(easysimd_float32, INT8_MIN))
          r_.i8[i] = INT8_MIN;
        else
          r_.i8[i] = EASYSIMD_CONVERT_FTOI(int8_t, easysimd_math_roundf(a_.f32[i]));
      }
      /* Note: the upper half is undefined */
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtps_pi8(a) easysimd_mm_cvtps_pi8((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtpu16_ps (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtpu16_ps(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcvtq_f32_u32(vmovl_u16(a.neon_u16));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m64_private a_ = easysimd__m64_to_private(a);

    #if defined(EASYSIMD_CONVERT_VECTOR_)
      EASYSIMD_CONVERT_VECTOR_(r_.f32, a_.u16);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = (easysimd_float32) a_.u16[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtpu16_ps(a) easysimd_mm_cvtpu16_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtpu8_ps (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtpu8_ps(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(a.neon_u8))));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd__m64_private a_ = easysimd__m64_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = HEDLEY_STATIC_CAST(easysimd_float32, a_.u8[i]);
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtpu8_ps(a) easysimd_mm_cvtpu8_ps(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtsi32_ss (easysimd__m128 a, int32_t b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cvtsi32_ss(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsetq_lane_f32(HEDLEY_STATIC_CAST(float32_t, b), a.neon_f32, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f32[0] = HEDLEY_STATIC_CAST(float32_t, b);
    return a;
  #else
    easysimd__m128_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    r_ = a_;
    r_.f32[0] = HEDLEY_STATIC_CAST(easysimd_float32, b);

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtsi32_ss(a, b) easysimd_mm_cvtsi32_ss((a), b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cvtsi64_ss (easysimd__m128 a, int64_t b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
    #if !defined(__PGI)
      return _mm_cvtsi64_ss(a, b);
    #else
      return _mm_cvtsi64x_ss(a, b);
    #endif
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsetq_lane_f32(HEDLEY_STATIC_CAST(float32_t, b), a.neon_f32, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.f32[0] = HEDLEY_STATIC_CAST(easysimd_float32, b);
    return a;
  #else
    easysimd__m128_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_f32 = vsetq_lane_f32(HEDLEY_STATIC_CAST(float32_t, b), a_.neon_f32, 0);
    #else
      r_ = a_;
      r_.f32[0] = HEDLEY_STATIC_CAST(easysimd_float32, b);
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
#  define _mm_cvtsi64_ss(a, b) easysimd_mm_cvtsi64_ss((a), b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32
easysimd_mm_cvtss_f32 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cvtss_f32(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.f32[0];
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);
    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      return vgetq_lane_f32(a_.neon_f32, 0);
    #else
      return a_.f32[0];
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtss_f32(a) easysimd_mm_cvtss_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm_cvtss_i64 (easysimd__m128 a) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_i64 = svcvt_s64_f32_z(svptrue_b64(), a.sve_f32);
    return r.i64[0];
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);
    return HEDLEY_STATIC_CAST(int64_t, a_.f32[0]);
  #endif
}
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtss_i64(a) easysimd_mm_cvtss_i64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_cvtss_si32 (easysimd__m128 a) {
  return easysimd_mm_cvt_ss2si(a);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtss_si32(a) easysimd_mm_cvtss_si32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm_cvtss_si64 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_ARCH_AMD64)
    #if !defined(__PGI)
      return _mm_cvtss_si64(a);
    #else
      return _mm_cvtss_si64x(a);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return EASYSIMD_CONVERT_FTOI(int64_t, easysimd_math_roundf(a.f32[0]));
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);
    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      return EASYSIMD_CONVERT_FTOI(int64_t, easysimd_math_roundf(vgetq_lane_f32(a_.neon_f32, 0)));
    #else
      return EASYSIMD_CONVERT_FTOI(int64_t, easysimd_math_roundf(a_.f32[0]));
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
#  define _mm_cvtss_si64(a) easysimd_mm_cvtss_si64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_cvtt_ps2pi (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_cvtt_ps2pi(a);
  #else
    easysimd__m64_private r_;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_FAST_CONVERSION_RANGE)
      r_.neon_i32 = vcvt_s32_f32(vget_low_f32(a_.neon_f32));
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        easysimd_float32 v = a_.f32[i];
        #if !defined(EASYSIMD_FAST_CONVERSION_RANGE)
          r_.i32[i] = ((v > HEDLEY_STATIC_CAST(easysimd_float32, INT32_MIN)) && (v < HEDLEY_STATIC_CAST(easysimd_float32, INT32_MAX))) ?
            EASYSIMD_CONVERT_FTOI(int32_t, v) : INT32_MIN;
        #else
          r_.i32[i] = EASYSIMD_CONVERT_FTOI(int32_t, v);
        #endif
      }
    #endif

    return easysimd__m64_from_private(r_);
  #endif
}
#define easysimd_mm_cvttps_pi32(a) easysimd_mm_cvtt_ps2pi(a)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtt_ps2pi(a) easysimd_mm_cvtt_ps2pi((a))
#  define _mm_cvttps_pi32(a) easysimd_mm_cvttps_pi32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_mm_cvtt_ss2si (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cvtt_ss2si(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    if ((a.f32[0] > HEDLEY_STATIC_CAST(easysimd_float32, INT32_MIN)) && (a.f32[0] < HEDLEY_STATIC_CAST(easysimd_float32, INT32_MAX))) {
      return EASYSIMD_CONVERT_FTOI(int32_t, a.f32[0]);
    }
    return INT32_MIN;
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_FAST_CONVERSION_RANGE)
      return EASYSIMD_CONVERT_FTOI(int32_t, vgetq_lane_f32(a_.neon_f32, 0));
    #else
      easysimd_float32 v = a_.f32[0];
      #if !defined(EASYSIMD_FAST_CONVERSION_RANGE)
        return ((v > HEDLEY_STATIC_CAST(easysimd_float32, INT32_MIN)) && (v < HEDLEY_STATIC_CAST(easysimd_float32, INT32_MAX))) ?
          EASYSIMD_CONVERT_FTOI(int32_t, v) : INT32_MIN;
      #else
        return EASYSIMD_CONVERT_FTOI(int32_t, v);
      #endif
    #endif
  #endif
}
#define easysimd_mm_cvttss_si32(a) easysimd_mm_cvtt_ss2si((a))
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cvtt_ss2si(a) easysimd_mm_cvtt_ss2si((a))
#  define _mm_cvttss_si32(a) easysimd_mm_cvtt_ss2si((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_mm_cvttss_si64 (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_ARCH_AMD64) && !defined(_MSC_VER)
    #if defined(__PGI)
      return _mm_cvttss_si64x(a);
    #else
      return _mm_cvttss_si64(a);
    #endif
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_i64 = svcvt_s64_f32_z(svptrue_b64(), a.sve_f32);
    return r.i64[0];
    //return EASYSIMD_CONVERT_FTOI(int64_t, a.f32[0]);
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      return EASYSIMD_CONVERT_FTOI(int64_t, vgetq_lane_f32(a_.neon_f32, 0));
    #else
      return EASYSIMD_CONVERT_FTOI(int64_t, a_.f32[0]);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES) || (defined(EASYSIMD_ENABLE_NATIVE_ALIASES) && !defined(EASYSIMD_ARCH_AMD64))
#  define _mm_cvttss_si64(a) easysimd_mm_cvttss_si64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_cmpord_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_cmpord_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u32[0] = (easysimd_math_isnanf(easysimd_mm_cvtss_f32(a)) || easysimd_math_isnanf(easysimd_mm_cvtss_f32(b))) ? UINT32_C(0) : ~UINT32_C(0);
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpord_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_cmpord_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(easysimd_math_isnanf)
      r_.u32[0] = (easysimd_math_isnanf(easysimd_mm_cvtss_f32(a)) || easysimd_math_isnanf(easysimd_mm_cvtss_f32(b))) ? UINT32_C(0) : ~UINT32_C(0);
      EASYSIMD_VECTORIZE
      for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.u32[i] = a_.u32[i];
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_cmpord_ss(a, b) easysimd_mm_cmpord_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_div_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdiv_f32_z(svptrue_b32(), a.sve_f32, b.sve_f32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vdivq_f32(a.neon_f32, b.neon_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      float32x4_t recip0 = vrecpeq_f32(b_.neon_f32);
      float32x4_t recip1 = vmulq_f32(recip0, vrecpsq_f32(recip0, b_.neon_f32));
      r_.neon_f32 = vmulq_f32(a_.neon_f32, recip1);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 / b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i] / b_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_div_ps(a, b) easysimd_mm_div_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_div_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_div_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdiv_f32_z(svptrue_b32(), a.sve_f32, svdupq_n_f32(b.f32[0], 1.0, 1.0, 1.0));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    float32_t value = vgetq_lane_f32(easysimd_mm_div_ps(a, b).neon_f32, 0);
    r.neon_f32 = vsetq_lane_f32(value, a.neon_f32, 0);
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_div_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_div_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.f32[0] = a_.f32[0] / b_.f32[0];
    EASYSIMD_VECTORIZE
    for (size_t i = 1 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
      r_.f32[i] = a_.f32[i];
    }

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_div_ss(a, b) easysimd_mm_div_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_mm_extract_pi16 (easysimd__m64 a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return a.i16[imm8 & 3];
  #else
    easysimd__m64_private a_ = easysimd__m64_to_private(a);
    return a_.i16[imm8];
  #endif
}
#if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE) && !defined(HEDLEY_PGI_VERSION) && !defined(EASYSIMD_BUG_CLANG_44589)
  #define easysimd_mm_extract_pi16(a, imm8) HEDLEY_STATIC_CAST(int16_t, _mm_extract_pi16(a, imm8))
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_mm_extract_pi16(a, imm8) vget_lane_s16(easysimd__m64_to_private(a).neon_i16, imm8)
#endif
#define easysimd_m_pextrw(a, imm8) easysimd_mm_extract_pi16(a, imm8)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_extract_pi16(a, imm8) easysimd_mm_extract_pi16((a), (imm8))
#  define _m_pextrw(a, imm8) easysimd_mm_extract_pi16((a), (imm8))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_insert_pi16 (easysimd__m64 a, int16_t i, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 3) {
  easysimd__m64_private
    a_ = easysimd__m64_to_private(a);

  a_.i16[imm8] = i;

  return easysimd__m64_from_private(a_);
}
#if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE) && !defined(__PGI) && !defined(EASYSIMD_BUG_CLANG_44589)
  #define easysimd_mm_insert_pi16(a, i, imm8) _mm_insert_pi16(a, i, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  #define _mm_insert_pi16(a, i, imm8) easysimd_mm_insert_pi16(a, i, imm8)
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define easysimd_mm_insert_pi16(a, i, imm8) easysimd__m64_from_neon_i16(vset_lane_s16((i), easysimd__m64_to_neon_i16(a), (imm8)))
#endif
#define easysimd_m_pinsrw(a, i, imm8) (easysimd_mm_insert_pi16(a, i, imm8))
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_insert_pi16(a, i, imm8) easysimd_mm_insert_pi16(a, i, imm8)
#  define _m_pinsrw(a, i, imm8) easysimd_mm_insert_pi16(a, i, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_load_ps (easysimd_float32 const mem_addr[HEDLEY_ARRAY_PARAM(4)]) {
#if defined(EASYSIMD_X86_SSE_NATIVE)
  return _mm_load_ps(mem_addr);
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  easysimd__m128 r;
  r.sve_f32 = svld1_f32(svptrue_b32(), (float32_t const *)mem_addr);
  return r;
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  easysimd__m128 r;
  r.neon_f32 = vld1q_f32((float32_t const *)mem_addr);
  return r;
#else
  easysimd__m128_private r_;
  easysimd_memcpy(&r_, EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128), sizeof(r_));
  return easysimd__m128_from_private(r_);
#endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_load_ps(mem_addr) easysimd_mm_load_ps(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_load1_ps (easysimd_float32 const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_load_ps1(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdup_n_f32(*mem_addr);
    return r;
  #else
    easysimd__m128_private r_;

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_f32 = vld1q_dup_f32(mem_addr);
    #else
      r_ = easysimd__m128_to_private(easysimd_mm_set1_ps(*mem_addr));
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#define easysimd_mm_load_ps1(mem_addr) easysimd_mm_load1_ps(mem_addr)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_load_ps1(mem_addr) easysimd_mm_load1_ps(mem_addr)
#  define _mm_load1_ps(mem_addr) easysimd_mm_load1_ps(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_load_ss (easysimd_float32 const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_load_ss(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdupq_n_f32(*mem_addr, 0.0, 0.0, 0.0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsetq_lane_f32(*mem_addr, vdupq_n_f32(0), 0);
    return r;
  #else
    easysimd__m128_private r_;

    r_.f32[0] = *mem_addr;
    r_.i32[1] = 0;
    r_.i32[2] = 0;
    r_.i32[3] = 0;

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_load_ss(mem_addr) easysimd_mm_load_ss(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_loadh_pi (easysimd__m128 a, easysimd__m64 const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_loadh_pi(a, HEDLEY_REINTERPRET_CAST(__m64 const*, mem_addr));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcombine_f32(vget_low_f32(a.neon_f32), vld1_f32(HEDLEY_REINTERPRET_CAST(const float32_t*, mem_addr)));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[1] = *(uint64_t const *)mem_addr;
    return a;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    easysimd__m64_private b_ = *HEDLEY_REINTERPRET_CAST(easysimd__m64_private const*, mem_addr);
    r_.f32[0] = a_.f32[0];
    r_.f32[1] = a_.f32[1];
    r_.f32[2] = b_.f32[0];
    r_.f32[3] = b_.f32[1];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #if HEDLEY_HAS_WARNING("-Wold-style-cast")
    #define _mm_loadh_pi(a, mem_addr) easysimd_mm_loadh_pi((a), HEDLEY_REINTERPRET_CAST(easysimd__m64 const*, (mem_addr)))
  #else
    #define _mm_loadh_pi(a, mem_addr) easysimd_mm_loadh_pi((a), (easysimd__m64 const*) (mem_addr))
  #endif
#endif

/* The SSE documentation says that there are no alignment requirements
   for mem_addr.  Unfortunately they used the __m64 type for the argument
   which is supposed to be 8-byte aligned, so some compilers (like clang
   with -Wcast-align) will generate a warning if you try to cast, say,
   a easysimd_float32* to a easysimd__m64* for this function.

   I think the choice of argument type is unfortunate, but I do think we
   need to stick to it here.  If there is demand I can always add something
   like easysimd_x_mm_loadl_f32(easysimd__m128, easysimd_float32 mem_addr[2]) */
EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_loadl_pi (easysimd__m128 a, easysimd__m64 const* mem_addr) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_loadl_pi(a, HEDLEY_REINTERPRET_CAST(__m64 const*, mem_addr));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vcombine_f32(vld1_f32(HEDLEY_REINTERPRET_CAST(const float32_t*, mem_addr)), vget_high_f32(a.neon_f32));
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    a.u64[0] = *(uint64_t const *)mem_addr;
    return a;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    easysimd__m64_private b_;
    easysimd_memcpy(&b_, mem_addr, sizeof(b_));
    r_.i32[0] = b_.i32[0];
    r_.i32[1] = b_.i32[1];
    r_.i32[2] = a_.i32[2];
    r_.i32[3] = a_.i32[3];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #if HEDLEY_HAS_WARNING("-Wold-style-cast")
    #define _mm_loadl_pi(a, mem_addr) easysimd_mm_loadl_pi((a), HEDLEY_REINTERPRET_CAST(easysimd__m64 const*, (mem_addr)))
  #else
    #define _mm_loadl_pi(a, mem_addr) easysimd_mm_loadl_pi((a), (easysimd__m64 const*) (mem_addr))
  #endif
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_loadr_ps (easysimd_float32 const mem_addr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_loadr_ps(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svrev_f32(svld1_f32(svptrue_b32(), (float32_t const *)mem_addr));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 v, r;
    v = easysimd_mm_load_ps(mem_addr);
    r.neon_f32 = vrev64q_f32(v.neon_f32);
    r.neon_f32 = vextq_f32(r.neon_f32, r.neon_f32, 2);
    return r;
  #else
    easysimd__m128_private
      r_,
      v_ = easysimd__m128_to_private(easysimd_mm_load_ps(mem_addr));

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, v_.f32, v_.f32, 3, 2, 1, 0);
    #else
      r_.f32[0] = v_.f32[3];
      r_.f32[1] = v_.f32[2];
      r_.f32[2] = v_.f32[1];
      r_.f32[3] = v_.f32[0];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_loadr_ps(mem_addr) easysimd_mm_loadr_ps(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_loadu_ps (easysimd_float32 const mem_addr[HEDLEY_ARRAY_PARAM(4)]) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_loadu_ps(mem_addr);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svld1_f32(svptrue_b32(), &(mem_addr[0]));
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE))
    easysimd__m128 r;
    r.neon_f32 = vld1q_f32(HEDLEY_REINTERPRET_CAST(const float32_t*, mem_addr));
    return r;
  #else
    easysimd__m128_private r_;
    easysimd_memcpy(&r_, mem_addr, sizeof(r_));
    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_loadu_ps(mem_addr) easysimd_mm_loadu_ps(mem_addr)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_maskmove_si64 (easysimd__m64 a, easysimd__m64 mask, int8_t* mem_addr) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    _mm_maskmove_si64(a, mask, HEDLEY_REINTERPRET_CAST(char*, mem_addr));
  #else
    easysimd__m64_private
      a_ = easysimd__m64_to_private(a),
      mask_ = easysimd__m64_to_private(mask);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(a_.i8) / sizeof(a_.i8[0])) ; i++)
      if (mask_.i8[i] < 0)
        mem_addr[i] = a_.i8[i];
  #endif
}
#define easysimd_m_maskmovq(a, mask, mem_addr) easysimd_mm_maskmove_si64(a, mask, mem_addr)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_maskmove_si64(a, mask, mem_addr) easysimd_mm_maskmove_si64((a), (mask), EASYSIMD_CHECKED_REINTERPRET_CAST(int8_t*, char*, (mem_addr)))
#  define _m_maskmovq(a, mask, mem_addr) easysimd_mm_maskmove_si64((a), (mask), EASYSIMD_CHECKED_REINTERPRET_CAST(int8_t*, char*, (mem_addr)))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_max_pi16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_max_pi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_i16 = vmax_s16(a.neon_i16, b.neon_i16);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = (a_.i16[i] > b_.i16[i]) ? a_.i16[i] : b_.i16[i];
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#define easysimd_m_pmaxsw(a, b) easysimd_mm_max_pi16(a, b)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_max_pi16(a, b) easysimd_mm_max_pi16(a, b)
#  define _m_pmaxsw(a, b) easysimd_mm_max_pi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_max_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_max_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(svcmpgt_f32(svptrue_b32(), a.sve_f32, b.sve_f32), a.sve_f32, b.sve_f32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
    easysimd__m128 r;
    r.neon_f32 = vbslq_f32(vcgtq_f32(a.neon_f32, b.neon_f32), a.neon_f32, b.neon_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_FAST_NANS)
      r_.neon_f32 = vmaxq_f32(a_.neon_f32, b_.neon_f32);
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = (a_.f32[i] > b_.f32[i]) ? a_.f32[i] : b_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_max_ps(a, b) easysimd_mm_max_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_max_pu8 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_max_pu8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_u8 = vmax_u8(a.neon_u8, b.neon_u8);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = (a_.u8[i] > b_.u8[i]) ? a_.u8[i] : b_.u8[i];
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#define easysimd_m_pmaxub(a, b) easysimd_mm_max_pu8(a, b)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_max_pu8(a, b) easysimd_mm_max_pu8(a, b)
#  define _m_pmaxub(a, b) easysimd_mm_max_pu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_max_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_max_ss(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    float32_t value = vgetq_lane_f32(vmaxq_f32(a.neon_f32, b.neon_f32), 0);
    r.neon_f32 = vsetq_lane_f32(value, a.neon_f32, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp;
    tmp.sve_f32 = svmax_f32_z(svptrue_b32(), a.sve_f32, b.sve_f32);
    a.f32[0] = tmp.f32[0];
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_max_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_max_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.f32[0] = (a_.f32[0] > b_.f32[0]) ? a_.f32[0] : b_.f32[0];
    r_.f32[1] = a_.f32[1];
    r_.f32[2] = a_.f32[2];
    r_.f32[3] = a_.f32[3];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_max_ss(a, b) easysimd_mm_max_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_min_pi16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_min_pi16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_i16 = vmin_s16(a.neon_i16, b.neon_i16);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.i16) / sizeof(r_.i16[0])) ; i++) {
      r_.i16[i] = (a_.i16[i] < b_.i16[i]) ? a_.i16[i] : b_.i16[i];
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#define easysimd_m_pminsw(a, b) easysimd_mm_min_pi16(a, b)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_min_pi16(a, b) easysimd_mm_min_pi16(a, b)
#  define _m_pminsw(a, b) easysimd_mm_min_pi16(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_min_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_min_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsel_f32(svcmplt_f32(svptrue_b32(), a.sve_f32, b.sve_f32), a.sve_f32, b.sve_f32);
    return r;
  #elif (defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE))
    easysimd__m128 r;
    r.neon_f32 = vbslq_f32(vcltq_f32(a.neon_f32, b.neon_f32), a.neon_f32, b.neon_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_FAST_NANS) && defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_f32 = vminq_f32(a_.neon_f32, b_.neon_f32);
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      uint32_t EASYSIMD_VECTOR(16) m = HEDLEY_REINTERPRET_CAST(__typeof__(m), a_.f32 < b_.f32);
      r_.f32 =
        HEDLEY_REINTERPRET_CAST(
          __typeof__(r_.f32),
          ( (HEDLEY_REINTERPRET_CAST(__typeof__(m), a_.f32) &  m) |
            (HEDLEY_REINTERPRET_CAST(__typeof__(m), b_.f32) & ~m)
          )
        );
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = (a_.f32[i] < b_.f32[i]) ? a_.f32[i] : b_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_min_ps(a, b) easysimd_mm_min_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_min_pu8 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_min_pu8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    r.neon_u8 = vmin_u8(a.neon_u8, b.neon_u8);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      r_.u8[i] = (a_.u8[i] < b_.u8[i]) ? a_.u8[i] : b_.u8[i];
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#define easysimd_m_pminub(a, b) easysimd_mm_min_pu8(a, b)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_min_pu8(a, b) easysimd_mm_min_pu8(a, b)
#  define _m_pminub(a, b) easysimd_mm_min_pu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_min_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_min_ss(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    float32_t value = vgetq_lane_f32(vminq_f32(a.neon_f32, b.neon_f32), 0);
    r.neon_f32 = vsetq_lane_f32(value, a.neon_f32, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp;
    tmp.sve_f32 = svmin_f32_z(svptrue_b32(), a.sve_f32, b.sve_f32);
    a.f32[0] = tmp.f32[0];
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_min_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_min_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.f32[0] = (a_.f32[0] < b_.f32[0]) ? a_.f32[0] : b_.f32[0];
    r_.f32[1] = a_.f32[1];
    r_.f32[2] = a_.f32[2];
    r_.f32[3] = a_.f32[3];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_min_ss(a, b) easysimd_mm_min_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_movehl_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_movehl_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    float32x2_t a32 = vget_high_f32(a.neon_f32);
    float32x2_t b32 = vget_high_f32(b.neon_f32);
    r.neon_f32 = vcombine_f32(b32, a32);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svfloat32x2_t svdata = svcreate2_f32(a.sve_f32, b.sve_f32);
    r.sve_f32 = svtbl2_f32(svdata, svdupq_n_u32(6, 7, 2, 3));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, b_.f32, 6, 7, 2, 3);
    #else
      r_.f32[0] = b_.f32[2];
      r_.f32[1] = b_.f32[3];
      r_.f32[2] = a_.f32[2];
      r_.f32[3] = a_.f32[3];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_movehl_ps(a, b) easysimd_mm_movehl_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_movelh_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_movelh_ps(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    float32x2_t a10 = vget_low_f32(a.neon_f32);
    float32x2_t b10 = vget_low_f32(b.neon_f32);
    r.neon_f32 = vcombine_f32(a10, b10);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdupq_n_f32(a.f32[0], a.f32[1], b.f32[0], b.f32[1]);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, b_.f32, 0, 1, 4, 5);
    #else
      r_.f32[0] = a_.f32[0];
      r_.f32[1] = a_.f32[1];
      r_.f32[2] = b_.f32[0];
      r_.f32[3] = b_.f32[1];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_movelh_ps(a, b) easysimd_mm_movelh_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_movemask_pi8 (easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_movemask_pi8(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    int r = 0;
    uint8x8_t input = a.neon_u8;
    const int8_t xr[8] = {-7, -6, -5, -4, -3, -2, -1, 0};
    const uint8x8_t mask_and = vdup_n_u8(0x80);
    const int8x8_t mask_shift = vld1_s8(xr);
    const uint8x8_t mask_result = vshl_u8(vand_u8(input, mask_and), mask_shift);
    uint8x8_t lo = mask_result;
    r = vaddv_u8(lo);
    return r;
  #else
    easysimd__m64_private a_ = easysimd__m64_to_private(a);
    int r = 0;

    const size_t nmemb = sizeof(a_.i8) / sizeof(a_.i8[0]);
    EASYSIMD_VECTORIZE_REDUCTION(|:r)
    for (size_t i = 0 ; i < nmemb ; i++) {
      r |= (a_.u8[nmemb - 1 - i] >> 7) << (nmemb - 1 - i);
    }

    return r;
  #endif
}
#define easysimd_m_pmovmskb(a) easysimd_mm_movemask_pi8(a)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_movemask_pi8(a) easysimd_mm_movemask_pi8(a)
#  define _m_pmovmskb(a) easysimd_mm_movemask_pi8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_movemask_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_movemask_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    int k = 0;
    easysimd__m128 mask32 = {
      .i32 = {1, 2, 4, 8}
    };
    k = svaddv_s32(svcmplt_s32(svptrue_b32(), svreinterpret_s32_f32(a.sve_f32), svdup_n_s32(0)), mask32.sve_i32);
    return k;
  #else
    int r = 0;
    easysimd__m128_private a_ = easysimd__m128_to_private(a);
    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      // Shift out everything but the sign bits with a 32-bit unsigned shift right.
      uint64x2_t high_bits = vreinterpretq_u64_u32(vshrq_n_u32(a_.neon_u32, 31));
      // Merge the two pairs together with a 64-bit unsigned shift right + add.
      uint8x16_t paired = vreinterpretq_u8_u64(vsraq_n_u64(high_bits, high_bits, 31));
      // Extract the result.
      return vgetq_lane_u8(paired, 0) | (vgetq_lane_u8(paired, 8) << 2);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      static const uint32_t md[4] = {
        1 << 0, 1 << 1, 1 << 2, 1 << 3
      };

      uint32x4_t extended = vreinterpretq_u32_s32(vshrq_n_s32(a_.neon_i32, 31));
      uint32x4_t masked = vandq_u32(vld1q_u32(md), extended);
      #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
        return HEDLEY_STATIC_CAST(int32_t, vaddvq_u32(masked));
      #else
        uint64x2_t t64 = vpaddlq_u32(masked);
        return
          HEDLEY_STATIC_CAST(int, vgetq_lane_u64(t64, 0)) +
          HEDLEY_STATIC_CAST(int, vgetq_lane_u64(t64, 1));
      #endif
    #else
      EASYSIMD_VECTORIZE_REDUCTION(|:r)
      for (size_t i = 0 ; i < sizeof(a_.u32) / sizeof(a_.u32[0]) ; i++) {
        r |= (a_.u32[i] >> ((sizeof(a_.u32[i]) * CHAR_BIT) - 1)) << i;
      }
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_movemask_ps(a) easysimd_mm_movemask_ps((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mul_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_mul_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svmul_f32_z(svptrue_b32(), a.sve_f32, b.sve_f32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vmulq_f32(a.neon_f32, b.neon_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 * b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i] * b_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_mul_ps(a, b) easysimd_mm_mul_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_mul_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_mul_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svmul_f32_z(svptrue_b32(), a.sve_f32, svdupq_n_f32(b.f32[0], 1.0, 1.0, 1.0));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    float32_t value = vgetq_lane_f32(vmulq_f32(a.neon_f32, b.neon_f32), 0);
    r.neon_f32 = vsetq_lane_f32(value, a.neon_f32, 0);
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_mul_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_mul_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.f32[0] = a_.f32[0] * b_.f32[0];
    r_.f32[1] = a_.f32[1];
    r_.f32[2] = a_.f32[2];
    r_.f32[3] = a_.f32[3];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_mul_ss(a, b) easysimd_mm_mul_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_mulhi_pu16 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_mulhi_pu16(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    const uint32x4_t t1 = vmull_u16(a.neon_u16, b.neon_u16);
    const uint32x4_t t2 = vshrq_n_u32(t1, 16);
    const uint16x4_t t3 = vmovn_u32(t2);
    r.neon_u16 = t3;
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.u16) / sizeof(r_.u16[0])) ; i++) {
      r_.u16[i] = HEDLEY_STATIC_CAST(uint16_t, ((HEDLEY_STATIC_CAST(uint32_t, a_.u16[i]) * HEDLEY_STATIC_CAST(uint32_t, b_.u16[i])) >> UINT32_C(16)));
    }

    return easysimd__m64_from_private(r_);
  #endif
}
#define easysimd_m_pmulhuw(a, b) easysimd_mm_mulhi_pu16(a, b)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_mulhi_pu16(a, b) easysimd_mm_mulhi_pu16(a, b)
#  define _m_pmulhuw(a, b) easysimd_mm_mulhi_pu16(a, b)
#endif

#if defined(EASYSIMD_X86_SSE_NATIVE) && defined(HEDLEY_GCC_VERSION)
  #define EASYSIMD_MM_HINT_NTA  HEDLEY_STATIC_CAST(enum _mm_hint, 0)
  #define EASYSIMD_MM_HINT_T0   HEDLEY_STATIC_CAST(enum _mm_hint, 1)
  #define EASYSIMD_MM_HINT_T1   HEDLEY_STATIC_CAST(enum _mm_hint, 2)
  #define EASYSIMD_MM_HINT_T2   HEDLEY_STATIC_CAST(enum _mm_hint, 3)
  #define EASYSIMD_MM_HINT_ENTA HEDLEY_STATIC_CAST(enum _mm_hint, 4)
  #define EASYSIMD_MM_HINT_ET0  HEDLEY_STATIC_CAST(enum _mm_hint, 5)
  #define EASYSIMD_MM_HINT_ET1  HEDLEY_STATIC_CAST(enum _mm_hint, 6)
  #define EASYSIMD_MM_HINT_ET2  HEDLEY_STATIC_CAST(enum _mm_hint, 7)
#else
  #define EASYSIMD_MM_HINT_NTA  0
  #define EASYSIMD_MM_HINT_T0   1
  #define EASYSIMD_MM_HINT_T1   2
  #define EASYSIMD_MM_HINT_T2   3
  #define EASYSIMD_MM_HINT_ENTA 4
  #define EASYSIMD_MM_HINT_ET0  5
  #define EASYSIMD_MM_HINT_ET1  6
  #define EASYSIMD_MM_HINT_ET2  7
#endif

#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  HEDLEY_DIAGNOSTIC_PUSH
  #if HEDLEY_HAS_WARNING("-Wreserved-id-macro")
    _Pragma("clang diagnostic ignored \"-Wreserved-id-macro\"")
  #endif
  #undef  _MM_HINT_NTA
  #define _MM_HINT_NTA  EASYSIMD_MM_HINT_NTA
  #undef  _MM_HINT_T0
  #define _MM_HINT_T0   EASYSIMD_MM_HINT_T0
  #undef  _MM_HINT_T1
  #define _MM_HINT_T1   EASYSIMD_MM_HINT_T1
  #undef  _MM_HINT_T2
  #define _MM_HINT_T2   EASYSIMD_MM_HINT_T2
  #undef  _MM_HINT_ENTA
  #define _MM_HINT_ETNA EASYSIMD_MM_HINT_ENTA
  #undef  _MM_HINT_ET0
  #define _MM_HINT_ET0  EASYSIMD_MM_HINT_ET0
  #undef  _MM_HINT_ET1
  #define _MM_HINT_ET1  EASYSIMD_MM_HINT_ET1
  #undef  _MM_HINT_ET1
  #define _MM_HINT_ET2  EASYSIMD_MM_HINT_ET2
  HEDLEY_DIAGNOSTIC_POP
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void easysimd__pldx_nta(const void * p){
  __builtin_prefetch(p, 0, 0);
}

EASYSIMD_FUNCTION_ATTRIBUTES
void easysimd__pldx_t0(const void * p){
  __builtin_prefetch(p, 0, 3);
}

EASYSIMD_FUNCTION_ATTRIBUTES
void easysimd__pldx_t1(const void * p){
 __builtin_prefetch(p, 0, 2);
}

EASYSIMD_FUNCTION_ATTRIBUTES
void easysimd__pldx_t2(const void * p){
  __builtin_prefetch(p, 0, 1);
}

EASYSIMD_FUNCTION_ATTRIBUTES
void easysimd__pldx_enta(const void * p){
  __builtin_prefetch(p, 1, 0);
}

EASYSIMD_FUNCTION_ATTRIBUTES
void easysimd__pldx_et0(const void * p){
  __builtin_prefetch(p, 1, 3);
}

EASYSIMD_FUNCTION_ATTRIBUTES
void easysimd__pldx_et1(const void * p){
  __builtin_prefetch(p, 1, 2);
}

EASYSIMD_FUNCTION_ATTRIBUTES
void easysimd__pldx_et2(const void * p){
  __builtin_prefetch(p, 0, 1);
}

typedef struct {
  int OP;
  void (*pldxfun)(const void *);
}PldxFun;

static PldxFun pldxfunlist[] = {
  {EASYSIMD_MM_HINT_NTA , easysimd__pldx_nta }, {EASYSIMD_MM_HINT_T0 , easysimd__pldx_t0 }, {EASYSIMD_MM_HINT_T1 , easysimd__pldx_t1 }, {EASYSIMD_MM_HINT_T2 , easysimd__pldx_t2 },
  {EASYSIMD_MM_HINT_ENTA, easysimd__pldx_enta}, {EASYSIMD_MM_HINT_ET0, easysimd__pldx_et0}, {EASYSIMD_MM_HINT_ET1, easysimd__pldx_et1}, {EASYSIMD_MM_HINT_ET2, easysimd__pldx_et2}
};

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_prefetch (const void* p, int i) {
  #if \
      HEDLEY_HAS_BUILTIN(__builtin_prefetch) || \
      HEDLEY_GCC_VERSION_CHECK(3,4,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0)
    pldxfunlist[i].pldxfun(p);
  #elif defined(__ARM_ACLE)
    #if (__ARM_ACLE >= 101)
      switch(i) {
        case EASYSIMD_MM_HINT_NTA:
          __pldx(0, 0, 1, p);
          break;
        case EASYSIMD_MM_HINT_T0:
          __pldx(0, 0, 0, p);
          break;
        case EASYSIMD_MM_HINT_T1:
          __pldx(0, 1, 0, p);
          break;
        case EASYSIMD_MM_HINT_T2:
          __pldx(0, 2, 0, p);
          break;
        case EASYSIMD_MM_HINT_ENTA:
          __pldx(1, 0, 1, p);
          break;
        case EASYSIMD_MM_HINT_ET0:
          __pldx(1, 0, 0, p);
          break;
        case EASYSIMD_MM_HINT_ET1:
          __pldx(1, 1, 0, p);
          break;
        case EASYSIMD_MM_HINT_ET2:
          __pldx(1, 2, 0, p);
          break;
      }
    #else
      (void) i;
      __pld(p)
    #endif
  #elif HEDLEY_PGI_VERSION_CHECK(10,0,0)
    (void) i;
    #pragma mem prefetch p
  #elif HEDLEY_CRAY_VERSION_CHECK(8,1,0)
    switch (i) {
      case EASYSIMD_MM_HINT_NTA:
        #pragma _CRI prefetch (nt) p
        break;
      case EASYSIMD_MM_HINT_T0:
      case EASYSIMD_MM_HINT_T1:
      case EASYSIMD_MM_HINT_T2:
        #pragma _CRI prefetch p
        break;
      case EASYSIMD_MM_HINT_ENTA:
        #pragma _CRI prefetch (write, nt) p
        break;
      case EASYSIMD_MM_HINT_ET0:
      case EASYSIMD_MM_HINT_ET1:
      case EASYSIMD_MM_HINT_ET2:
        #pragma _CRI prefetch (write) p
        break;
    }
  #elif HEDLEY_IBM_VERSION_CHECK(11,0,0)
    switch(i) {
      case EASYSIMD_MM_HINT_NTA:
        __prefetch_by_load(p, 0, 0);
        break;
      case EASYSIMD_MM_HINT_T0:
        __prefetch_by_load(p, 0, 3);
        break;
      case EASYSIMD_MM_HINT_T1:
        __prefetch_by_load(p, 0, 2);
        break;
      case EASYSIMD_MM_HINT_T2:
        __prefetch_by_load(p, 0, 1);
        break;
      case EASYSIMD_MM_HINT_ENTA:
        __prefetch_by_load(p, 1, 0);
        break;
      case EASYSIMD_MM_HINT_ET0:
        __prefetch_by_load(p, 1, 3);
        break;
      case EASYSIMD_MM_HINT_ET1:
        __prefetch_by_load(p, 1, 2);
        break;
      case EASYSIMD_MM_HINT_ET2:
        __prefetch_by_load(p, 0, 1);
        break;
    }
  #endif
}
#if defined(EASYSIMD_X86_SSE_NATIVE)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(10,0,0) /* https://reviews.llvm.org/D71718 */
    #define easysimd_mm_prefetch(p, i) \
      (__extension__({ \
        HEDLEY_DIAGNOSTIC_PUSH \
        HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL \
        _mm_prefetch((p), (i)); \
        HEDLEY_DIAGNOSTIC_POP \
      }))
  #else
    #define easysimd_mm_prefetch(p, i) _mm_prefetch(p, i)
  #endif
#endif
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
  #define _mm_prefetch(p, i) easysimd_mm_prefetch(p, i)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_negate_ps(easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return easysimd_mm_xor_ps(a, _mm_set1_ps(EASYSIMD_FLOAT32_C(-0.0)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_f32 = vnegq_f32(a_.neon_f32);
    #elif defined(EASYSIMD_VECTOR_NEGATE)
      r_.f32 = -a_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = -a_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_rcp_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_rcp_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svrecpe_f32(a.sve_f32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vrecpeq_f32(a.neon_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      float32x4_t recip = vrecpeq_f32(a_.neon_f32);

      #if EASYSIMD_ACCURACY_PREFERENCE > 0
        for (int i = 0; i < EASYSIMD_ACCURACY_PREFERENCE ; ++i) {
          recip = vmulq_f32(recip, vrecpsq_f32(recip, a_.neon_f32));
        }
      #endif

      r_.neon_f32 = recip;
    #elif defined(EASYSIMD_VECTOR_SUBSCRIPT_SCALAR)
      r_.f32 = 1.0f / a_.f32;
    #elif defined(EASYSIMD_IEEE754_STORAGE)
      /* https://stackoverflow.com/questions/12227126/division-as-multiply-and-lut-fast-float-division-reciprocal/12228234#12228234 */
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        int32_t ix;
        easysimd_float32 fx = a_.f32[i];
        easysimd_memcpy(&ix, &fx, sizeof(ix));
        int32_t x = INT32_C(0x7EF311C3) - ix;
        easysimd_float32 temp;
        easysimd_memcpy(&temp, &x, sizeof(temp));
        r_.f32[i] = temp * (EASYSIMD_FLOAT32_C(2.0) - temp * fx);
      }
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = 1.0f / a_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_rcp_ps(a) easysimd_mm_rcp_ps((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_rcp_ss (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_rcp_ss(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsetq_lane_f32(vgetq_lane_f32(easysimd_mm_rcp_ps(a).neon_f32, 0), a.neon_f32, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp;
    tmp.sve_f32 = svrecpe_f32(a.sve_f32);
    a.f32[0] = tmp.f32[0];
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_rcp_ps(a));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_rcp_ps(easysimd_x_mm_broadcastlow_ps(a)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    r_.f32[0] = 1.0f / a_.f32[0];
    r_.f32[1] = a_.f32[1];
    r_.f32[2] = a_.f32[2];
    r_.f32[3] = a_.f32[3];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_rcp_ss(a) easysimd_mm_rcp_ss((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_rsqrt_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_rsqrt_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svrsqrte_f32(a.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      r_.neon_f32 = vrsqrteq_f32(a_.neon_f32);
    #elif defined(EASYSIMD_IEEE754_STORAGE)
      /* https://basesandframes.files.wordpress.com/2020/04/even_faster_math_functions_green_2020.pdf
        Pages 100 - 103 */
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        #if EASYSIMD_ACCURACY_PREFERENCE <= 0
          r_.i32[i] = INT32_C(0x5F37624F) - (a_.i32[i] >> 1);
        #else
          easysimd_float32 x = a_.f32[i];
          easysimd_float32 xhalf = EASYSIMD_FLOAT32_C(0.5) * x;
          int32_t ix;

          easysimd_memcpy(&ix, &x, sizeof(ix));

          #if EASYSIMD_ACCURACY_PREFERENCE == 1
            ix = INT32_C(0x5F375A82) - (ix >> 1);
          #else
            ix = INT32_C(0x5F37599E) - (ix >> 1);
          #endif

          easysimd_memcpy(&x, &ix, sizeof(x));

          #if EASYSIMD_ACCURACY_PREFERENCE >= 2
            x = x * (EASYSIMD_FLOAT32_C(1.5008909) - xhalf * x * x);
          #endif
          x = x * (EASYSIMD_FLOAT32_C(1.5008909) - xhalf * x * x);

          r_.f32[i] = x;
        #endif
      }
    #elif defined(easysimd_math_sqrtf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = 1.0f / easysimd_math_sqrtf(a_.f32[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_rsqrt_ps(a) easysimd_mm_rsqrt_ps((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_rsqrt_ss (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_rsqrt_ss(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsetq_lane_f32(vgetq_lane_f32(easysimd_mm_rsqrt_ps(a).neon_f32, 0), a.neon_f32, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp;
    tmp.sve_f32 = svrsqrte_f32(a.sve_f32);
    a.f32[0] = tmp.f32[0];
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_rsqrt_ps(a));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_rsqrt_ps(easysimd_x_mm_broadcastlow_ps(a)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

  #if defined(EASYSIMD_IEEE754_STORAGE)
    {
      #if EASYSIMD_ACCURACY_PREFERENCE <= 0
        r_.i32[0] = INT32_C(0x5F37624F) - (a_.i32[0] >> 1);
      #else
        easysimd_float32 x = a_.f32[0];
        easysimd_float32 xhalf = EASYSIMD_FLOAT32_C(0.5) * x;
        int32_t ix;

        easysimd_memcpy(&ix, &x, sizeof(ix));

        #if EASYSIMD_ACCURACY_PREFERENCE == 1
          ix = INT32_C(0x5F375A82) - (ix >> 1);
        #else
          ix = INT32_C(0x5F37599E) - (ix >> 1);
        #endif

        easysimd_memcpy(&x, &ix, sizeof(x));

        #if EASYSIMD_ACCURACY_PREFERENCE >= 2
          x = x * (EASYSIMD_FLOAT32_C(1.5008909) - xhalf * x * x);
        #endif
        x = x * (EASYSIMD_FLOAT32_C(1.5008909) - xhalf * x * x);

        r_.f32[0] = x;
      #endif
    }
    r_.f32[1] = a_.f32[1];
    r_.f32[2] = a_.f32[2];
    r_.f32[3] = a_.f32[3];
  #elif defined(easysimd_math_sqrtf)
    r_.f32[0] = 1.0f / easysimd_math_sqrtf(a_.f32[0]);
    r_.f32[1] = a_.f32[1];
    r_.f32[2] = a_.f32[2];
    r_.f32[3] = a_.f32[3];
  #else
    HEDLEY_UNREACHABLE();
  #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_rsqrt_ss(a) easysimd_mm_rsqrt_ss((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_sad_pu8 (easysimd__m64 a, easysimd__m64 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_sad_pu8(a, b);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m64 r;
    uint64x1_t t = vpaddl_u32(vpaddl_u16(vpaddl_u8(vabd_u8(a.neon_u8, b.neon_u8))));
    r.neon_u16 = vset_lane_u16(HEDLEY_STATIC_CAST(uint64_t, vget_lane_u64(t, 0)), vdup_n_u16(0), 0);
    return r;
  #else
    easysimd__m64_private
      r_,
      a_ = easysimd__m64_to_private(a),
      b_ = easysimd__m64_to_private(b);

    uint16_t sum = 0;

    EASYSIMD_VECTORIZE_REDUCTION(+:sum)
    for (size_t i = 0 ; i < (sizeof(r_.u8) / sizeof(r_.u8[0])) ; i++) {
      sum += HEDLEY_STATIC_CAST(uint8_t, easysimd_math_abs(a_.u8[i] - b_.u8[i]));
    }

    r_.i16[0] = HEDLEY_STATIC_CAST(int16_t, sum);
    r_.i16[1] = 0;
    r_.i16[2] = 0;
    r_.i16[3] = 0;

    return easysimd__m64_from_private(r_);
  #endif
}
#define easysimd_m_psadbw(a, b) easysimd_mm_sad_pu8(a, b)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_sad_pu8(a, b) easysimd_mm_sad_pu8(a, b)
#  define _m_psadbw(a, b) easysimd_mm_sad_pu8(a, b)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_set_ss (easysimd_float32 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_set_ss(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdupq_n_f32(a, 0.0, 0.0, 0.0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsetq_lane_f32(a, vdupq_n_f32(EASYSIMD_FLOAT32_C(0.0)), 0);
    return r;
  #else
    return easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(0.0), EASYSIMD_FLOAT32_C(0.0), EASYSIMD_FLOAT32_C(0.0), a);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_set_ss(a) easysimd_mm_set_ss(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_setr_ps (easysimd_float32 e3, easysimd_float32 e2, easysimd_float32 e1, easysimd_float32 e0) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_setr_ps(e3, e2, e1, e0);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdupq_n_f32(e3, e2, e1, e0);
    return r;
  #else
    return easysimd_mm_set_ps(e0, e1, e2, e3);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_setr_ps(e3, e2, e1, e0) easysimd_mm_setr_ps(e3, e2, e1, e0)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_setzero_ps (void) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_setzero_ps();
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svdup_n_f32(0.0);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vdupq_n_f32(EASYSIMD_FLOAT32_C(0.0));
    return r;
  #else
    easysimd__m128 r;
    easysimd_memset(&r, 0, sizeof(r));
    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_setzero_ps() easysimd_mm_setzero_ps()
#endif

#if defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_undefined_ps (void) {
  easysimd__m128_private r_;

  #if defined(EASYSIMD_HAVE_UNDEFINED128)
    r_.n = _mm_undefined_ps();
  #elif !defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
    r_ = easysimd__m128_to_private(easysimd_mm_setzero_ps());
  #endif

  return easysimd__m128_from_private(r_);
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_undefined_ps() easysimd_mm_undefined_ps()
#endif

#if defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
HEDLEY_DIAGNOSTIC_POP
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_x_mm_setone_ps (void) {
  easysimd__m128 t = easysimd_mm_setzero_ps();
  return easysimd_mm_cmpeq_ps(t, t);
}

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_sfence (void) {
    /* TODO: Use Hedley. */
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_sfence();
  #elif defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7))
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  #elif !defined(__INTEL_COMPILER) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && !defined(__STDC_NO_ATOMICS__)
    #if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ < 9)
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
    #else
      atomic_thread_fence(memory_order_seq_cst);
    #endif
  #elif defined(_MSC_VER)
    MemoryBarrier();
  #elif HEDLEY_HAS_EXTENSION(c_atomic)
    __c11_atomic_thread_fence(__ATOMIC_SEQ_CST);
  #elif defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 1))
    __sync_synchronize();
  #elif defined(_OPENMP)
    #pragma omp critical(easysimd_mm_sfence_)
    { }
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_sfence() easysimd_mm_sfence()
#endif

#define EASYSIMD_MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _MM_SHUFFLE(z, y, x, w) EASYSIMD_MM_SHUFFLE(z, y, x, w)
#endif

#if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE) && !defined(__PGI)
#  define easysimd_mm_shuffle_pi16(a, imm8) _mm_shuffle_pi16(a, imm8)
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
#  define easysimd_mm_shuffle_pi16(a, imm8) (__extension__ ({ \
      const easysimd__m64_private easysimd__tmp_a_ = easysimd__m64_to_private(a); \
      easysimd__m64_from_private((easysimd__m64_private) { .i16 = \
        EASYSIMD_SHUFFLE_VECTOR_(16, 8, \
          (easysimd__tmp_a_).i16, \
          (easysimd__tmp_a_).i16, \
          (((imm8)     ) & 3), \
          (((imm8) >> 2) & 3), \
          (((imm8) >> 4) & 3), \
          (((imm8) >> 6) & 3)) }); }))
#else
EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m64
easysimd_mm_shuffle_pi16 (easysimd__m64 a, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  easysimd__m64_private r_;
  easysimd__m64_private a_ = easysimd__m64_to_private(a);

  for (size_t i = 0 ; i < sizeof(r_.i16) / sizeof(r_.i16[0]) ; i++) {
    r_.i16[i] = a_.i16[(imm8 >> (i * 2)) & 3];
  }

HEDLEY_DIAGNOSTIC_PUSH
#if HEDLEY_HAS_WARNING("-Wconditional-uninitialized")
#  pragma clang diagnostic ignored "-Wconditional-uninitialized"
#endif
  return easysimd__m64_from_private(r_);
HEDLEY_DIAGNOSTIC_POP
}
#endif
#if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE) && !defined(__PGI)
#  define easysimd_m_pshufw(a, imm8) _m_pshufw(a, imm8)
#else
#  define easysimd_m_pshufw(a, imm8) easysimd_mm_shuffle_pi16(a, imm8)
#endif
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_shuffle_pi16(a, imm8) easysimd_mm_shuffle_pi16(a, imm8)
#  define _m_pshufw(a, imm8) easysimd_mm_shuffle_pi16(a, imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_shuffle_ps (easysimd__m128 a, easysimd__m128 b, const int imm8)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(imm8, 0, 255) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    svbool_t   pgsel    = svdupq_n_b32(1, 1, 0, 0);
    svuint32_t svindexa = svdupq_n_u32(imm8        & 0x03, (imm8 >> 2) & 0x03, 0, 0);
    svuint32_t svindexb = svdupq_n_u32((imm8 >> 4) & 0x03, (imm8 >> 6) & 0x03, 0, 0);
    r.sve_f32 = svsplice(pgsel, svtbl_f32(a.sve_f32, svindexa), svtbl_f32(b.sve_f32, svindexb));
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.f32[0] = a_.f32[(imm8 >> 0) & 3];
    r_.f32[1] = a_.f32[(imm8 >> 2) & 3];
    r_.f32[2] = b_.f32[(imm8 >> 4) & 3];
    r_.f32[3] = b_.f32[(imm8 >> 6) & 3];

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_NATIVE) && !defined(__PGI)
#  define easysimd_mm_shuffle_ps(a, b, imm8) _mm_shuffle_ps(a, b, imm8)
#elif defined(EASYSIMD_ARM_SVE_NATIVE)
  #define _mm_shuffle_ps(a, b, imm8) easysimd_mm_shuffle_ps((a), (b), imm8)
#elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define easysimd_mm_shuffle_ps(a, b, imm8) ({easysimd__m128 r; \
    r.neon_f32 = vmovq_n_f32(vgetq_lane_f32(a.neon_f32, (imm8) & (0x3))); \
    r.neon_f32 = vsetq_lane_f32(vgetq_lane_f32(a.neon_f32, ((imm8) >> 2) & 0x3), r.neon_f32, 1); \
    r.neon_f32 = vsetq_lane_f32(vgetq_lane_f32(b.neon_f32, ((imm8) >> 4) & 0x3), r.neon_f32, 2); \
    r.neon_f32 = vsetq_lane_f32(vgetq_lane_f32(b.neon_f32, ((imm8) >> 6) & 0x3), r.neon_f32, 3); \
    r; \
  })
#elif defined(EASYSIMD_SHUFFLE_VECTOR_)
  #define easysimd_mm_shuffle_ps(a, b, imm8) (__extension__ ({ \
      easysimd__m128_from_private((easysimd__m128_private) { .f32 = \
        EASYSIMD_SHUFFLE_VECTOR_(32, 16, \
          easysimd__m128_to_private(a).f32, \
          easysimd__m128_to_private(b).f32, \
          (((imm8)     ) & 3), \
          (((imm8) >> 2) & 3), \
          (((imm8) >> 4) & 3) + 4, \
          (((imm8) >> 6) & 3) + 4) }); }))
#elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && defined(EASYSIMD_STATEMENT_EXPR_)
  #define easysimd_mm_shuffle_ps(a, b, imm8) \
    (__extension__({ \
      float32x4_t easysimd_mm_shuffle_ps_a_ = easysimd__m128i_to_neon_f32(a); \
      float32x4_t easysimd_mm_shuffle_ps_b_ = easysimd__m128i_to_neon_f32(b); \
      float32x4_t easysimd_mm_shuffle_ps_r_; \
      \
      easysimd_mm_shuffle_ps_r_ = vmovq_n_f32(vgetq_lane_f32(easysimd_mm_shuffle_ps_a_, (imm8) & (0x3))); \
      easysimd_mm_shuffle_ps_r_ = vsetq_lane_f32(vgetq_lane_f32(easysimd_mm_shuffle_ps_a_, ((imm8) >> 2) & 0x3), easysimd_mm_shuffle_ps_r_, 1); \
      easysimd_mm_shuffle_ps_r_ = vsetq_lane_f32(vgetq_lane_f32(easysimd_mm_shuffle_ps_b_, ((imm8) >> 4) & 0x3), easysimd_mm_shuffle_ps_r_, 2); \
                               vsetq_lane_f32(vgetq_lane_f32(easysimd_mm_shuffle_ps_b_, ((imm8) >> 6) & 0x3), easysimd_mm_shuffle_ps_r_, 3); \
    }))
#endif
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_shuffle_ps(a, b, imm8) easysimd_mm_shuffle_ps((a), (b), imm8)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_sqrt_ps (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sqrt_ps(a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsqrt_f32_z(svptrue_b32(), a.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f32 = vsqrtq_f32(a_.neon_f32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      float32x4_t est = vrsqrteq_f32(a_.neon_f32);
      for (int i = 0 ; i <= EASYSIMD_ACCURACY_PREFERENCE ; i++) {
        est = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a_.neon_f32, est), est), est);
      }
      r_.neon_f32 = vmulq_f32(a_.neon_f32, est);
    #elif defined(easysimd_math_sqrt)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < sizeof(r_.f32) / sizeof(r_.f32[0]) ; i++) {
        r_.f32[i] = easysimd_math_sqrtf(a_.f32[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_sqrt_ps(a) easysimd_mm_sqrt_ps((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_sqrt_ss (easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sqrt_ss(a);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    float32_t value =
            vgetq_lane_f32(easysimd__m128_to_private(easysimd_mm_sqrt_ps(a)).neon_f32, 0);
    r.neon_f32 = vsetq_lane_f32(value, a.neon_f32, 0);
    return r;
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 tmp;
    tmp.sve_f32 = svsqrt_f32_z(svptrue_b32(), a.sve_f32);
    a.f32[0] = tmp.f32[0];
    return a;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_sqrt_ps(a));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_sqrt_ps(easysimd_x_mm_broadcastlow_ps(a)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      float32_t value =
            vgetq_lane_f32(easysimd__m128_to_private(easysimd_mm_sqrt_ps(a)).neon_f32, 0);
      r_.neon_f32 = vsetq_lane_f32(value, a_.neon_f32, 0);
    #elif defined(easysimd_math_sqrtf)
      r_.f32[0] = easysimd_math_sqrtf(a_.f32[0]);
      r_.f32[1] = a_.f32[1];
      r_.f32[2] = a_.f32[2];
      r_.f32[3] = a_.f32[3];
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_sqrt_ss(a) easysimd_mm_sqrt_ss((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_store_ps (easysimd_float32 mem_addr[4], easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_store_ps(mem_addr, a);
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    svst1_f32(svptrue_b32(), &(mem_addr[0]), a.sve_f32);
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_f32(mem_addr, a.neon_f32);
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);
    easysimd_memcpy(mem_addr, &a_, sizeof(a));
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_store_ps(mem_addr, a) easysimd_mm_store_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float*, easysimd_float32*, mem_addr), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_store1_ps (easysimd_float32 mem_addr[4], easysimd__m128 a) {
  easysimd_float32* mem_addr_ = EASYSIMD_ALIGN_ASSUME_LIKE(mem_addr, easysimd__m128);
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_store_ps1(mem_addr_, a);
  #elif defined (EASYSIMD_ARM_SVE_NATIVE)
    svst1_f32(svptrue_b32(), &(mem_addr_[0]), svdup_n_f32(a.f32[0]));
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      vst1q_f32(mem_addr_, vdupq_lane_f32(vget_low_f32(a_.neon_f32), 0));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      easysimd__m128_private tmp_;
      tmp_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, a_.f32, 0, 0, 0, 0);
      easysimd_mm_store_ps(mem_addr_, tmp_);
    #else
      EASYSIMD_VECTORIZE_ALIGNED(mem_addr_:16)
      for (size_t i = 0 ; i < sizeof(a_.f32) / sizeof(a_.f32[0]) ; i++) {
        mem_addr_[i] = a_.f32[0];
      }
    #endif
  #endif
}
#define easysimd_mm_store_ps1(mem_addr, a) easysimd_mm_store1_ps(mem_addr, a)
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_store_ps1(mem_addr, a) easysimd_mm_store1_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float*, easysimd_float32*, mem_addr), (a))
#  define _mm_store1_ps(mem_addr, a) easysimd_mm_store1_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float*, easysimd_float32*, mem_addr), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_store_ss (easysimd_float32* mem_addr, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_store_ss(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    *mem_addr = a.f32[0];
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    vst1q_lane_f32(mem_addr, a.neon_f32, 0);
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    *mem_addr = a_.f32[0];
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_store_ss(mem_addr, a) easysimd_mm_store_ss(EASYSIMD_CHECKED_REINTERPRET_CAST(float*, easysimd_float32*, mem_addr), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storeh_pi (easysimd__m64* mem_addr, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_storeh_pi(HEDLEY_REINTERPRET_CAST(__m64*, mem_addr), a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_memcpy(mem_addr, &(a.f64[1]), sizeof(a.f64[1]));
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      vst1_f32(HEDLEY_REINTERPRET_CAST(float32_t*, mem_addr), vget_high_f32(a_.neon_f32));
    #else
      easysimd_memcpy(mem_addr, &(a_.m64[1]), sizeof(a_.m64[1]));
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_storeh_pi(mem_addr, a) easysimd_mm_storeh_pi(mem_addr, (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storel_pi (easysimd__m64* mem_addr, easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_storel_pi(HEDLEY_REINTERPRET_CAST(__m64*, mem_addr), a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_memcpy(mem_addr, &(a.f64[0]), sizeof(a.f64[0]));
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    mem_addr->neon_f32 = vget_low_f32(a.neon_f32);
  #else
    easysimd__m64_private* dest_ = HEDLEY_REINTERPRET_CAST(easysimd__m64_private*, mem_addr);
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    dest_->f32[0] = a_.f32[0];
    dest_->f32[1] = a_.f32[1];
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_storel_pi(mem_addr, a) easysimd_mm_storel_pi(mem_addr, (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storer_ps (easysimd_float32 mem_addr[4], easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_storer_ps(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f32(svptrue_b32(), &(mem_addr[0]), svrev_f32(a.sve_f32));
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      float32x4_t tmp = vrev64q_f32(a_.neon_f32);
      vst1q_f32(mem_addr, vextq_f32(tmp, tmp, 2));
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      a_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, a_.f32, 3, 2, 1, 0);
      easysimd_mm_store_ps(mem_addr, easysimd__m128_from_private(a_));
    #else
      EASYSIMD_VECTORIZE_ALIGNED(mem_addr:16)
      for (size_t i = 0 ; i < sizeof(a_.f32) / sizeof(a_.f32[0]) ; i++) {
        mem_addr[i] = a_.f32[((sizeof(a_.f32) / sizeof(a_.f32[0])) - 1) - i];
      }
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_storer_ps(mem_addr, a) easysimd_mm_storer_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float*, easysimd_float32*, mem_addr), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_storeu_ps (easysimd_float32 mem_addr[4], easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_storeu_ps(mem_addr, a);
  #else
    easysimd__m128_private a_ = easysimd__m128_to_private(a);

    #if defined(EASYSIMD_ARM_SVE_NATIVE)
      svst1_f32(svptrue_b32(), mem_addr, a_.sve_f32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      vst1q_f32(mem_addr, a_.neon_f32);
    #else
      easysimd_memcpy(mem_addr, &a_, sizeof(a_));
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_storeu_ps(mem_addr, a) easysimd_mm_storeu_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float*, easysimd_float32*, mem_addr), (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_sub_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sub_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsub_f32_z(svptrue_b32(), a.sve_f32, b.sve_f32);
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsubq_f32(a.neon_f32, b.neon_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
      r_.f32 = a_.f32 - b_.f32;
    #else
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.f32) / sizeof(r_.f32[0])) ; i++) {
        r_.f32[i] = a_.f32[i] - b_.f32[i];
      }
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_sub_ps(a, b) easysimd_mm_sub_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_sub_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_sub_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svsub_f32_z(svptrue_b32(), a.sve_f32, svdupq_n_f32(b.f32[0], 0.0, 0.0, 0.0));
    return r;
  #elif defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m128 r;
    r.neon_f32 = vsubq_f32(a.neon_f32, vsetq_lane_f32(vgetq_lane_f32(b.neon_f32, 0), vdupq_n_f32(0), 0));
    return r;
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0) && defined(EASYSIMD_FAST_EXCEPTIONS)
    return easysimd_mm_move_ss(a, easysimd_mm_sub_ps(a, b));
  #elif (EASYSIMD_NATURAL_VECTOR_SIZE > 0)
    return easysimd_mm_move_ss(a, easysimd_mm_sub_ps(easysimd_x_mm_broadcastlow_ps(a), easysimd_x_mm_broadcastlow_ps(b)));
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    r_.f32[0] = a_.f32[0] - b_.f32[0];
    r_.f32[1] = a_.f32[1];
    r_.f32[2] = a_.f32[2];
    r_.f32[3] = a_.f32[3];

    return easysimd__m128_from_private(r_);
  #endif
}

#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_sub_ss(a, b) easysimd_mm_sub_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomieq_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_ucomieq_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (fabsf(a.f32[0] - b.f32[0]) < 1e-9f);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_or_b_nan = vmvnq_u32(vandq_u32(a_not_nan, b_not_nan));
      uint32x4_t a_eq_b = vceqq_f32(a_.neon_f32, b_.neon_f32);
      r = !!(vgetq_lane_u32(vorrq_u32(a_or_b_nan, a_eq_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f);
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f);
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_ucomieq_ss(a, b) easysimd_mm_ucomieq_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomige_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_ucomige_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return ((a.f32[0] > b.f32[0]) || (fabsf(a.f32[0] - b.f32[0]) < 1e-9f));
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_and_b_not_nan = vandq_u32(a_not_nan, b_not_nan);
      uint32x4_t a_ge_b = vcgeq_f32(a_.neon_f32, b_.neon_f32);
      r = !!(vgetq_lane_u32(vandq_u32(a_and_b_not_nan, a_ge_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = ((a_.f32[0] > b_.f32[0]) || (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f));
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = ((a_.f32[0] > b_.f32[0]) || (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f));
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_ucomige_ss(a, b) easysimd_mm_ucomige_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomigt_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_ucomigt_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (a.f32[0] > b.f32[0]);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_and_b_not_nan = vandq_u32(a_not_nan, b_not_nan);
      uint32x4_t a_gt_b = vcgtq_f32(a_.neon_f32, b_.neon_f32);
      r = !!(vgetq_lane_u32(vandq_u32(a_and_b_not_nan, a_gt_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = a_.f32[0] > b_.f32[0];
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = a_.f32[0] > b_.f32[0];
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_ucomigt_ss(a, b) easysimd_mm_ucomigt_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomile_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_ucomile_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return ((a.f32[0] < b.f32[0]) || (fabsf(a.f32[0] - b.f32[0]) < 1e-9f));
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_or_b_nan = vmvnq_u32(vandq_u32(a_not_nan, b_not_nan));
      uint32x4_t a_le_b = vcleq_f32(a_.neon_f32, b_.neon_f32);
      r = !!(vgetq_lane_u32(vorrq_u32(a_or_b_nan, a_le_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = ((a_.f32[0] < b_.f32[0]) || (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f));
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = ((a_.f32[0] < b_.f32[0]) || (fabsf(a_.f32[0] - b_.f32[0]) < 1e-9f));
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_ucomile_ss(a, b) easysimd_mm_ucomile_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomilt_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_ucomilt_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (a.f32[0] < b.f32[0]);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_or_b_nan = vmvnq_u32(vandq_u32(a_not_nan, b_not_nan));
      uint32x4_t a_lt_b = vcltq_f32(a_.neon_f32, b_.neon_f32);
      r = !!(vgetq_lane_u32(vorrq_u32(a_or_b_nan, a_lt_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = a_.f32[0] < b_.f32[0];
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = a_.f32[0] < b_.f32[0];
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_ucomilt_ss(a, b) easysimd_mm_ucomilt_ss((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int
easysimd_mm_ucomineq_ss (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_ucomineq_ss(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    return (fabsf(a.f32[0] - b.f32[0]) > 1e-9f);
  #else
    easysimd__m128_private
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);
    int r;

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      uint32x4_t a_not_nan = vceqq_f32(a_.neon_f32, a_.neon_f32);
      uint32x4_t b_not_nan = vceqq_f32(b_.neon_f32, b_.neon_f32);
      uint32x4_t a_and_b_not_nan = vandq_u32(a_not_nan, b_not_nan);
      uint32x4_t a_neq_b = vmvnq_u32(vceqq_f32(a_.neon_f32, b_.neon_f32));
      r = !!(vgetq_lane_u32(vandq_u32(a_and_b_not_nan, a_neq_b), 0) != 0);
    #elif defined(EASYSIMD_HAVE_FENV_H)
      fenv_t envp;
      int x = feholdexcept(&envp);
      r = (fabsf(a_.f32[0] - b_.f32[0]) > 1e-9f);
      if (HEDLEY_LIKELY(x == 0))
        fesetenv(&envp);
    #else
      r = (fabsf(a_.f32[0] - b_.f32[0]) > 1e-9f);
    #endif

    return r;
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_ucomineq_ss(a, b) easysimd_mm_ucomineq_ss((a), (b))
#endif

#if defined(EASYSIMD_X86_SSE_NATIVE)
#  if defined(__has_builtin)
#    if __has_builtin(__builtin_ia32_undef128)
#      define EASYSIMD_HAVE_UNDEFINED128
#    endif
#  elif !defined(__PGI) && !defined(EASYSIMD_BUG_GCC_REV_208793) && !defined(_MSC_VER)
#    define EASYSIMD_HAVE_UNDEFINED128
#  endif
#endif

#if defined(EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_)
  HEDLEY_DIAGNOSTIC_PUSH
  EASYSIMD_DIAGNOSTIC_DISABLE_UNINITIALIZED_
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_unpackhi_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_unpackhi_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svzip2_f32(a.sve_f32, b.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f32 = vzip2q_f32(a_.neon_f32, b_.neon_f32);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      float32x2_t a1 = vget_high_f32(a_.neon_f32);
      float32x2_t b1 = vget_high_f32(b_.neon_f32);
      float32x2x2_t result = vzip_f32(a1, b1);
      r_.neon_f32 = vcombine_f32(result.val[0], result.val[1]);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, b_.f32, 2, 6, 3, 7);
    #else
      r_.f32[0] = a_.f32[2];
      r_.f32[1] = b_.f32[2];
      r_.f32[2] = a_.f32[3];
      r_.f32[3] = b_.f32[3];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_unpackhi_ps(a, b) easysimd_mm_unpackhi_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__m128
easysimd_mm_unpacklo_ps (easysimd__m128 a, easysimd__m128 b) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    return _mm_unpacklo_ps(a, b);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd__m128 r;
    r.sve_f32 = svzip1_f32(a.sve_f32, b.sve_f32);
    return r;
  #else
    easysimd__m128_private
      r_,
      a_ = easysimd__m128_to_private(a),
      b_ = easysimd__m128_to_private(b);

    #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
      r_.neon_f32 = vzip1q_f32(a_.neon_f32, b_.neon_f32);
    #elif defined(EASYSIMD_SHUFFLE_VECTOR_)
      r_.f32 = EASYSIMD_SHUFFLE_VECTOR_(32, 16, a_.f32, b_.f32, 0, 4, 1, 5);
    #elif defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      float32x2_t a1 = vget_low_f32(a_.neon_f32);
      float32x2_t b1 = vget_low_f32(b_.neon_f32);
      float32x2x2_t result = vzip_f32(a1, b1);
      r_.neon_f32 = vcombine_f32(result.val[0], result.val[1]);
    #else
      r_.f32[0] = a_.f32[0];
      r_.f32[1] = b_.f32[0];
      r_.f32[2] = a_.f32[1];
      r_.f32[3] = b_.f32[1];
    #endif

    return easysimd__m128_from_private(r_);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_unpacklo_ps(a, b) easysimd_mm_unpacklo_ps((a), (b))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_stream_pi (easysimd__m64* mem_addr, easysimd__m64 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    _mm_stream_pi(HEDLEY_REINTERPRET_CAST(__m64*, mem_addr), a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    easysimd_memcpy(mem_addr, &(a.i64[0]), sizeof(a.i64[0]));
  #else
    easysimd__m64_private*
      dest = HEDLEY_REINTERPRET_CAST(easysimd__m64_private*, mem_addr),
      a_ = easysimd__m64_to_private(a);

    #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
      dest->i64[0] = vget_lane_s64(a_.neon_i64, 0);
    #else
      dest->i64[0] = a_.i64[0];
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_stream_pi(mem_addr, a) easysimd_mm_stream_pi(mem_addr, (a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_stream_ps (easysimd_float32 mem_addr[4], easysimd__m128 a) {
  #if defined(EASYSIMD_X86_SSE_NATIVE)
    _mm_stream_ps(mem_addr, a);
  #elif defined(EASYSIMD_ARM_SVE_NATIVE)
    svst1_f32(svptrue_b32(), mem_addr, a.sve_f32);
  #elif HEDLEY_HAS_BUILTIN(__builtin_nontemporal_store) && defined(EASYSIMD_VECTOR_SUBSCRIPT_OPS)
    easysimd__m128_private a_ = easysimd__m128_to_private(a);
    __builtin_nontemporal_store(a_.f32, EASYSIMD_ALIGN_CAST(__typeof__(a_.f32)*, mem_addr));
  #else
    easysimd_mm_store_ps(mem_addr, a);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_stream_ps(mem_addr, a) easysimd_mm_stream_ps(EASYSIMD_CHECKED_REINTERPRET_CAST(float*, easysimd_float32*, mem_addr), (a))
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #define EASYSIMD_MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
    do { \
          float32x4x2_t EASYSIMD_MM_TRANSPOSE4_PS_ROW01 = vtrnq_f32(row0, row1); \
          float32x4x2_t EASYSIMD_MM_TRANSPOSE4_PS_ROW23 = vtrnq_f32(row2, row3); \
          row0 = vcombine_f32(vget_low_f32(EASYSIMD_MM_TRANSPOSE4_PS_ROW01.val[0]), \
                              vget_low_f32(EASYSIMD_MM_TRANSPOSE4_PS_ROW23.val[0])); \
          row1 = vcombine_f32(vget_low_f32(EASYSIMD_MM_TRANSPOSE4_PS_ROW01.val[1]), \
                              vget_low_f32(EASYSIMD_MM_TRANSPOSE4_PS_ROW23.val[1])); \
          row2 = vcombine_f32(vget_high_f32(EASYSIMD_MM_TRANSPOSE4_PS_ROW01.val[0]), \
                              vget_high_f32(EASYSIMD_MM_TRANSPOSE4_PS_ROW23.val[0])); \
          row3 = vcombine_f32(vget_high_f32(EASYSIMD_MM_TRANSPOSE4_PS_ROW01.val[1]), \
                              vget_high_f32(EASYSIMD_MM_TRANSPOSE4_PS_ROW23.val[1])); \
      } while (0)
#else
  #define EASYSIMD_MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
    do { \
      easysimd__m128 EASYSIMD_MM_TRANSPOSE4_PS_tmp3, EASYSIMD_MM_TRANSPOSE4_PS_tmp2, EASYSIMD_MM_TRANSPOSE4_PS_tmp1, EASYSIMD_MM_TRANSPOSE4_PS_tmp0; \
      EASYSIMD_MM_TRANSPOSE4_PS_tmp0 = easysimd_mm_unpacklo_ps((row0), (row1)); \
      EASYSIMD_MM_TRANSPOSE4_PS_tmp2 = easysimd_mm_unpacklo_ps((row2), (row3)); \
      EASYSIMD_MM_TRANSPOSE4_PS_tmp1 = easysimd_mm_unpackhi_ps((row0), (row1)); \
      EASYSIMD_MM_TRANSPOSE4_PS_tmp3 = easysimd_mm_unpackhi_ps((row2), (row3)); \
      row0 = easysimd_mm_movelh_ps(EASYSIMD_MM_TRANSPOSE4_PS_tmp0, EASYSIMD_MM_TRANSPOSE4_PS_tmp2); \
      row1 = easysimd_mm_movehl_ps(EASYSIMD_MM_TRANSPOSE4_PS_tmp2, EASYSIMD_MM_TRANSPOSE4_PS_tmp0); \
      row2 = easysimd_mm_movelh_ps(EASYSIMD_MM_TRANSPOSE4_PS_tmp1, EASYSIMD_MM_TRANSPOSE4_PS_tmp3); \
      row3 = easysimd_mm_movehl_ps(EASYSIMD_MM_TRANSPOSE4_PS_tmp3, EASYSIMD_MM_TRANSPOSE4_PS_tmp1); \
    } while (0)
#endif
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) EASYSIMD_MM_TRANSPOSE4_PS(row0, row1, row2, row3)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void*
easysimd_mm_malloc(size_t size, size_t align) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    return _mm_malloc(size, align);
  #else
    #if defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
    void *ptr;
    if (align == 1)
        return malloc (size);
    if (align == 2 || (sizeof (void *) == 8 && align == 4))
        align = sizeof (void *);
    if (posix_memalign(&ptr, align, size) == 0)
        return ptr;
    else
        return NULL;
    #else
        return malloc (size);
    #endif
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_malloc(size, align) easysimd_mm_malloc((size), (align))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
void
easysimd_mm_free(void * mem_addr) {
  #if defined(EASYSIMD_X86_SSE_NATIVE) && defined(EASYSIMD_X86_MMX_NATIVE)
    _mm_free(mem_addr);
  #else
    free(mem_addr);
  #endif
}
#if defined(EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES)
#  define _mm_free(mem_addr) easysimd_mm_free((mem_addr))
#endif

EASYSIMD_END_DECLS_

HEDLEY_DIAGNOSTIC_POP

#ifdef EASYSIMD_ENABLE_TEST_PERF
#pragma GCC pop_options
#endif
#endif /* !defined(EASYSIMD_X86_SSE_H) */
