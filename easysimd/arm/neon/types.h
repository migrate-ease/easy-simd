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

#if !defined(EASYSIMD_ARM_NEON_TYPES_H)
#define EASYSIMD_ARM_NEON_TYPES_H

#include "../../easysimd-common.h"
#include "../../easysimd-f16.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

#if defined(EASYSIMD_VECTOR_SUBSCRIPT)
  #define EASYSIMD_ARM_NEON_DECLARE_VECTOR(Element_Type, Name, Vector_Size) Element_Type Name EASYSIMD_VECTOR(Vector_Size)
#else
  #define EASYSIMD_ARM_NEON_DECLARE_VECTOR(Element_Type, Name, Vector_Size) Element_Type Name[(Vector_Size) / sizeof(Element_Type)]
#endif

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(int8_t, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_int8x8_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(int16_t, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_int16x4_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(int32_t, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_int32x2_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(int64_t, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_int64x1_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(uint8_t, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_uint8x8_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(uint16_t, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_uint16x4_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(uint32_t, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_uint32x2_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(uint64_t, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_uint64x1_private;

typedef union {
  #if EASYSIMD_FLOAT16_API != EASYSIMD_FLOAT16_API_PORTABLE && EASYSIMD_FLOAT16_API != EASYSIMD_FLOAT16_API_FP16_NO_ABI
    EASYSIMD_ARM_NEON_DECLARE_VECTOR(easysimd_float16, values, 8);
  #else
    easysimd_float16 values[4];
  #endif

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_float16x4_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(easysimd_float32, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_float32x2_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(easysimd_float64, values, 8);

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    __m64 m64;
  #endif
} easysimd_float64x1_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(int8_t, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m128i;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int8x16_t neon;
  #endif
} easysimd_int8x16_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(int16_t, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m128i;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int16x8_t neon;
  #endif
} easysimd_int16x8_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(int32_t, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m128i;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x4_t neon;
  #endif
} easysimd_int32x4_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(int64_t, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m128i;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int64x2_t neon;
  #endif
} easysimd_int64x2_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(uint8_t, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m128i;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int8x16_t neon;
  #endif
} easysimd_uint8x16_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(uint16_t, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m128i;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int16x8_t neon;
  #endif
} easysimd_uint16x8_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(uint32_t, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m128i;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x4_t neon;
  #endif
} easysimd_uint32x4_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(uint64_t, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128i m128i;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int64x2_t neon;
  #endif
} easysimd_uint64x2_private;

typedef union {
  #if EASYSIMD_FLOAT16_API != EASYSIMD_FLOAT16_API_PORTABLE && EASYSIMD_FLOAT16_API != EASYSIMD_FLOAT16_API_FP16_NO_ABI
    EASYSIMD_ARM_NEON_DECLARE_VECTOR(easysimd_float16, values, 16);
  #else
    easysimd_float16 values[8];
  #endif

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128 m128;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x4_t neon;
  #endif
} easysimd_float16x8_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(easysimd_float32, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128 m128;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int32x4_t neon;
  #endif
} easysimd_float32x4_private;

typedef union {
  EASYSIMD_ARM_NEON_DECLARE_VECTOR(easysimd_float64, values, 16);

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    __m128d m128d;
  #endif

  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    int64x2_t neon;
  #endif
} easysimd_float64x2_private;

#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  typedef     float32_t     easysimd_float32_t;

  typedef      int8x8_t      easysimd_int8x8_t;
  typedef     int16x4_t     easysimd_int16x4_t;
  typedef     int32x2_t     easysimd_int32x2_t;
  typedef     int64x1_t     easysimd_int64x1_t;
  typedef     uint8x8_t     easysimd_uint8x8_t;
  typedef    uint16x4_t    easysimd_uint16x4_t;
  typedef    uint32x2_t    easysimd_uint32x2_t;
  typedef    uint64x1_t    easysimd_uint64x1_t;
  typedef   float32x2_t   easysimd_float32x2_t;

  typedef     int8x16_t     easysimd_int8x16_t;
  typedef     int16x8_t     easysimd_int16x8_t;
  typedef     int32x4_t     easysimd_int32x4_t;
  typedef     int64x2_t     easysimd_int64x2_t;
  typedef    uint8x16_t    easysimd_uint8x16_t;
  typedef    uint16x8_t    easysimd_uint16x8_t;
  typedef    uint32x4_t    easysimd_uint32x4_t;
  typedef    uint64x2_t    easysimd_uint64x2_t;
  typedef   float32x4_t   easysimd_float32x4_t;

  typedef    int8x8x2_t    easysimd_int8x8x2_t;
  typedef   int16x4x2_t   easysimd_int16x4x2_t;
  typedef   int32x2x2_t   easysimd_int32x2x2_t;
  typedef   int64x1x2_t   easysimd_int64x1x2_t;
  typedef   uint8x8x2_t   easysimd_uint8x8x2_t;
  typedef  uint16x4x2_t  easysimd_uint16x4x2_t;
  typedef  uint32x2x2_t  easysimd_uint32x2x2_t;
  typedef  uint64x1x2_t  easysimd_uint64x1x2_t;
  typedef float32x2x2_t easysimd_float32x2x2_t;

  typedef   int8x16x2_t   easysimd_int8x16x2_t;
  typedef   int16x8x2_t   easysimd_int16x8x2_t;
  typedef   int32x4x2_t   easysimd_int32x4x2_t;
  typedef   int64x2x2_t   easysimd_int64x2x2_t;
  typedef  uint8x16x2_t  easysimd_uint8x16x2_t;
  typedef  uint16x8x2_t  easysimd_uint16x8x2_t;
  typedef  uint32x4x2_t  easysimd_uint32x4x2_t;
  typedef  uint64x2x2_t  easysimd_uint64x2x2_t;
  typedef float32x4x2_t easysimd_float32x4x2_t;

  typedef    int8x8x3_t    easysimd_int8x8x3_t;
  typedef   int16x4x3_t   easysimd_int16x4x3_t;
  typedef   int32x2x3_t   easysimd_int32x2x3_t;
  typedef   int64x1x3_t   easysimd_int64x1x3_t;
  typedef   uint8x8x3_t   easysimd_uint8x8x3_t;
  typedef  uint16x4x3_t  easysimd_uint16x4x3_t;
  typedef  uint32x2x3_t  easysimd_uint32x2x3_t;
  typedef  uint64x1x3_t  easysimd_uint64x1x3_t;
  typedef float32x2x3_t easysimd_float32x2x3_t;

  typedef   int8x16x3_t   easysimd_int8x16x3_t;
  typedef   int16x8x3_t   easysimd_int16x8x3_t;
  typedef   int32x4x3_t   easysimd_int32x4x3_t;
  typedef   int64x2x3_t   easysimd_int64x2x3_t;
  typedef  uint8x16x3_t  easysimd_uint8x16x3_t;
  typedef  uint16x8x3_t  easysimd_uint16x8x3_t;
  typedef  uint32x4x3_t  easysimd_uint32x4x3_t;
  typedef  uint64x2x3_t  easysimd_uint64x2x3_t;
  typedef float32x4x3_t easysimd_float32x4x3_t;

  typedef    int8x8x4_t    easysimd_int8x8x4_t;
  typedef   int16x4x4_t   easysimd_int16x4x4_t;
  typedef   int32x2x4_t   easysimd_int32x2x4_t;
  typedef   int64x1x4_t   easysimd_int64x1x4_t;
  typedef   uint8x8x4_t   easysimd_uint8x8x4_t;
  typedef  uint16x4x4_t  easysimd_uint16x4x4_t;
  typedef  uint32x2x4_t  easysimd_uint32x2x4_t;
  typedef  uint64x1x4_t  easysimd_uint64x1x4_t;
  typedef float32x2x4_t easysimd_float32x2x4_t;

  typedef   int8x16x4_t   easysimd_int8x16x4_t;
  typedef   int16x8x4_t   easysimd_int16x8x4_t;
  typedef   int32x4x4_t   easysimd_int32x4x4_t;
  typedef   int64x2x4_t   easysimd_int64x2x4_t;
  typedef  uint8x16x4_t  easysimd_uint8x16x4_t;
  typedef  uint16x8x4_t  easysimd_uint16x8x4_t;
  typedef  uint32x4x4_t  easysimd_uint32x4x4_t;
  typedef  uint64x2x4_t  easysimd_uint64x2x4_t;
  typedef float32x4x4_t easysimd_float32x4x4_t;

  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    typedef     float64_t     easysimd_float64_t;
    typedef   float64x1_t   easysimd_float64x1_t;
    typedef   float64x2_t   easysimd_float64x2_t;
    typedef float64x1x2_t easysimd_float64x1x2_t;
    typedef float64x2x2_t easysimd_float64x2x2_t;
    typedef float64x1x3_t easysimd_float64x1x3_t;
    typedef float64x2x3_t easysimd_float64x2x3_t;
    typedef float64x1x4_t easysimd_float64x1x4_t;
    typedef float64x2x4_t easysimd_float64x2x4_t;
  #else
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X1
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X2
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X1XN
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X2XN
  #endif

  #if EASYSIMD_FLOAT16_API == EASYSIMD_FLOAT16_API_FP16
    typedef     float16_t     easysimd_float16_t;
    typedef   float16x4_t   easysimd_float16x4_t;
    typedef   float16x8_t   easysimd_float16x8_t;
  #else
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F16
  #endif
#elif (defined(EASYSIMD_X86_MMX_NATIVE) || defined(EASYSIMD_X86_SSE_NATIVE)) && defined(EASYSIMD_ARM_NEON_FORCE_NATIVE_TYPES)
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F32
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64

  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_VXN
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X1XN
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X2XN

  #if defined(EASYSIMD_X86_MMX_NATIVE)
    typedef __m64    easysimd_int8x8_t;
    typedef __m64   easysimd_int16x4_t;
    typedef __m64   easysimd_int32x2_t;
    typedef __m64   easysimd_int64x1_t;
    typedef __m64   easysimd_uint8x8_t;
    typedef __m64  easysimd_uint16x4_t;
    typedef __m64  easysimd_uint32x2_t;
    typedef __m64  easysimd_uint64x1_t;
    typedef __m64 easysimd_float32x2_t;
    typedef __m64 easysimd_float64x1_t;
  #else
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_I8X8
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_I16X4
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_I32X2
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_I64X1
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_U8X8
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_U16X4
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_U32X2
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_U64X1
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F32X2
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X1
  #endif

  #if defined(EASYSIMD_X86_SSE_NATIVE)
    typedef __m128 easysimd_float32x4_t;
  #else
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F32X4
  #endif

  #if defined(EASYSIMD_X86_SSE2_NATIVE)
    typedef  __m128i  easysimd_int8x16_t;
    typedef  __m128i  easysimd_int16x8_t;
    typedef  __m128i  easysimd_int32x4_t;
    typedef  __m128i  easysimd_int64x2_t;
    typedef __m128i  easysimd_uint8x16_t;
    typedef __m128i  easysimd_uint16x8_t;
    typedef __m128i  easysimd_uint32x4_t;
    typedef __m128i  easysimd_uint64x2_t;
    typedef __m128d easysimd_float64x2_t;
  #else
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_I8X16
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_I16X8
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_I32X4
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_I64X2
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_U8X16
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_U16X8
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_U32X4
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_U64X2
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X2
  #endif

  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F16
#elif defined(EASYSIMD_VECTOR)
  typedef easysimd_float32 easysimd_float32_t;
  typedef easysimd_float64 easysimd_float64_t;
  typedef int8_t          easysimd_int8x8_t    EASYSIMD_VECTOR(8);
  typedef int16_t         easysimd_int16x4_t   EASYSIMD_VECTOR(8);
  typedef int32_t         easysimd_int32x2_t   EASYSIMD_VECTOR(8);
  typedef int64_t         easysimd_int64x1_t   EASYSIMD_VECTOR(8);
  typedef uint8_t         easysimd_uint8x8_t   EASYSIMD_VECTOR(8);
  typedef uint16_t        easysimd_uint16x4_t  EASYSIMD_VECTOR(8);
  typedef uint32_t        easysimd_uint32x2_t  EASYSIMD_VECTOR(8);
  typedef uint64_t        easysimd_uint64x1_t  EASYSIMD_VECTOR(8);
  typedef easysimd_float32_t easysimd_float32x2_t EASYSIMD_VECTOR(8);
  typedef easysimd_float64_t easysimd_float64x1_t EASYSIMD_VECTOR(8);
  typedef int8_t          easysimd_int8x16_t   EASYSIMD_VECTOR(16);
  typedef int16_t         easysimd_int16x8_t   EASYSIMD_VECTOR(16);
  typedef int32_t         easysimd_int32x4_t   EASYSIMD_VECTOR(16);
  typedef int64_t         easysimd_int64x2_t   EASYSIMD_VECTOR(16);
  typedef uint8_t         easysimd_uint8x16_t  EASYSIMD_VECTOR(16);
  typedef uint16_t        easysimd_uint16x8_t  EASYSIMD_VECTOR(16);
  typedef uint32_t        easysimd_uint32x4_t  EASYSIMD_VECTOR(16);
  typedef uint64_t        easysimd_uint64x2_t  EASYSIMD_VECTOR(16);
  typedef easysimd_float32_t easysimd_float32x4_t EASYSIMD_VECTOR(16);
  typedef easysimd_float64_t easysimd_float64x2_t EASYSIMD_VECTOR(16);

  #if defined(EASYSIMD_ARM_NEON_FP16)
    typedef easysimd_float16 easysimd_float16_t;
    typedef easysimd_float16_t easysimd_float16x4_t EASYSIMD_VECTOR(8);
    typedef easysimd_float16_t easysimd_float16x8_t EASYSIMD_VECTOR(16);
  #else
    #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F16
  #endif

  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_VXN
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X1XN
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X2XN
#else
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F16
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F32
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT

  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_VXN
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X1XN
  #define EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X2XN
#endif

#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_I8X8) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_int8x8_private easysimd_int8x8_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_I16X4) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_int16x4_private easysimd_int16x4_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_I32X2) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_int32x2_private easysimd_int32x2_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_I64X1) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_int64x1_private easysimd_int64x1_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_U8X8) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_uint8x8_private easysimd_uint8x8_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_U16X4) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_uint16x4_private easysimd_uint16x4_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_U32X2) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_uint32x2_private easysimd_uint32x2_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_U64X1) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_uint64x1_private easysimd_uint64x1_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_F32X2) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_float32x2_private easysimd_float32x2_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X1) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_64BIT)
  typedef easysimd_float64x1_private easysimd_float64x1_t;
#endif

#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_I8X16) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_int8x16_private easysimd_int8x16_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_I16X8) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_int16x8_private easysimd_int16x8_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_I32X4) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_int32x4_private easysimd_int32x4_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_I64X2) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_int64x2_private easysimd_int64x2_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_U8X16) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_uint8x16_private easysimd_uint8x16_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_U16X8) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_uint16x8_private easysimd_uint16x8_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_U32X4) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_uint32x4_private easysimd_uint32x4_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_U64X2) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_uint64x2_private easysimd_uint64x2_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_F32X4) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_float32x4_private easysimd_float32x4_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X2) || defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_128BIT)
  typedef easysimd_float64x2_private easysimd_float64x2_t;
#endif

#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_F16)
  typedef easysimd_float16 easysimd_float16_t;
  typedef easysimd_float16x4_private easysimd_float16x4_t;
  typedef easysimd_float16x8_private easysimd_float16x8_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_F32)
  typedef easysimd_float32 easysimd_float32_t;
#endif
#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_F64)
  typedef easysimd_float64 easysimd_float64_t;
#endif

#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_VXN) && !defined(EASYSIMD_BUG_INTEL_857088)
  typedef struct    easysimd_int8x8x2_t {
    easysimd_int8x8_t val[2];
  } easysimd_int8x8x2_t;
  typedef struct   easysimd_int16x4x2_t {
    easysimd_int16x4_t val[2];
  } easysimd_int16x4x2_t;
  typedef struct   easysimd_int32x2x2_t {
    easysimd_int32x2_t val[2];
  } easysimd_int32x2x2_t;
  typedef struct   easysimd_int64x1x2_t {
    easysimd_int64x1_t val[2];
  } easysimd_int64x1x2_t;
  typedef struct   easysimd_uint8x8x2_t {
    easysimd_uint8x8_t val[2];
  } easysimd_uint8x8x2_t;
  typedef struct  easysimd_uint16x4x2_t {
    easysimd_uint16x4_t val[2];
  } easysimd_uint16x4x2_t;
  typedef struct  easysimd_uint32x2x2_t {
    easysimd_uint32x2_t val[2];
  } easysimd_uint32x2x2_t;
  typedef struct  easysimd_uint64x1x2_t {
    easysimd_uint64x1_t val[2];
  } easysimd_uint64x1x2_t;
  typedef struct easysimd_float32x2x2_t {
    easysimd_float32x2_t val[2];
  } easysimd_float32x2x2_t;

  typedef struct   easysimd_int8x16x2_t {
    easysimd_int8x16_t val[2];
  } easysimd_int8x16x2_t;
  typedef struct   easysimd_int16x8x2_t {
    easysimd_int16x8_t val[2];
  } easysimd_int16x8x2_t;
  typedef struct   easysimd_int32x4x2_t {
    easysimd_int32x4_t val[2];
  } easysimd_int32x4x2_t;
  typedef struct   easysimd_int64x2x2_t {
    easysimd_int64x2_t val[2];
  } easysimd_int64x2x2_t;
  typedef struct  easysimd_uint8x16x2_t {
    easysimd_uint8x16_t val[2];
  } easysimd_uint8x16x2_t;
  typedef struct  easysimd_uint16x8x2_t {
    easysimd_uint16x8_t val[2];
  } easysimd_uint16x8x2_t;
  typedef struct  easysimd_uint32x4x2_t {
    easysimd_uint32x4_t val[2];
  } easysimd_uint32x4x2_t;
  typedef struct  easysimd_uint64x2x2_t {
    easysimd_uint64x2_t val[2];
  } easysimd_uint64x2x2_t;
  typedef struct easysimd_float32x4x2_t {
    easysimd_float32x4_t val[2];
  } easysimd_float32x4x2_t;

  typedef struct    easysimd_int8x8x3_t {
    easysimd_int8x8_t val[3];
  } easysimd_int8x8x3_t;
  typedef struct   easysimd_int16x4x3_t {
    easysimd_int16x4_t val[3];
  } easysimd_int16x4x3_t;
  typedef struct   easysimd_int32x2x3_t {
    easysimd_int32x2_t val[3];
  } easysimd_int32x2x3_t;
  typedef struct   easysimd_int64x1x3_t {
    easysimd_int64x1_t val[3];
  } easysimd_int64x1x3_t;
  typedef struct   easysimd_uint8x8x3_t {
    easysimd_uint8x8_t val[3];
  } easysimd_uint8x8x3_t;
  typedef struct  easysimd_uint16x4x3_t {
    easysimd_uint16x4_t val[3];
  } easysimd_uint16x4x3_t;
  typedef struct  easysimd_uint32x2x3_t {
    easysimd_uint32x2_t val[3];
  } easysimd_uint32x2x3_t;
  typedef struct  easysimd_uint64x1x3_t {
    easysimd_uint64x1_t val[3];
  } easysimd_uint64x1x3_t;
  typedef struct easysimd_float32x2x3_t {
    easysimd_float32x2_t val[3];
  } easysimd_float32x2x3_t;

  typedef struct   easysimd_int8x16x3_t {
    easysimd_int8x16_t val[3];
  } easysimd_int8x16x3_t;
  typedef struct   easysimd_int16x8x3_t {
    easysimd_int16x8_t val[3];
  } easysimd_int16x8x3_t;
  typedef struct   easysimd_int32x4x3_t {
    easysimd_int32x4_t val[3];
  } easysimd_int32x4x3_t;
  typedef struct   easysimd_int64x2x3_t {
    easysimd_int64x2_t val[3];
  } easysimd_int64x2x3_t;
  typedef struct  easysimd_uint8x16x3_t {
    easysimd_uint8x16_t val[3];
  } easysimd_uint8x16x3_t;
  typedef struct  easysimd_uint16x8x3_t {
    easysimd_uint16x8_t val[3];
  } easysimd_uint16x8x3_t;
  typedef struct  easysimd_uint32x4x3_t {
    easysimd_uint32x4_t val[3];
  } easysimd_uint32x4x3_t;
  typedef struct  easysimd_uint64x2x3_t {
    easysimd_uint64x2_t val[3];
  } easysimd_uint64x2x3_t;
  typedef struct easysimd_float32x4x3_t {
    easysimd_float32x4_t val[3];
  } easysimd_float32x4x3_t;

  typedef struct    easysimd_int8x8x4_t {
    easysimd_int8x8_t val[4];
  } easysimd_int8x8x4_t;
  typedef struct   easysimd_int16x4x4_t {
    easysimd_int16x4_t val[4];
  } easysimd_int16x4x4_t;
  typedef struct   easysimd_int32x2x4_t {
    easysimd_int32x2_t val[4];
  } easysimd_int32x2x4_t;
  typedef struct   easysimd_int64x1x4_t {
    easysimd_int64x1_t val[4];
  } easysimd_int64x1x4_t;
  typedef struct   easysimd_uint8x8x4_t {
    easysimd_uint8x8_t val[4];
  } easysimd_uint8x8x4_t;
  typedef struct  easysimd_uint16x4x4_t {
    easysimd_uint16x4_t val[4];
  } easysimd_uint16x4x4_t;
  typedef struct  easysimd_uint32x2x4_t {
    easysimd_uint32x2_t val[4];
  } easysimd_uint32x2x4_t;
  typedef struct  easysimd_uint64x1x4_t {
    easysimd_uint64x1_t val[4];
  } easysimd_uint64x1x4_t;
  typedef struct easysimd_float32x2x4_t {
    easysimd_float32x2_t val[4];
  } easysimd_float32x2x4_t;

  typedef struct   easysimd_int8x16x4_t {
    easysimd_int8x16_t val[4];
  } easysimd_int8x16x4_t;
  typedef struct   easysimd_int16x8x4_t {
    easysimd_int16x8_t val[4];
  } easysimd_int16x8x4_t;
  typedef struct   easysimd_int32x4x4_t {
    easysimd_int32x4_t val[4];
  } easysimd_int32x4x4_t;
  typedef struct   easysimd_int64x2x4_t {
    easysimd_int64x2_t val[4];
  } easysimd_int64x2x4_t;
  typedef struct  easysimd_uint8x16x4_t {
    easysimd_uint8x16_t val[4];
  } easysimd_uint8x16x4_t;
  typedef struct  easysimd_uint16x8x4_t {
    easysimd_uint16x8_t val[4];
  } easysimd_uint16x8x4_t;
  typedef struct  easysimd_uint32x4x4_t {
    easysimd_uint32x4_t val[4];
  } easysimd_uint32x4x4_t;
  typedef struct  easysimd_uint64x2x4_t {
    easysimd_uint64x2_t val[4];
  } easysimd_uint64x2x4_t;
  typedef struct easysimd_float32x4x4_t {
    easysimd_float32x4_t val[4];
  } easysimd_float32x4x4_t;
#endif

#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X1XN)
  typedef struct   easysimd_float64x1x2_t {
    easysimd_float64x1_t val[2];
  } easysimd_float64x1x2_t;

  typedef struct   easysimd_float64x1x3_t {
    easysimd_float64x1_t val[3];
  } easysimd_float64x1x3_t;

  typedef struct   easysimd_float64x1x4_t {
    easysimd_float64x1_t val[4];
  } easysimd_float64x1x4_t;
#endif

#if defined(EASYSIMD_ARM_NEON_NEED_PORTABLE_F64X2XN)
  typedef struct   easysimd_float64x2x2_t {
    easysimd_float64x2_t val[2];
  } easysimd_float64x2x2_t;

 typedef struct   easysimd_float64x2x3_t {
   easysimd_float64x2_t val[3];
 } easysimd_float64x2x3_t;

 typedef struct   easysimd_float64x2x4_t {
   easysimd_float64x2_t val[4];
 } easysimd_float64x2x4_t;
#endif

#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES) || defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  typedef   easysimd_float16_t     float16_t;
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  typedef   easysimd_float32_t     float32_t;

  typedef    easysimd_int8x8_t      int8x8_t;
  typedef   easysimd_int16x4_t     int16x4_t;
  typedef   easysimd_int32x2_t     int32x2_t;
  typedef   easysimd_int64x1_t     int64x1_t;
  typedef   easysimd_uint8x8_t     uint8x8_t;
  typedef  easysimd_uint16x4_t    uint16x4_t;
  typedef  easysimd_uint32x2_t    uint32x2_t;
  typedef  easysimd_uint64x1_t    uint64x1_t;
  typedef easysimd_float32x2_t   float32x2_t;

  typedef   easysimd_int8x16_t     int8x16_t;
  typedef   easysimd_int16x8_t     int16x8_t;
  typedef   easysimd_int32x4_t     int32x4_t;
  typedef   easysimd_int64x2_t     int64x2_t;
  typedef  easysimd_uint8x16_t    uint8x16_t;
  typedef  easysimd_uint16x8_t    uint16x8_t;
  typedef  easysimd_uint32x4_t    uint32x4_t;
  typedef  easysimd_uint64x2_t    uint64x2_t;
  typedef easysimd_float32x4_t   float32x4_t;

  typedef  easysimd_int8x8x2_t    int8x8x2_t;
  typedef easysimd_int16x4x2_t   int16x4x2_t;
  typedef easysimd_int32x2x2_t   int32x2x2_t;
  typedef easysimd_int64x1x2_t   int64x1x2_t;
  typedef easysimd_uint8x8x2_t   uint8x8x2_t;
  typedef easysimd_uint16x4x2_t  uint16x4x2_t;
  typedef easysimd_uint32x2x2_t  uint32x2x2_t;
  typedef easysimd_uint64x1x2_t  uint64x1x2_t;
  typedef easysimd_float32x2x2_t float32x2x2_t;

  typedef easysimd_int8x16x2_t   int8x16x2_t;
  typedef easysimd_int16x8x2_t   int16x8x2_t;
  typedef easysimd_int32x4x2_t   int32x4x2_t;
  typedef easysimd_int64x2x2_t   int64x2x2_t;
  typedef easysimd_uint8x16x2_t  uint8x16x2_t;
  typedef easysimd_uint16x8x2_t  uint16x8x2_t;
  typedef easysimd_uint32x4x2_t  uint32x4x2_t;
  typedef easysimd_uint64x2x2_t  uint64x2x2_t;
  typedef easysimd_float32x4x2_t float32x4x2_t;

  typedef  easysimd_int8x8x3_t    int8x8x3_t;
  typedef easysimd_int16x4x3_t   int16x4x3_t;
  typedef easysimd_int32x2x3_t   int32x2x3_t;
  typedef easysimd_int64x1x3_t   int64x1x3_t;
  typedef easysimd_uint8x8x3_t   uint8x8x3_t;
  typedef easysimd_uint16x4x3_t  uint16x4x3_t;
  typedef easysimd_uint32x2x3_t  uint32x2x3_t;
  typedef easysimd_uint64x1x3_t  uint64x1x3_t;
  typedef easysimd_float32x2x3_t float32x2x3_t;

  typedef easysimd_int8x16x3_t   int8x16x3_t;
  typedef easysimd_int16x8x3_t   int16x8x3_t;
  typedef easysimd_int32x4x3_t   int32x4x3_t;
  typedef easysimd_int64x2x3_t   int64x2x3_t;
  typedef easysimd_uint8x16x3_t  uint8x16x3_t;
  typedef easysimd_uint16x8x3_t  uint16x8x3_t;
  typedef easysimd_uint32x4x3_t  uint32x4x3_t;
  typedef easysimd_uint64x2x3_t  uint64x2x3_t;
  typedef easysimd_float32x4x3_t float32x4x3_t;

  typedef  easysimd_int8x8x4_t    int8x8x4_t;
  typedef easysimd_int16x4x4_t   int16x4x4_t;
  typedef easysimd_int32x2x4_t   int32x2x4_t;
  typedef easysimd_int64x1x4_t   int64x1x4_t;
  typedef easysimd_uint8x8x4_t   uint8x8x4_t;
  typedef easysimd_uint16x4x4_t  uint16x4x4_t;
  typedef easysimd_uint32x2x4_t  uint32x2x4_t;
  typedef easysimd_uint64x1x4_t  uint64x1x4_t;
  typedef easysimd_float32x2x4_t float32x2x4_t;

  typedef easysimd_int8x16x4_t   int8x16x4_t;
  typedef easysimd_int16x8x4_t   int16x8x4_t;
  typedef easysimd_int32x4x4_t   int32x4x4_t;
  typedef easysimd_int64x2x4_t   int64x2x4_t;
  typedef easysimd_uint8x16x4_t  uint8x16x4_t;
  typedef easysimd_uint16x8x4_t  uint16x8x4_t;
  typedef easysimd_uint32x4x4_t  uint32x4x4_t;
  typedef easysimd_uint64x2x4_t  uint64x2x4_t;
  typedef easysimd_float32x4x4_t float32x4x4_t;
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  typedef   easysimd_float64_t     float64_t;
  typedef easysimd_float16x4_t   float16x4_t;
  typedef easysimd_float64x1_t   float64x1_t;
  typedef easysimd_float16x8_t   float16x8_t;
  typedef easysimd_float64x2_t   float64x2_t;
  typedef easysimd_float64x1x2_t float64x1x2_t;
  typedef easysimd_float64x2x2_t float64x2x2_t;
  typedef easysimd_float64x1x3_t float64x1x3_t;
  typedef easysimd_float64x2x3_t float64x2x3_t;
  typedef easysimd_float64x1x4_t float64x1x4_t;
  typedef easysimd_float64x2x4_t float64x2x4_t;
#endif

#if defined(EASYSIMD_X86_MMX_NATIVE)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int8x8_to_m64,                  __m64,    easysimd_int8x8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int16x4_to_m64,                 __m64,   easysimd_int16x4_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int32x2_to_m64,                 __m64,   easysimd_int32x2_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int64x1_to_m64,                 __m64,   easysimd_int64x1_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint8x8_to_m64,                 __m64,   easysimd_uint8x8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint16x4_to_m64,                __m64,  easysimd_uint16x4_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint32x2_to_m64,                __m64,  easysimd_uint32x2_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint64x1_to_m64,                __m64,  easysimd_uint64x1_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_float32x2_to_m64,               __m64, easysimd_float32x2_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_float64x1_to_m64,               __m64, easysimd_float64x1_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int8x8_from_m64,       easysimd_int8x8_t,             __m64)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int16x4_from_m64,     easysimd_int16x4_t,             __m64)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int32x2_from_m64,     easysimd_int32x2_t,             __m64)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int64x1_from_m64,     easysimd_int64x1_t,             __m64)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint8x8_from_m64,     easysimd_uint8x8_t,             __m64)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint16x4_from_m64,   easysimd_uint16x4_t,             __m64)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint32x2_from_m64,   easysimd_uint32x2_t,             __m64)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint64x1_from_m64,   easysimd_uint64x1_t,             __m64)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_float32x2_from_m64, easysimd_float32x2_t,             __m64)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_float64x1_from_m64, easysimd_float64x1_t,             __m64)
#endif
#if defined(EASYSIMD_X86_SSE_NATIVE)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_float32x4_to_m128,              __m128, easysimd_float32x4_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_float32x4_from_m128, easysimd_float32x4_t,            __m128)
#endif
#if defined(EASYSIMD_X86_SSE2_NATIVE)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int8x16_to_m128i,               __m128i,   easysimd_int8x16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int16x8_to_m128i,               __m128i,   easysimd_int16x8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int32x4_to_m128i,               __m128i,   easysimd_int32x4_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int64x2_to_m128i,               __m128i,   easysimd_int64x2_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint8x16_to_m128i,              __m128i,  easysimd_uint8x16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint16x8_to_m128i,              __m128i,  easysimd_uint16x8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint32x4_to_m128i,              __m128i,  easysimd_uint32x4_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint64x2_to_m128i,              __m128i,  easysimd_uint64x2_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_float64x2_to_m128d,             __m128d, easysimd_float64x2_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int8x16_from_m128i,     easysimd_int8x16_t,           __m128i)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int16x8_from_m128i,     easysimd_int16x8_t,           __m128i)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int32x4_from_m128i,     easysimd_int32x4_t,           __m128i)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_int64x2_from_m128i,     easysimd_int64x2_t,           __m128i)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint8x16_from_m128i,   easysimd_uint8x16_t,           __m128i)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint16x8_from_m128i,   easysimd_uint16x8_t,           __m128i)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint32x4_from_m128i,   easysimd_uint32x4_t,           __m128i)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_uint64x2_from_m128i,   easysimd_uint64x2_t,           __m128i)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_float64x2_from_m128d, easysimd_float64x2_t,           __m128d)
#endif

#define EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(T) \
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_##T##_to_private,   easysimd_##T##_private, easysimd_##T##_t) \
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(easysimd_##T##_from_private, easysimd_##T##_t,       easysimd_##T##_private) \

EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(int8x8)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(int16x4)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(int32x2)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(int64x1)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(uint8x8)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(uint16x4)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(uint32x2)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(uint64x1)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(float16x4)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(float32x2)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(float64x1)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(int8x16)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(int16x8)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(int32x4)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(int64x2)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(uint8x16)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(uint16x8)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(uint32x4)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(uint64x2)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(float16x8)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(float32x4)
EASYSIMD_ARM_NEON_TYPE_DEFINE_CONVERSIONS_(float64x2)

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_NEON_TYPES_H */
