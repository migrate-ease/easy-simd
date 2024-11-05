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

/* easysimd-arch.h is used to determine which features are available according
   to the compiler.  However, we want to make it possible to forcibly enable
   or disable APIs */

#if !defined(EASYSIMD_FEATURES_H)
#define EASYSIMD_FEATURES_H

#include "easysimd-arch.h"
#include "easysimd-diagnostic.h"

#if !defined(EASYSIMD_X86_SVML_NATIVE) && !defined(EASYSIMD_X86_SVML_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_SVML)
    #define EASYSIMD_X86_SVML_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_SVML_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512VP2INTERSECT_NATIVE) && !defined(EASYSIMD_X86_AVX512VP2INTERSECT_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512VP2INTERSECT)
    #define EASYSIMD_X86_AVX512VP2INTERSECT_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512VP2INTERSECT_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && !defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512VPOPCNTDQ)
    #define EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && !defined(EASYSIMD_X86_AVX512BITALG_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512BITALG)
    #define EASYSIMD_X86_AVX512BITALG_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512BITALG_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && !defined(EASYSIMD_X86_AVX512VBMI_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512VBMI)
    #define EASYSIMD_X86_AVX512VBMI_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512VBMI_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512VBMI2_NATIVE) && !defined(EASYSIMD_X86_AVX512VBMI2_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512VBMI2)
    #define EASYSIMD_X86_AVX512VBMI2_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512VBMI2_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512VNNI_NATIVE) && !defined(EASYSIMD_X86_AVX512VNNI_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512VNNI)
    #define EASYSIMD_X86_AVX512VNNI_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512VNNI_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX5124VNNIW_NATIVE) && !defined(EASYSIMD_X86_AVX5124VNNIW_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX5124VNNIW)
    #define EASYSIMD_X86_AVX5124VNNIW_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX5124VNNIW_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512CD_NATIVE) && !defined(EASYSIMD_X86_AVX512CD_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512CD)
    #define EASYSIMD_X86_AVX512CD_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512CD_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512DQ_NATIVE) && !defined(EASYSIMD_X86_AVX512DQ_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512DQ)
    #define EASYSIMD_X86_AVX512DQ_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512VL_NATIVE) && !defined(EASYSIMD_X86_AVX512VL_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512VL)
    #define EASYSIMD_X86_AVX512VL_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512VL_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512BW_NATIVE) && !defined(EASYSIMD_X86_AVX512BW_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512BW)
    #define EASYSIMD_X86_AVX512BW_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512BF16_NATIVE) && !defined(EASYSIMD_X86_AVX512BF16_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512BF16)
    #define EASYSIMD_X86_AVX512BF16_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512BF16_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NATIVE)
  #define EASYSIMD_X86_AVX512F_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_X86_AVX512F_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX512F)
    #define EASYSIMD_X86_AVX512F_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && !defined(EASYSIMD_X86_AVX2_NATIVE)
  #define EASYSIMD_X86_AVX2_NATIVE
#endif

#if !defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_X86_FMA_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_FMA)
    #define EASYSIMD_X86_FMA_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_FMA_NATIVE) && !defined(EASYSIMD_X86_AVX_NATIVE)
  #define EASYSIMD_X86_AVX_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX2_NATIVE) && !defined(EASYSIMD_X86_AVX2_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX2)
    #define EASYSIMD_X86_AVX2_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX2_NATIVE) && !defined(EASYSIMD_X86_AVX_NATIVE)
  #define EASYSIMD_X86_AVX_NATIVE
#endif

#if !defined(EASYSIMD_X86_AVX_NATIVE) && !defined(EASYSIMD_X86_AVX_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_AVX)
    #define EASYSIMD_X86_AVX_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_AVX_NATIVE) && !defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #define EASYSIMD_X86_SSE4_2_NATIVE
#endif

#if !defined(EASYSIMD_X86_XOP_NATIVE) && !defined(EASYSIMD_X86_XOP_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_XOP)
    #define EASYSIMD_X86_XOP_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_XOP_NATIVE) && !defined(EASYSIMD_X86_SSE4_2_NATIVE)
  #define EASYSIMD_X86_SSE4_2_NATIVE
#endif

#if !defined(EASYSIMD_X86_SSE4_2_NATIVE) && !defined(EASYSIMD_X86_SSE4_2_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_SSE4_2)
    #define EASYSIMD_X86_SSE4_2_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_SSE4_2_NATIVE) && !defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #define EASYSIMD_X86_SSE4_1_NATIVE
#endif

#if !defined(EASYSIMD_X86_SSE4_1_NATIVE) && !defined(EASYSIMD_X86_SSE4_1_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_SSE4_1)
    #define EASYSIMD_X86_SSE4_1_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_SSE4_1_NATIVE) && !defined(EASYSIMD_X86_SSSE3_NATIVE)
  #define EASYSIMD_X86_SSSE3_NATIVE
#endif

#if !defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_X86_SSSE3_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_SSSE3)
    #define EASYSIMD_X86_SSSE3_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_SSSE3_NATIVE) && !defined(EASYSIMD_X86_SSE3_NATIVE)
  #define EASYSIMD_X86_SSE3_NATIVE
#endif

#if !defined(EASYSIMD_X86_SSE3_NATIVE) && !defined(EASYSIMD_X86_SSE3_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_SSE3)
    #define EASYSIMD_X86_SSE3_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_SSE3_NATIVE) && !defined(EASYSIMD_X86_SSE2_NATIVE)
  #define EASYSIMD_X86_SSE2_NATIVE
#endif

#if !defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(EASYSIMD_X86_SSE2_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_SSE2)
    #define EASYSIMD_X86_SSE2_NATIVE
  #endif
#endif
#if defined(EASYSIMD_X86_SSE2_NATIVE) && !defined(EASYSIMD_X86_SSE_NATIVE)
  #define EASYSIMD_X86_SSE_NATIVE
#endif

#if !defined(EASYSIMD_X86_SSE_NATIVE) && !defined(EASYSIMD_X86_SSE_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_SSE)
    #define EASYSIMD_X86_SSE_NATIVE
  #endif
#endif

#if !defined(EASYSIMD_X86_MMX_NATIVE) && !defined(EASYSIMD_X86_MMX_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_MMX)
    #define EASYSIMD_X86_MMX_NATIVE
  #endif
#endif

#if !defined(EASYSIMD_X86_GFNI_NATIVE) && !defined(EASYSIMD_X86_GFNI_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_GFNI)
    #define EASYSIMD_X86_GFNI_NATIVE
  #endif
#endif

#if !defined(EASYSIMD_X86_PCLMUL_NATIVE) && !defined(EASYSIMD_X86_PCLMUL_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_PCLMUL)
    #define EASYSIMD_X86_PCLMUL_NATIVE
  #endif
#endif

#if !defined(EASYSIMD_X86_VPCLMULQDQ_NATIVE) && !defined(EASYSIMD_X86_VPCLMULQDQ_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_VPCLMULQDQ)
    #define EASYSIMD_X86_VPCLMULQDQ_NATIVE
  #endif
#endif

#if !defined(EASYSIMD_X86_F16C_NATIVE) && !defined(EASYSIMD_X86_F16C_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_X86_F16C)
    #define EASYSIMD_X86_F16C_NATIVE
  #endif
#endif

#if !defined(EASYSIMD_X86_SVML_NATIVE) && !defined(EASYSIMD_X86_SVML_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(__INTEL_COMPILER)
    #define EASYSIMD_X86_SVML_NATIVE
  #endif
#endif

#if defined(HEDLEY_MSVC_VERSION)
  #pragma warning(push)
  #pragma warning(disable:4799)
#endif

#if \
    defined(EASYSIMD_X86_AVX_NATIVE) || defined(EASYSIMD_X86_GFNI_NATIVE)
  #include <immintrin.h>
#elif defined(EASYSIMD_X86_SSE4_2_NATIVE)
  #include <nmmintrin.h>
#elif defined(EASYSIMD_X86_SSE4_1_NATIVE)
  #include <smmintrin.h>
#elif defined(EASYSIMD_X86_SSSE3_NATIVE)
  #include <tmmintrin.h>
#elif defined(EASYSIMD_X86_SSE3_NATIVE)
  #include <pmmintrin.h>
#elif defined(EASYSIMD_X86_SSE2_NATIVE)
  #include <emmintrin.h>
#elif defined(EASYSIMD_X86_SSE_NATIVE)
  #include <xmmintrin.h>
#elif defined(EASYSIMD_X86_MMX_NATIVE)
  #include <mmintrin.h>
#endif

#if defined(EASYSIMD_X86_XOP_NATIVE)
  #if defined(_MSC_VER)
    #include <intrin.h>
  #else
    #include <x86intrin.h>
  #endif
#endif

#if defined(HEDLEY_MSVC_VERSION)
  #pragma warning(pop)
#endif

#if !defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && !defined(EASYSIMD_ARM_NEON_A64V8_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_ARM_NEON) && defined(EASYSIMD_ARCH_AARCH64) && EASYSIMD_ARCH_ARM_CHECK(8,0)
    #define EASYSIMD_ARM_NEON_A64V8_NATIVE
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && !defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
  #define EASYSIMD_ARM_NEON_A32V8_NATIVE
#endif

#if !defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && !defined(EASYSIMD_ARM_NEON_A32V8_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_ARM_NEON) && EASYSIMD_ARCH_ARM_CHECK(8,0) && (__ARM_NEON_FP & 0x02)
    #define EASYSIMD_ARM_NEON_A32V8_NATIVE
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V8_NATIVE) && !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #define EASYSIMD_ARM_NEON_A32V7_NATIVE
#endif

#if !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE) && !defined(EASYSIMD_ARM_NEON_A32V7_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_ARM_NEON) && EASYSIMD_ARCH_ARM_CHECK(7,0)
    #define EASYSIMD_ARM_NEON_A32V7_NATIVE
  #endif
#endif
#if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
  #include <arm_neon.h>
  #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #include <arm_fp16.h>
  #endif
#endif

#if !defined(EASYSIMD_ARM_SVE_NATIVE) && !defined(EASYSIMD_ARM_SVE_NO_NATIVE) && !defined(EASYSIMD_NO_NATIVE)
  #if defined(EASYSIMD_ARCH_ARM_SVE)
    #define EASYSIMD_ARM_SVE_NATIVE
    #include <arm_sve.h>
  #endif
#endif

/* This is used to determine whether or not to fall back on a vector
 * function in an earlier ISA extensions, as well as whether
 * we expected any attempts at vectorization to be fruitful or if we
 * expect to always be running serial code.
 *
 * Note that, for some architectures (okay, *one* architecture) there
 * can be a split where some types are supported for one vector length
 * but others only for a shorter length.  Therefore, it is possible to
 * provide separate values for float/int/double types. */

#if !defined(EASYSIMD_NATURAL_VECTOR_SIZE)
  #if defined(EASYSIMD_X86_AVX512F_NATIVE)
    #define EASYSIMD_NATURAL_VECTOR_SIZE (512)
  #elif defined(EASYSIMD_X86_AVX2_NATIVE)
    #define EASYSIMD_NATURAL_VECTOR_SIZE (256)
  #elif defined(EASYSIMD_X86_AVX_NATIVE)
    #define EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE (256)
    #define EASYSIMD_NATURAL_INT_VECTOR_SIZE (128)
    #define EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE (128)
  #elif \
      defined(EASYSIMD_X86_SSE2_NATIVE) || \
      defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    #define EASYSIMD_NATURAL_VECTOR_SIZE (128)
  #elif defined(EASYSIMD_X86_SSE_NATIVE)
    #define EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE (128)
    #define EASYSIMD_NATURAL_INT_VECTOR_SIZE (64)
    #define EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE (0)
  #endif

  #if !defined(EASYSIMD_NATURAL_VECTOR_SIZE)
    #if defined(EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE)
      #define EASYSIMD_NATURAL_VECTOR_SIZE EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE
    #elif defined(EASYSIMD_NATURAL_INT_VECTOR_SIZE)
      #define EASYSIMD_NATURAL_VECTOR_SIZE EASYSIMD_NATURAL_INT_VECTOR_SIZE
    #elif defined(EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE)
      #define EASYSIMD_NATURAL_VECTOR_SIZE EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE
    #else
      #define EASYSIMD_NATURAL_VECTOR_SIZE (0)
    #endif
  #endif

  #if !defined(EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE)
    #define EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE EASYSIMD_NATURAL_VECTOR_SIZE
  #endif
  #if !defined(EASYSIMD_NATURAL_INT_VECTOR_SIZE)
    #define EASYSIMD_NATURAL_INT_VECTOR_SIZE EASYSIMD_NATURAL_VECTOR_SIZE
  #endif
  #if !defined(EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE)
    #define EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE EASYSIMD_NATURAL_VECTOR_SIZE
  #endif
#endif

#define EASYSIMD_NATURAL_VECTOR_SIZE_LE(x) ((EASYSIMD_NATURAL_VECTOR_SIZE > 0) && (EASYSIMD_NATURAL_VECTOR_SIZE <= (x)))
#define EASYSIMD_NATURAL_VECTOR_SIZE_GE(x) ((EASYSIMD_NATURAL_VECTOR_SIZE > 0) && (EASYSIMD_NATURAL_VECTOR_SIZE >= (x)))
#define EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE_LE(x) ((EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE > 0) && (EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE <= (x)))
#define EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE_GE(x) ((EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE > 0) && (EASYSIMD_NATURAL_FLOAT_VECTOR_SIZE >= (x)))
#define EASYSIMD_NATURAL_INT_VECTOR_SIZE_LE(x) ((EASYSIMD_NATURAL_INT_VECTOR_SIZE > 0) && (EASYSIMD_NATURAL_INT_VECTOR_SIZE <= (x)))
#define EASYSIMD_NATURAL_INT_VECTOR_SIZE_GE(x) ((EASYSIMD_NATURAL_INT_VECTOR_SIZE > 0) && (EASYSIMD_NATURAL_INT_VECTOR_SIZE >= (x)))
#define EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE_LE(x) ((EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE > 0) && (EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE <= (x)))
#define EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE_GE(x) ((EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE > 0) && (EASYSIMD_NATURAL_DOUBLE_VECTOR_SIZE >= (x)))

/* Native aliases */
#if defined(EASYSIMD_ENABLE_NATIVE_ALIASES)
  #if !defined(EASYSIMD_X86_MMX_NATIVE)
    #define EASYSIMD_X86_MMX_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_SSE_NATIVE)
    #define EASYSIMD_X86_SSE_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_SSE2_NATIVE)
    #define EASYSIMD_X86_SSE2_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_SSE3_NATIVE)
    #define EASYSIMD_X86_SSE3_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_SSSE3_NATIVE)
    #define EASYSIMD_X86_SSSE3_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_SSE4_1_NATIVE)
    #define EASYSIMD_X86_SSE4_1_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_SSE4_2_NATIVE)
    #define EASYSIMD_X86_SSE4_2_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX_NATIVE)
    #define EASYSIMD_X86_AVX_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX2_NATIVE)
    #define EASYSIMD_X86_AVX2_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_FMA_NATIVE)
    #define EASYSIMD_X86_FMA_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512F_NATIVE)
    #define EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512VL_NATIVE)
    #define EASYSIMD_X86_AVX512VL_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512VBMI_NATIVE)
    #define EASYSIMD_X86_AVX512VBMI_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512VBMI2_NATIVE)
    #define EASYSIMD_X86_AVX512VBMI2_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512BW_NATIVE)
    #define EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512VNNI_NATIVE)
    #define EASYSIMD_X86_AVX512VNNI_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX5124VNNIW_NATIVE)
    #define EASYSIMD_X86_AVX5124VNNIW_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512BF16_NATIVE)
    #define EASYSIMD_X86_AVX512BF16_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512BITALG_NATIVE)
    #define EASYSIMD_X86_AVX512BITALG_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512VPOPCNTDQ_NATIVE)
    #define EASYSIMD_X86_AVX512VPOPCNTDQ_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512DQ_NATIVE)
    #define EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_AVX512CD_NATIVE)
    #define EASYSIMD_X86_AVX512CD_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_GFNI_NATIVE)
    #define EASYSIMD_X86_GFNI_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_PCLMUL_NATIVE)
    #define EASYSIMD_X86_PCLMUL_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_VPCLMULQDQ_NATIVE)
    #define EASYSIMD_X86_VPCLMULQDQ_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_X86_F16C_NATIVE)
    #define EASYSIMD_X86_F16C_ENABLE_NATIVE_ALIASES
  #endif

  #if !defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    #define EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_ARM_NEON_A32V8_NATIVE)
    #define EASYSIMD_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES
  #endif
  #if !defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    #define EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES
  #endif

  #if !defined(EASYSIMD_ARM_SVE_NATIVE)
    #define EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES
  #endif

  #if !defined(EASYSIMD_X86_SHA_NATIVE)
    #define EASYSIMD_X86_SHA_ENABLE_NATIVE_ALIASES
  #endif

  #if !defined(EASYSIMD_X86_AES_NATIVE)
    #define EASYSIMD_X86_AES_ENABLE_NATIVE_ALIASES
  #endif

#endif

/* Are floating point values stored using IEEE 754?  Knowing
 * this at during preprocessing is a bit tricky, mostly because what
 * we're curious about is how values are stored and not whether the
 * implementation is fully conformant in terms of rounding, NaN
 * handling, etc.
 *
 * For example, if you use -ffast-math or -Ofast on
 * GCC or clang IEEE 754 isn't strictly followed, therefore IEE 754
 * support is not advertised (by defining __STDC_IEC_559__).
 *
 * However, what we care about is whether it is safe to assume that
 * floating point values are stored in IEEE 754 format, in which case
 * we can provide faster implementations of some functions.
 *
 * Luckily every vaugely modern architecture I'm aware of uses IEEE 754-
 * so we just assume IEEE 754 for now.  There is a test which verifies
 * this, if that test fails sowewhere please let us know and we'll add
 * an exception for that platform.  Meanwhile, you can define
 * EASYSIMD_NO_IEEE754_STORAGE. */
#if !defined(EASYSIMD_IEEE754_STORAGE) && !defined(EASYSIMD_NO_IEE754_STORAGE)
  #define EASYSIMD_IEEE754_STORAGE
#endif

#if defined(EASYSIMD_ARCH_ARM_NEON_FP16)
  #define EASYSIMD_ARM_NEON_FP16
#endif

#endif /* !defined(EASYSIMD_FEATURES_H) */
