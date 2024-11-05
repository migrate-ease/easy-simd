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

/* Attempt to find math functions.  Functions may be in <cmath>,
 * <math.h>, compiler built-ins/intrinsics, or platform/architecture
 * specific headers.  In some cases, especially those not built in to
 * libm, we may need to define our own implementations. */

#if !defined(EASYSIMD_MATH_H)
#define EASYSIMD_MATH_H 1

#include "hedley.h"
#include "easysimd-features.h"

#include <stdint.h>
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
  #include <arm_neon.h>
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

/* SLEEF support
 * https://sleef.org/
 *
 * If you include <sleef.h> prior to including SIMDe, SIMDe will use
 * SLEEF.  You can also define EASYSIMD_MATH_SLEEF_ENABLE prior to
 * including SIMDe to force the issue.
 *
 * Note that SLEEF does requires linking to libsleef.
 *
 * By default, SIMDe will use the 1 ULP functions, but if you use
 * EASYSIMD_ACCURACY_PREFERENCE of 0 we will use up to 4 ULP.  This is
 * only the case for the easysimd_math_* functions; for code in other
 * SIMDe headers which calls SLEEF directly we may use functions with
 * greater error if the API we're implementing is less precise (for
 * example, SVML guarantees 4 ULP, so we will generally use the 3.5
 * ULP functions from SLEEF). */
#if !defined(EASYSIMD_MATH_SLEEF_DISABLE)
  #if defined(__SLEEF_H__)
    #define EASYSIMD_MATH_SLEEF_ENABLE
  #endif
#endif

#if defined(EASYSIMD_MATH_SLEEF_ENABLE) && !defined(__SLEEF_H__)
  HEDLEY_DIAGNOSTIC_PUSH
  EASYSIMD_DIAGNOSTIC_DISABLE_IGNORED_QUALIFIERS_
  #include <sleef.h>
  HEDLEY_DIAGNOSTIC_POP
#endif

#if defined(EASYSIMD_MATH_SLEEF_ENABLE) && defined(__SLEEF_H__)
  #if defined(SLEEF_VERSION_MAJOR)
    #define EASYSIMD_MATH_SLEEF_VERSION_CHECK(major, minor, patch) (HEDLEY_VERSION_ENCODE(SLEEF_VERSION_MAJOR, SLEEF_VERSION_MINOR, SLEEF_VERSION_PATCHLEVEL) >= HEDLEY_VERSION_ENCODE(major, minor, patch))
  #else
    #define EASYSIMD_MATH_SLEEF_VERSION_CHECK(major, minor, patch) (HEDLEY_VERSION_ENCODE(3,0,0) >= HEDLEY_VERSION_ENCODE(major, minor, patch))
  #endif
#else
  #define EASYSIMD_MATH_SLEEF_VERSION_CHECK(major, minor, patch) (0)
#endif

#if defined(__has_builtin)
  #define EASYSIMD_MATH_BUILTIN_LIBM(func) __has_builtin(__builtin_##func)
#elif \
    HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
    HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
    HEDLEY_GCC_VERSION_CHECK(4,4,0)
  #define EASYSIMD_MATH_BUILTIN_LIBM(func) (1)
#else
  #define EASYSIMD_MATH_BUILTIN_LIBM(func) (0)
#endif

#if defined(HUGE_VAL)
  /* Looks like <math.h> or <cmath> has already been included. */

  /* The math.h from libc++ (yes, the C header from the C++ standard
   * library) will define an isnan function, but not an isnan macro
   * like the C standard requires.  So we detect the header guards
   * macro libc++ uses. */
  #if defined(isnan) || (defined(_LIBCPP_MATH_H) && !defined(_LIBCPP_CMATH))
    #define EASYSIMD_MATH_HAVE_MATH_H
  #elif defined(__cplusplus)
    #define EASYSIMD_MATH_HAVE_CMATH
  #endif
#elif defined(__has_include)
  #if defined(__cplusplus) && (__cplusplus >= 201103L) && __has_include(<cmath>)
    #define EASYSIMD_MATH_HAVE_CMATH
    #include <cmath>
  #elif __has_include(<math.h>)
    #define EASYSIMD_MATH_HAVE_MATH_H
    #include <math.h>
  #elif !defined(EASYSIMD_MATH_NO_LIBM)
    #define EASYSIMD_MATH_NO_LIBM
  #endif
#elif !defined(EASYSIMD_MATH_NO_LIBM)
  #if defined(__cplusplus) && (__cplusplus >= 201103L)
    #define EASYSIMD_MATH_HAVE_CMATH
    HEDLEY_DIAGNOSTIC_PUSH
    #if defined(HEDLEY_MSVC_VERSION)
      /* VS 14 emits this diagnostic about noexcept being used on a
       * <cmath> function, which we can't do anything about. */
      #pragma warning(disable:4996)
    #endif
    #include <cmath>
    HEDLEY_DIAGNOSTIC_POP
  #else
    #define EASYSIMD_MATH_HAVE_MATH_H
    #include <math.h>
  #endif
#endif

#if !defined(EASYSIMD_MATH_INFINITY)
  #if \
      HEDLEY_HAS_BUILTIN(__builtin_inf) || \
      HEDLEY_GCC_VERSION_CHECK(3,3,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
      HEDLEY_CRAY_VERSION_CHECK(8,1,0)
    #define EASYSIMD_MATH_INFINITY (__builtin_inf())
  #elif defined(INFINITY)
    #define EASYSIMD_MATH_INFINITY INFINITY
  #endif
#endif

#if !defined(EASYSIMD_INFINITYF)
  #if \
      HEDLEY_HAS_BUILTIN(__builtin_inff) || \
      HEDLEY_GCC_VERSION_CHECK(3,3,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      HEDLEY_CRAY_VERSION_CHECK(8,1,0) || \
      HEDLEY_IBM_VERSION_CHECK(13,1,0)
    #define EASYSIMD_MATH_INFINITYF (__builtin_inff())
  #elif defined(INFINITYF)
    #define EASYSIMD_MATH_INFINITYF INFINITYF
  #elif defined(EASYSIMD_MATH_INFINITY)
    #define EASYSIMD_MATH_INFINITYF HEDLEY_STATIC_CAST(float, EASYSIMD_MATH_INFINITY)
  #endif
#endif

#if !defined(EASYSIMD_MATH_NAN)
  #if \
      HEDLEY_HAS_BUILTIN(__builtin_nan) || \
      HEDLEY_GCC_VERSION_CHECK(3,3,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
      HEDLEY_CRAY_VERSION_CHECK(8,1,0) || \
      HEDLEY_IBM_VERSION_CHECK(13,1,0)
    #define EASYSIMD_MATH_NAN (__builtin_nan(""))
  #elif defined(NAN)
    #define EASYSIMD_MATH_NAN NAN
  #endif
#endif

#if !defined(EASYSIMD_NANF)
  #if \
      HEDLEY_HAS_BUILTIN(__builtin_nanf) || \
      HEDLEY_GCC_VERSION_CHECK(3,3,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
      HEDLEY_CRAY_VERSION_CHECK(8,1,0)
    #define EASYSIMD_MATH_NANF (__builtin_nanf(""))
  #elif defined(NANF)
    #define EASYSIMD_MATH_NANF NANF
  #elif defined(EASYSIMD_MATH_NAN)
    #define EASYSIMD_MATH_NANF HEDLEY_STATIC_CAST(float, EASYSIMD_MATH_NAN)
  #endif
#endif

#if !defined(EASYSIMD_MATH_PI)
  #if defined(M_PI)
    #define EASYSIMD_MATH_PI M_PI
  #else
    #define EASYSIMD_MATH_PI 3.14159265358979323846
  #endif
#endif

#if !defined(EASYSIMD_MATH_PIF)
  #if defined(M_PI)
    #define EASYSIMD_MATH_PIF HEDLEY_STATIC_CAST(float, M_PI)
  #else
    #define EASYSIMD_MATH_PIF 3.14159265358979323846f
  #endif
#endif

#if !defined(EASYSIMD_MATH_PI_OVER_180)
  #define EASYSIMD_MATH_PI_OVER_180 0.0174532925199432957692369076848861271344287188854172545609719144
#endif

#if !defined(EASYSIMD_MATH_PI_OVER_180F)
  #define EASYSIMD_MATH_PI_OVER_180F 0.0174532925199432957692369076848861271344287188854172545609719144f
#endif

#if !defined(EASYSIMD_MATH_180_OVER_PI)
  #define EASYSIMD_MATH_180_OVER_PI 57.295779513082320876798154814105170332405472466564321549160243861
#endif

#if !defined(EASYSIMD_MATH_180_OVER_PIF)
  #define EASYSIMD_MATH_180_OVER_PIF 57.295779513082320876798154814105170332405472466564321549160243861f
#endif

#if !defined(EASYSIMD_MATH_FLT_MIN)
  #if defined(__FLT_MIN__)
    #define EASYSIMD_MATH_FLT_MIN __FLT_MIN__
  #else
    #if !defined(FLT_MIN)
      #if defined(__cplusplus)
        #include <cfloat>
      #else
        #include <float.h>
      #endif
    #endif
    #define EASYSIMD_MATH_FLT_MIN FLT_MIN
  #endif
#endif

#if !defined(EASYSIMD_MATH_FLT_MAX)
  #if defined(__FLT_MAX__)
    #define EASYSIMD_MATH_FLT_MAX __FLT_MAX__
  #else
    #if !defined(FLT_MAX)
      #if defined(__cplusplus)
        #include <cfloat>
      #else
        #include <float.h>
      #endif
    #endif
    #define EASYSIMD_MATH_FLT_MAX FLT_MAX
  #endif
#endif

#if !defined(EASYSIMD_MATH_DBL_MIN)
  #if defined(__DBL_MIN__)
    #define EASYSIMD_MATH_DBL_MIN __DBL_MIN__
  #else
    #if !defined(DBL_MIN)
      #if defined(__cplusplus)
        #include <cfloat>
      #else
        #include <float.h>
      #endif
    #endif
    #define EASYSIMD_MATH_DBL_MIN DBL_MIN
  #endif
#endif

#if !defined(EASYSIMD_MATH_DBL_MAX)
  #if defined(__DBL_MAX__)
    #define EASYSIMD_MATH_DBL_MAX __DBL_MAX__
  #else
    #if !defined(DBL_MAX)
      #if defined(__cplusplus)
        #include <cfloat>
      #else
        #include <float.h>
      #endif
    #endif
    #define EASYSIMD_MATH_DBL_MAX DBL_MAX
  #endif
#endif

/*** Classification macros from C99 ***/

#if !defined(easysimd_math_isinf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(isinf)
    #define easysimd_math_isinf(v) __builtin_isinf(v)
  #elif defined(isinf) || defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_isinf(v) isinf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_isinf(v) std::isinf(v)
  #endif
#endif

#if !defined(easysimd_math_isinff)
  #if HEDLEY_HAS_BUILTIN(__builtin_isinff) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      HEDLEY_ARM_VERSION_CHECK(4,1,0)
    #define easysimd_math_isinff(v) __builtin_isinff(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_isinff(v) std::isinf(v)
  #elif defined(easysimd_math_isinf)
    #define easysimd_math_isinff(v) easysimd_math_isinf(HEDLEY_STATIC_CAST(double, v))
  #endif
#endif

#if !defined(easysimd_math_isnan)
  #if EASYSIMD_MATH_BUILTIN_LIBM(isnan)
    #define easysimd_math_isnan(v) __builtin_isnan(v)
  #elif defined(isnan) || defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_isnan(v) isnan(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_isnan(v) std::isnan(v)
  #endif
#endif

#if !defined(easysimd_math_isnanf)
  #if HEDLEY_HAS_BUILTIN(__builtin_isnanf) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0) || \
      HEDLEY_ARM_VERSION_CHECK(4,1,0)
    /* XL C/C++ has __builtin_isnan but not __builtin_isnanf */
    #define easysimd_math_isnanf(v) __builtin_isnanf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_isnanf(v) std::isnan(v)
  #elif defined(easysimd_math_isnan)
    #define easysimd_math_isnanf(v) easysimd_math_isnan(HEDLEY_STATIC_CAST(double, v))
  #endif
#endif

#if !defined(easysimd_math_isnormal)
  #if EASYSIMD_MATH_BUILTIN_LIBM(isnormal)
    #define easysimd_math_isnormal(v) __builtin_isnormal(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_isnormal(v) isnormal(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_isnormal(v) std::isnormal(v)
  #endif
#endif

#if !defined(easysimd_math_isnormalf)
  #if HEDLEY_HAS_BUILTIN(__builtin_isnormalf)
    #define easysimd_math_isnormalf(v) __builtin_isnormalf(v)
  #elif EASYSIMD_MATH_BUILTIN_LIBM(isnormal)
    #define easysimd_math_isnormalf(v) __builtin_isnormal(v)
  #elif defined(isnormalf)
    #define easysimd_math_isnormalf(v) isnormalf(v)
  #elif defined(isnormal) || defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_isnormalf(v) isnormal(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_isnormalf(v) std::isnormal(v)
  #elif defined(easysimd_math_isnormal)
    #define easysimd_math_isnormalf(v) easysimd_math_isnormal(v)
  #endif
#endif

#if !defined(easysimd_math_issubnormalf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(fpclassify)
    #define easysimd_math_issubnormalf(v) __builtin_fpclassify(0, 0, 0, 1, 0, v)
  #elif defined(fpclassify)
    #define easysimd_math_issubnormalf(v) (fpclassify(v) == FP_SUBNORMAL)
  #elif defined(EASYSIMD_IEEE754_STORAGE)
    #define easysimd_math_issubnormalf(v) (((easysimd_float32_as_uint32(v) & UINT32_C(0x7F800000)) == UINT32_C(0)) && ((easysimd_float32_as_uint32(v) & UINT32_C(0x007FFFFF)) != UINT32_C(0)))
  #endif
#endif

#if !defined(easysimd_math_issubnormal)
  #if EASYSIMD_MATH_BUILTIN_LIBM(fpclassify)
    #define easysimd_math_issubnormal(v) __builtin_fpclassify(0, 0, 0, 1, 0, v)
  #elif defined(fpclassify)
    #define easysimd_math_issubnormal(v) (fpclassify(v) == FP_SUBNORMAL)
  #elif defined(EASYSIMD_IEEE754_STORAGE)
    #define easysimd_math_issubnormal(v) (((easysimd_float64_as_uint64(v) & UINT64_C(0x7FF0000000000000)) == UINT64_C(0)) && ((easysimd_float64_as_uint64(v) & UINT64_C(0x00FFFFFFFFFFFFF)) != UINT64_C(0)))
  #endif
#endif

#if defined(FP_NAN)
  #define EASYSIMD_MATH_FP_NAN FP_NAN
#else
  #define EASYSIMD_MATH_FP_NAN 0
#endif
#if defined(FP_INFINITE)
  #define EASYSIMD_MATH_FP_INFINITE FP_INFINITE
#else
  #define EASYSIMD_MATH_FP_INFINITE 1
#endif
#if defined(FP_ZERO)
  #define EASYSIMD_MATH_FP_ZERO FP_ZERO
#else
  #define EASYSIMD_MATH_FP_ZERO 2
#endif
#if defined(FP_SUBNORMAL)
  #define EASYSIMD_MATH_FP_SUBNORMAL FP_SUBNORMAL
#else
  #define EASYSIMD_MATH_FP_SUBNORMAL 3
#endif
#if defined(FP_NORMAL)
  #define EASYSIMD_MATH_FP_NORMAL FP_NORMAL
#else
  #define EASYSIMD_MATH_FP_NORMAL 4
#endif

static HEDLEY_INLINE
int
easysimd_math_fpclassifyf(float v) {
  #if EASYSIMD_MATH_BUILTIN_LIBM(fpclassify)
    return __builtin_fpclassify(EASYSIMD_MATH_FP_NAN, EASYSIMD_MATH_FP_INFINITE, EASYSIMD_MATH_FP_NORMAL, EASYSIMD_MATH_FP_SUBNORMAL, EASYSIMD_MATH_FP_ZERO, v);
  #elif defined(fpclassify)
    return fpclassify(v);
  #else
    return
      easysimd_math_isnormalf(v) ? EASYSIMD_MATH_FP_NORMAL    :
      (v == 0.0f)             ? EASYSIMD_MATH_FP_ZERO      :
      easysimd_math_isnanf(v)    ? EASYSIMD_MATH_FP_NAN       :
      easysimd_math_isinff(v)    ? EASYSIMD_MATH_FP_INFINITE  :
                                EASYSIMD_MATH_FP_SUBNORMAL;
  #endif
}

static HEDLEY_INLINE
int
easysimd_math_fpclassify(double v) {
  #if EASYSIMD_MATH_BUILTIN_LIBM(fpclassify)
    return __builtin_fpclassify(EASYSIMD_MATH_FP_NAN, EASYSIMD_MATH_FP_INFINITE, EASYSIMD_MATH_FP_NORMAL, EASYSIMD_MATH_FP_SUBNORMAL, EASYSIMD_MATH_FP_ZERO, v);
  #elif defined(fpclassify)
    return fpclassify(v);
  #else
    return
      easysimd_math_isnormal(v) ? EASYSIMD_MATH_FP_NORMAL    :
      (v == 0.0)             ? EASYSIMD_MATH_FP_ZERO      :
      easysimd_math_isnan(v)    ? EASYSIMD_MATH_FP_NAN       :
      easysimd_math_isinf(v)    ? EASYSIMD_MATH_FP_INFINITE  :
                               EASYSIMD_MATH_FP_SUBNORMAL;
  #endif
}

/*** Manipulation functions ***/

#if !defined(easysimd_math_nextafter)
  #if \
      (HEDLEY_HAS_BUILTIN(__builtin_nextafter) && !defined(HEDLEY_IBM_VERSION)) || \
      HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
      HEDLEY_GCC_VERSION_CHECK(3,4,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0)
    #define easysimd_math_nextafter(x, y) __builtin_nextafter(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_nextafter(x, y) std::nextafter(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_nextafter(x, y) nextafter(x, y)
  #endif
#endif

#if !defined(easysimd_math_nextafterf)
  #if \
      (HEDLEY_HAS_BUILTIN(__builtin_nextafterf) && !defined(HEDLEY_IBM_VERSION)) || \
      HEDLEY_ARM_VERSION_CHECK(4,1,0) || \
      HEDLEY_GCC_VERSION_CHECK(3,4,0) || \
      HEDLEY_INTEL_VERSION_CHECK(13,0,0)
    #define easysimd_math_nextafterf(x, y) __builtin_nextafterf(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_nextafterf(x, y) std::nextafter(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_nextafterf(x, y) nextafterf(x, y)
  #endif
#endif

/*** Functions from C99 ***/

#if !defined(easysimd_math_abs)
  #if EASYSIMD_MATH_BUILTIN_LIBM(abs)
    #define easysimd_math_abs(v) __builtin_abs(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_abs(v) std::abs(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_abs(v) abs(v)
  #endif
#endif

#if !defined(easysimd_math_labs)
  #if EASYSIMD_MATH_BUILTIN_LIBM(labs)
    #define easysimd_math_labs(v) __builtin_labs(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_labs(v) std::labs(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_labs(v) labs(v)
  #endif
#endif

#if !defined(easysimd_math_llabs)
  #if EASYSIMD_MATH_BUILTIN_LIBM(llabs)
    #define easysimd_math_llabs(v) __builtin_llabs(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_llabs(v) std::llabs(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_llabs(v) llabs(v)
  #endif
#endif

#if !defined(easysimd_math_fabsf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(fabsf)
    #define easysimd_math_fabsf(v) __builtin_fabsf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_fabsf(v) std::abs(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_fabsf(v) fabsf(v)
  #endif
#endif

#if !defined(easysimd_math_acos)
  #if EASYSIMD_MATH_BUILTIN_LIBM(acos)
    #define easysimd_math_acos(v) __builtin_acos(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_acos(v) std::acos(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_acos(v) acos(v)
  #endif
#endif

#if !defined(easysimd_math_acosf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(acosf)
    #define easysimd_math_acosf(v) __builtin_acosf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_acosf(v) std::acos(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_acosf(v) acosf(v)
  #endif
#endif

#if !defined(easysimd_math_acosh)
  #if EASYSIMD_MATH_BUILTIN_LIBM(acosh)
    #define easysimd_math_acosh(v) __builtin_acosh(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_acosh(v) std::acosh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_acosh(v) acosh(v)
  #endif
#endif

#if !defined(easysimd_math_acoshf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(acoshf)
    #define easysimd_math_acoshf(v) __builtin_acoshf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_acoshf(v) std::acosh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_acoshf(v) acoshf(v)
  #endif
#endif

#if !defined(easysimd_math_asin)
  #if EASYSIMD_MATH_BUILTIN_LIBM(asin)
    #define easysimd_math_asin(v) __builtin_asin(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_asin(v) std::asin(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_asin(v) asin(v)
  #endif
#endif

#if !defined(easysimd_math_asinf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(asinf)
    #define easysimd_math_asinf(v) __builtin_asinf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_asinf(v) std::asin(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_asinf(v) asinf(v)
  #endif
#endif

#if !defined(easysimd_math_asinh)
  #if EASYSIMD_MATH_BUILTIN_LIBM(asinh)
    #define easysimd_math_asinh(v) __builtin_asinh(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_asinh(v) std::asinh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_asinh(v) asinh(v)
  #endif
#endif

#if !defined(easysimd_math_asinhf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(asinhf)
    #define easysimd_math_asinhf(v) __builtin_asinhf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_asinhf(v) std::asinh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_asinhf(v) asinhf(v)
  #endif
#endif

#if !defined(easysimd_math_atan)
  #if EASYSIMD_MATH_BUILTIN_LIBM(atan)
    #define easysimd_math_atan(v) __builtin_atan(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_atan(v) std::atan(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_atan(v) atan(v)
  #endif
#endif

#if !defined(easysimd_math_atan2)
  #if EASYSIMD_MATH_BUILTIN_LIBM(atan2)
    #define easysimd_math_atan2(y, x) __builtin_atan2(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_atan2(y, x) std::atan2(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_atan2(y, x) atan2(y, x)
  #endif
#endif

#if !defined(easysimd_math_atan2f)
  #if EASYSIMD_MATH_BUILTIN_LIBM(atan2f)
    #define easysimd_math_atan2f(y, x) __builtin_atan2f(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_atan2f(y, x) std::atan2(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_atan2f(y, x) atan2f(y, x)
  #endif
#endif

#if !defined(easysimd_math_atanf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(atanf)
    #define easysimd_math_atanf(v) __builtin_atanf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_atanf(v) std::atan(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_atanf(v) atanf(v)
  #endif
#endif

#if !defined(easysimd_math_atanh)
  #if EASYSIMD_MATH_BUILTIN_LIBM(atanh)
    #define easysimd_math_atanh(v) __builtin_atanh(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_atanh(v) std::atanh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_atanh(v) atanh(v)
  #endif
#endif

#if !defined(easysimd_math_atanhf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(atanhf)
    #define easysimd_math_atanhf(v) __builtin_atanhf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_atanhf(v) std::atanh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_atanhf(v) atanhf(v)
  #endif
#endif

#if !defined(easysimd_math_cbrt)
  #if EASYSIMD_MATH_BUILTIN_LIBM(cbrt)
    #define easysimd_math_cbrt(v) __builtin_cbrt(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_cbrt(v) std::cbrt(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_cbrt(v) cbrt(v)
  #endif
#endif

#if !defined(easysimd_math_cbrtf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(cbrtf)
    #define easysimd_math_cbrtf(v) __builtin_cbrtf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_cbrtf(v) std::cbrt(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_cbrtf(v) cbrtf(v)
  #endif
#endif

#if !defined(easysimd_math_ceil)
  #if EASYSIMD_MATH_BUILTIN_LIBM(ceil)
    #define easysimd_math_ceil(v) __builtin_ceil(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_ceil(v) std::ceil(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_ceil(v) ceil(v)
  #endif
#endif

#if !defined(easysimd_math_ceilf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(ceilf)
    #define easysimd_math_ceilf(v) __builtin_ceilf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_ceilf(v) std::ceil(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_ceilf(v) ceilf(v)
  #endif
#endif

#if !defined(easysimd_math_copysign)
  #if EASYSIMD_MATH_BUILTIN_LIBM(copysign)
    #define easysimd_math_copysign(x, y) __builtin_copysign(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_copysign(x, y) std::copysign(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_copysign(x, y) copysign(x, y)
  #endif
#endif

#if !defined(easysimd_math_copysignf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(copysignf)
    #define easysimd_math_copysignf(x, y) __builtin_copysignf(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_copysignf(x, y) std::copysignf(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_copysignf(x, y) copysignf(x, y)
  #endif
#endif

#if !defined(easysimd_math_cos)
  #if EASYSIMD_MATH_BUILTIN_LIBM(cos)
    #define easysimd_math_cos(v) __builtin_cos(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_cos(v) std::cos(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_cos(v) cos(v)
  #endif
#endif

#if !defined(easysimd_math_cosf)
  #if defined(EASYSIMD_MATH_SLEEF_ENABLE)
    #if EASYSIMD_ACCURACY_PREFERENCE < 1
      #define easysimd_math_cosf(v) Sleef_cosf_u35(v)
    #else
      #define easysimd_math_cosf(v) Sleef_cosf_u10(v)
    #endif
  #elif EASYSIMD_MATH_BUILTIN_LIBM(cosf)
    #define easysimd_math_cosf(v) __builtin_cosf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_cosf(v) std::cos(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_cosf(v) cosf(v)
  #endif
#endif

#if !defined(easysimd_math_cosh)
  #if EASYSIMD_MATH_BUILTIN_LIBM(cosh)
    #define easysimd_math_cosh(v) __builtin_cosh(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_cosh(v) std::cosh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_cosh(v) cosh(v)
  #endif
#endif

#if !defined(easysimd_math_coshf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(coshf)
    #define easysimd_math_coshf(v) __builtin_coshf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_coshf(v) std::cosh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_coshf(v) coshf(v)
  #endif
#endif

#if !defined(easysimd_math_erf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(erf)
    #define easysimd_math_erf(v) __builtin_erf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_erf(v) std::erf(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_erf(v) erf(v)
  #endif
#endif

#if !defined(easysimd_math_erff)
  #if EASYSIMD_MATH_BUILTIN_LIBM(erff)
    #define easysimd_math_erff(v) __builtin_erff(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_erff(v) std::erf(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_erff(v) erff(v)
  #endif
#endif

#if !defined(easysimd_math_erfc)
  #if EASYSIMD_MATH_BUILTIN_LIBM(erfc)
    #define easysimd_math_erfc(v) __builtin_erfc(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_erfc(v) std::erfc(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_erfc(v) erfc(v)
  #endif
#endif

#if !defined(easysimd_math_erfcf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(erfcf)
    #define easysimd_math_erfcf(v) __builtin_erfcf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_erfcf(v) std::erfc(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_erfcf(v) erfcf(v)
  #endif
#endif

#if !defined(easysimd_math_exp)
  #if EASYSIMD_MATH_BUILTIN_LIBM(exp)
    #define easysimd_math_exp(v) __builtin_exp(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_exp(v) std::exp(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_exp(v) exp(v)
  #endif
#endif

#if !defined(easysimd_math_expf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(expf)
    #define easysimd_math_expf(v) __builtin_expf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_expf(v) std::exp(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_expf(v) expf(v)
  #endif
#endif

#if !defined(easysimd_math_expm1)
  #if EASYSIMD_MATH_BUILTIN_LIBM(expm1)
    #define easysimd_math_expm1(v) __builtin_expm1(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_expm1(v) std::expm1(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_expm1(v) expm1(v)
  #endif
#endif

#if !defined(easysimd_math_expm1f)
  #if EASYSIMD_MATH_BUILTIN_LIBM(expm1f)
    #define easysimd_math_expm1f(v) __builtin_expm1f(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_expm1f(v) std::expm1(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_expm1f(v) expm1f(v)
  #endif
#endif

#if !defined(easysimd_math_exp2)
  #if EASYSIMD_MATH_BUILTIN_LIBM(exp2)
    #define easysimd_math_exp2(v) __builtin_exp2(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_exp2(v) std::exp2(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_exp2(v) exp2(v)
  #endif
#endif

#if !defined(easysimd_math_exp2f)
  #if EASYSIMD_MATH_BUILTIN_LIBM(exp2f)
    #define easysimd_math_exp2f(v) __builtin_exp2f(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_exp2f(v) std::exp2(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_exp2f(v) exp2f(v)
  #endif
#endif

#if HEDLEY_HAS_BUILTIN(__builtin_exp10) ||  HEDLEY_GCC_VERSION_CHECK(3,4,0)
  #  define easysimd_math_exp10(v) __builtin_exp10(v)
#else
#  define easysimd_math_exp10(v) pow(10.0, (v))
#endif

#if HEDLEY_HAS_BUILTIN(__builtin_exp10f) ||  HEDLEY_GCC_VERSION_CHECK(3,4,0)
  #  define easysimd_math_exp10f(v) __builtin_exp10f(v)
#else
#  define easysimd_math_exp10f(v) powf(10.0f, (v))
#endif

#if !defined(easysimd_math_fabs)
  #if EASYSIMD_MATH_BUILTIN_LIBM(fabs)
    #define easysimd_math_fabs(v) __builtin_fabs(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_fabs(v) std::fabs(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_fabs(v) fabs(v)
  #endif
#endif

#if !defined(easysimd_math_fabsf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(fabsf)
    #define easysimd_math_fabsf(v) __builtin_fabsf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_fabsf(v) std::fabs(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_fabsf(v) fabsf(v)
  #endif
#endif

#if !defined(easysimd_math_floor)
  #if EASYSIMD_MATH_BUILTIN_LIBM(floor)
    #define easysimd_math_floor(v) __builtin_floor(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_floor(v) std::floor(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_floor(v) floor(v)
  #endif
#endif

#if !defined(easysimd_math_floorf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(floorf)
    #define easysimd_math_floorf(v) __builtin_floorf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_floorf(v) std::floor(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_floorf(v) floorf(v)
  #endif
#endif

#if !defined(easysimd_math_fma)
  #if EASYSIMD_MATH_BUILTIN_LIBM(fma)
    #define easysimd_math_fma(x, y, z) __builtin_fma(x, y, z)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_fma(x, y, z) std::fma(x, y, z)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_fma(x, y, z) fma(x, y, z)
  #endif
#endif

#if !defined(easysimd_math_fmaf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(fmaf)
    #define easysimd_math_fmaf(x, y, z) __builtin_fmaf(x, y, z)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_fmaf(x, y, z) std::fma(x, y, z)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_fmaf(x, y, z) fmaf(x, y, z)
  #endif
#endif

#if !defined(easysimd_math_fmax)
  #if EASYSIMD_MATH_BUILTIN_LIBM(fmax)
    #define easysimd_math_fmax(x, y) __builtin_fmax(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_fmax(x, y) std::fmax(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_fmax(x, y) fmax(x, y)
  #endif
#endif

#if !defined(easysimd_math_fmaxf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(fmaxf)
    #define easysimd_math_fmaxf(x, y) __builtin_fmaxf(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_fmaxf(x, y) std::fmax(x, y)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_fmaxf(x, y) fmaxf(x, y)
  #endif
#endif

#if !defined(easysimd_math_hypot)
  #if EASYSIMD_MATH_BUILTIN_LIBM(hypot)
    #define easysimd_math_hypot(y, x) __builtin_hypot(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_hypot(y, x) std::hypot(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_hypot(y, x) hypot(y, x)
  #endif
#endif

#if !defined(easysimd_math_hypotf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(hypotf)
    #define easysimd_math_hypotf(y, x) __builtin_hypotf(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_hypotf(y, x) std::hypot(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_hypotf(y, x) hypotf(y, x)
  #endif
#endif

#if !defined(easysimd_math_log)
  #if EASYSIMD_MATH_BUILTIN_LIBM(log)
    #define easysimd_math_log(v) __builtin_log(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_log(v) std::log(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_log(v) log(v)
  #endif
#endif

#if !defined(easysimd_math_logf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(logf)
    #define easysimd_math_logf(v) __builtin_logf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_logf(v) std::log(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_logf(v) logf(v)
  #endif
#endif

#if !defined(easysimd_math_logb)
  #if EASYSIMD_MATH_BUILTIN_LIBM(logb)
    #define easysimd_math_logb(v) __builtin_logb(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_logb(v) std::logb(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_logb(v) logb(v)
  #endif
#endif

#if !defined(easysimd_math_logbf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(logbf)
    #define easysimd_math_logbf(v) __builtin_logbf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_logbf(v) std::logb(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_logbf(v) logbf(v)
  #endif
#endif

#if !defined(easysimd_math_log1p)
  #if EASYSIMD_MATH_BUILTIN_LIBM(log1p)
    #define easysimd_math_log1p(v) __builtin_log1p(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_log1p(v) std::log1p(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_log1p(v) log1p(v)
  #endif
#endif

#if !defined(easysimd_math_log1pf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(log1pf)
    #define easysimd_math_log1pf(v) __builtin_log1pf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_log1pf(v) std::log1p(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_log1pf(v) log1pf(v)
  #endif
#endif

#if !defined(easysimd_math_log2)
  #if EASYSIMD_MATH_BUILTIN_LIBM(log2)
    #define easysimd_math_log2(v) __builtin_log2(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_log2(v) std::log2(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_log2(v) log2(v)
  #endif
#endif

#if !defined(easysimd_math_log2f)
  #if EASYSIMD_MATH_BUILTIN_LIBM(log2f)
    #define easysimd_math_log2f(v) __builtin_log2f(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_log2f(v) std::log2(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_log2f(v) log2f(v)
  #endif
#endif

#if !defined(easysimd_math_log10)
  #if EASYSIMD_MATH_BUILTIN_LIBM(log10)
    #define easysimd_math_log10(v) __builtin_log10(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_log10(v) std::log10(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_log10(v) log10(v)
  #endif
#endif

#if !defined(easysimd_math_log10f)
  #if EASYSIMD_MATH_BUILTIN_LIBM(log10f)
    #define easysimd_math_log10f(v) __builtin_log10f(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_log10f(v) std::log10(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_log10f(v) log10f(v)
  #endif
#endif

#if !defined(easysimd_math_modf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(modf)
    #define easysimd_math_modf(x, iptr) __builtin_modf(x, iptr)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_modf(x, iptr) std::modf(x, iptr)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_modf(x, iptr) modf(x, iptr)
  #endif
#endif

#if !defined(easysimd_math_modff)
  #if EASYSIMD_MATH_BUILTIN_LIBM(modff)
    #define easysimd_math_modff(x, iptr) __builtin_modff(x, iptr)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_modff(x, iptr) std::modf(x, iptr)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_modff(x, iptr) modff(x, iptr)
  #endif
#endif

#if !defined(easysimd_math_nearbyint)
  #if EASYSIMD_MATH_BUILTIN_LIBM(nearbyint)
    #define easysimd_math_nearbyint(v) __builtin_nearbyint(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_nearbyint(v) std::nearbyint(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_nearbyint(v) nearbyint(v)
  #endif
#endif

#if !defined(easysimd_math_nearbyintf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(nearbyintf)
    #define easysimd_math_nearbyintf(v) __builtin_nearbyintf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_nearbyintf(v) std::nearbyint(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_nearbyintf(v) nearbyintf(v)
  #endif
#endif

#if !defined(easysimd_math_pow)
  #if EASYSIMD_MATH_BUILTIN_LIBM(pow)
    #define easysimd_math_pow(y, x) __builtin_pow(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_pow(y, x) std::pow(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_pow(y, x) pow(y, x)
  #endif
#endif

#if !defined(easysimd_math_powf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(powf)
    #define easysimd_math_powf(y, x) __builtin_powf(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_powf(y, x) std::pow(y, x)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_powf(y, x) powf(y, x)
  #endif
#endif

#if !defined(easysimd_math_rint)
  #if EASYSIMD_MATH_BUILTIN_LIBM(rint)
    #define easysimd_math_rint(v) __builtin_rint(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_rint(v) std::rint(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_rint(v) rint(v)
  #endif
#endif

#if !defined(easysimd_math_rintf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(rintf)
    #define easysimd_math_rintf(v) __builtin_rintf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_rintf(v) std::rint(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_rintf(v) rintf(v)
  #endif
#endif

#if !defined(easysimd_math_round)
  #if EASYSIMD_MATH_BUILTIN_LIBM(round)
    #define easysimd_math_round(v) __builtin_round(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_round(v) std::round(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_round(v) round(v)
  #endif
#endif

#if !defined(easysimd_math_roundf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(roundf)
    #define easysimd_math_roundf(v) __builtin_roundf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_roundf(v) std::round(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_roundf(v) roundf(v)
  #endif
#endif

#if !defined(easysimd_math_roundeven)
  #if \
      HEDLEY_HAS_BUILTIN(__builtin_roundeven) || \
      HEDLEY_GCC_VERSION_CHECK(10,0,0)
    #define easysimd_math_roundeven(v) __builtin_roundeven(v)
  #elif defined(easysimd_math_round) && defined(easysimd_math_fabs)
    static HEDLEY_INLINE
    double
    easysimd_math_roundeven(double v) {
      double rounded = easysimd_math_round(v);
      double diff = rounded - v;
      if (HEDLEY_UNLIKELY(easysimd_math_fabs(diff) == 0.5) && (HEDLEY_STATIC_CAST(int64_t, rounded) & 1)) {
        rounded = v - diff;
      }
      return rounded;
    }
    #define easysimd_math_roundeven easysimd_math_roundeven
  #endif
#endif

#if !defined(easysimd_math_roundevenf)
  #if \
      HEDLEY_HAS_BUILTIN(__builtin_roundevenf) || \
      HEDLEY_GCC_VERSION_CHECK(10,0,0)
    #define easysimd_math_roundevenf(v) __builtin_roundevenf(v)
  #elif defined(easysimd_math_roundf) && defined(easysimd_math_fabsf)
    static HEDLEY_INLINE
    float
    easysimd_math_roundevenf(float v) {
      float rounded = easysimd_math_roundf(v);
      float diff = rounded - v;
      if (HEDLEY_UNLIKELY(easysimd_math_fabsf(diff) == 0.5f) && (HEDLEY_STATIC_CAST(int32_t, rounded) & 1)) {
        rounded = v - diff;
      }
      return rounded;
    }
    #define easysimd_math_roundevenf easysimd_math_roundevenf
  #endif
#endif

#if !defined(easysimd_math_sin)
  #if EASYSIMD_MATH_BUILTIN_LIBM(sin)
    #define easysimd_math_sin(v) __builtin_sin(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_sin(v) std::sin(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_sin(v) sin(v)
  #endif
#endif

#if !defined(easysimd_math_sinf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(sinf)
    #define easysimd_math_sinf(v) __builtin_sinf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_sinf(v) std::sin(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_sinf(v) sinf(v)
  #endif
#endif

#if !defined(easysimd_math_sinh)
  #if EASYSIMD_MATH_BUILTIN_LIBM(sinh)
    #define easysimd_math_sinh(v) __builtin_sinh(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_sinh(v) std::sinh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_sinh(v) sinh(v)
  #endif
#endif

#if !defined(easysimd_math_sinhf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(sinhf)
    #define easysimd_math_sinhf(v) __builtin_sinhf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_sinhf(v) std::sinh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_sinhf(v) sinhf(v)
  #endif
#endif

#if !defined(easysimd_math_sqrt)
  #if EASYSIMD_MATH_BUILTIN_LIBM(sqrt)
    #define easysimd_math_sqrt(v) __builtin_sqrt(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_sqrt(v) std::sqrt(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_sqrt(v) sqrt(v)
  #endif
#endif

#if !defined(easysimd_math_sqrtf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(sqrtf)
    #define easysimd_math_sqrtf(v) __builtin_sqrtf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_sqrtf(v) std::sqrt(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_sqrtf(v) sqrtf(v)
  #endif
#endif

#if !defined(easysimd_math_tan)
  #if EASYSIMD_MATH_BUILTIN_LIBM(tan)
    #define easysimd_math_tan(v) __builtin_tan(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_tan(v) std::tan(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_tan(v) tan(v)
  #endif
#endif

#if !defined(easysimd_math_tanf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(tanf)
    #define easysimd_math_tanf(v) __builtin_tanf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_tanf(v) std::tan(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_tanf(v) tanf(v)
  #endif
#endif

#if !defined(easysimd_math_tanh)
  #if EASYSIMD_MATH_BUILTIN_LIBM(tanh)
    #define easysimd_math_tanh(v) __builtin_tanh(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_tanh(v) std::tanh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_tanh(v) tanh(v)
  #endif
#endif

#if !defined(easysimd_math_tanhf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(tanhf)
    #define easysimd_math_tanhf(v) __builtin_tanhf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_tanhf(v) std::tanh(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_tanhf(v) tanhf(v)
  #endif
#endif

#if !defined(easysimd_math_trunc)
  #if EASYSIMD_MATH_BUILTIN_LIBM(trunc)
    #define easysimd_math_trunc(v) __builtin_trunc(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_trunc(v) std::trunc(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_trunc(v) trunc(v)
  #endif
#endif

#if !defined(easysimd_math_truncf)
  #if EASYSIMD_MATH_BUILTIN_LIBM(truncf)
    #define easysimd_math_truncf(v) __builtin_truncf(v)
  #elif defined(EASYSIMD_MATH_HAVE_CMATH)
    #define easysimd_math_truncf(v) std::trunc(v)
  #elif defined(EASYSIMD_MATH_HAVE_MATH_H)
    #define easysimd_math_truncf(v) truncf(v)
  #endif
#endif

/*** Comparison macros (which don't raise invalid errors) ***/

#if defined(isunordered)
  #define easysimd_math_isunordered(x, y) isunordered(x, y)
#elif HEDLEY_HAS_BUILTIN(__builtin_isunordered)
  #define easysimd_math_isunordered(x, y) __builtin_isunordered(x, y)
#else
  static HEDLEY_INLINE
  int easysimd_math_isunordered(double x, double y) {
    return (x != y) && (x != x || y != y);
  }
  #define easysimd_math_isunordered easysimd_math_isunordered

  static HEDLEY_INLINE
  int easysimd_math_isunorderedf(float x, float y) {
    return (x != y) && (x != x || y != y);
  }
  #define easysimd_math_isunorderedf easysimd_math_isunorderedf
#endif
#if !defined(easysimd_math_isunorderedf)
  #define easysimd_math_isunorderedf easysimd_math_isunordered
#endif

/*** Additional functions not in libm ***/

#if defined(easysimd_math_fabs) && defined(easysimd_math_sqrt) && defined(easysimd_math_exp)
  static HEDLEY_INLINE
  double
  easysimd_math_cdfnorm(double x) {
    /* https://www.johndcook.com/blog/cpp_phi/
    * Public Domain */
    static const double a1 =  0.254829592;
    static const double a2 = -0.284496736;
    static const double a3 =  1.421413741;
    static const double a4 = -1.453152027;
    static const double a5 =  1.061405429;
    static const double p  =  0.3275911;

    const int sign = x < 0;
    x = easysimd_math_fabs(x) / easysimd_math_sqrt(2.0);

    /* A&S formula 7.1.26 */
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * easysimd_math_exp(-x * x);

    return 0.5 * (1.0 + (sign ? -y : y));
  }
  #define easysimd_math_cdfnorm easysimd_math_cdfnorm
#endif

#if defined(easysimd_math_fabsf) && defined(easysimd_math_sqrtf) && defined(easysimd_math_expf)
  static HEDLEY_INLINE
  float
  easysimd_math_cdfnormf(float x) {
    /* https://www.johndcook.com/blog/cpp_phi/
    * Public Domain */
    static const float a1 =  0.254829592f;
    static const float a2 = -0.284496736f;
    static const float a3 =  1.421413741f;
    static const float a4 = -1.453152027f;
    static const float a5 =  1.061405429f;
    static const float p  =  0.3275911f;

    const int sign = x < 0;
    x = easysimd_math_fabsf(x) / easysimd_math_sqrtf(2.0f);

    /* A&S formula 7.1.26 */
    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * easysimd_math_expf(-x * x);

    return 0.5f * (1.0f + (sign ? -y : y));
  }
  #define easysimd_math_cdfnormf easysimd_math_cdfnormf
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DIAGNOSTIC_DISABLE_FLOAT_EQUAL_

#if !defined(easysimd_math_cdfnorminv) && defined(easysimd_math_log) && defined(easysimd_math_sqrt)
  /*https://web.archive.org/web/20150910081113/http://home.online.no/~pjacklam/notes/invnorm/impl/sprouse/ltqnorm.c*/
  static HEDLEY_INLINE
  double
  easysimd_math_cdfnorminv(double p) {
    static const double a[] = {
      -3.969683028665376e+01,
       2.209460984245205e+02,
      -2.759285104469687e+02,
       1.383577518672690e+02,
      -3.066479806614716e+01,
       2.506628277459239e+00
    };

    static const double b[] = {
      -5.447609879822406e+01,
       1.615858368580409e+02,
      -1.556989798598866e+02,
       6.680131188771972e+01,
      -1.328068155288572e+01
    };

    static const double c[] = {
      -7.784894002430293e-03,
      -3.223964580411365e-01,
      -2.400758277161838e+00,
      -2.549732539343734e+00,
       4.374664141464968e+00,
       2.938163982698783e+00
    };

    static const double d[] = {
      7.784695709041462e-03,
      3.224671290700398e-01,
      2.445134137142996e+00,
      3.754408661907416e+00
    };

    static const double low  = 0.02425;
    static const double high = 0.97575;
    double q, r;

    if (p < 0 || p > 1) {
      return 0.0;
    } else if (p == 0) {
      return -EASYSIMD_MATH_INFINITY;
    } else if (p == 1) {
      return EASYSIMD_MATH_INFINITY;
    } else if (p < low) {
      q = easysimd_math_sqrt(-2.0 * easysimd_math_log(p));
      return
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1));
    } else if (p > high) {
      q = easysimd_math_sqrt(-2.0 * easysimd_math_log(1.0 - p));
      return
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
         (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1));
    } else {
      q = p - 0.5;
      r = q * q;
      return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
}
#define easysimd_math_cdfnorminv easysimd_math_cdfnorminv
#endif

#if !defined(easysimd_math_cdfnorminvf) && defined(easysimd_math_logf) && defined(easysimd_math_sqrtf)
  static HEDLEY_INLINE
  float
  easysimd_math_cdfnorminvf(float p) {
    static const float a[] = {
      -3.969683028665376e+01f,
       2.209460984245205e+02f,
      -2.759285104469687e+02f,
       1.383577518672690e+02f,
      -3.066479806614716e+01f,
       2.506628277459239e+00f
    };
    static const float b[] = {
      -5.447609879822406e+01f,
       1.615858368580409e+02f,
      -1.556989798598866e+02f,
       6.680131188771972e+01f,
      -1.328068155288572e+01f
    };
    static const float c[] = {
      -7.784894002430293e-03f,
      -3.223964580411365e-01f,
      -2.400758277161838e+00f,
      -2.549732539343734e+00f,
       4.374664141464968e+00f,
       2.938163982698783e+00f
    };
    static const float d[] = {
      7.784695709041462e-03f,
      3.224671290700398e-01f,
      2.445134137142996e+00f,
      3.754408661907416e+00f
    };
    static const float low  = 0.02425f;
    static const float high = 0.97575f;
    float q, r;

    if (p < 0 || p > 1) {
      return 0.0f;
    } else if (p == 0) {
      return -EASYSIMD_MATH_INFINITYF;
    } else if (p == 1) {
      return EASYSIMD_MATH_INFINITYF;
    } else if (p < low) {
      q = easysimd_math_sqrtf(-2.0f * easysimd_math_logf(p));
      return
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1));
    } else if (p > high) {
      q = easysimd_math_sqrtf(-2.0f * easysimd_math_logf(1.0f - p));
      return
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
         (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1));
    } else {
      q = p - 0.5f;
      r = q * q;
      return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
         q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
  }
  #define easysimd_math_cdfnorminvf easysimd_math_cdfnorminvf
#endif

#if !defined(easysimd_math_erfinv) && defined(easysimd_math_log) && defined(easysimd_math_copysign) && defined(easysimd_math_sqrt)
  static HEDLEY_INLINE
  double
  easysimd_math_erfinv(double x) {
    /* https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
     *
     * The original answer on SO uses a constant of 0.147, but in my
     * testing 0.14829094707965850830078125 gives a lower average absolute error
     * (0.0001410958211636170744895935 vs. 0.0001465479290345683693885803).
     * That said, if your goal is to minimize the *maximum* absolute
     * error, 0.15449436008930206298828125 provides significantly better
     * results; 0.0009250640869140625000000000 vs ~ 0.005. */
    double tt1, tt2, lnx;
    double sgn = easysimd_math_copysign(1.0, x);

    x = (1.0 - x) * (1.0 + x);
    lnx = easysimd_math_log(x);

    tt1 = 2.0 / (EASYSIMD_MATH_PI * 0.14829094707965850830078125) + 0.5 * lnx;
    tt2 = (1.0 / 0.14829094707965850830078125) * lnx;

    return sgn * easysimd_math_sqrt(-tt1 + easysimd_math_sqrt(tt1 * tt1 - tt2));
  }
  #define easysimd_math_erfinv easysimd_math_erfinv
#endif

#if !defined(easysimd_math_erfinvf) && defined(easysimd_math_logf) && defined(easysimd_math_copysignf) && defined(easysimd_math_sqrtf)
  static HEDLEY_INLINE
  float
  easysimd_math_erfinvf(float x) {
    float tt1, tt2, lnx;
    float sgn = easysimd_math_copysignf(1.0f, x);

    x = (1.0f - x) * (1.0f + x);
    lnx = easysimd_math_logf(x);

    tt1 = 2.0f / (EASYSIMD_MATH_PIF * 0.14829094707965850830078125f) + 0.5f * lnx;
    tt2 = (1.0f / 0.14829094707965850830078125f) * lnx;

    return sgn * easysimd_math_sqrtf(-tt1 + easysimd_math_sqrtf(tt1 * tt1 - tt2));
  }
  #define easysimd_math_erfinvf easysimd_math_erfinvf
#endif

#if !defined(easysimd_math_erfcinv) && defined(easysimd_math_erfinv) && defined(easysimd_math_log) && defined(easysimd_math_sqrt)
  static HEDLEY_INLINE
  double
  easysimd_math_erfcinv(double x) {
    if(x >= 0.0625 && x < 2.0) {
      return easysimd_math_erfinv(1.0 - x);
    } else if (x < 0.0625 && x >= 1.0e-100) {
      double p[6] = {
        0.1550470003116,
        1.382719649631,
        0.690969348887,
        -1.128081391617,
        0.680544246825,
        -0.16444156791
      };
      double q[3] = {
        0.155024849822,
        1.385228141995,
        1.000000000000
      };

      const double t = 1.0 / easysimd_math_sqrt(-easysimd_math_log(x));
      return (p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])))) /
            (q[0] + t * (q[1] + t * (q[2])));
    } else if (x < 1.0e-100 && x >= EASYSIMD_MATH_DBL_MIN) {
      double p[4] = {
        0.00980456202915,
        0.363667889171,
        0.97302949837,
        -0.5374947401
      };
      double q[3] = {
        0.00980451277802,
        0.363699971544,
        1.000000000000
      };

      const double t = 1.0 / easysimd_math_sqrt(-easysimd_math_log(x));
      return (p[0] / t + p[1] + t * (p[2] + t * p[3])) /
             (q[0] + t * (q[1] + t * (q[2])));
    } else if (!easysimd_math_isnormal(x)) {
      return EASYSIMD_MATH_INFINITY;
    } else {
      return -EASYSIMD_MATH_INFINITY;
    }
  }

  #define easysimd_math_erfcinv easysimd_math_erfcinv
#endif

#if !defined(easysimd_math_erfcinvf) && defined(easysimd_math_erfinvf) && defined(easysimd_math_logf) && defined(easysimd_math_sqrtf)
  static HEDLEY_INLINE
  float
  easysimd_math_erfcinvf(float x) {
    if(x >= 0.0625f && x < 2.0f) {
      return easysimd_math_erfinvf(1.0f - x);
    } else if (x < 0.0625f && x >= EASYSIMD_MATH_FLT_MIN) {
      static const float p[6] = {
         0.1550470003116f,
         1.382719649631f,
         0.690969348887f,
        -1.128081391617f,
         0.680544246825f
        -0.164441567910f
      };
      static const float q[3] = {
        0.155024849822f,
        1.385228141995f,
        1.000000000000f
      };

      const float t = 1.0f / easysimd_math_sqrtf(-easysimd_math_logf(x));
      return (p[0] / t + p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])))) /
             (q[0] + t * (q[1] + t * (q[2])));
    } else if (x < EASYSIMD_MATH_FLT_MIN && easysimd_math_isnormalf(x)) {
      static const float p[4] = {
        0.00980456202915f,
        0.36366788917100f,
        0.97302949837000f,
        -0.5374947401000f
      };
      static const float q[3] = {
        0.00980451277802f,
        0.36369997154400f,
        1.00000000000000f
      };

      const float t = 1.0f / easysimd_math_sqrtf(-easysimd_math_logf(x));
      return (p[0] / t + p[1] + t * (p[2] + t * p[3])) /
             (q[0] + t * (q[1] + t * (q[2])));
    } else {
      return easysimd_math_isnormalf(x) ? -EASYSIMD_MATH_INFINITYF : EASYSIMD_MATH_INFINITYF;
    }
  }

  #define easysimd_math_erfcinvf easysimd_math_erfcinvf
#endif

HEDLEY_DIAGNOSTIC_POP

static HEDLEY_INLINE
double
easysimd_math_rad2deg(double radians) {
 return radians * EASYSIMD_MATH_180_OVER_PI;
}

static HEDLEY_INLINE
float
easysimd_math_rad2degf(float radians) {
    return radians * EASYSIMD_MATH_180_OVER_PIF;
}

static HEDLEY_INLINE
double
easysimd_math_deg2rad(double degrees) {
  return degrees * EASYSIMD_MATH_PI_OVER_180;
}

static HEDLEY_INLINE
float
easysimd_math_deg2radf(float degrees) {
    return degrees * (EASYSIMD_MATH_PI_OVER_180F);
}

/***  Saturated arithmetic ***/

static HEDLEY_INLINE
int8_t
easysimd_math_adds_i8(int8_t a, int8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqaddb_s8(a, b);
  #else
    uint8_t a_ = HEDLEY_STATIC_CAST(uint8_t, a);
    uint8_t b_ = HEDLEY_STATIC_CAST(uint8_t, b);
    uint8_t r_ = a_ + b_;

    a_ = (a_ >> ((8 * sizeof(r_)) - 1)) + INT8_MAX;
    if (HEDLEY_STATIC_CAST(int8_t, ((a_ ^ b_) | ~(b_ ^ r_))) >= 0) {
      r_ = a_;
    }

    return HEDLEY_STATIC_CAST(int8_t, r_);
  #endif
}

static HEDLEY_INLINE
int16_t
easysimd_math_adds_i16(int16_t a, int16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqaddh_s16(a, b);
  #else
    uint16_t a_ = HEDLEY_STATIC_CAST(uint16_t, a);
    uint16_t b_ = HEDLEY_STATIC_CAST(uint16_t, b);
    uint16_t r_ = a_ + b_;

    a_ = (a_ >> ((8 * sizeof(r_)) - 1)) + INT16_MAX;
    if (HEDLEY_STATIC_CAST(int16_t, ((a_ ^ b_) | ~(b_ ^ r_))) >= 0) {
      r_ = a_;
    }

    return HEDLEY_STATIC_CAST(int16_t, r_);
  #endif
}

static HEDLEY_INLINE
int32_t
easysimd_math_adds_i32(int32_t a, int32_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqadds_s32(a, b);
  #else
    uint32_t a_ = HEDLEY_STATIC_CAST(uint32_t, a);
    uint32_t b_ = HEDLEY_STATIC_CAST(uint32_t, b);
    uint32_t r_ = a_ + b_;

    a_ = (a_ >> ((8 * sizeof(r_)) - 1)) + INT32_MAX;
    if (HEDLEY_STATIC_CAST(int32_t, ((a_ ^ b_) | ~(b_ ^ r_))) >= 0) {
      r_ = a_;
    }

    return HEDLEY_STATIC_CAST(int32_t, r_);
  #endif
}

static HEDLEY_INLINE
int64_t
easysimd_math_adds_i64(int64_t a, int64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqaddd_s64(a, b);
  #else
    uint64_t a_ = HEDLEY_STATIC_CAST(uint64_t, a);
    uint64_t b_ = HEDLEY_STATIC_CAST(uint64_t, b);
    uint64_t r_ = a_ + b_;

    a_ = (a_ >> ((8 * sizeof(r_)) - 1)) + INT64_MAX;
    if (HEDLEY_STATIC_CAST(int64_t, ((a_ ^ b_) | ~(b_ ^ r_))) >= 0) {
      r_ = a_;
    }

    return HEDLEY_STATIC_CAST(int64_t, r_);
  #endif
}

static HEDLEY_INLINE
uint8_t
easysimd_math_adds_u8(uint8_t a, uint8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqaddb_u8(a, b);
  #else
    uint8_t r = a + b;
    r |= -(r < a);
    return r;
  #endif
}

static HEDLEY_INLINE
uint16_t
easysimd_math_adds_u16(uint16_t a, uint16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqaddh_u16(a, b);
  #else
    uint16_t r = a + b;
    r |= -(r < a);
    return r;
  #endif
}

static HEDLEY_INLINE
uint32_t
easysimd_math_adds_u32(uint32_t a, uint32_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqadds_u32(a, b);
  #else
    uint32_t r = a + b;
    r |= -(r < a);
    return r;
  #endif
}

static HEDLEY_INLINE
uint64_t
easysimd_math_adds_u64(uint64_t a, uint64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqaddd_u64(a, b);
  #else
    uint64_t r = a + b;
    r |= -(r < a);
    return r;
  #endif
}

static HEDLEY_INLINE
int8_t
easysimd_math_subs_i8(int8_t a, int8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqsubb_s8(a, b);
  #else
    uint8_t a_ = HEDLEY_STATIC_CAST(uint8_t, a);
    uint8_t b_ = HEDLEY_STATIC_CAST(uint8_t, b);
    uint8_t r_ = a_ - b_;

    a_ = (a_ >> 7) + INT8_MAX;

    if (HEDLEY_STATIC_CAST(int8_t, (a_ ^ b_) & (a_ ^ r_)) < 0) {
      r_ = a_;
    }

    return HEDLEY_STATIC_CAST(int8_t, r_);
  #endif
}

static HEDLEY_INLINE
int16_t
easysimd_math_subs_i16(int16_t a, int16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqsubh_s16(a, b);
  #else
    uint16_t a_ = HEDLEY_STATIC_CAST(uint16_t, a);
    uint16_t b_ = HEDLEY_STATIC_CAST(uint16_t, b);
    uint16_t r_ = a_ - b_;

    a_ = (a_ >> 15) + INT16_MAX;

    if (HEDLEY_STATIC_CAST(int16_t, (a_ ^ b_) & (a_ ^ r_)) < 0) {
      r_ = a_;
    }

    return HEDLEY_STATIC_CAST(int16_t, r_);
  #endif
}

static HEDLEY_INLINE
int32_t
easysimd_math_subs_i32(int32_t a, int32_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqsubs_s32(a, b);
  #else
    uint32_t a_ = HEDLEY_STATIC_CAST(uint32_t, a);
    uint32_t b_ = HEDLEY_STATIC_CAST(uint32_t, b);
    uint32_t r_ = a_ - b_;

    a_ = (a_ >> 31) + INT32_MAX;

    if (HEDLEY_STATIC_CAST(int32_t, (a_ ^ b_) & (a_ ^ r_)) < 0) {
      r_ = a_;
    }

    return HEDLEY_STATIC_CAST(int32_t, r_);
  #endif
}

static HEDLEY_INLINE
int64_t
easysimd_math_subs_i64(int64_t a, int64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqsubd_s64(a, b);
  #else
    uint64_t a_ = HEDLEY_STATIC_CAST(uint64_t, a);
    uint64_t b_ = HEDLEY_STATIC_CAST(uint64_t, b);
    uint64_t r_ = a_ - b_;

    a_ = (a_ >> 63) + INT64_MAX;

    if (HEDLEY_STATIC_CAST(int64_t, (a_ ^ b_) & (a_ ^ r_)) < 0) {
      r_ = a_;
    }

    return HEDLEY_STATIC_CAST(int64_t, r_);
  #endif
}

static HEDLEY_INLINE
uint8_t
easysimd_math_subs_u8(uint8_t a, uint8_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqsubb_u8(a, b);
  #else
    uint8_t res = a - b;
    res &= -(res <= a);
    return res;
  #endif
}

static HEDLEY_INLINE
uint16_t
easysimd_math_subs_u16(uint16_t a, uint16_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqsubh_u16(a, b);
  #else
    uint16_t res = a - b;
    res &= -(res <= a);
    return res;
  #endif
}

static HEDLEY_INLINE
uint32_t
easysimd_math_subs_u32(uint32_t a, uint32_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqsubs_u32(a, b);
  #else
    uint32_t res = a - b;
    res &= -(res <= a);
    return res;
  #endif
}

static HEDLEY_INLINE
uint64_t
easysimd_math_subs_u64(uint64_t a, uint64_t b) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqsubd_u64(a, b);
  #else
    uint64_t res = a - b;
    res &= -(res <= a);
    return res;
  #endif
}

HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_MATH_H) */
