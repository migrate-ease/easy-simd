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
 *   2021      Zhi An Ng <zhin@google.com> (Copyright owned by Google, LLC)
 */

#if !defined(EASYSIMD_ARM_NEON_RSQRTE_H)
#define EASYSIMD_ARM_NEON_RSQRTE_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32_t
easysimd_vrsqrtes_f32(easysimd_float32_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrsqrtes_f32(a);
  #else
    #if defined(EASYSIMD_IEEE754_STORAGE)
      /* https://basesandframes.files.wordpress.com/2020/04/even_faster_math_functions_green_2020.pdf
        Pages 100 - 103 */
      #if EASYSIMD_ACCURACY_PREFERENCE <= 0
        return (INT32_C(0x5F37624F) - (a >> 1));
      #else
        easysimd_float32 x = a;
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
        return x;
      #endif
    #elif defined(easysimd_math_sqrtf)
      return 1.0f / easysimd_math_sqrtf(a);
    #else
      HEDLEY_UNREACHABLE();
    #endif
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrsqrtes_f32
  #define vrsqrtes_f32(a) easysimd_vrsqrtes_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64_t
easysimd_vrsqrted_f64(easysimd_float64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrsqrted_f64(a);
  #else
    #if defined(EASYSIMD_IEEE754_STORAGE)
      //https://www.mdpi.com/1099-4300/23/1/86/htm
      easysimd_float64_t x = a;
      easysimd_float64_t xhalf = EASYSIMD_FLOAT64_C(0.5) * x;
      int64_t ix;

      easysimd_memcpy(&ix, &x, sizeof(ix));
      ix = INT64_C(0x5FE6ED2102DCBFDA) - (ix >> 1);
      easysimd_memcpy(&x, &ix, sizeof(x));
      x = x * (EASYSIMD_FLOAT64_C(1.50087895511633457) - xhalf * x * x);
      x = x * (EASYSIMD_FLOAT64_C(1.50000057967625766) - xhalf * x * x);
      return x;
    #elif defined(easysimd_math_sqrtf)
      return EASYSIMD_FLOAT64_C(1.0) / easysimd_math_sqrt(a_.values[i]);
    #else
      HEDLEY_UNREACHABLE();
    #endif
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrsqrted_f64
  #define vrsqrted_f64(a) easysimd_vrsqrted_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x2_t
easysimd_vrsqrte_u32(easysimd_uint32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrsqrte_u32(a);
  #else
    easysimd_uint32x2_private
      a_ = easysimd_uint32x2_to_private(a),
      r_;

    for(size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[i])) ; i++) {
      if(a_.values[i] < 0x3FFFFFFF) {
        r_.values[i] = UINT32_MAX;
      } else {
        uint32_t a_temp = (a_.values[i] >> 23) & 511;
        if(a_temp < 256) {
          a_temp = a_temp * 2 + 1;
        } else {
          a_temp = (a_temp >> 1) << 1;
          a_temp = (a_temp + 1) * 2;
        }
        uint32_t b = 512;
        while((a_temp * (b + 1) * (b + 1)) < (1 << 28))
          b = b + 1;
        r_.values[i] = (b + 1) / 2;
        r_.values[i] = r_.values[i] << 23;
      }
    }
    return easysimd_uint32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrsqrte_u32
  #define vrsqrte_u32(a) easysimd_vrsqrte_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x2_t
easysimd_vrsqrte_f32(easysimd_float32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrsqrte_f32(a);
  #else
    easysimd_float32x2_private
      r_,
      a_ = easysimd_float32x2_to_private(a);

    #if defined(EASYSIMD_IEEE754_STORAGE)
      /* https://basesandframes.files.wordpress.com/2020/04/even_faster_math_functions_green_2020.pdf
        Pages 100 - 103 */
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        #if EASYSIMD_ACCURACY_PREFERENCE <= 0
          r_.i32[i] = INT32_C(0x5F37624F) - (a_.i32[i] >> 1);
        #else
          easysimd_float32 x = a_.values[i];
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

          r_.values[i] = x;
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

    return easysimd_float32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrsqrte_f32
  #define vrsqrte_f32(a) easysimd_vrsqrte_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x1_t
easysimd_vrsqrte_f64(easysimd_float64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrsqrte_f64(a);
  #else
    easysimd_float64x1_private
      r_,
      a_ = easysimd_float64x1_to_private(a);

    #if defined(EASYSIMD_IEEE754_STORAGE)
      //https://www.mdpi.com/1099-4300/23/1/86/htm
      EASYSIMD_VECTORIZE
      for(size_t i = 0 ; i < (sizeof(r_.values)/sizeof(r_.values[0])) ; i++) {
        easysimd_float64_t x = a_.values[i];
        easysimd_float64_t xhalf = EASYSIMD_FLOAT64_C(0.5) * x;
        int64_t ix;

        easysimd_memcpy(&ix, &x, sizeof(ix));
        ix = INT64_C(0x5FE6ED2102DCBFDA) - (ix >> 1);
        easysimd_memcpy(&x, &ix, sizeof(x));
        x = x * (EASYSIMD_FLOAT64_C(1.50087895511633457) - xhalf * x * x);
        x = x * (EASYSIMD_FLOAT64_C(1.50000057967625766) - xhalf * x * x);
        r_.values[i] = x;
      }
    #elif defined(easysimd_math_sqrtf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = EASYSIMD_FLOAT64_C(1.0) / easysimd_math_sqrt(a_.values[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd_float64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrsqrte_f64
  #define vrsqrte_f64(a) easysimd_vrsqrte_f64((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_uint32x4_t
easysimd_vrsqrteq_u32(easysimd_uint32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrsqrteq_u32(a);
  #else
    easysimd_uint32x4_private
      a_ = easysimd_uint32x4_to_private(a),
      r_;

    for(size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[i])) ; i++) {
      if(a_.values[i] < 0x3FFFFFFF) {
        r_.values[i] = UINT32_MAX;
      } else {
        uint32_t a_temp = (a_.values[i] >> 23) & 511;
        if(a_temp < 256) {
          a_temp = a_temp * 2 + 1;
        } else {
          a_temp = (a_temp >> 1) << 1;
          a_temp = (a_temp + 1) * 2;
        }
        uint32_t b = 512;
        while((a_temp * (b + 1) * (b + 1)) < (1 << 28))
          b = b + 1;
        r_.values[i] = (b + 1) / 2;
        r_.values[i] = r_.values[i] << 23;
      }
    }
    return easysimd_uint32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrsqrteq_u32
  #define vrsqrteq_u32(a) easysimd_vrsqrteq_u32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float32x4_t
easysimd_vrsqrteq_f32(easysimd_float32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vrsqrteq_f32(a);
  #else
    easysimd_float32x4_private
      r_,
      a_ = easysimd_float32x4_to_private(a);

    #if defined(EASYSIMD_X86_SSE_NATIVE)
      r_.m128 = _mm_rsqrt_ps(a_.m128);
    #elif defined(EASYSIMD_IEEE754_STORAGE)
      /* https://basesandframes.files.wordpress.com/2020/04/even_faster_math_functions_green_2020.pdf
        Pages 100 - 103 */
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        #if EASYSIMD_ACCURACY_PREFERENCE <= 0
          r_.i32[i] = INT32_C(0x5F37624F) - (a_.i32[i] >> 1);
        #else
          easysimd_float32 x = a_.values[i];
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

          r_.values[i] = x;
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

    return easysimd_float32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vrsqrteq_f32
  #define vrsqrteq_f32(a) easysimd_vrsqrteq_f32((a))
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_float64x2_t
easysimd_vrsqrteq_f64(easysimd_float64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vrsqrteq_f64(a);
  #else
    easysimd_float64x2_private
      r_,
      a_ = easysimd_float64x2_to_private(a);

    #if defined(EASYSIMD_IEEE754_STORAGE)
      //https://www.mdpi.com/1099-4300/23/1/86/htm
      EASYSIMD_VECTORIZE
      for(size_t i = 0 ; i < (sizeof(r_.values)/sizeof(r_.values[0])) ; i++) {
        easysimd_float64_t x = a_.values[i];
        easysimd_float64_t xhalf = EASYSIMD_FLOAT64_C(0.5) * x;
        int64_t ix;

        easysimd_memcpy(&ix, &x, sizeof(ix));
        ix = INT64_C(0x5FE6ED2102DCBFDA) - (ix >> 1);
        easysimd_memcpy(&x, &ix, sizeof(x));
        x = x * (EASYSIMD_FLOAT64_C(1.50087895511633457) - xhalf * x * x);
        x = x * (EASYSIMD_FLOAT64_C(1.50000057967625766) - xhalf * x * x);
        r_.values[i] = x;
      }
    #elif defined(easysimd_math_sqrtf)
      EASYSIMD_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        r_.values[i] = EASYSIMD_FLOAT64_C(1.0) / easysimd_math_sqrt(a_.values[i]);
      }
    #else
      HEDLEY_UNREACHABLE();
    #endif

    return easysimd_float64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vrsqrteq_f64
  #define vrsqrteq_f64(a) easysimd_vrsqrteq_f64((a))
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP
#endif /* !defined(EASYSIMD_ARM_NEON_RSQRTE_H) */
