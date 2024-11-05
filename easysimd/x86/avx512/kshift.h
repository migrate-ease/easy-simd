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
 *   2020      Christopher Moore <moore@free.fr>
 */

#if !defined(EASYSIMD_X86_AVX512_KSHIFT_H)
#define EASYSIMD_X86_AVX512_KSHIFT_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_kshiftli_mask16 (easysimd__mmask16 a, unsigned int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  return HEDLEY_STATIC_CAST(easysimd__mmask16, (count <= 15) ? (a << count) : 0);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && (!defined(EASYSIMD_DETECT_CLANG_VERSION) && EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0))
  #define easysimd_kshiftli_mask16(a, count) _kshiftli_mask16(a, count)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _kshiftli_mask16
  #define _kshiftli_mask16(a, count) easysimd_kshiftli_mask16(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_kshiftli_mask32 (easysimd__mmask32 a, unsigned int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  return (count <= 31) ? (a << count) : 0;
}
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && (!defined(EASYSIMD_DETECT_CLANG_VERSION) && EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0))
  #define easysimd_kshiftli_mask32(a, count) _kshiftli_mask32(a, count)
#endif
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _kshiftli_mask32
  #define _kshiftli_mask32(a, count) easysimd_kshiftli_mask32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_kshiftli_mask64 (easysimd__mmask64 a, unsigned int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  return (count <= 63) ? (a << count) : 0;
}
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && (!defined(EASYSIMD_DETECT_CLANG_VERSION) && EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0))
  #define easysimd_kshiftli_mask64(a, count) _kshiftli_mask64(a, count)
#endif
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _kshiftli_mask64
  #define _kshiftli_mask64(a, count) easysimd_kshiftli_mask64(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_kshiftli_mask8 (easysimd__mmask8 a, unsigned int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  return HEDLEY_STATIC_CAST(easysimd__mmask8, (count <= 7) ? (a << count) : 0);
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && (!defined(EASYSIMD_DETECT_CLANG_VERSION) && EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0))
  #define easysimd_kshiftli_mask8(a, count) _kshiftli_mask8(a, count)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _kshiftli_mask8
  #define _kshiftli_mask8(a, count) easysimd_kshiftli_mask8(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask16
easysimd_kshiftri_mask16 (easysimd__mmask16 a, unsigned int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  return HEDLEY_STATIC_CAST(easysimd__mmask16, (count <= 15) ? (a >> count) : 0);
}
#if defined(EASYSIMD_X86_AVX512F_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && (!defined(EASYSIMD_DETECT_CLANG_VERSION) && EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0))
  #define easysimd_kshiftri_mask16(a, count) _kshiftri_mask16(a, count)
#endif
#if defined(EASYSIMD_X86_AVX512F_ENABLE_NATIVE_ALIASES)
  #undef _kshiftri_mask16
  #define _kshiftri_mask16(a, count) easysimd_kshiftri_mask16(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask32
easysimd_kshiftri_mask32 (easysimd__mmask32 a, unsigned int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  return (count <= 31) ? (a >> count) : 0;
}
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && (!defined(EASYSIMD_DETECT_CLANG_VERSION) && EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0))
  #define easysimd_kshiftri_mask32(a, count) _kshiftri_mask32(a, count)
#endif
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _kshiftri_mask32
  #define _kshiftri_mask32(a, count) easysimd_kshiftri_mask32(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask64
easysimd_kshiftri_mask64 (easysimd__mmask64 a, unsigned int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  return (count <= 63) ? (a >> count) : 0;
}
#if defined(EASYSIMD_X86_AVX512BW_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && (!defined(EASYSIMD_DETECT_CLANG_VERSION) && EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0))
  #define easysimd_kshiftri_mask64(a, count) _kshiftri_mask64(a, count)
#endif
#if defined(EASYSIMD_X86_AVX512BW_ENABLE_NATIVE_ALIASES)
  #undef _kshiftri_mask64
  #define _kshiftri_mask64(a, count) easysimd_kshiftri_mask64(a, count)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd__mmask8
easysimd_kshiftri_mask8 (easysimd__mmask8 a, unsigned int count)
    EASYSIMD_REQUIRE_CONSTANT_RANGE(count, 0, 255) {
  return HEDLEY_STATIC_CAST(easysimd__mmask8, (count <= 7) ? (a >> count) : 0);
}
#if defined(EASYSIMD_X86_AVX512DQ_NATIVE) && (!defined(HEDLEY_GCC_VERSION) || HEDLEY_GCC_VERSION_CHECK(7,0,0)) && (!defined(EASYSIMD_DETECT_CLANG_VERSION) && EASYSIMD_DETECT_CLANG_VERSION_CHECK(8,0,0))
  #define easysimd_kshiftri_mask8(a, count) _kshiftri_mask8(a, count)
#endif
#if defined(EASYSIMD_X86_AVX512DQ_ENABLE_NATIVE_ALIASES)
  #undef _kshiftri_mask8
  #define _kshiftri_mask8(a, count) easysimd_kshiftri_mask8(a, count)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_X86_AVX512_KSHIFT_H) */
