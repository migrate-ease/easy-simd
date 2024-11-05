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
 *   2021      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_ARM_SVE_CNT_H)
#define EASYSIMD_ARM_SVE_CNT_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

#if defined(EASYSIMD_ARM_SVE_128_PIPELINE)
#define easysimd_svcntb() 16
#define easysimd_svcnth() 8
#define easysimd_svcntw() 4
#define easysimd_svcntd() 2
#else
EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_svcntb(void) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcntb();
  #else
    return sizeof(easysimd_svint8_t) / sizeof(int8_t);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcntb
  #define svcntb() easysimd_svcntb()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_svcnth(void) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcnth();
  #else
    return sizeof(easysimd_svint16_t) / sizeof(int16_t);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcnth
  #define svcnth() easysimd_svcnth()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_svcntw(void) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcntw();
  #else
    return sizeof(easysimd_svint32_t) / sizeof(int32_t);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcntw
  #define svcntw() easysimd_svcntw()
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
uint64_t
easysimd_svcntd(void) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svcntd();
  #else
    return sizeof(easysimd_svint64_t) / sizeof(int64_t);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svcntd
  #define svcntd() easysimd_svcntd()
#endif

#endif

HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_SVE_CNT_H */
