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

#if !defined(EASYSIMD_ARM_SVE_WHILELT_H)
#define EASYSIMD_ARM_SVE_WHILELT_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b8_s32(int32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b8_s32(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask64(HEDLEY_STATIC_CAST(__mmask64, 0));

    int_fast32_t remaining = (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));
    __mmask64 r = ~HEDLEY_STATIC_CAST(__mmask64, 0);
    if (HEDLEY_UNLIKELY(remaining < 64)) {
      r >>= 64 - remaining;
    }

    return easysimd_svbool_from_mmask64(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask32(HEDLEY_STATIC_CAST(__mmask32, 0));

    int_fast32_t remaining = (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));
    __mmask32 r = HEDLEY_STATIC_CAST(__mmask32, ~UINT32_C(0));
    if (HEDLEY_UNLIKELY(remaining < 32)) {
      r >>= 32 - remaining;
    }

    return easysimd_svbool_from_mmask32(r);
  #else
    easysimd_svint8_t r;

    int_fast32_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntb()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT8_C(0) : UINT8_C(0);
    }

    return easysimd_svbool_from_svint8(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b8_s32
  #define svwhilelt_b8_s32(op1, op2) easysimd_svwhilelt_b8_s32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b16_s32(int32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b16_s32(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask32(HEDLEY_STATIC_CAST(__mmask32, 0));

    int_fast32_t remaining = (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));
    __mmask32 r = HEDLEY_STATIC_CAST(__mmask32, ~UINT32_C(0));
    if (HEDLEY_UNLIKELY(remaining < 32)) {
      r >>= 32 - remaining;
    }

    return easysimd_svbool_from_mmask32(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask16(HEDLEY_STATIC_CAST(__mmask16, 0));

    int_fast32_t remaining = (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));
    __mmask16 r = HEDLEY_STATIC_CAST(__mmask16, ~UINT16_C(0));
    if (HEDLEY_UNLIKELY(remaining < 16)) {
      r >>= 16 - remaining;
    }

    return easysimd_svbool_from_mmask16(r);
  #else
    easysimd_svint16_t r;

    int_fast32_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcnth()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT16_C(0) : UINT16_C(0);
    }

    return easysimd_svbool_from_svint16(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b16_s32
  #define svwhilelt_b16_s32(op1, op2) easysimd_svwhilelt_b16_s32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b32_s32(int32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b32_s32(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask16(HEDLEY_STATIC_CAST(__mmask16, 0));

    int_fast32_t remaining = (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));
    __mmask16 r = HEDLEY_STATIC_CAST(__mmask16, ~UINT16_C(0));
    if (HEDLEY_UNLIKELY(remaining < 16)) {
      r >>= 16 - remaining;
    }

    return easysimd_svbool_from_mmask16(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask8(HEDLEY_STATIC_CAST(__mmask8, 0));

    int_fast32_t remaining = (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, ~UINT8_C(0));
    if (HEDLEY_UNLIKELY(remaining < 8)) {
      r >>= 8 - remaining;
    }

    return easysimd_svbool_from_mmask8(r);
  #else
    easysimd_svint32_t r;

    int_fast32_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntw()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~INT32_C(0) : INT32_C(0);
    }

    return easysimd_svbool_from_svint32(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b32_s32
  #define svwhilelt_b32_s32(op1, op2) easysimd_svwhilelt_b32_s32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b64_s32(int32_t op1, int32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b64_s32(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask8(HEDLEY_STATIC_CAST(__mmask8, 0));

    int_fast32_t remaining = (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, ~UINT8_C(0));
    if (HEDLEY_UNLIKELY(remaining < 8)) {
      r >>= 8 - remaining;
    }

    return easysimd_svbool_from_mmask8(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask4(HEDLEY_STATIC_CAST(__mmask8, 0));

    int_fast32_t remaining = (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, 0x0f);
    if (HEDLEY_UNLIKELY(remaining < 4)) {
      r >>= 4 - remaining;
    }

    return easysimd_svbool_from_mmask4(r);
  #else
    easysimd_svint64_t r;

    int_fast32_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(int_fast32_t, op2) - HEDLEY_STATIC_CAST(int_fast32_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntd()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~INT64_C(0) : INT64_C(0);
    }

    return easysimd_svbool_from_svint64(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b64_s32
  #define svwhilelt_b64_s32(op1, op2) easysimd_svwhilelt_b64_s32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b8_s64(int64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b8_s64(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask64(HEDLEY_STATIC_CAST(__mmask64, 0));

    int_fast64_t remaining = (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));
    __mmask64 r = ~HEDLEY_STATIC_CAST(__mmask64, 0);
    if (HEDLEY_UNLIKELY(remaining < 64)) {
      r >>= 64 - remaining;
    }

    return easysimd_svbool_from_mmask64(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask32(HEDLEY_STATIC_CAST(__mmask32, 0));

    int_fast64_t remaining = (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));
    __mmask32 r = HEDLEY_STATIC_CAST(__mmask32, ~UINT32_C(0));
    if (HEDLEY_UNLIKELY(remaining < 32)) {
      r >>= 32 - remaining;
    }

    return easysimd_svbool_from_mmask32(r);
  #else
    easysimd_svint8_t r;

    int_fast64_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntb()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT8_C(0) : UINT8_C(0);
    }

    return easysimd_svbool_from_svint8(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b8_s64
  #define svwhilelt_b8_s64(op1, op2) easysimd_svwhilelt_b8_s64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b16_s64(int64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b16_s64(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask32(HEDLEY_STATIC_CAST(__mmask32, 0));

    int_fast64_t remaining = (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));
    __mmask32 r = HEDLEY_STATIC_CAST(__mmask32, ~UINT64_C(0));
    if (HEDLEY_UNLIKELY(remaining < 32)) {
      r >>= 32 - remaining;
    }

    return easysimd_svbool_from_mmask32(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask16(HEDLEY_STATIC_CAST(__mmask16, 0));

    int_fast64_t remaining = (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));
    __mmask16 r = HEDLEY_STATIC_CAST(__mmask16, ~UINT16_C(0));
    if (HEDLEY_UNLIKELY(remaining < 16)) {
      r >>= 16 - remaining;
    }

    return easysimd_svbool_from_mmask16(r);
  #else
    easysimd_svint16_t r;

    int_fast64_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcnth()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT16_C(0) : UINT16_C(0);
    }

    return easysimd_svbool_from_svint16(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b16_s64
  #define svwhilelt_b16_s64(op1, op2) easysimd_svwhilelt_b16_s64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b32_s64(int64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b32_s64(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask16(HEDLEY_STATIC_CAST(__mmask16, 0));

    int_fast64_t remaining = (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));
    __mmask16 r = HEDLEY_STATIC_CAST(__mmask16, ~UINT16_C(0));
    if (HEDLEY_UNLIKELY(remaining < 16)) {
      r >>= 16 - remaining;
    }

    return easysimd_svbool_from_mmask16(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask8(HEDLEY_STATIC_CAST(__mmask8, 0));

    int_fast64_t remaining = (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, ~UINT8_C(0));
    if (HEDLEY_UNLIKELY(remaining < 8)) {
      r >>= 8 - remaining;
    }

    return easysimd_svbool_from_mmask8(r);
  #else
    easysimd_svint64_t r;

    int_fast64_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntw()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~INT64_C(0) : INT64_C(0);
    }

    return easysimd_svbool_from_svint64(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b32_s64
  #define svwhilelt_b32_s64(op1, op2) easysimd_svwhilelt_b32_s64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b64_s64(int64_t op1, int64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b64_s64(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask8(HEDLEY_STATIC_CAST(__mmask8, 0));

    int_fast64_t remaining = (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, ~UINT8_C(0));
    if (HEDLEY_UNLIKELY(remaining < 8)) {
      r >>= 8 - remaining;
    }

    return easysimd_svbool_from_mmask8(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask4(HEDLEY_STATIC_CAST(__mmask8, 0));

    int_fast64_t remaining = (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, 0x0f);
    if (HEDLEY_UNLIKELY(remaining < 4)) {
      r >>= 4 - remaining;
    }

    return easysimd_svbool_from_mmask4(r);
  #else
    easysimd_svint64_t r;

    int_fast64_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(int_fast64_t, op2) - HEDLEY_STATIC_CAST(int_fast64_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntd()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~INT64_C(0) : INT64_C(0);
    }

    return easysimd_svbool_from_svint64(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b64_s64
  #define svwhilelt_b64_s64(op1, op2) easysimd_svwhilelt_b64_s64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b8_u32(uint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b8_u32(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask64(HEDLEY_STATIC_CAST(__mmask64, 0));

    uint_fast32_t remaining = (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));
    __mmask64 r = ~HEDLEY_STATIC_CAST(__mmask64, 0);
    if (HEDLEY_UNLIKELY(remaining < 64)) {
      r >>= 64 - remaining;
    }

    return easysimd_svbool_from_mmask64(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask32(HEDLEY_STATIC_CAST(__mmask32, 0));

    uint_fast32_t remaining = (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));
    __mmask32 r = HEDLEY_STATIC_CAST(__mmask32, ~UINT32_C(0));
    if (HEDLEY_UNLIKELY(remaining < 32)) {
      r >>= 32 - remaining;
    }

    return easysimd_svbool_from_mmask32(r);
  #else
    easysimd_svint8_t r;

    uint_fast32_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntb()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT8_C(0) : UINT8_C(0);
    }

    return easysimd_svbool_from_svint8(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b8_u32
  #define svwhilelt_b8_u32(op1, op2) easysimd_svwhilelt_b8_u32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b16_u32(uint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b16_u32(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask32(HEDLEY_STATIC_CAST(__mmask32, 0));

    uint_fast32_t remaining = (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));
    __mmask32 r = HEDLEY_STATIC_CAST(__mmask32, ~UINT32_C(0));
    if (HEDLEY_UNLIKELY(remaining < 32)) {
      r >>= 32 - remaining;
    }

    return easysimd_svbool_from_mmask32(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask16(HEDLEY_STATIC_CAST(__mmask16, 0));

    uint_fast32_t remaining = (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));
    __mmask16 r = HEDLEY_STATIC_CAST(__mmask16, ~UINT16_C(0));
    if (HEDLEY_UNLIKELY(remaining < 16)) {
      r >>= 16 - remaining;
    }

    return easysimd_svbool_from_mmask16(r);
  #else
    easysimd_svint16_t r;

    uint_fast32_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcnth()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT16_C(0) : UINT16_C(0);
    }

    return easysimd_svbool_from_svint16(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b16_u32
  #define svwhilelt_b16_u32(op1, op2) easysimd_svwhilelt_b16_u32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b32_u32(uint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b32_u32(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask16(HEDLEY_STATIC_CAST(__mmask16, 0));

    uint_fast32_t remaining = (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));
    __mmask16 r = HEDLEY_STATIC_CAST(__mmask16, ~UINT16_C(0));
    if (HEDLEY_UNLIKELY(remaining < 16)) {
      r >>= 16 - remaining;
    }

    return easysimd_svbool_from_mmask16(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask8(HEDLEY_STATIC_CAST(__mmask8, 0));

    uint_fast32_t remaining = (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, ~UINT8_C(0));
    if (HEDLEY_UNLIKELY(remaining < 8)) {
      r >>= 8 - remaining;
    }

    return easysimd_svbool_from_mmask8(r);
  #else
    easysimd_svuint32_t r;

    uint_fast32_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntw()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT32_C(0) : UINT32_C(0);
    }

    return easysimd_svbool_from_svuint32(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b32_u32
  #define svwhilelt_b32_u32(op1, op2) easysimd_svwhilelt_b32_u32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b64_u32(uint32_t op1, uint32_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b64_u32(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask8(HEDLEY_STATIC_CAST(__mmask8, 0));

    uint_fast32_t remaining = (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, ~UINT8_C(0));
    if (HEDLEY_UNLIKELY(remaining < 8)) {
      r >>= 8 - remaining;
    }

    return easysimd_svbool_from_mmask8(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask4(HEDLEY_STATIC_CAST(__mmask8, 0));

    uint_fast32_t remaining = (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, 0x0f);
    if (HEDLEY_UNLIKELY(remaining < 4)) {
      r >>= 4 - remaining;
    }

    return easysimd_svbool_from_mmask4(r);
  #else
    easysimd_svint64_t r;

    uint_fast32_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(uint_fast32_t, op2) - HEDLEY_STATIC_CAST(uint_fast32_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntd()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~INT64_C(0) : INT64_C(0);
    }

    return easysimd_svbool_from_svint64(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b64_u32
  #define svwhilelt_b64_u32(op1, op2) easysimd_svwhilelt_b64_u32(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b8_u64(uint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b8_u64(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask64(HEDLEY_STATIC_CAST(__mmask64, 0));

    uint_fast64_t remaining = (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));
    __mmask64 r = ~HEDLEY_STATIC_CAST(__mmask64, 0);
    if (HEDLEY_UNLIKELY(remaining < 64)) {
      r >>= 64 - remaining;
    }

    return easysimd_svbool_from_mmask64(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask32(HEDLEY_STATIC_CAST(__mmask32, 0));

    uint_fast64_t remaining = (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));
    __mmask32 r = HEDLEY_STATIC_CAST(__mmask32, ~UINT64_C(0));
    if (HEDLEY_UNLIKELY(remaining < 32)) {
      r >>= 32 - remaining;
    }

    return easysimd_svbool_from_mmask32(r);
  #else
    easysimd_svint8_t r;

    uint_fast64_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntb()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT8_C(0) : UINT8_C(0);
    }

    return easysimd_svbool_from_svint8(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b8_u64
  #define svwhilelt_b8_u64(op1, op2) easysimd_svwhilelt_b8_u64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b16_u64(uint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b16_u64(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask32(HEDLEY_STATIC_CAST(__mmask32, 0));

    uint_fast64_t remaining = (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));
    __mmask32 r = HEDLEY_STATIC_CAST(__mmask32, ~UINT64_C(0));
    if (HEDLEY_UNLIKELY(remaining < 32)) {
      r >>= 32 - remaining;
    }

    return easysimd_svbool_from_mmask32(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask16(HEDLEY_STATIC_CAST(__mmask16, 0));

    uint_fast64_t remaining = (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));
    __mmask16 r = HEDLEY_STATIC_CAST(__mmask16, ~UINT16_C(0));
    if (HEDLEY_UNLIKELY(remaining < 16)) {
      r >>= 16 - remaining;
    }

    return easysimd_svbool_from_mmask16(r);
  #else
    easysimd_svint16_t r;

    uint_fast64_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcnth()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT16_C(0) : UINT16_C(0);
    }

    return easysimd_svbool_from_svint16(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b16_u64
  #define svwhilelt_b16_u64(op1, op2) easysimd_svwhilelt_b16_u64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b32_u64(uint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b32_u64(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask16(HEDLEY_STATIC_CAST(__mmask16, 0));

    uint_fast64_t remaining = (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));
    __mmask16 r = HEDLEY_STATIC_CAST(__mmask16, ~UINT16_C(0));
    if (HEDLEY_UNLIKELY(remaining < 16)) {
      r >>= 16 - remaining;
    }

    return easysimd_svbool_from_mmask16(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask8(HEDLEY_STATIC_CAST(__mmask8, 0));

    uint_fast64_t remaining = (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, ~UINT8_C(0));
    if (HEDLEY_UNLIKELY(remaining < 8)) {
      r >>= 8 - remaining;
    }

    return easysimd_svbool_from_mmask8(r);
  #else
    easysimd_svuint64_t r;

    uint_fast64_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntw()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~UINT64_C(0) : UINT64_C(0);
    }

    return easysimd_svbool_from_svuint64(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b32_u64
  #define svwhilelt_b32_u64(op1, op2) easysimd_svwhilelt_b32_u64(op1, op2)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_svbool_t
easysimd_svwhilelt_b64_u64(uint64_t op1, uint64_t op2) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE)
    return svwhilelt_b64_u64(op1, op2);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && (EASYSIMD_ARM_SVE_VECTOR_SIZE >= 512)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask8(HEDLEY_STATIC_CAST(__mmask8, 0));

    uint_fast64_t remaining = (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, ~UINT8_C(0));
    if (HEDLEY_UNLIKELY(remaining < 8)) {
      r >>= 8 - remaining;
    }

    return easysimd_svbool_from_mmask8(r);
  #elif defined(EASYSIMD_X86_AVX512BW_NATIVE) && defined(EASYSIMD_X86_AVX512VL_NATIVE)
    if (HEDLEY_UNLIKELY(op1 >= op2))
      return easysimd_svbool_from_mmask4(HEDLEY_STATIC_CAST(__mmask8, 0));

    uint_fast64_t remaining = (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));
    __mmask8 r = HEDLEY_STATIC_CAST(__mmask8, 0x0f);
    if (HEDLEY_UNLIKELY(remaining < 4)) {
      r >>= 4 - remaining;
    }

    return easysimd_svbool_from_mmask4(r);
  #else
    easysimd_svint64_t r;

    uint_fast64_t remaining = (op1 >= op2) ? 0 : (HEDLEY_STATIC_CAST(uint_fast64_t, op2) - HEDLEY_STATIC_CAST(uint_fast64_t, op1));

    EASYSIMD_VECTORIZE
    for (int i = 0 ; i < HEDLEY_STATIC_CAST(int, easysimd_svcntd()) ; i++) {
      r.values[i] = (remaining-- > 0) ? ~INT64_C(0) : INT64_C(0);
    }

    return easysimd_svbool_from_svint64(r);
  #endif
}
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef easysimd_svwhilelt_b64_u64
  #define svwhilelt_b64_u64(op1, op2) easysimd_svwhilelt_b64_u64(op1, op2)
#endif

#if defined(__cplusplus)
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b8 ( int32_t op1,  int32_t op2) { return  easysimd_svwhilelt_b8_s32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b8 ( int64_t op1,  int64_t op2) { return  easysimd_svwhilelt_b8_s64(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b8 (uint32_t op1, uint32_t op2) { return  easysimd_svwhilelt_b8_u32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b8 (uint64_t op1, uint64_t op2) { return  easysimd_svwhilelt_b8_u64(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b16( int32_t op1,  int32_t op2) { return easysimd_svwhilelt_b16_s32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b16( int64_t op1,  int64_t op2) { return easysimd_svwhilelt_b16_s64(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b16(uint32_t op1, uint32_t op2) { return easysimd_svwhilelt_b16_u32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b16(uint64_t op1, uint64_t op2) { return easysimd_svwhilelt_b16_u64(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b32( int32_t op1,  int32_t op2) { return easysimd_svwhilelt_b32_s32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b32( int64_t op1,  int64_t op2) { return easysimd_svwhilelt_b32_s64(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b32(uint32_t op1, uint32_t op2) { return easysimd_svwhilelt_b32_u32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b32(uint64_t op1, uint64_t op2) { return easysimd_svwhilelt_b32_u64(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b64( int32_t op1,  int32_t op2) { return easysimd_svwhilelt_b64_s32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b64( int64_t op1,  int64_t op2) { return easysimd_svwhilelt_b64_s64(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b64(uint32_t op1, uint32_t op2) { return easysimd_svwhilelt_b64_u32(op1, op2); }
  EASYSIMD_FUNCTION_ATTRIBUTES easysimd_svbool_t easysimd_svwhilelt_b64(uint64_t op1, uint64_t op2) { return easysimd_svwhilelt_b64_u64(op1, op2); }
#elif defined(EASYSIMD_GENERIC_)
  #define easysimd_svwhilelt_b8(op1, op2) \
    (EASYSIMD_GENERIC_((op1), \
       int32_t: easysimd_svwhilelt_b8_s32, \
      uint32_t: easysimd_svwhilelt_b8_u32, \
       int64_t: easysimd_svwhilelt_b8_s64, \
      uint64_t: easysimd_svwhilelt_b8_u64)((op1), (op2)))
  #define easysimd_svwhilelt_b16(op1, op2) \
    (EASYSIMD_GENERIC_((op1), \
       int32_t: easysimd_svwhilelt_b16_s32, \
      uint32_t: easysimd_svwhilelt_b16_u32, \
       int64_t: easysimd_svwhilelt_b16_s64, \
      uint64_t: easysimd_svwhilelt_b16_u64)((op1), (op2)))
  #define easysimd_svwhilelt_b32(op1, op2) \
    (EASYSIMD_GENERIC_((op1), \
       int32_t: easysimd_svwhilelt_b32_s32, \
      uint32_t: easysimd_svwhilelt_b32_u32, \
       int64_t: easysimd_svwhilelt_b32_s64, \
      uint64_t: easysimd_svwhilelt_b32_u64)((op1), (op2)))
  #define easysimd_svwhilelt_b64(op1, op2) \
    (EASYSIMD_GENERIC_((op1), \
       int32_t: easysimd_svwhilelt_b64_s32, \
      uint32_t: easysimd_svwhilelt_b64_u32, \
       int64_t: easysimd_svwhilelt_b64_s64, \
      uint64_t: easysimd_svwhilelt_b64_u64)((op1), (op2)))
#endif
#if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #undef svwhilelt_b8
  #undef svwhilelt_b16
  #undef svwhilelt_b32
  #undef svwhilelt_b64
  #define svwhilelt_b8(op1, op2) easysimd_svwhilelt_b8((op1), (op2))
  #define svwhilelt_b16(op1, op2) easysimd_svwhilelt_b16((op1), (op2))
  #define svwhilelt_b32(op1, op2) easysimd_svwhilelt_b32((op1), (op2))
  #define svwhilelt_b64(op1, op2) easysimd_svwhilelt_b64((op1), (op2))
#endif

HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_SVE_WHILELT_H */
