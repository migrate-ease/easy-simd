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

#if !defined(EASYSIMD_ARM_NEON_QNEG_H)
#define EASYSIMD_ARM_NEON_QNEG_H

#include "types.h"

#if !defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE) || 1
  #include "dup_n.h"
  #include "max.h"
  #include "neg.h"
#endif

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

EASYSIMD_FUNCTION_ATTRIBUTES
int8_t
easysimd_vqnegb_s8(int8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqnegb_s8(a);
  #else
    return a == INT8_MIN ? INT8_MAX : -a;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqnegb_s8
  #define vqnegb_s8(a) easysimd_vqnegb_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int16_t
easysimd_vqnegh_s16(int16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqnegh_s16(a);
  #else
    return a == INT16_MIN ? INT16_MAX : -a;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqnegh_s16
  #define vqnegh_s16(a) easysimd_vqnegh_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int32_t
easysimd_vqnegs_s32(int32_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqnegs_s32(a);
  #else
    return a == INT32_MIN ? INT32_MAX : -a;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqnegs_s32
  #define vqnegs_s32(a) easysimd_vqnegs_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
int64_t
easysimd_vqnegd_s64(int64_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqnegd_s64(a);
  #else
    return a == INT64_MIN ? INT64_MAX : -a;
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vqnegd_s64
  #define vqnegd_s64(a) easysimd_vqnegd_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x8_t
easysimd_vqneg_s8(easysimd_int8x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqneg_s8(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(64)
    return easysimd_vneg_s8(easysimd_vmax_s8(a, easysimd_vdup_n_s8(INT8_MIN + 1)));
  #else
    easysimd_int8x8_private
      r_,
      a_ = easysimd_int8x8_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] == INT8_MIN) ? INT8_MAX : -(a_.values[i]);
    }

    return easysimd_int8x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqneg_s8
  #define vqneg_s8(a) easysimd_vqneg_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x4_t
easysimd_vqneg_s16(easysimd_int16x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqneg_s16(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(64)
    return easysimd_vneg_s16(easysimd_vmax_s16(a, easysimd_vdup_n_s16(INT16_MIN + 1)));
  #else
    easysimd_int16x4_private
      r_,
      a_ = easysimd_int16x4_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] == INT16_MIN) ? INT16_MAX : -(a_.values[i]);
    }

    return easysimd_int16x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqneg_s16
  #define vqneg_s16(a) easysimd_vqneg_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x2_t
easysimd_vqneg_s32(easysimd_int32x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqneg_s32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(64)
    return easysimd_vneg_s32(easysimd_vmax_s32(a, easysimd_vdup_n_s32(INT32_MIN + 1)));
  #else
    easysimd_int32x2_private
      r_,
      a_ = easysimd_int32x2_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] == INT32_MIN) ? INT32_MAX : -(a_.values[i]);
    }

    return easysimd_int32x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqneg_s32
  #define vqneg_s32(a) easysimd_vqneg_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x1_t
easysimd_vqneg_s64(easysimd_int64x1_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqneg_s64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_vneg_s64(easysimd_x_vmax_s64(a, easysimd_vdup_n_s64(INT64_MIN + 1)));
  #else
    easysimd_int64x1_private
      r_,
      a_ = easysimd_int64x1_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] == INT64_MIN) ? INT64_MAX : -(a_.values[i]);
    }

    return easysimd_int64x1_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqneg_s64
  #define vqneg_s64(a) easysimd_vqneg_s64(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int8x16_t
easysimd_vqnegq_s8(easysimd_int8x16_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqnegq_s8(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_vnegq_s8(easysimd_vmaxq_s8(a, easysimd_vdupq_n_s8(INT8_MIN + 1)));
  #else
    easysimd_int8x16_private
      r_,
      a_ = easysimd_int8x16_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] == INT8_MIN) ? INT8_MAX : -(a_.values[i]);
    }

    return easysimd_int8x16_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqnegq_s8
  #define vqnegq_s8(a) easysimd_vqnegq_s8(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int16x8_t
easysimd_vqnegq_s16(easysimd_int16x8_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqnegq_s16(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_vnegq_s16(easysimd_vmaxq_s16(a, easysimd_vdupq_n_s16(INT16_MIN + 1)));
  #else
    easysimd_int16x8_private
      r_,
      a_ = easysimd_int16x8_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] == INT16_MIN) ? INT16_MAX : -(a_.values[i]);
    }

    return easysimd_int16x8_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqnegq_s16
  #define vqnegq_s16(a) easysimd_vqnegq_s16(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int32x4_t
easysimd_vqnegq_s32(easysimd_int32x4_t a) {
  #if defined(EASYSIMD_ARM_NEON_A32V7_NATIVE)
    return vqnegq_s32(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_vnegq_s32(easysimd_vmaxq_s32(a, easysimd_vdupq_n_s32(INT32_MIN + 1)));
  #else
    easysimd_int32x4_private
      r_,
      a_ = easysimd_int32x4_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] == INT32_MIN) ? INT32_MAX : -(a_.values[i]);
    }

    return easysimd_int32x4_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqnegq_s32
  #define vqnegq_s32(a) easysimd_vqnegq_s32(a)
#endif

EASYSIMD_FUNCTION_ATTRIBUTES
easysimd_int64x2_t
easysimd_vqnegq_s64(easysimd_int64x2_t a) {
  #if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    return vqnegq_s64(a);
  #elif EASYSIMD_NATURAL_VECTOR_SIZE_GE(128)
    return easysimd_vnegq_s64(easysimd_x_vmaxq_s64(a, easysimd_vdupq_n_s64(INT64_MIN + 1)));
  #else
    easysimd_int64x2_private
      r_,
      a_ = easysimd_int64x2_to_private(a);

    EASYSIMD_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = (a_.values[i] == INT64_MIN) ? INT64_MAX : -(a_.values[i]);
    }

    return easysimd_int64x2_from_private(r_);
  #endif
}
#if defined(EASYSIMD_ARM_NEON_A32V7_ENABLE_NATIVE_ALIASES)
  #undef vqnegq_s64
  #define vqnegq_s64(a) easysimd_vqnegq_s64(a)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_QNEG_H) */
