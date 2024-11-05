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
*   2021      Atharva Nimbalkar <atharvakn@gmail.com>
*/

#if !defined(EASYSIMD_ARM_NEON_FMA_LANE_H)
#define EASYSIMD_ARM_NEON_FMA_LANE_H

#include "add.h"
#include "dup_n.h"
#include "get_lane.h"
#include "mul.h"
#include "mul_lane.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS
EASYSIMD_BEGIN_DECLS_

/* easysimd_vfmad_lane_f64 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(11,0,0)
    #define easysimd_vfmad_lane_f64(a, b, v, lane) \
    EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vfmad_lane_f64(a, b, v, lane))
  #else
    #define easysimd_vfmad_lane_f64(a, b, v, lane) vfmad_lane_f64((a), (b), (v), (lane))
  #endif
#else
  #define easysimd_vfmad_lane_f64(a, b, v, lane) \
  easysimd_vget_lane_f64( \
    easysimd_vadd_f64( \
      easysimd_vdup_n_f64(a), \
      easysimd_vdup_n_f64(easysimd_vmuld_lane_f64(b, v, lane)) \
    ), \
    0 \
  )
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfmad_lane_f64
  #define vfmad_lane_f64(a, b, v, lane) easysimd_vfmad_lane_f64(a, b, v, lane)
#endif

/* easysimd_vfmad_laneq_f64 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(11,0,0)
    #define easysimd_vfmad_laneq_f64(a, b, v, lane) \
    EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vfmad_laneq_f64(a, b, v, lane))
  #else
    #define easysimd_vfmad_laneq_f64(a, b, v, lane) vfmad_laneq_f64((a), (b), (v), (lane))
  #endif
#else
  #define easysimd_vfmad_laneq_f64(a, b, v, lane) \
  easysimd_vget_lane_f64( \
    easysimd_vadd_f64( \
      easysimd_vdup_n_f64(a), \
      easysimd_vdup_n_f64(easysimd_vmuld_laneq_f64(b, v, lane)) \
    ), \
    0 \
  )
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfmad_laneq_f64
  #define vfmad_laneq_f64(a, b, v, lane) easysimd_vfmad_laneq_f64(a, b, v, lane)
#endif

/* easysimd_vfmas_lane_f32 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(11,0,0)
    #define easysimd_vfmas_lane_f32(a, b, v, lane) \
    EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vfmas_lane_f32(a, b, v, lane))
  #else
    #define easysimd_vfmas_lane_f32(a, b, v, lane) vfmas_lane_f32((a), (b), (v), (lane))
  #endif
#else
  #define easysimd_vfmas_lane_f32(a, b, v, lane) \
  easysimd_vget_lane_f32( \
    easysimd_vadd_f32( \
      easysimd_vdup_n_f32(a), \
      easysimd_vdup_n_f32(easysimd_vmuls_lane_f32(b, v, lane)) \
    ), \
    0 \
  )
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfmas_lane_f32
  #define vfmas_lane_f32(a, b, v, lane) easysimd_vfmas_lane_f32(a, b, v, lane)
#endif

/* easysimd_vfmas_laneq_f32 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #if defined(__clang__) && !EASYSIMD_DETECT_CLANG_VERSION_CHECK(11,0,0)
    #define easysimd_vfmas_laneq_f32(a, b, v, lane) \
    EASYSIMD_DISABLE_DIAGNOSTIC_EXPR_(EASYSIMD_DIAGNOSTIC_DISABLE_VECTOR_CONVERSION_, vfmas_laneq_f32(a, b, v, lane))
  #else
    #define easysimd_vfmas_laneq_f32(a, b, v, lane) vfmas_laneq_f32((a), (b), (v), (lane))
  #endif
#else
  #define easysimd_vfmas_laneq_f32(a, b, v, lane) \
  easysimd_vget_lane_f32( \
    easysimd_vadd_f32( \
      easysimd_vdup_n_f32(a), \
      easysimd_vdup_n_f32(easysimd_vmuls_laneq_f32(b, v, lane)) \
    ), \
    0 \
  )
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfmas_laneq_f32
  #define vfmas_laneq_f32(a, b, v, lane) easysimd_vfmas_laneq_f32(a, b, v, lane)
#endif

/* easysimd_vfma_lane_f32 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #define easysimd_vfma_lane_f32(a, b, v, lane) vfma_lane_f32(a, b, v, lane)
#else
  #define easysimd_vfma_lane_f32(a, b, v, lane) easysimd_vadd_f32(a, easysimd_vmul_lane_f32(b, v, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfma_lane_f32
  #define vfma_lane_f32(a, b, v, lane) easysimd_vfma_lane_f32(a, b, v, lane)
#endif

/* easysimd_vfma_lane_f64 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #define easysimd_vfma_lane_f64(a, b, v, lane) vfma_lane_f64((a), (b), (v), (lane))
#else
  #define easysimd_vfma_lane_f64(a, b, v, lane) easysimd_vadd_f64(a, easysimd_vmul_lane_f64(b, v, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfma_lane_f64
  #define vfma_lane_f64(a, b, v, lane) easysimd_vfma_lane_f64(a, b, v, lane)
#endif

/* easysimd_vfma_laneq_f32 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #define easysimd_vfma_laneq_f32(a, b, v, lane) vfma_laneq_f32((a), (b), (v), (lane))
#else
  #define easysimd_vfma_laneq_f32(a, b, v, lane) easysimd_vadd_f32(a, easysimd_vmul_laneq_f32(b, v, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfma_laneq_f32
  #define vfma_laneq_f32(a, b, v, lane) easysimd_vfma_laneq_f32(a, b, v, lane)
#endif

/* easysimd_vfma_laneq_f64 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #define easysimd_vfma_laneq_f64(a, b, v, lane) vfma_laneq_f64((a), (b), (v), (lane))
#else
  #define easysimd_vfma_laneq_f64(a, b, v, lane) easysimd_vadd_f64(a, easysimd_vmul_laneq_f64(b, v, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfma_laneq_f64
  #define vfma_laneq_f64(a, b, v, lane) easysimd_vfma_laneq_f64(a, b, v, lane)
#endif

/* easysimd_vfmaq_lane_f64 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #define easysimd_vfmaq_lane_f64(a, b, v, lane) vfmaq_lane_f64((a), (b), (v), (lane))
#else
  #define easysimd_vfmaq_lane_f64(a, b, v, lane) easysimd_vaddq_f64(a, easysimd_vmulq_lane_f64(b, v, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfmaq_lane_f64
  #define vfmaq_lane_f64(a, b, v, lane) easysimd_vfmaq_lane_f64(a, b, v, lane)
#endif

/* easysimd_vfmaq_lane_f32 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #define easysimd_vfmaq_lane_f32(a, b, v, lane) vfmaq_lane_f32((a), (b), (v), (lane))
#else
  #define easysimd_vfmaq_lane_f32(a, b, v, lane) easysimd_vaddq_f32(a, easysimd_vmulq_lane_f32(b, v, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfmaq_lane_f32
  #define vfmaq_lane_f32(a, b, v, lane) easysimd_vfmaq_lane_f32(a, b, v, lane)
#endif

/* easysimd_vfmaq_laneq_f32 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #define easysimd_vfmaq_laneq_f32(a, b, v, lane) vfmaq_laneq_f32((a), (b), (v), (lane))
#else
  #define easysimd_vfmaq_laneq_f32(a, b, v, lane) \
  easysimd_vaddq_f32(a, easysimd_vmulq_laneq_f32(b, v, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfmaq_laneq_f32
  #define vfmaq_laneq_f32(a, b, v, lane) easysimd_vfmaq_laneq_f32(a, b, v, lane)
#endif

/* easysimd_vfmaq_laneq_f64 */
#if defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) && (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
  #define easysimd_vfmaq_laneq_f64(a, b, v, lane) vfmaq_laneq_f64((a), (b), (v), (lane))
#else
  #define easysimd_vfmaq_laneq_f64(a, b, v, lane) \
  easysimd_vaddq_f64(a, easysimd_vmulq_laneq_f64(b, v, lane))
#endif
#if defined(EASYSIMD_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vfmaq_laneq_f64
  #define vfmaq_laneq_f64(a, b, v, lane) easysimd_vfmaq_laneq_f64(a, b, v, lane)
#endif

EASYSIMD_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(EASYSIMD_ARM_NEON_FMA_LANE_H) */
