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

#define EASYSIMD_TEST_X86_AVX512_INSN setzero

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/set1.h>
#include <easysimd/x86/avx512/setzero.h>

static int
test_easysimd_mm512_setzero_si512(EASYSIMD_MUNIT_TEST_ARGS) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512i r;
    easysimd__m512i a =
      easysimd_mm512_set_epi32(INT32_C(0), INT32_C(0), INT32_C(0), INT32_C(0),
                            INT32_C(0), INT32_C(0), INT32_C(0), INT32_C(0),
                            INT32_C(0), INT32_C(0), INT32_C(0), INT32_C(0),
                            INT32_C(0), INT32_C(0), INT32_C(0), INT32_C(0));
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_setzero_si512();
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_setzero_si512");
    easysimd_assert_m512i_i32(a, ==, r);
  #else
    easysimd_assert_m512i_i32(easysimd_mm512_setzero_si512(), ==, easysimd_mm512_set1_epi32(INT32_C(0)));
  #endif
  return 0;
}

static int
test_easysimd_mm512_setzero_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512 r;
    easysimd__m512 a =
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    0.00), EASYSIMD_FLOAT32_C(    0.00),
                         EASYSIMD_FLOAT32_C(    0.00), EASYSIMD_FLOAT32_C(    0.00),
                         EASYSIMD_FLOAT32_C(    0.00), EASYSIMD_FLOAT32_C(    0.00),
                         EASYSIMD_FLOAT32_C(    0.00), EASYSIMD_FLOAT32_C(    0.00),
                         EASYSIMD_FLOAT32_C(    0.00), EASYSIMD_FLOAT32_C(    0.00),
                         EASYSIMD_FLOAT32_C(    0.00), EASYSIMD_FLOAT32_C(    0.00),
                         EASYSIMD_FLOAT32_C(    0.00), EASYSIMD_FLOAT32_C(    0.00),
                         EASYSIMD_FLOAT32_C(    0.00), EASYSIMD_FLOAT32_C(    0.00));
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_setzero_ps();
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_setzero_ps");
    easysimd_assert_m512_close(a, r, 1);
  #else
    easysimd_assert_m512_close(easysimd_mm512_setzero_ps(), easysimd_mm512_set1_ps(EASYSIMD_FLOAT32_C(0.0)), 1);
  #endif
  return 0;
}

static int
test_easysimd_mm512_setzero_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  #if defined(EASYSIMD_ARM_SVE_NATIVE) || defined(EASYSIMD_ARM_NEON_A64V8_NATIVE)
    easysimd__m512d r;
    easysimd__m512d a =
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00));
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_setzero_pd();
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_setzero_pd");
    easysimd_assert_m512d_close(a, r, 1);
  #else
    easysimd_assert_m512d_close(easysimd_mm512_setzero_pd(), easysimd_mm512_set1_pd(EASYSIMD_FLOAT64_C(0.0)), 1);
  #endif
  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_setzero_si512)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_setzero_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_setzero_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
