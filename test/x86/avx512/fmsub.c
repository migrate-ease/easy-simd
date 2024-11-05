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
 *   2020      kitegi <kitegi@users.noreply.github.com>
 */

#define EASYSIMD_TEST_X86_AVX512_INSN fmsub

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/fmsub.h>

static int
test_easysimd_mm256_mask3_fmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd__mmask8 k;
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -954.42), EASYSIMD_FLOAT64_C(  -453.32), EASYSIMD_FLOAT64_C(  -202.19), EASYSIMD_FLOAT64_C(  -856.55) },
      { EASYSIMD_FLOAT64_C(   617.91), EASYSIMD_FLOAT64_C(   844.52), EASYSIMD_FLOAT64_C(  -195.80), EASYSIMD_FLOAT64_C(   367.41) },
      { EASYSIMD_FLOAT64_C(   809.85), EASYSIMD_FLOAT64_C(  -324.84), EASYSIMD_FLOAT64_C(   457.53), EASYSIMD_FLOAT64_C(   131.49) },
      UINT8_C( 88),
      { EASYSIMD_FLOAT64_C(   809.85), EASYSIMD_FLOAT64_C(  -324.84), EASYSIMD_FLOAT64_C(   457.53), EASYSIMD_FLOAT64_C(-314836.53) } },
    { { EASYSIMD_FLOAT64_C(   836.65), EASYSIMD_FLOAT64_C(   763.53), EASYSIMD_FLOAT64_C(   194.01), EASYSIMD_FLOAT64_C(   -41.39) },
      { EASYSIMD_FLOAT64_C(  -329.34), EASYSIMD_FLOAT64_C(   298.74), EASYSIMD_FLOAT64_C(  -519.73), EASYSIMD_FLOAT64_C(  -460.87) },
      { EASYSIMD_FLOAT64_C(   478.28), EASYSIMD_FLOAT64_C(   152.02), EASYSIMD_FLOAT64_C(   235.10), EASYSIMD_FLOAT64_C(   904.68) },
      UINT8_C(107),
      { EASYSIMD_FLOAT64_C(-276020.59), EASYSIMD_FLOAT64_C(227944.93), EASYSIMD_FLOAT64_C(   235.10), EASYSIMD_FLOAT64_C( 18170.73) } },
    { { EASYSIMD_FLOAT64_C(   -36.65), EASYSIMD_FLOAT64_C(   588.02), EASYSIMD_FLOAT64_C(   822.54), EASYSIMD_FLOAT64_C(   882.45) },
      { EASYSIMD_FLOAT64_C(   159.85), EASYSIMD_FLOAT64_C(   868.13), EASYSIMD_FLOAT64_C(  -570.87), EASYSIMD_FLOAT64_C(   957.66) },
      { EASYSIMD_FLOAT64_C(  -988.43), EASYSIMD_FLOAT64_C(  -952.96), EASYSIMD_FLOAT64_C(   802.18), EASYSIMD_FLOAT64_C(  -184.23) },
      UINT8_C(218),
      { EASYSIMD_FLOAT64_C(  -988.43), EASYSIMD_FLOAT64_C(511430.76), EASYSIMD_FLOAT64_C(   802.18), EASYSIMD_FLOAT64_C(845271.30) } },
    { { EASYSIMD_FLOAT64_C(   612.03), EASYSIMD_FLOAT64_C(   490.93), EASYSIMD_FLOAT64_C(  -128.01), EASYSIMD_FLOAT64_C(  -256.48) },
      { EASYSIMD_FLOAT64_C(   611.73), EASYSIMD_FLOAT64_C(  -291.36), EASYSIMD_FLOAT64_C(  -492.95), EASYSIMD_FLOAT64_C(  -194.26) },
      { EASYSIMD_FLOAT64_C(   667.25), EASYSIMD_FLOAT64_C(   177.71), EASYSIMD_FLOAT64_C(  -895.52), EASYSIMD_FLOAT64_C(  -852.48) },
      UINT8_C( 92),
      { EASYSIMD_FLOAT64_C(   667.25), EASYSIMD_FLOAT64_C(   177.71), EASYSIMD_FLOAT64_C( 63998.05), EASYSIMD_FLOAT64_C( 50676.28) } },
    { { EASYSIMD_FLOAT64_C(   582.76), EASYSIMD_FLOAT64_C(   299.54), EASYSIMD_FLOAT64_C(   -48.07), EASYSIMD_FLOAT64_C(   487.43) },
      { EASYSIMD_FLOAT64_C(   493.05), EASYSIMD_FLOAT64_C(   915.28), EASYSIMD_FLOAT64_C(    75.46), EASYSIMD_FLOAT64_C(   315.60) },
      { EASYSIMD_FLOAT64_C(   797.73), EASYSIMD_FLOAT64_C(  -764.70), EASYSIMD_FLOAT64_C(   183.72), EASYSIMD_FLOAT64_C(  -773.14) },
      UINT8_C( 99),
      { EASYSIMD_FLOAT64_C(286532.09), EASYSIMD_FLOAT64_C(274927.67), EASYSIMD_FLOAT64_C(   183.72), EASYSIMD_FLOAT64_C(  -773.14) } },
    { { EASYSIMD_FLOAT64_C(   195.30), EASYSIMD_FLOAT64_C(  -726.10), EASYSIMD_FLOAT64_C(   995.14), EASYSIMD_FLOAT64_C(  -988.93) },
      { EASYSIMD_FLOAT64_C(   688.35), EASYSIMD_FLOAT64_C(   607.17), EASYSIMD_FLOAT64_C(   502.00), EASYSIMD_FLOAT64_C(  -439.67) },
      { EASYSIMD_FLOAT64_C(  -649.31), EASYSIMD_FLOAT64_C(   113.73), EASYSIMD_FLOAT64_C(   268.97), EASYSIMD_FLOAT64_C(  -142.26) },
      UINT8_C(195),
      { EASYSIMD_FLOAT64_C(135084.07), EASYSIMD_FLOAT64_C(-440979.87), EASYSIMD_FLOAT64_C(   268.97), EASYSIMD_FLOAT64_C(  -142.26) } },
    { { EASYSIMD_FLOAT64_C(   -63.77), EASYSIMD_FLOAT64_C(  -964.56), EASYSIMD_FLOAT64_C(  -976.06), EASYSIMD_FLOAT64_C(    83.75) },
      { EASYSIMD_FLOAT64_C(   752.28), EASYSIMD_FLOAT64_C(   606.70), EASYSIMD_FLOAT64_C(  -616.71), EASYSIMD_FLOAT64_C(  -295.79) },
      { EASYSIMD_FLOAT64_C(    94.13), EASYSIMD_FLOAT64_C(   876.35), EASYSIMD_FLOAT64_C(  -380.51), EASYSIMD_FLOAT64_C(  -830.42) },
      UINT8_C(252),
      { EASYSIMD_FLOAT64_C(    94.13), EASYSIMD_FLOAT64_C(   876.35), EASYSIMD_FLOAT64_C(602326.47), EASYSIMD_FLOAT64_C(-23941.99) } },
    { { EASYSIMD_FLOAT64_C(  -582.78), EASYSIMD_FLOAT64_C(  -595.11), EASYSIMD_FLOAT64_C(  -624.34), EASYSIMD_FLOAT64_C(  -355.92) },
      { EASYSIMD_FLOAT64_C(  -402.15), EASYSIMD_FLOAT64_C(   570.96), EASYSIMD_FLOAT64_C(   -82.02), EASYSIMD_FLOAT64_C(  -407.00) },
      { EASYSIMD_FLOAT64_C(   582.03), EASYSIMD_FLOAT64_C(  -393.67), EASYSIMD_FLOAT64_C(  -799.83), EASYSIMD_FLOAT64_C(    84.03) },
      UINT8_C(168),
      { EASYSIMD_FLOAT64_C(   582.03), EASYSIMD_FLOAT64_C(  -393.67), EASYSIMD_FLOAT64_C(  -799.83), EASYSIMD_FLOAT64_C(144775.41) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r = easysimd_mm256_mask3_fmsub_pd(a, b, c, test_vec[i].k);
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d c = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d r = easysimd_mm256_mask3_fmsub_pd(a, b, c, k);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_fmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const easysimd__mmask8 k;
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   679.94), EASYSIMD_FLOAT64_C(  -650.92), EASYSIMD_FLOAT64_C(  -451.39), EASYSIMD_FLOAT64_C(   782.22) },
      UINT8_C(247),
      { EASYSIMD_FLOAT64_C(  -298.33), EASYSIMD_FLOAT64_C(   981.11), EASYSIMD_FLOAT64_C(  -715.76), EASYSIMD_FLOAT64_C(   567.16) },
      { EASYSIMD_FLOAT64_C(   977.82), EASYSIMD_FLOAT64_C(   -15.46), EASYSIMD_FLOAT64_C(   580.37), EASYSIMD_FLOAT64_C(   684.51) },
      { EASYSIMD_FLOAT64_C(-203824.32), EASYSIMD_FLOAT64_C(-638608.66), EASYSIMD_FLOAT64_C(322506.54), EASYSIMD_FLOAT64_C(   782.22) } },
    { { EASYSIMD_FLOAT64_C(   427.00), EASYSIMD_FLOAT64_C(  -211.11), EASYSIMD_FLOAT64_C(  -196.66), EASYSIMD_FLOAT64_C(   942.46) },
      UINT8_C(  6),
      { EASYSIMD_FLOAT64_C(   760.51), EASYSIMD_FLOAT64_C(   834.79), EASYSIMD_FLOAT64_C(   915.81), EASYSIMD_FLOAT64_C(   632.36) },
      { EASYSIMD_FLOAT64_C(  -813.85), EASYSIMD_FLOAT64_C(   615.59), EASYSIMD_FLOAT64_C(   177.28), EASYSIMD_FLOAT64_C(  -429.10) },
      { EASYSIMD_FLOAT64_C(   427.00), EASYSIMD_FLOAT64_C(-176848.11), EASYSIMD_FLOAT64_C(-180280.47), EASYSIMD_FLOAT64_C(   942.46) } },
    { { EASYSIMD_FLOAT64_C(  -557.62), EASYSIMD_FLOAT64_C(   876.64), EASYSIMD_FLOAT64_C(  -592.31), EASYSIMD_FLOAT64_C(   192.97) },
      UINT8_C(  7),
      { EASYSIMD_FLOAT64_C(  -912.37), EASYSIMD_FLOAT64_C(   542.05), EASYSIMD_FLOAT64_C(   794.79), EASYSIMD_FLOAT64_C(   869.85) },
      { EASYSIMD_FLOAT64_C(  -648.72), EASYSIMD_FLOAT64_C(  -503.53), EASYSIMD_FLOAT64_C(   850.96), EASYSIMD_FLOAT64_C(  -364.48) },
      { EASYSIMD_FLOAT64_C(509404.48), EASYSIMD_FLOAT64_C(475686.24), EASYSIMD_FLOAT64_C(-471613.02), EASYSIMD_FLOAT64_C(   192.97) } },
    { { EASYSIMD_FLOAT64_C(  -936.37), EASYSIMD_FLOAT64_C(   828.79), EASYSIMD_FLOAT64_C(   620.07), EASYSIMD_FLOAT64_C(   644.01) },
      UINT8_C( 86),
      { EASYSIMD_FLOAT64_C(    47.06), EASYSIMD_FLOAT64_C(  -567.11), EASYSIMD_FLOAT64_C(  -683.37), EASYSIMD_FLOAT64_C(   -10.48) },
      { EASYSIMD_FLOAT64_C(  -691.38), EASYSIMD_FLOAT64_C(  -922.86), EASYSIMD_FLOAT64_C(  -175.69), EASYSIMD_FLOAT64_C(  -775.57) },
      { EASYSIMD_FLOAT64_C(  -936.37), EASYSIMD_FLOAT64_C(-469092.24), EASYSIMD_FLOAT64_C(-423561.55), EASYSIMD_FLOAT64_C(   644.01) } },
    { { EASYSIMD_FLOAT64_C(   709.51), EASYSIMD_FLOAT64_C(    10.46), EASYSIMD_FLOAT64_C(   840.02), EASYSIMD_FLOAT64_C(  -113.21) },
      UINT8_C( 27),
      { EASYSIMD_FLOAT64_C(  -717.60), EASYSIMD_FLOAT64_C(  -236.58), EASYSIMD_FLOAT64_C(   989.06), EASYSIMD_FLOAT64_C(   475.37) },
      { EASYSIMD_FLOAT64_C(  -990.39), EASYSIMD_FLOAT64_C(  -923.31), EASYSIMD_FLOAT64_C(    17.42), EASYSIMD_FLOAT64_C(   804.41) },
      { EASYSIMD_FLOAT64_C(-508153.99), EASYSIMD_FLOAT64_C( -1551.32), EASYSIMD_FLOAT64_C(   840.02), EASYSIMD_FLOAT64_C(-54621.05) } },
    { { EASYSIMD_FLOAT64_C(   946.54), EASYSIMD_FLOAT64_C(   368.70), EASYSIMD_FLOAT64_C(  -699.12), EASYSIMD_FLOAT64_C(   797.51) },
      UINT8_C(148),
      { EASYSIMD_FLOAT64_C(  -635.49), EASYSIMD_FLOAT64_C(   626.29), EASYSIMD_FLOAT64_C(   624.29), EASYSIMD_FLOAT64_C(  -991.49) },
      { EASYSIMD_FLOAT64_C(   139.59), EASYSIMD_FLOAT64_C(  -328.65), EASYSIMD_FLOAT64_C(  -558.59), EASYSIMD_FLOAT64_C(   456.22) },
      { EASYSIMD_FLOAT64_C(   946.54), EASYSIMD_FLOAT64_C(   368.70), EASYSIMD_FLOAT64_C(-435895.03), EASYSIMD_FLOAT64_C(   797.51) } },
    { { EASYSIMD_FLOAT64_C(   660.87), EASYSIMD_FLOAT64_C(  -249.98), EASYSIMD_FLOAT64_C(   533.36), EASYSIMD_FLOAT64_C(  -514.82) },
      UINT8_C(206),
      { EASYSIMD_FLOAT64_C(   242.87), EASYSIMD_FLOAT64_C(   495.64), EASYSIMD_FLOAT64_C(  -185.53), EASYSIMD_FLOAT64_C(  -870.34) },
      { EASYSIMD_FLOAT64_C(    77.00), EASYSIMD_FLOAT64_C(    96.87), EASYSIMD_FLOAT64_C(  -106.92), EASYSIMD_FLOAT64_C(    66.06) },
      { EASYSIMD_FLOAT64_C(   660.87), EASYSIMD_FLOAT64_C(-123996.96), EASYSIMD_FLOAT64_C(-98847.36), EASYSIMD_FLOAT64_C(448002.38) } },
    { { EASYSIMD_FLOAT64_C(  -427.76), EASYSIMD_FLOAT64_C(   -97.30), EASYSIMD_FLOAT64_C(   142.74), EASYSIMD_FLOAT64_C(   589.66) },
      UINT8_C(211),
      { EASYSIMD_FLOAT64_C(    89.28), EASYSIMD_FLOAT64_C(   -41.64), EASYSIMD_FLOAT64_C(     7.98), EASYSIMD_FLOAT64_C(  -113.21) },
      { EASYSIMD_FLOAT64_C(   -37.42), EASYSIMD_FLOAT64_C(   372.49), EASYSIMD_FLOAT64_C(  -486.92), EASYSIMD_FLOAT64_C(  -413.14) },
      { EASYSIMD_FLOAT64_C(-38152.99), EASYSIMD_FLOAT64_C(  3679.08), EASYSIMD_FLOAT64_C(   142.74), EASYSIMD_FLOAT64_C(   589.66) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_fmsub_pd(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_fmsub_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d c = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_fmsub_pd(a, k, b, c);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_fmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C( 25),
      { EASYSIMD_FLOAT64_C(   232.69), EASYSIMD_FLOAT64_C(    64.70), EASYSIMD_FLOAT64_C(  -891.95), EASYSIMD_FLOAT64_C(  -994.97) },
      { EASYSIMD_FLOAT64_C(  -220.84), EASYSIMD_FLOAT64_C(  -581.73), EASYSIMD_FLOAT64_C(  -763.36), EASYSIMD_FLOAT64_C(   140.20) },
      { EASYSIMD_FLOAT64_C(  -188.54), EASYSIMD_FLOAT64_C(   795.19), EASYSIMD_FLOAT64_C(  -191.89), EASYSIMD_FLOAT64_C(  -597.31) },
      { EASYSIMD_FLOAT64_C(-51198.72), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-138897.48) } },
    { UINT8_C(126),
      { EASYSIMD_FLOAT64_C(  -417.76), EASYSIMD_FLOAT64_C(  -376.20), EASYSIMD_FLOAT64_C(  -480.75), EASYSIMD_FLOAT64_C(   587.20) },
      { EASYSIMD_FLOAT64_C(   654.93), EASYSIMD_FLOAT64_C(   320.39), EASYSIMD_FLOAT64_C(  -333.82), EASYSIMD_FLOAT64_C(  -841.08) },
      { EASYSIMD_FLOAT64_C(  -684.25), EASYSIMD_FLOAT64_C(   264.60), EASYSIMD_FLOAT64_C(   239.65), EASYSIMD_FLOAT64_C(  -504.73) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-120795.32), EASYSIMD_FLOAT64_C(160244.32), EASYSIMD_FLOAT64_C(-493377.45) } },
    { UINT8_C(109),
      { EASYSIMD_FLOAT64_C(   758.39), EASYSIMD_FLOAT64_C(  -246.89), EASYSIMD_FLOAT64_C(   128.18), EASYSIMD_FLOAT64_C(   432.15) },
      { EASYSIMD_FLOAT64_C(  -342.44), EASYSIMD_FLOAT64_C(  -639.14), EASYSIMD_FLOAT64_C(  -503.15), EASYSIMD_FLOAT64_C(  -234.38) },
      { EASYSIMD_FLOAT64_C(  -634.10), EASYSIMD_FLOAT64_C(   276.01), EASYSIMD_FLOAT64_C(   183.88), EASYSIMD_FLOAT64_C(  -397.47) },
      { EASYSIMD_FLOAT64_C(-259068.97), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-64677.65), EASYSIMD_FLOAT64_C(-100889.85) } },
    { UINT8_C( 93),
      { EASYSIMD_FLOAT64_C(   995.34), EASYSIMD_FLOAT64_C(  -602.28), EASYSIMD_FLOAT64_C(   224.32), EASYSIMD_FLOAT64_C(  -601.97) },
      { EASYSIMD_FLOAT64_C(  -911.34), EASYSIMD_FLOAT64_C(   806.56), EASYSIMD_FLOAT64_C(    21.83), EASYSIMD_FLOAT64_C(  -392.09) },
      { EASYSIMD_FLOAT64_C(   393.76), EASYSIMD_FLOAT64_C(  -323.24), EASYSIMD_FLOAT64_C(   928.30), EASYSIMD_FLOAT64_C(  -940.06) },
      { EASYSIMD_FLOAT64_C(-907486.92), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  3968.61), EASYSIMD_FLOAT64_C(236966.48) } },
    { UINT8_C( 91),
      { EASYSIMD_FLOAT64_C(  -755.95), EASYSIMD_FLOAT64_C(   324.54), EASYSIMD_FLOAT64_C(  -924.67), EASYSIMD_FLOAT64_C(  -260.68) },
      { EASYSIMD_FLOAT64_C(  -253.16), EASYSIMD_FLOAT64_C(   833.72), EASYSIMD_FLOAT64_C(   492.43), EASYSIMD_FLOAT64_C(   875.02) },
      { EASYSIMD_FLOAT64_C(   265.87), EASYSIMD_FLOAT64_C(  -850.00), EASYSIMD_FLOAT64_C(  -764.12), EASYSIMD_FLOAT64_C(   762.72) },
      { EASYSIMD_FLOAT64_C(191110.43), EASYSIMD_FLOAT64_C(271425.49), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-228862.93) } },
    { UINT8_C( 13),
      { EASYSIMD_FLOAT64_C(  -398.22), EASYSIMD_FLOAT64_C(    38.73), EASYSIMD_FLOAT64_C(  -900.51), EASYSIMD_FLOAT64_C(   204.31) },
      { EASYSIMD_FLOAT64_C(   454.94), EASYSIMD_FLOAT64_C(  -905.17), EASYSIMD_FLOAT64_C(   602.03), EASYSIMD_FLOAT64_C(  -320.74) },
      { EASYSIMD_FLOAT64_C(  -507.13), EASYSIMD_FLOAT64_C(   690.69), EASYSIMD_FLOAT64_C(  -514.18), EASYSIMD_FLOAT64_C(   514.70) },
      { EASYSIMD_FLOAT64_C(-180659.08), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-541619.86), EASYSIMD_FLOAT64_C(-66045.09) } },
    { UINT8_C(114),
      { EASYSIMD_FLOAT64_C(   879.58), EASYSIMD_FLOAT64_C(  -808.55), EASYSIMD_FLOAT64_C(  -773.10), EASYSIMD_FLOAT64_C(   939.51) },
      { EASYSIMD_FLOAT64_C(    27.13), EASYSIMD_FLOAT64_C(  -529.04), EASYSIMD_FLOAT64_C(   264.05), EASYSIMD_FLOAT64_C(   102.46) },
      { EASYSIMD_FLOAT64_C(   210.28), EASYSIMD_FLOAT64_C(  -989.11), EASYSIMD_FLOAT64_C(   -63.82), EASYSIMD_FLOAT64_C(  -297.29) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(428744.40), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(228),
      { EASYSIMD_FLOAT64_C(  -797.95), EASYSIMD_FLOAT64_C(  -147.29), EASYSIMD_FLOAT64_C(  -878.21), EASYSIMD_FLOAT64_C(   964.77) },
      { EASYSIMD_FLOAT64_C(   768.32), EASYSIMD_FLOAT64_C(  -276.43), EASYSIMD_FLOAT64_C(     3.50), EASYSIMD_FLOAT64_C(   867.81) },
      { EASYSIMD_FLOAT64_C(   927.88), EASYSIMD_FLOAT64_C(  -541.56), EASYSIMD_FLOAT64_C(   962.65), EASYSIMD_FLOAT64_C(   529.91) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C( -4036.39), EASYSIMD_FLOAT64_C(     0.00) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_fmsub_pd(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_fmsub_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d c = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_fmsub_pd(k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask3_fmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd__mmask8 k;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   487.95), EASYSIMD_FLOAT64_C(   974.14) },
      { EASYSIMD_FLOAT64_C(   218.01), EASYSIMD_FLOAT64_C(  -532.31) },
      { EASYSIMD_FLOAT64_C(   262.30), EASYSIMD_FLOAT64_C(  -672.99) },
      UINT8_C(137),
      { EASYSIMD_FLOAT64_C(106115.68), EASYSIMD_FLOAT64_C(  -672.99) } },
    { { EASYSIMD_FLOAT64_C(   -47.66), EASYSIMD_FLOAT64_C(   274.55) },
      { EASYSIMD_FLOAT64_C(  -634.93), EASYSIMD_FLOAT64_C(  -690.86) },
      { EASYSIMD_FLOAT64_C(  -817.46), EASYSIMD_FLOAT64_C(   844.40) },
      UINT8_C( 30),
      { EASYSIMD_FLOAT64_C(  -817.46), EASYSIMD_FLOAT64_C(-190520.01) } },
    { { EASYSIMD_FLOAT64_C(   735.27), EASYSIMD_FLOAT64_C(   -23.68) },
      { EASYSIMD_FLOAT64_C(  -790.02), EASYSIMD_FLOAT64_C(  -627.96) },
      { EASYSIMD_FLOAT64_C(  -178.61), EASYSIMD_FLOAT64_C(   799.75) },
      UINT8_C( 62),
      { EASYSIMD_FLOAT64_C(  -178.61), EASYSIMD_FLOAT64_C( 14070.34) } },
    { { EASYSIMD_FLOAT64_C(  -281.07), EASYSIMD_FLOAT64_C(  -249.44) },
      { EASYSIMD_FLOAT64_C(  -363.18), EASYSIMD_FLOAT64_C(   -49.00) },
      { EASYSIMD_FLOAT64_C(  -545.91), EASYSIMD_FLOAT64_C(  -837.74) },
      UINT8_C(189),
      { EASYSIMD_FLOAT64_C(102624.91), EASYSIMD_FLOAT64_C(  -837.74) } },
    { { EASYSIMD_FLOAT64_C(  -192.42), EASYSIMD_FLOAT64_C(  -886.89) },
      { EASYSIMD_FLOAT64_C(   112.78), EASYSIMD_FLOAT64_C(  -704.47) },
      { EASYSIMD_FLOAT64_C(  -912.75), EASYSIMD_FLOAT64_C(  -669.21) },
      UINT8_C( 93),
      { EASYSIMD_FLOAT64_C(-20788.38), EASYSIMD_FLOAT64_C(  -669.21) } },
    { { EASYSIMD_FLOAT64_C(   349.56), EASYSIMD_FLOAT64_C(  -342.21) },
      { EASYSIMD_FLOAT64_C(  -284.85), EASYSIMD_FLOAT64_C(  -698.11) },
      { EASYSIMD_FLOAT64_C(   932.35), EASYSIMD_FLOAT64_C(    80.22) },
      UINT8_C(115),
      { EASYSIMD_FLOAT64_C(-100504.52), EASYSIMD_FLOAT64_C(238820.00) } },
    { { EASYSIMD_FLOAT64_C(  -885.11), EASYSIMD_FLOAT64_C(   -75.37) },
      { EASYSIMD_FLOAT64_C(  -400.13), EASYSIMD_FLOAT64_C(   850.16) },
      { EASYSIMD_FLOAT64_C(   900.94), EASYSIMD_FLOAT64_C(  -190.15) },
      UINT8_C(190),
      { EASYSIMD_FLOAT64_C(   900.94), EASYSIMD_FLOAT64_C(-63886.41) } },
    { { EASYSIMD_FLOAT64_C(  -277.66), EASYSIMD_FLOAT64_C(  -390.39) },
      { EASYSIMD_FLOAT64_C(  -507.69), EASYSIMD_FLOAT64_C(   441.27) },
      { EASYSIMD_FLOAT64_C(   360.17), EASYSIMD_FLOAT64_C(   129.12) },
      UINT8_C(  2),
      { EASYSIMD_FLOAT64_C(   360.17), EASYSIMD_FLOAT64_C(-172396.52) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask3_fmsub_pd(a, b, c, test_vec[i].k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask3_fmsub_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d c = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d r = easysimd_mm_mask3_fmsub_pd(a, b, c, k);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask3_fmsubadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd__mmask8 k;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   399.93), EASYSIMD_FLOAT64_C(  -256.71) },
      { EASYSIMD_FLOAT64_C(   485.93), EASYSIMD_FLOAT64_C(   212.45) },
      { EASYSIMD_FLOAT64_C(   385.07), EASYSIMD_FLOAT64_C(   434.09) },
      UINT8_C( 82),
      { EASYSIMD_FLOAT64_C(   385.07), EASYSIMD_FLOAT64_C(-54972.13) } },
    { { EASYSIMD_FLOAT64_C(  -172.79), EASYSIMD_FLOAT64_C(   862.50) },
      { EASYSIMD_FLOAT64_C(   252.29), EASYSIMD_FLOAT64_C(   473.91) },
      { EASYSIMD_FLOAT64_C(   327.60), EASYSIMD_FLOAT64_C(   182.27) },
      UINT8_C(166),
      { EASYSIMD_FLOAT64_C(   327.60), EASYSIMD_FLOAT64_C(408565.10) } },
    { { EASYSIMD_FLOAT64_C(   -83.86), EASYSIMD_FLOAT64_C(   463.04) },
      { EASYSIMD_FLOAT64_C(   102.28), EASYSIMD_FLOAT64_C(  -951.79) },
      { EASYSIMD_FLOAT64_C(    -2.07), EASYSIMD_FLOAT64_C(   114.82) },
      UINT8_C(219),
      { EASYSIMD_FLOAT64_C( -8579.27), EASYSIMD_FLOAT64_C(-440831.66) } },
    { { EASYSIMD_FLOAT64_C(   -11.32), EASYSIMD_FLOAT64_C(   973.48) },
      { EASYSIMD_FLOAT64_C(   736.20), EASYSIMD_FLOAT64_C(   575.22) },
      { EASYSIMD_FLOAT64_C(  -925.87), EASYSIMD_FLOAT64_C(  -448.72) },
      UINT8_C( 37),
      { EASYSIMD_FLOAT64_C( -9259.65), EASYSIMD_FLOAT64_C(  -448.72) } },
    { { EASYSIMD_FLOAT64_C(  -506.08), EASYSIMD_FLOAT64_C(  -671.00) },
      { EASYSIMD_FLOAT64_C(   316.82), EASYSIMD_FLOAT64_C(   893.84) },
      { EASYSIMD_FLOAT64_C(    72.29), EASYSIMD_FLOAT64_C(  -197.25) },
      UINT8_C( 38),
      { EASYSIMD_FLOAT64_C(    72.29), EASYSIMD_FLOAT64_C(-599569.39) } },
    { { EASYSIMD_FLOAT64_C(  -542.63), EASYSIMD_FLOAT64_C(  -763.16) },
      { EASYSIMD_FLOAT64_C(   758.19), EASYSIMD_FLOAT64_C(   284.58) },
      { EASYSIMD_FLOAT64_C(  -900.66), EASYSIMD_FLOAT64_C(    10.48) },
      UINT8_C(233),
      { EASYSIMD_FLOAT64_C(-412317.30), EASYSIMD_FLOAT64_C(    10.48) } },
    { { EASYSIMD_FLOAT64_C(   426.94), EASYSIMD_FLOAT64_C(  -807.25) },
      { EASYSIMD_FLOAT64_C(  -612.34), EASYSIMD_FLOAT64_C(  -656.92) },
      { EASYSIMD_FLOAT64_C(   655.79), EASYSIMD_FLOAT64_C(   489.94) },
      UINT8_C(104),
      { EASYSIMD_FLOAT64_C(   655.79), EASYSIMD_FLOAT64_C(   489.94) } },
    { { EASYSIMD_FLOAT64_C(  -346.28), EASYSIMD_FLOAT64_C(  -395.24) },
      { EASYSIMD_FLOAT64_C(   689.57), EASYSIMD_FLOAT64_C(   642.40) },
      { EASYSIMD_FLOAT64_C(  -421.76), EASYSIMD_FLOAT64_C(   425.78) },
      UINT8_C( 38),
      { EASYSIMD_FLOAT64_C(  -421.76), EASYSIMD_FLOAT64_C(-254327.96) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask3_fmsubadd_pd(a, b, c, test_vec[i].k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask3_fmsubadd_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d c = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d r = easysimd_mm_mask3_fmsubadd_pd(a, b, c, k);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd__mmask8 k;
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   819.06), EASYSIMD_FLOAT64_C(   -89.62) },
      UINT8_C( 20),
      { EASYSIMD_FLOAT64_C(   863.60), EASYSIMD_FLOAT64_C(   956.60) },
      { EASYSIMD_FLOAT64_C(   184.19), EASYSIMD_FLOAT64_C(  -173.26) },
      { EASYSIMD_FLOAT64_C(   819.06), EASYSIMD_FLOAT64_C(   -89.62) } },
    { { EASYSIMD_FLOAT64_C(  -907.16), EASYSIMD_FLOAT64_C(    78.38) },
      UINT8_C(253),
      { EASYSIMD_FLOAT64_C(   510.70), EASYSIMD_FLOAT64_C(  -694.94) },
      { EASYSIMD_FLOAT64_C(   141.48), EASYSIMD_FLOAT64_C(  -897.93) },
      { EASYSIMD_FLOAT64_C(-463428.09), EASYSIMD_FLOAT64_C(    78.38) } },
    { { EASYSIMD_FLOAT64_C(  -315.81), EASYSIMD_FLOAT64_C(   993.53) },
      UINT8_C( 80),
      { EASYSIMD_FLOAT64_C(   423.85), EASYSIMD_FLOAT64_C(   561.78) },
      { EASYSIMD_FLOAT64_C(   178.61), EASYSIMD_FLOAT64_C(   606.93) },
      { EASYSIMD_FLOAT64_C(  -315.81), EASYSIMD_FLOAT64_C(   993.53) } },
    { { EASYSIMD_FLOAT64_C(  -282.73), EASYSIMD_FLOAT64_C(   374.02) },
      UINT8_C(234),
      { EASYSIMD_FLOAT64_C(   382.75), EASYSIMD_FLOAT64_C(    69.97) },
      { EASYSIMD_FLOAT64_C(   987.46), EASYSIMD_FLOAT64_C(  -743.87) },
      { EASYSIMD_FLOAT64_C(  -282.73), EASYSIMD_FLOAT64_C( 26914.05) } },
    { { EASYSIMD_FLOAT64_C(  -650.14), EASYSIMD_FLOAT64_C(   792.18) },
      UINT8_C(215),
      { EASYSIMD_FLOAT64_C(  -831.08), EASYSIMD_FLOAT64_C(  -297.44) },
      { EASYSIMD_FLOAT64_C(   908.88), EASYSIMD_FLOAT64_C(  -967.48) },
      { EASYSIMD_FLOAT64_C(539409.47), EASYSIMD_FLOAT64_C(-234658.54) } },
    { { EASYSIMD_FLOAT64_C(  -340.84), EASYSIMD_FLOAT64_C(    93.07) },
      UINT8_C(198),
      { EASYSIMD_FLOAT64_C(  -248.00), EASYSIMD_FLOAT64_C(  -828.55) },
      { EASYSIMD_FLOAT64_C(  -776.88), EASYSIMD_FLOAT64_C(  -737.30) },
      { EASYSIMD_FLOAT64_C(  -340.84), EASYSIMD_FLOAT64_C(-76375.85) } },
    { { EASYSIMD_FLOAT64_C(  -523.49), EASYSIMD_FLOAT64_C(   364.60) },
      UINT8_C(236),
      { EASYSIMD_FLOAT64_C(   160.70), EASYSIMD_FLOAT64_C(   358.13) },
      { EASYSIMD_FLOAT64_C(  -902.39), EASYSIMD_FLOAT64_C(  -415.45) },
      { EASYSIMD_FLOAT64_C(  -523.49), EASYSIMD_FLOAT64_C(   364.60) } },
    { { EASYSIMD_FLOAT64_C(   -80.09), EASYSIMD_FLOAT64_C(   276.22) },
      UINT8_C( 89),
      { EASYSIMD_FLOAT64_C(   637.18), EASYSIMD_FLOAT64_C(  -349.76) },
      { EASYSIMD_FLOAT64_C(    41.52), EASYSIMD_FLOAT64_C(    19.94) },
      { EASYSIMD_FLOAT64_C(-51073.27), EASYSIMD_FLOAT64_C(   276.22) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_fmsub_pd(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_fmsub_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d c = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_fmsub_pd(a, k, b, c);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C( 95),
      { EASYSIMD_FLOAT64_C(   920.64), EASYSIMD_FLOAT64_C(   725.81) },
      { EASYSIMD_FLOAT64_C(   360.26), EASYSIMD_FLOAT64_C(   687.84) },
      { EASYSIMD_FLOAT64_C(   262.25), EASYSIMD_FLOAT64_C(   878.73) },
      { EASYSIMD_FLOAT64_C(331407.52), EASYSIMD_FLOAT64_C(498362.42) } },
    { UINT8_C(142),
      { EASYSIMD_FLOAT64_C(  -320.65), EASYSIMD_FLOAT64_C(  -354.71) },
      { EASYSIMD_FLOAT64_C(   942.15), EASYSIMD_FLOAT64_C(   899.30) },
      { EASYSIMD_FLOAT64_C(   212.41), EASYSIMD_FLOAT64_C(   995.00) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-319985.70) } },
    { UINT8_C(220),
      { EASYSIMD_FLOAT64_C(  -851.11), EASYSIMD_FLOAT64_C(  -551.67) },
      { EASYSIMD_FLOAT64_C(   695.44), EASYSIMD_FLOAT64_C(   899.92) },
      { EASYSIMD_FLOAT64_C(   350.71), EASYSIMD_FLOAT64_C(   416.83) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(163),
      { EASYSIMD_FLOAT64_C(   836.16), EASYSIMD_FLOAT64_C(  -798.79) },
      { EASYSIMD_FLOAT64_C(   679.77), EASYSIMD_FLOAT64_C(    47.52) },
      { EASYSIMD_FLOAT64_C(  -961.36), EASYSIMD_FLOAT64_C(  -234.63) },
      { EASYSIMD_FLOAT64_C(569357.84), EASYSIMD_FLOAT64_C(-37723.87) } },
    { UINT8_C(112),
      { EASYSIMD_FLOAT64_C(  -847.67), EASYSIMD_FLOAT64_C(  -617.07) },
      { EASYSIMD_FLOAT64_C(  -319.47), EASYSIMD_FLOAT64_C(  -927.03) },
      { EASYSIMD_FLOAT64_C(  -891.26), EASYSIMD_FLOAT64_C(  -959.21) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 59),
      { EASYSIMD_FLOAT64_C(   370.99), EASYSIMD_FLOAT64_C(   919.51) },
      { EASYSIMD_FLOAT64_C(   339.75), EASYSIMD_FLOAT64_C(  -949.67) },
      { EASYSIMD_FLOAT64_C(  -435.19), EASYSIMD_FLOAT64_C(   281.89) },
      { EASYSIMD_FLOAT64_C(126479.04), EASYSIMD_FLOAT64_C(-873512.95) } },
    { UINT8_C( 15),
      { EASYSIMD_FLOAT64_C(   777.22), EASYSIMD_FLOAT64_C(   276.89) },
      { EASYSIMD_FLOAT64_C(  -243.21), EASYSIMD_FLOAT64_C(   926.10) },
      { EASYSIMD_FLOAT64_C(   725.23), EASYSIMD_FLOAT64_C(  -547.77) },
      { EASYSIMD_FLOAT64_C(-189752.91), EASYSIMD_FLOAT64_C(256975.60) } },
    { UINT8_C(202),
      { EASYSIMD_FLOAT64_C(    75.93), EASYSIMD_FLOAT64_C(   869.06) },
      { EASYSIMD_FLOAT64_C(  -988.11), EASYSIMD_FLOAT64_C(   -87.90) },
      { EASYSIMD_FLOAT64_C(  -929.73), EASYSIMD_FLOAT64_C(   691.66) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-77082.03) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_fmsub_pd(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_fmsub_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d c = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_fmsub_pd(k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fmsubadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd__mmask8 k;
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -360.25), EASYSIMD_FLOAT64_C(  -865.35) },
      UINT8_C(148),
      { EASYSIMD_FLOAT64_C(  -488.89), EASYSIMD_FLOAT64_C(   397.57) },
      { EASYSIMD_FLOAT64_C(  -152.07), EASYSIMD_FLOAT64_C(  -929.37) },
      { EASYSIMD_FLOAT64_C(  -360.25), EASYSIMD_FLOAT64_C(  -865.35) } },
    { { EASYSIMD_FLOAT64_C(   121.78), EASYSIMD_FLOAT64_C(  -252.18) },
      UINT8_C(245),
      { EASYSIMD_FLOAT64_C(  -917.30), EASYSIMD_FLOAT64_C(  -825.56) },
      { EASYSIMD_FLOAT64_C(  -969.46), EASYSIMD_FLOAT64_C(   279.09) },
      { EASYSIMD_FLOAT64_C(-112678.25), EASYSIMD_FLOAT64_C(  -252.18) } },
    { { EASYSIMD_FLOAT64_C(   817.86), EASYSIMD_FLOAT64_C(    34.19) },
      UINT8_C(176),
      { EASYSIMD_FLOAT64_C(  -640.26), EASYSIMD_FLOAT64_C(   987.56) },
      { EASYSIMD_FLOAT64_C(    -0.97), EASYSIMD_FLOAT64_C(   305.68) },
      { EASYSIMD_FLOAT64_C(   817.86), EASYSIMD_FLOAT64_C(    34.19) } },
    { { EASYSIMD_FLOAT64_C(   280.84), EASYSIMD_FLOAT64_C(  -641.01) },
      UINT8_C( 60),
      { EASYSIMD_FLOAT64_C(   251.92), EASYSIMD_FLOAT64_C(  -165.46) },
      { EASYSIMD_FLOAT64_C(   727.97), EASYSIMD_FLOAT64_C(   372.78) },
      { EASYSIMD_FLOAT64_C(   280.84), EASYSIMD_FLOAT64_C(  -641.01) } },
    { { EASYSIMD_FLOAT64_C(    84.93), EASYSIMD_FLOAT64_C(   807.95) },
      UINT8_C(234),
      { EASYSIMD_FLOAT64_C(   724.68), EASYSIMD_FLOAT64_C(   942.61) },
      { EASYSIMD_FLOAT64_C(  -248.36), EASYSIMD_FLOAT64_C(  -764.21) },
      { EASYSIMD_FLOAT64_C(    84.93), EASYSIMD_FLOAT64_C(762345.96) } },
    { { EASYSIMD_FLOAT64_C(   340.18), EASYSIMD_FLOAT64_C(   599.57) },
      UINT8_C( 83),
      { EASYSIMD_FLOAT64_C(  -538.04), EASYSIMD_FLOAT64_C(  -652.61) },
      { EASYSIMD_FLOAT64_C(  -357.29), EASYSIMD_FLOAT64_C(  -455.34) },
      { EASYSIMD_FLOAT64_C(-183387.74), EASYSIMD_FLOAT64_C(-390830.04) } },
    { { EASYSIMD_FLOAT64_C(  -478.17), EASYSIMD_FLOAT64_C(  -326.75) },
      UINT8_C(180),
      { EASYSIMD_FLOAT64_C(  -660.31), EASYSIMD_FLOAT64_C(   707.44) },
      { EASYSIMD_FLOAT64_C(  -281.55), EASYSIMD_FLOAT64_C(  -300.57) },
      { EASYSIMD_FLOAT64_C(  -478.17), EASYSIMD_FLOAT64_C(  -326.75) } },
    { { EASYSIMD_FLOAT64_C(   695.01), EASYSIMD_FLOAT64_C(   717.49) },
      UINT8_C( 65),
      { EASYSIMD_FLOAT64_C(   -24.15), EASYSIMD_FLOAT64_C(  -923.52) },
      { EASYSIMD_FLOAT64_C(   944.73), EASYSIMD_FLOAT64_C(  -772.23) },
      { EASYSIMD_FLOAT64_C(-15839.76), EASYSIMD_FLOAT64_C(   717.49) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_fmsubadd_pd(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_fmsubadd_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d c = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_fmsubadd_pd(a, k, b, c);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fmsubadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd__mmask8 k;
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -88.98), EASYSIMD_FLOAT64_C(   672.70) },
      UINT8_C(152),
      { EASYSIMD_FLOAT64_C(   995.95), EASYSIMD_FLOAT64_C(   480.65) },
      { EASYSIMD_FLOAT64_C(   730.01), EASYSIMD_FLOAT64_C(   720.63) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   423.26), EASYSIMD_FLOAT64_C(  -518.35) },
      UINT8_C(240),
      { EASYSIMD_FLOAT64_C(  -236.56), EASYSIMD_FLOAT64_C(  -918.77) },
      { EASYSIMD_FLOAT64_C(  -737.15), EASYSIMD_FLOAT64_C(   225.40) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -571.38), EASYSIMD_FLOAT64_C(   -94.44) },
      UINT8_C(229),
      { EASYSIMD_FLOAT64_C(   -49.56), EASYSIMD_FLOAT64_C(   578.81) },
      { EASYSIMD_FLOAT64_C(   593.80), EASYSIMD_FLOAT64_C(   290.13) },
      { EASYSIMD_FLOAT64_C( 28911.39), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   286.26), EASYSIMD_FLOAT64_C(  -687.74) },
      UINT8_C( 47),
      { EASYSIMD_FLOAT64_C(   -18.74), EASYSIMD_FLOAT64_C(  -970.26) },
      { EASYSIMD_FLOAT64_C(   994.66), EASYSIMD_FLOAT64_C(   957.11) },
      { EASYSIMD_FLOAT64_C( -4369.85), EASYSIMD_FLOAT64_C(666329.50) } },
    { { EASYSIMD_FLOAT64_C(  -893.78), EASYSIMD_FLOAT64_C(   939.38) },
      UINT8_C(130),
      { EASYSIMD_FLOAT64_C(    17.24), EASYSIMD_FLOAT64_C(   612.08) },
      { EASYSIMD_FLOAT64_C(   785.44), EASYSIMD_FLOAT64_C(    13.18) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(574962.53) } },
    { { EASYSIMD_FLOAT64_C(    92.74), EASYSIMD_FLOAT64_C(   515.45) },
      UINT8_C(221),
      { EASYSIMD_FLOAT64_C(  -484.00), EASYSIMD_FLOAT64_C(   997.11) },
      { EASYSIMD_FLOAT64_C(  -309.77), EASYSIMD_FLOAT64_C(   279.44) },
      { EASYSIMD_FLOAT64_C(-45195.93), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -921.67), EASYSIMD_FLOAT64_C(   -46.92) },
      UINT8_C(156),
      { EASYSIMD_FLOAT64_C(  -493.05), EASYSIMD_FLOAT64_C(   858.64) },
      { EASYSIMD_FLOAT64_C(  -725.11), EASYSIMD_FLOAT64_C(   457.40) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   437.45), EASYSIMD_FLOAT64_C(   868.70) },
      UINT8_C( 36),
      { EASYSIMD_FLOAT64_C(  -276.29), EASYSIMD_FLOAT64_C(  -819.05) },
      { EASYSIMD_FLOAT64_C(  -262.92), EASYSIMD_FLOAT64_C(   704.97) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_fmsubadd_pd(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_fmsubadd_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d c = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_fmsubadd_pd(k, a, b, c);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask3_fmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd__mmask8 k;
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   652.59), EASYSIMD_FLOAT32_C(   -65.81), EASYSIMD_FLOAT32_C(  -587.88), EASYSIMD_FLOAT32_C(   591.56),
        EASYSIMD_FLOAT32_C(   981.92), EASYSIMD_FLOAT32_C(   460.71), EASYSIMD_FLOAT32_C(   160.98), EASYSIMD_FLOAT32_C(  -198.70) },
      { EASYSIMD_FLOAT32_C(   831.07), EASYSIMD_FLOAT32_C(     2.11), EASYSIMD_FLOAT32_C(   287.80), EASYSIMD_FLOAT32_C(  -865.37),
        EASYSIMD_FLOAT32_C(   984.61), EASYSIMD_FLOAT32_C(  -509.15), EASYSIMD_FLOAT32_C(   535.44), EASYSIMD_FLOAT32_C(  -822.80) },
      { EASYSIMD_FLOAT32_C(   697.16), EASYSIMD_FLOAT32_C(  -953.29), EASYSIMD_FLOAT32_C(  -349.92), EASYSIMD_FLOAT32_C(   768.86),
        EASYSIMD_FLOAT32_C(  -907.91), EASYSIMD_FLOAT32_C(   546.96), EASYSIMD_FLOAT32_C(   176.82), EASYSIMD_FLOAT32_C(  -900.99) },
      UINT8_C( 95),
      { EASYSIMD_FLOAT32_C(541650.81), EASYSIMD_FLOAT32_C(   814.43), EASYSIMD_FLOAT32_C(-168841.94), EASYSIMD_FLOAT32_C(-512687.12),
        EASYSIMD_FLOAT32_C(967716.12), EASYSIMD_FLOAT32_C(   546.96), EASYSIMD_FLOAT32_C( 86018.31), EASYSIMD_FLOAT32_C(  -900.99) } },
    { { EASYSIMD_FLOAT32_C(  -246.16), EASYSIMD_FLOAT32_C(   965.54), EASYSIMD_FLOAT32_C(   831.57), EASYSIMD_FLOAT32_C(   414.39),
        EASYSIMD_FLOAT32_C(   787.88), EASYSIMD_FLOAT32_C(   810.49), EASYSIMD_FLOAT32_C(    66.98), EASYSIMD_FLOAT32_C(  -277.94) },
      { EASYSIMD_FLOAT32_C(  -777.39), EASYSIMD_FLOAT32_C(  -341.46), EASYSIMD_FLOAT32_C(  -296.02), EASYSIMD_FLOAT32_C(   683.31),
        EASYSIMD_FLOAT32_C(   819.52), EASYSIMD_FLOAT32_C(   505.28), EASYSIMD_FLOAT32_C(   514.38), EASYSIMD_FLOAT32_C(  -178.37) },
      { EASYSIMD_FLOAT32_C(  -206.92), EASYSIMD_FLOAT32_C(   649.02), EASYSIMD_FLOAT32_C(  -193.76), EASYSIMD_FLOAT32_C(   283.93),
        EASYSIMD_FLOAT32_C(   184.46), EASYSIMD_FLOAT32_C(   -16.56), EASYSIMD_FLOAT32_C(   -18.91), EASYSIMD_FLOAT32_C(   231.16) },
      UINT8_C( 77),
      { EASYSIMD_FLOAT32_C(191569.25), EASYSIMD_FLOAT32_C(   649.02), EASYSIMD_FLOAT32_C(-245967.58), EASYSIMD_FLOAT32_C(282872.91),
        EASYSIMD_FLOAT32_C(   184.46), EASYSIMD_FLOAT32_C(   -16.56), EASYSIMD_FLOAT32_C( 34472.09), EASYSIMD_FLOAT32_C(   231.16) } },
    { { EASYSIMD_FLOAT32_C(  -250.05), EASYSIMD_FLOAT32_C(   323.26), EASYSIMD_FLOAT32_C(   180.47), EASYSIMD_FLOAT32_C(   926.77),
        EASYSIMD_FLOAT32_C(   422.27), EASYSIMD_FLOAT32_C(   -45.19), EASYSIMD_FLOAT32_C(  -319.39), EASYSIMD_FLOAT32_C(   387.80) },
      { EASYSIMD_FLOAT32_C(  -213.62), EASYSIMD_FLOAT32_C(  -905.00), EASYSIMD_FLOAT32_C(   175.68), EASYSIMD_FLOAT32_C(  -403.13),
        EASYSIMD_FLOAT32_C(   161.99), EASYSIMD_FLOAT32_C(   897.74), EASYSIMD_FLOAT32_C(  -180.52), EASYSIMD_FLOAT32_C(   820.53) },
      { EASYSIMD_FLOAT32_C(  -398.28), EASYSIMD_FLOAT32_C(  -497.21), EASYSIMD_FLOAT32_C(   640.04), EASYSIMD_FLOAT32_C(  -893.00),
        EASYSIMD_FLOAT32_C(  -982.83), EASYSIMD_FLOAT32_C(  -538.33), EASYSIMD_FLOAT32_C(   -99.92), EASYSIMD_FLOAT32_C(   666.19) },
      UINT8_C(113),
      { EASYSIMD_FLOAT32_C( 53813.96), EASYSIMD_FLOAT32_C(  -497.21), EASYSIMD_FLOAT32_C(   640.04), EASYSIMD_FLOAT32_C(  -893.00),
        EASYSIMD_FLOAT32_C( 69386.35), EASYSIMD_FLOAT32_C(-40030.54), EASYSIMD_FLOAT32_C( 57756.21), EASYSIMD_FLOAT32_C(   666.19) } },
    { { EASYSIMD_FLOAT32_C(  -815.99), EASYSIMD_FLOAT32_C(  -149.35), EASYSIMD_FLOAT32_C(  -748.65), EASYSIMD_FLOAT32_C(   165.11),
        EASYSIMD_FLOAT32_C(  -918.19), EASYSIMD_FLOAT32_C(   884.86), EASYSIMD_FLOAT32_C(   915.06), EASYSIMD_FLOAT32_C(   405.07) },
      { EASYSIMD_FLOAT32_C(    65.34), EASYSIMD_FLOAT32_C(   841.83), EASYSIMD_FLOAT32_C(  -172.67), EASYSIMD_FLOAT32_C(  -979.85),
        EASYSIMD_FLOAT32_C(  -477.56), EASYSIMD_FLOAT32_C(  -784.86), EASYSIMD_FLOAT32_C(  -193.47), EASYSIMD_FLOAT32_C(  -382.56) },
      { EASYSIMD_FLOAT32_C(   390.82), EASYSIMD_FLOAT32_C(   403.39), EASYSIMD_FLOAT32_C(   779.43), EASYSIMD_FLOAT32_C(   288.56),
        EASYSIMD_FLOAT32_C(  -777.13), EASYSIMD_FLOAT32_C(   599.96), EASYSIMD_FLOAT32_C(   890.28), EASYSIMD_FLOAT32_C(  -274.34) },
      UINT8_C(155),
      { EASYSIMD_FLOAT32_C(-53707.60), EASYSIMD_FLOAT32_C(-126130.71), EASYSIMD_FLOAT32_C(   779.43), EASYSIMD_FLOAT32_C(-162071.59),
        EASYSIMD_FLOAT32_C(439267.94), EASYSIMD_FLOAT32_C(   599.96), EASYSIMD_FLOAT32_C(   890.28), EASYSIMD_FLOAT32_C(-154689.23) } },
    { { EASYSIMD_FLOAT32_C(   997.29), EASYSIMD_FLOAT32_C(  -257.17), EASYSIMD_FLOAT32_C(   701.68), EASYSIMD_FLOAT32_C(  -102.63),
        EASYSIMD_FLOAT32_C(  -590.98), EASYSIMD_FLOAT32_C(   -30.41), EASYSIMD_FLOAT32_C(    81.39), EASYSIMD_FLOAT32_C(   259.67) },
      { EASYSIMD_FLOAT32_C(   220.94), EASYSIMD_FLOAT32_C(  -753.51), EASYSIMD_FLOAT32_C(   341.48), EASYSIMD_FLOAT32_C(   105.80),
        EASYSIMD_FLOAT32_C(  -838.45), EASYSIMD_FLOAT32_C(  -253.45), EASYSIMD_FLOAT32_C(  -828.87), EASYSIMD_FLOAT32_C(  -996.62) },
      { EASYSIMD_FLOAT32_C(   573.89), EASYSIMD_FLOAT32_C(  -808.72), EASYSIMD_FLOAT32_C(  -474.18), EASYSIMD_FLOAT32_C(   789.02),
        EASYSIMD_FLOAT32_C(    -2.19), EASYSIMD_FLOAT32_C(   143.26), EASYSIMD_FLOAT32_C(   179.84), EASYSIMD_FLOAT32_C(  -598.80) },
      UINT8_C( 96),
      { EASYSIMD_FLOAT32_C(   573.89), EASYSIMD_FLOAT32_C(  -808.72), EASYSIMD_FLOAT32_C(  -474.18), EASYSIMD_FLOAT32_C(   789.02),
        EASYSIMD_FLOAT32_C(    -2.19), EASYSIMD_FLOAT32_C(  7564.15), EASYSIMD_FLOAT32_C(-67641.57), EASYSIMD_FLOAT32_C(  -598.80) } },
    { { EASYSIMD_FLOAT32_C(  -531.60), EASYSIMD_FLOAT32_C(  -375.93), EASYSIMD_FLOAT32_C(  -477.35), EASYSIMD_FLOAT32_C(  -641.32),
        EASYSIMD_FLOAT32_C(   349.73), EASYSIMD_FLOAT32_C(   762.65), EASYSIMD_FLOAT32_C(  -644.03), EASYSIMD_FLOAT32_C(  -907.44) },
      { EASYSIMD_FLOAT32_C(   464.33), EASYSIMD_FLOAT32_C(   253.35), EASYSIMD_FLOAT32_C(  -498.42), EASYSIMD_FLOAT32_C(  -566.08),
        EASYSIMD_FLOAT32_C(  -665.27), EASYSIMD_FLOAT32_C(   761.25), EASYSIMD_FLOAT32_C(   654.85), EASYSIMD_FLOAT32_C(  -418.77) },
      { EASYSIMD_FLOAT32_C(   102.73), EASYSIMD_FLOAT32_C(  -239.35), EASYSIMD_FLOAT32_C(  -257.22), EASYSIMD_FLOAT32_C(   849.29),
        EASYSIMD_FLOAT32_C(   -68.22), EASYSIMD_FLOAT32_C(  -253.84), EASYSIMD_FLOAT32_C(   423.17), EASYSIMD_FLOAT32_C(   123.07) },
      UINT8_C(225),
      { EASYSIMD_FLOAT32_C(-246940.55), EASYSIMD_FLOAT32_C(  -239.35), EASYSIMD_FLOAT32_C(  -257.22), EASYSIMD_FLOAT32_C(   849.29),
        EASYSIMD_FLOAT32_C(   -68.22), EASYSIMD_FLOAT32_C(580821.19), EASYSIMD_FLOAT32_C(-422166.22), EASYSIMD_FLOAT32_C(379885.56) } },
    { { EASYSIMD_FLOAT32_C(   212.19), EASYSIMD_FLOAT32_C(  -879.13), EASYSIMD_FLOAT32_C(  -584.76), EASYSIMD_FLOAT32_C(  -607.97),
        EASYSIMD_FLOAT32_C(  -477.92), EASYSIMD_FLOAT32_C(   337.93), EASYSIMD_FLOAT32_C(  -139.57), EASYSIMD_FLOAT32_C(   146.15) },
      { EASYSIMD_FLOAT32_C(   860.58), EASYSIMD_FLOAT32_C(   219.12), EASYSIMD_FLOAT32_C(  -504.12), EASYSIMD_FLOAT32_C(   623.23),
        EASYSIMD_FLOAT32_C(   575.09), EASYSIMD_FLOAT32_C(  -411.56), EASYSIMD_FLOAT32_C(    87.56), EASYSIMD_FLOAT32_C(  -171.56) },
      { EASYSIMD_FLOAT32_C(    90.03), EASYSIMD_FLOAT32_C(   521.48), EASYSIMD_FLOAT32_C(   163.17), EASYSIMD_FLOAT32_C(  -148.72),
        EASYSIMD_FLOAT32_C(   176.33), EASYSIMD_FLOAT32_C(   744.40), EASYSIMD_FLOAT32_C(   954.01), EASYSIMD_FLOAT32_C(   936.98) },
      UINT8_C( 28),
      { EASYSIMD_FLOAT32_C(    90.03), EASYSIMD_FLOAT32_C(   521.48), EASYSIMD_FLOAT32_C(294626.03), EASYSIMD_FLOAT32_C(-378756.41),
        EASYSIMD_FLOAT32_C(-275023.38), EASYSIMD_FLOAT32_C(   744.40), EASYSIMD_FLOAT32_C(   954.01), EASYSIMD_FLOAT32_C(   936.98) } },
    { { EASYSIMD_FLOAT32_C(   803.30), EASYSIMD_FLOAT32_C(  -131.24), EASYSIMD_FLOAT32_C(   233.34), EASYSIMD_FLOAT32_C(   226.47),
        EASYSIMD_FLOAT32_C(   991.83), EASYSIMD_FLOAT32_C(  -494.68), EASYSIMD_FLOAT32_C(  -561.34), EASYSIMD_FLOAT32_C(  -887.30) },
      { EASYSIMD_FLOAT32_C(   -79.44), EASYSIMD_FLOAT32_C(  -169.31), EASYSIMD_FLOAT32_C(  -365.22), EASYSIMD_FLOAT32_C(  -741.51),
        EASYSIMD_FLOAT32_C(   691.13), EASYSIMD_FLOAT32_C(   780.93), EASYSIMD_FLOAT32_C(  -880.93), EASYSIMD_FLOAT32_C(   -89.76) },
      { EASYSIMD_FLOAT32_C(  -723.19), EASYSIMD_FLOAT32_C(   742.31), EASYSIMD_FLOAT32_C(  -514.67), EASYSIMD_FLOAT32_C(  -134.75),
        EASYSIMD_FLOAT32_C(  -170.13), EASYSIMD_FLOAT32_C(   313.77), EASYSIMD_FLOAT32_C(   955.28), EASYSIMD_FLOAT32_C(  -648.65) },
      UINT8_C(253),
      { EASYSIMD_FLOAT32_C(-63090.96), EASYSIMD_FLOAT32_C(   742.31), EASYSIMD_FLOAT32_C(-84705.77), EASYSIMD_FLOAT32_C(-167795.02),
        EASYSIMD_FLOAT32_C(685653.62), EASYSIMD_FLOAT32_C(-386624.22), EASYSIMD_FLOAT32_C(493546.00), EASYSIMD_FLOAT32_C( 80292.70) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r = easysimd_mm256_mask3_fmsub_ps(a, b, c, test_vec[i].k);
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256 c = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 r = easysimd_mm256_mask3_fmsub_ps(a, b, c, k);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_fmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const easysimd__mmask8 k;
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   760.80), EASYSIMD_FLOAT32_C(  -869.71), EASYSIMD_FLOAT32_C(   650.39), EASYSIMD_FLOAT32_C(  -704.52),
        EASYSIMD_FLOAT32_C(  -365.12), EASYSIMD_FLOAT32_C(  -720.70), EASYSIMD_FLOAT32_C(  -182.13), EASYSIMD_FLOAT32_C(   336.49) },
      UINT8_C(235),
      { EASYSIMD_FLOAT32_C(  -819.89), EASYSIMD_FLOAT32_C(  -357.32), EASYSIMD_FLOAT32_C(   -17.10), EASYSIMD_FLOAT32_C(  -725.98),
        EASYSIMD_FLOAT32_C(  -252.19), EASYSIMD_FLOAT32_C(   625.68), EASYSIMD_FLOAT32_C(   369.39), EASYSIMD_FLOAT32_C(   314.52) },
      { EASYSIMD_FLOAT32_C(   348.46), EASYSIMD_FLOAT32_C(   164.55), EASYSIMD_FLOAT32_C(   456.93), EASYSIMD_FLOAT32_C(   475.55),
        EASYSIMD_FLOAT32_C(  -663.97), EASYSIMD_FLOAT32_C(  -697.06), EASYSIMD_FLOAT32_C(  -282.66), EASYSIMD_FLOAT32_C(   -30.80) },
      { EASYSIMD_FLOAT32_C(-624120.75), EASYSIMD_FLOAT32_C(310600.25), EASYSIMD_FLOAT32_C(   650.39), EASYSIMD_FLOAT32_C(510991.88),
        EASYSIMD_FLOAT32_C(  -365.12), EASYSIMD_FLOAT32_C(-450230.53), EASYSIMD_FLOAT32_C(-66994.34), EASYSIMD_FLOAT32_C(105863.62) } },
    { { EASYSIMD_FLOAT32_C(  -684.38), EASYSIMD_FLOAT32_C(   338.18), EASYSIMD_FLOAT32_C(  -475.07), EASYSIMD_FLOAT32_C(   194.10),
        EASYSIMD_FLOAT32_C(   633.33), EASYSIMD_FLOAT32_C(  -519.45), EASYSIMD_FLOAT32_C(   -45.10), EASYSIMD_FLOAT32_C(   763.62) },
      UINT8_C( 51),
      { EASYSIMD_FLOAT32_C(   250.38), EASYSIMD_FLOAT32_C(  -601.50), EASYSIMD_FLOAT32_C(  -589.77), EASYSIMD_FLOAT32_C(  -931.75),
        EASYSIMD_FLOAT32_C(   734.99), EASYSIMD_FLOAT32_C(  -989.98), EASYSIMD_FLOAT32_C(  -751.64), EASYSIMD_FLOAT32_C(  -622.33) },
      { EASYSIMD_FLOAT32_C(    -7.08), EASYSIMD_FLOAT32_C(  -477.62), EASYSIMD_FLOAT32_C(   125.48), EASYSIMD_FLOAT32_C(  -381.41),
        EASYSIMD_FLOAT32_C(   891.77), EASYSIMD_FLOAT32_C(  -560.00), EASYSIMD_FLOAT32_C(   967.05), EASYSIMD_FLOAT32_C(    56.31) },
      { EASYSIMD_FLOAT32_C(-171347.98), EASYSIMD_FLOAT32_C(-202937.64), EASYSIMD_FLOAT32_C(  -475.07), EASYSIMD_FLOAT32_C(   194.10),
        EASYSIMD_FLOAT32_C(464599.44), EASYSIMD_FLOAT32_C(514805.12), EASYSIMD_FLOAT32_C(   -45.10), EASYSIMD_FLOAT32_C(   763.62) } },
    { { EASYSIMD_FLOAT32_C(   896.92), EASYSIMD_FLOAT32_C(   442.60), EASYSIMD_FLOAT32_C(   392.34), EASYSIMD_FLOAT32_C(  -800.14),
        EASYSIMD_FLOAT32_C(  -840.06), EASYSIMD_FLOAT32_C(  -638.46), EASYSIMD_FLOAT32_C(  -484.52), EASYSIMD_FLOAT32_C(   498.12) },
      UINT8_C( 14),
      { EASYSIMD_FLOAT32_C(   709.58), EASYSIMD_FLOAT32_C(   131.45), EASYSIMD_FLOAT32_C(   367.02), EASYSIMD_FLOAT32_C(  -335.52),
        EASYSIMD_FLOAT32_C(  -104.93), EASYSIMD_FLOAT32_C(   497.96), EASYSIMD_FLOAT32_C(   914.86), EASYSIMD_FLOAT32_C(   293.57) },
      { EASYSIMD_FLOAT32_C(   908.19), EASYSIMD_FLOAT32_C(   983.11), EASYSIMD_FLOAT32_C(    28.57), EASYSIMD_FLOAT32_C(   918.21),
        EASYSIMD_FLOAT32_C(  -768.53), EASYSIMD_FLOAT32_C(   406.24), EASYSIMD_FLOAT32_C(   -88.87), EASYSIMD_FLOAT32_C(  -246.15) },
      { EASYSIMD_FLOAT32_C(   896.92), EASYSIMD_FLOAT32_C( 57196.66), EASYSIMD_FLOAT32_C(143968.05), EASYSIMD_FLOAT32_C(267544.75),
        EASYSIMD_FLOAT32_C(  -840.06), EASYSIMD_FLOAT32_C(  -638.46), EASYSIMD_FLOAT32_C(  -484.52), EASYSIMD_FLOAT32_C(   498.12) } },
    { { EASYSIMD_FLOAT32_C(  -468.28), EASYSIMD_FLOAT32_C(   529.72), EASYSIMD_FLOAT32_C(  -354.38), EASYSIMD_FLOAT32_C(   -28.29),
        EASYSIMD_FLOAT32_C(   496.77), EASYSIMD_FLOAT32_C(   701.94), EASYSIMD_FLOAT32_C(  -131.36), EASYSIMD_FLOAT32_C(   -60.63) },
      UINT8_C(124),
      { EASYSIMD_FLOAT32_C(    68.50), EASYSIMD_FLOAT32_C(    99.31), EASYSIMD_FLOAT32_C(   455.82), EASYSIMD_FLOAT32_C(   583.98),
        EASYSIMD_FLOAT32_C(  -402.57), EASYSIMD_FLOAT32_C(  -657.71), EASYSIMD_FLOAT32_C(   293.57), EASYSIMD_FLOAT32_C(   728.88) },
      { EASYSIMD_FLOAT32_C(   709.31), EASYSIMD_FLOAT32_C(   958.05), EASYSIMD_FLOAT32_C(  -376.05), EASYSIMD_FLOAT32_C(   207.27),
        EASYSIMD_FLOAT32_C(   872.91), EASYSIMD_FLOAT32_C(   917.52), EASYSIMD_FLOAT32_C(   115.46), EASYSIMD_FLOAT32_C(   856.02) },
      { EASYSIMD_FLOAT32_C(  -468.28), EASYSIMD_FLOAT32_C(   529.72), EASYSIMD_FLOAT32_C(-161157.45), EASYSIMD_FLOAT32_C(-16728.06),
        EASYSIMD_FLOAT32_C(-200857.61), EASYSIMD_FLOAT32_C(-462590.50), EASYSIMD_FLOAT32_C(-38678.82), EASYSIMD_FLOAT32_C(   -60.63) } },
    { { EASYSIMD_FLOAT32_C(   -53.91), EASYSIMD_FLOAT32_C(    33.67), EASYSIMD_FLOAT32_C(  -912.51), EASYSIMD_FLOAT32_C(  -647.67),
        EASYSIMD_FLOAT32_C(   944.80), EASYSIMD_FLOAT32_C(  -158.65), EASYSIMD_FLOAT32_C(  -115.96), EASYSIMD_FLOAT32_C(   474.52) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT32_C(   855.75), EASYSIMD_FLOAT32_C(   -28.71), EASYSIMD_FLOAT32_C(   188.90), EASYSIMD_FLOAT32_C(  -275.61),
        EASYSIMD_FLOAT32_C(   910.66), EASYSIMD_FLOAT32_C(  -716.82), EASYSIMD_FLOAT32_C(   792.89), EASYSIMD_FLOAT32_C(     9.98) },
      { EASYSIMD_FLOAT32_C(   739.00), EASYSIMD_FLOAT32_C(   376.87), EASYSIMD_FLOAT32_C(   607.41), EASYSIMD_FLOAT32_C(  -918.71),
        EASYSIMD_FLOAT32_C(  -329.56), EASYSIMD_FLOAT32_C(   336.28), EASYSIMD_FLOAT32_C(   790.60), EASYSIMD_FLOAT32_C(  -371.52) },
      { EASYSIMD_FLOAT32_C(-46872.48), EASYSIMD_FLOAT32_C(    33.67), EASYSIMD_FLOAT32_C(  -912.51), EASYSIMD_FLOAT32_C(  -647.67),
        EASYSIMD_FLOAT32_C(860721.06), EASYSIMD_FLOAT32_C(  -158.65), EASYSIMD_FLOAT32_C(-92734.12), EASYSIMD_FLOAT32_C(   474.52) } },
    { { EASYSIMD_FLOAT32_C(   960.23), EASYSIMD_FLOAT32_C(    -2.14), EASYSIMD_FLOAT32_C(  -498.61), EASYSIMD_FLOAT32_C(   877.75),
        EASYSIMD_FLOAT32_C(  -886.68), EASYSIMD_FLOAT32_C(  -642.59), EASYSIMD_FLOAT32_C(  -176.16), EASYSIMD_FLOAT32_C(   146.99) },
      UINT8_C(110),
      { EASYSIMD_FLOAT32_C(   176.16), EASYSIMD_FLOAT32_C(    91.79), EASYSIMD_FLOAT32_C(   286.25), EASYSIMD_FLOAT32_C(  -939.79),
        EASYSIMD_FLOAT32_C(  -433.69), EASYSIMD_FLOAT32_C(  -226.78), EASYSIMD_FLOAT32_C(   915.96), EASYSIMD_FLOAT32_C(   537.60) },
      { EASYSIMD_FLOAT32_C(   962.13), EASYSIMD_FLOAT32_C(  -359.65), EASYSIMD_FLOAT32_C(   448.26), EASYSIMD_FLOAT32_C(  -754.69),
        EASYSIMD_FLOAT32_C(  -566.76), EASYSIMD_FLOAT32_C(  -541.77), EASYSIMD_FLOAT32_C(   984.31), EASYSIMD_FLOAT32_C(   810.11) },
      { EASYSIMD_FLOAT32_C(   960.23), EASYSIMD_FLOAT32_C(   163.22), EASYSIMD_FLOAT32_C(-143175.38), EASYSIMD_FLOAT32_C(-824145.94),
        EASYSIMD_FLOAT32_C(  -886.68), EASYSIMD_FLOAT32_C(146268.33), EASYSIMD_FLOAT32_C(-162339.83), EASYSIMD_FLOAT32_C(   146.99) } },
    { { EASYSIMD_FLOAT32_C(  -934.36), EASYSIMD_FLOAT32_C(  -934.40), EASYSIMD_FLOAT32_C(  -519.45), EASYSIMD_FLOAT32_C(   401.92),
        EASYSIMD_FLOAT32_C(   856.20), EASYSIMD_FLOAT32_C(   109.03), EASYSIMD_FLOAT32_C(   362.15), EASYSIMD_FLOAT32_C(  -145.94) },
      UINT8_C(125),
      { EASYSIMD_FLOAT32_C(   239.90), EASYSIMD_FLOAT32_C(   -32.61), EASYSIMD_FLOAT32_C(   967.84), EASYSIMD_FLOAT32_C(  -936.26),
        EASYSIMD_FLOAT32_C(  -885.62), EASYSIMD_FLOAT32_C(  -587.25), EASYSIMD_FLOAT32_C(   239.91), EASYSIMD_FLOAT32_C(   206.16) },
      { EASYSIMD_FLOAT32_C(   699.00), EASYSIMD_FLOAT32_C(   300.11), EASYSIMD_FLOAT32_C(   772.47), EASYSIMD_FLOAT32_C(  -527.77),
        EASYSIMD_FLOAT32_C(   216.07), EASYSIMD_FLOAT32_C(   310.07), EASYSIMD_FLOAT32_C(  -565.65), EASYSIMD_FLOAT32_C(   856.43) },
      { EASYSIMD_FLOAT32_C(-224851.95), EASYSIMD_FLOAT32_C(  -934.40), EASYSIMD_FLOAT32_C(-503516.97), EASYSIMD_FLOAT32_C(-375773.88),
        EASYSIMD_FLOAT32_C(-758483.94), EASYSIMD_FLOAT32_C(-64337.94), EASYSIMD_FLOAT32_C( 87449.05), EASYSIMD_FLOAT32_C(  -145.94) } },
    { { EASYSIMD_FLOAT32_C(  -241.67), EASYSIMD_FLOAT32_C(  -320.34), EASYSIMD_FLOAT32_C(  -710.33), EASYSIMD_FLOAT32_C(   216.56),
        EASYSIMD_FLOAT32_C(  -336.02), EASYSIMD_FLOAT32_C(  -900.22), EASYSIMD_FLOAT32_C(   282.20), EASYSIMD_FLOAT32_C(  -270.42) },
      UINT8_C(149),
      { EASYSIMD_FLOAT32_C(  -315.88), EASYSIMD_FLOAT32_C(  -414.22), EASYSIMD_FLOAT32_C(   689.36), EASYSIMD_FLOAT32_C(  -953.73),
        EASYSIMD_FLOAT32_C(   439.85), EASYSIMD_FLOAT32_C(   299.78), EASYSIMD_FLOAT32_C(   286.18), EASYSIMD_FLOAT32_C(  -592.77) },
      { EASYSIMD_FLOAT32_C(   267.62), EASYSIMD_FLOAT32_C(   349.92), EASYSIMD_FLOAT32_C(  -478.39), EASYSIMD_FLOAT32_C(   680.37),
        EASYSIMD_FLOAT32_C(  -410.18), EASYSIMD_FLOAT32_C(   727.77), EASYSIMD_FLOAT32_C(   379.37), EASYSIMD_FLOAT32_C(   889.94) },
      { EASYSIMD_FLOAT32_C( 76071.10), EASYSIMD_FLOAT32_C(  -320.34), EASYSIMD_FLOAT32_C(-489194.69), EASYSIMD_FLOAT32_C(   216.56),
        EASYSIMD_FLOAT32_C(-147388.22), EASYSIMD_FLOAT32_C(  -900.22), EASYSIMD_FLOAT32_C(   282.20), EASYSIMD_FLOAT32_C(159406.94) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_fmsub_ps(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_fmsub_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 c = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_fmsub_ps(a, k, b, c);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_fmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(241),
      { EASYSIMD_FLOAT32_C(  -894.70), EASYSIMD_FLOAT32_C(  -318.54), EASYSIMD_FLOAT32_C(   690.66), EASYSIMD_FLOAT32_C(  -902.45),
        EASYSIMD_FLOAT32_C(  -599.03), EASYSIMD_FLOAT32_C(   385.31), EASYSIMD_FLOAT32_C(  -662.85), EASYSIMD_FLOAT32_C(   949.38) },
      { EASYSIMD_FLOAT32_C(   -41.50), EASYSIMD_FLOAT32_C(  -230.23), EASYSIMD_FLOAT32_C(  -619.36), EASYSIMD_FLOAT32_C(  -932.10),
        EASYSIMD_FLOAT32_C(   694.10), EASYSIMD_FLOAT32_C(   357.34), EASYSIMD_FLOAT32_C(  -490.89), EASYSIMD_FLOAT32_C(   404.47) },
      { EASYSIMD_FLOAT32_C(    97.18), EASYSIMD_FLOAT32_C(   -45.79), EASYSIMD_FLOAT32_C(   163.45), EASYSIMD_FLOAT32_C(  -463.68),
        EASYSIMD_FLOAT32_C(  -742.60), EASYSIMD_FLOAT32_C(  -196.27), EASYSIMD_FLOAT32_C(   186.61), EASYSIMD_FLOAT32_C(  -938.11) },
      { EASYSIMD_FLOAT32_C( 37032.87), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-415044.12), EASYSIMD_FLOAT32_C(137882.94), EASYSIMD_FLOAT32_C(325199.81), EASYSIMD_FLOAT32_C(384933.84) } },
    { UINT8_C(219),
      { EASYSIMD_FLOAT32_C(   559.49), EASYSIMD_FLOAT32_C(   -58.33), EASYSIMD_FLOAT32_C(   654.48), EASYSIMD_FLOAT32_C(  -999.12),
        EASYSIMD_FLOAT32_C(  -656.50), EASYSIMD_FLOAT32_C(  -460.74), EASYSIMD_FLOAT32_C(  -893.81), EASYSIMD_FLOAT32_C(    24.96) },
      { EASYSIMD_FLOAT32_C(  -770.08), EASYSIMD_FLOAT32_C(  -796.26), EASYSIMD_FLOAT32_C(   425.93), EASYSIMD_FLOAT32_C(   615.23),
        EASYSIMD_FLOAT32_C(  -459.10), EASYSIMD_FLOAT32_C(   375.32), EASYSIMD_FLOAT32_C(  -426.27), EASYSIMD_FLOAT32_C(   310.66) },
      { EASYSIMD_FLOAT32_C(   755.96), EASYSIMD_FLOAT32_C(  -358.37), EASYSIMD_FLOAT32_C(     4.76), EASYSIMD_FLOAT32_C(   113.30),
        EASYSIMD_FLOAT32_C(   150.74), EASYSIMD_FLOAT32_C(  -590.77), EASYSIMD_FLOAT32_C(  -789.53), EASYSIMD_FLOAT32_C(  -895.06) },
      { EASYSIMD_FLOAT32_C(-431608.03), EASYSIMD_FLOAT32_C( 46804.22), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-614801.88),
        EASYSIMD_FLOAT32_C(301248.41), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(381793.91), EASYSIMD_FLOAT32_C(  8649.13) } },
    { UINT8_C( 31),
      { EASYSIMD_FLOAT32_C(  -253.21), EASYSIMD_FLOAT32_C(  -637.65), EASYSIMD_FLOAT32_C(  -623.59), EASYSIMD_FLOAT32_C(   933.40),
        EASYSIMD_FLOAT32_C(  -575.76), EASYSIMD_FLOAT32_C(   563.65), EASYSIMD_FLOAT32_C(   492.89), EASYSIMD_FLOAT32_C(   365.91) },
      { EASYSIMD_FLOAT32_C(   218.13), EASYSIMD_FLOAT32_C(   493.78), EASYSIMD_FLOAT32_C(   709.40), EASYSIMD_FLOAT32_C(   757.39),
        EASYSIMD_FLOAT32_C(   599.96), EASYSIMD_FLOAT32_C(  -265.63), EASYSIMD_FLOAT32_C(   987.30), EASYSIMD_FLOAT32_C(   803.71) },
      { EASYSIMD_FLOAT32_C(  -839.70), EASYSIMD_FLOAT32_C(   602.53), EASYSIMD_FLOAT32_C(  -655.40), EASYSIMD_FLOAT32_C(   535.62),
        EASYSIMD_FLOAT32_C(  -823.73), EASYSIMD_FLOAT32_C(   655.27), EASYSIMD_FLOAT32_C(   291.58), EASYSIMD_FLOAT32_C(  -182.10) },
      { EASYSIMD_FLOAT32_C(-54393.00), EASYSIMD_FLOAT32_C(-315461.34), EASYSIMD_FLOAT32_C(-441719.38), EASYSIMD_FLOAT32_C(706412.25),
        EASYSIMD_FLOAT32_C(-344609.25), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(129),
      { EASYSIMD_FLOAT32_C(  -595.13), EASYSIMD_FLOAT32_C(   968.63), EASYSIMD_FLOAT32_C(    69.26), EASYSIMD_FLOAT32_C(  -384.65),
        EASYSIMD_FLOAT32_C(  -926.42), EASYSIMD_FLOAT32_C(  -358.05), EASYSIMD_FLOAT32_C(   362.14), EASYSIMD_FLOAT32_C(  -564.08) },
      { EASYSIMD_FLOAT32_C(    18.36), EASYSIMD_FLOAT32_C(   295.54), EASYSIMD_FLOAT32_C(  -139.84), EASYSIMD_FLOAT32_C(  -417.99),
        EASYSIMD_FLOAT32_C(  -211.57), EASYSIMD_FLOAT32_C(  -773.93), EASYSIMD_FLOAT32_C(   800.14), EASYSIMD_FLOAT32_C(  -717.79) },
      { EASYSIMD_FLOAT32_C(   935.47), EASYSIMD_FLOAT32_C(   557.52), EASYSIMD_FLOAT32_C(   882.18), EASYSIMD_FLOAT32_C(  -330.16),
        EASYSIMD_FLOAT32_C(   544.82), EASYSIMD_FLOAT32_C(   685.88), EASYSIMD_FLOAT32_C(  -169.86), EASYSIMD_FLOAT32_C(   147.36) },
      { EASYSIMD_FLOAT32_C(-11862.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(404743.62) } },
    { UINT8_C(135),
      { EASYSIMD_FLOAT32_C(  -634.24), EASYSIMD_FLOAT32_C(   323.62), EASYSIMD_FLOAT32_C(   685.75), EASYSIMD_FLOAT32_C(   657.33),
        EASYSIMD_FLOAT32_C(  -858.48), EASYSIMD_FLOAT32_C(  -654.22), EASYSIMD_FLOAT32_C(  -937.79), EASYSIMD_FLOAT32_C(  -889.85) },
      { EASYSIMD_FLOAT32_C(   415.04), EASYSIMD_FLOAT32_C(  -322.45), EASYSIMD_FLOAT32_C(  -816.27), EASYSIMD_FLOAT32_C(  -943.01),
        EASYSIMD_FLOAT32_C(  -960.31), EASYSIMD_FLOAT32_C(  -380.35), EASYSIMD_FLOAT32_C(    75.35), EASYSIMD_FLOAT32_C(   335.23) },
      { EASYSIMD_FLOAT32_C(   479.81), EASYSIMD_FLOAT32_C(   657.36), EASYSIMD_FLOAT32_C(  -876.33), EASYSIMD_FLOAT32_C(   705.88),
        EASYSIMD_FLOAT32_C(   457.50), EASYSIMD_FLOAT32_C(  -594.12), EASYSIMD_FLOAT32_C(   641.35), EASYSIMD_FLOAT32_C(    15.02) },
      { EASYSIMD_FLOAT32_C(-263714.78), EASYSIMD_FLOAT32_C(-105008.63), EASYSIMD_FLOAT32_C(-558880.81), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-298319.44) } },
    { UINT8_C(250),
      { EASYSIMD_FLOAT32_C(  -688.81), EASYSIMD_FLOAT32_C(  -440.15), EASYSIMD_FLOAT32_C(   973.94), EASYSIMD_FLOAT32_C(   141.33),
        EASYSIMD_FLOAT32_C(   707.20), EASYSIMD_FLOAT32_C(  -995.58), EASYSIMD_FLOAT32_C(   507.09), EASYSIMD_FLOAT32_C(    30.82) },
      { EASYSIMD_FLOAT32_C(   690.18), EASYSIMD_FLOAT32_C(   164.42), EASYSIMD_FLOAT32_C(   172.34), EASYSIMD_FLOAT32_C(  -964.04),
        EASYSIMD_FLOAT32_C(   226.62), EASYSIMD_FLOAT32_C(   282.50), EASYSIMD_FLOAT32_C(   451.00), EASYSIMD_FLOAT32_C(   904.18) },
      { EASYSIMD_FLOAT32_C(   466.22), EASYSIMD_FLOAT32_C(   507.99), EASYSIMD_FLOAT32_C(   943.87), EASYSIMD_FLOAT32_C(  -914.12),
        EASYSIMD_FLOAT32_C(  -416.66), EASYSIMD_FLOAT32_C(   279.10), EASYSIMD_FLOAT32_C(   565.69), EASYSIMD_FLOAT32_C(  -759.30) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-72877.45), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-135333.66),
        EASYSIMD_FLOAT32_C(160682.33), EASYSIMD_FLOAT32_C(-281530.47), EASYSIMD_FLOAT32_C(228131.89), EASYSIMD_FLOAT32_C( 28626.13) } },
    { UINT8_C( 26),
      { EASYSIMD_FLOAT32_C(   271.57), EASYSIMD_FLOAT32_C(   698.20), EASYSIMD_FLOAT32_C(   808.65), EASYSIMD_FLOAT32_C(   -87.07),
        EASYSIMD_FLOAT32_C(  -286.78), EASYSIMD_FLOAT32_C(  -903.30), EASYSIMD_FLOAT32_C(   224.12), EASYSIMD_FLOAT32_C(   273.06) },
      { EASYSIMD_FLOAT32_C(  -929.36), EASYSIMD_FLOAT32_C(  -634.55), EASYSIMD_FLOAT32_C(   -19.73), EASYSIMD_FLOAT32_C(  -924.93),
        EASYSIMD_FLOAT32_C(   872.54), EASYSIMD_FLOAT32_C(  -988.91), EASYSIMD_FLOAT32_C(   765.24), EASYSIMD_FLOAT32_C(    36.96) },
      { EASYSIMD_FLOAT32_C(   183.43), EASYSIMD_FLOAT32_C(   801.20), EASYSIMD_FLOAT32_C(  -736.42), EASYSIMD_FLOAT32_C(  -534.07),
        EASYSIMD_FLOAT32_C(   252.20), EASYSIMD_FLOAT32_C(  -832.24), EASYSIMD_FLOAT32_C(   932.15), EASYSIMD_FLOAT32_C(  -239.82) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-443844.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 81067.73),
        EASYSIMD_FLOAT32_C(-250479.22), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(190),
      { EASYSIMD_FLOAT32_C(  -981.97), EASYSIMD_FLOAT32_C(   343.52), EASYSIMD_FLOAT32_C(   390.73), EASYSIMD_FLOAT32_C(   583.72),
        EASYSIMD_FLOAT32_C(   584.22), EASYSIMD_FLOAT32_C(  -206.50), EASYSIMD_FLOAT32_C(  -144.71), EASYSIMD_FLOAT32_C(   282.42) },
      { EASYSIMD_FLOAT32_C(  -397.85), EASYSIMD_FLOAT32_C(   768.21), EASYSIMD_FLOAT32_C(   995.64), EASYSIMD_FLOAT32_C(  -301.14),
        EASYSIMD_FLOAT32_C(    -7.67), EASYSIMD_FLOAT32_C(   268.70), EASYSIMD_FLOAT32_C(  -230.50), EASYSIMD_FLOAT32_C(   357.78) },
      { EASYSIMD_FLOAT32_C(  -751.04), EASYSIMD_FLOAT32_C(  -155.43), EASYSIMD_FLOAT32_C(   230.32), EASYSIMD_FLOAT32_C(  -739.94),
        EASYSIMD_FLOAT32_C(  -390.19), EASYSIMD_FLOAT32_C(  -732.72), EASYSIMD_FLOAT32_C(   443.49), EASYSIMD_FLOAT32_C(  -588.99) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(264050.94), EASYSIMD_FLOAT32_C(388796.12), EASYSIMD_FLOAT32_C(-175041.50),
        EASYSIMD_FLOAT32_C( -4090.78), EASYSIMD_FLOAT32_C(-54753.83), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(101633.22) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_fmsub_ps(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_fmsub_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256 c = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_fmsub_ps(k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask3_fmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd__mmask8 k;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   240.91), EASYSIMD_FLOAT32_C(   -54.50), EASYSIMD_FLOAT32_C(  -637.17), EASYSIMD_FLOAT32_C(   388.12) },
      { EASYSIMD_FLOAT32_C(  -792.37), EASYSIMD_FLOAT32_C(  -885.59), EASYSIMD_FLOAT32_C(   487.51), EASYSIMD_FLOAT32_C(   603.37) },
      { EASYSIMD_FLOAT32_C(  -728.21), EASYSIMD_FLOAT32_C(   576.82), EASYSIMD_FLOAT32_C(   999.18), EASYSIMD_FLOAT32_C(   128.93) },
      UINT8_C(152),
      { EASYSIMD_FLOAT32_C(  -728.21), EASYSIMD_FLOAT32_C(   576.82), EASYSIMD_FLOAT32_C(   999.18), EASYSIMD_FLOAT32_C(234051.03) } },
    { { EASYSIMD_FLOAT32_C(   465.73), EASYSIMD_FLOAT32_C(   515.39), EASYSIMD_FLOAT32_C(  -683.78), EASYSIMD_FLOAT32_C(   180.15) },
      { EASYSIMD_FLOAT32_C(  -414.45), EASYSIMD_FLOAT32_C(    47.55), EASYSIMD_FLOAT32_C(   164.21), EASYSIMD_FLOAT32_C(   638.68) },
      { EASYSIMD_FLOAT32_C(   124.40), EASYSIMD_FLOAT32_C(   -10.14), EASYSIMD_FLOAT32_C(  -750.03), EASYSIMD_FLOAT32_C(    99.19) },
      UINT8_C(232),
      { EASYSIMD_FLOAT32_C(   124.40), EASYSIMD_FLOAT32_C(   -10.14), EASYSIMD_FLOAT32_C(  -750.03), EASYSIMD_FLOAT32_C(114959.01) } },
    { { EASYSIMD_FLOAT32_C(   -96.47), EASYSIMD_FLOAT32_C(  -394.08), EASYSIMD_FLOAT32_C(   145.35), EASYSIMD_FLOAT32_C(   876.06) },
      { EASYSIMD_FLOAT32_C(   981.90), EASYSIMD_FLOAT32_C(  -613.74), EASYSIMD_FLOAT32_C(  -178.44), EASYSIMD_FLOAT32_C(  -655.27) },
      { EASYSIMD_FLOAT32_C(   774.38), EASYSIMD_FLOAT32_C(    29.19), EASYSIMD_FLOAT32_C(  -540.86), EASYSIMD_FLOAT32_C(   261.90) },
      UINT8_C(133),
      { EASYSIMD_FLOAT32_C(-95498.27), EASYSIMD_FLOAT32_C(    29.19), EASYSIMD_FLOAT32_C(-25395.39), EASYSIMD_FLOAT32_C(   261.90) } },
    { { EASYSIMD_FLOAT32_C(  -269.06), EASYSIMD_FLOAT32_C(  -161.29), EASYSIMD_FLOAT32_C(  -368.26), EASYSIMD_FLOAT32_C(   859.86) },
      { EASYSIMD_FLOAT32_C(   884.53), EASYSIMD_FLOAT32_C(  -902.54), EASYSIMD_FLOAT32_C(   375.25), EASYSIMD_FLOAT32_C(  -799.25) },
      { EASYSIMD_FLOAT32_C(   277.62), EASYSIMD_FLOAT32_C(   960.80), EASYSIMD_FLOAT32_C(   248.30), EASYSIMD_FLOAT32_C(  -558.17) },
      UINT8_C(228),
      { EASYSIMD_FLOAT32_C(   277.62), EASYSIMD_FLOAT32_C(   960.80), EASYSIMD_FLOAT32_C(-138437.88), EASYSIMD_FLOAT32_C(  -558.17) } },
    { { EASYSIMD_FLOAT32_C(  -627.30), EASYSIMD_FLOAT32_C(   431.69), EASYSIMD_FLOAT32_C(   849.45), EASYSIMD_FLOAT32_C(   471.89) },
      { EASYSIMD_FLOAT32_C(   633.66), EASYSIMD_FLOAT32_C(  -247.01), EASYSIMD_FLOAT32_C(  -922.19), EASYSIMD_FLOAT32_C(  -220.99) },
      { EASYSIMD_FLOAT32_C(  -370.95), EASYSIMD_FLOAT32_C(  -940.29), EASYSIMD_FLOAT32_C(   165.27), EASYSIMD_FLOAT32_C(   450.61) },
      UINT8_C(203),
      { EASYSIMD_FLOAT32_C(-397123.94), EASYSIMD_FLOAT32_C(-105691.45), EASYSIMD_FLOAT32_C(   165.27), EASYSIMD_FLOAT32_C(-104733.59) } },
    { { EASYSIMD_FLOAT32_C(   -60.34), EASYSIMD_FLOAT32_C(  -520.20), EASYSIMD_FLOAT32_C(  -136.42), EASYSIMD_FLOAT32_C(  -798.45) },
      { EASYSIMD_FLOAT32_C(   112.36), EASYSIMD_FLOAT32_C(   594.52), EASYSIMD_FLOAT32_C(    40.26), EASYSIMD_FLOAT32_C(   744.09) },
      { EASYSIMD_FLOAT32_C(   454.39), EASYSIMD_FLOAT32_C(   -75.20), EASYSIMD_FLOAT32_C(   841.56), EASYSIMD_FLOAT32_C(  -170.36) },
      UINT8_C(176),
      { EASYSIMD_FLOAT32_C(   454.39), EASYSIMD_FLOAT32_C(   -75.20), EASYSIMD_FLOAT32_C(   841.56), EASYSIMD_FLOAT32_C(  -170.36) } },
    { { EASYSIMD_FLOAT32_C(   119.18), EASYSIMD_FLOAT32_C(  -209.56), EASYSIMD_FLOAT32_C(  -626.16), EASYSIMD_FLOAT32_C(   561.00) },
      { EASYSIMD_FLOAT32_C(  -610.08), EASYSIMD_FLOAT32_C(  -253.46), EASYSIMD_FLOAT32_C(    -7.31), EASYSIMD_FLOAT32_C(  -760.62) },
      { EASYSIMD_FLOAT32_C(  -781.56), EASYSIMD_FLOAT32_C(  -373.65), EASYSIMD_FLOAT32_C(    -7.64), EASYSIMD_FLOAT32_C(  -703.75) },
      UINT8_C(104),
      { EASYSIMD_FLOAT32_C(  -781.56), EASYSIMD_FLOAT32_C(  -373.65), EASYSIMD_FLOAT32_C(    -7.64), EASYSIMD_FLOAT32_C(-426004.06) } },
    { { EASYSIMD_FLOAT32_C(   621.41), EASYSIMD_FLOAT32_C(  -644.04), EASYSIMD_FLOAT32_C(  -429.37), EASYSIMD_FLOAT32_C(    72.02) },
      { EASYSIMD_FLOAT32_C(  -239.60), EASYSIMD_FLOAT32_C(   510.28), EASYSIMD_FLOAT32_C(   551.82), EASYSIMD_FLOAT32_C(   623.98) },
      { EASYSIMD_FLOAT32_C(   711.83), EASYSIMD_FLOAT32_C(  -335.82), EASYSIMD_FLOAT32_C(   218.50), EASYSIMD_FLOAT32_C(  -247.90) },
      UINT8_C(135),
      { EASYSIMD_FLOAT32_C(-149601.66), EASYSIMD_FLOAT32_C(-328304.91), EASYSIMD_FLOAT32_C(-237153.45), EASYSIMD_FLOAT32_C(  -247.90) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask3_fmsub_ps(a, b, c, test_vec[i].k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask3_fmsub_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 c = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 r = easysimd_mm_mask3_fmsub_ps(a, b, c, k);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask3_fmsubadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd__mmask8 k;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -479.75), EASYSIMD_FLOAT32_C(   334.28), EASYSIMD_FLOAT32_C(   -28.35), EASYSIMD_FLOAT32_C(   111.59) },
      { EASYSIMD_FLOAT32_C(   498.30), EASYSIMD_FLOAT32_C(  -541.52), EASYSIMD_FLOAT32_C(   845.40), EASYSIMD_FLOAT32_C(   304.01) },
      { EASYSIMD_FLOAT32_C(   576.28), EASYSIMD_FLOAT32_C(  -834.00), EASYSIMD_FLOAT32_C(  -204.94), EASYSIMD_FLOAT32_C(   980.44) },
      UINT8_C(226),
      { EASYSIMD_FLOAT32_C(   576.28), EASYSIMD_FLOAT32_C(-180185.31), EASYSIMD_FLOAT32_C(  -204.94), EASYSIMD_FLOAT32_C(   980.44) } },
    { { EASYSIMD_FLOAT32_C(  -986.01), EASYSIMD_FLOAT32_C(  -216.05), EASYSIMD_FLOAT32_C(  -448.48), EASYSIMD_FLOAT32_C(  -595.49) },
      { EASYSIMD_FLOAT32_C(   767.30), EASYSIMD_FLOAT32_C(   613.36), EASYSIMD_FLOAT32_C(   879.22), EASYSIMD_FLOAT32_C(  -982.03) },
      { EASYSIMD_FLOAT32_C(   255.10), EASYSIMD_FLOAT32_C(  -693.20), EASYSIMD_FLOAT32_C(   749.14), EASYSIMD_FLOAT32_C(  -499.02) },
      UINT8_C(182),
      { EASYSIMD_FLOAT32_C(   255.10), EASYSIMD_FLOAT32_C(-131823.22), EASYSIMD_FLOAT32_C(-393563.47), EASYSIMD_FLOAT32_C(  -499.02) } },
    { { EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(  -212.99), EASYSIMD_FLOAT32_C(  -655.51), EASYSIMD_FLOAT32_C(   587.52) },
      { EASYSIMD_FLOAT32_C(  -695.25), EASYSIMD_FLOAT32_C(  -135.26), EASYSIMD_FLOAT32_C(   -78.20), EASYSIMD_FLOAT32_C(   276.40) },
      { EASYSIMD_FLOAT32_C(   976.32), EASYSIMD_FLOAT32_C(  -579.91), EASYSIMD_FLOAT32_C(   734.88), EASYSIMD_FLOAT32_C(   821.73) },
      UINT8_C( 55),
      { EASYSIMD_FLOAT32_C(-243751.69), EASYSIMD_FLOAT32_C( 29388.94), EASYSIMD_FLOAT32_C( 51995.76), EASYSIMD_FLOAT32_C(   821.73) } },
    { { EASYSIMD_FLOAT32_C(   311.17), EASYSIMD_FLOAT32_C(   987.73), EASYSIMD_FLOAT32_C(  -480.83), EASYSIMD_FLOAT32_C(   291.61) },
      { EASYSIMD_FLOAT32_C(     8.07), EASYSIMD_FLOAT32_C(  -466.84), EASYSIMD_FLOAT32_C(  -924.45), EASYSIMD_FLOAT32_C(   559.59) },
      { EASYSIMD_FLOAT32_C(   -62.32), EASYSIMD_FLOAT32_C(   842.85), EASYSIMD_FLOAT32_C(   172.95), EASYSIMD_FLOAT32_C(  -183.10) },
      UINT8_C(123),
      { EASYSIMD_FLOAT32_C(  2448.82), EASYSIMD_FLOAT32_C(-461954.72), EASYSIMD_FLOAT32_C(   172.95), EASYSIMD_FLOAT32_C(163365.14) } },
    { { EASYSIMD_FLOAT32_C(  -571.95), EASYSIMD_FLOAT32_C(   123.70), EASYSIMD_FLOAT32_C(   609.96), EASYSIMD_FLOAT32_C(   -70.97) },
      { EASYSIMD_FLOAT32_C(  -675.45), EASYSIMD_FLOAT32_C(   -38.04), EASYSIMD_FLOAT32_C(   716.04), EASYSIMD_FLOAT32_C(  -330.97) },
      { EASYSIMD_FLOAT32_C(  -450.52), EASYSIMD_FLOAT32_C(  -979.21), EASYSIMD_FLOAT32_C(   533.77), EASYSIMD_FLOAT32_C(   471.28) },
      UINT8_C(207),
      { EASYSIMD_FLOAT32_C(385873.12), EASYSIMD_FLOAT32_C( -3726.34), EASYSIMD_FLOAT32_C(437289.53), EASYSIMD_FLOAT32_C( 23017.66) } },
    { { EASYSIMD_FLOAT32_C(   510.10), EASYSIMD_FLOAT32_C(   891.37), EASYSIMD_FLOAT32_C(    32.07), EASYSIMD_FLOAT32_C(   331.82) },
      { EASYSIMD_FLOAT32_C(   615.48), EASYSIMD_FLOAT32_C(  -656.77), EASYSIMD_FLOAT32_C(   319.55), EASYSIMD_FLOAT32_C(  -865.35) },
      { EASYSIMD_FLOAT32_C(   634.84), EASYSIMD_FLOAT32_C(  -672.38), EASYSIMD_FLOAT32_C(  -332.19), EASYSIMD_FLOAT32_C(   710.39) },
      UINT8_C( 62),
      { EASYSIMD_FLOAT32_C(   634.84), EASYSIMD_FLOAT32_C(-584752.69), EASYSIMD_FLOAT32_C(  9915.78), EASYSIMD_FLOAT32_C(-287850.81) } },
    { { EASYSIMD_FLOAT32_C(   605.49), EASYSIMD_FLOAT32_C(   553.24), EASYSIMD_FLOAT32_C(    60.16), EASYSIMD_FLOAT32_C(  -577.61) },
      { EASYSIMD_FLOAT32_C(   414.06), EASYSIMD_FLOAT32_C(   488.21), EASYSIMD_FLOAT32_C(   546.08), EASYSIMD_FLOAT32_C(    24.03) },
      { EASYSIMD_FLOAT32_C(  -582.76), EASYSIMD_FLOAT32_C(   870.63), EASYSIMD_FLOAT32_C(   985.99), EASYSIMD_FLOAT32_C(  -866.72) },
      UINT8_C(247),
      { EASYSIMD_FLOAT32_C(250126.42), EASYSIMD_FLOAT32_C(269226.66), EASYSIMD_FLOAT32_C( 33838.16), EASYSIMD_FLOAT32_C(  -866.72) } },
    { { EASYSIMD_FLOAT32_C(  -464.53), EASYSIMD_FLOAT32_C(  -845.94), EASYSIMD_FLOAT32_C(  -926.56), EASYSIMD_FLOAT32_C(  -993.25) },
      { EASYSIMD_FLOAT32_C(   451.25), EASYSIMD_FLOAT32_C(   583.53), EASYSIMD_FLOAT32_C(   898.12), EASYSIMD_FLOAT32_C(  -516.68) },
      { EASYSIMD_FLOAT32_C(   -84.64), EASYSIMD_FLOAT32_C(   513.59), EASYSIMD_FLOAT32_C(  -173.45), EASYSIMD_FLOAT32_C(  -765.09) },
      UINT8_C(183),
      { EASYSIMD_FLOAT32_C(-209703.80), EASYSIMD_FLOAT32_C(-494145.00), EASYSIMD_FLOAT32_C(-832335.50), EASYSIMD_FLOAT32_C(  -765.09) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask3_fmsubadd_ps(a, b, c, test_vec[i].k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask3_fmsubadd_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 c = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 r = easysimd_mm_mask3_fmsubadd_ps(a, b, c, k);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd__mmask8 k;
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -467.30), EASYSIMD_FLOAT32_C(   -97.89), EASYSIMD_FLOAT32_C(   613.57), EASYSIMD_FLOAT32_C(   530.60) },
      UINT8_C( 50),
      { EASYSIMD_FLOAT32_C(  -607.55), EASYSIMD_FLOAT32_C(  -702.66), EASYSIMD_FLOAT32_C(   709.68), EASYSIMD_FLOAT32_C(   121.97) },
      { EASYSIMD_FLOAT32_C(    76.84), EASYSIMD_FLOAT32_C(  -579.92), EASYSIMD_FLOAT32_C(  -412.78), EASYSIMD_FLOAT32_C(  -324.84) },
      { EASYSIMD_FLOAT32_C(  -467.30), EASYSIMD_FLOAT32_C( 69363.30), EASYSIMD_FLOAT32_C(   613.57), EASYSIMD_FLOAT32_C(   530.60) } },
    { { EASYSIMD_FLOAT32_C(   623.05), EASYSIMD_FLOAT32_C(   228.16), EASYSIMD_FLOAT32_C(   749.42), EASYSIMD_FLOAT32_C(  -113.00) },
      UINT8_C( 99),
      { EASYSIMD_FLOAT32_C(   474.58), EASYSIMD_FLOAT32_C(   621.99), EASYSIMD_FLOAT32_C(  -360.00), EASYSIMD_FLOAT32_C(  -226.66) },
      { EASYSIMD_FLOAT32_C(  -631.29), EASYSIMD_FLOAT32_C(   160.93), EASYSIMD_FLOAT32_C(   873.78), EASYSIMD_FLOAT32_C(   653.70) },
      { EASYSIMD_FLOAT32_C(296318.34), EASYSIMD_FLOAT32_C(141752.31), EASYSIMD_FLOAT32_C(   749.42), EASYSIMD_FLOAT32_C(  -113.00) } },
    { { EASYSIMD_FLOAT32_C(  -866.75), EASYSIMD_FLOAT32_C(  -352.64), EASYSIMD_FLOAT32_C(  -395.86), EASYSIMD_FLOAT32_C(  -884.71) },
      UINT8_C( 96),
      { EASYSIMD_FLOAT32_C(   136.84), EASYSIMD_FLOAT32_C(    17.40), EASYSIMD_FLOAT32_C(   376.10), EASYSIMD_FLOAT32_C(  -332.56) },
      { EASYSIMD_FLOAT32_C(  -337.18), EASYSIMD_FLOAT32_C(   768.55), EASYSIMD_FLOAT32_C(   -35.22), EASYSIMD_FLOAT32_C(  -627.50) },
      { EASYSIMD_FLOAT32_C(  -866.75), EASYSIMD_FLOAT32_C(  -352.64), EASYSIMD_FLOAT32_C(  -395.86), EASYSIMD_FLOAT32_C(  -884.71) } },
    { { EASYSIMD_FLOAT32_C(  -109.48), EASYSIMD_FLOAT32_C(  -958.37), EASYSIMD_FLOAT32_C(  -207.41), EASYSIMD_FLOAT32_C(   477.74) },
      UINT8_C(251),
      { EASYSIMD_FLOAT32_C(  -584.37), EASYSIMD_FLOAT32_C(  -294.09), EASYSIMD_FLOAT32_C(  -533.80), EASYSIMD_FLOAT32_C(   302.63) },
      { EASYSIMD_FLOAT32_C(   839.32), EASYSIMD_FLOAT32_C(   940.79), EASYSIMD_FLOAT32_C(   -75.39), EASYSIMD_FLOAT32_C(  -520.68) },
      { EASYSIMD_FLOAT32_C( 63137.51), EASYSIMD_FLOAT32_C(280906.25), EASYSIMD_FLOAT32_C(  -207.41), EASYSIMD_FLOAT32_C(145099.14) } },
    { { EASYSIMD_FLOAT32_C(  -285.88), EASYSIMD_FLOAT32_C(   293.32), EASYSIMD_FLOAT32_C(   640.24), EASYSIMD_FLOAT32_C(  -412.09) },
      UINT8_C(194),
      { EASYSIMD_FLOAT32_C(   773.49), EASYSIMD_FLOAT32_C(   235.27), EASYSIMD_FLOAT32_C(   551.16), EASYSIMD_FLOAT32_C(   888.78) },
      { EASYSIMD_FLOAT32_C(    -2.20), EASYSIMD_FLOAT32_C(  -312.00), EASYSIMD_FLOAT32_C(   -93.83), EASYSIMD_FLOAT32_C(  -626.10) },
      { EASYSIMD_FLOAT32_C(  -285.88), EASYSIMD_FLOAT32_C( 69321.40), EASYSIMD_FLOAT32_C(   640.24), EASYSIMD_FLOAT32_C(  -412.09) } },
    { { EASYSIMD_FLOAT32_C(   355.44), EASYSIMD_FLOAT32_C(   569.00), EASYSIMD_FLOAT32_C(  -857.55), EASYSIMD_FLOAT32_C(  -679.77) },
      UINT8_C(216),
      { EASYSIMD_FLOAT32_C(    32.97), EASYSIMD_FLOAT32_C(  -638.15), EASYSIMD_FLOAT32_C(  -265.91), EASYSIMD_FLOAT32_C(  -489.28) },
      { EASYSIMD_FLOAT32_C(    78.64), EASYSIMD_FLOAT32_C(   149.72), EASYSIMD_FLOAT32_C(   216.62), EASYSIMD_FLOAT32_C(   544.84) },
      { EASYSIMD_FLOAT32_C(   355.44), EASYSIMD_FLOAT32_C(   569.00), EASYSIMD_FLOAT32_C(  -857.55), EASYSIMD_FLOAT32_C(332053.03) } },
    { { EASYSIMD_FLOAT32_C(  -547.66), EASYSIMD_FLOAT32_C(    55.95), EASYSIMD_FLOAT32_C(   485.63), EASYSIMD_FLOAT32_C(   376.96) },
      UINT8_C(205),
      { EASYSIMD_FLOAT32_C(  -800.25), EASYSIMD_FLOAT32_C(  -329.72), EASYSIMD_FLOAT32_C(   175.51), EASYSIMD_FLOAT32_C(  -212.34) },
      { EASYSIMD_FLOAT32_C(   617.30), EASYSIMD_FLOAT32_C(   -51.00), EASYSIMD_FLOAT32_C(  -977.07), EASYSIMD_FLOAT32_C(   168.46) },
      { EASYSIMD_FLOAT32_C(437647.59), EASYSIMD_FLOAT32_C(    55.95), EASYSIMD_FLOAT32_C( 86209.99), EASYSIMD_FLOAT32_C(-80212.14) } },
    { { EASYSIMD_FLOAT32_C(  -162.22), EASYSIMD_FLOAT32_C(    20.73), EASYSIMD_FLOAT32_C(   856.47), EASYSIMD_FLOAT32_C(   743.95) },
      UINT8_C(134),
      { EASYSIMD_FLOAT32_C(   211.91), EASYSIMD_FLOAT32_C(   312.95), EASYSIMD_FLOAT32_C(   537.08), EASYSIMD_FLOAT32_C(   532.14) },
      { EASYSIMD_FLOAT32_C(   254.45), EASYSIMD_FLOAT32_C(  -429.95), EASYSIMD_FLOAT32_C(   893.99), EASYSIMD_FLOAT32_C(   988.53) },
      { EASYSIMD_FLOAT32_C(  -162.22), EASYSIMD_FLOAT32_C(  6917.40), EASYSIMD_FLOAT32_C(459098.91), EASYSIMD_FLOAT32_C(   743.95) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_fmsub_ps(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_fmsub_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 c = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_fmsub_ps(a, k, b, c);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(156),
      { EASYSIMD_FLOAT32_C(   917.71), EASYSIMD_FLOAT32_C(    74.36), EASYSIMD_FLOAT32_C(  -951.19), EASYSIMD_FLOAT32_C(  -856.64) },
      { EASYSIMD_FLOAT32_C(   344.94), EASYSIMD_FLOAT32_C(   345.27), EASYSIMD_FLOAT32_C(  -815.67), EASYSIMD_FLOAT32_C(   154.88) },
      { EASYSIMD_FLOAT32_C(   515.87), EASYSIMD_FLOAT32_C(  -190.94), EASYSIMD_FLOAT32_C(  -775.96), EASYSIMD_FLOAT32_C(  -306.64) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(776633.06), EASYSIMD_FLOAT32_C(-132369.77) } },
    { UINT8_C(211),
      { EASYSIMD_FLOAT32_C(  -299.56), EASYSIMD_FLOAT32_C(  -428.64), EASYSIMD_FLOAT32_C(  -152.45), EASYSIMD_FLOAT32_C(   -15.13) },
      { EASYSIMD_FLOAT32_C(  -343.82), EASYSIMD_FLOAT32_C(  -484.97), EASYSIMD_FLOAT32_C(   189.45), EASYSIMD_FLOAT32_C(  -897.15) },
      { EASYSIMD_FLOAT32_C(   661.07), EASYSIMD_FLOAT32_C(  -705.49), EASYSIMD_FLOAT32_C(   153.63), EASYSIMD_FLOAT32_C(  -846.44) },
      { EASYSIMD_FLOAT32_C(102333.65), EASYSIMD_FLOAT32_C(208583.03), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(250),
      { EASYSIMD_FLOAT32_C(   609.45), EASYSIMD_FLOAT32_C(   278.71), EASYSIMD_FLOAT32_C(  -466.57), EASYSIMD_FLOAT32_C(   -54.72) },
      { EASYSIMD_FLOAT32_C(  -107.34), EASYSIMD_FLOAT32_C(  -548.86), EASYSIMD_FLOAT32_C(  -980.36), EASYSIMD_FLOAT32_C(   -58.53) },
      { EASYSIMD_FLOAT32_C(  -405.51), EASYSIMD_FLOAT32_C(   364.58), EASYSIMD_FLOAT32_C(  -713.26), EASYSIMD_FLOAT32_C(  -221.17) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-153337.34), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  3423.93) } },
    { UINT8_C( 20),
      { EASYSIMD_FLOAT32_C(   802.61), EASYSIMD_FLOAT32_C(   587.88), EASYSIMD_FLOAT32_C(  -256.50), EASYSIMD_FLOAT32_C(  -504.02) },
      { EASYSIMD_FLOAT32_C(  -353.28), EASYSIMD_FLOAT32_C(   443.94), EASYSIMD_FLOAT32_C(    67.33), EASYSIMD_FLOAT32_C(   494.27) },
      { EASYSIMD_FLOAT32_C(  -571.18), EASYSIMD_FLOAT32_C(   723.51), EASYSIMD_FLOAT32_C(  -990.70), EASYSIMD_FLOAT32_C(   618.26) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-16279.45), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 50),
      { EASYSIMD_FLOAT32_C(   670.36), EASYSIMD_FLOAT32_C(   912.78), EASYSIMD_FLOAT32_C(   -20.02), EASYSIMD_FLOAT32_C(   823.92) },
      { EASYSIMD_FLOAT32_C(   843.54), EASYSIMD_FLOAT32_C(  -410.57), EASYSIMD_FLOAT32_C(   102.63), EASYSIMD_FLOAT32_C(  -623.02) },
      { EASYSIMD_FLOAT32_C(   534.71), EASYSIMD_FLOAT32_C(   995.30), EASYSIMD_FLOAT32_C(  -171.89), EASYSIMD_FLOAT32_C(   554.35) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-375755.41), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(172),
      { EASYSIMD_FLOAT32_C(   422.61), EASYSIMD_FLOAT32_C(   -81.07), EASYSIMD_FLOAT32_C(   223.51), EASYSIMD_FLOAT32_C(  -798.56) },
      { EASYSIMD_FLOAT32_C(   438.40), EASYSIMD_FLOAT32_C(    26.12), EASYSIMD_FLOAT32_C(   789.32), EASYSIMD_FLOAT32_C(  -818.10) },
      { EASYSIMD_FLOAT32_C(   522.09), EASYSIMD_FLOAT32_C(  -563.96), EASYSIMD_FLOAT32_C(   625.85), EASYSIMD_FLOAT32_C(  -410.57) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(175795.06), EASYSIMD_FLOAT32_C(653712.50) } },
    { UINT8_C( 18),
      { EASYSIMD_FLOAT32_C(  -945.33), EASYSIMD_FLOAT32_C(  -687.07), EASYSIMD_FLOAT32_C(   939.60), EASYSIMD_FLOAT32_C(   672.93) },
      { EASYSIMD_FLOAT32_C(  -860.71), EASYSIMD_FLOAT32_C(   609.96), EASYSIMD_FLOAT32_C(   585.71), EASYSIMD_FLOAT32_C(   119.27) },
      { EASYSIMD_FLOAT32_C(   433.89), EASYSIMD_FLOAT32_C(   429.25), EASYSIMD_FLOAT32_C(   708.70), EASYSIMD_FLOAT32_C(  -463.48) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-419514.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(  5),
      { EASYSIMD_FLOAT32_C(   243.41), EASYSIMD_FLOAT32_C(  -468.18), EASYSIMD_FLOAT32_C(  -365.66), EASYSIMD_FLOAT32_C(  -202.24) },
      { EASYSIMD_FLOAT32_C(   468.58), EASYSIMD_FLOAT32_C(  -943.05), EASYSIMD_FLOAT32_C(   716.69), EASYSIMD_FLOAT32_C(  -307.91) },
      { EASYSIMD_FLOAT32_C(  -741.62), EASYSIMD_FLOAT32_C(   155.09), EASYSIMD_FLOAT32_C(   718.21), EASYSIMD_FLOAT32_C(  -952.30) },
      { EASYSIMD_FLOAT32_C(114798.68), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-262783.09), EASYSIMD_FLOAT32_C(     0.00) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_fmsub_ps(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_fmsub_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 c = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_fmsub_ps(k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fmsubadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd__mmask8 k;
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -789.30), EASYSIMD_FLOAT32_C(  -268.26), EASYSIMD_FLOAT32_C(   662.08), EASYSIMD_FLOAT32_C(  -683.08) },
      UINT8_C(178),
      { EASYSIMD_FLOAT32_C(   846.97), EASYSIMD_FLOAT32_C(   334.16), EASYSIMD_FLOAT32_C(  -716.79), EASYSIMD_FLOAT32_C(   632.41) },
      { EASYSIMD_FLOAT32_C(  -652.66), EASYSIMD_FLOAT32_C(   375.95), EASYSIMD_FLOAT32_C(   147.86), EASYSIMD_FLOAT32_C(    81.15) },
      { EASYSIMD_FLOAT32_C(  -789.30), EASYSIMD_FLOAT32_C(-90017.72), EASYSIMD_FLOAT32_C(   662.08), EASYSIMD_FLOAT32_C(  -683.08) } },
    { { EASYSIMD_FLOAT32_C(   891.95), EASYSIMD_FLOAT32_C(   144.97), EASYSIMD_FLOAT32_C(   771.39), EASYSIMD_FLOAT32_C(   171.38) },
      UINT8_C( 31),
      { EASYSIMD_FLOAT32_C(  -275.53), EASYSIMD_FLOAT32_C(   676.22), EASYSIMD_FLOAT32_C(   730.26), EASYSIMD_FLOAT32_C(  -416.89) },
      { EASYSIMD_FLOAT32_C(   951.11), EASYSIMD_FLOAT32_C(   187.65), EASYSIMD_FLOAT32_C(  -979.44), EASYSIMD_FLOAT32_C(   819.80) },
      { EASYSIMD_FLOAT32_C(-244807.88), EASYSIMD_FLOAT32_C( 97843.96), EASYSIMD_FLOAT32_C(562335.81), EASYSIMD_FLOAT32_C(-72266.41) } },
    { { EASYSIMD_FLOAT32_C(   935.18), EASYSIMD_FLOAT32_C(  -255.74), EASYSIMD_FLOAT32_C(  -999.24), EASYSIMD_FLOAT32_C(  -327.74) },
      UINT8_C( 32),
      { EASYSIMD_FLOAT32_C(  -788.54), EASYSIMD_FLOAT32_C(   404.00), EASYSIMD_FLOAT32_C(  -888.68), EASYSIMD_FLOAT32_C(  -471.63) },
      { EASYSIMD_FLOAT32_C(  -924.87), EASYSIMD_FLOAT32_C(   958.28), EASYSIMD_FLOAT32_C(   862.53), EASYSIMD_FLOAT32_C(  -641.66) },
      { EASYSIMD_FLOAT32_C(   935.18), EASYSIMD_FLOAT32_C(  -255.74), EASYSIMD_FLOAT32_C(  -999.24), EASYSIMD_FLOAT32_C(  -327.74) } },
    { { EASYSIMD_FLOAT32_C(   590.69), EASYSIMD_FLOAT32_C(  -790.13), EASYSIMD_FLOAT32_C(   734.29), EASYSIMD_FLOAT32_C(  -261.44) },
      UINT8_C( 91),
      { EASYSIMD_FLOAT32_C(   626.23), EASYSIMD_FLOAT32_C(   883.53), EASYSIMD_FLOAT32_C(    62.41), EASYSIMD_FLOAT32_C(  -202.39) },
      { EASYSIMD_FLOAT32_C(   106.83), EASYSIMD_FLOAT32_C(   786.88), EASYSIMD_FLOAT32_C(  -526.17), EASYSIMD_FLOAT32_C(  -162.91) },
      { EASYSIMD_FLOAT32_C(370014.62), EASYSIMD_FLOAT32_C(-698890.44), EASYSIMD_FLOAT32_C(   734.29), EASYSIMD_FLOAT32_C( 53075.75) } },
    { { EASYSIMD_FLOAT32_C(  -630.02), EASYSIMD_FLOAT32_C(  -575.06), EASYSIMD_FLOAT32_C(  -975.26), EASYSIMD_FLOAT32_C(  -609.46) },
      UINT8_C(185),
      { EASYSIMD_FLOAT32_C(   959.92), EASYSIMD_FLOAT32_C(   134.80), EASYSIMD_FLOAT32_C(  -754.50), EASYSIMD_FLOAT32_C(  -367.82) },
      { EASYSIMD_FLOAT32_C(   584.03), EASYSIMD_FLOAT32_C(  -543.05), EASYSIMD_FLOAT32_C(  -963.81), EASYSIMD_FLOAT32_C(   695.34) },
      { EASYSIMD_FLOAT32_C(-604184.81), EASYSIMD_FLOAT32_C(  -575.06), EASYSIMD_FLOAT32_C(  -975.26), EASYSIMD_FLOAT32_C(223476.25) } },
    { { EASYSIMD_FLOAT32_C(   -14.67), EASYSIMD_FLOAT32_C(  -888.68), EASYSIMD_FLOAT32_C(   653.63), EASYSIMD_FLOAT32_C(  -152.14) },
      UINT8_C( 87),
      { EASYSIMD_FLOAT32_C(   244.32), EASYSIMD_FLOAT32_C(    57.73), EASYSIMD_FLOAT32_C(  -796.06), EASYSIMD_FLOAT32_C(   982.88) },
      { EASYSIMD_FLOAT32_C(  -651.25), EASYSIMD_FLOAT32_C(   830.18), EASYSIMD_FLOAT32_C(   866.41), EASYSIMD_FLOAT32_C(   411.16) },
      { EASYSIMD_FLOAT32_C( -4235.42), EASYSIMD_FLOAT32_C(-52133.68), EASYSIMD_FLOAT32_C(-519462.28), EASYSIMD_FLOAT32_C(  -152.14) } },
    { { EASYSIMD_FLOAT32_C(  -372.21), EASYSIMD_FLOAT32_C(   -26.75), EASYSIMD_FLOAT32_C(   198.03), EASYSIMD_FLOAT32_C(   101.62) },
      UINT8_C( 19),
      { EASYSIMD_FLOAT32_C(   568.01), EASYSIMD_FLOAT32_C(   526.56), EASYSIMD_FLOAT32_C(   835.08), EASYSIMD_FLOAT32_C(   958.55) },
      { EASYSIMD_FLOAT32_C(   771.30), EASYSIMD_FLOAT32_C(   795.00), EASYSIMD_FLOAT32_C(    93.35), EASYSIMD_FLOAT32_C(  -983.20) },
      { EASYSIMD_FLOAT32_C(-210647.70), EASYSIMD_FLOAT32_C(-14880.48), EASYSIMD_FLOAT32_C(   198.03), EASYSIMD_FLOAT32_C(   101.62) } },
    { { EASYSIMD_FLOAT32_C(  -572.81), EASYSIMD_FLOAT32_C(  -322.62), EASYSIMD_FLOAT32_C(  -526.24), EASYSIMD_FLOAT32_C(  -536.62) },
      UINT8_C(139),
      { EASYSIMD_FLOAT32_C(   459.08), EASYSIMD_FLOAT32_C(  -425.31), EASYSIMD_FLOAT32_C(  -973.65), EASYSIMD_FLOAT32_C(  -693.06) },
      { EASYSIMD_FLOAT32_C(    44.35), EASYSIMD_FLOAT32_C(   270.67), EASYSIMD_FLOAT32_C(   364.67), EASYSIMD_FLOAT32_C(   248.29) },
      { EASYSIMD_FLOAT32_C(-262921.25), EASYSIMD_FLOAT32_C(136942.84), EASYSIMD_FLOAT32_C(  -526.24), EASYSIMD_FLOAT32_C(371661.56) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_fmsubadd_ps(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_fmsubadd_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 c = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_fmsubadd_ps(a, k, b, c);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fmsubadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd__mmask8 k;
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   253.55), EASYSIMD_FLOAT32_C(   713.42), EASYSIMD_FLOAT32_C(    78.47), EASYSIMD_FLOAT32_C(   119.97) },
      UINT8_C(168),
      { EASYSIMD_FLOAT32_C(   706.26), EASYSIMD_FLOAT32_C(  -906.79), EASYSIMD_FLOAT32_C(  -677.39), EASYSIMD_FLOAT32_C(  -192.12) },
      { EASYSIMD_FLOAT32_C(   903.55), EASYSIMD_FLOAT32_C(   890.62), EASYSIMD_FLOAT32_C(  -665.56), EASYSIMD_FLOAT32_C(   738.63) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-23787.27) } },
    { { EASYSIMD_FLOAT32_C(   849.17), EASYSIMD_FLOAT32_C(  -894.26), EASYSIMD_FLOAT32_C(   533.63), EASYSIMD_FLOAT32_C(   -57.48) },
      UINT8_C(228),
      { EASYSIMD_FLOAT32_C(   960.82), EASYSIMD_FLOAT32_C(   619.90), EASYSIMD_FLOAT32_C(  -403.71), EASYSIMD_FLOAT32_C(  -575.81) },
      { EASYSIMD_FLOAT32_C(   992.62), EASYSIMD_FLOAT32_C(  -944.62), EASYSIMD_FLOAT32_C(    -1.11), EASYSIMD_FLOAT32_C(  -981.03) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-215432.88), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -637.68), EASYSIMD_FLOAT32_C(  -956.77), EASYSIMD_FLOAT32_C(   289.64), EASYSIMD_FLOAT32_C(   726.99) },
      UINT8_C(204),
      { EASYSIMD_FLOAT32_C(  -456.81), EASYSIMD_FLOAT32_C(   440.41), EASYSIMD_FLOAT32_C(  -630.01), EASYSIMD_FLOAT32_C(   663.16) },
      { EASYSIMD_FLOAT32_C(  -435.01), EASYSIMD_FLOAT32_C(  -923.75), EASYSIMD_FLOAT32_C(   756.37), EASYSIMD_FLOAT32_C(  -112.40) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-181719.73), EASYSIMD_FLOAT32_C(482223.06) } },
    { { EASYSIMD_FLOAT32_C(  -115.87), EASYSIMD_FLOAT32_C(   659.92), EASYSIMD_FLOAT32_C(  -221.78), EASYSIMD_FLOAT32_C(   218.56) },
      UINT8_C(102),
      { EASYSIMD_FLOAT32_C(  -372.61), EASYSIMD_FLOAT32_C(   324.30), EASYSIMD_FLOAT32_C(   -67.82), EASYSIMD_FLOAT32_C(   569.91) },
      { EASYSIMD_FLOAT32_C(   446.84), EASYSIMD_FLOAT32_C(  -107.00), EASYSIMD_FLOAT32_C(   189.81), EASYSIMD_FLOAT32_C(  -956.87) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(214119.05), EASYSIMD_FLOAT32_C( 15230.93), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   317.19), EASYSIMD_FLOAT32_C(   182.43), EASYSIMD_FLOAT32_C(  -901.49), EASYSIMD_FLOAT32_C(  -683.92) },
      UINT8_C(112),
      { EASYSIMD_FLOAT32_C(  -539.16), EASYSIMD_FLOAT32_C(  -640.69), EASYSIMD_FLOAT32_C(  -508.96), EASYSIMD_FLOAT32_C(  -812.17) },
      { EASYSIMD_FLOAT32_C(   650.84), EASYSIMD_FLOAT32_C(    34.23), EASYSIMD_FLOAT32_C(   628.24), EASYSIMD_FLOAT32_C(  -979.17) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -302.61), EASYSIMD_FLOAT32_C(  -806.77), EASYSIMD_FLOAT32_C(  -902.92), EASYSIMD_FLOAT32_C(  -546.24) },
      UINT8_C(245),
      { EASYSIMD_FLOAT32_C(   -18.80), EASYSIMD_FLOAT32_C(  -886.32), EASYSIMD_FLOAT32_C(   859.04), EASYSIMD_FLOAT32_C(  -800.23) },
      { EASYSIMD_FLOAT32_C(   512.22), EASYSIMD_FLOAT32_C(  -513.56), EASYSIMD_FLOAT32_C(   524.07), EASYSIMD_FLOAT32_C(  -555.60) },
      { EASYSIMD_FLOAT32_C(  6201.29), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-775120.31), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -943.65), EASYSIMD_FLOAT32_C(   -29.09), EASYSIMD_FLOAT32_C(   337.40), EASYSIMD_FLOAT32_C(   246.16) },
      UINT8_C(228),
      { EASYSIMD_FLOAT32_C(  -345.41), EASYSIMD_FLOAT32_C(  -571.40), EASYSIMD_FLOAT32_C(   112.56), EASYSIMD_FLOAT32_C(   -29.33) },
      { EASYSIMD_FLOAT32_C(   630.00), EASYSIMD_FLOAT32_C(   573.39), EASYSIMD_FLOAT32_C(   329.98), EASYSIMD_FLOAT32_C(  -878.96) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 38307.72), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   761.22), EASYSIMD_FLOAT32_C(   -19.19), EASYSIMD_FLOAT32_C(   155.27), EASYSIMD_FLOAT32_C(   389.46) },
      UINT8_C(191),
      { EASYSIMD_FLOAT32_C(   852.66), EASYSIMD_FLOAT32_C(   582.69), EASYSIMD_FLOAT32_C(    98.72), EASYSIMD_FLOAT32_C(  -693.58) },
      { EASYSIMD_FLOAT32_C(  -336.48), EASYSIMD_FLOAT32_C(  -920.08), EASYSIMD_FLOAT32_C(  -579.90), EASYSIMD_FLOAT32_C(  -477.44) },
      { EASYSIMD_FLOAT32_C(648725.31), EASYSIMD_FLOAT32_C(-10261.74), EASYSIMD_FLOAT32_C( 14748.35), EASYSIMD_FLOAT32_C(-269644.22) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_fmsubadd_ps(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_fmsubadd_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 c = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_fmsubadd_ps(k, a, b, c);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask3_fmsubadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const uint8_t k;
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   857.69), EASYSIMD_FLOAT32_C(   158.56), EASYSIMD_FLOAT32_C(  -162.47), EASYSIMD_FLOAT32_C(    11.16),
        EASYSIMD_FLOAT32_C(   145.59), EASYSIMD_FLOAT32_C(   361.28), EASYSIMD_FLOAT32_C(   286.29), EASYSIMD_FLOAT32_C(  -654.18) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT32_C(   435.51), EASYSIMD_FLOAT32_C(   826.70), EASYSIMD_FLOAT32_C(    52.71), EASYSIMD_FLOAT32_C(  -340.62),
        EASYSIMD_FLOAT32_C(  -396.48), EASYSIMD_FLOAT32_C(   728.36), EASYSIMD_FLOAT32_C(  -629.24), EASYSIMD_FLOAT32_C(     3.49) },
      { EASYSIMD_FLOAT32_C(  -344.98), EASYSIMD_FLOAT32_C(     3.61), EASYSIMD_FLOAT32_C(   674.60), EASYSIMD_FLOAT32_C(  -808.62),
        EASYSIMD_FLOAT32_C(   236.12), EASYSIMD_FLOAT32_C(   662.38), EASYSIMD_FLOAT32_C(   309.88), EASYSIMD_FLOAT32_C(   -33.23) },
      { EASYSIMD_FLOAT32_C(373187.62), EASYSIMD_FLOAT32_C(     3.61), EASYSIMD_FLOAT32_C(   674.60), EASYSIMD_FLOAT32_C(  -808.62),
        EASYSIMD_FLOAT32_C(-57487.40), EASYSIMD_FLOAT32_C(   662.38), EASYSIMD_FLOAT32_C(-179835.25), EASYSIMD_FLOAT32_C(   -33.23) } },
    { { EASYSIMD_FLOAT32_C(   -77.34), EASYSIMD_FLOAT32_C(   160.32), EASYSIMD_FLOAT32_C(  -500.90), EASYSIMD_FLOAT32_C(  -772.31),
        EASYSIMD_FLOAT32_C(  -972.40), EASYSIMD_FLOAT32_C(  -852.56), EASYSIMD_FLOAT32_C(  -914.62), EASYSIMD_FLOAT32_C(   186.16) },
      UINT8_C(180),
      { EASYSIMD_FLOAT32_C(    96.54), EASYSIMD_FLOAT32_C(  -668.24), EASYSIMD_FLOAT32_C(  -653.76), EASYSIMD_FLOAT32_C(  -617.17),
        EASYSIMD_FLOAT32_C(  -322.42), EASYSIMD_FLOAT32_C(  -718.45), EASYSIMD_FLOAT32_C(   818.35), EASYSIMD_FLOAT32_C(  -495.72) },
      { EASYSIMD_FLOAT32_C(   334.26), EASYSIMD_FLOAT32_C(  -522.27), EASYSIMD_FLOAT32_C(   107.80), EASYSIMD_FLOAT32_C(    62.62),
        EASYSIMD_FLOAT32_C(  -151.51), EASYSIMD_FLOAT32_C(  -888.71), EASYSIMD_FLOAT32_C(   717.64), EASYSIMD_FLOAT32_C(   852.10) },
      { EASYSIMD_FLOAT32_C(   334.26), EASYSIMD_FLOAT32_C(  -522.27), EASYSIMD_FLOAT32_C(327576.19), EASYSIMD_FLOAT32_C(    62.62),
        EASYSIMD_FLOAT32_C(313369.72), EASYSIMD_FLOAT32_C(613410.44), EASYSIMD_FLOAT32_C(   717.64), EASYSIMD_FLOAT32_C(-93135.34) } },
    { { EASYSIMD_FLOAT32_C(   785.89), EASYSIMD_FLOAT32_C(   909.02), EASYSIMD_FLOAT32_C(    88.22), EASYSIMD_FLOAT32_C(   448.27),
        EASYSIMD_FLOAT32_C(   218.90), EASYSIMD_FLOAT32_C(  -945.00), EASYSIMD_FLOAT32_C(  -629.07), EASYSIMD_FLOAT32_C(  -620.78) },
      UINT8_C( 10),
      { EASYSIMD_FLOAT32_C(  -401.38), EASYSIMD_FLOAT32_C(  -593.18), EASYSIMD_FLOAT32_C(  -298.47), EASYSIMD_FLOAT32_C(  -316.00),
        EASYSIMD_FLOAT32_C(   592.99), EASYSIMD_FLOAT32_C(   686.50), EASYSIMD_FLOAT32_C(   780.55), EASYSIMD_FLOAT32_C(   924.74) },
      { EASYSIMD_FLOAT32_C(  -967.26), EASYSIMD_FLOAT32_C(  -836.62), EASYSIMD_FLOAT32_C(  -397.68), EASYSIMD_FLOAT32_C(  -685.71),
        EASYSIMD_FLOAT32_C(   981.73), EASYSIMD_FLOAT32_C(   106.60), EASYSIMD_FLOAT32_C(   648.55), EASYSIMD_FLOAT32_C(  -540.54) },
      { EASYSIMD_FLOAT32_C(  -967.26), EASYSIMD_FLOAT32_C(-538375.88), EASYSIMD_FLOAT32_C(  -397.68), EASYSIMD_FLOAT32_C(-140967.61),
        EASYSIMD_FLOAT32_C(   981.73), EASYSIMD_FLOAT32_C(   106.60), EASYSIMD_FLOAT32_C(   648.55), EASYSIMD_FLOAT32_C(  -540.54) } },
    { { EASYSIMD_FLOAT32_C(  -785.61), EASYSIMD_FLOAT32_C(  -288.82), EASYSIMD_FLOAT32_C(   307.95), EASYSIMD_FLOAT32_C(  -674.32),
        EASYSIMD_FLOAT32_C(  -571.18), EASYSIMD_FLOAT32_C(   160.06), EASYSIMD_FLOAT32_C(  -888.43), EASYSIMD_FLOAT32_C(  -662.16) },
      UINT8_C( 70),
      { EASYSIMD_FLOAT32_C(   559.84), EASYSIMD_FLOAT32_C(   556.74), EASYSIMD_FLOAT32_C(  -696.73), EASYSIMD_FLOAT32_C(   930.77),
        EASYSIMD_FLOAT32_C(   935.96), EASYSIMD_FLOAT32_C(  -142.63), EASYSIMD_FLOAT32_C(  -470.60), EASYSIMD_FLOAT32_C(  -657.21) },
      { EASYSIMD_FLOAT32_C(   558.90), EASYSIMD_FLOAT32_C(   213.40), EASYSIMD_FLOAT32_C(   935.77), EASYSIMD_FLOAT32_C(   245.39),
        EASYSIMD_FLOAT32_C(    -6.05), EASYSIMD_FLOAT32_C(   860.52), EASYSIMD_FLOAT32_C(   278.13), EASYSIMD_FLOAT32_C(   157.33) },
      { EASYSIMD_FLOAT32_C(   558.90), EASYSIMD_FLOAT32_C(-161011.05), EASYSIMD_FLOAT32_C(-213622.23), EASYSIMD_FLOAT32_C(   245.39),
        EASYSIMD_FLOAT32_C(    -6.05), EASYSIMD_FLOAT32_C(   860.52), EASYSIMD_FLOAT32_C(418373.28), EASYSIMD_FLOAT32_C(   157.33) } },
    { { EASYSIMD_FLOAT32_C(  -537.16), EASYSIMD_FLOAT32_C(   592.43), EASYSIMD_FLOAT32_C(   139.06), EASYSIMD_FLOAT32_C(   569.44),
        EASYSIMD_FLOAT32_C(   240.98), EASYSIMD_FLOAT32_C(   598.52), EASYSIMD_FLOAT32_C(   783.83), EASYSIMD_FLOAT32_C(   952.16) },
      UINT8_C(100),
      { EASYSIMD_FLOAT32_C(  -890.48), EASYSIMD_FLOAT32_C(  -619.02), EASYSIMD_FLOAT32_C(  -933.47), EASYSIMD_FLOAT32_C(  -778.91),
        EASYSIMD_FLOAT32_C(  -281.18), EASYSIMD_FLOAT32_C(  -685.20), EASYSIMD_FLOAT32_C(   780.93), EASYSIMD_FLOAT32_C(  -724.44) },
      { EASYSIMD_FLOAT32_C(  -381.92), EASYSIMD_FLOAT32_C(   711.71), EASYSIMD_FLOAT32_C(  -788.48), EASYSIMD_FLOAT32_C(   475.44),
        EASYSIMD_FLOAT32_C(  -758.90), EASYSIMD_FLOAT32_C(  -445.70), EASYSIMD_FLOAT32_C(    34.34), EASYSIMD_FLOAT32_C(   454.50) },
      { EASYSIMD_FLOAT32_C(  -381.92), EASYSIMD_FLOAT32_C(   711.71), EASYSIMD_FLOAT32_C(-130596.80), EASYSIMD_FLOAT32_C(   475.44),
        EASYSIMD_FLOAT32_C(  -758.90), EASYSIMD_FLOAT32_C(-409660.25), EASYSIMD_FLOAT32_C(612150.69), EASYSIMD_FLOAT32_C(   454.50) } },
    { { EASYSIMD_FLOAT32_C(  -509.92), EASYSIMD_FLOAT32_C(  -720.27), EASYSIMD_FLOAT32_C(  -551.55), EASYSIMD_FLOAT32_C(  -649.41),
        EASYSIMD_FLOAT32_C(   557.87), EASYSIMD_FLOAT32_C(   605.78), EASYSIMD_FLOAT32_C(  -186.57), EASYSIMD_FLOAT32_C(   150.29) },
      UINT8_C( 65),
      { EASYSIMD_FLOAT32_C(  -617.13), EASYSIMD_FLOAT32_C(  -608.73), EASYSIMD_FLOAT32_C(  -656.65), EASYSIMD_FLOAT32_C(  -833.30),
        EASYSIMD_FLOAT32_C(  -656.57), EASYSIMD_FLOAT32_C(   249.83), EASYSIMD_FLOAT32_C(  -723.79), EASYSIMD_FLOAT32_C(  -275.60) },
      { EASYSIMD_FLOAT32_C(   316.35), EASYSIMD_FLOAT32_C(  -502.70), EASYSIMD_FLOAT32_C(   443.22), EASYSIMD_FLOAT32_C(   631.15),
        EASYSIMD_FLOAT32_C(  -721.77), EASYSIMD_FLOAT32_C(   718.78), EASYSIMD_FLOAT32_C(  -750.77), EASYSIMD_FLOAT32_C(   989.94) },
      { EASYSIMD_FLOAT32_C(315003.28), EASYSIMD_FLOAT32_C(  -502.70), EASYSIMD_FLOAT32_C(   443.22), EASYSIMD_FLOAT32_C(   631.15),
        EASYSIMD_FLOAT32_C(  -721.77), EASYSIMD_FLOAT32_C(   718.78), EASYSIMD_FLOAT32_C(134286.73), EASYSIMD_FLOAT32_C(   989.94) } },
    { { EASYSIMD_FLOAT32_C(   930.29), EASYSIMD_FLOAT32_C(   724.67), EASYSIMD_FLOAT32_C(  -768.96), EASYSIMD_FLOAT32_C(  -515.40),
        EASYSIMD_FLOAT32_C(  -240.99), EASYSIMD_FLOAT32_C(   685.55), EASYSIMD_FLOAT32_C(   -25.33), EASYSIMD_FLOAT32_C(    38.74) },
      UINT8_C( 12),
      { EASYSIMD_FLOAT32_C(   325.26), EASYSIMD_FLOAT32_C(  -403.39), EASYSIMD_FLOAT32_C(   739.78), EASYSIMD_FLOAT32_C(  -861.30),
        EASYSIMD_FLOAT32_C(   746.90), EASYSIMD_FLOAT32_C(  -515.38), EASYSIMD_FLOAT32_C(  -478.44), EASYSIMD_FLOAT32_C(  -861.82) },
      { EASYSIMD_FLOAT32_C(  -172.03), EASYSIMD_FLOAT32_C(  -311.74), EASYSIMD_FLOAT32_C(  -518.39), EASYSIMD_FLOAT32_C(  -922.20),
        EASYSIMD_FLOAT32_C(   -35.52), EASYSIMD_FLOAT32_C(   206.01), EASYSIMD_FLOAT32_C(   394.15), EASYSIMD_FLOAT32_C(   461.78) },
      { EASYSIMD_FLOAT32_C(  -172.03), EASYSIMD_FLOAT32_C(  -311.74), EASYSIMD_FLOAT32_C(-569379.62), EASYSIMD_FLOAT32_C(444836.22),
        EASYSIMD_FLOAT32_C(   -35.52), EASYSIMD_FLOAT32_C(   206.01), EASYSIMD_FLOAT32_C(   394.15), EASYSIMD_FLOAT32_C(   461.78) } },
    { { EASYSIMD_FLOAT32_C(  -350.77), EASYSIMD_FLOAT32_C(    25.30), EASYSIMD_FLOAT32_C(   740.01), EASYSIMD_FLOAT32_C(  -632.00),
        EASYSIMD_FLOAT32_C(   274.53), EASYSIMD_FLOAT32_C(   729.96), EASYSIMD_FLOAT32_C(  -701.70), EASYSIMD_FLOAT32_C(    -0.80) },
      UINT8_C( 92),
      { EASYSIMD_FLOAT32_C(  -217.11), EASYSIMD_FLOAT32_C(   758.22), EASYSIMD_FLOAT32_C(   646.55), EASYSIMD_FLOAT32_C(   757.56),
        EASYSIMD_FLOAT32_C(  -203.04), EASYSIMD_FLOAT32_C(   780.55), EASYSIMD_FLOAT32_C(    82.83), EASYSIMD_FLOAT32_C(   393.57) },
      { EASYSIMD_FLOAT32_C(   520.33), EASYSIMD_FLOAT32_C(   221.53), EASYSIMD_FLOAT32_C(   140.48), EASYSIMD_FLOAT32_C(  -995.06),
        EASYSIMD_FLOAT32_C(   743.09), EASYSIMD_FLOAT32_C(   278.65), EASYSIMD_FLOAT32_C(  -167.09), EASYSIMD_FLOAT32_C(  -568.65) },
      { EASYSIMD_FLOAT32_C(   520.33), EASYSIMD_FLOAT32_C(   221.53), EASYSIMD_FLOAT32_C(478593.94), EASYSIMD_FLOAT32_C(-477782.84),
        EASYSIMD_FLOAT32_C(-54997.48), EASYSIMD_FLOAT32_C(   278.65), EASYSIMD_FLOAT32_C(-58288.90), EASYSIMD_FLOAT32_C(  -568.65) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask3_fmsubadd_ps(a, b, c, k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask3_fmsubadd_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 c = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask3_fmsubadd_ps(a, b, c, k);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_fmsubadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const uint8_t k;
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   760.26), EASYSIMD_FLOAT32_C(   -89.29), EASYSIMD_FLOAT32_C(   395.83), EASYSIMD_FLOAT32_C(   -33.73),
        EASYSIMD_FLOAT32_C(  -695.14), EASYSIMD_FLOAT32_C(  -142.39), EASYSIMD_FLOAT32_C(   615.50), EASYSIMD_FLOAT32_C(   330.16) },
      UINT8_C( 98),
      { EASYSIMD_FLOAT32_C(   983.50), EASYSIMD_FLOAT32_C(  -395.31), EASYSIMD_FLOAT32_C(  -672.42), EASYSIMD_FLOAT32_C(  -718.20),
        EASYSIMD_FLOAT32_C(   603.90), EASYSIMD_FLOAT32_C(  -711.42), EASYSIMD_FLOAT32_C(    64.69), EASYSIMD_FLOAT32_C(   362.11) },
      { EASYSIMD_FLOAT32_C(   935.13), EASYSIMD_FLOAT32_C(  -177.75), EASYSIMD_FLOAT32_C(  -840.93), EASYSIMD_FLOAT32_C(   715.68),
        EASYSIMD_FLOAT32_C(   905.08), EASYSIMD_FLOAT32_C(   552.64), EASYSIMD_FLOAT32_C(   236.00), EASYSIMD_FLOAT32_C(   126.61) },
      { EASYSIMD_FLOAT32_C(   760.26), EASYSIMD_FLOAT32_C( 35474.98), EASYSIMD_FLOAT32_C(   395.83), EASYSIMD_FLOAT32_C(   -33.73),
        EASYSIMD_FLOAT32_C(  -695.14), EASYSIMD_FLOAT32_C(100746.45), EASYSIMD_FLOAT32_C( 40052.70), EASYSIMD_FLOAT32_C(   330.16) } },
    { { EASYSIMD_FLOAT32_C(  -306.88), EASYSIMD_FLOAT32_C(   240.95), EASYSIMD_FLOAT32_C(  -130.30), EASYSIMD_FLOAT32_C(   971.77),
        EASYSIMD_FLOAT32_C(  -926.14), EASYSIMD_FLOAT32_C(   301.05), EASYSIMD_FLOAT32_C(   732.03), EASYSIMD_FLOAT32_C(   -15.43) },
      UINT8_C(250),
      { EASYSIMD_FLOAT32_C(  -301.70), EASYSIMD_FLOAT32_C(   289.43), EASYSIMD_FLOAT32_C(   554.50), EASYSIMD_FLOAT32_C(  -686.21),
        EASYSIMD_FLOAT32_C(  -380.41), EASYSIMD_FLOAT32_C(  -847.88), EASYSIMD_FLOAT32_C(  -702.70), EASYSIMD_FLOAT32_C(   224.28) },
      { EASYSIMD_FLOAT32_C(  -520.29), EASYSIMD_FLOAT32_C(  -420.91), EASYSIMD_FLOAT32_C(  -171.83), EASYSIMD_FLOAT32_C(  -231.71),
        EASYSIMD_FLOAT32_C(   643.78), EASYSIMD_FLOAT32_C(  -809.71), EASYSIMD_FLOAT32_C(  -296.58), EASYSIMD_FLOAT32_C(  -533.96) },
      { EASYSIMD_FLOAT32_C(  -306.88), EASYSIMD_FLOAT32_C( 70159.06), EASYSIMD_FLOAT32_C(  -130.30), EASYSIMD_FLOAT32_C(-666606.62),
        EASYSIMD_FLOAT32_C(352956.72), EASYSIMD_FLOAT32_C(-254444.56), EASYSIMD_FLOAT32_C(-514694.09), EASYSIMD_FLOAT32_C( -2926.68) } },
    { { EASYSIMD_FLOAT32_C(  -650.64), EASYSIMD_FLOAT32_C(  -580.90), EASYSIMD_FLOAT32_C(  -628.88), EASYSIMD_FLOAT32_C(   902.01),
        EASYSIMD_FLOAT32_C(   655.10), EASYSIMD_FLOAT32_C(   497.73), EASYSIMD_FLOAT32_C(  -404.87), EASYSIMD_FLOAT32_C(  -103.95) },
      UINT8_C( 18),
      { EASYSIMD_FLOAT32_C(  -433.10), EASYSIMD_FLOAT32_C(   -30.09), EASYSIMD_FLOAT32_C(   668.47), EASYSIMD_FLOAT32_C(  -701.07),
        EASYSIMD_FLOAT32_C(   954.48), EASYSIMD_FLOAT32_C(  -634.64), EASYSIMD_FLOAT32_C(    -2.77), EASYSIMD_FLOAT32_C(   243.91) },
      { EASYSIMD_FLOAT32_C(   919.85), EASYSIMD_FLOAT32_C(   311.02), EASYSIMD_FLOAT32_C(   863.49), EASYSIMD_FLOAT32_C(  -928.02),
        EASYSIMD_FLOAT32_C(   608.32), EASYSIMD_FLOAT32_C(    87.77), EASYSIMD_FLOAT32_C(  -448.31), EASYSIMD_FLOAT32_C(  -812.59) },
      { EASYSIMD_FLOAT32_C(  -650.64), EASYSIMD_FLOAT32_C( 17168.26), EASYSIMD_FLOAT32_C(  -628.88), EASYSIMD_FLOAT32_C(   902.01),
        EASYSIMD_FLOAT32_C(625888.12), EASYSIMD_FLOAT32_C(   497.73), EASYSIMD_FLOAT32_C(  -404.87), EASYSIMD_FLOAT32_C(  -103.95) } },
    { { EASYSIMD_FLOAT32_C(   915.94), EASYSIMD_FLOAT32_C(   319.98), EASYSIMD_FLOAT32_C(   831.19), EASYSIMD_FLOAT32_C(  -893.77),
        EASYSIMD_FLOAT32_C(  -976.60), EASYSIMD_FLOAT32_C(  -702.77), EASYSIMD_FLOAT32_C(  -544.41), EASYSIMD_FLOAT32_C(  -557.51) },
      UINT8_C( 60),
      { EASYSIMD_FLOAT32_C(  -642.40), EASYSIMD_FLOAT32_C(  -902.40), EASYSIMD_FLOAT32_C(  -833.93), EASYSIMD_FLOAT32_C(   -47.28),
        EASYSIMD_FLOAT32_C(    -6.35), EASYSIMD_FLOAT32_C(  -466.50), EASYSIMD_FLOAT32_C(   519.62), EASYSIMD_FLOAT32_C(   963.56) },
      { EASYSIMD_FLOAT32_C(  -798.03), EASYSIMD_FLOAT32_C(   818.55), EASYSIMD_FLOAT32_C(   918.04), EASYSIMD_FLOAT32_C(  -432.67),
        EASYSIMD_FLOAT32_C(  -184.22), EASYSIMD_FLOAT32_C(   161.94), EASYSIMD_FLOAT32_C(  -512.81), EASYSIMD_FLOAT32_C(  -873.19) },
      { EASYSIMD_FLOAT32_C(   915.94), EASYSIMD_FLOAT32_C(   319.98), EASYSIMD_FLOAT32_C(-692236.19), EASYSIMD_FLOAT32_C( 42690.12),
        EASYSIMD_FLOAT32_C(  6017.19), EASYSIMD_FLOAT32_C(327680.28), EASYSIMD_FLOAT32_C(  -544.41), EASYSIMD_FLOAT32_C(  -557.51) } },
    { { EASYSIMD_FLOAT32_C(    25.43), EASYSIMD_FLOAT32_C(  -440.84), EASYSIMD_FLOAT32_C(   735.13), EASYSIMD_FLOAT32_C(  -886.80),
        EASYSIMD_FLOAT32_C(   110.85), EASYSIMD_FLOAT32_C(   922.54), EASYSIMD_FLOAT32_C(  -970.85), EASYSIMD_FLOAT32_C(  -569.18) },
      UINT8_C(126),
      { EASYSIMD_FLOAT32_C(  -864.62), EASYSIMD_FLOAT32_C(  -545.78), EASYSIMD_FLOAT32_C(  -949.04), EASYSIMD_FLOAT32_C(  -409.03),
        EASYSIMD_FLOAT32_C(  -103.28), EASYSIMD_FLOAT32_C(  -280.69), EASYSIMD_FLOAT32_C(   -51.43), EASYSIMD_FLOAT32_C(    -5.69) },
      { EASYSIMD_FLOAT32_C(  -114.61), EASYSIMD_FLOAT32_C(   901.30), EASYSIMD_FLOAT32_C(   987.96), EASYSIMD_FLOAT32_C(   418.89),
        EASYSIMD_FLOAT32_C(   420.92), EASYSIMD_FLOAT32_C(   951.52), EASYSIMD_FLOAT32_C(   620.86), EASYSIMD_FLOAT32_C(   239.48) },
      { EASYSIMD_FLOAT32_C(    25.43), EASYSIMD_FLOAT32_C(239700.38), EASYSIMD_FLOAT32_C(-696679.81), EASYSIMD_FLOAT32_C(362308.94),
        EASYSIMD_FLOAT32_C(-11027.67), EASYSIMD_FLOAT32_C(-259899.27), EASYSIMD_FLOAT32_C( 50551.68), EASYSIMD_FLOAT32_C(  -569.18) } },
    { { EASYSIMD_FLOAT32_C(   869.56), EASYSIMD_FLOAT32_C(  -811.81), EASYSIMD_FLOAT32_C(  -944.74), EASYSIMD_FLOAT32_C(    31.50),
        EASYSIMD_FLOAT32_C(  -324.62), EASYSIMD_FLOAT32_C(  -817.93), EASYSIMD_FLOAT32_C(  -943.07), EASYSIMD_FLOAT32_C(   234.54) },
      UINT8_C( 85),
      { EASYSIMD_FLOAT32_C(  -829.86), EASYSIMD_FLOAT32_C(  -654.61), EASYSIMD_FLOAT32_C(   839.73), EASYSIMD_FLOAT32_C(  -800.71),
        EASYSIMD_FLOAT32_C(  -223.78), EASYSIMD_FLOAT32_C(   593.47), EASYSIMD_FLOAT32_C(  -665.33), EASYSIMD_FLOAT32_C(   230.44) },
      { EASYSIMD_FLOAT32_C(   644.43), EASYSIMD_FLOAT32_C(   -74.36), EASYSIMD_FLOAT32_C(  -872.85), EASYSIMD_FLOAT32_C(  -636.26),
        EASYSIMD_FLOAT32_C(   874.22), EASYSIMD_FLOAT32_C(   121.47), EASYSIMD_FLOAT32_C(   249.13), EASYSIMD_FLOAT32_C(   775.51) },
      { EASYSIMD_FLOAT32_C(-720968.62), EASYSIMD_FLOAT32_C(  -811.81), EASYSIMD_FLOAT32_C(-794199.38), EASYSIMD_FLOAT32_C(    31.50),
        EASYSIMD_FLOAT32_C( 73517.68), EASYSIMD_FLOAT32_C(  -817.93), EASYSIMD_FLOAT32_C(627701.94), EASYSIMD_FLOAT32_C(   234.54) } },
    { { EASYSIMD_FLOAT32_C(   109.43), EASYSIMD_FLOAT32_C(  -331.98), EASYSIMD_FLOAT32_C(   196.44), EASYSIMD_FLOAT32_C(    60.95),
        EASYSIMD_FLOAT32_C(  -711.12), EASYSIMD_FLOAT32_C(  -564.09), EASYSIMD_FLOAT32_C(   -69.50), EASYSIMD_FLOAT32_C(  -522.93) },
      UINT8_C(232),
      { EASYSIMD_FLOAT32_C(   962.00), EASYSIMD_FLOAT32_C(   152.45), EASYSIMD_FLOAT32_C(  -326.76), EASYSIMD_FLOAT32_C(  -981.07),
        EASYSIMD_FLOAT32_C(  -613.01), EASYSIMD_FLOAT32_C(  -409.57), EASYSIMD_FLOAT32_C(  -810.93), EASYSIMD_FLOAT32_C(  -267.62) },
      { EASYSIMD_FLOAT32_C(  -569.84), EASYSIMD_FLOAT32_C(  -611.64), EASYSIMD_FLOAT32_C(   508.60), EASYSIMD_FLOAT32_C(  -976.37),
        EASYSIMD_FLOAT32_C(  -276.97), EASYSIMD_FLOAT32_C(  -260.97), EASYSIMD_FLOAT32_C(   668.06), EASYSIMD_FLOAT32_C(   648.67) },
      { EASYSIMD_FLOAT32_C(   109.43), EASYSIMD_FLOAT32_C(  -331.98), EASYSIMD_FLOAT32_C(   196.44), EASYSIMD_FLOAT32_C(-58819.85),
        EASYSIMD_FLOAT32_C(  -711.12), EASYSIMD_FLOAT32_C(231295.33), EASYSIMD_FLOAT32_C( 57027.69), EASYSIMD_FLOAT32_C(139297.84) } },
    { { EASYSIMD_FLOAT32_C(  -133.81), EASYSIMD_FLOAT32_C(  -968.20), EASYSIMD_FLOAT32_C(   522.89), EASYSIMD_FLOAT32_C(   987.65),
        EASYSIMD_FLOAT32_C(   280.93), EASYSIMD_FLOAT32_C(   298.40), EASYSIMD_FLOAT32_C(    97.08), EASYSIMD_FLOAT32_C(   948.95) },
      UINT8_C(140),
      { EASYSIMD_FLOAT32_C(  -841.97), EASYSIMD_FLOAT32_C(  -762.17), EASYSIMD_FLOAT32_C(   -69.25), EASYSIMD_FLOAT32_C(    88.53),
        EASYSIMD_FLOAT32_C(  -285.10), EASYSIMD_FLOAT32_C(   421.92), EASYSIMD_FLOAT32_C(    50.53), EASYSIMD_FLOAT32_C(   867.35) },
      { EASYSIMD_FLOAT32_C(  -904.84), EASYSIMD_FLOAT32_C(    69.46), EASYSIMD_FLOAT32_C(  -745.66), EASYSIMD_FLOAT32_C(  -314.41),
        EASYSIMD_FLOAT32_C(   258.53), EASYSIMD_FLOAT32_C(   -13.27), EASYSIMD_FLOAT32_C(   115.75), EASYSIMD_FLOAT32_C(   646.89) },
      { EASYSIMD_FLOAT32_C(  -133.81), EASYSIMD_FLOAT32_C(  -968.20), EASYSIMD_FLOAT32_C(-36955.79), EASYSIMD_FLOAT32_C( 87751.06),
        EASYSIMD_FLOAT32_C(   280.93), EASYSIMD_FLOAT32_C(   298.40), EASYSIMD_FLOAT32_C(    97.08), EASYSIMD_FLOAT32_C(822424.88) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_fmsubadd_ps(a, k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_fmsubadd_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 c = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_fmsubadd_ps(a, k, b, c);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_fmsubadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const uint8_t k;
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -504.68), EASYSIMD_FLOAT32_C(   139.38), EASYSIMD_FLOAT32_C(  -630.08), EASYSIMD_FLOAT32_C(   234.35),
        EASYSIMD_FLOAT32_C(  -192.57), EASYSIMD_FLOAT32_C(  -981.41), EASYSIMD_FLOAT32_C(  -899.46), EASYSIMD_FLOAT32_C(  -160.77) },
      UINT8_C(223),
      { EASYSIMD_FLOAT32_C(  -911.81), EASYSIMD_FLOAT32_C(  -879.84), EASYSIMD_FLOAT32_C(  -160.12), EASYSIMD_FLOAT32_C(   185.27),
        EASYSIMD_FLOAT32_C(  -930.88), EASYSIMD_FLOAT32_C(   334.72), EASYSIMD_FLOAT32_C(   343.30), EASYSIMD_FLOAT32_C(  -693.05) },
      { EASYSIMD_FLOAT32_C(  -734.52), EASYSIMD_FLOAT32_C(  -568.17), EASYSIMD_FLOAT32_C(    21.85), EASYSIMD_FLOAT32_C(   687.40),
        EASYSIMD_FLOAT32_C(   482.36), EASYSIMD_FLOAT32_C(  -110.79), EASYSIMD_FLOAT32_C(   782.56), EASYSIMD_FLOAT32_C(  -448.18) },
      { EASYSIMD_FLOAT32_C(459437.72), EASYSIMD_FLOAT32_C(-122063.94), EASYSIMD_FLOAT32_C(100910.26), EASYSIMD_FLOAT32_C( 42730.63),
        EASYSIMD_FLOAT32_C(179741.92), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-308002.06), EASYSIMD_FLOAT32_C(111869.83) } },
    { { EASYSIMD_FLOAT32_C(   143.55), EASYSIMD_FLOAT32_C(  -531.86), EASYSIMD_FLOAT32_C(   810.35), EASYSIMD_FLOAT32_C(  -869.72),
        EASYSIMD_FLOAT32_C(   583.89), EASYSIMD_FLOAT32_C(   457.24), EASYSIMD_FLOAT32_C(  -374.40), EASYSIMD_FLOAT32_C(  -276.73) },
      UINT8_C(209),
      { EASYSIMD_FLOAT32_C(   859.95), EASYSIMD_FLOAT32_C(   530.70), EASYSIMD_FLOAT32_C(   845.75), EASYSIMD_FLOAT32_C(   960.49),
        EASYSIMD_FLOAT32_C(  -630.07), EASYSIMD_FLOAT32_C(   387.23), EASYSIMD_FLOAT32_C(  -951.32), EASYSIMD_FLOAT32_C(  -509.90) },
      { EASYSIMD_FLOAT32_C(  -772.89), EASYSIMD_FLOAT32_C(   233.96), EASYSIMD_FLOAT32_C(  -440.79), EASYSIMD_FLOAT32_C(   561.83),
        EASYSIMD_FLOAT32_C(  -422.74), EASYSIMD_FLOAT32_C(  -133.84), EASYSIMD_FLOAT32_C(   827.31), EASYSIMD_FLOAT32_C(     9.08) },
      { EASYSIMD_FLOAT32_C(122672.94), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-368314.34), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(357001.53), EASYSIMD_FLOAT32_C(141095.55) } },
    { { EASYSIMD_FLOAT32_C(   888.01), EASYSIMD_FLOAT32_C(   514.71), EASYSIMD_FLOAT32_C(  -508.56), EASYSIMD_FLOAT32_C(  -222.78),
        EASYSIMD_FLOAT32_C(   297.26), EASYSIMD_FLOAT32_C(    43.26), EASYSIMD_FLOAT32_C(   920.77), EASYSIMD_FLOAT32_C(   765.41) },
      UINT8_C(124),
      { EASYSIMD_FLOAT32_C(  -948.95), EASYSIMD_FLOAT32_C(   349.30), EASYSIMD_FLOAT32_C(  -689.16), EASYSIMD_FLOAT32_C(  -323.36),
        EASYSIMD_FLOAT32_C(  -927.43), EASYSIMD_FLOAT32_C(  -862.00), EASYSIMD_FLOAT32_C(  -463.40), EASYSIMD_FLOAT32_C(   603.27) },
      { EASYSIMD_FLOAT32_C(   983.75), EASYSIMD_FLOAT32_C(  -502.91), EASYSIMD_FLOAT32_C(   973.20), EASYSIMD_FLOAT32_C(   370.97),
        EASYSIMD_FLOAT32_C(  -454.23), EASYSIMD_FLOAT32_C(  -536.71), EASYSIMD_FLOAT32_C(   598.08), EASYSIMD_FLOAT32_C(   779.73) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(351452.38), EASYSIMD_FLOAT32_C( 71667.17),
        EASYSIMD_FLOAT32_C(-276142.06), EASYSIMD_FLOAT32_C(-36753.41), EASYSIMD_FLOAT32_C(-426086.72), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(    22.50), EASYSIMD_FLOAT32_C(   159.92), EASYSIMD_FLOAT32_C(  -643.01), EASYSIMD_FLOAT32_C(   888.67),
        EASYSIMD_FLOAT32_C(   -12.77), EASYSIMD_FLOAT32_C(   366.07), EASYSIMD_FLOAT32_C(   776.68), EASYSIMD_FLOAT32_C(  -498.07) },
      UINT8_C( 72),
      { EASYSIMD_FLOAT32_C(  -446.10), EASYSIMD_FLOAT32_C(   799.20), EASYSIMD_FLOAT32_C(   -99.24), EASYSIMD_FLOAT32_C(  -525.33),
        EASYSIMD_FLOAT32_C(   564.60), EASYSIMD_FLOAT32_C(   754.37), EASYSIMD_FLOAT32_C(  -474.28), EASYSIMD_FLOAT32_C(   -86.10) },
      { EASYSIMD_FLOAT32_C(  -934.79), EASYSIMD_FLOAT32_C(   202.36), EASYSIMD_FLOAT32_C(   -13.53), EASYSIMD_FLOAT32_C(  -796.79),
        EASYSIMD_FLOAT32_C(   738.96), EASYSIMD_FLOAT32_C(  -410.27), EASYSIMD_FLOAT32_C(  -813.04), EASYSIMD_FLOAT32_C(  -763.96) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-466048.25),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-369176.81), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -437.07), EASYSIMD_FLOAT32_C(   557.93), EASYSIMD_FLOAT32_C(  -218.18), EASYSIMD_FLOAT32_C(    26.22),
        EASYSIMD_FLOAT32_C(   156.01), EASYSIMD_FLOAT32_C(  -438.46), EASYSIMD_FLOAT32_C(  -951.27), EASYSIMD_FLOAT32_C(  -684.07) },
      UINT8_C(110),
      { EASYSIMD_FLOAT32_C(   937.39), EASYSIMD_FLOAT32_C(   303.15), EASYSIMD_FLOAT32_C(  -715.40), EASYSIMD_FLOAT32_C(   714.07),
        EASYSIMD_FLOAT32_C(   805.09), EASYSIMD_FLOAT32_C(  -857.89), EASYSIMD_FLOAT32_C(  -732.03), EASYSIMD_FLOAT32_C(   604.28) },
      { EASYSIMD_FLOAT32_C(    42.87), EASYSIMD_FLOAT32_C(  -257.35), EASYSIMD_FLOAT32_C(   168.89), EASYSIMD_FLOAT32_C(  -202.76),
        EASYSIMD_FLOAT32_C(   268.36), EASYSIMD_FLOAT32_C(  -917.21), EASYSIMD_FLOAT32_C(  -137.55), EASYSIMD_FLOAT32_C(  -529.28) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(169393.81), EASYSIMD_FLOAT32_C(156254.86), EASYSIMD_FLOAT32_C( 18925.68),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(377067.66), EASYSIMD_FLOAT32_C(696220.69), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(    69.26), EASYSIMD_FLOAT32_C(    65.66), EASYSIMD_FLOAT32_C(  -790.32), EASYSIMD_FLOAT32_C(   658.99),
        EASYSIMD_FLOAT32_C(   252.62), EASYSIMD_FLOAT32_C(  -554.28), EASYSIMD_FLOAT32_C(  -778.08), EASYSIMD_FLOAT32_C(  -189.45) },
      UINT8_C(176),
      { EASYSIMD_FLOAT32_C(   248.14), EASYSIMD_FLOAT32_C(   966.56), EASYSIMD_FLOAT32_C(   789.08), EASYSIMD_FLOAT32_C(   296.87),
        EASYSIMD_FLOAT32_C(  -717.51), EASYSIMD_FLOAT32_C(  -292.39), EASYSIMD_FLOAT32_C(   234.27), EASYSIMD_FLOAT32_C(   585.65) },
      { EASYSIMD_FLOAT32_C(    -7.79), EASYSIMD_FLOAT32_C(   -51.66), EASYSIMD_FLOAT32_C(   390.73), EASYSIMD_FLOAT32_C(   134.32),
        EASYSIMD_FLOAT32_C(   216.31), EASYSIMD_FLOAT32_C(    -4.98), EASYSIMD_FLOAT32_C(  -822.81), EASYSIMD_FLOAT32_C(   958.96) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-181041.06), EASYSIMD_FLOAT32_C(162070.94), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-111910.36) } },
    { { EASYSIMD_FLOAT32_C(  -836.09), EASYSIMD_FLOAT32_C(   -25.57), EASYSIMD_FLOAT32_C(   227.32), EASYSIMD_FLOAT32_C(  -753.30),
        EASYSIMD_FLOAT32_C(   836.89), EASYSIMD_FLOAT32_C(   698.04), EASYSIMD_FLOAT32_C(   315.95), EASYSIMD_FLOAT32_C(   -97.45) },
      UINT8_C(214),
      { EASYSIMD_FLOAT32_C(   -25.06), EASYSIMD_FLOAT32_C(  -844.83), EASYSIMD_FLOAT32_C(  -646.55), EASYSIMD_FLOAT32_C(   196.86),
        EASYSIMD_FLOAT32_C(   -34.27), EASYSIMD_FLOAT32_C(   580.98), EASYSIMD_FLOAT32_C(  -554.99), EASYSIMD_FLOAT32_C(   -67.71) },
      { EASYSIMD_FLOAT32_C(   370.07), EASYSIMD_FLOAT32_C(   741.88), EASYSIMD_FLOAT32_C(   214.78), EASYSIMD_FLOAT32_C(  -922.32),
        EASYSIMD_FLOAT32_C(   -23.85), EASYSIMD_FLOAT32_C(  -199.57), EASYSIMD_FLOAT32_C(    69.89), EASYSIMD_FLOAT32_C(   924.49) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 20860.42), EASYSIMD_FLOAT32_C(-146758.97), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-28704.07), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-175279.20), EASYSIMD_FLOAT32_C(  5673.85) } },
    { { EASYSIMD_FLOAT32_C(  -808.84), EASYSIMD_FLOAT32_C(  -795.78), EASYSIMD_FLOAT32_C(   140.80), EASYSIMD_FLOAT32_C(   186.18),
        EASYSIMD_FLOAT32_C(  -618.59), EASYSIMD_FLOAT32_C(    99.76), EASYSIMD_FLOAT32_C(   350.09), EASYSIMD_FLOAT32_C(   355.84) },
      UINT8_C(207),
      { EASYSIMD_FLOAT32_C(   596.78), EASYSIMD_FLOAT32_C(   192.73), EASYSIMD_FLOAT32_C(  -974.88), EASYSIMD_FLOAT32_C(   -87.27),
        EASYSIMD_FLOAT32_C(  -904.72), EASYSIMD_FLOAT32_C(   932.85), EASYSIMD_FLOAT32_C(   887.68), EASYSIMD_FLOAT32_C(  -749.54) },
      { EASYSIMD_FLOAT32_C(  -713.70), EASYSIMD_FLOAT32_C(    84.54), EASYSIMD_FLOAT32_C(   216.19), EASYSIMD_FLOAT32_C(   867.28),
        EASYSIMD_FLOAT32_C(   529.55), EASYSIMD_FLOAT32_C(  -851.53), EASYSIMD_FLOAT32_C(   237.35), EASYSIMD_FLOAT32_C(   271.43) },
      { EASYSIMD_FLOAT32_C(-483413.25), EASYSIMD_FLOAT32_C(-153455.23), EASYSIMD_FLOAT32_C(-137046.92), EASYSIMD_FLOAT32_C(-17115.21),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(311005.22), EASYSIMD_FLOAT32_C(-266987.75) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_fmsubadd_ps(k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_fmsubadd_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 c = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_fmsubadd_ps(k, a, b, c);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask3_fmsubadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const uint8_t k;
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   363.26), EASYSIMD_FLOAT64_C(   315.03), EASYSIMD_FLOAT64_C(  -752.43), EASYSIMD_FLOAT64_C(  -836.32) },
      UINT8_C(  4),
      { EASYSIMD_FLOAT64_C(  -827.94), EASYSIMD_FLOAT64_C(  -645.15), EASYSIMD_FLOAT64_C(  -410.86), EASYSIMD_FLOAT64_C(   312.86) },
      { EASYSIMD_FLOAT64_C(   541.03), EASYSIMD_FLOAT64_C(   -29.45), EASYSIMD_FLOAT64_C(  -587.38), EASYSIMD_FLOAT64_C(  -108.89) },
      { EASYSIMD_FLOAT64_C(   541.03), EASYSIMD_FLOAT64_C(   -29.45), EASYSIMD_FLOAT64_C(308556.01), EASYSIMD_FLOAT64_C(  -108.89) } },
    { { EASYSIMD_FLOAT64_C(  -673.61), EASYSIMD_FLOAT64_C(  -260.30), EASYSIMD_FLOAT64_C(  -512.11), EASYSIMD_FLOAT64_C(   519.12) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT64_C(   400.63), EASYSIMD_FLOAT64_C(   614.41), EASYSIMD_FLOAT64_C(  -302.33), EASYSIMD_FLOAT64_C(   288.31) },
      { EASYSIMD_FLOAT64_C(   864.87), EASYSIMD_FLOAT64_C(   -16.04), EASYSIMD_FLOAT64_C(  -627.15), EASYSIMD_FLOAT64_C(    81.05) },
      { EASYSIMD_FLOAT64_C(-269003.50), EASYSIMD_FLOAT64_C(   -16.04), EASYSIMD_FLOAT64_C(  -627.15), EASYSIMD_FLOAT64_C(    81.05) } },
    { { EASYSIMD_FLOAT64_C(  -148.76), EASYSIMD_FLOAT64_C(   902.40), EASYSIMD_FLOAT64_C(   229.53), EASYSIMD_FLOAT64_C(  -911.41) },
      UINT8_C( 48),
      { EASYSIMD_FLOAT64_C(  -407.22), EASYSIMD_FLOAT64_C(   403.62), EASYSIMD_FLOAT64_C(   421.40), EASYSIMD_FLOAT64_C(  -243.53) },
      { EASYSIMD_FLOAT64_C(   788.54), EASYSIMD_FLOAT64_C(   593.45), EASYSIMD_FLOAT64_C(   111.31), EASYSIMD_FLOAT64_C(  -622.32) },
      { EASYSIMD_FLOAT64_C(   788.54), EASYSIMD_FLOAT64_C(   593.45), EASYSIMD_FLOAT64_C(   111.31), EASYSIMD_FLOAT64_C(  -622.32) } },
    { { EASYSIMD_FLOAT64_C(   -93.69), EASYSIMD_FLOAT64_C(  -347.66), EASYSIMD_FLOAT64_C(   348.23), EASYSIMD_FLOAT64_C(   318.93) },
      UINT8_C( 13),
      { EASYSIMD_FLOAT64_C(   674.62), EASYSIMD_FLOAT64_C(  -941.38), EASYSIMD_FLOAT64_C(  -968.66), EASYSIMD_FLOAT64_C(   193.74) },
      { EASYSIMD_FLOAT64_C(  -176.56), EASYSIMD_FLOAT64_C(   431.97), EASYSIMD_FLOAT64_C(  -191.85), EASYSIMD_FLOAT64_C(   521.11) },
      { EASYSIMD_FLOAT64_C(-63381.71), EASYSIMD_FLOAT64_C(   431.97), EASYSIMD_FLOAT64_C(-337508.32), EASYSIMD_FLOAT64_C( 61268.39) } },
    { { EASYSIMD_FLOAT64_C(  -279.72), EASYSIMD_FLOAT64_C(  -326.99), EASYSIMD_FLOAT64_C(  -494.93), EASYSIMD_FLOAT64_C(    93.13) },
      UINT8_C( 23),
      { EASYSIMD_FLOAT64_C(   356.32), EASYSIMD_FLOAT64_C(    -4.48), EASYSIMD_FLOAT64_C(   -16.41), EASYSIMD_FLOAT64_C(   444.91) },
      { EASYSIMD_FLOAT64_C(  -830.65), EASYSIMD_FLOAT64_C(   576.37), EASYSIMD_FLOAT64_C(  -151.47), EASYSIMD_FLOAT64_C(   590.74) },
      { EASYSIMD_FLOAT64_C(-100500.48), EASYSIMD_FLOAT64_C(   888.55), EASYSIMD_FLOAT64_C(  7970.33), EASYSIMD_FLOAT64_C(   590.74) } },
    { { EASYSIMD_FLOAT64_C(  -667.16), EASYSIMD_FLOAT64_C(  -362.93), EASYSIMD_FLOAT64_C(   184.20), EASYSIMD_FLOAT64_C(   444.15) },
      UINT8_C(212),
      { EASYSIMD_FLOAT64_C(  -909.49), EASYSIMD_FLOAT64_C(  -903.51), EASYSIMD_FLOAT64_C(  -637.02), EASYSIMD_FLOAT64_C(   409.43) },
      { EASYSIMD_FLOAT64_C(   639.94), EASYSIMD_FLOAT64_C(  -962.39), EASYSIMD_FLOAT64_C(   468.06), EASYSIMD_FLOAT64_C(   671.28) },
      { EASYSIMD_FLOAT64_C(   639.94), EASYSIMD_FLOAT64_C(  -962.39), EASYSIMD_FLOAT64_C(-116871.02), EASYSIMD_FLOAT64_C(   671.28) } },
    { { EASYSIMD_FLOAT64_C(   231.35), EASYSIMD_FLOAT64_C(  -708.50), EASYSIMD_FLOAT64_C(   103.25), EASYSIMD_FLOAT64_C(  -960.50) },
      UINT8_C( 66),
      { EASYSIMD_FLOAT64_C(   823.53), EASYSIMD_FLOAT64_C(  -287.49), EASYSIMD_FLOAT64_C(  -682.32), EASYSIMD_FLOAT64_C(   -83.34) },
      { EASYSIMD_FLOAT64_C(  -533.42), EASYSIMD_FLOAT64_C(   674.00), EASYSIMD_FLOAT64_C(   912.19), EASYSIMD_FLOAT64_C(   450.17) },
      { EASYSIMD_FLOAT64_C(  -533.42), EASYSIMD_FLOAT64_C(203012.67), EASYSIMD_FLOAT64_C(   912.19), EASYSIMD_FLOAT64_C(   450.17) } },
    { { EASYSIMD_FLOAT64_C(   118.91), EASYSIMD_FLOAT64_C(  -918.47), EASYSIMD_FLOAT64_C(    26.54), EASYSIMD_FLOAT64_C(   967.43) },
      UINT8_C(178),
      { EASYSIMD_FLOAT64_C(   359.38), EASYSIMD_FLOAT64_C(  -395.49), EASYSIMD_FLOAT64_C(  -143.52), EASYSIMD_FLOAT64_C(  -196.48) },
      { EASYSIMD_FLOAT64_C(   619.26), EASYSIMD_FLOAT64_C(   -53.02), EASYSIMD_FLOAT64_C(   -99.99), EASYSIMD_FLOAT64_C(   982.25) },
      { EASYSIMD_FLOAT64_C(   619.26), EASYSIMD_FLOAT64_C(363298.72), EASYSIMD_FLOAT64_C(   -99.99), EASYSIMD_FLOAT64_C(   982.25) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask3_fmsubadd_pd(a, b, c, k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask3_fmsubadd_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d c = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask3_fmsubadd_pd(a, b, c, k);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_fmsubadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const uint8_t k;
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -643.58), EASYSIMD_FLOAT64_C(  -460.06), EASYSIMD_FLOAT64_C(  -980.15), EASYSIMD_FLOAT64_C(   824.47) },
      UINT8_C( 11),
      { EASYSIMD_FLOAT64_C(   251.20), EASYSIMD_FLOAT64_C(  -884.03), EASYSIMD_FLOAT64_C(   314.48), EASYSIMD_FLOAT64_C(   290.70) },
      { EASYSIMD_FLOAT64_C(   928.58), EASYSIMD_FLOAT64_C(   138.01), EASYSIMD_FLOAT64_C(  -996.79), EASYSIMD_FLOAT64_C(  -753.74) },
      { EASYSIMD_FLOAT64_C(-160738.72), EASYSIMD_FLOAT64_C(406568.83), EASYSIMD_FLOAT64_C(  -980.15), EASYSIMD_FLOAT64_C(240427.17) } },
    { { EASYSIMD_FLOAT64_C(  -945.33), EASYSIMD_FLOAT64_C(  -530.22), EASYSIMD_FLOAT64_C(   920.26), EASYSIMD_FLOAT64_C(   966.86) },
      UINT8_C(142),
      { EASYSIMD_FLOAT64_C(    39.16), EASYSIMD_FLOAT64_C(  -951.61), EASYSIMD_FLOAT64_C(   -53.51), EASYSIMD_FLOAT64_C(     6.60) },
      { EASYSIMD_FLOAT64_C(   720.67), EASYSIMD_FLOAT64_C(  -694.14), EASYSIMD_FLOAT64_C(   611.10), EASYSIMD_FLOAT64_C(  -422.85) },
      { EASYSIMD_FLOAT64_C(  -945.33), EASYSIMD_FLOAT64_C(505256.79), EASYSIMD_FLOAT64_C(-48632.01), EASYSIMD_FLOAT64_C(  6804.13) } },
    { { EASYSIMD_FLOAT64_C(   109.39), EASYSIMD_FLOAT64_C(   230.36), EASYSIMD_FLOAT64_C(   524.13), EASYSIMD_FLOAT64_C(  -990.60) },
      UINT8_C(201),
      { EASYSIMD_FLOAT64_C(   880.55), EASYSIMD_FLOAT64_C(  -450.66), EASYSIMD_FLOAT64_C(   232.46), EASYSIMD_FLOAT64_C(   705.02) },
      { EASYSIMD_FLOAT64_C(  -239.44), EASYSIMD_FLOAT64_C(  -516.34), EASYSIMD_FLOAT64_C(   820.99), EASYSIMD_FLOAT64_C(  -924.96) },
      { EASYSIMD_FLOAT64_C( 96083.92), EASYSIMD_FLOAT64_C(   230.36), EASYSIMD_FLOAT64_C(   524.13), EASYSIMD_FLOAT64_C(-697467.85) } },
    { { EASYSIMD_FLOAT64_C(   774.36), EASYSIMD_FLOAT64_C(   749.57), EASYSIMD_FLOAT64_C(   213.05), EASYSIMD_FLOAT64_C(   777.56) },
      UINT8_C(200),
      { EASYSIMD_FLOAT64_C(   267.72), EASYSIMD_FLOAT64_C(  -752.65), EASYSIMD_FLOAT64_C(   916.09), EASYSIMD_FLOAT64_C(   234.58) },
      { EASYSIMD_FLOAT64_C(  -832.70), EASYSIMD_FLOAT64_C(   -44.75), EASYSIMD_FLOAT64_C(   282.97), EASYSIMD_FLOAT64_C(   113.78) },
      { EASYSIMD_FLOAT64_C(   774.36), EASYSIMD_FLOAT64_C(   749.57), EASYSIMD_FLOAT64_C(   213.05), EASYSIMD_FLOAT64_C(182286.24) } },
    { { EASYSIMD_FLOAT64_C(   961.85), EASYSIMD_FLOAT64_C(     3.64), EASYSIMD_FLOAT64_C(   419.65), EASYSIMD_FLOAT64_C(   572.95) },
      UINT8_C(167),
      { EASYSIMD_FLOAT64_C(  -470.97), EASYSIMD_FLOAT64_C(  -196.69), EASYSIMD_FLOAT64_C(   104.92), EASYSIMD_FLOAT64_C(  -461.57) },
      { EASYSIMD_FLOAT64_C(  -984.08), EASYSIMD_FLOAT64_C(   -14.53), EASYSIMD_FLOAT64_C(    87.77), EASYSIMD_FLOAT64_C(   248.38) },
      { EASYSIMD_FLOAT64_C(-453986.57), EASYSIMD_FLOAT64_C(  -701.42), EASYSIMD_FLOAT64_C( 44117.45), EASYSIMD_FLOAT64_C(   572.95) } },
    { { EASYSIMD_FLOAT64_C(  -309.51), EASYSIMD_FLOAT64_C(   848.33), EASYSIMD_FLOAT64_C(   732.04), EASYSIMD_FLOAT64_C(  -488.52) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT64_C(   506.40), EASYSIMD_FLOAT64_C(  -738.95), EASYSIMD_FLOAT64_C(   136.43), EASYSIMD_FLOAT64_C(   283.96) },
      { EASYSIMD_FLOAT64_C(  -743.11), EASYSIMD_FLOAT64_C(  -595.85), EASYSIMD_FLOAT64_C(   531.31), EASYSIMD_FLOAT64_C(  -827.03) },
      { EASYSIMD_FLOAT64_C(-157478.97), EASYSIMD_FLOAT64_C(   848.33), EASYSIMD_FLOAT64_C(   732.04), EASYSIMD_FLOAT64_C(  -488.52) } },
    { { EASYSIMD_FLOAT64_C(   638.73), EASYSIMD_FLOAT64_C(   698.61), EASYSIMD_FLOAT64_C(   128.22), EASYSIMD_FLOAT64_C(   -78.29) },
      UINT8_C( 79),
      { EASYSIMD_FLOAT64_C(    90.07), EASYSIMD_FLOAT64_C(   925.35), EASYSIMD_FLOAT64_C(  -767.96), EASYSIMD_FLOAT64_C(  -336.98) },
      { EASYSIMD_FLOAT64_C(   506.14), EASYSIMD_FLOAT64_C(  -238.93), EASYSIMD_FLOAT64_C(   466.33), EASYSIMD_FLOAT64_C(  -388.94) },
      { EASYSIMD_FLOAT64_C( 58036.55), EASYSIMD_FLOAT64_C(646697.69), EASYSIMD_FLOAT64_C(-98001.50), EASYSIMD_FLOAT64_C( 26771.10) } },
    { { EASYSIMD_FLOAT64_C(   299.50), EASYSIMD_FLOAT64_C(   482.25), EASYSIMD_FLOAT64_C(   596.52), EASYSIMD_FLOAT64_C(  -612.73) },
      UINT8_C( 43),
      { EASYSIMD_FLOAT64_C(  -712.99), EASYSIMD_FLOAT64_C(  -764.40), EASYSIMD_FLOAT64_C(  -537.33), EASYSIMD_FLOAT64_C(  -201.51) },
      { EASYSIMD_FLOAT64_C(  -841.02), EASYSIMD_FLOAT64_C(   969.07), EASYSIMD_FLOAT64_C(    59.55), EASYSIMD_FLOAT64_C(   295.41) },
      { EASYSIMD_FLOAT64_C(-214381.52), EASYSIMD_FLOAT64_C(-369600.97), EASYSIMD_FLOAT64_C(   596.52), EASYSIMD_FLOAT64_C(123175.81) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_fmsubadd_pd(a, k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_fmsubadd_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d c = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_fmsubadd_pd(a, k, b, c);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_fmsubadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const uint8_t k;
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   253.03), EASYSIMD_FLOAT64_C(   316.43), EASYSIMD_FLOAT64_C(   699.56), EASYSIMD_FLOAT64_C(  -215.66) },
      UINT8_C( 45),
      { EASYSIMD_FLOAT64_C(   338.29), EASYSIMD_FLOAT64_C(  -517.05), EASYSIMD_FLOAT64_C(  -382.37), EASYSIMD_FLOAT64_C(  -740.01) },
      { EASYSIMD_FLOAT64_C(   295.34), EASYSIMD_FLOAT64_C(   707.70), EASYSIMD_FLOAT64_C(  -814.66), EASYSIMD_FLOAT64_C(   527.38) },
      { EASYSIMD_FLOAT64_C( 85892.86), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-268305.42), EASYSIMD_FLOAT64_C(159063.18) } },
    { { EASYSIMD_FLOAT64_C(  -629.28), EASYSIMD_FLOAT64_C(   691.48), EASYSIMD_FLOAT64_C(  -711.55), EASYSIMD_FLOAT64_C(   837.04) },
      UINT8_C(203),
      { EASYSIMD_FLOAT64_C(   587.95), EASYSIMD_FLOAT64_C(   319.29), EASYSIMD_FLOAT64_C(   899.07), EASYSIMD_FLOAT64_C(   975.22) },
      { EASYSIMD_FLOAT64_C(  -950.08), EASYSIMD_FLOAT64_C(  -813.92), EASYSIMD_FLOAT64_C(  -789.17), EASYSIMD_FLOAT64_C(  -487.41) },
      { EASYSIMD_FLOAT64_C(-370935.26), EASYSIMD_FLOAT64_C(221596.57), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(816785.56) } },
    { { EASYSIMD_FLOAT64_C(   -15.43), EASYSIMD_FLOAT64_C(  -630.19), EASYSIMD_FLOAT64_C(  -518.34), EASYSIMD_FLOAT64_C(  -955.88) },
      UINT8_C( 42),
      { EASYSIMD_FLOAT64_C(   734.69), EASYSIMD_FLOAT64_C(   360.55), EASYSIMD_FLOAT64_C(   364.77), EASYSIMD_FLOAT64_C(  -480.96) },
      { EASYSIMD_FLOAT64_C(  -150.04), EASYSIMD_FLOAT64_C(  -296.94), EASYSIMD_FLOAT64_C(     1.99), EASYSIMD_FLOAT64_C(   467.59) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-226918.06), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(459272.45) } },
    { { EASYSIMD_FLOAT64_C(   -36.95), EASYSIMD_FLOAT64_C(  -702.67), EASYSIMD_FLOAT64_C(   175.28), EASYSIMD_FLOAT64_C(   148.39) },
      UINT8_C(142),
      { EASYSIMD_FLOAT64_C(   546.00), EASYSIMD_FLOAT64_C(  -160.12), EASYSIMD_FLOAT64_C(  -886.84), EASYSIMD_FLOAT64_C(   383.05) },
      { EASYSIMD_FLOAT64_C(   142.42), EASYSIMD_FLOAT64_C(   701.11), EASYSIMD_FLOAT64_C(  -297.66), EASYSIMD_FLOAT64_C(    41.49) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(111810.41), EASYSIMD_FLOAT64_C(-155742.98), EASYSIMD_FLOAT64_C( 56799.30) } },
    { { EASYSIMD_FLOAT64_C(   676.34), EASYSIMD_FLOAT64_C(  -247.74), EASYSIMD_FLOAT64_C(   227.57), EASYSIMD_FLOAT64_C(   887.17) },
      UINT8_C( 41),
      { EASYSIMD_FLOAT64_C(  -787.86), EASYSIMD_FLOAT64_C(  -743.02), EASYSIMD_FLOAT64_C(   746.52), EASYSIMD_FLOAT64_C(  -743.74) },
      { EASYSIMD_FLOAT64_C(   922.19), EASYSIMD_FLOAT64_C(   481.22), EASYSIMD_FLOAT64_C(   616.81), EASYSIMD_FLOAT64_C(   286.96) },
      { EASYSIMD_FLOAT64_C(-531939.04), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-660110.78) } },
    { { EASYSIMD_FLOAT64_C(  -999.75), EASYSIMD_FLOAT64_C(  -533.23), EASYSIMD_FLOAT64_C(   990.01), EASYSIMD_FLOAT64_C(     2.24) },
      UINT8_C( 51),
      { EASYSIMD_FLOAT64_C(   -46.94), EASYSIMD_FLOAT64_C(   299.56), EASYSIMD_FLOAT64_C(   109.64), EASYSIMD_FLOAT64_C(  -898.54) },
      { EASYSIMD_FLOAT64_C(   124.27), EASYSIMD_FLOAT64_C(  -344.36), EASYSIMD_FLOAT64_C(   -58.67), EASYSIMD_FLOAT64_C(   237.43) },
      { EASYSIMD_FLOAT64_C( 47052.53), EASYSIMD_FLOAT64_C(-159390.02), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -961.31), EASYSIMD_FLOAT64_C(  -916.24), EASYSIMD_FLOAT64_C(   -61.45), EASYSIMD_FLOAT64_C(  -258.97) },
      UINT8_C(138),
      { EASYSIMD_FLOAT64_C(  -385.11), EASYSIMD_FLOAT64_C(   493.29), EASYSIMD_FLOAT64_C(  -647.19), EASYSIMD_FLOAT64_C(  -497.95) },
      { EASYSIMD_FLOAT64_C(  -241.85), EASYSIMD_FLOAT64_C(  -435.05), EASYSIMD_FLOAT64_C(  -240.97), EASYSIMD_FLOAT64_C(  -495.33) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-451536.98), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(129449.44) } },
    { { EASYSIMD_FLOAT64_C(  -178.79), EASYSIMD_FLOAT64_C(  -318.78), EASYSIMD_FLOAT64_C(   985.88), EASYSIMD_FLOAT64_C(  -561.97) },
      UINT8_C(197),
      { EASYSIMD_FLOAT64_C(   986.13), EASYSIMD_FLOAT64_C(   -95.20), EASYSIMD_FLOAT64_C(   958.19), EASYSIMD_FLOAT64_C(   -11.63) },
      { EASYSIMD_FLOAT64_C(  -160.85), EASYSIMD_FLOAT64_C(   -88.74), EASYSIMD_FLOAT64_C(  -712.06), EASYSIMD_FLOAT64_C(   948.79) },
      { EASYSIMD_FLOAT64_C(-176471.03), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(943948.30), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_fmsubadd_pd(k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_fmsubadd_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d c = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_fmsubadd_pd(k, a, b, c);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_fmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 c[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    39.18), EASYSIMD_FLOAT32_C(    72.95), EASYSIMD_FLOAT32_C(     2.39), EASYSIMD_FLOAT32_C(   -99.28),
        EASYSIMD_FLOAT32_C(   -27.76), EASYSIMD_FLOAT32_C(    78.92), EASYSIMD_FLOAT32_C(    97.46), EASYSIMD_FLOAT32_C(   -75.13),
        EASYSIMD_FLOAT32_C(   -78.86), EASYSIMD_FLOAT32_C(    62.73), EASYSIMD_FLOAT32_C(    46.15), EASYSIMD_FLOAT32_C(   -15.69),
        EASYSIMD_FLOAT32_C(   -66.26), EASYSIMD_FLOAT32_C(    -2.97), EASYSIMD_FLOAT32_C(    64.58), EASYSIMD_FLOAT32_C(   -96.46) },
      { EASYSIMD_FLOAT32_C(    87.93), EASYSIMD_FLOAT32_C(   -57.28), EASYSIMD_FLOAT32_C(   -38.98), EASYSIMD_FLOAT32_C(    16.76),
        EASYSIMD_FLOAT32_C(    -7.19), EASYSIMD_FLOAT32_C(   -49.24), EASYSIMD_FLOAT32_C(   -45.18), EASYSIMD_FLOAT32_C(    16.45),
        EASYSIMD_FLOAT32_C(   -68.84), EASYSIMD_FLOAT32_C(   -51.68), EASYSIMD_FLOAT32_C(    46.34), EASYSIMD_FLOAT32_C(    24.50),
        EASYSIMD_FLOAT32_C(   -28.63), EASYSIMD_FLOAT32_C(   -74.05), EASYSIMD_FLOAT32_C(   -21.93), EASYSIMD_FLOAT32_C(   -89.45) },
      { EASYSIMD_FLOAT32_C(    98.91), EASYSIMD_FLOAT32_C(    80.47), EASYSIMD_FLOAT32_C(   -88.73), EASYSIMD_FLOAT32_C(   -28.86),
        EASYSIMD_FLOAT32_C(    59.39), EASYSIMD_FLOAT32_C(   -91.27), EASYSIMD_FLOAT32_C(    -3.99), EASYSIMD_FLOAT32_C(    80.52),
        EASYSIMD_FLOAT32_C(    71.46), EASYSIMD_FLOAT32_C(   -57.84), EASYSIMD_FLOAT32_C(   -35.17), EASYSIMD_FLOAT32_C(   -94.81),
        EASYSIMD_FLOAT32_C(    39.19), EASYSIMD_FLOAT32_C(   -70.59), EASYSIMD_FLOAT32_C(   -91.26), EASYSIMD_FLOAT32_C(    27.12) },
      { EASYSIMD_FLOAT32_C(  3346.19), EASYSIMD_FLOAT32_C( -4259.05), EASYSIMD_FLOAT32_C(    -4.43), EASYSIMD_FLOAT32_C( -1635.07),
        EASYSIMD_FLOAT32_C(   140.20), EASYSIMD_FLOAT32_C( -3794.75), EASYSIMD_FLOAT32_C( -4399.25), EASYSIMD_FLOAT32_C( -1316.41),
        EASYSIMD_FLOAT32_C(  5357.26), EASYSIMD_FLOAT32_C( -3184.05), EASYSIMD_FLOAT32_C(  2173.76), EASYSIMD_FLOAT32_C(  -289.60),
        EASYSIMD_FLOAT32_C(  1857.83), EASYSIMD_FLOAT32_C(   290.52), EASYSIMD_FLOAT32_C( -1324.98), EASYSIMD_FLOAT32_C(  8601.23) } },
    { { EASYSIMD_FLOAT32_C(   -27.87), EASYSIMD_FLOAT32_C(   -30.25), EASYSIMD_FLOAT32_C(   -56.12), EASYSIMD_FLOAT32_C(    64.94),
        EASYSIMD_FLOAT32_C(    20.51), EASYSIMD_FLOAT32_C(    -1.30), EASYSIMD_FLOAT32_C(   -18.61), EASYSIMD_FLOAT32_C(    51.67),
        EASYSIMD_FLOAT32_C(    47.02), EASYSIMD_FLOAT32_C(   -72.27), EASYSIMD_FLOAT32_C(   -23.83), EASYSIMD_FLOAT32_C(   -81.60),
        EASYSIMD_FLOAT32_C(   -46.31), EASYSIMD_FLOAT32_C(    54.24), EASYSIMD_FLOAT32_C(   -71.05), EASYSIMD_FLOAT32_C(   -47.41) },
      { EASYSIMD_FLOAT32_C(    34.71), EASYSIMD_FLOAT32_C(   -59.79), EASYSIMD_FLOAT32_C(    23.73), EASYSIMD_FLOAT32_C(    -5.90),
        EASYSIMD_FLOAT32_C(   -51.06), EASYSIMD_FLOAT32_C(   -80.26), EASYSIMD_FLOAT32_C(   -25.38), EASYSIMD_FLOAT32_C(   -79.60),
        EASYSIMD_FLOAT32_C(   -38.10), EASYSIMD_FLOAT32_C(    39.45), EASYSIMD_FLOAT32_C(   -74.41), EASYSIMD_FLOAT32_C(   -98.91),
        EASYSIMD_FLOAT32_C(    68.87), EASYSIMD_FLOAT32_C(   -65.67), EASYSIMD_FLOAT32_C(    28.21), EASYSIMD_FLOAT32_C(   -59.01) },
      { EASYSIMD_FLOAT32_C(     4.08), EASYSIMD_FLOAT32_C(    72.09), EASYSIMD_FLOAT32_C(   -94.06), EASYSIMD_FLOAT32_C(   -75.41),
        EASYSIMD_FLOAT32_C(   -29.20), EASYSIMD_FLOAT32_C(   -12.67), EASYSIMD_FLOAT32_C(    76.26), EASYSIMD_FLOAT32_C(   -82.18),
        EASYSIMD_FLOAT32_C(    15.06), EASYSIMD_FLOAT32_C(   -47.57), EASYSIMD_FLOAT32_C(   -63.79), EASYSIMD_FLOAT32_C(    68.75),
        EASYSIMD_FLOAT32_C(   -93.33), EASYSIMD_FLOAT32_C(   -34.84), EASYSIMD_FLOAT32_C(   -78.66), EASYSIMD_FLOAT32_C(    41.38) },
      { EASYSIMD_FLOAT32_C(  -971.45), EASYSIMD_FLOAT32_C(  1736.56), EASYSIMD_FLOAT32_C( -1237.67), EASYSIMD_FLOAT32_C(  -307.74),
        EASYSIMD_FLOAT32_C( -1018.04), EASYSIMD_FLOAT32_C(   117.01), EASYSIMD_FLOAT32_C(   396.06), EASYSIMD_FLOAT32_C( -4030.75),
        EASYSIMD_FLOAT32_C( -1806.52), EASYSIMD_FLOAT32_C( -2803.48), EASYSIMD_FLOAT32_C(  1836.98), EASYSIMD_FLOAT32_C(  8002.31),
        EASYSIMD_FLOAT32_C( -3096.04), EASYSIMD_FLOAT32_C( -3527.10), EASYSIMD_FLOAT32_C( -1925.66), EASYSIMD_FLOAT32_C(  2756.28) } },
    { { EASYSIMD_FLOAT32_C(     5.37), EASYSIMD_FLOAT32_C(    45.08), EASYSIMD_FLOAT32_C(   -64.53), EASYSIMD_FLOAT32_C(    54.31),
        EASYSIMD_FLOAT32_C(    64.82), EASYSIMD_FLOAT32_C(    10.09), EASYSIMD_FLOAT32_C(    74.71), EASYSIMD_FLOAT32_C(   -73.28),
        EASYSIMD_FLOAT32_C(   -50.46), EASYSIMD_FLOAT32_C(   -99.70), EASYSIMD_FLOAT32_C(   -72.19), EASYSIMD_FLOAT32_C(   -81.59),
        EASYSIMD_FLOAT32_C(   -65.37), EASYSIMD_FLOAT32_C(    56.02), EASYSIMD_FLOAT32_C(   -40.60), EASYSIMD_FLOAT32_C(    38.71) },
      { EASYSIMD_FLOAT32_C(    28.11), EASYSIMD_FLOAT32_C(   -34.66), EASYSIMD_FLOAT32_C(    63.29), EASYSIMD_FLOAT32_C(    98.90),
        EASYSIMD_FLOAT32_C(    52.67), EASYSIMD_FLOAT32_C(    39.55), EASYSIMD_FLOAT32_C(   -83.28), EASYSIMD_FLOAT32_C(   -32.27),
        EASYSIMD_FLOAT32_C(    91.98), EASYSIMD_FLOAT32_C(   -47.07), EASYSIMD_FLOAT32_C(   -63.52), EASYSIMD_FLOAT32_C(    98.65),
        EASYSIMD_FLOAT32_C(    18.09), EASYSIMD_FLOAT32_C(   -42.18), EASYSIMD_FLOAT32_C(    40.03), EASYSIMD_FLOAT32_C(   -76.54) },
      { EASYSIMD_FLOAT32_C(   -97.10), EASYSIMD_FLOAT32_C(    75.50), EASYSIMD_FLOAT32_C(    77.78), EASYSIMD_FLOAT32_C(    67.71),
        EASYSIMD_FLOAT32_C(   -14.41), EASYSIMD_FLOAT32_C(    52.49), EASYSIMD_FLOAT32_C(    94.43), EASYSIMD_FLOAT32_C(    35.13),
        EASYSIMD_FLOAT32_C(    52.79), EASYSIMD_FLOAT32_C(   -77.76), EASYSIMD_FLOAT32_C(    53.54), EASYSIMD_FLOAT32_C(    87.41),
        EASYSIMD_FLOAT32_C(    78.26), EASYSIMD_FLOAT32_C(   -87.05), EASYSIMD_FLOAT32_C(    26.12), EASYSIMD_FLOAT32_C(     6.37) },
      { EASYSIMD_FLOAT32_C(   248.05), EASYSIMD_FLOAT32_C( -1637.97), EASYSIMD_FLOAT32_C( -4161.88), EASYSIMD_FLOAT32_C(  5303.55),
        EASYSIMD_FLOAT32_C(  3428.48), EASYSIMD_FLOAT32_C(   346.57), EASYSIMD_FLOAT32_C( -6316.28), EASYSIMD_FLOAT32_C(  2329.62),
        EASYSIMD_FLOAT32_C( -4694.10), EASYSIMD_FLOAT32_C(  4770.64), EASYSIMD_FLOAT32_C(  4531.97), EASYSIMD_FLOAT32_C( -8136.26),
        EASYSIMD_FLOAT32_C( -1260.80), EASYSIMD_FLOAT32_C( -2275.87), EASYSIMD_FLOAT32_C( -1651.34), EASYSIMD_FLOAT32_C( -2969.23) } },
    { { EASYSIMD_FLOAT32_C(   -21.71), EASYSIMD_FLOAT32_C(   -10.59), EASYSIMD_FLOAT32_C(     5.27), EASYSIMD_FLOAT32_C(   -69.04),
        EASYSIMD_FLOAT32_C(   -71.03), EASYSIMD_FLOAT32_C(    22.00), EASYSIMD_FLOAT32_C(    -1.31), EASYSIMD_FLOAT32_C(   -79.05),
        EASYSIMD_FLOAT32_C(    74.93), EASYSIMD_FLOAT32_C(    35.16), EASYSIMD_FLOAT32_C(   -80.40), EASYSIMD_FLOAT32_C(    -6.98),
        EASYSIMD_FLOAT32_C(    92.98), EASYSIMD_FLOAT32_C(    59.62), EASYSIMD_FLOAT32_C(    16.48), EASYSIMD_FLOAT32_C(    95.88) },
      { EASYSIMD_FLOAT32_C(    35.12), EASYSIMD_FLOAT32_C(    -5.74), EASYSIMD_FLOAT32_C(    63.59), EASYSIMD_FLOAT32_C(   -79.29),
        EASYSIMD_FLOAT32_C(   -53.25), EASYSIMD_FLOAT32_C(    58.02), EASYSIMD_FLOAT32_C(    55.85), EASYSIMD_FLOAT32_C(    99.54),
        EASYSIMD_FLOAT32_C(    80.26), EASYSIMD_FLOAT32_C(     9.39), EASYSIMD_FLOAT32_C(    86.95), EASYSIMD_FLOAT32_C(    58.52),
        EASYSIMD_FLOAT32_C(    22.34), EASYSIMD_FLOAT32_C(    13.07), EASYSIMD_FLOAT32_C(   -35.11), EASYSIMD_FLOAT32_C(   -99.37) },
      { EASYSIMD_FLOAT32_C(   -97.51), EASYSIMD_FLOAT32_C(    70.17), EASYSIMD_FLOAT32_C(   -68.42), EASYSIMD_FLOAT32_C(   -68.55),
        EASYSIMD_FLOAT32_C(    -7.84), EASYSIMD_FLOAT32_C(    30.27), EASYSIMD_FLOAT32_C(   -47.60), EASYSIMD_FLOAT32_C(   -32.91),
        EASYSIMD_FLOAT32_C(   -34.57), EASYSIMD_FLOAT32_C(   -28.00), EASYSIMD_FLOAT32_C(    60.11), EASYSIMD_FLOAT32_C(   -41.59),
        EASYSIMD_FLOAT32_C(   -68.38), EASYSIMD_FLOAT32_C(   -23.41), EASYSIMD_FLOAT32_C(   -45.71), EASYSIMD_FLOAT32_C(    66.75) },
      { EASYSIMD_FLOAT32_C(  -664.95), EASYSIMD_FLOAT32_C(    -9.38), EASYSIMD_FLOAT32_C(   403.54), EASYSIMD_FLOAT32_C(  5542.73),
        EASYSIMD_FLOAT32_C(  3790.19), EASYSIMD_FLOAT32_C(  1246.17), EASYSIMD_FLOAT32_C(   -25.56), EASYSIMD_FLOAT32_C( -7835.73),
        EASYSIMD_FLOAT32_C(  6048.45), EASYSIMD_FLOAT32_C(   358.15), EASYSIMD_FLOAT32_C( -7050.89), EASYSIMD_FLOAT32_C(  -366.88),
        EASYSIMD_FLOAT32_C(  2145.55), EASYSIMD_FLOAT32_C(   802.64), EASYSIMD_FLOAT32_C(  -532.90), EASYSIMD_FLOAT32_C( -9594.35) } },
    { { EASYSIMD_FLOAT32_C(    70.85), EASYSIMD_FLOAT32_C(   -82.11), EASYSIMD_FLOAT32_C(    87.46), EASYSIMD_FLOAT32_C(   -82.40),
        EASYSIMD_FLOAT32_C(    75.91), EASYSIMD_FLOAT32_C(    43.31), EASYSIMD_FLOAT32_C(   -82.86), EASYSIMD_FLOAT32_C(    56.17),
        EASYSIMD_FLOAT32_C(   -47.30), EASYSIMD_FLOAT32_C(   -95.91), EASYSIMD_FLOAT32_C(    14.69), EASYSIMD_FLOAT32_C(    75.04),
        EASYSIMD_FLOAT32_C(    17.16), EASYSIMD_FLOAT32_C(    79.59), EASYSIMD_FLOAT32_C(    75.66), EASYSIMD_FLOAT32_C(    19.64) },
      { EASYSIMD_FLOAT32_C(    49.75), EASYSIMD_FLOAT32_C(   -92.75), EASYSIMD_FLOAT32_C(    51.10), EASYSIMD_FLOAT32_C(   -58.08),
        EASYSIMD_FLOAT32_C(    37.51), EASYSIMD_FLOAT32_C(   -96.50), EASYSIMD_FLOAT32_C(     9.01), EASYSIMD_FLOAT32_C(   -97.06),
        EASYSIMD_FLOAT32_C(   -24.50), EASYSIMD_FLOAT32_C(   -30.88), EASYSIMD_FLOAT32_C(   -38.64), EASYSIMD_FLOAT32_C(     7.12),
        EASYSIMD_FLOAT32_C(    45.71), EASYSIMD_FLOAT32_C(    15.65), EASYSIMD_FLOAT32_C(   -26.14), EASYSIMD_FLOAT32_C(    16.56) },
      { EASYSIMD_FLOAT32_C(    33.54), EASYSIMD_FLOAT32_C(   -38.68), EASYSIMD_FLOAT32_C(    34.16), EASYSIMD_FLOAT32_C(     9.45),
        EASYSIMD_FLOAT32_C(   -95.37), EASYSIMD_FLOAT32_C(    51.30), EASYSIMD_FLOAT32_C(   -34.38), EASYSIMD_FLOAT32_C(   -42.67),
        EASYSIMD_FLOAT32_C(    55.39), EASYSIMD_FLOAT32_C(    80.31), EASYSIMD_FLOAT32_C(   -67.64), EASYSIMD_FLOAT32_C(   -27.46),
        EASYSIMD_FLOAT32_C(    59.90), EASYSIMD_FLOAT32_C(   -91.97), EASYSIMD_FLOAT32_C(    92.19), EASYSIMD_FLOAT32_C(     9.65) },
      { EASYSIMD_FLOAT32_C(  3491.25), EASYSIMD_FLOAT32_C(  7654.38), EASYSIMD_FLOAT32_C(  4435.05), EASYSIMD_FLOAT32_C(  4776.34),
        EASYSIMD_FLOAT32_C(  2942.75), EASYSIMD_FLOAT32_C( -4230.71), EASYSIMD_FLOAT32_C(  -712.19), EASYSIMD_FLOAT32_C( -5409.19),
        EASYSIMD_FLOAT32_C(  1103.46), EASYSIMD_FLOAT32_C(  2881.39), EASYSIMD_FLOAT32_C(  -499.98), EASYSIMD_FLOAT32_C(   561.74),
        EASYSIMD_FLOAT32_C(   724.48), EASYSIMD_FLOAT32_C(  1337.55), EASYSIMD_FLOAT32_C( -2069.94), EASYSIMD_FLOAT32_C(   315.59) } },
    { { EASYSIMD_FLOAT32_C(   -84.73), EASYSIMD_FLOAT32_C(    43.28), EASYSIMD_FLOAT32_C(    51.57), EASYSIMD_FLOAT32_C(    52.79),
        EASYSIMD_FLOAT32_C(    46.78), EASYSIMD_FLOAT32_C(   -39.42), EASYSIMD_FLOAT32_C(    55.73), EASYSIMD_FLOAT32_C(   -77.72),
        EASYSIMD_FLOAT32_C(    29.69), EASYSIMD_FLOAT32_C(   -82.91), EASYSIMD_FLOAT32_C(    29.40), EASYSIMD_FLOAT32_C(   -24.60),
        EASYSIMD_FLOAT32_C(    32.74), EASYSIMD_FLOAT32_C(   -96.74), EASYSIMD_FLOAT32_C(    91.96), EASYSIMD_FLOAT32_C(   -33.72) },
      { EASYSIMD_FLOAT32_C(   -35.42), EASYSIMD_FLOAT32_C(    26.13), EASYSIMD_FLOAT32_C(    75.73), EASYSIMD_FLOAT32_C(   -30.79),
        EASYSIMD_FLOAT32_C(   -22.57), EASYSIMD_FLOAT32_C(   -58.65), EASYSIMD_FLOAT32_C(    26.54), EASYSIMD_FLOAT32_C(   -67.19),
        EASYSIMD_FLOAT32_C(   -78.34), EASYSIMD_FLOAT32_C(    58.90), EASYSIMD_FLOAT32_C(     5.36), EASYSIMD_FLOAT32_C(    81.57),
        EASYSIMD_FLOAT32_C(    66.92), EASYSIMD_FLOAT32_C(    -2.46), EASYSIMD_FLOAT32_C(    -8.78), EASYSIMD_FLOAT32_C(    82.20) },
      { EASYSIMD_FLOAT32_C(   -59.17), EASYSIMD_FLOAT32_C(   -57.21), EASYSIMD_FLOAT32_C(    34.98), EASYSIMD_FLOAT32_C(    87.61),
        EASYSIMD_FLOAT32_C(     3.36), EASYSIMD_FLOAT32_C(    -9.29), EASYSIMD_FLOAT32_C(   -90.11), EASYSIMD_FLOAT32_C(   -66.95),
        EASYSIMD_FLOAT32_C(     7.80), EASYSIMD_FLOAT32_C(    39.28), EASYSIMD_FLOAT32_C(     8.46), EASYSIMD_FLOAT32_C(   -59.46),
        EASYSIMD_FLOAT32_C(    42.54), EASYSIMD_FLOAT32_C(     0.42), EASYSIMD_FLOAT32_C(     6.82), EASYSIMD_FLOAT32_C(   -92.87) },
      { EASYSIMD_FLOAT32_C(  3060.31), EASYSIMD_FLOAT32_C(  1188.12), EASYSIMD_FLOAT32_C(  3870.42), EASYSIMD_FLOAT32_C( -1713.01),
        EASYSIMD_FLOAT32_C( -1059.18), EASYSIMD_FLOAT32_C(  2321.27), EASYSIMD_FLOAT32_C(  1569.18), EASYSIMD_FLOAT32_C(  5288.96),
        EASYSIMD_FLOAT32_C( -2333.71), EASYSIMD_FLOAT32_C( -4922.68), EASYSIMD_FLOAT32_C(   149.12), EASYSIMD_FLOAT32_C( -1947.16),
        EASYSIMD_FLOAT32_C(  2148.42), EASYSIMD_FLOAT32_C(   237.56), EASYSIMD_FLOAT32_C(  -814.23), EASYSIMD_FLOAT32_C( -2678.91) } },
    { { EASYSIMD_FLOAT32_C(   -73.46), EASYSIMD_FLOAT32_C(   -17.45), EASYSIMD_FLOAT32_C(   -23.66), EASYSIMD_FLOAT32_C(     3.97),
        EASYSIMD_FLOAT32_C(    23.90), EASYSIMD_FLOAT32_C(   -97.13), EASYSIMD_FLOAT32_C(    36.78), EASYSIMD_FLOAT32_C(    45.56),
        EASYSIMD_FLOAT32_C(    61.77), EASYSIMD_FLOAT32_C(   -57.86), EASYSIMD_FLOAT32_C(    27.13), EASYSIMD_FLOAT32_C(    28.69),
        EASYSIMD_FLOAT32_C(    39.68), EASYSIMD_FLOAT32_C(   -81.65), EASYSIMD_FLOAT32_C(    10.89), EASYSIMD_FLOAT32_C(    80.51) },
      { EASYSIMD_FLOAT32_C(   -38.86), EASYSIMD_FLOAT32_C(   -54.13), EASYSIMD_FLOAT32_C(    68.12), EASYSIMD_FLOAT32_C(    64.50),
        EASYSIMD_FLOAT32_C(    36.58), EASYSIMD_FLOAT32_C(    78.01), EASYSIMD_FLOAT32_C(    97.56), EASYSIMD_FLOAT32_C(   -55.62),
        EASYSIMD_FLOAT32_C(    17.29), EASYSIMD_FLOAT32_C(     6.01), EASYSIMD_FLOAT32_C(   -15.08), EASYSIMD_FLOAT32_C(   -40.17),
        EASYSIMD_FLOAT32_C(   -93.57), EASYSIMD_FLOAT32_C(    91.74), EASYSIMD_FLOAT32_C(   -33.04), EASYSIMD_FLOAT32_C(   -67.03) },
      { EASYSIMD_FLOAT32_C(   -25.71), EASYSIMD_FLOAT32_C(    43.29), EASYSIMD_FLOAT32_C(    36.94), EASYSIMD_FLOAT32_C(    98.19),
        EASYSIMD_FLOAT32_C(    46.17), EASYSIMD_FLOAT32_C(   -26.28), EASYSIMD_FLOAT32_C(    43.76), EASYSIMD_FLOAT32_C(     7.94),
        EASYSIMD_FLOAT32_C(    15.86), EASYSIMD_FLOAT32_C(   -29.11), EASYSIMD_FLOAT32_C(   -63.37), EASYSIMD_FLOAT32_C(   -44.46),
        EASYSIMD_FLOAT32_C(   -10.77), EASYSIMD_FLOAT32_C(    47.52), EASYSIMD_FLOAT32_C(   -63.95), EASYSIMD_FLOAT32_C(    50.37) },
      { EASYSIMD_FLOAT32_C(  2880.37), EASYSIMD_FLOAT32_C(   901.28), EASYSIMD_FLOAT32_C( -1648.66), EASYSIMD_FLOAT32_C(   157.88),
        EASYSIMD_FLOAT32_C(   828.09), EASYSIMD_FLOAT32_C( -7550.83), EASYSIMD_FLOAT32_C(  3544.50), EASYSIMD_FLOAT32_C( -2541.99),
        EASYSIMD_FLOAT32_C(  1052.14), EASYSIMD_FLOAT32_C(  -318.63), EASYSIMD_FLOAT32_C(  -345.75), EASYSIMD_FLOAT32_C( -1108.02),
        EASYSIMD_FLOAT32_C( -3702.09), EASYSIMD_FLOAT32_C( -7538.09), EASYSIMD_FLOAT32_C(  -295.86), EASYSIMD_FLOAT32_C( -5446.96) } },
    { { EASYSIMD_FLOAT32_C(    93.39), EASYSIMD_FLOAT32_C(   -95.84), EASYSIMD_FLOAT32_C(    14.87), EASYSIMD_FLOAT32_C(    29.97),
        EASYSIMD_FLOAT32_C(    82.17), EASYSIMD_FLOAT32_C(    12.43), EASYSIMD_FLOAT32_C(    74.35), EASYSIMD_FLOAT32_C(    -0.54),
        EASYSIMD_FLOAT32_C(   -81.56), EASYSIMD_FLOAT32_C(   -40.72), EASYSIMD_FLOAT32_C(    59.29), EASYSIMD_FLOAT32_C(   -75.13),
        EASYSIMD_FLOAT32_C(   -48.98), EASYSIMD_FLOAT32_C(   -73.75), EASYSIMD_FLOAT32_C(   -42.16), EASYSIMD_FLOAT32_C(    25.31) },
      { EASYSIMD_FLOAT32_C(    69.55), EASYSIMD_FLOAT32_C(    94.78), EASYSIMD_FLOAT32_C(    23.51), EASYSIMD_FLOAT32_C(    15.71),
        EASYSIMD_FLOAT32_C(   -31.49), EASYSIMD_FLOAT32_C(   -32.74), EASYSIMD_FLOAT32_C(   -76.35), EASYSIMD_FLOAT32_C(    84.37),
        EASYSIMD_FLOAT32_C(    38.15), EASYSIMD_FLOAT32_C(   -39.72), EASYSIMD_FLOAT32_C(   -60.09), EASYSIMD_FLOAT32_C(   -72.62),
        EASYSIMD_FLOAT32_C(   -92.20), EASYSIMD_FLOAT32_C(   -24.04), EASYSIMD_FLOAT32_C(    77.76), EASYSIMD_FLOAT32_C(   -98.81) },
      { EASYSIMD_FLOAT32_C(   -19.88), EASYSIMD_FLOAT32_C(    -7.37), EASYSIMD_FLOAT32_C(    31.16), EASYSIMD_FLOAT32_C(   -37.71),
        EASYSIMD_FLOAT32_C(   -94.94), EASYSIMD_FLOAT32_C(     5.51), EASYSIMD_FLOAT32_C(    61.75), EASYSIMD_FLOAT32_C(   -76.50),
        EASYSIMD_FLOAT32_C(    64.79), EASYSIMD_FLOAT32_C(    21.04), EASYSIMD_FLOAT32_C(   -51.63), EASYSIMD_FLOAT32_C(   -84.19),
        EASYSIMD_FLOAT32_C(    47.29), EASYSIMD_FLOAT32_C(     6.21), EASYSIMD_FLOAT32_C(    41.12), EASYSIMD_FLOAT32_C(    16.84) },
      { EASYSIMD_FLOAT32_C(  6515.15), EASYSIMD_FLOAT32_C( -9076.34), EASYSIMD_FLOAT32_C(   318.43), EASYSIMD_FLOAT32_C(   508.54),
        EASYSIMD_FLOAT32_C( -2492.59), EASYSIMD_FLOAT32_C(  -412.47), EASYSIMD_FLOAT32_C( -5738.37), EASYSIMD_FLOAT32_C(    30.94),
        EASYSIMD_FLOAT32_C( -3176.30), EASYSIMD_FLOAT32_C(  1596.36), EASYSIMD_FLOAT32_C( -3511.11), EASYSIMD_FLOAT32_C(  5540.13),
        EASYSIMD_FLOAT32_C(  4468.67), EASYSIMD_FLOAT32_C(  1766.74), EASYSIMD_FLOAT32_C( -3319.48), EASYSIMD_FLOAT32_C( -2517.72) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 c = easysimd_mm512_loadu_ps(test_vec[i].c);
    easysimd__m512 r = easysimd_mm512_fmsub_ps(a, b, c);
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_fmsub_ps(a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_fmsub_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_fmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const easysimd__mmask16 k;
    const easysimd_float32 b[16];
    const easysimd_float32 c[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -467.27), EASYSIMD_FLOAT32_C(  -468.28), EASYSIMD_FLOAT32_C(   196.38), EASYSIMD_FLOAT32_C(   925.20),
        EASYSIMD_FLOAT32_C(    -6.19), EASYSIMD_FLOAT32_C(   169.04), EASYSIMD_FLOAT32_C(  -951.84), EASYSIMD_FLOAT32_C(  -906.14),
        EASYSIMD_FLOAT32_C(   386.84), EASYSIMD_FLOAT32_C(  -831.02), EASYSIMD_FLOAT32_C(   513.65), EASYSIMD_FLOAT32_C(   729.39),
        EASYSIMD_FLOAT32_C(  -929.74), EASYSIMD_FLOAT32_C(  -671.50), EASYSIMD_FLOAT32_C(    77.04), EASYSIMD_FLOAT32_C(   167.35) },
      UINT16_C(10936),
      { EASYSIMD_FLOAT32_C(    27.11), EASYSIMD_FLOAT32_C(  -807.19), EASYSIMD_FLOAT32_C(  -667.17), EASYSIMD_FLOAT32_C(   878.86),
        EASYSIMD_FLOAT32_C(  -131.52), EASYSIMD_FLOAT32_C(   675.81), EASYSIMD_FLOAT32_C(    12.27), EASYSIMD_FLOAT32_C(  -443.33),
        EASYSIMD_FLOAT32_C(   523.80), EASYSIMD_FLOAT32_C(  -362.92), EASYSIMD_FLOAT32_C(    89.83), EASYSIMD_FLOAT32_C(  -281.00),
        EASYSIMD_FLOAT32_C(  -628.63), EASYSIMD_FLOAT32_C(   622.56), EASYSIMD_FLOAT32_C(   250.72), EASYSIMD_FLOAT32_C(   567.76) },
      { EASYSIMD_FLOAT32_C(   547.75), EASYSIMD_FLOAT32_C(  -755.47), EASYSIMD_FLOAT32_C(  -263.20), EASYSIMD_FLOAT32_C(   595.91),
        EASYSIMD_FLOAT32_C(  -661.60), EASYSIMD_FLOAT32_C(  -876.36), EASYSIMD_FLOAT32_C(   764.89), EASYSIMD_FLOAT32_C(   852.04),
        EASYSIMD_FLOAT32_C(   853.03), EASYSIMD_FLOAT32_C(   835.15), EASYSIMD_FLOAT32_C(  -819.46), EASYSIMD_FLOAT32_C(   -69.93),
        EASYSIMD_FLOAT32_C(     2.50), EASYSIMD_FLOAT32_C(   882.44), EASYSIMD_FLOAT32_C(   406.64), EASYSIMD_FLOAT32_C(  -970.39) },
      { EASYSIMD_FLOAT32_C(  -467.27), EASYSIMD_FLOAT32_C(  -468.28), EASYSIMD_FLOAT32_C(   196.38), EASYSIMD_FLOAT32_C(812525.31),
        EASYSIMD_FLOAT32_C(  1475.71), EASYSIMD_FLOAT32_C(115115.27), EASYSIMD_FLOAT32_C(  -951.84), EASYSIMD_FLOAT32_C(400867.00),
        EASYSIMD_FLOAT32_C(   386.84), EASYSIMD_FLOAT32_C(300758.62), EASYSIMD_FLOAT32_C(   513.65), EASYSIMD_FLOAT32_C(-204888.66),
        EASYSIMD_FLOAT32_C(  -929.74), EASYSIMD_FLOAT32_C(-418931.47), EASYSIMD_FLOAT32_C(    77.04), EASYSIMD_FLOAT32_C(   167.35) } },
    { { EASYSIMD_FLOAT32_C(  -924.75), EASYSIMD_FLOAT32_C(   739.47), EASYSIMD_FLOAT32_C(   908.47), EASYSIMD_FLOAT32_C(   -56.27),
        EASYSIMD_FLOAT32_C(   415.28), EASYSIMD_FLOAT32_C(   -79.26), EASYSIMD_FLOAT32_C(   500.39), EASYSIMD_FLOAT32_C(   -60.93),
        EASYSIMD_FLOAT32_C(   557.82), EASYSIMD_FLOAT32_C(  -409.78), EASYSIMD_FLOAT32_C(   658.07), EASYSIMD_FLOAT32_C(   929.20),
        EASYSIMD_FLOAT32_C(  -787.22), EASYSIMD_FLOAT32_C(   -91.21), EASYSIMD_FLOAT32_C(   496.95), EASYSIMD_FLOAT32_C(   760.53) },
      UINT16_C(50822),
      { EASYSIMD_FLOAT32_C(   356.44), EASYSIMD_FLOAT32_C(   491.72), EASYSIMD_FLOAT32_C(  -642.61), EASYSIMD_FLOAT32_C(   121.33),
        EASYSIMD_FLOAT32_C(   343.76), EASYSIMD_FLOAT32_C(  -789.58), EASYSIMD_FLOAT32_C(   -43.52), EASYSIMD_FLOAT32_C(   524.30),
        EASYSIMD_FLOAT32_C(   140.49), EASYSIMD_FLOAT32_C(   958.98), EASYSIMD_FLOAT32_C(   406.74), EASYSIMD_FLOAT32_C(  -452.87),
        EASYSIMD_FLOAT32_C(   988.59), EASYSIMD_FLOAT32_C(   481.99), EASYSIMD_FLOAT32_C(  -713.40), EASYSIMD_FLOAT32_C(   897.07) },
      { EASYSIMD_FLOAT32_C(  -574.28), EASYSIMD_FLOAT32_C(   701.88), EASYSIMD_FLOAT32_C(  -182.19), EASYSIMD_FLOAT32_C(   926.11),
        EASYSIMD_FLOAT32_C(  -359.05), EASYSIMD_FLOAT32_C(  -624.36), EASYSIMD_FLOAT32_C(  -483.67), EASYSIMD_FLOAT32_C(  -700.97),
        EASYSIMD_FLOAT32_C(  -695.17), EASYSIMD_FLOAT32_C(  -270.90), EASYSIMD_FLOAT32_C(   207.82), EASYSIMD_FLOAT32_C(   801.79),
        EASYSIMD_FLOAT32_C(  -510.37), EASYSIMD_FLOAT32_C(  -638.85), EASYSIMD_FLOAT32_C(  -964.47), EASYSIMD_FLOAT32_C(   846.07) },
      { EASYSIMD_FLOAT32_C(  -924.75), EASYSIMD_FLOAT32_C(362910.31), EASYSIMD_FLOAT32_C(-583609.69), EASYSIMD_FLOAT32_C(   -56.27),
        EASYSIMD_FLOAT32_C(   415.28), EASYSIMD_FLOAT32_C(   -79.26), EASYSIMD_FLOAT32_C(   500.39), EASYSIMD_FLOAT32_C(-31244.63),
        EASYSIMD_FLOAT32_C(   557.82), EASYSIMD_FLOAT32_C(-392699.91), EASYSIMD_FLOAT32_C(267455.56), EASYSIMD_FLOAT32_C(   929.20),
        EASYSIMD_FLOAT32_C(  -787.22), EASYSIMD_FLOAT32_C(   -91.21), EASYSIMD_FLOAT32_C(-353559.69), EASYSIMD_FLOAT32_C(681402.62) } },
    { { EASYSIMD_FLOAT32_C(   852.87), EASYSIMD_FLOAT32_C(  -607.08), EASYSIMD_FLOAT32_C(   -32.60), EASYSIMD_FLOAT32_C(   196.63),
        EASYSIMD_FLOAT32_C(  -396.66), EASYSIMD_FLOAT32_C(   923.88), EASYSIMD_FLOAT32_C(  -279.06), EASYSIMD_FLOAT32_C(   743.82),
        EASYSIMD_FLOAT32_C(   882.86), EASYSIMD_FLOAT32_C(  -872.32), EASYSIMD_FLOAT32_C(  -709.05), EASYSIMD_FLOAT32_C(   871.46),
        EASYSIMD_FLOAT32_C(   609.67), EASYSIMD_FLOAT32_C(  -422.45), EASYSIMD_FLOAT32_C(   768.52), EASYSIMD_FLOAT32_C(  -964.62) },
      UINT16_C(38122),
      { EASYSIMD_FLOAT32_C(   961.49), EASYSIMD_FLOAT32_C(   -79.61), EASYSIMD_FLOAT32_C(   -38.03), EASYSIMD_FLOAT32_C(  -522.18),
        EASYSIMD_FLOAT32_C(   219.42), EASYSIMD_FLOAT32_C(   266.81), EASYSIMD_FLOAT32_C(   206.92), EASYSIMD_FLOAT32_C(  -572.77),
        EASYSIMD_FLOAT32_C(    68.59), EASYSIMD_FLOAT32_C(   696.55), EASYSIMD_FLOAT32_C(  -211.62), EASYSIMD_FLOAT32_C(   104.13),
        EASYSIMD_FLOAT32_C(   542.62), EASYSIMD_FLOAT32_C(  -358.75), EASYSIMD_FLOAT32_C(   497.05), EASYSIMD_FLOAT32_C(  -489.98) },
      { EASYSIMD_FLOAT32_C(   837.88), EASYSIMD_FLOAT32_C(  -899.62), EASYSIMD_FLOAT32_C(  -566.11), EASYSIMD_FLOAT32_C(  -441.18),
        EASYSIMD_FLOAT32_C(   844.20), EASYSIMD_FLOAT32_C(  -683.25), EASYSIMD_FLOAT32_C(  -313.51), EASYSIMD_FLOAT32_C(  -864.85),
        EASYSIMD_FLOAT32_C(  -811.79), EASYSIMD_FLOAT32_C(  -703.84), EASYSIMD_FLOAT32_C(  -287.29), EASYSIMD_FLOAT32_C(   956.73),
        EASYSIMD_FLOAT32_C(  -668.46), EASYSIMD_FLOAT32_C(    -7.86), EASYSIMD_FLOAT32_C(  -456.93), EASYSIMD_FLOAT32_C(  -706.97) },
      { EASYSIMD_FLOAT32_C(   852.87), EASYSIMD_FLOAT32_C( 49229.26), EASYSIMD_FLOAT32_C(   -32.60), EASYSIMD_FLOAT32_C(-102235.08),
        EASYSIMD_FLOAT32_C(  -396.66), EASYSIMD_FLOAT32_C(247183.67), EASYSIMD_FLOAT32_C(-57429.58), EASYSIMD_FLOAT32_C(-425172.97),
        EASYSIMD_FLOAT32_C(   882.86), EASYSIMD_FLOAT32_C(  -872.32), EASYSIMD_FLOAT32_C(150336.45), EASYSIMD_FLOAT32_C(   871.46),
        EASYSIMD_FLOAT32_C(331487.59), EASYSIMD_FLOAT32_C(  -422.45), EASYSIMD_FLOAT32_C(   768.52), EASYSIMD_FLOAT32_C(473351.50) } },
    { { EASYSIMD_FLOAT32_C(   912.53), EASYSIMD_FLOAT32_C(   505.04), EASYSIMD_FLOAT32_C(  -229.15), EASYSIMD_FLOAT32_C(   131.95),
        EASYSIMD_FLOAT32_C(  -228.15), EASYSIMD_FLOAT32_C(   977.77), EASYSIMD_FLOAT32_C(   559.18), EASYSIMD_FLOAT32_C(   840.44),
        EASYSIMD_FLOAT32_C(   674.31), EASYSIMD_FLOAT32_C(  -652.44), EASYSIMD_FLOAT32_C(   -55.43), EASYSIMD_FLOAT32_C(   216.93),
        EASYSIMD_FLOAT32_C(   -11.19), EASYSIMD_FLOAT32_C(  -558.39), EASYSIMD_FLOAT32_C(   726.94), EASYSIMD_FLOAT32_C(  -173.31) },
      UINT16_C(12199),
      { EASYSIMD_FLOAT32_C(   385.50), EASYSIMD_FLOAT32_C(  -613.80), EASYSIMD_FLOAT32_C(  -522.41), EASYSIMD_FLOAT32_C(  -928.01),
        EASYSIMD_FLOAT32_C(  -478.65), EASYSIMD_FLOAT32_C(  -334.20), EASYSIMD_FLOAT32_C(  -631.85), EASYSIMD_FLOAT32_C(   234.06),
        EASYSIMD_FLOAT32_C(  -377.47), EASYSIMD_FLOAT32_C(  -300.31), EASYSIMD_FLOAT32_C(  -773.80), EASYSIMD_FLOAT32_C(   165.60),
        EASYSIMD_FLOAT32_C(    -7.28), EASYSIMD_FLOAT32_C(  -861.27), EASYSIMD_FLOAT32_C(  -329.35), EASYSIMD_FLOAT32_C(   763.57) },
      { EASYSIMD_FLOAT32_C(   270.68), EASYSIMD_FLOAT32_C(   442.50), EASYSIMD_FLOAT32_C(   741.34), EASYSIMD_FLOAT32_C(  -170.14),
        EASYSIMD_FLOAT32_C(   282.94), EASYSIMD_FLOAT32_C(   415.65), EASYSIMD_FLOAT32_C(   177.42), EASYSIMD_FLOAT32_C(  -772.49),
        EASYSIMD_FLOAT32_C(  -367.42), EASYSIMD_FLOAT32_C(  -833.78), EASYSIMD_FLOAT32_C(  -330.88), EASYSIMD_FLOAT32_C(  -640.48),
        EASYSIMD_FLOAT32_C(    -7.09), EASYSIMD_FLOAT32_C(   211.12), EASYSIMD_FLOAT32_C(  -479.64), EASYSIMD_FLOAT32_C(  -621.59) },
      { EASYSIMD_FLOAT32_C(351509.62), EASYSIMD_FLOAT32_C(-310436.06), EASYSIMD_FLOAT32_C(118968.90), EASYSIMD_FLOAT32_C(   131.95),
        EASYSIMD_FLOAT32_C(  -228.15), EASYSIMD_FLOAT32_C(-327186.41), EASYSIMD_FLOAT32_C(   559.18), EASYSIMD_FLOAT32_C(197485.88),
        EASYSIMD_FLOAT32_C(-254164.38), EASYSIMD_FLOAT32_C(196768.03), EASYSIMD_FLOAT32_C( 43222.61), EASYSIMD_FLOAT32_C( 36564.09),
        EASYSIMD_FLOAT32_C(   -11.19), EASYSIMD_FLOAT32_C(480713.47), EASYSIMD_FLOAT32_C(   726.94), EASYSIMD_FLOAT32_C(  -173.31) } },
    { { EASYSIMD_FLOAT32_C(   597.32), EASYSIMD_FLOAT32_C(    -2.05), EASYSIMD_FLOAT32_C(  -549.60), EASYSIMD_FLOAT32_C(  -881.33),
        EASYSIMD_FLOAT32_C(   663.75), EASYSIMD_FLOAT32_C(  -181.44), EASYSIMD_FLOAT32_C(   352.73), EASYSIMD_FLOAT32_C(  -713.72),
        EASYSIMD_FLOAT32_C(   518.25), EASYSIMD_FLOAT32_C(   578.93), EASYSIMD_FLOAT32_C(   451.88), EASYSIMD_FLOAT32_C(  -489.03),
        EASYSIMD_FLOAT32_C(   717.66), EASYSIMD_FLOAT32_C(  -877.47), EASYSIMD_FLOAT32_C(  -725.46), EASYSIMD_FLOAT32_C(   -11.66) },
      UINT16_C(58400),
      { EASYSIMD_FLOAT32_C(   818.20), EASYSIMD_FLOAT32_C(  -152.03), EASYSIMD_FLOAT32_C(   431.52), EASYSIMD_FLOAT32_C(    -4.38),
        EASYSIMD_FLOAT32_C(    75.48), EASYSIMD_FLOAT32_C(  -935.90), EASYSIMD_FLOAT32_C(   161.84), EASYSIMD_FLOAT32_C(   744.61),
        EASYSIMD_FLOAT32_C(  -576.38), EASYSIMD_FLOAT32_C(  -845.25), EASYSIMD_FLOAT32_C(   -44.27), EASYSIMD_FLOAT32_C(   -56.02),
        EASYSIMD_FLOAT32_C(  -466.84), EASYSIMD_FLOAT32_C(  -446.96), EASYSIMD_FLOAT32_C(   941.93), EASYSIMD_FLOAT32_C(   -16.43) },
      { EASYSIMD_FLOAT32_C(  -328.29), EASYSIMD_FLOAT32_C(   605.68), EASYSIMD_FLOAT32_C(   802.12), EASYSIMD_FLOAT32_C(  -975.56),
        EASYSIMD_FLOAT32_C(   891.97), EASYSIMD_FLOAT32_C(   320.37), EASYSIMD_FLOAT32_C(   603.38), EASYSIMD_FLOAT32_C(   343.85),
        EASYSIMD_FLOAT32_C(   831.34), EASYSIMD_FLOAT32_C(   321.04), EASYSIMD_FLOAT32_C(   466.38), EASYSIMD_FLOAT32_C(  -894.12),
        EASYSIMD_FLOAT32_C(  -690.62), EASYSIMD_FLOAT32_C(    31.41), EASYSIMD_FLOAT32_C(  -878.24), EASYSIMD_FLOAT32_C(  -872.42) },
      { EASYSIMD_FLOAT32_C(   597.32), EASYSIMD_FLOAT32_C(    -2.05), EASYSIMD_FLOAT32_C(  -549.60), EASYSIMD_FLOAT32_C(  -881.33),
        EASYSIMD_FLOAT32_C(   663.75), EASYSIMD_FLOAT32_C(169489.33), EASYSIMD_FLOAT32_C(   352.73), EASYSIMD_FLOAT32_C(  -713.72),
        EASYSIMD_FLOAT32_C(   518.25), EASYSIMD_FLOAT32_C(   578.93), EASYSIMD_FLOAT32_C(-20471.11), EASYSIMD_FLOAT32_C(  -489.03),
        EASYSIMD_FLOAT32_C(   717.66), EASYSIMD_FLOAT32_C(392162.56), EASYSIMD_FLOAT32_C(-682454.31), EASYSIMD_FLOAT32_C(  1063.99) } },
    { { EASYSIMD_FLOAT32_C(   879.39), EASYSIMD_FLOAT32_C(   553.28), EASYSIMD_FLOAT32_C(   123.20), EASYSIMD_FLOAT32_C(   -45.13),
        EASYSIMD_FLOAT32_C(   617.39), EASYSIMD_FLOAT32_C(  -714.96), EASYSIMD_FLOAT32_C(  -300.53), EASYSIMD_FLOAT32_C(  -958.99),
        EASYSIMD_FLOAT32_C(  -560.20), EASYSIMD_FLOAT32_C(   655.20), EASYSIMD_FLOAT32_C(   -15.01), EASYSIMD_FLOAT32_C(   -27.04),
        EASYSIMD_FLOAT32_C(  -791.76), EASYSIMD_FLOAT32_C(   -73.07), EASYSIMD_FLOAT32_C(   956.53), EASYSIMD_FLOAT32_C(  -120.04) },
      UINT16_C(27098),
      { EASYSIMD_FLOAT32_C(   -95.60), EASYSIMD_FLOAT32_C(  -575.42), EASYSIMD_FLOAT32_C(    79.03), EASYSIMD_FLOAT32_C(  -492.22),
        EASYSIMD_FLOAT32_C(   768.43), EASYSIMD_FLOAT32_C(   -89.63), EASYSIMD_FLOAT32_C(   828.82), EASYSIMD_FLOAT32_C(   234.81),
        EASYSIMD_FLOAT32_C(    16.25), EASYSIMD_FLOAT32_C(  -861.80), EASYSIMD_FLOAT32_C(  -733.78), EASYSIMD_FLOAT32_C(   138.01),
        EASYSIMD_FLOAT32_C(  -734.22), EASYSIMD_FLOAT32_C(  -854.39), EASYSIMD_FLOAT32_C(  -308.70), EASYSIMD_FLOAT32_C(   388.98) },
      { EASYSIMD_FLOAT32_C(   100.48), EASYSIMD_FLOAT32_C(  -691.32), EASYSIMD_FLOAT32_C(   674.03), EASYSIMD_FLOAT32_C(   799.95),
        EASYSIMD_FLOAT32_C(  -650.31), EASYSIMD_FLOAT32_C(  -886.17), EASYSIMD_FLOAT32_C(   455.15), EASYSIMD_FLOAT32_C(   334.69),
        EASYSIMD_FLOAT32_C(    86.79), EASYSIMD_FLOAT32_C(   663.40), EASYSIMD_FLOAT32_C(  -738.38), EASYSIMD_FLOAT32_C(    43.32),
        EASYSIMD_FLOAT32_C(  -456.64), EASYSIMD_FLOAT32_C(  -205.77), EASYSIMD_FLOAT32_C(  -198.03), EASYSIMD_FLOAT32_C(   447.76) },
      { EASYSIMD_FLOAT32_C(   879.39), EASYSIMD_FLOAT32_C(-317677.06), EASYSIMD_FLOAT32_C(   123.20), EASYSIMD_FLOAT32_C( 21413.94),
        EASYSIMD_FLOAT32_C(475071.31), EASYSIMD_FLOAT32_C(  -714.96), EASYSIMD_FLOAT32_C(-249540.44), EASYSIMD_FLOAT32_C(-225515.12),
        EASYSIMD_FLOAT32_C( -9190.04), EASYSIMD_FLOAT32_C(   655.20), EASYSIMD_FLOAT32_C(   -15.01), EASYSIMD_FLOAT32_C( -3775.11),
        EASYSIMD_FLOAT32_C(  -791.76), EASYSIMD_FLOAT32_C( 62636.05), EASYSIMD_FLOAT32_C(-295082.81), EASYSIMD_FLOAT32_C(  -120.04) } },
    { { EASYSIMD_FLOAT32_C(   218.81), EASYSIMD_FLOAT32_C(   881.00), EASYSIMD_FLOAT32_C(   955.53), EASYSIMD_FLOAT32_C(   -12.77),
        EASYSIMD_FLOAT32_C(  -208.63), EASYSIMD_FLOAT32_C(   784.35), EASYSIMD_FLOAT32_C(  -777.95), EASYSIMD_FLOAT32_C(   807.62),
        EASYSIMD_FLOAT32_C(   922.55), EASYSIMD_FLOAT32_C(  -511.73), EASYSIMD_FLOAT32_C(   -54.37), EASYSIMD_FLOAT32_C(  -811.67),
        EASYSIMD_FLOAT32_C(  -366.12), EASYSIMD_FLOAT32_C(   636.93), EASYSIMD_FLOAT32_C(   577.32), EASYSIMD_FLOAT32_C(   734.36) },
      UINT16_C(63411),
      { EASYSIMD_FLOAT32_C(   534.31), EASYSIMD_FLOAT32_C(  -704.69), EASYSIMD_FLOAT32_C(   365.17), EASYSIMD_FLOAT32_C(   -10.54),
        EASYSIMD_FLOAT32_C(   629.99), EASYSIMD_FLOAT32_C(  -548.04), EASYSIMD_FLOAT32_C(  -347.14), EASYSIMD_FLOAT32_C(   891.61),
        EASYSIMD_FLOAT32_C(   495.28), EASYSIMD_FLOAT32_C(   196.21), EASYSIMD_FLOAT32_C(  -314.16), EASYSIMD_FLOAT32_C(  -702.75),
        EASYSIMD_FLOAT32_C(  -356.03), EASYSIMD_FLOAT32_C(   904.65), EASYSIMD_FLOAT32_C(  -821.75), EASYSIMD_FLOAT32_C(  -400.50) },
      { EASYSIMD_FLOAT32_C(  -108.12), EASYSIMD_FLOAT32_C(   -30.38), EASYSIMD_FLOAT32_C(  -616.15), EASYSIMD_FLOAT32_C(   113.93),
        EASYSIMD_FLOAT32_C(  -222.76), EASYSIMD_FLOAT32_C(  -693.60), EASYSIMD_FLOAT32_C(   602.20), EASYSIMD_FLOAT32_C(   722.88),
        EASYSIMD_FLOAT32_C(  -505.26), EASYSIMD_FLOAT32_C(  -763.92), EASYSIMD_FLOAT32_C(   359.81), EASYSIMD_FLOAT32_C(  -927.95),
        EASYSIMD_FLOAT32_C(   970.43), EASYSIMD_FLOAT32_C(   305.42), EASYSIMD_FLOAT32_C(   323.40), EASYSIMD_FLOAT32_C(   504.74) },
      { EASYSIMD_FLOAT32_C(117020.48), EASYSIMD_FLOAT32_C(-620801.50), EASYSIMD_FLOAT32_C(   955.53), EASYSIMD_FLOAT32_C(   -12.77),
        EASYSIMD_FLOAT32_C(-131212.05), EASYSIMD_FLOAT32_C(-429161.56), EASYSIMD_FLOAT32_C(  -777.95), EASYSIMD_FLOAT32_C(719359.19),
        EASYSIMD_FLOAT32_C(457425.81), EASYSIMD_FLOAT32_C(-99642.62), EASYSIMD_FLOAT32_C( 16721.07), EASYSIMD_FLOAT32_C(  -811.67),
        EASYSIMD_FLOAT32_C(129379.27), EASYSIMD_FLOAT32_C(575893.31), EASYSIMD_FLOAT32_C(-474736.12), EASYSIMD_FLOAT32_C(-294615.94) } },
    { { EASYSIMD_FLOAT32_C(   600.73), EASYSIMD_FLOAT32_C(  -311.43), EASYSIMD_FLOAT32_C(  -505.80), EASYSIMD_FLOAT32_C(   230.72),
        EASYSIMD_FLOAT32_C(   140.53), EASYSIMD_FLOAT32_C(   147.06), EASYSIMD_FLOAT32_C(   122.33), EASYSIMD_FLOAT32_C(  -364.20),
        EASYSIMD_FLOAT32_C(  -656.73), EASYSIMD_FLOAT32_C(   808.17), EASYSIMD_FLOAT32_C(   -66.95), EASYSIMD_FLOAT32_C(   -12.76),
        EASYSIMD_FLOAT32_C(   712.82), EASYSIMD_FLOAT32_C(   111.30), EASYSIMD_FLOAT32_C(   586.74), EASYSIMD_FLOAT32_C(  -395.30) },
      UINT16_C(10485),
      { EASYSIMD_FLOAT32_C(   718.62), EASYSIMD_FLOAT32_C(  -141.84), EASYSIMD_FLOAT32_C(  -723.00), EASYSIMD_FLOAT32_C(   320.82),
        EASYSIMD_FLOAT32_C(  -418.97), EASYSIMD_FLOAT32_C(  -228.26), EASYSIMD_FLOAT32_C(   556.89), EASYSIMD_FLOAT32_C(   940.84),
        EASYSIMD_FLOAT32_C(  -156.21), EASYSIMD_FLOAT32_C(   527.33), EASYSIMD_FLOAT32_C(   246.26), EASYSIMD_FLOAT32_C(  -832.82),
        EASYSIMD_FLOAT32_C(    32.07), EASYSIMD_FLOAT32_C(  -153.01), EASYSIMD_FLOAT32_C(  -144.25), EASYSIMD_FLOAT32_C(   526.27) },
      { EASYSIMD_FLOAT32_C(  -922.29), EASYSIMD_FLOAT32_C(   996.28), EASYSIMD_FLOAT32_C(  -326.68), EASYSIMD_FLOAT32_C(   200.04),
        EASYSIMD_FLOAT32_C(  -367.92), EASYSIMD_FLOAT32_C(    16.59), EASYSIMD_FLOAT32_C(     8.21), EASYSIMD_FLOAT32_C(   565.13),
        EASYSIMD_FLOAT32_C(  -996.17), EASYSIMD_FLOAT32_C(  -278.97), EASYSIMD_FLOAT32_C(  -323.57), EASYSIMD_FLOAT32_C(   590.58),
        EASYSIMD_FLOAT32_C(   325.72), EASYSIMD_FLOAT32_C(  -242.65), EASYSIMD_FLOAT32_C(   561.18), EASYSIMD_FLOAT32_C(    44.35) },
      { EASYSIMD_FLOAT32_C(432618.84), EASYSIMD_FLOAT32_C(  -311.43), EASYSIMD_FLOAT32_C(366020.09), EASYSIMD_FLOAT32_C(   230.72),
        EASYSIMD_FLOAT32_C(-58509.93), EASYSIMD_FLOAT32_C(-33584.50), EASYSIMD_FLOAT32_C( 68116.15), EASYSIMD_FLOAT32_C(-343219.06),
        EASYSIMD_FLOAT32_C(  -656.73), EASYSIMD_FLOAT32_C(   808.17), EASYSIMD_FLOAT32_C(   -66.95), EASYSIMD_FLOAT32_C( 10036.20),
        EASYSIMD_FLOAT32_C(   712.82), EASYSIMD_FLOAT32_C(-16787.36), EASYSIMD_FLOAT32_C(   586.74), EASYSIMD_FLOAT32_C(  -395.30) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 c = easysimd_mm512_loadu_ps(test_vec[i].c);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_fmsub_ps(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_fmsub_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 c = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_mask_fmsub_ps(a, k, b, c);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_fmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const easysimd__mmask16 k;
    const easysimd_float32 b[16];
    const easysimd_float32 c[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -860.20), EASYSIMD_FLOAT32_C(  -712.10), EASYSIMD_FLOAT32_C(   243.95), EASYSIMD_FLOAT32_C(  -406.44),
        EASYSIMD_FLOAT32_C(  -470.25), EASYSIMD_FLOAT32_C(   363.77), EASYSIMD_FLOAT32_C(  -533.64), EASYSIMD_FLOAT32_C(  -994.30),
        EASYSIMD_FLOAT32_C(   391.28), EASYSIMD_FLOAT32_C(    24.23), EASYSIMD_FLOAT32_C(    91.54), EASYSIMD_FLOAT32_C(  -479.09),
        EASYSIMD_FLOAT32_C(   -77.37), EASYSIMD_FLOAT32_C(  -358.81), EASYSIMD_FLOAT32_C(  -994.97), EASYSIMD_FLOAT32_C(  -698.85) },
      UINT16_C(45343),
      { EASYSIMD_FLOAT32_C(  -274.58), EASYSIMD_FLOAT32_C(   615.02), EASYSIMD_FLOAT32_C(    62.52), EASYSIMD_FLOAT32_C(  -414.52),
        EASYSIMD_FLOAT32_C(  -161.36), EASYSIMD_FLOAT32_C(   -90.40), EASYSIMD_FLOAT32_C(   745.57), EASYSIMD_FLOAT32_C(  -159.85),
        EASYSIMD_FLOAT32_C(  -173.21), EASYSIMD_FLOAT32_C(   353.80), EASYSIMD_FLOAT32_C(   885.21), EASYSIMD_FLOAT32_C(   253.37),
        EASYSIMD_FLOAT32_C(    51.75), EASYSIMD_FLOAT32_C(  -974.98), EASYSIMD_FLOAT32_C(   541.27), EASYSIMD_FLOAT32_C(  -704.30) },
      { EASYSIMD_FLOAT32_C(  -381.42), EASYSIMD_FLOAT32_C(  -928.98), EASYSIMD_FLOAT32_C(   659.47), EASYSIMD_FLOAT32_C(    84.94),
        EASYSIMD_FLOAT32_C(  -923.29), EASYSIMD_FLOAT32_C(    50.75), EASYSIMD_FLOAT32_C(  -890.83), EASYSIMD_FLOAT32_C(   168.26),
        EASYSIMD_FLOAT32_C(   571.66), EASYSIMD_FLOAT32_C(    31.80), EASYSIMD_FLOAT32_C(   809.45), EASYSIMD_FLOAT32_C(   576.69),
        EASYSIMD_FLOAT32_C(   332.95), EASYSIMD_FLOAT32_C(  -387.81), EASYSIMD_FLOAT32_C(   220.88), EASYSIMD_FLOAT32_C(  -941.63) },
      { EASYSIMD_FLOAT32_C(236575.12), EASYSIMD_FLOAT32_C(-437026.78), EASYSIMD_FLOAT32_C( 14592.28), EASYSIMD_FLOAT32_C(168392.56),
        EASYSIMD_FLOAT32_C( 76802.83), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-68345.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C( -4336.85), EASYSIMD_FLOAT32_C(350220.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(493141.66) } },
    { { EASYSIMD_FLOAT32_C(  -772.79), EASYSIMD_FLOAT32_C(  -716.60), EASYSIMD_FLOAT32_C(  -356.15), EASYSIMD_FLOAT32_C(    65.85),
        EASYSIMD_FLOAT32_C(   193.00), EASYSIMD_FLOAT32_C(  -610.58), EASYSIMD_FLOAT32_C(   906.00), EASYSIMD_FLOAT32_C(  -980.21),
        EASYSIMD_FLOAT32_C(   743.21), EASYSIMD_FLOAT32_C(   791.21), EASYSIMD_FLOAT32_C(   273.16), EASYSIMD_FLOAT32_C(  -205.04),
        EASYSIMD_FLOAT32_C(   816.23), EASYSIMD_FLOAT32_C(  -185.56), EASYSIMD_FLOAT32_C(    90.66), EASYSIMD_FLOAT32_C(  -565.20) },
      UINT16_C(15428),
      { EASYSIMD_FLOAT32_C(   519.74), EASYSIMD_FLOAT32_C(   -37.84), EASYSIMD_FLOAT32_C(   800.88), EASYSIMD_FLOAT32_C(   628.91),
        EASYSIMD_FLOAT32_C(  -869.58), EASYSIMD_FLOAT32_C(   372.54), EASYSIMD_FLOAT32_C(  -339.29), EASYSIMD_FLOAT32_C(   939.87),
        EASYSIMD_FLOAT32_C(   -50.77), EASYSIMD_FLOAT32_C(   993.65), EASYSIMD_FLOAT32_C(  -447.94), EASYSIMD_FLOAT32_C(  -829.89),
        EASYSIMD_FLOAT32_C(  -947.97), EASYSIMD_FLOAT32_C(  -220.73), EASYSIMD_FLOAT32_C(  -546.49), EASYSIMD_FLOAT32_C(  -304.12) },
      { EASYSIMD_FLOAT32_C(   845.11), EASYSIMD_FLOAT32_C(   646.51), EASYSIMD_FLOAT32_C(    85.29), EASYSIMD_FLOAT32_C(   751.11),
        EASYSIMD_FLOAT32_C(   666.30), EASYSIMD_FLOAT32_C(  -171.49), EASYSIMD_FLOAT32_C(   542.32), EASYSIMD_FLOAT32_C(   -60.53),
        EASYSIMD_FLOAT32_C(   623.47), EASYSIMD_FLOAT32_C(   358.55), EASYSIMD_FLOAT32_C(   753.90), EASYSIMD_FLOAT32_C(  -285.87),
        EASYSIMD_FLOAT32_C(   793.35), EASYSIMD_FLOAT32_C(  -360.65), EASYSIMD_FLOAT32_C(   464.26), EASYSIMD_FLOAT32_C(   313.09) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-285318.69), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-307939.06), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-123113.19), EASYSIMD_FLOAT32_C(170446.52),
        EASYSIMD_FLOAT32_C(-774554.88), EASYSIMD_FLOAT32_C( 41319.30), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   601.52), EASYSIMD_FLOAT32_C(   265.13), EASYSIMD_FLOAT32_C(   -58.00), EASYSIMD_FLOAT32_C(   731.94),
        EASYSIMD_FLOAT32_C(  -362.33), EASYSIMD_FLOAT32_C(   602.71), EASYSIMD_FLOAT32_C(   671.81), EASYSIMD_FLOAT32_C(   586.90),
        EASYSIMD_FLOAT32_C(   596.36), EASYSIMD_FLOAT32_C(  -776.13), EASYSIMD_FLOAT32_C(   757.01), EASYSIMD_FLOAT32_C(   648.39),
        EASYSIMD_FLOAT32_C(     3.14), EASYSIMD_FLOAT32_C(  -789.48), EASYSIMD_FLOAT32_C(  -655.73), EASYSIMD_FLOAT32_C(  -151.75) },
      UINT16_C(57273),
      { EASYSIMD_FLOAT32_C(  -400.64), EASYSIMD_FLOAT32_C(   523.33), EASYSIMD_FLOAT32_C(  -741.93), EASYSIMD_FLOAT32_C(  -858.31),
        EASYSIMD_FLOAT32_C(  -537.20), EASYSIMD_FLOAT32_C(   881.54), EASYSIMD_FLOAT32_C(   500.24), EASYSIMD_FLOAT32_C(  -783.30),
        EASYSIMD_FLOAT32_C(  -404.34), EASYSIMD_FLOAT32_C(   293.59), EASYSIMD_FLOAT32_C(  -143.95), EASYSIMD_FLOAT32_C(  -940.08),
        EASYSIMD_FLOAT32_C(  -393.32), EASYSIMD_FLOAT32_C(  -542.43), EASYSIMD_FLOAT32_C(   325.05), EASYSIMD_FLOAT32_C(   548.68) },
      { EASYSIMD_FLOAT32_C(  -810.49), EASYSIMD_FLOAT32_C(   962.73), EASYSIMD_FLOAT32_C(   151.39), EASYSIMD_FLOAT32_C(   861.32),
        EASYSIMD_FLOAT32_C(   549.63), EASYSIMD_FLOAT32_C(  -252.24), EASYSIMD_FLOAT32_C(  -914.81), EASYSIMD_FLOAT32_C(   306.64),
        EASYSIMD_FLOAT32_C(  -603.86), EASYSIMD_FLOAT32_C(    88.33), EASYSIMD_FLOAT32_C(   517.15), EASYSIMD_FLOAT32_C(  -259.59),
        EASYSIMD_FLOAT32_C(   936.58), EASYSIMD_FLOAT32_C(   374.18), EASYSIMD_FLOAT32_C(  -830.03), EASYSIMD_FLOAT32_C(  -464.06) },
      { EASYSIMD_FLOAT32_C(-240182.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-629092.75),
        EASYSIMD_FLOAT32_C(194094.05), EASYSIMD_FLOAT32_C(531565.25), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-460025.41),
        EASYSIMD_FLOAT32_C(-240528.33), EASYSIMD_FLOAT32_C(-227952.33), EASYSIMD_FLOAT32_C(-109488.73), EASYSIMD_FLOAT32_C(-609278.94),
        EASYSIMD_FLOAT32_C( -2171.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-212314.98), EASYSIMD_FLOAT32_C(-82798.12) } },
    { { EASYSIMD_FLOAT32_C(  -102.49), EASYSIMD_FLOAT32_C(  -571.96), EASYSIMD_FLOAT32_C(  -322.38), EASYSIMD_FLOAT32_C(   360.30),
        EASYSIMD_FLOAT32_C(  -690.43), EASYSIMD_FLOAT32_C(  -822.14), EASYSIMD_FLOAT32_C(   577.00), EASYSIMD_FLOAT32_C(   -94.76),
        EASYSIMD_FLOAT32_C(   471.45), EASYSIMD_FLOAT32_C(  -566.95), EASYSIMD_FLOAT32_C(   -34.85), EASYSIMD_FLOAT32_C(  -921.87),
        EASYSIMD_FLOAT32_C(  -109.37), EASYSIMD_FLOAT32_C(  -709.79), EASYSIMD_FLOAT32_C(   626.81), EASYSIMD_FLOAT32_C(    80.14) },
      UINT16_C(33482),
      { EASYSIMD_FLOAT32_C(   -58.54), EASYSIMD_FLOAT32_C(   802.56), EASYSIMD_FLOAT32_C(   525.96), EASYSIMD_FLOAT32_C(    26.65),
        EASYSIMD_FLOAT32_C(   109.20), EASYSIMD_FLOAT32_C(   922.11), EASYSIMD_FLOAT32_C(  -885.02), EASYSIMD_FLOAT32_C(  -373.65),
        EASYSIMD_FLOAT32_C(  -337.48), EASYSIMD_FLOAT32_C(  -948.45), EASYSIMD_FLOAT32_C(  -999.47), EASYSIMD_FLOAT32_C(  -167.51),
        EASYSIMD_FLOAT32_C(  -412.51), EASYSIMD_FLOAT32_C(  -101.97), EASYSIMD_FLOAT32_C(   260.52), EASYSIMD_FLOAT32_C(   265.12) },
      { EASYSIMD_FLOAT32_C(  -741.67), EASYSIMD_FLOAT32_C(   570.09), EASYSIMD_FLOAT32_C(   442.98), EASYSIMD_FLOAT32_C(   835.33),
        EASYSIMD_FLOAT32_C(  -524.67), EASYSIMD_FLOAT32_C(   -85.57), EASYSIMD_FLOAT32_C(  -731.61), EASYSIMD_FLOAT32_C(   440.49),
        EASYSIMD_FLOAT32_C(    -7.44), EASYSIMD_FLOAT32_C(   159.01), EASYSIMD_FLOAT32_C(   730.69), EASYSIMD_FLOAT32_C(  -380.63),
        EASYSIMD_FLOAT32_C(  -760.85), EASYSIMD_FLOAT32_C(   983.62), EASYSIMD_FLOAT32_C(   397.57), EASYSIMD_FLOAT32_C(   180.62) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-459602.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  8766.66),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-509924.94), EASYSIMD_FLOAT32_C( 34966.59),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(537564.75), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 21066.10) } },
    { { EASYSIMD_FLOAT32_C(   786.18), EASYSIMD_FLOAT32_C(   -76.47), EASYSIMD_FLOAT32_C(  -792.73), EASYSIMD_FLOAT32_C(  -104.62),
        EASYSIMD_FLOAT32_C(  -154.36), EASYSIMD_FLOAT32_C(  -677.75), EASYSIMD_FLOAT32_C(   521.73), EASYSIMD_FLOAT32_C(   508.16),
        EASYSIMD_FLOAT32_C(  -626.20), EASYSIMD_FLOAT32_C(   522.26), EASYSIMD_FLOAT32_C(  -659.36), EASYSIMD_FLOAT32_C(   -38.71),
        EASYSIMD_FLOAT32_C(  -579.71), EASYSIMD_FLOAT32_C(   601.16), EASYSIMD_FLOAT32_C(  -773.59), EASYSIMD_FLOAT32_C(  -321.38) },
      UINT16_C( 1072),
      { EASYSIMD_FLOAT32_C(  -486.04), EASYSIMD_FLOAT32_C(   646.59), EASYSIMD_FLOAT32_C(  -416.19), EASYSIMD_FLOAT32_C(  -217.66),
        EASYSIMD_FLOAT32_C(    87.07), EASYSIMD_FLOAT32_C(   576.37), EASYSIMD_FLOAT32_C(   941.36), EASYSIMD_FLOAT32_C(  -182.23),
        EASYSIMD_FLOAT32_C(  -804.26), EASYSIMD_FLOAT32_C(  -819.49), EASYSIMD_FLOAT32_C(  -198.61), EASYSIMD_FLOAT32_C(   593.31),
        EASYSIMD_FLOAT32_C(   361.13), EASYSIMD_FLOAT32_C(  -412.42), EASYSIMD_FLOAT32_C(  -483.15), EASYSIMD_FLOAT32_C(   568.40) },
      { EASYSIMD_FLOAT32_C(   482.96), EASYSIMD_FLOAT32_C(   362.49), EASYSIMD_FLOAT32_C(   890.64), EASYSIMD_FLOAT32_C(     4.69),
        EASYSIMD_FLOAT32_C(  -129.36), EASYSIMD_FLOAT32_C(  -735.56), EASYSIMD_FLOAT32_C(  -473.06), EASYSIMD_FLOAT32_C(   211.28),
        EASYSIMD_FLOAT32_C(   225.74), EASYSIMD_FLOAT32_C(   -52.77), EASYSIMD_FLOAT32_C(  -187.56), EASYSIMD_FLOAT32_C(   452.15),
        EASYSIMD_FLOAT32_C(   625.86), EASYSIMD_FLOAT32_C(   983.70), EASYSIMD_FLOAT32_C(   121.54), EASYSIMD_FLOAT32_C(  -860.19) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-13310.76), EASYSIMD_FLOAT32_C(-389899.19), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(131143.05), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   630.29), EASYSIMD_FLOAT32_C(   705.35), EASYSIMD_FLOAT32_C(   -77.84), EASYSIMD_FLOAT32_C(  -282.64),
        EASYSIMD_FLOAT32_C(   281.72), EASYSIMD_FLOAT32_C(  -136.48), EASYSIMD_FLOAT32_C(   535.13), EASYSIMD_FLOAT32_C(   477.46),
        EASYSIMD_FLOAT32_C(    44.03), EASYSIMD_FLOAT32_C(  -663.48), EASYSIMD_FLOAT32_C(    70.77), EASYSIMD_FLOAT32_C(  -594.84),
        EASYSIMD_FLOAT32_C(   -75.90), EASYSIMD_FLOAT32_C(   587.61), EASYSIMD_FLOAT32_C(   973.55), EASYSIMD_FLOAT32_C(  -592.95) },
      UINT16_C(56983),
      { EASYSIMD_FLOAT32_C(   411.74), EASYSIMD_FLOAT32_C(   820.74), EASYSIMD_FLOAT32_C(  -871.36), EASYSIMD_FLOAT32_C(   938.68),
        EASYSIMD_FLOAT32_C(    32.02), EASYSIMD_FLOAT32_C(   354.38), EASYSIMD_FLOAT32_C(  -114.08), EASYSIMD_FLOAT32_C(   844.46),
        EASYSIMD_FLOAT32_C(  -193.47), EASYSIMD_FLOAT32_C(  -488.22), EASYSIMD_FLOAT32_C(   828.17), EASYSIMD_FLOAT32_C(   928.07),
        EASYSIMD_FLOAT32_C(  -348.41), EASYSIMD_FLOAT32_C(   458.46), EASYSIMD_FLOAT32_C(   633.42), EASYSIMD_FLOAT32_C(   573.75) },
      { EASYSIMD_FLOAT32_C(  -824.18), EASYSIMD_FLOAT32_C(   -84.86), EASYSIMD_FLOAT32_C(  -562.73), EASYSIMD_FLOAT32_C(   710.95),
        EASYSIMD_FLOAT32_C(  -607.40), EASYSIMD_FLOAT32_C(   481.30), EASYSIMD_FLOAT32_C(  -952.53), EASYSIMD_FLOAT32_C(   463.36),
        EASYSIMD_FLOAT32_C(   886.45), EASYSIMD_FLOAT32_C(   -28.44), EASYSIMD_FLOAT32_C(    50.98), EASYSIMD_FLOAT32_C(   860.01),
        EASYSIMD_FLOAT32_C(   378.62), EASYSIMD_FLOAT32_C(  -998.92), EASYSIMD_FLOAT32_C(   724.20), EASYSIMD_FLOAT32_C(  -209.64) },
      { EASYSIMD_FLOAT32_C(260339.78), EASYSIMD_FLOAT32_C(578993.81), EASYSIMD_FLOAT32_C( 68389.38), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  9628.08), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(402732.50),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(323952.62), EASYSIMD_FLOAT32_C( 58558.61), EASYSIMD_FLOAT32_C(-552913.19),
        EASYSIMD_FLOAT32_C( 26065.70), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(615941.81), EASYSIMD_FLOAT32_C(-339995.44) } },
    { { EASYSIMD_FLOAT32_C(   821.82), EASYSIMD_FLOAT32_C(   852.84), EASYSIMD_FLOAT32_C(  -270.96), EASYSIMD_FLOAT32_C(  -146.16),
        EASYSIMD_FLOAT32_C(   207.22), EASYSIMD_FLOAT32_C(   614.96), EASYSIMD_FLOAT32_C(  -301.70), EASYSIMD_FLOAT32_C(  -986.25),
        EASYSIMD_FLOAT32_C(  -873.27), EASYSIMD_FLOAT32_C(  -473.53), EASYSIMD_FLOAT32_C(   941.82), EASYSIMD_FLOAT32_C(  -221.68),
        EASYSIMD_FLOAT32_C(   984.92), EASYSIMD_FLOAT32_C(   575.24), EASYSIMD_FLOAT32_C(  -647.93), EASYSIMD_FLOAT32_C(  -839.26) },
      UINT16_C(43672),
      { EASYSIMD_FLOAT32_C(   871.69), EASYSIMD_FLOAT32_C(  -117.02), EASYSIMD_FLOAT32_C(  -729.37), EASYSIMD_FLOAT32_C(   919.16),
        EASYSIMD_FLOAT32_C(  -653.65), EASYSIMD_FLOAT32_C(  -842.91), EASYSIMD_FLOAT32_C(  -109.27), EASYSIMD_FLOAT32_C(   397.33),
        EASYSIMD_FLOAT32_C(  -982.90), EASYSIMD_FLOAT32_C(  -730.66), EASYSIMD_FLOAT32_C(   398.40), EASYSIMD_FLOAT32_C(   741.30),
        EASYSIMD_FLOAT32_C(    59.70), EASYSIMD_FLOAT32_C(   220.22), EASYSIMD_FLOAT32_C(   594.14), EASYSIMD_FLOAT32_C(   788.74) },
      { EASYSIMD_FLOAT32_C(  -925.94), EASYSIMD_FLOAT32_C(  -198.64), EASYSIMD_FLOAT32_C(   403.69), EASYSIMD_FLOAT32_C(  -227.64),
        EASYSIMD_FLOAT32_C(  -184.88), EASYSIMD_FLOAT32_C(   530.43), EASYSIMD_FLOAT32_C(   298.83), EASYSIMD_FLOAT32_C(  -243.06),
        EASYSIMD_FLOAT32_C(  -691.25), EASYSIMD_FLOAT32_C(   283.75), EASYSIMD_FLOAT32_C(  -667.81), EASYSIMD_FLOAT32_C(  -339.18),
        EASYSIMD_FLOAT32_C(   444.49), EASYSIMD_FLOAT32_C(  -177.43), EASYSIMD_FLOAT32_C(   450.16), EASYSIMD_FLOAT32_C(   316.19) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-134116.78),
        EASYSIMD_FLOAT32_C(-135264.48), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-391623.62),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(345705.66), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-163992.19),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(126856.78), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-662274.12) } },
    { { EASYSIMD_FLOAT32_C(   705.55), EASYSIMD_FLOAT32_C(   720.79), EASYSIMD_FLOAT32_C(   235.35), EASYSIMD_FLOAT32_C(  -948.10),
        EASYSIMD_FLOAT32_C(   877.88), EASYSIMD_FLOAT32_C(  -873.92), EASYSIMD_FLOAT32_C(   449.22), EASYSIMD_FLOAT32_C(   894.98),
        EASYSIMD_FLOAT32_C(  -604.58), EASYSIMD_FLOAT32_C(  -152.37), EASYSIMD_FLOAT32_C(   636.28), EASYSIMD_FLOAT32_C(   455.12),
        EASYSIMD_FLOAT32_C(  -932.16), EASYSIMD_FLOAT32_C(   230.41), EASYSIMD_FLOAT32_C(   243.85), EASYSIMD_FLOAT32_C(  -858.10) },
      UINT16_C(  867),
      { EASYSIMD_FLOAT32_C(   -85.74), EASYSIMD_FLOAT32_C(  -153.11), EASYSIMD_FLOAT32_C(  -822.03), EASYSIMD_FLOAT32_C(  -786.92),
        EASYSIMD_FLOAT32_C(   603.83), EASYSIMD_FLOAT32_C(  -513.28), EASYSIMD_FLOAT32_C(   496.83), EASYSIMD_FLOAT32_C(   936.02),
        EASYSIMD_FLOAT32_C(   147.54), EASYSIMD_FLOAT32_C(   -58.67), EASYSIMD_FLOAT32_C(  -241.41), EASYSIMD_FLOAT32_C(  -402.30),
        EASYSIMD_FLOAT32_C(  -742.48), EASYSIMD_FLOAT32_C(  -535.86), EASYSIMD_FLOAT32_C(  -681.50), EASYSIMD_FLOAT32_C(   492.87) },
      { EASYSIMD_FLOAT32_C(  -483.96), EASYSIMD_FLOAT32_C(  -803.62), EASYSIMD_FLOAT32_C(   618.94), EASYSIMD_FLOAT32_C(   965.27),
        EASYSIMD_FLOAT32_C(  -908.64), EASYSIMD_FLOAT32_C(  -985.64), EASYSIMD_FLOAT32_C(  -187.11), EASYSIMD_FLOAT32_C(   727.63),
        EASYSIMD_FLOAT32_C(   469.48), EASYSIMD_FLOAT32_C(  -119.26), EASYSIMD_FLOAT32_C(   -41.95), EASYSIMD_FLOAT32_C(  -286.67),
        EASYSIMD_FLOAT32_C(    22.64), EASYSIMD_FLOAT32_C(   -10.18), EASYSIMD_FLOAT32_C(   360.88), EASYSIMD_FLOAT32_C(   936.90) },
      { EASYSIMD_FLOAT32_C(-60009.89), EASYSIMD_FLOAT32_C(-109556.54), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(449551.31), EASYSIMD_FLOAT32_C(223373.08), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-89669.21), EASYSIMD_FLOAT32_C(  9058.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 c = easysimd_mm512_loadu_ps(test_vec[i].c);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_fmsub_ps(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_fmsub_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 c = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_maskz_fmsub_ps(k, a, b, c);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_fmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 c[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -46.12), EASYSIMD_FLOAT64_C(   -61.46), EASYSIMD_FLOAT64_C(    12.22), EASYSIMD_FLOAT64_C(   -13.13),
        EASYSIMD_FLOAT64_C(    17.13), EASYSIMD_FLOAT64_C(    34.93), EASYSIMD_FLOAT64_C(    51.88), EASYSIMD_FLOAT64_C(    19.93) },
      { EASYSIMD_FLOAT64_C(    73.17), EASYSIMD_FLOAT64_C(   -90.86), EASYSIMD_FLOAT64_C(   -64.69), EASYSIMD_FLOAT64_C(    30.40),
        EASYSIMD_FLOAT64_C(   -78.85), EASYSIMD_FLOAT64_C(    -2.23), EASYSIMD_FLOAT64_C(    38.81), EASYSIMD_FLOAT64_C(   -25.38) },
      { EASYSIMD_FLOAT64_C(     6.25), EASYSIMD_FLOAT64_C(   -22.61), EASYSIMD_FLOAT64_C(     8.61), EASYSIMD_FLOAT64_C(   -27.29),
        EASYSIMD_FLOAT64_C(   -33.10), EASYSIMD_FLOAT64_C(    99.80), EASYSIMD_FLOAT64_C(   -95.28), EASYSIMD_FLOAT64_C(    59.13) },
      { EASYSIMD_FLOAT64_C( -3380.85), EASYSIMD_FLOAT64_C(  5606.87), EASYSIMD_FLOAT64_C(  -799.12), EASYSIMD_FLOAT64_C(  -371.86),
        EASYSIMD_FLOAT64_C( -1317.60), EASYSIMD_FLOAT64_C(  -177.69), EASYSIMD_FLOAT64_C(  2108.74), EASYSIMD_FLOAT64_C(  -564.95) } },
    { { EASYSIMD_FLOAT64_C(    28.50), EASYSIMD_FLOAT64_C(   -32.67), EASYSIMD_FLOAT64_C(   -95.72), EASYSIMD_FLOAT64_C(    21.20),
        EASYSIMD_FLOAT64_C(    40.97), EASYSIMD_FLOAT64_C(   -90.91), EASYSIMD_FLOAT64_C(    40.73), EASYSIMD_FLOAT64_C(    94.85) },
      { EASYSIMD_FLOAT64_C(   -52.37), EASYSIMD_FLOAT64_C(   -47.05), EASYSIMD_FLOAT64_C(   -18.28), EASYSIMD_FLOAT64_C(    64.76),
        EASYSIMD_FLOAT64_C(    87.88), EASYSIMD_FLOAT64_C(   -66.39), EASYSIMD_FLOAT64_C(   -15.31), EASYSIMD_FLOAT64_C(    61.05) },
      { EASYSIMD_FLOAT64_C(   -57.26), EASYSIMD_FLOAT64_C(    19.99), EASYSIMD_FLOAT64_C(    -8.55), EASYSIMD_FLOAT64_C(   -36.11),
        EASYSIMD_FLOAT64_C(   -82.23), EASYSIMD_FLOAT64_C(   -69.73), EASYSIMD_FLOAT64_C(    38.51), EASYSIMD_FLOAT64_C(    24.02) },
      { EASYSIMD_FLOAT64_C( -1435.28), EASYSIMD_FLOAT64_C(  1517.13), EASYSIMD_FLOAT64_C(  1758.31), EASYSIMD_FLOAT64_C(  1409.02),
        EASYSIMD_FLOAT64_C(  3682.67), EASYSIMD_FLOAT64_C(  6105.24), EASYSIMD_FLOAT64_C(  -662.09), EASYSIMD_FLOAT64_C(  5766.57) } },
    { { EASYSIMD_FLOAT64_C(     7.65), EASYSIMD_FLOAT64_C(   -52.88), EASYSIMD_FLOAT64_C(    96.73), EASYSIMD_FLOAT64_C(    74.56),
        EASYSIMD_FLOAT64_C(   -53.08), EASYSIMD_FLOAT64_C(   -98.55), EASYSIMD_FLOAT64_C(    33.69), EASYSIMD_FLOAT64_C(    75.41) },
      { EASYSIMD_FLOAT64_C(   -31.22), EASYSIMD_FLOAT64_C(    37.96), EASYSIMD_FLOAT64_C(    -3.39), EASYSIMD_FLOAT64_C(   -90.25),
        EASYSIMD_FLOAT64_C(    47.06), EASYSIMD_FLOAT64_C(   -62.65), EASYSIMD_FLOAT64_C(   -95.40), EASYSIMD_FLOAT64_C(    94.69) },
      { EASYSIMD_FLOAT64_C(    -9.70), EASYSIMD_FLOAT64_C(   -13.67), EASYSIMD_FLOAT64_C(    59.45), EASYSIMD_FLOAT64_C(   -21.82),
        EASYSIMD_FLOAT64_C(    19.93), EASYSIMD_FLOAT64_C(   -55.86), EASYSIMD_FLOAT64_C(   -60.77), EASYSIMD_FLOAT64_C(    62.67) },
      { EASYSIMD_FLOAT64_C(  -229.13), EASYSIMD_FLOAT64_C( -1993.65), EASYSIMD_FLOAT64_C(  -387.36), EASYSIMD_FLOAT64_C( -6707.22),
        EASYSIMD_FLOAT64_C( -2517.87), EASYSIMD_FLOAT64_C(  6230.02), EASYSIMD_FLOAT64_C( -3153.26), EASYSIMD_FLOAT64_C(  7077.90) } },
    { { EASYSIMD_FLOAT64_C(    64.13), EASYSIMD_FLOAT64_C(    30.68), EASYSIMD_FLOAT64_C(   -73.44), EASYSIMD_FLOAT64_C(    81.90),
        EASYSIMD_FLOAT64_C(    60.95), EASYSIMD_FLOAT64_C(    65.08), EASYSIMD_FLOAT64_C(     5.91), EASYSIMD_FLOAT64_C(   -31.40) },
      { EASYSIMD_FLOAT64_C(   -87.81), EASYSIMD_FLOAT64_C(     2.64), EASYSIMD_FLOAT64_C(   -56.84), EASYSIMD_FLOAT64_C(   -40.89),
        EASYSIMD_FLOAT64_C(     4.09), EASYSIMD_FLOAT64_C(    76.84), EASYSIMD_FLOAT64_C(   -65.48), EASYSIMD_FLOAT64_C(    72.87) },
      { EASYSIMD_FLOAT64_C(    14.81), EASYSIMD_FLOAT64_C(    31.14), EASYSIMD_FLOAT64_C(    82.62), EASYSIMD_FLOAT64_C(   -38.13),
        EASYSIMD_FLOAT64_C(    68.49), EASYSIMD_FLOAT64_C(    87.23), EASYSIMD_FLOAT64_C(   -43.44), EASYSIMD_FLOAT64_C(   -41.22) },
      { EASYSIMD_FLOAT64_C( -5646.07), EASYSIMD_FLOAT64_C(    49.86), EASYSIMD_FLOAT64_C(  4091.71), EASYSIMD_FLOAT64_C( -3310.76),
        EASYSIMD_FLOAT64_C(   180.80), EASYSIMD_FLOAT64_C(  4913.52), EASYSIMD_FLOAT64_C(  -343.55), EASYSIMD_FLOAT64_C( -2246.90) } },
    { { EASYSIMD_FLOAT64_C(   -26.44), EASYSIMD_FLOAT64_C(   -83.99), EASYSIMD_FLOAT64_C(    36.96), EASYSIMD_FLOAT64_C(    93.49),
        EASYSIMD_FLOAT64_C(   -39.85), EASYSIMD_FLOAT64_C(    76.19), EASYSIMD_FLOAT64_C(    56.16), EASYSIMD_FLOAT64_C(   -75.72) },
      { EASYSIMD_FLOAT64_C(     6.87), EASYSIMD_FLOAT64_C(    82.72), EASYSIMD_FLOAT64_C(   -93.82), EASYSIMD_FLOAT64_C(   -32.18),
        EASYSIMD_FLOAT64_C(    47.80), EASYSIMD_FLOAT64_C(    12.09), EASYSIMD_FLOAT64_C(    36.42), EASYSIMD_FLOAT64_C(    59.99) },
      { EASYSIMD_FLOAT64_C(   -85.27), EASYSIMD_FLOAT64_C(    79.58), EASYSIMD_FLOAT64_C(   -80.90), EASYSIMD_FLOAT64_C(    18.82),
        EASYSIMD_FLOAT64_C(    56.42), EASYSIMD_FLOAT64_C(   -46.38), EASYSIMD_FLOAT64_C(    -8.31), EASYSIMD_FLOAT64_C(   -28.77) },
      { EASYSIMD_FLOAT64_C(   -96.37), EASYSIMD_FLOAT64_C( -7027.23), EASYSIMD_FLOAT64_C( -3386.69), EASYSIMD_FLOAT64_C( -3027.33),
        EASYSIMD_FLOAT64_C( -1961.25), EASYSIMD_FLOAT64_C(   967.52), EASYSIMD_FLOAT64_C(  2053.66), EASYSIMD_FLOAT64_C( -4513.67) } },
    { { EASYSIMD_FLOAT64_C(    84.76), EASYSIMD_FLOAT64_C(   -25.68), EASYSIMD_FLOAT64_C(    33.09), EASYSIMD_FLOAT64_C(    53.25),
        EASYSIMD_FLOAT64_C(   -38.45), EASYSIMD_FLOAT64_C(    89.65), EASYSIMD_FLOAT64_C(   -87.97), EASYSIMD_FLOAT64_C(    35.10) },
      { EASYSIMD_FLOAT64_C(   -94.34), EASYSIMD_FLOAT64_C(    48.99), EASYSIMD_FLOAT64_C(    28.59), EASYSIMD_FLOAT64_C(   -34.19),
        EASYSIMD_FLOAT64_C(    25.18), EASYSIMD_FLOAT64_C(   -15.25), EASYSIMD_FLOAT64_C(    -9.91), EASYSIMD_FLOAT64_C(   -67.95) },
      { EASYSIMD_FLOAT64_C(   -32.53), EASYSIMD_FLOAT64_C(    -3.73), EASYSIMD_FLOAT64_C(    -0.13), EASYSIMD_FLOAT64_C(   -84.73),
        EASYSIMD_FLOAT64_C(   -91.64), EASYSIMD_FLOAT64_C(   -63.71), EASYSIMD_FLOAT64_C(    75.26), EASYSIMD_FLOAT64_C(   -76.91) },
      { EASYSIMD_FLOAT64_C( -7963.73), EASYSIMD_FLOAT64_C( -1254.33), EASYSIMD_FLOAT64_C(   946.17), EASYSIMD_FLOAT64_C( -1735.89),
        EASYSIMD_FLOAT64_C(  -876.53), EASYSIMD_FLOAT64_C( -1303.45), EASYSIMD_FLOAT64_C(   796.52), EASYSIMD_FLOAT64_C( -2308.14) } },
    { { EASYSIMD_FLOAT64_C(   -84.14), EASYSIMD_FLOAT64_C(    94.36), EASYSIMD_FLOAT64_C(    41.92), EASYSIMD_FLOAT64_C(    72.28),
        EASYSIMD_FLOAT64_C(   -52.01), EASYSIMD_FLOAT64_C(   -66.39), EASYSIMD_FLOAT64_C(   -56.49), EASYSIMD_FLOAT64_C(   -67.25) },
      { EASYSIMD_FLOAT64_C(     7.93), EASYSIMD_FLOAT64_C(    76.60), EASYSIMD_FLOAT64_C(    85.99), EASYSIMD_FLOAT64_C(    69.48),
        EASYSIMD_FLOAT64_C(    66.25), EASYSIMD_FLOAT64_C(    98.03), EASYSIMD_FLOAT64_C(     4.58), EASYSIMD_FLOAT64_C(    71.92) },
      { EASYSIMD_FLOAT64_C(    47.02), EASYSIMD_FLOAT64_C(   -66.83), EASYSIMD_FLOAT64_C(   -62.27), EASYSIMD_FLOAT64_C(   -27.80),
        EASYSIMD_FLOAT64_C(    17.92), EASYSIMD_FLOAT64_C(    27.82), EASYSIMD_FLOAT64_C(     4.25), EASYSIMD_FLOAT64_C(    85.39) },
      { EASYSIMD_FLOAT64_C(  -714.25), EASYSIMD_FLOAT64_C(  7294.81), EASYSIMD_FLOAT64_C(  3666.97), EASYSIMD_FLOAT64_C(  5049.81),
        EASYSIMD_FLOAT64_C( -3463.58), EASYSIMD_FLOAT64_C( -6536.03), EASYSIMD_FLOAT64_C(  -262.97), EASYSIMD_FLOAT64_C( -4922.01) } },
    { { EASYSIMD_FLOAT64_C(   -75.91), EASYSIMD_FLOAT64_C(   -95.88), EASYSIMD_FLOAT64_C(   -99.34), EASYSIMD_FLOAT64_C(   -67.54),
        EASYSIMD_FLOAT64_C(   -59.59), EASYSIMD_FLOAT64_C(    75.92), EASYSIMD_FLOAT64_C(   -44.45), EASYSIMD_FLOAT64_C(   -43.72) },
      { EASYSIMD_FLOAT64_C(    70.29), EASYSIMD_FLOAT64_C(    97.47), EASYSIMD_FLOAT64_C(   -71.44), EASYSIMD_FLOAT64_C(   -81.73),
        EASYSIMD_FLOAT64_C(   -68.92), EASYSIMD_FLOAT64_C(   -27.93), EASYSIMD_FLOAT64_C(   -48.98), EASYSIMD_FLOAT64_C(    39.01) },
      { EASYSIMD_FLOAT64_C(   -51.32), EASYSIMD_FLOAT64_C(   -62.98), EASYSIMD_FLOAT64_C(     8.48), EASYSIMD_FLOAT64_C(   -85.07),
        EASYSIMD_FLOAT64_C(   -64.96), EASYSIMD_FLOAT64_C(   -86.94), EASYSIMD_FLOAT64_C(    86.85), EASYSIMD_FLOAT64_C(    82.06) },
      { EASYSIMD_FLOAT64_C( -5284.39), EASYSIMD_FLOAT64_C( -9282.44), EASYSIMD_FLOAT64_C(  7088.37), EASYSIMD_FLOAT64_C(  5605.11),
        EASYSIMD_FLOAT64_C(  4171.90), EASYSIMD_FLOAT64_C( -2033.51), EASYSIMD_FLOAT64_C(  2090.31), EASYSIMD_FLOAT64_C( -1787.58) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d c = easysimd_mm512_loadu_pd(test_vec[i].c);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_fmsub_pd(a, b, c);
    }
    EASYSIMD_TEST_PERF_END("_mm512_fmsub_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask3_fmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask3_fmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask3_fmsubadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fmsubadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fmsubadd_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask3_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask3_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask3_fmsubadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fmsubadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fmsubadd_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask3_fmsubadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fmsubadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fmsubadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask3_fmsubadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fmsubadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fmsubadd_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_fmsub_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
